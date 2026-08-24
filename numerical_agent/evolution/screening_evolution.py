"""Bounded single-agent evolution of task-conditioned screening policies."""
from __future__ import annotations

import ast
import json
import math
import pprint
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object

from .execution import CRASHED, INVALID, NOT_APPLICABLE, SUCCESS, Outcome, Task
from .filtering import FilterDictionary
from .screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningGateResult,
    ScreeningConstraints,
    ScreeningPolicy,
    ScreeningScore,
    compare_screening,
    evaluate_screening,
    materialize_active_dictionary,
    profile_task,
    profile_tags,
    task_group_tags,
)


SCREENING_SYSTEM = """You are one self-evolving task-conditioned numerical Filter Agent.
You receive only aggregate Train evidence and a current candidate screening policy. Every method
is compared with the named baseline inside history-only strata for periodicity, trend,
intermittency, regime, frequency, history length, and horizon. Propose a conservative Child policy.
Keep a method broad only when it is reliable across supported strata. Mark it specialized when the
evidence shows a coherent regime where it beats or safely complements the baseline. Statistical,
TSFM, and Combined methods are equally eligible for typed specialization. Treat not_applicable as
valid specialist behavior; use repair for crashes/invalid outputs and quarantine only for severe
unreliability. oracle_profiles lists anonymous Train strata where a method is globally best; every
listed profile must remain covered by keep or a matching clause. Never manufacture dictionary
differences without performance evidence. Preserve
the per-task historical oracle, reduce crash/invalid exposure, and move candidate counts toward the
supplied bounds. Discard is unavailable in this stage. Applicability is OR across any_of clauses
and AND within each clause. Use only supplied profile fields, tags, operators, and evidence. Do not
invent methods, results, task IDs, source code, documents, or labels. Every required target must
appear exactly once and no unrequested target may appear. Return only one JSON object:
{"summary":"...","actions":[{"name":"...","status":"keep|specialized|repair|quarantine",
"any_of":[{"all_tags":["..."],"feature_tests":[{"field":"...","operator":">=",
"value":0.5}]}],"reason":"..."}]}
For broad keep use an empty any_of. Repair and quarantine must use an empty any_of. Return no more
than 24 actions."""


class ScreeningEvolutionError(ValueError):
    """An Agent response or screening artifact violates the typed boundary."""


@dataclass(frozen=True)
class ScreeningGeneration:
    generation: int
    parent: ScreeningPolicy
    child: ScreeningPolicy
    train_parent: ScreeningScore
    train_child: ScreeningScore
    dev_parent: ScreeningScore
    dev_child: ScreeningScore
    gate: ScreeningGateResult
    accepted: bool
    required_targets: tuple[str, ...]
    agent_calls: int = 1
    oracle_shields: tuple["OracleShield", ...] = ()
    action_decisions: tuple["ScreeningActionDecision", ...] = ()


@dataclass(frozen=True)
class OracleShield:
    """One anonymous Train profile clause restored to retain a historical oracle."""

    method: str
    profile_tags: tuple[str, ...]


@dataclass(frozen=True)
class ScreeningActionDecision:
    """Dev-gate decision for one Train-generated action salvaged from a failed batch."""

    name: str
    accepted: bool
    reason: str


def migrate_filter_dictionary(
    dictionary: FilterDictionary,
    *,
    fallback_names: Sequence[str],
) -> ScreeningPolicy:
    """Convert legacy single-AND applicability tags without changing identities."""
    return ScreeningPolicy(
        entries=tuple(
            ScreeningEntry(
                entry.name,
                entry.family,
                entry.status,
                ApplicabilityPolicy(
                    (ApplicabilityClause(tuple(entry.applicability)),)
                    if entry.applicability
                    else ()
                ),
                entry.reason,
            )
            for entry in dictionary.entries
        ),
        fallback_names=tuple(str(name) for name in fallback_names),
    )


def render_screening_source(policy: ScreeningPolicy) -> str:
    """Render one policy as safe Python literals, never executable expressions."""
    entries = [_entry_payload(entry) for entry in policy.entries]
    return (
        '"""Task-conditioned numerical screening policy."""\n\n'
        f"CANDIDATES = {pprint.pformat(entries, width=100, sort_dicts=False)}\n\n"
        f"FALLBACK_NAMES = {pprint.pformat(policy.fallback_names, width=100)}\n"
    )


def parse_screening_source(source: str) -> ScreeningPolicy:
    """Parse exact literal assignments without importing the policy file."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ScreeningEvolutionError(f"screening source does not parse: {error}") from error
    values: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"CANDIDATES", "FALLBACK_NAMES"}
        ):
            if node.targets[0].id in values:
                raise ScreeningEvolutionError("duplicate screening source assignment")
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError) as error:
                raise ScreeningEvolutionError("screening source must contain literals") from error
    if set(values) != {"CANDIDATES", "FALLBACK_NAMES"}:
        raise ScreeningEvolutionError("screening source needs CANDIDATES and FALLBACK_NAMES")
    raw_entries = values["CANDIDATES"]
    raw_fallbacks = values["FALLBACK_NAMES"]
    if not isinstance(raw_entries, (list, tuple)) or not isinstance(raw_fallbacks, (list, tuple)):
        raise ScreeningEvolutionError("screening source assignments must be sequences")
    return ScreeningPolicy(
        tuple(_parse_entry(raw) for raw in raw_entries),
        tuple(str(name) for name in raw_fallbacks),
    )


def apply_screening_response(
    parent: ScreeningPolicy,
    response: str,
    *,
    required_names: frozenset[str],
) -> ScreeningPolicy:
    """Apply one Agent mutation while preserving candidate identity and family."""
    payload = parse_json_object(response)
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise ScreeningEvolutionError("Agent response actions must be a list")
    if len(raw_actions) > 24:
        raise ScreeningEvolutionError("Agent may change at most 24 screening entries")
    replacements: dict[str, ScreeningEntry] = {}
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ScreeningEvolutionError("each screening action must be an object")
        name = str(raw.get("name", "")).strip()
        current = parent.get(name)
        if current is None:
            raise ScreeningEvolutionError(f"unknown screening candidate {name!r}")
        if name in replacements:
            raise ScreeningEvolutionError(f"duplicate screening action for {name}")
        status = str(raw.get("status", "")).strip()
        if status not in {"keep", "specialized", "repair", "quarantine"}:
            raise ScreeningEvolutionError(f"unsupported screening status {status!r}")
        any_of = _parse_any_of(raw.get("any_of", []))
        if status in {"repair", "quarantine"} and any_of.any_of:
            raise ScreeningEvolutionError(f"{status} candidates cannot define applicability")
        replacements[name] = ScreeningEntry(
            name,
            current.family,
            status,
            any_of,
            str(raw.get("reason", "")).strip(),
        )
    if missing := required_names - replacements.keys():
        raise ScreeningEvolutionError(
            "Agent must address required targets: " + ", ".join(sorted(missing))
        )
    if unexpected := replacements.keys() - required_names:
        raise ScreeningEvolutionError(
            "Agent response contains unexpected targets: "
            + ", ".join(sorted(unexpected))
        )
    return ScreeningPolicy(
        tuple(replacements.get(entry.name, entry) for entry in parent.entries),
        parent.fallback_names,
    )


def complete_target_batches(
    policy: ScreeningPolicy,
    batches: Sequence[Sequence[str]],
    *,
    batch_size: int = 24,
) -> tuple[tuple[str, ...], ...]:
    """Preserve target priority, append missing candidates, then make bounded batches."""
    if not 1 <= batch_size <= 24:
        raise ScreeningEvolutionError("screening batch_size must be between 1 and 24")
    known = {entry.name for entry in policy.entries}
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_batch in batches:
        batch = tuple(str(name) for name in raw_batch)
        if not 1 <= len(batch) <= 24:
            raise ScreeningEvolutionError("each screening batch must be nonempty and bounded")
        if len(batch) != len(set(batch)) or any(name in seen for name in batch):
            raise ScreeningEvolutionError("screening target names must appear exactly once")
        if unknown := set(batch) - known:
            raise ScreeningEvolutionError(f"unknown screening targets: {sorted(unknown)!r}")
        ordered.extend(batch)
        seen.update(batch)
    ordered.extend(entry.name for entry in policy.entries if entry.name not in seen)
    return tuple(
        tuple(ordered[index : index + batch_size])
        for index in range(0, len(ordered), batch_size)
    )


def select_refinement_targets(
    policy: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    *,
    constraints: ScreeningConstraints,
    excluded_names: frozenset[str] = frozenset(),
    required_families: Sequence[str] = (),
    limit: int = 24,
) -> tuple[str, ...]:
    """Choose broad Train-only candidates for another bounded evolution pass."""
    if not 1 <= limit <= 24:
        raise ScreeningEvolutionError("refinement target limit must be between 1 and 24")
    eligible = tuple(
        entry
        for entry in policy.entries
        if entry.status in {"keep", "specialized"}
        and entry.name != constraints.baseline_method
        and entry.name not in excluded_names
    )
    broad = tuple(
        entry
        for entry in eligible
        if entry.status == "keep" and not entry.applicability.any_of
    )
    if not eligible:
        return ()
    evidence = {
        str(row["name"]): row
        for row in build_train_evidence(
            frozenset(entry.name for entry in eligible),
            tasks,
            outcomes,
            policy=policy,
            baseline_method=constraints.baseline_method,
            min_group_support=constraints.min_group_support,
        )
    }

    def quality(entry: ScreeningEntry) -> tuple[float, float, float, str]:
        row = evidence[entry.name]
        relative = row["relative_to_baseline"]
        assert isinstance(relative, Mapping)
        group_rates = [
            float(group["win_rate"])
            for group in row["groups"]  # type: ignore[index]
            if isinstance(group, Mapping) and group.get("win_rate") is not None
        ]
        best_group = max(group_rates, default=-1.0)
        failures = float(row["crashed"]) + float(row["invalid"])
        mean_relative = relative.get("mean_relative_mase")
        harm = float(mean_relative) if mean_relative is not None else float("inf")
        return failures, harm, best_group, entry.name

    ordered: list[str] = []
    forced = tuple(str(family) for family in required_families)
    if len(forced) != len(set(forced)) or set(forced) - {"statistical", "tsfm", "combined"}:
        raise ScreeningEvolutionError("required refinement families must be unique and known")
    conditioned = {
        entry.family for entry in policy.entries if entry.status == "specialized"
    }
    families_to_prioritize = forced or constraints.required_conditioned_families
    for family in families_to_prioritize:
        if not forced and family in conditioned:
            continue
        candidates = [entry for entry in eligible if entry.family == family]
        if candidates:
            chosen = max(candidates, key=lambda entry: (quality(entry)[2], quality(entry)))
            ordered.append(chosen.name)

    for entry in sorted(broad, key=quality, reverse=True):
        if entry.name not in ordered:
            ordered.append(entry.name)
        if len(ordered) >= limit:
            break
    return tuple(ordered[:limit])


def protect_train_oracles(
    child: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
) -> tuple[ScreeningPolicy, tuple[OracleShield, ...]]:
    """Project a Child onto 100% Train-oracle coverage using history-only clauses."""
    policy_names = {entry.name for entry in child.entries}
    required: dict[str, set[tuple[str, ...]]] = {}
    for task in tasks:
        successful = [
            row
            for row in outcomes
            if row.task_id == task.task_id
            and row.method in policy_names
            and row.status == SUCCESS
            and row.mase is not None
            and math.isfinite(float(row.mase))
        ]
        if not successful:
            continue
        best_mase = min(float(row.mase) for row in successful)  # type: ignore[arg-type]
        oracles = sorted(
            row.method
            for row in successful
            if abs(float(row.mase) - best_mase) <= 1e-12  # type: ignore[arg-type]
        )
        profile = profile_task(task)
        active = {
            candidate.name for candidate in materialize_active_dictionary(child, profile).active
        }
        if active.intersection(oracles):
            continue
        chosen = oracles[0]
        required.setdefault(chosen, set()).add(tuple(sorted(task_group_tags(profile))))

    if not required:
        return child, ()
    replacements: dict[str, ScreeningEntry] = {}
    shields: list[OracleShield] = []
    for name, signatures in sorted(required.items()):
        entry = child.get(name)
        if entry is None:
            raise ScreeningEvolutionError(f"oracle shield references unknown method {name!r}")
        clauses = list(entry.applicability.any_of if entry.status == "specialized" else ())
        existing = {clause.all_tags for clause in clauses if not clause.feature_tests}
        for signature in sorted(signatures):
            if signature not in existing:
                clauses.append(ApplicabilityClause(signature))
            shields.append(OracleShield(name, signature))
        replacements[name] = ScreeningEntry(
            name,
            entry.family,
            "specialized",
            ApplicabilityPolicy(tuple(clauses)),
            entry.reason + "; protected on anonymous Train-oracle profiles",
        )
    return (
        ScreeningPolicy(
            tuple(replacements.get(entry.name, entry) for entry in child.entries),
            child.fallback_names,
        ),
        tuple(shields),
    )


def evolve_screening_once(
    parent: ScreeningPolicy,
    train_tasks: Sequence[Task],
    dev_tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    agent: LLMClient,
    *,
    generation: int,
    required_targets: Sequence[str],
    transcript_dir: str | Path,
    constraints: ScreeningConstraints | None = None,
    enforce_final_constraints: bool = False,
    required_conditioning_families: Sequence[str] = (),
) -> ScreeningGeneration:
    """Propose one Child from Train evidence and accept it with a read-only Dev gate."""
    target_names = tuple(str(name) for name in required_targets)
    if not 1 <= len(target_names) <= 24 or len(target_names) != len(set(target_names)):
        raise ScreeningEvolutionError("required targets must contain 1 to 24 unique names")
    if unknown := set(target_names) - {entry.name for entry in parent.entries}:
        raise ScreeningEvolutionError(f"unknown required targets: {sorted(unknown)!r}")
    train_ids = {task.task_id for task in train_tasks}
    dev_ids = {task.task_id for task in dev_tasks}
    train_outcomes = tuple(row for row in outcomes if row.task_id in train_ids)
    dev_outcomes = tuple(row for row in outcomes if row.task_id in dev_ids)
    train_parent = evaluate_screening(parent, train_tasks, train_outcomes)
    dev_parent = evaluate_screening(parent, dev_tasks, dev_outcomes)
    effective_constraints = constraints or ScreeningConstraints(
        baseline_method=parent.fallback_names[0],
        min_active_candidates=1,
        max_active_candidates=len(parent.entries),
        min_unique_active_dictionaries=1,
        max_mean_pairwise_jaccard=1.0,
        min_group_support=1,
        required_conditioned_families=(),
    )
    if parent.get(effective_constraints.baseline_method) is None:
        raise ScreeningEvolutionError(
            f"unknown baseline method {effective_constraints.baseline_method!r}"
        )
    conditioning_families = tuple(str(family) for family in required_conditioning_families)
    if (
        len(conditioning_families) != len(set(conditioning_families))
        or set(conditioning_families) - {"statistical", "tsfm", "combined"}
    ):
        raise ScreeningEvolutionError("required conditioning families must be unique and known")
    request_payload = {
            "generation": generation,
            "task_count": len(train_tasks),
            "required_targets": list(target_names),
            "allowed_profile_fields": list(_allowed_profile_fields()),
            "allowed_profile_tags": sorted(
                {
                    tag
                    for task in train_tasks
                    for tag in profile_tags(profile_task(task))
                }
            ),
            "allowed_operators": ["<", "<=", "==", ">=", ">", "in"],
            "baseline_method": effective_constraints.baseline_method,
            "constraints": {
                "min_active_candidates": effective_constraints.min_active_candidates,
                "max_active_candidates": effective_constraints.max_active_candidates,
                "min_unique_active_dictionaries": (
                    effective_constraints.min_unique_active_dictionaries
                ),
                "max_mean_pairwise_jaccard": (
                    effective_constraints.max_mean_pairwise_jaccard
                ),
                "oracle_retention": 1.0,
                "failure_exposure_must_not_increase": True,
                "required_conditioned_families": list(
                    effective_constraints.required_conditioned_families
                ),
            },
            "parent_train_score": _score_payload(train_parent),
            "current_policy": [_entry_payload(entry) for entry in parent.entries],
            "train_evidence": build_train_evidence(
                frozenset(target_names),
                train_tasks,
                train_outcomes,
                policy=parent,
                baseline_method=effective_constraints.baseline_method,
                min_group_support=effective_constraints.min_group_support,
            ),
        }
    if conditioning_families:
        request_payload["required_conditioning_families"] = list(conditioning_families)
        request_payload["required_conditioning_instruction"] = (
            "At least one requested method in each named family must become specialized "
            "with evidence-supported clauses; broad keep is not valid for that family."
        )
    request = json.dumps(
        request_payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    response = agent.complete(
        system=SCREENING_SYSTEM,
        messages=[{"role": "user", "content": request}],
        temperature=0.0,
    )
    directory = Path(transcript_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"generation_{generation:03d}_screening"
    (directory / f"{prefix}_request.txt").write_text(request, encoding="utf-8")
    (directory / f"{prefix}_response.json").write_text(response.text, encoding="utf-8")
    agent_calls = 1
    applied_response = response.text
    try:
        child = apply_screening_response(
            parent, response.text, required_names=frozenset(target_names)
        )
        _require_conditioned_families(child, target_names, conditioning_families)
    except ValueError as first_error:
        repair_request = json.dumps(
            {
                "validation_error": str(first_error),
                "required_targets": list(target_names),
                "instruction": (
                    "Return one complete replacement JSON object. Address every required "
                    "target exactly once and obey the original typed schema. Do not explain."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        (directory / f"{prefix}_repair_request.txt").write_text(
            repair_request, encoding="utf-8"
        )
        repaired = agent.complete(
            system=SCREENING_SYSTEM,
            messages=[
                {"role": "user", "content": request},
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": repair_request},
            ],
            temperature=0.0,
        )
        agent_calls = 2
        (directory / f"{prefix}_repair_response.json").write_text(
            repaired.text, encoding="utf-8"
        )
        try:
            child = apply_screening_response(
                parent, repaired.text, required_names=frozenset(target_names)
            )
            _require_conditioned_families(child, target_names, conditioning_families)
            applied_response = repaired.text
        except ValueError as second_error:
            return ScreeningGeneration(
                generation,
                parent,
                parent,
                train_parent,
                train_parent,
                dev_parent,
                dev_parent,
                ScreeningGateResult(
                    False,
                    f"rejected: invalid Agent response after one retry: {second_error}",
                ),
                False,
                target_names,
                agent_calls,
                (),
                (),
            )
    child, oracle_shields = protect_train_oracles(child, train_tasks, train_outcomes)
    train_child = evaluate_screening(child, train_tasks, train_outcomes)
    dev_child = evaluate_screening(child, dev_tasks, dev_outcomes)
    gate = compare_screening(
        train_parent,
        train_child,
        dev_parent,
        dev_child,
        constraints=effective_constraints,
        enforce_final_constraints=enforce_final_constraints,
    )
    action_decisions: tuple[ScreeningActionDecision, ...] = ()
    if not gate.accepted:
        (
            salvaged_child,
            salvaged_train,
            salvaged_dev,
            salvaged_shields,
            action_decisions,
            salvaged_dimensions,
        ) = _salvage_screening_actions(
            parent,
            applied_response,
            train_tasks,
            dev_tasks,
            train_outcomes,
            dev_outcomes,
            effective_constraints,
        )
        if any(decision.accepted for decision in action_decisions):
            child = salvaged_child
            train_child = salvaged_train
            dev_child = salvaged_dev
            oracle_shields = salvaged_shields
            gate = ScreeningGateResult(
                True,
                "accepted: safe actions salvaged individually from rejected batch",
                salvaged_dimensions,
            )
    return ScreeningGeneration(
        generation,
        parent,
        child,
        train_parent,
        train_child,
        dev_parent,
        dev_child,
        gate,
        gate.accepted,
        target_names,
        agent_calls,
        oracle_shields,
        action_decisions,
    )


def _salvage_screening_actions(
    parent: ScreeningPolicy,
    response: str,
    train_tasks: Sequence[Task],
    dev_tasks: Sequence[Task],
    train_outcomes: Sequence[Outcome],
    dev_outcomes: Sequence[Outcome],
    constraints: ScreeningConstraints,
) -> tuple[
    ScreeningPolicy,
    ScreeningScore,
    ScreeningScore,
    tuple[OracleShield, ...],
    tuple[ScreeningActionDecision, ...],
    tuple[str, ...],
]:
    """Evaluate independent Train-generated actions without feeding Dev data to the Agent."""
    payload = parse_json_object(response)
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise ScreeningEvolutionError("validated screening response lost its actions")
    current = parent
    current_train = evaluate_screening(current, train_tasks, train_outcomes)
    current_dev = evaluate_screening(current, dev_tasks, dev_outcomes)
    decisions = []
    accepted_shields: list[OracleShield] = []
    improved: set[str] = set()
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ScreeningEvolutionError("validated screening action is not an object")
        name = str(raw.get("name", ""))
        single = json.dumps(
            {"summary": "single-action salvage", "actions": [raw]},
            ensure_ascii=False,
            sort_keys=True,
        )
        candidate = apply_screening_response(
            current, single, required_names=frozenset({name})
        )
        candidate, shields = protect_train_oracles(candidate, train_tasks, train_outcomes)
        candidate_train = evaluate_screening(candidate, train_tasks, train_outcomes)
        candidate_dev = evaluate_screening(candidate, dev_tasks, dev_outcomes)
        result = compare_screening(
            current_train,
            candidate_train,
            current_dev,
            candidate_dev,
            constraints=constraints,
            enforce_final_constraints=False,
        )
        decisions.append(ScreeningActionDecision(name, result.accepted, result.reason))
        if result.accepted:
            current = candidate
            current_train = candidate_train
            current_dev = candidate_dev
            accepted_shields.extend(shields)
            improved.update(result.improved_dimensions)
    return (
        current,
        current_train,
        current_dev,
        tuple(accepted_shields),
        tuple(decisions),
        tuple(sorted(improved)),
    )


def _require_conditioned_families(
    child: ScreeningPolicy,
    target_names: Sequence[str],
    required_families: Sequence[str],
) -> None:
    targets = set(target_names)
    missing = {
        family
        for family in required_families
        if not any(
            entry.name in targets
            and entry.family == family
            and entry.status == "specialized"
            and bool(entry.applicability.any_of)
            for entry in child.entries
        )
    }
    if missing:
        raise ScreeningEvolutionError(
            "required families must receive a specialized target: "
            + ", ".join(sorted(missing))
        )


def _parse_entry(raw: object) -> ScreeningEntry:
    if not isinstance(raw, Mapping) or set(raw) != {
        "name", "family", "status", "any_of", "reason"
    }:
        raise ScreeningEvolutionError("invalid screening entry literal")
    return ScreeningEntry(
        str(raw["name"]),
        str(raw["family"]),
        str(raw["status"]),
        _parse_any_of(raw["any_of"]),
        str(raw["reason"]),
    )


def _parse_any_of(raw: object) -> ApplicabilityPolicy:
    if not isinstance(raw, (list, tuple)):
        raise ScreeningEvolutionError("any_of must be a sequence")
    clauses = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or not item
            or not set(item).issubset({"all_tags", "feature_tests"})
        ):
            raise ScreeningEvolutionError("invalid applicability clause")
        tags = item.get("all_tags", ())
        tests = item.get("feature_tests", ())
        if not isinstance(tags, (list, tuple)) or not isinstance(tests, (list, tuple)):
            raise ScreeningEvolutionError("clause fields must be sequences")
        feature_tests = []
        for test in tests:
            if not isinstance(test, Mapping) or set(test) != {"field", "operator", "value"}:
                raise ScreeningEvolutionError("invalid feature test")
            value = test["value"]
            if str(test["operator"]) == "in" and isinstance(value, list):
                value = tuple(value)
            feature_tests.append(
                FeatureTest(str(test["field"]), str(test["operator"]), value)
            )
        clauses.append(
            ApplicabilityClause(
                tuple(str(tag) for tag in tags),
                tuple(feature_tests),
            )
        )
    return ApplicabilityPolicy(tuple(clauses))


def _entry_payload(entry: ScreeningEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "family": entry.family,
        "status": entry.status,
        "any_of": [
            {
                "all_tags": list(clause.all_tags),
                "feature_tests": [
                    {"field": test.field, "operator": test.operator, "value": test.value}
                    for test in clause.feature_tests
                ],
            }
            for clause in entry.applicability.any_of
        ],
        "reason": entry.reason,
    }


def _allowed_profile_fields() -> tuple[str, ...]:
    return (
        "frequency", "history_length", "horizon", "zero_fraction", "signed",
        "integer_valued", "trend_direction", "trend_strength", "periodicity_periods",
        "periodicity_strength", "periodicity_confidence", "outlier_fraction",
        "noise_relative_scale", "likely_stationary", "stationarity_score",
        "recent_regime_start", "recent_regime_confidence", "intermittency_adi",
        "intermittency_cv2",
    )


def build_train_evidence(
    names: frozenset[str],
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    *,
    policy: ScreeningPolicy,
    baseline_method: str,
    min_group_support: int,
) -> list[dict[str, object]]:
    profiles = {task.task_id: profile_task(task) for task in tasks}
    groups: dict[str, set[str]] = {}
    for task_id, profile in profiles.items():
        for group in task_group_tags(profile):
            groups.setdefault(group, set()).add(task_id)
    baseline = {
        row.task_id: row for row in outcomes if row.method == baseline_method
    }
    oracle_profiles: dict[str, set[tuple[str, ...]]] = {}
    policy_names = {entry.name for entry in policy.entries}
    for task in tasks:
        successful = [
            row
            for row in outcomes
            if row.task_id == task.task_id
            and row.method in policy_names
            and row.status == SUCCESS
            and row.mase is not None
            and math.isfinite(float(row.mase))
        ]
        if not successful:
            continue
        best_mase = min(float(row.mase) for row in successful)  # type: ignore[arg-type]
        signature = tuple(sorted(task_group_tags(profiles[task.task_id])))
        for row in successful:
            if abs(float(row.mase) - best_mase) <= 1e-12:  # type: ignore[arg-type]
                oracle_profiles.setdefault(row.method, set()).add(signature)
    result = []
    for name in sorted(names):
        rows = [row for row in outcomes if row.method == name]
        entry = policy.get(name)
        if entry is None:
            raise ScreeningEvolutionError(f"unknown screening evidence method {name!r}")
        conditioned = []
        for group, task_ids in sorted(groups.items()):
            if len(task_ids) < min_group_support:
                continue
            selected = [row for row in rows if row.task_id in task_ids]
            conditioned.append(
                {
                    "group": group,
                    **_row_summary(selected),
                    **_relative_summary(selected, baseline),
                }
            )
        result.append(
            {
                "name": name,
                "family": entry.family,
                **_row_summary(rows),
                "relative_to_baseline": _relative_summary(rows, baseline),
                "oracle_profiles": [
                    list(signature)
                    for signature in sorted(oracle_profiles.get(name, set()))
                ],
                "groups": conditioned,
            }
        )
    return result


def _relative_summary(
    rows: Sequence[Outcome], baseline: Mapping[str, Outcome]
) -> dict[str, object]:
    comparable = []
    for row in rows:
        reference = baseline.get(row.task_id)
        if (
            row.status == SUCCESS
            and row.mase is not None
            and math.isfinite(float(row.mase))
            and reference is not None
            and reference.status == SUCCESS
            and reference.mase is not None
            and math.isfinite(float(reference.mase))
        ):
            comparable.append((float(row.mase), float(reference.mase)))
    tolerance = 1e-12
    wins = sum(value < reference - tolerance for value, reference in comparable)
    ties = sum(abs(value - reference) <= tolerance for value, reference in comparable)
    losses = len(comparable) - wins - ties
    return {
        "comparable": len(comparable),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": (wins + 0.5 * ties) / len(comparable) if comparable else None,
        "mean_relative_mase": statistics.fmean(
            (value - reference) / (1.0 + reference)
            for value, reference in comparable
        ) if comparable else None,
    }


def _score_payload(score: ScreeningScore) -> dict[str, object]:
    return {
        "active_success_rate": score.active_success_rate,
        "failure_exposure": score.failure_exposure,
        "not_applicable_exposure": score.not_applicable_exposure,
        "global_oracle_retention": score.global_oracle_retention,
        "mean_active_oracle_regret": score.mean_active_oracle_regret,
        "mean_active_candidates": score.mean_active_candidates,
        "min_active_candidates": score.min_active_candidates,
        "max_active_candidates": score.max_active_candidates,
        "unique_active_dictionaries": score.unique_active_dictionaries,
        "mean_pairwise_jaccard": score.mean_pairwise_jaccard,
        "conditioned_entries_by_family": dict(score.conditioned_entries_by_family),
    }


def _row_summary(rows: Sequence[Outcome]) -> dict[str, object]:
    successful = [
        row for row in rows
        if row.status == SUCCESS and row.mase is not None and math.isfinite(float(row.mase))
    ]
    return {
        "tasks": len(rows),
        "success": len(successful),
        "not_applicable": sum(row.status == NOT_APPLICABLE for row in rows),
        "crashed": sum(row.status == CRASHED for row in rows),
        "invalid": sum(row.status == INVALID for row in rows),
        "mean_mase": statistics.fmean(float(row.mase) for row in successful)
        if successful else None,
    }
