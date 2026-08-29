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
from common.metrics import joint_scaled_error, pareto_scaled_improvement

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
valid specialist behavior. Repair or quarantine requires observed Crash/Invalid Train evidence;
poor scaled forecast quality alone must remain keep or become specialized because a weak standalone method can still
be useful as a diverse ensemble member. Every specialized clause must pass its exact joint Train subset: the supplied
minimum support, at least 75% successful executions, at least 50% Pareto wins over the named
baseline, and non-regressing median sMAE/sRMSE deltas with at least one strict improvement. Do not combine marginally supported tags into
an unsupported conjunction. For every specialized action, trusted Python will compile your intent
into at most three validated atomic Train strata; your any_of clauses are preferences, not an
authority to bypass that compiler. oracle_profiles lists anonymous Train strata where a method is globally best; every
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


@dataclass(frozen=True)
class SpecializedClauseEvidence:
    """Exact Train evidence for one proposed applicability conjunction."""

    method: str
    clause_index: int
    support: int
    successes: int
    comparable: int
    win_rate: float
    median_delta_smae: float
    median_delta_srmse: float


@dataclass(frozen=True)
class ScreeningTrainDevResult:
    """Train-only screening evolution followed by one read-only Dev gate."""

    original_parent: ScreeningPolicy
    train_winner: ScreeningPolicy
    frozen: ScreeningPolicy
    generations: tuple[ScreeningGeneration, ...]
    train_parent: ScreeningScore
    train_winner_score: ScreeningScore
    dev_parent: ScreeningScore
    dev_winner: ScreeningScore
    final_gate: ScreeningGateResult


def validate_specialized_evidence(
    parent: ScreeningPolicy,
    child: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    *,
    baseline_method: str,
    min_group_support: int,
    target_names: Sequence[str],
) -> tuple[SpecializedClauseEvidence, ...]:
    """Reject unsupported Agent-created conjunctions before they reach a gate.

    The prompt contains marginal stratum summaries.  A Child may combine those
    marginals, so trusted Python must verify the *joint* clause on Train tasks.
    Trusted oracle shields are added later and therefore are not constrained by
    this Agent-proposal check.
    """
    if min_group_support < 1:
        raise ValueError("min_group_support must be positive")
    by_key = {(row.method, row.task_id): row for row in outcomes}
    profiles = {task.task_id: profile_task(task) for task in tasks}
    evidence: list[SpecializedClauseEvidence] = []
    for name in tuple(str(item) for item in target_names):
        before = parent.get(name)
        after = child.get(name)
        if before is None or after is None or after == before or after.status != "specialized":
            continue
        for clause_index, clause in enumerate(after.applicability.any_of):
            matched = [task for task in tasks if clause.matches(profiles[task.task_id])]
            support = len(matched)
            if support < min_group_support:
                raise ScreeningEvolutionError(
                    f"{name} clause {clause_index} has joint support {support}; "
                    f"requires at least {min_group_support}"
                )
            successes = 0
            delta_smae: list[float] = []
            delta_srmse: list[float] = []
            wins = 0
            for task in matched:
                method = by_key.get((name, task.task_id))
                baseline = by_key.get((baseline_method, task.task_id))
                if method is not None and method.status == SUCCESS and _finite_outcome(method):
                    successes += 1
                if (
                    method is None
                    or baseline is None
                    or method.status != SUCCESS
                    or baseline.status != SUCCESS
                    or not _finite_outcome(method)
                    or not _finite_outcome(baseline)
                ):
                    continue
                method_smae = float(method.smae)
                method_srmse = float(method.srmse)
                baseline_smae = float(baseline.smae)
                baseline_srmse = float(baseline.srmse)
                wins += int(
                    pareto_scaled_improvement(
                        baseline_smae,
                        baseline_srmse,
                        method_smae,
                        method_srmse,
                    )
                )
                delta_smae.append(method_smae - baseline_smae)
                delta_srmse.append(method_srmse - baseline_srmse)
            comparable = len(delta_smae)
            if successes / support < 0.75 - 1e-12:
                raise ScreeningEvolutionError(
                    f"{name} clause {clause_index} is unreliable on its joint support"
                )
            if comparable < min_group_support:
                raise ScreeningEvolutionError(
                    f"{name} clause {clause_index} has only {comparable} comparable "
                    f"Train outcomes; requires at least {min_group_support}"
                )
            win_rate = wins / comparable
            median_smae = statistics.median(delta_smae)
            median_srmse = statistics.median(delta_srmse)
            median_pareto = (
                median_smae <= 1e-12
                and median_srmse <= 1e-12
                and (median_smae < -1e-12 or median_srmse < -1e-12)
            )
            if win_rate < 0.5 - 1e-12 or not median_pareto:
                raise ScreeningEvolutionError(
                    f"{name} clause {clause_index} lacks reliable baseline uplift"
                )
            evidence.append(
                SpecializedClauseEvidence(
                    name,
                    clause_index,
                    support,
                    successes,
                    comparable,
                    win_rate,
                    median_smae,
                    median_srmse,
                )
            )
    return tuple(evidence)


def compile_supported_specialists(
    parent: ScreeningPolicy,
    child: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    *,
    baseline_method: str,
    min_group_support: int,
    target_names: Sequence[str],
    max_clauses: int = 3,
) -> ScreeningPolicy:
    """Compile Agent specialization intent into exact, trusted Train strata.

    The Agent decides *which* methods should become specialists.  Python owns
    the executable applicability clauses: it enumerates approved atomic task
    strata, validates each one against the same trusted gate, removes duplicate
    task subsets, and keeps a small best-first OR policy.  This prevents a weak
    LLM conjunction from invalidating an otherwise useful generation.
    """
    if max_clauses < 1:
        raise ValueError("max_clauses must be positive")
    profiles = {task.task_id: profile_task(task) for task in tasks}
    replacements: dict[str, ScreeningEntry] = {}
    for name in tuple(str(item) for item in target_names):
        before = parent.get(name)
        proposed = child.get(name)
        if (
            before is None
            or proposed is None
            or proposed == before
            or proposed.status != "specialized"
        ):
            continue
        preferred = {
            tag
            for clause in proposed.applicability.any_of
            for tag in clause.all_tags
        }
        supported: list[
            tuple[
                bool,
                float,
                float,
                int,
                str,
                tuple[str, ...],
                ApplicabilityClause,
            ]
        ] = []
        tags = sorted(
            {tag for profile in profiles.values() for tag in task_group_tags(profile)}
        )
        for tag in tags:
            clause = ApplicabilityClause((tag,))
            matched_ids = tuple(
                sorted(
                    task_id
                    for task_id, profile in profiles.items()
                    if clause.matches(profile)
                )
            )
            candidate_entry = ScreeningEntry(
                proposed.name,
                proposed.family,
                "specialized",
                ApplicabilityPolicy((clause,)),
                proposed.reason,
            )
            candidate = ScreeningPolicy(
                tuple(
                    candidate_entry if entry.name == name else entry
                    for entry in child.entries
                ),
                child.fallback_names,
            )
            try:
                evidence = validate_specialized_evidence(
                    parent,
                    candidate,
                    tasks,
                    outcomes,
                    baseline_method=baseline_method,
                    min_group_support=min_group_support,
                    target_names=(name,),
                )[0]
            except (IndexError, ScreeningEvolutionError):
                continue
            supported.append(
                (
                    tag not in preferred,
                    max(evidence.median_delta_smae, evidence.median_delta_srmse),
                    -evidence.win_rate,
                    -evidence.support,
                    tag,
                    matched_ids,
                    clause,
                )
            )
        if not supported:
            raise ScreeningEvolutionError(
                f"{name} has no evidence-supported atomic Train stratum"
            )
        supported.sort(key=lambda row: row[:5])
        selected: list[ApplicabilityClause] = []
        seen_subsets: set[tuple[str, ...]] = set()
        for row in supported:
            if row[-2] in seen_subsets:
                continue
            seen_subsets.add(row[-2])
            selected.append(row[-1])
            if len(selected) == max_clauses:
                break
        clauses = tuple(selected)
        replacements[name] = ScreeningEntry(
            proposed.name,
            proposed.family,
            "specialized",
            ApplicabilityPolicy(clauses),
            proposed.reason + "; applicability compiled from trusted Train strata",
        )
    if not replacements:
        return child
    return ScreeningPolicy(
        tuple(replacements.get(entry.name, entry) for entry in child.entries),
        child.fallback_names,
    )


def _finite_outcome(outcome: Outcome) -> bool:
    return (
        outcome.smae is not None
        and outcome.srmse is not None
        and math.isfinite(float(outcome.smae))
        and math.isfinite(float(outcome.srmse))
    )


def validate_failure_status_evidence(
    parent: ScreeningPolicy,
    child: ScreeningPolicy,
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    *,
    target_names: Sequence[str],
) -> dict[str, int]:
    """Require trusted execution failures before making a method non-selectable.

    Forecast error is not an implementation failure. A method with weak
    standalone scaled accuracy can still contribute a useful shape or residual to a
    guarded combination, so only Crash/Invalid outcomes authorize ``repair``
    or ``quarantine``.
    """
    task_ids = {task.task_id for task in tasks}
    failure_counts: dict[str, int] = {}
    for name in tuple(str(item) for item in target_names):
        before = parent.get(name)
        after = child.get(name)
        if (
            before is None
            or after is None
            or after == before
            or after.status not in {"repair", "quarantine"}
        ):
            continue
        failures = sum(
            row.method == name
            and row.task_id in task_ids
            and row.status in {CRASHED, INVALID}
            for row in outcomes
        )
        if failures == 0:
            raise ScreeningEvolutionError(
                f"{name} cannot become {after.status}: no Crash/Invalid Train evidence"
            )
        failure_counts[name] = failures
    return failure_counts


def evolve_screening_on_train_once(
    parent: ScreeningPolicy,
    train_tasks: Sequence[Task],
    train_outcomes: Sequence[Outcome],
    agent: LLMClient,
    **kwargs: object,
) -> ScreeningGeneration:
    """Run one mutation using Train for proposal and acceptance, never Dev."""
    return evolve_screening_once(
        parent,
        train_tasks,
        train_tasks,
        train_outcomes,
        agent,
        **kwargs,
    )


def evolve_screening_train_then_dev(
    parent: ScreeningPolicy,
    train_tasks: Sequence[Task],
    dev_tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
    agent: LLMClient,
    *,
    batches: Sequence[Sequence[str]],
    transcript_dir: str | Path,
    constraints: ScreeningConstraints,
) -> ScreeningTrainDevResult:
    """Evolve every requested batch on Train, then expose Dev exactly once."""
    if not batches:
        raise ValueError("screening evolution requires at least one target batch")
    original = parent
    current = parent
    train_ids = {task.task_id for task in train_tasks}
    train_outcomes = tuple(row for row in outcomes if row.task_id in train_ids)
    generations: list[ScreeningGeneration] = []
    for generation, batch in enumerate(batches, 1):
        # Reuse the existing proposal/salvage machinery with Train as both the
        # optimization and safety partition.  No Dev task or outcome crosses
        # this boundary; Dev is read only after the final Train winner exists.
        result = evolve_screening_on_train_once(
            current,
            train_tasks,
            train_outcomes,
            agent,
            generation=generation,
            required_targets=batch,
            transcript_dir=transcript_dir,
            constraints=constraints,
            enforce_final_constraints=False,
        )
        generations.append(result)
        if result.accepted:
            current = result.child

    train_parent = evaluate_screening(original, train_tasks, train_outcomes)
    train_winner = evaluate_screening(current, train_tasks, train_outcomes)
    dev_ids = {task.task_id for task in dev_tasks}
    dev_outcomes = tuple(row for row in outcomes if row.task_id in dev_ids)
    dev_parent = evaluate_screening(original, dev_tasks, dev_outcomes)
    dev_winner = evaluate_screening(current, dev_tasks, dev_outcomes)
    final_gate = compare_screening(
        train_parent,
        train_winner,
        dev_parent,
        dev_winner,
        constraints=constraints,
        enforce_final_constraints=True,
    )
    return ScreeningTrainDevResult(
        original,
        current,
        current if final_gate.accepted else original,
        tuple(generations),
        train_parent,
        train_winner,
        dev_parent,
        dev_winner,
        final_gate,
    )


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
        mean_delta_smae = relative.get("delta_smae")
        mean_delta_srmse = relative.get("delta_srmse")
        harm = (
            (float(mean_delta_smae) + float(mean_delta_srmse)) / 2.0
            if mean_delta_smae is not None and mean_delta_srmse is not None
            else float("inf")
        )
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
            and _finite_outcome(row)
        ]
        if not successful:
            continue
        best = min(successful, key=_scaled_order)
        oracles = sorted(
            row.method
            for row in successful
            if abs(float(row.smae) - float(best.smae)) <= 1e-12
            and abs(float(row.srmse) - float(best.srmse)) <= 1e-12
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
        child = compile_supported_specialists(
            parent,
            child,
            train_tasks,
            train_outcomes,
            baseline_method=effective_constraints.baseline_method,
            min_group_support=effective_constraints.min_group_support,
            target_names=target_names,
        )
        validate_failure_status_evidence(
            parent,
            child,
            train_tasks,
            train_outcomes,
            target_names=target_names,
        )
        validate_specialized_evidence(
            parent,
            child,
            train_tasks,
            train_outcomes,
            baseline_method=effective_constraints.baseline_method,
            min_group_support=effective_constraints.min_group_support,
            target_names=target_names,
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
            child = compile_supported_specialists(
                parent,
                child,
                train_tasks,
                train_outcomes,
                baseline_method=effective_constraints.baseline_method,
                min_group_support=effective_constraints.min_group_support,
                target_names=target_names,
            )
            validate_failure_status_evidence(
                parent,
                child,
                train_tasks,
                train_outcomes,
                target_names=target_names,
            )
            validate_specialized_evidence(
                parent,
                child,
                train_tasks,
                train_outcomes,
                baseline_method=effective_constraints.baseline_method,
                min_group_support=effective_constraints.min_group_support,
                target_names=target_names,
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
        candidate = compile_supported_specialists(
            current,
            candidate,
            train_tasks,
            train_outcomes,
            baseline_method=constraints.baseline_method,
            min_group_support=constraints.min_group_support,
            target_names=(name,),
        )
        validate_failure_status_evidence(
            current,
            candidate,
            train_tasks,
            train_outcomes,
            target_names=(name,),
        )
        validate_specialized_evidence(
            current,
            candidate,
            train_tasks,
            train_outcomes,
            baseline_method=constraints.baseline_method,
            min_group_support=constraints.min_group_support,
            target_names=(name,),
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
            and _finite_outcome(row)
        ]
        if not successful:
            continue
        best = min(successful, key=_scaled_order)
        signature = tuple(sorted(task_group_tags(profiles[task.task_id])))
        for row in successful:
            if (
                abs(float(row.smae) - float(best.smae)) <= 1e-12
                and abs(float(row.srmse) - float(best.srmse)) <= 1e-12
            ):
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
    comparable: list[tuple[float, float, float, float]] = []
    for row in rows:
        reference = baseline.get(row.task_id)
        if (
            row.status == SUCCESS
            and _finite_outcome(row)
            and reference is not None
            and reference.status == SUCCESS
            and _finite_outcome(reference)
        ):
            comparable.append(
                (
                    float(row.smae),
                    float(row.srmse),
                    float(reference.smae),
                    float(reference.srmse),
                )
            )
    tolerance = 1e-12
    wins = sum(
        pareto_scaled_improvement(
            reference_smae,
            reference_srmse,
            value_smae,
            value_srmse,
            tolerance=tolerance,
        )
        for value_smae, value_srmse, reference_smae, reference_srmse in comparable
    )
    ties = sum(
        abs(value_smae - reference_smae) <= tolerance
        and abs(value_srmse - reference_srmse) <= tolerance
        for value_smae, value_srmse, reference_smae, reference_srmse in comparable
    )
    losses = len(comparable) - wins - ties
    return {
        "comparable": len(comparable),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": (wins + 0.5 * ties) / len(comparable) if comparable else None,
        "delta_smae": statistics.fmean(
            value_smae - reference_smae
            for value_smae, _, reference_smae, _ in comparable
        ) if comparable else None,
        "delta_srmse": statistics.fmean(
            value_srmse - reference_srmse
            for _, value_srmse, _, reference_srmse in comparable
        ) if comparable else None,
    }


def _score_payload(score: ScreeningScore) -> dict[str, object]:
    return {
        "mean_active_smae": score.mean_active_smae,
        "mean_active_srmse": score.mean_active_srmse,
        "active_success_rate": score.active_success_rate,
        "failure_exposure": score.failure_exposure,
        "not_applicable_exposure": score.not_applicable_exposure,
        "mean_active_failures": score.mean_active_failures,
        "mean_active_not_applicable": score.mean_active_not_applicable,
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
        if row.status == SUCCESS and _finite_outcome(row)
    ]
    return {
        "tasks": len(rows),
        "success": len(successful),
        "not_applicable": sum(row.status == NOT_APPLICABLE for row in rows),
        "crashed": sum(row.status == CRASHED for row in rows),
        "invalid": sum(row.status == INVALID for row in rows),
        "mean_smae": statistics.fmean(float(row.smae) for row in successful)
        if successful else None,
        "mean_srmse": statistics.fmean(float(row.srmse) for row in successful)
        if successful else None,
        "coverage": len(successful) / len(rows) if rows else 0.0,
        "failure_rate": (
            sum(row.status in {CRASHED, INVALID} for row in rows) / len(rows)
            if rows else 0.0
        ),
    }


def _scaled_order(row: Outcome) -> tuple[float, float, float, str]:
    assert row.smae is not None and row.srmse is not None
    smae = float(row.smae)
    srmse = float(row.srmse)
    return joint_scaled_error(smae, srmse), smae, srmse, row.method
