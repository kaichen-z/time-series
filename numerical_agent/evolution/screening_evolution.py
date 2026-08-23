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
    ScreeningPolicy,
    ScreeningScore,
    compare_screening,
    evaluate_screening,
    profile_task,
    profile_tags,
)


SCREENING_SYSTEM = """You are one self-evolving task-conditioned numerical Filter Agent.
You receive only aggregate Train evidence and a current candidate screening policy. Propose a
conservative Child policy. Keep broad reliable methods. Mark a method specialized when it is
useful in one coherent history-only regime. Mark faulty but valuable implementations repair and
severely unsafe implementations quarantine. Discard is unavailable in this stage. Applicability
is OR across any_of clauses and AND within each clause. Use only supplied profile fields, tags,
operators, and evidence. Do not invent methods, results, task IDs, source code, documents, or
labels. Every required target must appear exactly once. Return only one JSON object:
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
    return ScreeningPolicy(
        tuple(replacements.get(entry.name, entry) for entry in parent.entries),
        parent.fallback_names,
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
    request = json.dumps(
        {
            "generation": generation,
            "task_count": len(train_tasks),
            "required_targets": list(target_names),
            "allowed_profile_fields": list(_allowed_profile_fields()),
            "allowed_operators": ["<", "<=", "==", ">=", ">", "in"],
            "current_policy": [_entry_payload(entry) for entry in parent.entries],
            "train_evidence": _train_evidence(
                frozenset(target_names), train_tasks, train_outcomes
            ),
        },
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
    child = apply_screening_response(
        parent, response.text, required_names=frozenset(target_names)
    )
    train_child = evaluate_screening(child, train_tasks, train_outcomes)
    dev_child = evaluate_screening(child, dev_tasks, dev_outcomes)
    gate = compare_screening(train_parent, train_child, dev_parent, dev_child)
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
        if not isinstance(item, Mapping) or set(item) != {"all_tags", "feature_tests"}:
            raise ScreeningEvolutionError("invalid applicability clause")
        tags = item["all_tags"]
        tests = item["feature_tests"]
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


def _train_evidence(
    names: frozenset[str],
    tasks: Sequence[Task],
    outcomes: Sequence[Outcome],
) -> list[dict[str, object]]:
    profiles = {task.task_id: profile_task(task) for task in tasks}
    buckets = (
        ("intermittent", lambda profile: "intermittent" in profile_tags(profile)),
        ("periodicity_strength>=0.6", lambda profile: profile.periodicity_strength >= 0.6),
        ("trend_strength>=0.6", lambda profile: profile.trend_strength >= 0.6),
        ("recent_regime_confidence>=0.5", lambda profile: profile.recent_regime_confidence >= 0.5),
        ("horizon<=24", lambda profile: profile.horizon <= 24),
        ("history_length<=48", lambda profile: profile.history_length <= 48),
    )
    result = []
    for name in sorted(names):
        rows = [row for row in outcomes if row.method == name]
        result.append(
            {
                "name": name,
                **_row_summary(rows),
                "conditioned": {
                    label: _row_summary(
                        [row for row in rows if predicate(profiles[row.task_id])]
                    )
                    for label, predicate in buckets
                },
            }
        )
    return result


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
