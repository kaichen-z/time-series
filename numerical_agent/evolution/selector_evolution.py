"""Typed self-evolution and trusted evaluation for the Numerical Selector."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import pprint
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.metrics import mae, mase, smape

from .execution import Task
from .numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    SelectionDecision,
    select_numerical_forecast,
)


SELECTOR_SYSTEM = """You are a bounded Meta-Harness Engineer for one history-only Numerical
Selector. You receive only aggregate Train diagnostics. Propose one conservative DecisionPolicy.
You may change only the supplied ranking order, recent-regime preference, minimum completed folds,
catastrophic MASE threshold, and guarded-ensemble thresholds. You cannot change identities,
screening, task partition, measurements, evaluator, runtime, cache, code, or labels. Return exactly
one JSON object with keys summary and policy. The policy must contain every allowed field exactly
once and no other fields."""


class SelectorEvolutionError(ValueError):
    """The Selector Agent crossed its typed mutation boundary."""


@dataclass(frozen=True)
class DecisionCase:
    task: Task
    active_names: tuple[str, ...]
    diagnostics: Mapping[str, CandidateDiagnostics]
    forecasts: Mapping[str, tuple[float, ...]]
    families: Mapping[str, str]


@dataclass(frozen=True)
class DecisionScore:
    task_count: int
    coverage: float
    mean_mase: float
    median_mase: float
    mean_mae: float
    median_mae: float
    mean_smape: float
    catastrophic_rate: float
    mean_active_oracle_regret: float
    method_diversity: int
    family_diversity: int
    ensemble_rate: float
    fallback_rate: float


@dataclass(frozen=True)
class DecisionGateResult:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class SelectorGeneration:
    generation: int
    parent: DecisionPolicy
    child: DecisionPolicy
    train_parent: DecisionScore
    train_child: DecisionScore
    dev_parent: DecisionScore
    dev_child: DecisionScore
    gate: DecisionGateResult
    accepted: bool
    screening_policy_hash: str
    agent_calls: int = 1


_FIELDS = (
    "ranking_order",
    "recent_regime_first",
    "min_successful_folds",
    "catastrophic_mase",
    "ensemble_enabled",
    "ensemble_max_members",
    "ensemble_min_diversity",
    "ensemble_min_improvement",
)


def render_decision_source(policy: DecisionPolicy, *, screening_policy_hash: str = "") -> str:
    payload = _policy_payload(policy)
    return (
        '"""Frozen task-conditioned numerical Decision policy."""\n\n'
        f"SCREENING_POLICY_HASH = {screening_policy_hash!r}\n"
        f"DECISION_POLICY = {pprint.pformat(payload, width=100, sort_dicts=False)}\n"
    )


def parse_decision_source(source: str) -> DecisionPolicy:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SelectorEvolutionError(f"decision source does not parse: {error}") from error
    payload = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DECISION_POLICY"
        ):
            if payload is not None:
                raise SelectorEvolutionError("duplicate DECISION_POLICY assignment")
            try:
                payload = ast.literal_eval(node.value)
            except (TypeError, ValueError) as error:
                raise SelectorEvolutionError("decision source must contain literals") from error
    if not isinstance(payload, Mapping):
        raise SelectorEvolutionError("decision source needs a DECISION_POLICY mapping")
    return _parse_policy(payload)


def decision_policy_hash(policy: DecisionPolicy, *, screening_policy_hash: str = "") -> str:
    return hashlib.sha256(
        render_decision_source(policy, screening_policy_hash=screening_policy_hash).encode("utf-8")
    ).hexdigest()


def apply_decision_response(parent: DecisionPolicy, response: str) -> DecisionPolicy:
    del parent  # The response is an exact replacement inside a fixed typed schema.
    payload = parse_json_object(response)
    if set(payload) != {"summary", "policy"} or not isinstance(payload["summary"], str):
        raise SelectorEvolutionError("response must contain exactly summary and policy")
    raw = payload["policy"]
    if not isinstance(raw, Mapping) or set(raw) != set(_FIELDS):
        raise SelectorEvolutionError("policy must contain exactly the approved fields")
    return _parse_policy(raw)


def evaluate_decision(
    policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
) -> DecisionScore:
    """Trusted evaluation: the Selector is frozen before this function reads labels."""
    records: list[tuple[DecisionCase, SelectionDecision, float, float, float, float]] = []
    for case in cases:
        try:
            decision = select_numerical_forecast(
                policy,
                active_names=case.active_names,
                diagnostics=case.diagnostics,
                forecasts=case.forecasts,
            )
        except (ValueError, KeyError):
            continue
        truth = list(case.task.future)
        prediction = list(decision.forecast)
        if len(truth) != len(prediction):
            continue
        task_mase = mase(truth, prediction, list(case.task.history))
        task_mae = mae(truth, prediction)
        task_smape = smape(truth, prediction)
        active_scores = [
            mase(truth, list(case.forecasts[name]), list(case.task.history))
            for name in case.active_names
            if name in case.forecasts and len(case.forecasts[name]) == len(truth)
        ]
        oracle = min(active_scores, default=task_mase)
        regret = (task_mase - oracle) / (1.0 + oracle)
        records.append((case, decision, task_mase, task_mae, task_smape, regret))

    mases = [record[2] for record in records]
    maes = [record[3] for record in records]
    smapes = [record[4] for record in records]
    selected = {name for _, decision, *_ in records for name in decision.selected}
    families = {
        case.families.get(name, "unknown")
        for case, decision, *_ in records
        for name in decision.selected
    }
    count = len(cases)
    completed = len(records)
    return DecisionScore(
        task_count=count,
        coverage=completed / count if count else 0.0,
        mean_mase=statistics.fmean(mases) if mases else math.inf,
        median_mase=statistics.median(mases) if mases else math.inf,
        mean_mae=statistics.fmean(maes) if maes else math.inf,
        median_mae=statistics.median(maes) if maes else math.inf,
        mean_smape=statistics.fmean(smapes) if smapes else math.inf,
        catastrophic_rate=(sum(value > 10.0 for value in mases) / completed if completed else 1.0),
        mean_active_oracle_regret=(
            statistics.fmean(record[5] for record in records) if records else math.inf
        ),
        method_diversity=len(selected),
        family_diversity=len(families),
        ensemble_rate=(
            sum(record[1].mode == "ensemble" for record in records) / completed
            if completed else 0.0
        ),
        fallback_rate=0.0,
    )


def compare_decisions(
    train_parent: DecisionScore,
    train_child: DecisionScore,
    dev_parent: DecisionScore,
    dev_child: DecisionScore,
) -> DecisionGateResult:
    if train_child.coverage < train_parent.coverage or dev_child.coverage < dev_parent.coverage:
        return DecisionGateResult(False, "coverage decreased")
    if dev_child.catastrophic_rate > dev_parent.catastrophic_rate + 1e-12:
        return DecisionGateResult(False, "Dev catastrophic-tail rate increased")
    if dev_child.mean_active_oracle_regret > dev_parent.mean_active_oracle_regret + 1e-12:
        return DecisionGateResult(False, "Dev active-oracle regret increased")
    mean_improved = dev_child.mean_mase < dev_parent.mean_mase - 1e-12
    median_safe = (
        dev_child.median_mase < dev_parent.median_mase - 1e-12
        and dev_child.mean_mase <= dev_parent.mean_mase * 1.01 + 1e-12
    )
    if not (mean_improved or median_safe):
        return DecisionGateResult(False, "Dev MASE did not improve")
    train_signal = (
        train_child.mean_mase < train_parent.mean_mase - 1e-12
        or train_child.median_mase < train_parent.median_mase - 1e-12
    )
    if not train_signal:
        return DecisionGateResult(False, "Child has no Train improvement")
    return DecisionGateResult(True, "Train proposal passed all read-only Dev gates")


def evolve_selector_once(
    parent: DecisionPolicy,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    agent: LLMClient,
    *,
    generation: int,
    screening_policy_hash: str,
    transcript_dir: str | Path,
) -> SelectorGeneration:
    train_parent = evaluate_decision(parent, train_cases)
    dev_parent = evaluate_decision(parent, dev_cases)
    request = json.dumps(
        {
            "generation": generation,
            "screening_policy_hash": screening_policy_hash,
            "current_policy": _policy_payload(parent),
            "train_summary": asdict(train_parent),
            "train_failure_summary": _failure_summary(parent, train_cases),
            "instruction": "Make one conservative typed policy proposal.",
        },
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    response = agent.complete(
        system=SELECTOR_SYSTEM,
        messages=[{"role": "user", "content": request}],
        temperature=0.0,
    )
    directory = Path(transcript_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"generation_{generation:03d}_selector"
    (directory / f"{prefix}_request.txt").write_text(request, encoding="utf-8")
    (directory / f"{prefix}_response.json").write_text(response.text, encoding="utf-8")
    child = apply_decision_response(parent, response.text)
    train_child = evaluate_decision(child, train_cases)
    dev_child = evaluate_decision(child, dev_cases)
    gate = compare_decisions(train_parent, train_child, dev_parent, dev_child)
    return SelectorGeneration(
        generation,
        parent,
        child,
        train_parent,
        train_child,
        dev_parent,
        dev_child,
        gate,
        gate.accepted,
        screening_policy_hash,
    )


def _parse_policy(raw: Mapping[str, object]) -> DecisionPolicy:
    if set(raw) != set(_FIELDS):
        raise SelectorEvolutionError("invalid DecisionPolicy fields")
    ranking = raw["ranking_order"]
    if not isinstance(ranking, (list, tuple)):
        raise SelectorEvolutionError("ranking_order must be a sequence")
    try:
        return DecisionPolicy(
            ranking_order=tuple(str(value) for value in ranking),
            recent_regime_first=_strict_bool(raw["recent_regime_first"]),
            min_successful_folds=_strict_int(raw["min_successful_folds"]),
            catastrophic_mase=_finite_float(raw["catastrophic_mase"]),
            ensemble_enabled=_strict_bool(raw["ensemble_enabled"]),
            ensemble_max_members=_strict_int(raw["ensemble_max_members"]),
            ensemble_min_diversity=_finite_float(raw["ensemble_min_diversity"]),
            ensemble_min_improvement=_finite_float(raw["ensemble_min_improvement"]),
        )
    except (TypeError, ValueError) as error:
        raise SelectorEvolutionError(f"invalid DecisionPolicy: {error}") from error


def _policy_payload(policy: DecisionPolicy) -> dict[str, object]:
    return {
        "ranking_order": list(policy.ranking_order),
        "recent_regime_first": policy.recent_regime_first,
        "min_successful_folds": policy.min_successful_folds,
        "catastrophic_mase": policy.catastrophic_mase,
        "ensemble_enabled": policy.ensemble_enabled,
        "ensemble_max_members": policy.ensemble_max_members,
        "ensemble_min_diversity": policy.ensemble_min_diversity,
        "ensemble_min_improvement": policy.ensemble_min_improvement,
    }


def _failure_summary(
    policy: DecisionPolicy, cases: Sequence[DecisionCase]
) -> dict[str, object]:
    selected_counts: dict[str, int] = {}
    failures = 0
    for case in cases:
        try:
            decision = select_numerical_forecast(
                policy,
                active_names=case.active_names,
                diagnostics=case.diagnostics,
                forecasts=case.forecasts,
            )
        except (ValueError, KeyError):
            failures += 1
            continue
        for name in decision.selected:
            selected_counts[name] = selected_counts.get(name, 0) + 1
    return {
        "selection_failures": failures,
        "selection_counts": selected_counts,
        "diagnostic_ranges": {
            field: _range(
                getattr(diagnostic, field)
                for case in cases
                for diagnostic in case.diagnostics.values()
                if diagnostic.eligible
            )
            for field in ("median_mase", "recent_mase", "worst_mase", "mase_mad")
        },
    }


def _range(values) -> list[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return [min(finite), max(finite)] if finite else []


def _strict_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("expected boolean")
    return bool(value)


def _strict_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected integer")
    return int(value)


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected finite number")
    return result
