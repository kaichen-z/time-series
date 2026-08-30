"""Typed self-evolution and trusted evaluation for the Numerical Selector."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import pprint
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.metrics import (
    drcik_point_metrics,
    joint_scaled_error,
    linear_quantile,
    mae,
    mase,
    pareto_scaled_improvement,
    smape,
    standard_error,
)

from .execution import Task
from .numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    SelectionDecision,
    _long_horizon_route_matches,
    select_assumption_guided_forecast,
)
from .screening import profile_task


SELECTOR_SYSTEM = """You are a bounded Meta-Harness Engineer for one history-only Numerical
Selector. You receive only aggregate Train diagnostics. Propose one conservative DecisionPolicy.
You may change only the supplied baseline strategy, ranking order, recent-regime preference,
minimum completed folds, independent raw sMAE/sRMSE catastrophe and fold-regret thresholds,
and guarded TSFM-plus-statistical combination thresholds, weight grids,
clipped residual strengths, and one typed task-conditioned long-horizon audit route. You cannot change identities,
screening, task partition, measurements, evaluator, runtime, cache, code, or labels. Return exactly
one JSON object with keys summary and policy. The policy must contain every allowed field exactly
once and no other fields. Never set min_successful_folds above the supplied
available_hindcast_folds. Use prior rejection feedback to propose a materially different policy.
Every ensemble_weight_grid value must be strictly between 0.5 and 1.0; every
ensemble_residual_strengths value must be in (0, 0.5]; ensemble_max_members must be between one and three;
ensemble_min_fold_wins must be a positive integer no larger than the available folds.
baseline_strategy must be toto_first, minimax_tsfm, conservative_tsfm,
conservative_combined, conservative_single_tsfm, conservative_tsfm_portfolio,
conservative_tsfm_statistical, conservative_joint_portfolio,
protected_single_tsfm, protected_tsfm_portfolio, or protected_joint_residual.
The protected_topk_single_tsfm, protected_topk_tsfm_portfolio, and
protected_topk_joint_residual variants preserve the learned minimax Top-k Parent
before applying the same challenger gates.
tsfm_router_min_improvement must be within [0, 1] and is the dedicated minimum
history-only improvement required before conservative_tsfm may route Toto to TimesFM.
tsfm_router_blend_weight must be within [0, 0.5]; zero preserves the legacy hard
route, while a positive value is a maximum cap for the adaptive 0.05/0.10/0.25
history-validated shrinkage ladder. The largest eligible weight must strictly improve
both sMAE and sRMSE on every ordinary fold and the long-horizon audit.
assumption_guidance_enabled controls whether a history-only assumption layer restricts the Verifier.
When enabled, assumption_top_k must be between 1 and 7, candidates per hypothesis between 1 and 3,
and minimum confidence within [0, 1]. Preserve reviewed TSFM anchors regardless of Top-k.
Use joint scaled error only to order proposals. Accept only Pareto improvements in
clipped Dr-CiK-aligned sMAE and sRMSE, with clipped-task counts, P90/P95 sMAE,
and full coverage as safety constraints."""


class SelectorEvolutionError(ValueError):
    """The Selector Agent crossed its typed mutation boundary."""


@dataclass(frozen=True)
class DecisionCase:
    task: Task
    active_names: tuple[str, ...]
    diagnostics: Mapping[str, CandidateDiagnostics]
    forecasts: Mapping[str, tuple[float, ...]]
    families: Mapping[str, str]
    conditioned_names: tuple[str, ...] = ()
    group_id: str = ""


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
    mean_smae: float = math.inf
    median_smae: float = math.inf
    se_smae: float = math.inf
    mean_srmse: float = math.inf
    median_srmse: float = math.inf
    se_srmse: float = math.inf
    p90_smae: float = math.inf
    p95_smae: float = math.inf
    p90_srmse: float = math.inf
    p95_srmse: float = math.inf
    p90_smae_raw: float = math.inf
    p95_smae_raw: float = math.inf
    p90_srmse_raw: float = math.inf
    p95_srmse_raw: float = math.inf
    smae_clipped_count: int = 0
    smae_clipped_rate: float = 1.0
    srmse_clipped_count: int = 0
    srmse_clipped_rate: float = 1.0
    mean_assumption_count: float = 0.0
    mean_considered_candidates: float = 0.0
    mean_considered_families: float = 0.0
    assumption_kind_diversity: int = 0
    task_scaled_pairs: Mapping[str, tuple[float, float]] = field(default_factory=dict)


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


@dataclass(frozen=True)
class SelectorTrainGeneration:
    """One Train-only proposal/search generation; it never receives Dev cases."""

    generation: int
    parent: DecisionPolicy
    proposal: DecisionPolicy
    child: DecisionPolicy
    train_parent: DecisionScore
    train_child: DecisionScore
    gate: DecisionGateResult
    accepted: bool
    screening_policy_hash: str
    candidate_count: int
    agent_calls: int = 1


@dataclass(frozen=True)
class SelectorTrainDevResult:
    """Train-evolved policy plus exactly one read-only Dev acceptance decision."""

    original_parent: DecisionPolicy
    train_winner: DecisionPolicy
    frozen: DecisionPolicy
    generations: tuple[SelectorTrainGeneration, ...]
    train_parent: DecisionScore
    train_winner_score: DecisionScore
    dev_parent: DecisionScore
    dev_winner: DecisionScore
    final_gate: DecisionGateResult


_FIELDS = (
    "ranking_order",
    "recent_regime_first",
    "min_successful_folds",
    "catastrophic_smae_raw",
    "catastrophic_srmse_raw",
    "max_smae_fold_regret",
    "max_srmse_fold_regret",
    "baseline_strategy",
    "tsfm_router_min_improvement",
    "tsfm_router_blend_weight",
    "assumption_guidance_enabled",
    "assumption_top_k",
    "assumption_candidates_per_hypothesis",
    "assumption_min_confidence",
    "ensemble_enabled",
    "ensemble_max_members",
    "ensemble_min_diversity",
    "ensemble_min_improvement",
    "ensemble_weight_grid",
    "ensemble_residual_strengths",
    "ensemble_correction_clip",
    "ensemble_min_fold_wins",
    "ensemble_max_worst_fold_regret",
    "long_horizon_audit_enabled",
    "long_horizon_penalty_weight",
    "long_horizon_route_feature",
    "long_horizon_route_operator",
    "long_horizon_route_threshold",
    "long_horizon_guard_enabled",
    "long_horizon_min_coverage",
    "long_horizon_max_regret",
    "fallback_to_best_available",
)

_SCALED_SAFETY_FIELDS = frozenset(
    {
        "catastrophic_smae_raw",
        "catastrophic_srmse_raw",
        "max_smae_fold_regret",
        "max_srmse_fold_regret",
    }
)
_MUTATION_FIELDS = _FIELDS

_ASSUMPTION_FIELDS = frozenset(
    {
        "assumption_guidance_enabled",
        "assumption_top_k",
        "assumption_candidates_per_hypothesis",
        "assumption_min_confidence",
    }
)

_TSFM_ROUTER_FIELDS = frozenset({
    "tsfm_router_min_improvement",
    "tsfm_router_blend_weight",
})
_TSFM_BLEND_FIELDS = frozenset({"tsfm_router_blend_weight"})

_LONG_HORIZON_ROUTE_FIELDS = frozenset(
    {
        "long_horizon_audit_enabled",
        "long_horizon_penalty_weight",
        "long_horizon_route_feature",
        "long_horizon_route_operator",
        "long_horizon_route_threshold",
    }
)

_LONG_HORIZON_GUARD_FIELDS = frozenset(
    {
        "long_horizon_guard_enabled",
        "long_horizon_min_coverage",
        "long_horizon_max_regret",
    }
)

_PRE_COMBINED_FIELDS = tuple(
    field for field in _FIELDS if field not in {
        "baseline_strategy",
        *_ASSUMPTION_FIELDS,
        *_LONG_HORIZON_ROUTE_FIELDS,
        *_LONG_HORIZON_GUARD_FIELDS,
        "ensemble_weight_grid",
        "ensemble_residual_strengths",
        "ensemble_correction_clip",
        "ensemble_min_fold_wins",
        "ensemble_max_worst_fold_regret",
    }
)


def render_decision_source(policy: DecisionPolicy, *, screening_policy_hash: str = "") -> str:
    payload = _policy_payload(policy)
    return (
        '"""Frozen task-conditioned numerical Decision policy."""\n\n'
        f"SCREENING_POLICY_HASH = {screening_policy_hash!r}\n"
        f"DECISION_POLICY = {pprint.pformat(payload, width=100, sort_dicts=False)}\n"
    )


def parse_decision_source(
    source: str, *, allow_legacy: bool = False
) -> DecisionPolicy:
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
    return _parse_policy(payload, allow_legacy=allow_legacy)


def decision_policy_hash(policy: DecisionPolicy, *, screening_policy_hash: str = "") -> str:
    return hashlib.sha256(
        render_decision_source(policy, screening_policy_hash=screening_policy_hash).encode("utf-8")
    ).hexdigest()


def apply_decision_response(parent: DecisionPolicy, response: str) -> DecisionPolicy:
    payload = parse_json_object(response)
    if set(payload) != {"summary", "policy"} or not isinstance(payload["summary"], str):
        raise SelectorEvolutionError("response must contain exactly summary and policy")
    raw = payload["policy"]
    mutation_fields = set(_MUTATION_FIELDS)
    accepted_fields = (
        mutation_fields,
        mutation_fields - {"baseline_strategy"},
        mutation_fields - _ASSUMPTION_FIELDS,
        mutation_fields - _ASSUMPTION_FIELDS - {"baseline_strategy"},
        mutation_fields - _LONG_HORIZON_ROUTE_FIELDS,
        mutation_fields - _LONG_HORIZON_ROUTE_FIELDS - {"baseline_strategy"},
        mutation_fields - _LONG_HORIZON_ROUTE_FIELDS - _ASSUMPTION_FIELDS,
        mutation_fields - _LONG_HORIZON_ROUTE_FIELDS - _ASSUMPTION_FIELDS - {"baseline_strategy"},
    )
    accepted_fields = (
        *accepted_fields,
        *(fields - _LONG_HORIZON_GUARD_FIELDS for fields in accepted_fields),
    )
    accepted_fields = (
        *accepted_fields,
        *(fields - _TSFM_ROUTER_FIELDS for fields in accepted_fields),
    )
    accepted_fields = (
        *accepted_fields,
        *(fields - _SCALED_SAFETY_FIELDS for fields in accepted_fields),
    )
    if not isinstance(raw, Mapping) or set(raw) not in accepted_fields:
        raise SelectorEvolutionError("policy must contain exactly the approved fields")
    normalized = dict(raw)
    for field in _SCALED_SAFETY_FIELDS:
        normalized.setdefault(field, getattr(parent, field))
    return _parse_policy(normalized)


def evaluate_decision(
    policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
) -> DecisionScore:
    """Trusted evaluation: the Selector is frozen before this function reads labels."""
    records: list[
        tuple[
            DecisionCase,
            SelectionDecision,
            float,
            float,
            float,
            float,
            Mapping[str, float | bool],
        ]
    ] = []
    for case in cases:
        try:
            decision = select_assumption_guided_forecast(
                policy,
                profile=profile_task(case.task),
                active_names=case.active_names,
                diagnostics=case.diagnostics,
                forecasts=case.forecasts,
                families=case.families,
                history=case.task.history,
                conditioned_names=case.conditioned_names,
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
        point = drcik_point_metrics(truth, prediction)
        active_scores = [
            float(drcik_point_metrics(truth, list(case.forecasts[name]))["smae"])
            for name in case.active_names
            if name in case.forecasts and len(case.forecasts[name]) == len(truth)
        ]
        task_smae = float(point["smae"])
        oracle = min(active_scores, default=task_smae)
        regret = (task_smae - oracle) / (1.0 + oracle)
        records.append((case, decision, task_mase, task_mae, task_smape, regret, point))

    mases = [record[2] for record in records]
    maes = [record[3] for record in records]
    smapes = [record[4] for record in records]
    smaes = [float(record[6]["smae"]) for record in records]
    srmses = [float(record[6]["srmse"]) for record in records]
    smaes_raw = [float(record[6]["smae_raw"]) for record in records]
    srmses_raw = [float(record[6]["srmse_raw"]) for record in records]
    smae_clipped_count = sum(bool(record[6]["smae_clipped"]) for record in records)
    srmse_clipped_count = sum(bool(record[6]["srmse_clipped"]) for record in records)
    selected = {name for _, decision, *_ in records for name in decision.selected}
    families = {
        case.families.get(name, "unknown")
        for case, decision, *_ in records
        for name in decision.selected
    }
    count = len(cases)
    completed = len(records)
    assumption_kinds = {
        kind for _, decision, *_ in records for kind in decision.assumption_kinds
    }
    assumption_counts = [len(decision.assumption_ids) for _, decision, *_ in records]
    considered_counts = [len(decision.considered_candidates) for _, decision, *_ in records]
    considered_family_counts = [
        len({case.families.get(name, "unknown") for name in decision.considered_candidates})
        for case, decision, *_ in records
    ]
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
            sum(record[1].mode in {"ensemble", "combined"} for record in records) / completed
            if completed else 0.0
        ),
        fallback_rate=(
            sum(
                "conservative_best_available_fallback" in record[1].reason_codes
                for record in records
            ) / completed
            if completed else 0.0
        ),
        mean_smae=statistics.fmean(smaes) if smaes else math.inf,
        median_smae=statistics.median(smaes) if smaes else math.inf,
        se_smae=standard_error(smaes) if smaes else math.inf,
        mean_srmse=statistics.fmean(srmses) if srmses else math.inf,
        median_srmse=statistics.median(srmses) if srmses else math.inf,
        se_srmse=standard_error(srmses) if srmses else math.inf,
        p90_smae=linear_quantile(smaes, 0.90) if smaes else math.inf,
        p95_smae=linear_quantile(smaes, 0.95) if smaes else math.inf,
        p90_srmse=linear_quantile(srmses, 0.90) if srmses else math.inf,
        p95_srmse=linear_quantile(srmses, 0.95) if srmses else math.inf,
        p90_smae_raw=linear_quantile(smaes_raw, 0.90) if smaes_raw else math.inf,
        p95_smae_raw=linear_quantile(smaes_raw, 0.95) if smaes_raw else math.inf,
        p90_srmse_raw=linear_quantile(srmses_raw, 0.90) if srmses_raw else math.inf,
        p95_srmse_raw=linear_quantile(srmses_raw, 0.95) if srmses_raw else math.inf,
        smae_clipped_count=smae_clipped_count,
        smae_clipped_rate=smae_clipped_count / completed if completed else 1.0,
        srmse_clipped_count=srmse_clipped_count,
        srmse_clipped_rate=srmse_clipped_count / completed if completed else 1.0,
        mean_assumption_count=(statistics.fmean(assumption_counts) if records else 0.0),
        mean_considered_candidates=(statistics.fmean(considered_counts) if records else 0.0),
        mean_considered_families=(statistics.fmean(considered_family_counts) if records else 0.0),
        assumption_kind_diversity=len(assumption_kinds),
        task_scaled_pairs={
            record[0].task.task_id: (
                float(record[6]["smae"]),
                float(record[6]["srmse"]),
            )
            for record in records
        },
    )


def compare_decisions(
    train_parent: DecisionScore,
    train_child: DecisionScore,
    dev_parent: DecisionScore,
    dev_child: DecisionScore,
) -> DecisionGateResult:
    if train_child.coverage < 1.0 - 1e-12 or dev_child.coverage < 1.0 - 1e-12:
        return DecisionGateResult(False, "Train and Dev coverage must remain 100%")
    if not pareto_scaled_improvement(
        dev_parent.mean_smae,
        dev_parent.mean_srmse,
        dev_child.mean_smae,
        dev_child.mean_srmse,
    ):
        return DecisionGateResult(False, "Dev scaled metric pair did not Pareto-improve")
    if dev_child.smae_clipped_count > dev_parent.smae_clipped_count:
        return DecisionGateResult(False, "Dev clipped-sMAE task count increased")
    if dev_child.srmse_clipped_count > dev_parent.srmse_clipped_count:
        return DecisionGateResult(False, "Dev clipped-sRMSE task count increased")
    if not _tail_is_safe(dev_child.p90_smae, dev_parent.p90_smae):
        return DecisionGateResult(False, "Dev P90 sMAE materially increased")
    if not _tail_is_safe(dev_child.p95_smae, dev_parent.p95_smae):
        return DecisionGateResult(False, "Dev P95 sMAE materially increased")
    if dev_child.mean_active_oracle_regret > dev_parent.mean_active_oracle_regret + 1e-12:
        return DecisionGateResult(False, "Dev active-oracle regret increased")
    if not pareto_scaled_improvement(
        train_parent.mean_smae,
        train_parent.mean_srmse,
        train_child.mean_smae,
        train_child.mean_srmse,
    ):
        return DecisionGateResult(False, "Train scaled metric pair did not Pareto-improve")
    return DecisionGateResult(True, "Train proposal passed all read-only Dev gates")


def _tail_is_safe(child: float, parent: float, *, relative_tolerance: float = 0.01) -> bool:
    """Permit at most a one-percent increase in a high-quantile tail statistic."""
    return child <= parent * (1.0 + relative_tolerance) + 1e-12


def bounded_combined_candidates(
    parent: DecisionPolicy,
    proposal: DecisionPolicy,
    *,
    available_hindcast_folds: int,
) -> tuple[DecisionPolicy, ...]:
    """Expand one LLM proposal into a small deterministic Combined neighborhood."""
    if available_hindcast_folds < 1:
        raise ValueError("available_hindcast_folds must be positive")
    if proposal.min_successful_folds > available_hindcast_folds:
        raise SelectorEvolutionError(
            "min_successful_folds exceeds available hindcast folds "
            f"({proposal.min_successful_folds} > {available_hindcast_folds})"
        )

    ranking_orders = (
        proposal.ranking_order,
        parent.ranking_order,
        (
            "worst_joint_scaled_error",
            "recent_joint_scaled_error",
            "median_joint_scaled_error",
            "median_srmse",
            "smae_mad",
        ),
    )
    fold_requirements = (
        available_hindcast_folds,
        min(2, available_hindcast_folds),
    )
    operators = (
        ((0.7, 0.8, 0.9), ()),
        ((), (0.1, 0.25)),
        ((0.8, 0.9), (0.1, 0.25)),
    )
    safety_gates = (
        (0.05, 0.00, min(2, available_hindcast_folds), 0.10),
        (0.10, 0.02, min(2, available_hindcast_folds), 0.05),
        (0.10, 0.05, available_hindcast_folds, 0.02),
    )

    candidates: list[DecisionPolicy] = []
    seen: set[DecisionPolicy] = set()

    def add(candidate: DecisionPolicy) -> None:
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    add(proposal)
    add(replace(proposal, baseline_strategy="minimax_tsfm"))
    for baseline_strategy in ("toto_first", "minimax_tsfm"):
        for top_k, per_assumption, confidence in (
            (3, 1, 0.25),
            (3, 2, 0.35),
            (5, 1, 0.25),
            (5, 2, 0.35),
        ):
            add(replace(
                proposal,
                baseline_strategy=baseline_strategy,
                assumption_guidance_enabled=True,
                assumption_top_k=top_k,
                assumption_candidates_per_hypothesis=per_assumption,
                assumption_min_confidence=confidence,
            ))
    for ranking in ranking_orders:
        for minimum_folds in fold_requirements:
            for weight_grid, residual_strengths in operators:
                for diversity, improvement, fold_wins, worst_regret in safety_gates:
                    add(replace(
                        proposal,
                        ranking_order=ranking,
                        min_successful_folds=minimum_folds,
                        ensemble_enabled=True,
                        ensemble_max_members=2,
                        ensemble_min_diversity=diversity,
                        ensemble_min_improvement=improvement,
                        ensemble_weight_grid=weight_grid,
                        ensemble_residual_strengths=residual_strengths,
                        ensemble_min_fold_wins=fold_wins,
                        ensemble_max_worst_fold_regret=worst_regret,
                    ))
    return tuple(candidates)


def bounded_long_horizon_route_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Enumerate an interpretable Train-only decision-stump neighborhood."""
    thresholds = {
        "audit_coverage": (0.5, 0.75, 1.0),
        "horizon_ratio": (0.1, 0.25, 0.5),
        "history_length": (48.0, 96.0, 192.0),
        "horizon": (12.0, 24.0, 56.0),
        "trend_strength": (0.25, 0.5, 0.75),
        "periodicity_strength": (0.25, 0.5, 0.75),
        "recent_regime_confidence": (0.25, 0.5, 0.75),
        "noise_relative_scale": (0.25, 0.5, 1.0),
        "intermittency_adi": (1.32, 2.0, 4.0),
        "zero_fraction": (0.1, 0.3, 0.6),
    }
    candidates = [parent]
    for feature, values in thresholds.items():
        for operator in ("at_least", "at_most"):
            for threshold in values:
                for weight in (0.25, 0.5, 1.0):
                    candidates.append(replace(
                        parent,
                        long_horizon_audit_enabled=True,
                        long_horizon_penalty_weight=weight,
                        long_horizon_route_feature=feature,
                        long_horizon_route_operator=operator,
                        long_horizon_route_threshold=threshold,
                    ))
    return tuple(dict.fromkeys(candidates))


def bounded_baseline_guard_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Return the parent plus four predeclared change-aware guard children."""
    children = tuple(
        replace(
            parent,
            baseline_strategy=baseline_strategy,
            long_horizon_guard_enabled=True,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=max_regret,
        )
        for baseline_strategy in ("toto_first", "minimax_tsfm")
        for max_regret in (0.0, 0.02)
    )
    return (parent, *children)


def bounded_conservative_tsfm_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Return two bounded adaptive-overlay caps without an open parameter search."""
    children = tuple(
        replace(
            parent,
            baseline_strategy="conservative_tsfm",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=blend_weight,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        )
        for blend_weight in (0.1, 0.25)
    )
    return (parent, *children)


def bounded_conservative_combined_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Return two bounded Statistical-on-TSFM overlay caps for Train-only search."""
    if parent.baseline_strategy != "toto_first" or parent.long_horizon_guard_enabled:
        return (parent,)
    children = tuple(
        replace(
            parent,
            baseline_strategy="conservative_combined",
            tsfm_router_min_improvement=0.02,
            tsfm_router_blend_weight=blend_weight,
            long_horizon_min_coverage=0.75,
            long_horizon_max_regret=0.0,
        )
        for blend_weight in (0.1, 0.25)
    )
    return (parent, *children)


def bounded_joint_portfolio_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Return the reference plus four fixed single/multi-TSFM/statistical ablations."""
    if parent.baseline_strategy != "toto_first" or parent.long_horizon_guard_enabled:
        return (parent,)
    common = {
        "ensemble_enabled": False,
        "recent_regime_first": False,
        "tsfm_router_min_improvement": 0.02,
        "long_horizon_guard_enabled": True,
        "long_horizon_min_coverage": 0.75,
        "long_horizon_max_regret": 0.0,
    }
    return (
        parent,
        replace(
            parent,
            **common,
            baseline_strategy="conservative_single_tsfm",
            tsfm_router_blend_weight=0.0,
        ),
        replace(
            parent,
            **common,
            baseline_strategy="conservative_tsfm_portfolio",
            tsfm_router_blend_weight=0.5,
        ),
        replace(
            parent,
            **common,
            baseline_strategy="conservative_tsfm_statistical",
            tsfm_router_blend_weight=0.25,
        ),
        replace(
            parent,
            **common,
            baseline_strategy="conservative_joint_portfolio",
            tsfm_router_blend_weight=0.25,
        ),
    )


def bounded_protected_portfolio_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Return Parent plus R1/R2/R3 without changing Parent's normal selector."""
    if parent.baseline_strategy != "toto_first":
        return (parent,)
    common = {
        "tsfm_router_min_improvement": 0.02,
        "long_horizon_min_coverage": 0.75,
        "long_horizon_max_regret": 0.0,
    }
    return (
        parent,
        replace(parent, **common, baseline_strategy="protected_single_tsfm"),
        replace(parent, **common, baseline_strategy="protected_tsfm_portfolio"),
        replace(
            parent,
            **common,
            baseline_strategy="protected_joint_residual",
            ensemble_residual_strengths=(0.05, 0.1, 0.2),
            ensemble_correction_clip=1.0,
        ),
    )


def bounded_protected_topk_candidates(
    parent: DecisionPolicy,
) -> tuple[DecisionPolicy, ...]:
    """Protect one learned assumption-guided Parent with fixed R1/R2/R3 children."""
    if not parent.assumption_guidance_enabled:
        return (parent,)
    common = {
        "tsfm_router_min_improvement": 0.02,
        "long_horizon_min_coverage": 0.75,
        "long_horizon_max_regret": 0.0,
    }
    return (
        parent,
        replace(parent, **common, baseline_strategy="protected_topk_single_tsfm"),
        replace(parent, **common, baseline_strategy="protected_topk_tsfm_portfolio"),
        replace(
            parent,
            **common,
            baseline_strategy="protected_topk_joint_residual",
        ),
    )


def _compare_train_decisions(
    parent: DecisionScore,
    child: DecisionScore,
) -> DecisionGateResult:
    if child.coverage < 1.0 - 1e-12:
        return DecisionGateResult(False, "Train coverage must remain 100%")
    if child.smae_clipped_count > parent.smae_clipped_count:
        return DecisionGateResult(False, "Train clipped-sMAE task count increased")
    if child.srmse_clipped_count > parent.srmse_clipped_count:
        return DecisionGateResult(False, "Train clipped-sRMSE task count increased")
    if not pareto_scaled_improvement(
        parent.mean_smae,
        parent.mean_srmse,
        child.mean_smae,
        child.mean_srmse,
    ):
        return DecisionGateResult(False, "Train scaled metric pair did not Pareto-improve")
    if not _tail_is_safe(child.p90_smae, parent.p90_smae, relative_tolerance=0.05):
        return DecisionGateResult(False, "Train P90 sMAE materially increased")
    if not _tail_is_safe(child.p95_smae, parent.p95_smae, relative_tolerance=0.05):
        return DecisionGateResult(False, "Train P95 sMAE materially increased")
    if child.mean_active_oracle_regret > parent.mean_active_oracle_regret + 1e-12:
        return DecisionGateResult(False, "Train active-oracle regret increased")
    return DecisionGateResult(True, "Candidate passed all Train search gates")


def compare_train_crossfolds(
    parent_policy: DecisionPolicy,
    child_policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
    *,
    folds: int = 4,
) -> DecisionGateResult:
    """Reject Train gains that are not stable across entity-disjoint folds."""
    if folds < 2:
        raise ValueError("cross-validation folds must be at least two")
    partitions = _group_balanced_folds(cases, folds)
    if len(partitions) < 2:
        return DecisionGateResult(False, "Train cross-fold validation needs two nonempty folds")

    improvements = 0
    for index, partition in enumerate(partitions):
        parent = evaluate_decision(parent_policy, partition)
        child = evaluate_decision(child_policy, partition)
        if child.coverage < 1.0 - 1e-12:
            return DecisionGateResult(False, f"Train fold {index} coverage decreased")
        if child.smae_clipped_count > parent.smae_clipped_count:
            return DecisionGateResult(False, f"Train fold {index} clipped-sMAE count increased")
        if child.srmse_clipped_count > parent.srmse_clipped_count:
            return DecisionGateResult(False, f"Train fold {index} clipped-sRMSE count increased")
        if child.mean_smae > parent.mean_smae + 1e-12:
            return DecisionGateResult(False, f"Train fold {index} sMAE regressed")
        if child.mean_srmse > parent.mean_srmse + 1e-12:
            return DecisionGateResult(False, f"Train fold {index} sRMSE regressed")
        if pareto_scaled_improvement(
            parent.mean_smae,
            parent.mean_srmse,
            child.mean_smae,
            child.mean_srmse,
        ):
            improvements += 1

    required = math.ceil(0.75 * len(partitions))
    if improvements < required:
        return DecisionGateResult(
            False,
            f"Train cross-fold scaled pair improved in only {improvements}/{len(partitions)} folds",
        )
    return DecisionGateResult(
        True,
        f"Train cross-fold scaled pair improved in {improvements}/{len(partitions)} folds",
    )


def compare_task_conditioned_crossfolds(
    parent_policy: DecisionPolicy,
    child_policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
    *,
    folds: int = 4,
) -> DecisionGateResult:
    """Validate a sparse audit route without requiring inactive folds to improve."""
    partitions = _group_balanced_folds(cases, folds)
    score_pairs = tuple(
        (
            evaluate_decision(parent_policy, partition),
            evaluate_decision(child_policy, partition),
        )
        for partition in partitions
    )
    matched_counts = tuple(
        sum(_route_matches_case(child_policy, case) for case in partition)
        for partition in partitions
    )
    return compare_activation_aware_fold_scores(
        score_pairs,
        matched_counts=matched_counts,
        total_matched=sum(matched_counts),
        total_tasks=len(cases),
    )


def compare_change_aware_crossfolds(
    parent_policy: DecisionPolicy,
    child_policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
    *,
    folds: int = 4,
) -> DecisionGateResult:
    """Validate only tasks whose final forecast differs from the parent."""
    partitions = _group_balanced_folds(cases, folds)
    score_pairs = tuple(
        (
            evaluate_decision(parent_policy, partition),
            evaluate_decision(child_policy, partition),
        )
        for partition in partitions
    )
    matched_counts = tuple(
        len(changed_decision_task_ids(parent_policy, child_policy, partition))
        for partition in partitions
    )
    return compare_activation_aware_fold_scores(
        score_pairs,
        matched_counts=matched_counts,
        total_matched=sum(matched_counts),
        total_tasks=len(cases),
        minimum_matches=(
            2 if child_policy.tsfm_router_blend_weight > 0.0 else None
        ),
    )


def changed_decision_task_ids(
    parent_policy: DecisionPolicy,
    child_policy: DecisionPolicy,
    cases: Sequence[DecisionCase],
) -> tuple[str, ...]:
    changed = []
    for case in cases:
        parent = _select_case(parent_policy, case)
        child = _select_case(child_policy, case)
        if parent is not None and child is not None and parent.forecast != child.forecast:
            changed.append(case.task.task_id)
    return tuple(changed)


def _select_case(
    policy: DecisionPolicy,
    case: DecisionCase,
) -> SelectionDecision | None:
    try:
        return select_assumption_guided_forecast(
            policy,
            profile=profile_task(case.task),
            active_names=case.active_names,
            diagnostics=case.diagnostics,
            forecasts=case.forecasts,
            families=case.families,
            history=case.task.history,
            conditioned_names=case.conditioned_names,
        )
    except (ValueError, KeyError):
        return None


def compare_activation_aware_fold_scores(
    score_pairs: Sequence[tuple[DecisionScore, DecisionScore]],
    *,
    matched_counts: Sequence[int],
    total_matched: int,
    total_tasks: int,
    minimum_matches: int | None = None,
) -> DecisionGateResult:
    if len(score_pairs) < 2 or len(score_pairs) != len(matched_counts):
        return DecisionGateResult(False, "Activation-aware validation needs aligned folds")
    minimum_matches = (
        max(2, math.ceil(0.10 * total_tasks))
        if minimum_matches is None
        else max(2, int(minimum_matches))
    )
    maximum_matches = math.floor(0.80 * total_tasks)
    if not minimum_matches <= total_matched <= maximum_matches:
        return DecisionGateResult(
            False,
            f"Route coverage {total_matched}/{total_tasks} is outside "
            f"[{minimum_matches}, {maximum_matches}]",
        )
    if sum(count > 0 for count in matched_counts) < 2:
        return DecisionGateResult(False, "Route must activate at least two entity folds")

    improvements = 0
    for index, (parent, child) in enumerate(score_pairs):
        if child.coverage < parent.coverage - 1e-12:
            return DecisionGateResult(False, f"Train fold {index} coverage decreased")
        if child.smae_clipped_count > parent.smae_clipped_count:
            return DecisionGateResult(False, f"Train fold {index} clipped-sMAE count increased")
        if child.srmse_clipped_count > parent.srmse_clipped_count:
            return DecisionGateResult(False, f"Train fold {index} clipped-sRMSE count increased")
        if child.mean_smae > parent.mean_smae + 1e-12:
            return DecisionGateResult(False, f"Train fold {index} sMAE regressed")
        if child.mean_srmse > parent.mean_srmse + 1e-12:
            return DecisionGateResult(False, f"Train fold {index} sRMSE regressed")
        if not _tail_is_safe(child.p90_smae, parent.p90_smae):
            return DecisionGateResult(False, f"Train fold {index} P90 sMAE materially increased")
        if not _tail_is_safe(child.p95_smae, parent.p95_smae):
            return DecisionGateResult(False, f"Train fold {index} P95 sMAE materially increased")
        if child.mean_active_oracle_regret > parent.mean_active_oracle_regret + 1e-12:
            return DecisionGateResult(False, f"Train fold {index} oracle regret increased")
        if pareto_scaled_improvement(
            parent.mean_smae,
            parent.mean_srmse,
            child.mean_smae,
            child.mean_srmse,
        ):
            improvements += 1
    if improvements < 2:
        return DecisionGateResult(
            False,
            f"Activation-aware scaled pair improved in only {improvements}/{len(score_pairs)} folds",
        )
    return DecisionGateResult(
        True,
        f"Activation-aware scaled pair improved in {improvements}/{len(score_pairs)} folds "
        "with no material fold regression",
    )


def _route_matches_case(policy: DecisionPolicy, case: DecisionCase) -> bool:
    coverage = min(case.task.horizon, max(1, len(case.task.history) // 3)) / case.task.horizon
    return _long_horizon_route_matches(policy, profile_task(case.task), coverage)


def _group_balanced_folds(
    cases: Sequence[DecisionCase], folds: int
) -> tuple[tuple[DecisionCase, ...], ...]:
    grouped: dict[str, list[DecisionCase]] = {}
    for case in cases:
        key = case.group_id or case.task.task_id
        grouped.setdefault(key, []).append(case)
    bucket_count = min(folds, len(grouped))
    if bucket_count == 0:
        return ()
    buckets: list[list[DecisionCase]] = [[] for _ in range(bucket_count)]
    for _, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(bucket_count), key=lambda index: (len(buckets[index]), index))
        buckets[target].extend(group)
    return tuple(tuple(bucket) for bucket in buckets if bucket)


def _train_rank(score: DecisionScore) -> tuple[float, ...]:
    return (
        joint_scaled_error(score.mean_smae, score.mean_srmse),
        score.mean_smae,
        score.mean_srmse,
        score.p95_smae,
        score.p90_smae,
        score.mean_active_oracle_regret,
    )


def _evolve_selector_on_train_once(
    parent: DecisionPolicy,
    train_cases: Sequence[DecisionCase],
    agent: LLMClient,
    *,
    generation: int,
    screening_policy_hash: str,
    transcript_dir: str | Path,
    available_hindcast_folds: int,
    train_validation_folds: int = 0,
    prior_rejections: Sequence[str] = (),
) -> SelectorTrainGeneration:
    train_parent = evaluate_decision(parent, train_cases)
    request = json.dumps(
        {
            "generation": generation,
            "screening_policy_hash": screening_policy_hash,
            "current_policy": _mutation_policy_payload(parent),
            "available_hindcast_folds": available_hindcast_folds,
            "prior_rejections": list(prior_rejections[-5:]),
            "train_summary": _scaled_train_summary(train_parent),
            "train_failure_summary": _failure_summary(parent, train_cases),
            "instruction": (
                "Propose one conservative typed policy. Python will expand it into a bounded "
                "Combined neighborhood and select using Train only."
            ),
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
    try:
        proposal = apply_decision_response(parent, response.text)
        candidates = bounded_combined_candidates(
            parent,
            proposal,
            available_hindcast_folds=available_hindcast_folds,
        )
    except (SelectorEvolutionError, ValueError) as error:
        gate = DecisionGateResult(False, f"Invalid child proposal: {error}")
        return SelectorTrainGeneration(
            generation,
            parent,
            parent,
            parent,
            train_parent,
            train_parent,
            gate,
            False,
            screening_policy_hash,
            0,
        )

    winner = parent
    winner_score = train_parent
    crossfold_rejection = ""
    for candidate in candidates:
        score = evaluate_decision(candidate, train_cases)
        gate = _compare_train_decisions(train_parent, score)
        if not gate.accepted:
            continue
        if train_validation_folds >= 2:
            crossfold = compare_train_crossfolds(
                parent,
                candidate,
                train_cases,
                folds=train_validation_folds,
            )
            if not crossfold.accepted:
                crossfold_rejection = crossfold.reason
                continue
        if _train_rank(score) < _train_rank(winner_score):
            winner = candidate
            winner_score = score
    accepted = winner != parent
    gate = (
        DecisionGateResult(True, "Best bounded candidate passed Train search gates")
        if accepted
        else DecisionGateResult(
            False,
            "No bounded candidate passed the Train search gates"
            + (f": {crossfold_rejection}" if crossfold_rejection else ""),
        )
    )
    return SelectorTrainGeneration(
        generation,
        parent,
        proposal,
        winner,
        train_parent,
        winner_score,
        gate,
        accepted,
        screening_policy_hash,
        len(candidates),
    )


def evolve_selector_train_then_dev(
    parent: DecisionPolicy,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    agent: LLMClient,
    *,
    generations: int,
    available_hindcast_folds: int,
    train_validation_folds: int = 0,
    screening_policy_hash: str,
    transcript_dir: str | Path,
) -> SelectorTrainDevResult:
    """Evolve on Train, then expose Dev once to accept or reject the frozen winner."""
    if generations < 1:
        raise ValueError("generations must be positive")
    original = parent
    current = parent
    rejected: list[str] = []
    results: list[SelectorTrainGeneration] = []
    for generation in range(1, generations + 1):
        result = _evolve_selector_on_train_once(
            current,
            train_cases,
            agent,
            generation=generation,
            screening_policy_hash=screening_policy_hash,
            transcript_dir=transcript_dir,
            available_hindcast_folds=available_hindcast_folds,
            train_validation_folds=train_validation_folds,
            prior_rejections=tuple(rejected),
        )
        results.append(result)
        if result.accepted:
            current = result.child
        else:
            rejected.append(f"Generation {generation}: {result.gate.reason}")

    train_parent = evaluate_decision(original, train_cases)
    train_winner = evaluate_decision(current, train_cases)
    dev_parent = evaluate_decision(original, dev_cases)
    dev_winner = evaluate_decision(current, dev_cases)
    final_gate = compare_decisions(
        train_parent,
        train_winner,
        dev_parent,
        dev_winner,
    )
    frozen = current if final_gate.accepted else original
    return SelectorTrainDevResult(
        original,
        current,
        frozen,
        tuple(results),
        train_parent,
        train_winner,
        dev_parent,
        dev_winner,
        final_gate,
    )


def evolve_selector_once(
    parent: DecisionPolicy,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    agent: LLMClient,
    *,
    generation: int,
    screening_policy_hash: str,
    transcript_dir: str | Path,
    available_hindcast_folds: int | None = None,
    prior_rejections: Sequence[str] = (),
) -> SelectorGeneration:
    if available_hindcast_folds is None:
        available_hindcast_folds = max(
            (
                len(diagnostic.folds)
                for case in train_cases
                for diagnostic in case.diagnostics.values()
            ),
            default=0,
        )
        if available_hindcast_folds == 0:
            available_hindcast_folds = parent.min_successful_folds
    if available_hindcast_folds < 1:
        raise ValueError("available_hindcast_folds must be positive")
    train_parent = evaluate_decision(parent, train_cases)
    dev_parent = evaluate_decision(parent, dev_cases)
    request = json.dumps(
        {
            "generation": generation,
            "screening_policy_hash": screening_policy_hash,
            "current_policy": _mutation_policy_payload(parent),
            "available_hindcast_folds": available_hindcast_folds,
            "prior_rejections": list(prior_rejections[-5:]),
            "train_summary": _scaled_train_summary(train_parent),
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
    try:
        child = apply_decision_response(parent, response.text)
        if child.min_successful_folds > available_hindcast_folds:
            raise SelectorEvolutionError(
                "min_successful_folds exceeds available hindcast folds "
                f"({child.min_successful_folds} > {available_hindcast_folds})"
            )
    except (SelectorEvolutionError, ValueError) as error:
        gate = DecisionGateResult(False, f"Invalid child proposal: {error}")
        return SelectorGeneration(
            generation,
            parent,
            parent,
            train_parent,
            train_parent,
            dev_parent,
            dev_parent,
            gate,
            False,
            screening_policy_hash,
        )
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


def evolve_selector_generations(
    parent: DecisionPolicy,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    agent: LLMClient,
    *,
    generations: int,
    available_hindcast_folds: int,
    screening_policy_hash: str,
    transcript_dir: str | Path,
) -> tuple[DecisionPolicy, tuple[SelectorGeneration, ...]]:
    """Run bounded generations while carrying rejection feedback forward."""
    if generations < 1:
        raise ValueError("generations must be positive")
    current = parent
    rejected: list[str] = []
    results: list[SelectorGeneration] = []
    for generation in range(1, generations + 1):
        result = evolve_selector_once(
            current,
            train_cases,
            dev_cases,
            agent,
            generation=generation,
            screening_policy_hash=screening_policy_hash,
            transcript_dir=transcript_dir,
            available_hindcast_folds=available_hindcast_folds,
            prior_rejections=tuple(rejected),
        )
        results.append(result)
        if result.accepted:
            current = result.child
        else:
            rejected.append(f"Generation {generation}: {result.gate.reason}")
    return current, tuple(results)


def _parse_policy(
    raw: Mapping[str, object], *, allow_legacy: bool = False
) -> DecisionPolicy:
    raw = dict(raw)
    if "catastrophic_mase" in raw:
        if not allow_legacy:
            raise SelectorEvolutionError(
                "legacy catastrophic_mase requires allow_legacy=True report-only parsing"
            )
        raw.pop("catastrophic_mase")
    legacy_fields = (
        set(_FIELDS) - _TSFM_BLEND_FIELDS,
        set(_PRE_COMBINED_FIELDS),
        set(_PRE_COMBINED_FIELDS) | _LONG_HORIZON_ROUTE_FIELDS,
        set(_PRE_COMBINED_FIELDS) | _LONG_HORIZON_ROUTE_FIELDS | _LONG_HORIZON_GUARD_FIELDS,
        set(_FIELDS) - _LONG_HORIZON_GUARD_FIELDS,
        set(_FIELDS) - {"baseline_strategy"},
        set(_FIELDS) - _ASSUMPTION_FIELDS,
        set(_FIELDS) - _ASSUMPTION_FIELDS - {"baseline_strategy"},
        set(_FIELDS) - _LONG_HORIZON_ROUTE_FIELDS,
        set(_FIELDS) - _LONG_HORIZON_ROUTE_FIELDS - {"baseline_strategy"},
        set(_FIELDS) - _LONG_HORIZON_ROUTE_FIELDS - _ASSUMPTION_FIELDS,
        set(_FIELDS) - _LONG_HORIZON_ROUTE_FIELDS - _ASSUMPTION_FIELDS - {"baseline_strategy"},
    )
    legacy_fields = (
        *legacy_fields,
        *(fields - _LONG_HORIZON_GUARD_FIELDS for fields in legacy_fields),
    )
    legacy_fields = (
        *legacy_fields,
        *(fields - _TSFM_BLEND_FIELDS for fields in legacy_fields),
    )
    legacy_fields = (
        *legacy_fields,
        *(fields - _TSFM_ROUTER_FIELDS for fields in legacy_fields),
    )
    legacy_fields = (
        *legacy_fields,
        *(fields - _SCALED_SAFETY_FIELDS for fields in legacy_fields),
    )
    if set(raw) in legacy_fields:
        defaults = _policy_payload(DecisionPolicy())
        raw = {**defaults, **raw}
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
            catastrophic_smae_raw=_finite_float(raw["catastrophic_smae_raw"]),
            catastrophic_srmse_raw=_finite_float(raw["catastrophic_srmse_raw"]),
            max_smae_fold_regret=_finite_float(raw["max_smae_fold_regret"]),
            max_srmse_fold_regret=_finite_float(raw["max_srmse_fold_regret"]),
            baseline_strategy=str(raw["baseline_strategy"]),
            tsfm_router_min_improvement=_finite_float(
                raw["tsfm_router_min_improvement"]
            ),
            tsfm_router_blend_weight=_finite_float(
                raw["tsfm_router_blend_weight"]
            ),
            assumption_guidance_enabled=_strict_bool(raw["assumption_guidance_enabled"]),
            assumption_top_k=_strict_int(raw["assumption_top_k"]),
            assumption_candidates_per_hypothesis=_strict_int(
                raw["assumption_candidates_per_hypothesis"]
            ),
            assumption_min_confidence=_finite_float(raw["assumption_min_confidence"]),
            ensemble_enabled=_strict_bool(raw["ensemble_enabled"]),
            ensemble_max_members=_strict_int(raw["ensemble_max_members"]),
            ensemble_min_diversity=_finite_float(raw["ensemble_min_diversity"]),
            ensemble_min_improvement=_finite_float(raw["ensemble_min_improvement"]),
            ensemble_weight_grid=_finite_float_tuple(raw["ensemble_weight_grid"]),
            ensemble_residual_strengths=_finite_float_tuple(
                raw["ensemble_residual_strengths"]
            ),
            ensemble_correction_clip=_finite_float(raw["ensemble_correction_clip"]),
            ensemble_min_fold_wins=_strict_int(raw["ensemble_min_fold_wins"]),
            ensemble_max_worst_fold_regret=_finite_float(
                raw["ensemble_max_worst_fold_regret"]
            ),
            long_horizon_audit_enabled=_strict_bool(raw["long_horizon_audit_enabled"]),
            long_horizon_penalty_weight=_finite_float(raw["long_horizon_penalty_weight"]),
            long_horizon_route_feature=str(raw["long_horizon_route_feature"]),
            long_horizon_route_operator=str(raw["long_horizon_route_operator"]),
            long_horizon_route_threshold=_finite_float(raw["long_horizon_route_threshold"]),
            long_horizon_guard_enabled=_strict_bool(raw["long_horizon_guard_enabled"]),
            long_horizon_min_coverage=_finite_float(raw["long_horizon_min_coverage"]),
            long_horizon_max_regret=_finite_float(raw["long_horizon_max_regret"]),
            fallback_to_best_available=_strict_bool(raw["fallback_to_best_available"]),
        )
    except (TypeError, ValueError) as error:
        raise SelectorEvolutionError(f"invalid DecisionPolicy: {error}") from error


def _policy_payload(policy: DecisionPolicy) -> dict[str, object]:
    return {
        "ranking_order": list(policy.ranking_order),
        "recent_regime_first": policy.recent_regime_first,
        "min_successful_folds": policy.min_successful_folds,
        "catastrophic_smae_raw": policy.catastrophic_smae_raw,
        "catastrophic_srmse_raw": policy.catastrophic_srmse_raw,
        "max_smae_fold_regret": policy.max_smae_fold_regret,
        "max_srmse_fold_regret": policy.max_srmse_fold_regret,
        "baseline_strategy": policy.baseline_strategy,
        "tsfm_router_min_improvement": policy.tsfm_router_min_improvement,
        "tsfm_router_blend_weight": policy.tsfm_router_blend_weight,
        "assumption_guidance_enabled": policy.assumption_guidance_enabled,
        "assumption_top_k": policy.assumption_top_k,
        "assumption_candidates_per_hypothesis": policy.assumption_candidates_per_hypothesis,
        "assumption_min_confidence": policy.assumption_min_confidence,
        "ensemble_enabled": policy.ensemble_enabled,
        "ensemble_max_members": policy.ensemble_max_members,
        "ensemble_min_diversity": policy.ensemble_min_diversity,
        "ensemble_min_improvement": policy.ensemble_min_improvement,
        "ensemble_weight_grid": list(policy.ensemble_weight_grid),
        "ensemble_residual_strengths": list(policy.ensemble_residual_strengths),
        "ensemble_correction_clip": policy.ensemble_correction_clip,
        "ensemble_min_fold_wins": policy.ensemble_min_fold_wins,
        "ensemble_max_worst_fold_regret": policy.ensemble_max_worst_fold_regret,
        "long_horizon_audit_enabled": policy.long_horizon_audit_enabled,
        "long_horizon_penalty_weight": policy.long_horizon_penalty_weight,
        "long_horizon_route_feature": policy.long_horizon_route_feature,
        "long_horizon_route_operator": policy.long_horizon_route_operator,
        "long_horizon_route_threshold": policy.long_horizon_route_threshold,
        "long_horizon_guard_enabled": policy.long_horizon_guard_enabled,
        "long_horizon_min_coverage": policy.long_horizon_min_coverage,
        "long_horizon_max_regret": policy.long_horizon_max_regret,
        "fallback_to_best_available": policy.fallback_to_best_available,
    }


def _mutation_policy_payload(policy: DecisionPolicy) -> dict[str, object]:
    """Project only live scaled mutation fields; legacy metrics stay read-only."""
    return _policy_payload(policy)


def _scaled_train_summary(score: DecisionScore) -> dict[str, object]:
    """Expose only formal scaled objectives and non-performance safety summaries."""
    payload = asdict(score)
    for field in (
        "mean_mase",
        "median_mase",
        "mean_mae",
        "median_mae",
        "mean_smape",
        "catastrophic_rate",
        "task_scaled_pairs",
    ):
        payload.pop(field)
    return payload


def _failure_summary(
    policy: DecisionPolicy, cases: Sequence[DecisionCase]
) -> dict[str, object]:
    selected_counts: dict[str, int] = {}
    failures = 0
    for case in cases:
        try:
            decision = select_assumption_guided_forecast(
                policy,
                profile=profile_task(case.task),
                active_names=case.active_names,
                diagnostics=case.diagnostics,
                forecasts=case.forecasts,
                families=case.families,
                history=case.task.history,
                conditioned_names=case.conditioned_names,
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
            for field in (
                "median_joint_scaled_error",
                "recent_joint_scaled_error",
                "worst_joint_scaled_error",
                "median_smae",
                "median_srmse",
                "worst_smae_raw",
                "worst_srmse_raw",
            )
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


def _finite_float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a sequence of finite numbers")
    return tuple(_finite_float(item) for item in value)
