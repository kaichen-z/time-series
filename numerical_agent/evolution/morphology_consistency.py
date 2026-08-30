"""Deterministic, history-only validation for grounded morphology assumptions."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from common.metrics import joint_scaled_error

from .assumptions import ForecastAssumption, rank_diverse_assumptions
from .morphology import AssumptionGrounding, MorphologyCard
from .numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastFold,
    passes_independent_scaled_regret,
)
from .screening import TaskProfile


@dataclass(frozen=True)
class AssumptionConsistencyResult:
    """Accepted grounded assumptions and typed reasons for advisory rejections."""

    accepted: tuple[AssumptionGrounding, ...]
    rejected: Mapping[str, str]

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted)
        rejected = dict(self.rejected)
        if any(not isinstance(item, AssumptionGrounding) for item in accepted):
            raise ValueError("accepted assumptions must be grounded artifacts")
        if len({item.assumption_id for item in accepted}) != len(accepted):
            raise ValueError("accepted assumptions must have unique ids")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in rejected.items()):
            raise ValueError("rejection trace must contain string ids and reasons")
        if set(item.assumption_id for item in accepted) & set(rejected):
            raise ValueError("an assumption cannot be both accepted and rejected")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejected", MappingProxyType(rejected))


def check_morphology_assumptions(
    card: MorphologyCard,
    *,
    profile: TaskProfile,
    active_names: Sequence[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    policy: DecisionPolicy | None = None,
    min_successful_folds: int | None = None,
    protected_anchor_name: str | None = None,
) -> AssumptionConsistencyResult:
    """Fail closed before a grounded assumption may advise numerical selection.

    This boundary deliberately consumes no assumption prose as an executable rule.  It only
    checks the canonical kind against the deterministic profile and requires every nominated
    candidate to have already passed the existing reliability prerequisites.  It never alters
    the candidate dictionary, forecasts, or a protected fallback.
    """
    if not isinstance(card, MorphologyCard):
        raise ValueError("card must be a MorphologyCard")
    assumptions = card.assumptions
    rejected: dict[str, str] = {}
    if not isinstance(profile, TaskProfile):
        return _reject_all(assumptions, "invalid_profile")
    if policy is None:
        policy = DecisionPolicy()
    if not isinstance(policy, DecisionPolicy):
        return _reject_all(assumptions, "invalid_policy")
    minimum = _effective_minimum_folds(policy, min_successful_folds)
    if minimum is None:
        return _reject_all(assumptions, "invalid_min_successful_folds")
    active = _active_set(active_names)
    if active is None:
        return _reject_all(assumptions, "invalid_active_names")
    if not isinstance(diagnostics, Mapping):
        return _reject_all(assumptions, "invalid_diagnostics")
    if not isinstance(forecasts, Mapping):
        return _reject_all(assumptions, "invalid_forecasts")
    if protected_anchor_name is not None and (
        not isinstance(protected_anchor_name, str)
        or not protected_anchor_name
        or protected_anchor_name not in active
    ):
        return _reject_all(assumptions, "invalid_protected_anchor")
    stable_diagnostics = _stable_diagnostics(
        assumptions,
        diagnostics,
        required_names=(protected_anchor_name,) if protected_anchor_name else (),
    )
    if stable_diagnostics is None:
        return _reject_all(assumptions, "invalid_diagnostics")
    stable_forecasts = _stable_forecasts(assumptions, forecasts)
    if stable_forecasts is None:
        return _reject_all(assumptions, "invalid_forecasts")
    protected_anchor = (
        stable_diagnostics.get(protected_anchor_name)
        if protected_anchor_name is not None
        else None
    )
    if protected_anchor_name is not None and not _valid_diagnostic(
        protected_anchor, protected_anchor_name
    ):
        return _reject_all(assumptions, "invalid_protected_anchor")

    survivors: list[AssumptionGrounding] = []
    for assumption in assumptions:
        reason = _supporting_window_reason(assumption, card, profile.history_length)
        if reason is None:
            reason = _profile_reason(assumption, profile)
        if reason is None and assumption.prior_confidence < policy.assumption_min_confidence:
            reason = "below_min_confidence"
        if reason is None:
            reason = _candidate_reason(
                assumption,
                active=active,
                diagnostics=stable_diagnostics,
                forecasts=stable_forecasts,
                minimum_folds=minimum,
                policy=policy,
                horizon=profile.horizon,
                protected_anchor=protected_anchor,
            )
        if reason is None:
            survivors.append(assumption)
        else:
            rejected[assumption.assumption_id] = reason

    ranked = rank_diverse_assumptions(
        tuple(_as_rankable(assumption) for assumption in survivors),
        stable_diagnostics,
        top_k=policy.assumption_top_k,
        candidates_per_assumption=policy.assumption_candidates_per_hypothesis,
        min_confidence=policy.assumption_min_confidence,
    )
    accepted_ids = {item.assumption.assumption_id for item in ranked}
    for assumption in survivors:
        if assumption.assumption_id not in accepted_ids:
            rejected[assumption.assumption_id] = "diversity_rejected"
    accepted = tuple(assumption for assumption in survivors if assumption.assumption_id in accepted_ids)
    return AssumptionConsistencyResult(accepted=accepted, rejected=rejected)


def _reject_all(
    assumptions: Sequence[AssumptionGrounding], reason: str
) -> AssumptionConsistencyResult:
    return AssumptionConsistencyResult(
        accepted=(),
        rejected={assumption.assumption_id: reason for assumption in assumptions},
    )


def _effective_minimum_folds(policy: DecisionPolicy, requested: int | None) -> int | None:
    if requested is None:
        return policy.min_successful_folds
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        return None
    # Advisory morphology must not weaken the selector's existing reliability gate.
    return max(policy.min_successful_folds, requested)


def _active_set(active_names: Sequence[str]) -> frozenset[str] | None:
    if isinstance(active_names, (str, bytes)):
        return None
    try:
        names = tuple(active_names)
    except (TypeError, ValueError):
        return None
    if not names or any(not isinstance(name, str) or not name or name.strip() != name for name in names):
        return None
    if len(names) != len(set(names)):
        return None
    return frozenset(names)


def _stable_diagnostics(
    assumptions: Sequence[AssumptionGrounding],
    diagnostics: Mapping[str, CandidateDiagnostics],
    *,
    required_names: Sequence[str] = (),
) -> dict[str, CandidateDiagnostics] | None:
    """Reject changing mappings, then rank only a stable local diagnostic snapshot."""
    stable: dict[str, CandidateDiagnostics] = {}
    try:
        names = dict.fromkeys(
            (
                candidate
                for assumption in assumptions
                for candidate in assumption.candidate_names
            ),
        )
        names.update(dict.fromkeys(required_names))
        for name in names:
            value = diagnostics.get(name)
            if diagnostics.get(name) != value:
                return None
            if value is not None:
                stable[name] = value
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    return stable


def _stable_forecasts(
    assumptions: Sequence[AssumptionGrounding], forecasts: Mapping[str, Sequence[float]]
) -> dict[str, Sequence[float]] | None:
    """Reject changing mappings before a forecast container can affect an advisory gate."""
    stable: dict[str, Sequence[float]] = {}
    try:
        names = dict.fromkeys(
            candidate
            for assumption in assumptions
            for candidate in assumption.candidate_names
        )
        for name in names:
            value = forecasts.get(name)
            if forecasts.get(name) != value:
                return None
            if value is not None:
                stable[name] = value
    except Exception:
        return None
    return stable


def _profile_reason(assumption: AssumptionGrounding, profile: TaskProfile) -> str | None:
    compatible = {
        "seasonality": bool(profile.periodicity_periods) and profile.periodicity_confidence >= 0.25,
        "trend": profile.trend_direction != "flat" and profile.trend_strength >= 0.25,
        "intermittency": profile.zero_fraction > 0.3 or profile.intermittency_adi > 1.32,
        "regime": (
            profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.25
        ),
        "noise": profile.noise_relative_scale >= 0.25 or profile.outlier_fraction >= 0.1,
        "level": profile.likely_stationary or (
            profile.trend_direction == "flat" and profile.trend_strength < 0.25
        ),
    }
    return None if compatible.get(assumption.kind, False) else "profile_incompatible"


def _supporting_window_reason(
    assumption: AssumptionGrounding, card: MorphologyCard, history_length: int
) -> str | None:
    calls = {call.call_id: call for call in card.tool_calls}
    cited = tuple(calls[call_id] for call_id in assumption.supporting_call_ids)
    has_full_history = any(call.start == 0 and call.end == history_length for call in cited)
    has_recent = any(call.start > 0 and call.end == history_length for call in cited)
    return None if has_full_history and has_recent else "insufficient_window_evidence"


def _candidate_reason(
    assumption: AssumptionGrounding,
    *,
    active: frozenset[str],
    diagnostics: Mapping[str, CandidateDiagnostics],
    forecasts: Mapping[str, Sequence[float]],
    minimum_folds: int,
    policy: DecisionPolicy,
    horizon: int,
    protected_anchor: CandidateDiagnostics | None,
) -> str | None:
    for name in assumption.candidate_names:
        if name not in active:
            return "inactive_candidate"
        try:
            diagnostic = diagnostics.get(name)
        except (AttributeError, TypeError, ValueError):
            return "invalid_diagnostics"
        if not _valid_diagnostic(diagnostic, name):
            return "invalid_diagnostics"
        assert diagnostic is not None
        if not diagnostic.eligible:
            return "candidate_ineligible"
        if diagnostic.successful_folds < minimum_folds:
            return "insufficient_successful_folds"
        if not _valid_fold_evidence(diagnostic):
            return "invalid_fold_evidence"
        if (
            diagnostic.explosion
            or diagnostic.worst_smae_raw > policy.catastrophic_smae_raw
            or diagnostic.worst_srmse_raw > policy.catastrophic_srmse_raw
        ):
            return "catastrophic_hindcast_tail"
        if protected_anchor is not None and not _passes_safe_anchor_regret(
            diagnostic, protected_anchor, policy
        ):
            return "safe_anchor_scaled_regret"
        if policy.long_horizon_guard_enabled:
            if diagnostic.long_horizon_coverage < policy.long_horizon_min_coverage:
                return "insufficient_long_horizon_coverage"
            if diagnostic.long_horizon_fold is None:
                return "missing_long_horizon_audit"
            if not isinstance(diagnostic.long_horizon_fold, HindcastFold):
                return "invalid_long_horizon_audit"
            if not _valid_long_horizon_audit(diagnostic):
                return "invalid_long_horizon_audit"
        try:
            forecast = forecasts.get(name)
        except (AttributeError, TypeError, ValueError):
            return "invalid_forecasts"
        if not _valid_forecast(forecast):
            return "invalid_forecast"
        assert forecast is not None
        if len(forecast) != horizon:
            return "forecast_horizon_mismatch"
    return None


def _passes_safe_anchor_regret(
    candidate: CandidateDiagnostics,
    anchor: CandidateDiagnostics,
    policy: DecisionPolicy,
) -> bool:
    """Use the runtime raw/clipping guard on complete truth-aligned folds."""
    if (
        not candidate.fold_forecasts
        or not candidate.fold_truths
        or not anchor.fold_forecasts
        or not anchor.fold_truths
        or candidate.fold_truths != anchor.fold_truths
        or len(candidate.fold_forecasts) != candidate.successful_folds
        or len(anchor.fold_forecasts) != anchor.successful_folds
        or candidate.successful_folds != anchor.successful_folds
    ):
        return False
    return passes_independent_scaled_regret(
        candidate.fold_forecasts,
        anchor.fold_forecasts,
        anchor.fold_truths,
        max_smae_regret=policy.max_smae_fold_regret,
        max_srmse_regret=policy.max_srmse_fold_regret,
    )


def _valid_diagnostic(value: object, expected_name: str) -> bool:
    if not isinstance(value, CandidateDiagnostics) or value.name != expected_name:
        return False
    if type(value.eligible) is not bool or type(value.explosion) is not bool:
        return False
    if isinstance(value.successful_folds, bool) or not isinstance(value.successful_folds, int):
        return False
    capped_pairs = (
        (value.median_joint_scaled_error, value.median_smae, value.median_srmse),
        (value.recent_joint_scaled_error, value.recent_smae, value.recent_srmse),
        (value.worst_joint_scaled_error, value.worst_smae, value.worst_srmse),
    )
    capped_numbers = (
        value.median_smae,
        value.recent_smae,
        value.worst_smae,
        value.smae_mad,
        value.median_srmse,
        value.recent_srmse,
        value.worst_srmse,
        value.srmse_mad,
    )
    numbers = (
        *(number for pair in capped_pairs for number in pair),
        value.smae_mad,
        value.srmse_mad,
        value.normalized_bias,
        value.slope_error,
        value.long_horizon_coverage,
    )
    optional_numbers = (value.phase_error, value.amplitude_ratio)
    if not all(
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and math.isfinite(number)
        for number in numbers
    ):
        return False
    if any(number < 0.0 or number > 5.0 for number in capped_numbers):
        return False
    valid_successful_folds = tuple(
        fold
        for fold in value.folds
        if fold.status == "success" and _valid_scaled_fold_metrics(fold)
    )
    if value.folds and len(valid_successful_folds) == len(value.folds):
        if not _scaled_summaries_match_folds(value, valid_successful_folds):
            return False
    elif not value.folds and any(
        joint != joint_scaled_error(smae, srmse)
        for joint, smae, srmse in capped_pairs
    ):
        # Read-only compatibility diagnostics lack serialized folds.  Their
        # joint values can only be checked against the paired summaries.
        return False
    raw_tails = (value.worst_smae_raw, value.worst_srmse_raw)
    if any(
        not isinstance(number, (int, float))
        or isinstance(number, bool)
        or math.isnan(float(number))
        or float(number) < 0.0
        or float(number) == -math.inf
        for number in raw_tails
    ):
        return False
    if value.worst_smae != min(5.0, value.worst_smae_raw):
        return False
    if value.worst_srmse != min(5.0, value.worst_srmse_raw):
        return False
    if any(
        number is not None
        and (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
        )
        for number in optional_numbers
    ):
        return False
    return True


def _scaled_summaries_match_folds(
    diagnostic: CandidateDiagnostics,
    folds: tuple[HindcastFold, ...],
) -> bool:
    smaes = tuple(float(fold.smae) for fold in folds if fold.smae is not None)
    srmses = tuple(float(fold.srmse) for fold in folds if fold.srmse is not None)
    smaes_raw = tuple(
        float(fold.smae_raw) for fold in folds if fold.smae_raw is not None
    )
    srmses_raw = tuple(
        float(fold.srmse_raw) for fold in folds if fold.srmse_raw is not None
    )
    if not smaes or not (len(smaes) == len(srmses) == len(smaes_raw) == len(srmses_raw)):
        return False
    joints = tuple(
        joint_scaled_error(smae, srmse)
        for smae, srmse in zip(smaes, srmses, strict=True)
    )
    median_smae = statistics.median(smaes)
    median_srmse = statistics.median(srmses)
    return (
        diagnostic.median_joint_scaled_error == statistics.median(joints)
        and diagnostic.recent_joint_scaled_error == joints[-1]
        and diagnostic.worst_joint_scaled_error == max(joints)
        and diagnostic.median_smae == median_smae
        and diagnostic.recent_smae == smaes[-1]
        and diagnostic.worst_smae == max(smaes)
        and diagnostic.smae_mad
        == statistics.median(abs(value - median_smae) for value in smaes)
        and diagnostic.median_srmse == median_srmse
        and diagnostic.recent_srmse == srmses[-1]
        and diagnostic.worst_srmse == max(srmses)
        and diagnostic.srmse_mad
        == statistics.median(abs(value - median_srmse) for value in srmses)
        and diagnostic.worst_smae_raw == max(smaes_raw)
        and diagnostic.worst_srmse_raw == max(srmses_raw)
    )


def _valid_fold_evidence(diagnostic: CandidateDiagnostics) -> bool:
    forecasts = diagnostic.fold_forecasts
    truths = diagnostic.fold_truths
    if (
        diagnostic.successful_folds < 1
        or not isinstance(forecasts, tuple)
        or not isinstance(truths, tuple)
        or len(forecasts) != diagnostic.successful_folds
        or len(truths) != diagnostic.successful_folds
    ):
        return False
    if not all(
        _valid_forecast(forecast)
        and _valid_forecast(truth)
        and len(forecast) == len(truth)
        for forecast, truth in zip(forecasts, truths)
    ):
        return False
    # Synthetic diagnostics used by read-only compatibility callers do not carry
    # serialized fold records.  When records are present, however, their scaled
    # evidence is mandatory and must fail closed.
    return not diagnostic.folds or (
        len(diagnostic.folds) == diagnostic.successful_folds
        and tuple(fold.forecast for fold in diagnostic.folds) == forecasts
        and tuple(fold.truth for fold in diagnostic.folds) == truths
        and all(
            isinstance(fold, HindcastFold)
            and fold.status == "success"
            and _valid_scaled_fold_metrics(fold)
            for fold in diagnostic.folds
        )
    )


def _valid_long_horizon_audit(diagnostic: CandidateDiagnostics) -> bool:
    audit = diagnostic.long_horizon_fold
    return (
        isinstance(audit, HindcastFold)
        and audit.status == "success"
        and _valid_forecast(audit.forecast)
        and _valid_forecast(audit.truth)
        and len(audit.forecast) == len(audit.truth)
        and _valid_scaled_fold_metrics(audit)
    )


def _valid_scaled_fold_metrics(fold: HindcastFold) -> bool:
    return _valid_scaled_metric_triplet(
        fold.smae, fold.smae_raw, fold.smae_clipped
    ) and _valid_scaled_metric_triplet(
        fold.srmse, fold.srmse_raw, fold.srmse_clipped
    )


def _valid_scaled_metric_triplet(
    capped: object,
    raw: object,
    clipped: object,
) -> bool:
    if (
        not isinstance(capped, (int, float))
        or isinstance(capped, bool)
        or not math.isfinite(float(capped))
        or not 0.0 <= float(capped) <= 5.0
        or not isinstance(raw, (int, float))
        or isinstance(raw, bool)
        or math.isnan(float(raw))
        or float(raw) < 0.0
        or float(raw) == -math.inf
        or type(clipped) is not bool
    ):
        return False
    return (
        float(capped) == min(5.0, float(raw))
        and clipped == (float(raw) > 5.0)
    )


def _valid_forecast(value: object) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError, OverflowError):
        return False


def _as_rankable(assumption: AssumptionGrounding) -> ForecastAssumption:
    return ForecastAssumption(
        assumption_id=assumption.assumption_id,
        kind=assumption.kind,
        claim=assumption.claim,
        supporting_signals=assumption.supporting_call_ids,
        failure_condition=assumption.failure_condition,
        candidate_names=assumption.candidate_names,
        prior_confidence=assumption.prior_confidence,
    )
