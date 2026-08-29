"""Deterministic, history-only validation for grounded morphology assumptions."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .assumptions import ForecastAssumption, rank_diverse_assumptions
from .morphology import AssumptionGrounding, MorphologyCard
from .numerical_selector import CandidateDiagnostics, DecisionPolicy
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
    stable_diagnostics = _stable_diagnostics(assumptions, diagnostics)
    if stable_diagnostics is None:
        return _reject_all(assumptions, "invalid_diagnostics")
    stable_forecasts = _stable_forecasts(assumptions, forecasts)
    if stable_forecasts is None:
        return _reject_all(assumptions, "invalid_forecasts")

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
    assumptions: Sequence[AssumptionGrounding], diagnostics: Mapping[str, CandidateDiagnostics]
) -> dict[str, CandidateDiagnostics] | None:
    """Reject changing mappings, then rank only a stable local diagnostic snapshot."""
    stable: dict[str, CandidateDiagnostics] = {}
    try:
        names = dict.fromkeys(
            candidate
            for assumption in assumptions
            for candidate in assumption.candidate_names
        )
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
        if diagnostic.explosion or diagnostic.worst_mase > policy.catastrophic_mase:
            return "catastrophic_hindcast_tail"
        if policy.long_horizon_guard_enabled:
            if diagnostic.long_horizon_coverage < policy.long_horizon_min_coverage:
                return "insufficient_long_horizon_coverage"
            if not _valid_long_horizon_audit(diagnostic):
                return "missing_long_horizon_audit"
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


def _valid_diagnostic(value: object, expected_name: str) -> bool:
    if not isinstance(value, CandidateDiagnostics) or value.name != expected_name:
        return False
    if type(value.eligible) is not bool or type(value.explosion) is not bool:
        return False
    if isinstance(value.successful_folds, bool) or not isinstance(value.successful_folds, int):
        return False
    numbers = (
        value.median_mase,
        value.recent_mase,
        value.worst_mase,
        value.mase_mad,
        value.median_mae,
        value.median_smape,
        value.median_rmsse,
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
    return all(
        _valid_forecast(forecast)
        and _valid_forecast(truth)
        and len(forecast) == len(truth)
        for forecast, truth in zip(forecasts, truths)
    )


def _valid_long_horizon_audit(diagnostic: CandidateDiagnostics) -> bool:
    audit = diagnostic.long_horizon_fold
    return (
        audit is not None
        and audit.status == "success"
        and _valid_forecast(audit.forecast)
        and _valid_forecast(audit.truth)
        and len(audit.forecast) == len(audit.truth)
        and audit.mase_scale is not None
        and isinstance(audit.mase_scale, (int, float))
        and not isinstance(audit.mase_scale, bool)
        and math.isfinite(audit.mase_scale)
        and audit.mase_scale > 0.0
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
