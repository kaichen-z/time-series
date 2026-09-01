"""History-only execution for frozen Numerical Champion policies."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .champion import (
    ChampionRecipe,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from .numerical_selector import CandidateDiagnostics
from .screening import TaskProfile


_PROFILE_FLOAT_FEATURES = frozenset({
    "zero_fraction",
    "trend_strength",
    "periodicity_strength",
    "periodicity_confidence",
    "outlier_fraction",
    "noise_relative_scale",
    "stationarity_score",
    "recent_regime_confidence",
    "intermittency_adi",
    "intermittency_cv2",
})
_PROFILE_INTEGER_FEATURES = frozenset({"history_length", "horizon"})
_PROFILE_FLOAT_BOUNDS = {
    "zero_fraction": (0.0, 1.0),
    "trend_strength": (0.0, 1.0),
    "periodicity_strength": (0.0, 1.0),
    "periodicity_confidence": (0.0, 1.0),
    "outlier_fraction": (0.0, 1.0),
    "noise_relative_scale": (0.0, 1_000_000.0),
    "stationarity_score": (0.0, 1.0),
    "recent_regime_confidence": (0.0, 1.0),
    "intermittency_adi": (0.0, 1_000_000.0),
    "intermittency_cv2": (0.0, 1_000_000.0),
}
_DIAGNOSTIC_FLOAT_FEATURES = frozenset({
    "median_mase",
    "recent_mase",
    "worst_mase",
    "mase_mad",
    "median_mae",
    "median_smape",
    "median_rmsse",
    "normalized_bias",
    "slope_error",
    "long_horizon_coverage",
    "median_joint_scaled_error",
    "recent_joint_scaled_error",
    "worst_joint_scaled_error",
    "median_smae",
    "recent_smae",
    "worst_smae",
    "smae_mad",
    "median_srmse",
    "recent_srmse",
    "worst_srmse",
    "srmse_mad",
    "worst_smae_raw",
    "worst_srmse_raw",
})
_NONNEGATIVE_DIAGNOSTIC_FEATURES = _DIAGNOSTIC_FLOAT_FEATURES - {
    "normalized_bias",
    "slope_error",
}


@dataclass(frozen=True)
class ChampionExecution:
    """A deterministic result from one frozen Champion policy."""

    forecast: tuple[float, ...]
    selected_names: tuple[str, ...]
    activated_assumptions: tuple[str, ...]
    fallback_reason: str | None


def _mapping_value(mapping: object, key: str) -> tuple[bool, object]:
    if not isinstance(mapping, Mapping):
        return False, None
    try:
        return True, mapping[key]
    except Exception:
        return False, None


def _fallback(forecast: object, name: str | None, reason: str) -> ChampionExecution:
    return ChampionExecution(
        forecast=forecast,
        selected_names=(name,) if name is not None else (),
        activated_assumptions=(),
        fallback_reason=reason,
    )


def _valid_policy(policy: object) -> bool:
    if type(policy) is not FittedChampionPolicy:
        return False
    try:
        if type(policy.recipe) is not ChampionRecipe:
            return False
        for assumption in policy.recipe.assumptions:
            if type(assumption) is not EvolutionAssumption:
                return False
            EvolutionAssumption.__post_init__(assumption)
        ChampionRecipe.__post_init__(policy.recipe)
        FittedChampionPolicy.__post_init__(policy)
    except Exception:
        return False
    return True


def _finite_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_values(value: object, *, length: int | None = None) -> tuple[float, ...] | None:
    if type(value) not in {list, tuple}:
        return None
    if length is not None and len(value) != length:
        return None
    converted: list[float] = []
    for item in value:
        number = _finite_number(item)
        if number is None:
            return None
        converted.append(number)
    return tuple(converted)


def _valid_profile(profile: object, *, history_length: int, horizon: int) -> bool:
    if type(profile) is not TaskProfile:
        return False
    try:
        if (
            type(profile.history_length) is not int
            or type(profile.horizon) is not int
            or not 1 <= profile.history_length <= 1_000_000
            or not 1 <= profile.horizon <= 1_000_000
            or profile.history_length != history_length
            or profile.horizon != horizon
        ):
            return False
        for field, (lower, upper) in _PROFILE_FLOAT_BOUNDS.items():
            value = getattr(profile, field)
            if type(value) is not float or not math.isfinite(value):
                return False
            if not lower <= value <= upper:
                return False
        if any(
            type(getattr(profile, field)) is not bool
            for field in ("signed", "integer_valued", "likely_stationary")
        ):
            return False
        if type(profile.trend_direction) is not str:
            return False
        if type(profile.periodicity_periods) is not tuple or any(
            type(period) is not int or not 1 <= period <= 1_000_000
            for period in profile.periodicity_periods
        ):
            return False
        if profile.recent_regime_start is not None and (
            type(profile.recent_regime_start) is not int
            or not 0 <= profile.recent_regime_start < profile.history_length
        ):
            return False
    except Exception:
        return False
    return True


def _valid_diagnostic(diagnostic: object, expected_name: str) -> bool:
    if type(diagnostic) is not CandidateDiagnostics:
        return False
    try:
        if type(diagnostic.name) is not str or diagnostic.name != expected_name:
            return False
        if type(diagnostic.family) is not str or type(diagnostic.reason_code) is not str:
            return False
        if type(diagnostic.eligible) is not bool or type(diagnostic.explosion) is not bool:
            return False
        if type(diagnostic.successful_folds) is not int or diagnostic.successful_folds < 0:
            return False
        for field in _DIAGNOSTIC_FLOAT_FEATURES:
            value = getattr(diagnostic, field)
            if type(value) is not float or not math.isfinite(value):
                return False
            if field in _NONNEGATIVE_DIAGNOSTIC_FEATURES and value < 0.0:
                return False
        if not 0.0 <= diagnostic.long_horizon_coverage <= 1.0:
            return False
        for field in ("phase_error", "amplitude_ratio"):
            value = getattr(diagnostic, field)
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                return False
        if diagnostic.amplitude_ratio is not None and diagnostic.amplitude_ratio < 0.0:
            return False
    except Exception:
        return False
    return True


def _safe_diagnostic(diagnostic: CandidateDiagnostics) -> bool:
    return (
        diagnostic.eligible
        and not diagnostic.explosion
        and diagnostic.successful_folds > 0
        and diagnostic.reason_code == "ok"
    )


def _measurement(
    assumption: EvolutionAssumption,
    profile: TaskProfile,
    diagnostics: Mapping[str, CandidateDiagnostics],
) -> tuple[str | None, float | None]:
    feature = assumption.feature
    if feature == "horizon_ratio":
        return None, profile.horizon / profile.history_length
    if feature in _PROFILE_FLOAT_FEATURES or feature in _PROFILE_INTEGER_FEATURES:
        value = _finite_number(getattr(profile, feature))
        return (None, value) if value is not None else ("invalid_profile", None)
    if feature in _DIAGNOSTIC_FLOAT_FEATURES:
        diagnostic = diagnostics[assumption.candidate_name]
        value = _finite_number(getattr(diagnostic, feature))
        return (None, value) if value is not None else ("invalid_diagnostics", None)
    return "invalid_profile", None


def _assumptions_satisfied(
    policy: FittedChampionPolicy,
    profile: TaskProfile,
    diagnostics: Mapping[str, CandidateDiagnostics],
) -> tuple[str | None, tuple[str, ...]]:
    activated: list[str] = []
    for assumption, (_, threshold) in zip(
        policy.recipe.assumptions, policy.thresholds, strict=True
    ):
        diagnostic = diagnostics[assumption.candidate_name]
        if not _safe_diagnostic(diagnostic):
            return "assumption_not_satisfied", ()
        reason, value = _measurement(assumption, profile, diagnostics)
        if reason is not None or value is None:
            return reason or "invalid_profile", ()
        matches = value >= threshold if assumption.direction == "above" else value < threshold
        if not matches:
            return "assumption_not_satisfied", ()
        activated.append(assumption.assumption_id)
    return None, tuple(activated)


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle - 1] / 2.0 + ordered[middle] / 2.0


def robust_history_scale(history: object) -> float:
    """Return a deterministic overflow-safe median absolute history level."""
    values = _finite_values(history)
    if values is None or not values:
        raise ValueError("history must be a non-empty finite list or tuple")
    return max(_median(tuple(abs(value) for value in values)), 1e-12)


def _finite_result(values: list[float]) -> tuple[float, ...] | None:
    if not all(math.isfinite(value) for value in values):
        return None
    return tuple(values)


def execute_champion(
    policy: FittedChampionPolicy,
    forecasts: Mapping[str, object],
    diagnostics: Mapping[str, CandidateDiagnostics],
    profile: TaskProfile,
    history: object,
    horizon: int,
) -> ChampionExecution:
    """Execute a frozen Champion from history-only, already materialized inputs."""
    fallback_name: str | None = None
    if type(policy) is FittedChampionPolicy and type(policy.recipe) is ChampionRecipe:
        if type(policy.recipe.fallback_parent) is str:
            fallback_name = policy.recipe.fallback_parent
    found_fallback, fallback_forecast = (
        _mapping_value(forecasts, fallback_name)
        if fallback_name is not None
        else (False, ())
    )
    if not found_fallback:
        fallback_forecast = ()

    if not _valid_policy(policy):
        return _fallback(fallback_forecast, fallback_name, "invalid_policy")
    if not found_fallback:
        return _fallback(fallback_forecast, fallback_name, "missing_parent")
    if type(horizon) is not int or not 1 <= horizon <= 1_000_000:
        return _fallback(fallback_forecast, fallback_name, "invalid_horizon")

    history_values = _finite_values(history)
    if history_values is None or not history_values:
        return _fallback(fallback_forecast, fallback_name, "invalid_history")
    if not _valid_profile(profile, history_length=len(history_values), horizon=horizon):
        return _fallback(fallback_forecast, fallback_name, "invalid_profile")

    parent_forecasts: dict[str, tuple[float, ...]] = {}
    original_forecasts: dict[str, object] = {}
    parent_diagnostics: dict[str, CandidateDiagnostics] = {}
    for name in policy.recipe.parents:
        found, raw_forecast = (
            (True, fallback_forecast)
            if name == fallback_name
            else _mapping_value(forecasts, name)
        )
        if not found:
            return _fallback(fallback_forecast, fallback_name, "missing_parent")
        original_forecasts[name] = raw_forecast
        values = _finite_values(raw_forecast, length=horizon)
        if type(raw_forecast) is not tuple or values is None:
            return _fallback(fallback_forecast, fallback_name, "invalid_parent_forecast")
        parent_forecasts[name] = values

        found, raw_diagnostic = _mapping_value(diagnostics, name)
        if not found or not _valid_diagnostic(raw_diagnostic, name):
            return _fallback(fallback_forecast, fallback_name, "invalid_diagnostics")
        parent_diagnostics[name] = raw_diagnostic

    reason, activated = _assumptions_satisfied(policy, profile, parent_diagnostics)
    if reason is not None:
        return _fallback(fallback_forecast, fallback_name, reason)
    if any(not _safe_diagnostic(item) for item in parent_diagnostics.values()):
        return _fallback(fallback_forecast, fallback_name, "invalid_diagnostics")

    parents = policy.recipe.parents
    kind = policy.recipe.kind
    try:
        if kind == "select":
            name = parents[0]
            return ChampionExecution(original_forecasts[name], (name,), activated, None)

        if kind == "route":
            routed_name = policy.recipe.assumptions[0].candidate_name
            if any(item.candidate_name != routed_name for item in policy.recipe.assumptions):
                return _fallback(fallback_forecast, fallback_name, "invalid_policy")
            return ChampionExecution(
                original_forecasts[routed_name], (routed_name,), activated, None
            )

        left_name, right_name = parents
        left = parent_forecasts[left_name]
        right = parent_forecasts[right_name]

        if kind == "horizon_route":
            split = int(horizon * policy.horizon_split)
            result = left[:split] + right[split:]
        elif kind == "weighted":
            result = tuple(
                left[index] * policy.weights[0] + right[index] * policy.weights[1]
                for index in range(horizon)
            )
        elif kind == "median":
            result = tuple(
                left[index] / 2.0 + right[index] / 2.0 for index in range(horizon)
            )
        elif kind == "bounded_overlay":
            specialist_name = right_name if left_name == fallback_name else left_name
            specialist = parent_forecasts[specialist_name]
            fallback_values = parent_forecasts[fallback_name]
            limit = policy.correction_cap * robust_history_scale(history_values)
            if not math.isfinite(limit):
                return _fallback(fallback_forecast, fallback_name, "invalid_arithmetic")
            overlaid: list[float] = []
            for baseline, candidate in zip(fallback_values, specialist, strict=True):
                delta = candidate - baseline
                correction = max(-limit, min(limit, policy.overlay_alpha * delta))
                value = baseline + correction
                if not math.isfinite(value) or abs(value - baseline) > limit:
                    return _fallback(fallback_forecast, fallback_name, "invalid_arithmetic")
                overlaid.append(value)
            result = tuple(overlaid)
        else:
            return _fallback(fallback_forecast, fallback_name, "invalid_policy")
    except (ArithmeticError, TypeError, ValueError):
        return _fallback(fallback_forecast, fallback_name, "invalid_arithmetic")

    finite = _finite_result(list(result))
    if finite is None:
        return _fallback(fallback_forecast, fallback_name, "invalid_arithmetic")
    return ChampionExecution(finite, parents, activated, None)
