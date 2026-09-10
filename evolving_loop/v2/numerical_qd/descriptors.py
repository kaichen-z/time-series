"""Deterministic history-only morphology descriptors for Numerical QD."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from types import MappingProxyType

from ..contracts import (
    _require_exact_schema,
    _strict_json_value,
    canonical_v2_bytes,
    fingerprint_payload,
)
from .contracts import MEMBER_FAMILIES, MorphologyCellV2


_POLICY_FIELDS = (
    "schema_version",
    "trend_low_threshold",
    "trend_high_threshold",
    "intermittency_high_threshold",
    "regime_shift_threshold",
    "horizon_short_threshold",
    "horizon_medium_threshold",
    "seasonality_threshold",
    "short_seasonal_lag_max",
    "variance_floor",
    "seasonal_lags",
)


def _finite_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    return value


@dataclass(frozen=True, slots=True)
class DescriptorPolicyV2:
    """Versioned thresholds and lag candidates for morphology classification."""

    schema_version: int
    trend_low_threshold: float
    trend_high_threshold: float
    intermittency_high_threshold: float
    regime_shift_threshold: float
    horizon_short_threshold: float
    horizon_medium_threshold: float
    seasonality_threshold: float
    short_seasonal_lag_max: int
    variance_floor: float
    seasonal_lags: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")

        trend_low = _finite_float(self.trend_low_threshold, "trend_low_threshold")
        trend_high = _finite_float(self.trend_high_threshold, "trend_high_threshold")
        if trend_low < 0.0 or trend_high <= trend_low:
            raise ValueError(
                "trend_low_threshold and trend_high_threshold must be ordered and non-negative"
            )

        intermittency = _finite_float(
            self.intermittency_high_threshold, "intermittency_high_threshold"
        )
        if not 0.0 < intermittency <= 1.0:
            raise ValueError("intermittency_high_threshold must be in (0, 1]")

        regime = _finite_float(self.regime_shift_threshold, "regime_shift_threshold")
        if regime <= 0.0:
            raise ValueError("regime_shift_threshold must be positive")

        horizon_short = _finite_float(
            self.horizon_short_threshold, "horizon_short_threshold"
        )
        horizon_medium = _finite_float(
            self.horizon_medium_threshold, "horizon_medium_threshold"
        )
        if not 0.0 < horizon_short < horizon_medium:
            raise ValueError(
                "horizon_short_threshold and horizon_medium_threshold must be ordered and positive"
            )

        seasonality = _finite_float(self.seasonality_threshold, "seasonality_threshold")
        if not 0.0 <= seasonality <= 1.0:
            raise ValueError("seasonality_threshold must be in [0, 1]")
        if type(self.short_seasonal_lag_max) is not int or self.short_seasonal_lag_max <= 0:
            raise ValueError("short_seasonal_lag_max must be a positive integer")
        variance_floor = _finite_float(self.variance_floor, "variance_floor")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")

        if not isinstance(self.seasonal_lags, Mapping):
            raise ValueError("seasonal_lags must be an object")
        lags: dict[str, tuple[int, ...]] = {}
        for frequency, raw_lags in self.seasonal_lags.items():
            if type(frequency) is not str or not frequency.strip():
                raise ValueError("seasonal_lags keys must be non-empty strings")
            if not isinstance(raw_lags, (list, tuple)):
                raise ValueError(f"seasonal_lags.{frequency} must be a list or tuple")
            normalized = tuple(raw_lags)
            if not normalized:
                raise ValueError(f"seasonal_lags.{frequency} must not be empty")
            if any(type(lag) is not int or lag <= 0 for lag in normalized):
                raise ValueError(f"seasonal_lags.{frequency} must contain positive integers")
            if normalized != tuple(sorted(set(normalized))):
                raise ValueError(f"seasonal_lags.{frequency} must be sorted and unique")
            lags[frequency] = normalized
        object.__setattr__(self, "seasonal_lags", MappingProxyType(dict(sorted(lags.items()))))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DescriptorPolicyV2":
        values = _require_exact_schema(payload, _POLICY_FIELDS, field="descriptor_policy")
        strict = _strict_json_value(values, field="descriptor_policy")
        assert isinstance(strict, dict)
        return cls(**strict)  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "trend_low_threshold": self.trend_low_threshold,
            "trend_high_threshold": self.trend_high_threshold,
            "intermittency_high_threshold": self.intermittency_high_threshold,
            "regime_shift_threshold": self.regime_shift_threshold,
            "horizon_short_threshold": self.horizon_short_threshold,
            "horizon_medium_threshold": self.horizon_medium_threshold,
            "seasonality_threshold": self.seasonality_threshold,
            "short_seasonal_lag_max": self.short_seasonal_lag_max,
            "variance_floor": self.variance_floor,
            "seasonal_lags": self.seasonal_lags,
        }
        strict = _strict_json_value(payload)
        assert isinstance(strict, dict)
        return strict

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


def _history_values(history: object) -> tuple[float, ...]:
    if not isinstance(history, (list, tuple)):
        raise ValueError("history must be a list or tuple")
    if len(history) < 4:
        raise ValueError("history must contain at least four observations")
    values = []
    for value in history:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("history must contain only finite numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("history must contain only finite numbers")
        values.append(numeric)
    return tuple(values)


def _autocorrelation(values: Sequence[float], lag: int) -> float | None:
    left_raw = values[:-lag]
    right_raw = values[lag:]
    value_scale = max(abs(value) for value in values)
    if value_scale == 0.0:
        return None
    left = tuple(value / value_scale for value in left_raw)
    right = tuple(value / value_scale for value in right_raw)
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    left_scale = max(abs(value) for value in left_centered)
    right_scale = max(abs(value) for value in right_centered)
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    left_centered = tuple(value / left_scale for value in left_centered)
    right_centered = tuple(value / right_scale for value in right_centered)
    left_energy = sum(value * value for value in left_centered)
    right_energy = sum(value * value for value in right_centered)
    denominator = math.sqrt(left_energy * right_energy)
    result = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator
    return result if math.isfinite(result) else None


def describe_history(
    history: Sequence[float],
    horizon: int,
    frequency: str,
    family: str,
    policy: DescriptorPolicyV2,
) -> MorphologyCellV2:
    """Classify one observed history without accepting future or label inputs."""
    values = _history_values(history)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if type(frequency) is not str or not frequency.strip():
        raise ValueError("frequency must be a non-empty string")
    if type(family) is not str or family not in MEMBER_FAMILIES:
        raise ValueError("family must be a registered Numerical member family")
    if type(policy) is not DescriptorPolicyV2:
        raise ValueError("policy must be a DescriptorPolicyV2")

    scale = max(pstdev(values), policy.variance_floor)
    quarter_size = len(values) // 4
    trend_score = abs(
        mean(values[-quarter_size:]) - mean(values[:quarter_size])
    ) / scale
    if trend_score < policy.trend_low_threshold:
        trend = "low"
    elif trend_score < policy.trend_high_threshold:
        trend = "medium"
    else:
        trend = "high"

    zero_fraction = sum(value == 0.0 for value in values) / len(values)
    intermittency = (
        "high" if zero_fraction >= policy.intermittency_high_threshold else "low"
    )

    midpoint = len(values) // 2
    regime_score = abs(
        mean(values[midpoint:]) - mean(values[:midpoint])
    ) / scale
    regime = "shift" if regime_score >= policy.regime_shift_threshold else "stable"

    horizon_ratio = horizon / len(values)
    if horizon_ratio <= policy.horizon_short_threshold:
        horizon_bin = "short"
    elif horizon_ratio <= policy.horizon_medium_threshold:
        horizon_bin = "medium"
    else:
        horizon_bin = "long"

    correlations = []
    for lag in policy.seasonal_lags.get(frequency, ()):
        if len(values) < 2 * lag:
            continue
        correlation = _autocorrelation(values, lag)
        if correlation is not None:
            correlations.append((correlation, lag))
    if not correlations:
        seasonality = "none"
    else:
        best_score, best_lag = max(correlations, key=lambda item: (item[0], -item[1]))
        if best_score < policy.seasonality_threshold:
            seasonality = "none"
        elif best_lag <= policy.short_seasonal_lag_max:
            seasonality = "short"
        else:
            seasonality = "long"

    return MorphologyCellV2(
        trend=trend,
        seasonality=seasonality,
        intermittency=intermittency,
        regime=regime,
        horizon=horizon_bin,
        family=family,
    )
