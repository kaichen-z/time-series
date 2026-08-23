"""Deterministic, history-only analysis skills available to forecasting methods."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


SKILL_API_VERSION = 1
ANALYSIS_SKILL_NAMES = (
    "detect_periodicity",
    "detect_outliers",
    "detect_trend",
    "detect_change_points",
    "detect_intermittency",
    "estimate_noise_scale",
    "assess_stationarity",
    "detect_recent_regime",
    "analyze_series",
)


def _values(history: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in history)
    if not values:
        raise ValueError("history must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("history must contain only finite values")
    return values


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2.0
    mean = statistics.fmean(values)
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return sum(
        (index - center) * (value - mean) for index, value in enumerate(values)
    ) / denominator


def detect_periodicity(history, frequency):
    """Return autocorrelation-supported historical periods without forecasting the future."""
    del frequency
    values = _values(history)
    if len(values) < 8 or statistics.pstdev(values) <= 1e-12:
        return {"candidate_periods": [], "strength": 0.0, "confidence": 0.0}
    slope = _linear_slope(values)
    centered = tuple(
        value - (statistics.fmean(values) + slope * (index - (len(values) - 1) / 2.0))
        for index, value in enumerate(values)
    )
    maximum = min(48, len(values) // 2)
    scores = [(lag, _correlation(centered[:-lag], centered[lag:])) for lag in range(2, maximum + 1)]
    peaks = []
    for index, (lag, score) in enumerate(scores):
        previous = scores[index - 1][1] if index else -1.0
        following = scores[index + 1][1] if index + 1 < len(scores) else -1.0
        if score >= 0.2 and score >= previous and score >= following:
            peaks.append((lag, score))
    peaks.sort(key=lambda item: (-round(item[1], 10), item[0]))
    if not peaks:
        return {"candidate_periods": [], "strength": 0.0, "confidence": 0.0}
    best = peaks[0][1]
    candidates = [lag for lag, score in peaks if score >= max(0.2, best - 0.1)][:3]
    cycles = len(values) / candidates[0]
    confidence = _clamp(best * min(1.0, cycles / 3.0))
    return {
        "candidate_periods": candidates,
        "strength": round(_clamp(best), 8),
        "confidence": round(confidence, 8),
    }


def detect_outliers(history):
    """Report robust marginal outlier candidates while leaving the history unchanged."""
    values = _values(history)
    median = statistics.median(values)
    deviations = tuple(abs(value - median) for value in values)
    mad = statistics.median(deviations)
    if mad <= 1e-12:
        scores = tuple(0.0 if deviation <= 1e-12 else 1e12 for deviation in deviations)
    else:
        scores = tuple(0.67448975 * deviation / mad for deviation in deviations)
    indices = [index for index, score in enumerate(scores) if score > 3.5]
    return {
        "indices": indices,
        "scores": [round(scores[index], 8) for index in indices],
        "median": float(median),
        "mad": float(mad),
        "confidence": round(_clamp(max((scores[index] for index in indices), default=0.0) / 7.0), 8),
    }


def detect_trend(history):
    """Measure the direction, slope, and linear strength of the historical level."""
    values = _values(history)
    slope = _linear_slope(values)
    fitted = tuple(slope * index for index in range(len(values)))
    strength = abs(_correlation(values, fitted)) if len(values) >= 3 and abs(slope) > 1e-12 else 0.0
    scale = statistics.pstdev(values) if len(values) > 1 else 0.0
    materiality = abs(slope) * max(1, len(values) - 1) / max(scale, 1e-12)
    if materiality < 0.25:
        direction = "flat"
    else:
        direction = "up" if slope > 0.0 else "down"
    return {
        "direction": direction,
        "slope_per_step": round(float(slope), 8),
        "strength": round(_clamp(strength), 8),
        "confidence": round(_clamp(strength * min(1.0, len(values) / 20.0)), 8),
    }


def detect_change_points(history):
    """Report mean-shift candidates supported by both pre- and post-change history."""
    values = _values(history)
    minimum = max(4, min(12, len(values) // 5))
    if len(values) < 2 * minimum:
        return {"indices": [], "scores": [], "confidence": 0.0}
    global_scale = statistics.pstdev(values) or 1.0
    candidates = []
    for split in range(minimum, len(values) - minimum + 1):
        left = values[max(0, split - minimum) : split]
        right = values[split : split + minimum]
        contrast = abs(statistics.fmean(right) - statistics.fmean(left)) / global_scale
        balance = math.sqrt(len(left) * len(right)) / minimum
        candidates.append((split, contrast * balance))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = []
    for split, score in candidates:
        if score < 0.75:
            break
        if all(abs(split - existing[0]) >= minimum for existing in selected):
            selected.append((split, score))
        if len(selected) == 3:
            break
    selected.sort()
    best = max((score for _, score in selected), default=0.0)
    return {
        "indices": [split for split, _ in selected],
        "scores": [round(score, 8) for _, score in selected],
        "confidence": round(_clamp(best / 2.0), 8),
    }


def detect_intermittency(history):
    """Measure zero prevalence and nonzero-arrival spacing in historical observations."""
    values = _values(history)
    nonzero = [index for index, value in enumerate(values) if abs(value) > 1e-12]
    zero_fraction = 1.0 - len(nonzero) / len(values)
    gaps = [right - left for left, right in zip(nonzero, nonzero[1:])]
    average_gap = statistics.fmean(gaps) if gaps else float(len(values))
    nonzero_values = [values[index] for index in nonzero]
    mean_nonzero = statistics.fmean(nonzero_values) if nonzero_values else 0.0
    cv2 = (
        (statistics.pstdev(nonzero_values) / mean_nonzero) ** 2
        if len(nonzero_values) > 1 and abs(mean_nonzero) > 1e-12
        else 0.0
    )
    return {
        "is_intermittent": bool(zero_fraction > 0.3 or average_gap > 1.32),
        "zero_fraction": round(zero_fraction, 8),
        "average_nonzero_gap": round(float(average_gap), 8),
        "nonzero_cv2": round(float(cv2), 8),
        "confidence": round(_clamp(max(zero_fraction, (average_gap - 1.0) / 4.0)), 8),
    }


def estimate_noise_scale(history):
    """Estimate historical innovation scale from robust first differences."""
    values = _values(history)
    if len(values) < 2:
        return {"robust_scale": 0.0, "relative_scale": 0.0, "confidence": 0.0}
    differences = tuple(values[index] - values[index - 1] for index in range(1, len(values)))
    median = statistics.median(differences)
    mad = statistics.median(abs(value - median) for value in differences)
    robust_scale = mad / 0.67448975 if mad > 1e-12 else statistics.pstdev(differences)
    level_scale = statistics.pstdev(values)
    relative = robust_scale / max(level_scale, 1e-12)
    return {
        "robust_scale": round(float(robust_scale), 8),
        "relative_scale": round(float(relative), 8),
        "confidence": round(_clamp(len(differences) / 20.0), 8),
    }


def assess_stationarity(history):
    """Heuristically compare historical halves for stable mean, variance, and trend."""
    values = _values(history)
    if len(values) < 12:
        return {"likely_stationary": False, "score": 0.0, "confidence": 0.0}
    middle = len(values) // 2
    left, right = values[:middle], values[middle:]
    scale = statistics.pstdev(values) or 1.0
    mean_shift = abs(statistics.fmean(right) - statistics.fmean(left)) / scale
    left_var = statistics.pvariance(left)
    right_var = statistics.pvariance(right)
    variance_ratio = min(left_var, right_var) / max(left_var, right_var, 1e-12)
    trend = detect_trend(values)
    score = _clamp(1.0 - mean_shift / 2.0) * variance_ratio * (1.0 - 0.5 * float(trend["strength"]))
    return {
        "likely_stationary": bool(score >= 0.5),
        "score": round(score, 8),
        "confidence": round(_clamp(len(values) / 40.0), 8),
    }


def detect_recent_regime(history):
    """Identify the strongest supported change point in the recent half of history."""
    values = _values(history)
    changes = detect_change_points(values)
    recent = [
        (index, score)
        for index, score in zip(changes["indices"], changes["scores"], strict=True)
        if index >= len(values) // 2
    ]
    if not recent:
        return {"regime_start": None, "confidence": 0.0, "level_shift": 0.0}
    index, score = max(recent, key=lambda item: item[1])
    width = max(2, min(12, index, len(values) - index))
    shift = statistics.fmean(values[index : index + width]) - statistics.fmean(values[index - width : index])
    return {
        "regime_start": int(index),
        "confidence": round(_clamp(float(score) / 2.0), 8),
        "level_shift": round(float(shift), 8),
    }


def analyze_series(history, frequency):
    """Build one reusable history-only profile for forecasting-method selection."""
    values = _values(history)
    return {
        "periodicity": detect_periodicity(values, frequency),
        "outliers": detect_outliers(values),
        "trend": detect_trend(values),
        "change_points": detect_change_points(values),
        "intermittency": detect_intermittency(values),
        "noise": estimate_noise_scale(values),
        "stationarity": assess_stationarity(values),
        "recent_regime": detect_recent_regime(values),
    }
