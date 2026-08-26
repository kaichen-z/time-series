"""Forecast metrics shared by trusted evaluation hosts."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def smape(y_true: list[float], y_pred: list[float]) -> float:
    """Symmetric MAPE in [0, 200]; a true/pred pair that's both zero contributes 0, not NaN."""
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    total = 0.0
    for t, p in zip(y_true, y_pred):
        denom = abs(t) + abs(p)
        total += 0.0 if denom == 0 else 200.0 * abs(t - p) / denom
    return total / len(y_true)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean absolute error."""
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root mean squared error."""
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    return math.hypot(*(t - p for t, p in zip(y_true, y_pred))) / math.sqrt(
        len(y_true)
    )


def drcik_point_metrics(
    y_true: Sequence[float],
    forecast: Sequence[float] | Sequence[Sequence[float]],
    *,
    cap: float = 5.0,
) -> dict[str, float | bool]:
    """Dr-CiK-aligned point metrics for one task.

    Dr-CiK scales both MAE and RMSE by the mean absolute value of the true
    forecast horizon, then independently caps each per-task scaled metric at
    ``cap`` before aggregating across tasks. ``forecast`` accepts either one
    deterministic trajectory or multiple trajectories; multiple trajectories
    are reduced to their step-wise mean before scoring.

    These values reproduce the public metric definition, but they are not a
    verified hidden-test leaderboard score because the official scorer and
    hidden labels remain private.
    """
    truth = [float(value) for value in y_true]
    if not truth:
        raise ValueError("Dr-CiK point metrics require a non-empty horizon")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap must be a positive finite value")
    raw = tuple(forecast)
    if not raw:
        raise ValueError("Dr-CiK point metrics require at least one forecast")
    first = raw[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        trajectories = tuple(
            tuple(float(value) for value in sample)  # type: ignore[arg-type]
            for sample in raw
        )
        if any(len(sample) != len(truth) for sample in trajectories):
            raise ValueError("every forecast trajectory must match the future horizon")
        prediction = [
            statistics.fmean(sample[step] for sample in trajectories)
            for step in range(len(truth))
        ]
    else:
        prediction = [float(value) for value in raw]  # type: ignore[arg-type]
        _check_same_length(truth, prediction)
    if not all(math.isfinite(value) for value in (*truth, *prediction)):
        raise ValueError("Dr-CiK inputs must contain only finite values")

    scale = statistics.fmean(abs(value) for value in truth)
    absolute_error = mae(truth, prediction)
    squared_error = rmse(truth, prediction)
    if scale > 0.0:
        smae_raw = absolute_error / scale
        srmse_raw = squared_error / scale
    else:
        smae_raw = 0.0 if absolute_error == 0.0 else math.inf
        srmse_raw = 0.0 if squared_error == 0.0 else math.inf
    return {
        "scale": scale,
        "mae": absolute_error,
        "rmse": squared_error,
        "smae_raw": smae_raw,
        "srmse_raw": srmse_raw,
        "smae": min(cap, smae_raw),
        "srmse": min(cap, srmse_raw),
        "smae_clipped": smae_raw > cap,
        "srmse_clipped": srmse_raw > cap,
    }


def aggregate_drcik_point_metrics(
    rows: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Average already capped Dr-CiK task metrics across tasks."""
    if not rows:
        raise ValueError("at least one task metric row is required")
    return {
        "smae": statistics.fmean(float(row["smae"]) for row in rows),
        "srmse": statistics.fmean(float(row["srmse"]) for row in rows),
    }


def standard_error(values: list[float]) -> float:
    """Standard error of a task-level sample; zero for fewer than two tasks."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def linear_quantile(values: list[float], probability: float) -> float:
    """Linearly interpolated quantile with endpoints included."""
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mase(
    y_true: list[float], y_pred: list[float], history: list[float], seasonal_period: int = 1
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006): MAE scaled by the in-sample naive
    error, so it stays finite and comparable across series instead of blowing up."""
    _check_same_length(y_true, y_pred)
    if not history:
        raise ValueError("history must not be empty for MASE scaling")
    period = seasonal_period if 0 < seasonal_period < len(history) else 1
    diffs = [abs(history[i] - history[i - period]) for i in range(period, len(history))]
    scale = sum(diffs) / len(diffs) if diffs else 0.0
    if scale <= 1e-8:
        # A flat in-sample history would otherwise divide by zero; fall back to the series'
        # own spread so a genuinely flat series still yields a finite, comparable score.
        spread = max(history) - min(history)
        scale = spread if spread > 1e-8 else 1.0
    return mae(y_true, y_pred) / scale


def score_forecast(y_true: list[float], y_pred: list[float]) -> dict:
    """sMAPE + MAE together, with sMAPE as the single scalar used for ranking/comparison."""
    return {"smape": smape(y_true, y_pred), "mae": mae(y_true, y_pred), "primary": smape(y_true, y_pred)}


def spearman_rank_correlation(left: list[float], right: list[float]) -> float:
    """Correlation of average ranks; return 0 when either ordering is constant."""
    _check_same_length(left, right)
    if len(left) < 2:
        return 0.0

    def average_ranks(values: list[float]) -> list[float]:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
        ranks = [0.0] * len(values)
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
                end += 1
            average = ((start + 1) + end) / 2.0
            for position in range(start, end):
                ranks[ordered[position]] = average
            start = end
        return ranks

    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_scale = sum((rank - left_mean) ** 2 for rank in left_ranks) ** 0.5
    right_scale = sum((rank - right_mean) ** 2 for rank in right_ranks) ** 0.5
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _check_same_length(y_true: list[float], y_pred: list[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true values vs {len(y_pred)} predicted")
