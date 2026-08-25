"""Forecast metrics, including the public Dr-CiK definitions."""
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


def mase(
    y_true: list[float],
    y_pred: list[float],
    history: list[float],
    seasonal_period: int = 1,
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
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "primary": smape(y_true, y_pred),
    }


def drcik_task_metrics(
    y_true: Sequence[float],
    samples: Sequence[Sequence[float]],
    *,
    cap: float = 5.0,
) -> dict[str, float]:
    """Compute the forecasting metrics defined in Dr-CiK Appendix H.2.

    Dr-CiK scales each task by the mean absolute value of that task's *future
    target*, not by a history-based MASE denominator.  The three scaled metrics
    are independently capped before cross-task aggregation.
    """
    truth = tuple(float(value) for value in y_true)
    trajectories = tuple(tuple(float(value) for value in sample) for sample in samples)
    if not truth:
        raise ValueError("Dr-CiK metrics require a non-empty future target")
    if not trajectories:
        raise ValueError("Dr-CiK metrics require at least one forecast trajectory")
    if cap <= 0:
        raise ValueError("Dr-CiK metric cap must be positive")
    if any(len(sample) != len(truth) for sample in trajectories):
        raise ValueError("every forecast trajectory must match the future horizon")
    if not all(math.isfinite(value) for value in truth):
        raise ValueError("future target contains a non-finite value")
    if not all(math.isfinite(value) for sample in trajectories for value in sample):
        raise ValueError("forecast trajectory contains a non-finite value")

    scale_denominator = statistics.fmean(abs(value) for value in truth)
    if scale_denominator == 0.0:
        # The paper does not specify a fallback for an all-zero future target.
        raise ValueError(
            "official Dr-CiK scaling is undefined for an all-zero future target"
        )

    point = tuple(
        statistics.fmean(sample[step] for sample in trajectories)
        for step in range(len(truth))
    )
    point_mae = statistics.fmean(
        abs(actual - predicted) for actual, predicted in zip(truth, point)
    )
    point_rmse = math.sqrt(
        statistics.fmean(
            (actual - predicted) ** 2 for actual, predicted in zip(truth, point)
        )
    )
    crps = statistics.fmean(
        _empirical_crps(actual, [sample[step] for sample in trajectories])
        for step, actual in enumerate(truth)
    )
    raw = {
        "smae_uncapped": point_mae / scale_denominator,
        "srmse_uncapped": point_rmse / scale_denominator,
        "scrps_uncapped": crps / scale_denominator,
    }
    return {
        "official_scale_denominator": scale_denominator,
        "official_point_mae": point_mae,
        "official_point_rmse": point_rmse,
        "official_crps": crps,
        **raw,
        "smae": min(cap, raw["smae_uncapped"]),
        "srmse": min(cap, raw["srmse_uncapped"]),
        "scrps": min(cap, raw["scrps_uncapped"]),
    }


def aggregate_drcik_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float | int]:
    """Aggregate already capped task metrics as mean and standard error."""
    if not rows:
        raise ValueError("at least one task metric row is required")
    summary: dict[str, float | int] = {"num_tasks": len(rows)}
    for name in ("smae", "srmse", "scrps"):
        values = [float(row[name]) for row in rows]
        summary[name] = statistics.fmean(values)
        summary[f"{name}_se"] = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
    return summary


def _empirical_crps(truth: float, samples: Sequence[float]) -> float:
    """Empirical CRPS in O(S log S), equivalent to the paper's pairwise formula."""
    ordered = sorted(samples)
    count = len(ordered)
    accuracy = statistics.fmean(abs(value - truth) for value in ordered)
    dispersion = sum(
        (2 * index - count + 1) * value for index, value in enumerate(ordered)
    ) / (count * count)
    return accuracy - dispersion


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
        raise ValueError(
            f"length mismatch: {len(y_true)} true values vs {len(y_pred)} predicted"
        )
