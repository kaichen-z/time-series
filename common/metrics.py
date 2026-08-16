"""sMAPE and MAE metrics."""
from __future__ import annotations


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
