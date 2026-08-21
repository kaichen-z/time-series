"""sMAPE, MAE, RMSE and forecast-shape metrics."""
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


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root mean squared error; penalizes large deviations more than MAE, in the same units."""
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    return (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def variance_ratio(y_true: list[float], y_pred: list[float]) -> float:
    """Spread of the forecast over spread of the truth; 0.0 means a flat line."""
    _check_same_length(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    truth_spread = _population_std(y_true)
    if truth_spread <= 1e-12:
        return 1.0 if _population_std(y_pred) <= 1e-12 else 0.0
    return _population_std(y_pred) / truth_spread


def shape_correlation(y_true: list[float], y_pred: list[float]) -> float:
    """Pearson correlation between forecast and truth; 0.0 when either is constant.

    A forecast that sits at the right level but never moves scores 0.0 here however good its
    MAE, which is what separates tracking the series from parking near its mean.
    """
    _check_same_length(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    true_mean = sum(y_true) / len(y_true)
    pred_mean = sum(y_pred) / len(y_pred)
    covariance = sum((t - true_mean) * (p - pred_mean) for t, p in zip(y_true, y_pred))
    true_scale = sum((t - true_mean) ** 2 for t in y_true) ** 0.5
    pred_scale = sum((p - pred_mean) ** 2 for p in y_pred) ** 0.5
    if true_scale <= 1e-12 or pred_scale <= 1e-12:
        return 0.0
    return covariance / (true_scale * pred_scale)


def change_mae(y_true: list[float], y_pred: list[float], last_observed: float) -> float:
    """MAE on first differences, counting the step from the last observation into the horizon.

    A flat forecast predicts zero change everywhere, so this equals the truth's own volatility
    for it; anything that tracks the dynamics beats that.
    """
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    true_steps = [y_true[0] - last_observed] + [
        y_true[i] - y_true[i - 1] for i in range(1, len(y_true))
    ]
    pred_steps = [y_pred[0] - last_observed] + [
        y_pred[i] - y_pred[i - 1] for i in range(1, len(y_pred))
    ]
    return mae(true_steps, pred_steps)


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


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
