"""MAE, RMSE, the Dr-CiK scaled metrics (sMAE, sRMSE, scaled change-MAE), and forecast-shape metrics.

Every metric returned here is rounded to 3 decimal places, since more precision than that is
noise for reporting and never worth carrying through a downstream mean.
"""
from __future__ import annotations


ROUND_DIGITS = 3 # I added this to round the results, much better than just letting it go.


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean absolute error, in the series' own units.

    A building block for scaled_mae and change_smae rather than a metric to rank methods by:
    a mean of raw MAE across tasks is decided by the largest-magnitude series.
    """
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    return round(sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true), ROUND_DIGITS)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root mean squared error, in the series' own units.

    A building block for scaled_rmse; not comparable across tasks of differing magnitude.
    """
    _check_same_length(y_true, y_pred)
    if not y_true:
        return 0.0
    value = (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5
    return round(value, ROUND_DIGITS)


def horizon_scale(y_true: list[float]) -> float:
    """Mean absolute value of the truth over the horizon, the denominator of sMAE and sRMSE.

    The Dr-CiK scale factor a = (1/T sum |y_t|)^-1, taken over the forecast horizon itself.
    Falls back to 1.0 for an all-zero horizon, which leaves the error in its own units rather
    than dividing by ~zero.
    """
    if not y_true:
        return 1.0
    scale = sum(abs(value) for value in y_true) / len(y_true)
    return scale if scale > 1e-8 else 1.0


def scaled_mae(y_true: list[float], y_pred: list[float]) -> float:
    """Dr-CiK sMAE: MAE divided by the mean absolute truth over the horizon.

    Read as a fraction of the series' own typical magnitude: 0.1 means the average error is a
    tenth of that, so a slow expensive series and a fast cheap one weigh the same.
    """
    return round(mae(y_true, y_pred) / horizon_scale(y_true), ROUND_DIGITS)


def scaled_rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Dr-CiK sRMSE: RMSE over the same scale, so large errors still weigh more."""
    return round(rmse(y_true, y_pred) / horizon_scale(y_true), ROUND_DIGITS)


def variance_ratio(y_true: list[float], y_pred: list[float]) -> float:
    """Spread of the forecast over spread of the truth; 0.0 means a flat line."""
    _check_same_length(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    truth_spread = _population_std(y_true)
    if truth_spread <= 1e-12:
        return 1.0 if _population_std(y_pred) <= 1e-12 else 0.0
    return round(_population_std(y_pred) / truth_spread, ROUND_DIGITS)


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
    return round(covariance / (true_scale * pred_scale), ROUND_DIGITS)


def change_mae(y_true: list[float], y_pred: list[float], last_observed: float) -> float:
    """MAE on first differences, counting the step from the last observation into the horizon.

    A flat forecast predicts zero change everywhere, so this equals the truth's own volatility
    for it; anything that tracks the dynamics beats that. A building block for change_smae
    rather than a metric to rank methods by, for the same reason as mae and rmse.
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


def change_smae(y_true: list[float], y_pred: list[float], last_observed: float) -> float:
    """change_mae divided by the same horizon scale as scaled_mae and scaled_rmse.

    Puts whether a forecast tracks the series' dynamics on the same footing across a slow
    expensive series and a fast cheap one, the way scaled_mae does for level error.
    """
    value = change_mae(y_true, y_pred, last_observed) / horizon_scale(y_true)
    return round(value, ROUND_DIGITS)


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


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
    return round(numerator / (left_scale * right_scale), ROUND_DIGITS)


def _check_same_length(y_true: list[float], y_pred: list[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true values vs {len(y_pred)} predicted")
