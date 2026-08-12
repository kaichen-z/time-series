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


def _check_same_length(y_true: list[float], y_pred: list[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true values vs {len(y_pred)} predicted")
