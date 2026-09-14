"""Seed program for whole-forecaster evolution."""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence


ForecastSkill = Callable[[Sequence[float], int, str], list[float]]


class NotApplicable(Exception):
    """Signal that a forecasting skill does not fit the supplied series."""


def _history(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("history must contain finite values")
    return result


def _validate(values: Sequence[float], horizon: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != horizon or not all(math.isfinite(value) for value in result):
        raise ValueError("forecast must contain exactly horizon finite values")
    return result


def _period(frequency: str) -> int | None:
    text = str(frequency).strip().lower().replace("_", " ")
    exact = {
        "h": 24,
        "hour": 24,
        "hourly": 24,
        "1 hour": 24,
        "d": 7,
        "day": 7,
        "daily": 7,
        "1 day": 7,
        "w": 52,
        "week": 52,
        "weekly": 52,
        "m": 12,
        "month": 12,
        "monthly": 12,
        "q": 4,
        "quarter": 4,
        "quarterly": 4,
        "5 minutes": 288,
        "1 minute": 1440,
        "1 second": 60,
    }
    return exact.get(text)


def naive_last(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Repeat the latest observation as a universal fallback."""
    del frequency
    values = _history(history)
    return _validate([values[-1]] * horizon, horizon)


def naive_drift(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Extend the average end-to-end change of a nontrivial series."""
    del frequency
    values = _history(history)
    if len(values) < 3:
        raise NotApplicable("drift requires at least three observations")
    drift = (values[-1] - values[0]) / (len(values) - 1)
    return _validate([values[-1] + drift * step for step in range(1, horizon + 1)], horizon)


def seasonal_naive(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Repeat the latest complete season when its period is known."""
    values = _history(history)
    period = _period(frequency)
    if period is None or len(values) < 2 * period:
        raise NotApplicable("seasonal naive requires two complete seasons")
    return _validate([values[-period + step % period] for step in range(horizon)], horizon)


def ses(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Fit a stable local level by optimized simple exponential smoothing."""
    del frequency
    values = _history(history)
    if len(values) < 4:
        raise NotApplicable("SES requires at least four observations")
    best_error = math.inf
    best_level = values[-1]
    for alpha in (0.1, 0.2, 0.4, 0.6, 0.8, 0.95):
        level = values[0]
        error = 0.0
        for actual in values[1:]:
            error += (actual - level) ** 2
            level = alpha * actual + (1.0 - alpha) * level
        if error < best_error:
            best_error, best_level = error, level
    return _validate([best_level] * horizon, horizon)


def _holt(values: Sequence[float], horizon: int, alpha: float, beta: float, phi: float) -> tuple[list[float], float]:
    level = float(values[0])
    trend = (float(values[-1]) - level) / max(len(values) - 1, 1)
    error = 0.0
    for actual in values[1:]:
        predicted = level + phi * trend
        error += (float(actual) - predicted) ** 2
        previous = level
        level = alpha * float(actual) + (1.0 - alpha) * predicted
        trend = beta * (level - previous) + (1.0 - beta) * phi * trend
    forecasts = []
    damping = 0.0
    for step in range(1, horizon + 1):
        damping += phi**step
        forecasts.append(level + damping * trend)
    return forecasts, error


def holt_damped_trend(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Extrapolate a locally estimated trend with geometric damping."""
    del frequency
    values = _history(history)
    if len(values) < 6:
        raise NotApplicable("damped Holt requires at least six observations")
    candidates = [
        _holt(values, horizon, alpha, beta, phi)
        for alpha in (0.2, 0.5, 0.8)
        for beta in (0.1, 0.3)
        for phi in (0.8, 0.9, 0.98)
    ]
    forecast_values, _ = min(candidates, key=lambda item: item[1])
    return _validate(forecast_values, horizon)


def ets_auto(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Choose a level, damped-trend, or seasonal exponential-smoothing form."""
    values = _history(history)
    if len(values) < 8:
        raise NotApplicable("automatic ETS requires at least eight observations")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    specifications: list[tuple[str | None, bool, str | None, int | None]] = [
        (None, False, None, None),
        ("add", True, None, None),
    ]
    period = _period(frequency)
    if period is not None and len(values) >= 2 * period:
        specifications.append(("add", True, "add", period))
    fits = []
    for trend, damped, seasonal, seasonal_periods in specifications:
        try:
            if trend is None:
                fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
            else:
                fit = ExponentialSmoothing(
                    values,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True)
            fits.append(fit)
        except (ValueError, RuntimeError, OverflowError):
            continue
    if not fits:
        raise NotApplicable("no ETS specification fitted")
    best = min(fits, key=lambda fit: float(fit.sse))
    return _validate(best.forecast(horizon), horizon)


def arima_auto(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Select a small ARIMA order grid by finite Akaike information criterion."""
    del frequency
    values = _history(history)
    if len(values) < 12:
        raise NotApplicable("automatic ARIMA requires at least twelve observations")
    from statsmodels.tsa.arima.model import ARIMA

    fits = []
    for order in ((1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1)):
        try:
            model = ARIMA(values, order=order)
            if order == (0, 1, 1):
                differences = [right - left for left, right in zip(values, values[1:])]
                innovation_variance = max(
                    statistics.fmean(value * value for value in differences),
                    1e-8,
                )
                fit = model.fit(start_params=[0.0, innovation_variance])
            else:
                fit = model.fit()
            if math.isfinite(float(fit.aic)):
                fits.append(fit)
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            continue
    if not fits:
        raise NotApplicable("no ARIMA order fitted")
    best = min(fits, key=lambda fit: float(fit.aic))
    return _validate(best.forecast(horizon), horizon)


def theta_classic(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Combine a linear drift projection with an exponentially smoothed level."""
    del frequency
    values = _history(history)
    if len(values) < 8:
        raise NotApplicable("Theta requires at least eight observations")
    level = ses(values, horizon, "")
    drift = (values[-1] - values[0]) / (len(values) - 1)
    result = [level[step - 1] + 0.5 * drift * step for step in range(1, horizon + 1)]
    return _validate(result, horizon)


def _intervals(values: Sequence[float]) -> tuple[list[int], list[float]]:
    indices = [index for index, value in enumerate(values) if value > 0.0]
    if len(indices) < 2:
        raise NotApplicable("intermittent method requires two positive observations")
    return [indices[index] - indices[index - 1] for index in range(1, len(indices))], [values[index] for index in indices]


def croston_sba(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Forecast intermittent demand with the bias-corrected Croston ratio."""
    del frequency
    values = _history(history)
    intervals, positive = _intervals(values)
    alpha = 0.1
    size = positive[0]
    gap = float(intervals[0])
    for observed, interval in zip(positive[1:], intervals):
        size += alpha * (observed - size)
        gap += alpha * (interval - gap)
    estimate = (1.0 - alpha / 2.0) * size / max(gap, 1.0)
    return _validate([estimate] * horizon, horizon)


def tsb(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Forecast intermittent demand by smoothing occurrence and positive size."""
    del frequency
    values = _history(history)
    positive = [value for value in values if value > 0.0]
    if len(positive) < 2:
        raise NotApplicable("TSB requires two positive observations")
    probability = float(values[0] > 0.0)
    size = positive[0]
    alpha = beta = 0.1
    for value in values[1:]:
        occurred = float(value > 0.0)
        probability += beta * (occurred - probability)
        if occurred:
            size += alpha * (value - size)
    return _validate([probability * size] * horizon, horizon)


SKILLS: dict[str, ForecastSkill] = {
    "naive_last": naive_last,
    "naive_drift": naive_drift,
    "seasonal_naive": seasonal_naive,
    "ses": ses,
    "holt_damped_trend": holt_damped_trend,
    "ets_auto": ets_auto,
    "arima_auto": arima_auto,
    "theta_classic": theta_classic,
    "croston_sba": croston_sba,
    "tsb": tsb,
}


def _folds(history: Sequence[float], horizon: int) -> tuple[tuple[list[float], list[float]], ...]:
    width = min(horizon, max(1, len(history) // 10))
    cutoffs = [len(history) - width * offset for offset in (3, 2, 1)]
    if cutoffs[0] < max(8, 2 * width):
        return ()
    return tuple((list(history[:cutoff]), list(history[cutoff : cutoff + width])) for cutoff in cutoffs)


def _point_metrics(truth: Sequence[float], prediction: Sequence[float]) -> tuple[float, float]:
    scale = statistics.fmean(abs(value) for value in truth)
    absolute = statistics.fmean(abs(actual - predicted) for actual, predicted in zip(truth, prediction))
    squared = math.sqrt(statistics.fmean((actual - predicted) ** 2 for actual, predicted in zip(truth, prediction)))
    if scale == 0.0:
        return (0.0, 0.0) if absolute == 0.0 else (5.0, 5.0)
    return min(5.0, absolute / scale), min(5.0, squared / scale)


def _skill_score(skill: ForecastSkill, folds: Sequence[tuple[list[float], list[float]]], frequency: str) -> float:
    scores = []
    for prefix, truth in folds:
        prediction = _validate(skill(prefix, len(truth), frequency), len(truth))
        smae, srmse = _point_metrics(truth, prediction)
        scores.append(0.5 * smae + 0.5 * srmse)
    return statistics.fmean(scores)


def forecast(history: Sequence[float], horizon: int, frequency: str) -> list[float]:
    """Select an applicable forecasting skill using three causal hindcasts."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    values = _history(history)
    folds = _folds(values, horizon)
    if len(folds) != 3:
        return naive_last(values, horizon, frequency)

    ranked = []
    for name, skill in SKILLS.items():
        try:
            ranked.append((_skill_score(skill, folds, frequency), name, skill))
        except (NotApplicable, ArithmeticError, RuntimeError, TypeError, ValueError):
            continue
    for _, _, skill in sorted(ranked, key=lambda item: (item[0], item[1])):
        try:
            return _validate(skill(values, horizon, frequency), horizon)
        except (NotApplicable, ArithmeticError, RuntimeError, TypeError, ValueError):
            continue
    return naive_last(values, horizon, frequency)
