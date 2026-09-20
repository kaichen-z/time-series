"""A pure-Python statistical forecaster library (no numpy/statsmodels). The Numerical loop
evolves a COMBINATION (weights) over this library + the cached TSFM candidates -- i.e. it
genuinely evolves the numerical 'dictionary', not just an expose on/off switch."""
from __future__ import annotations
from statistics import mean


def _naive(hv, h):
    return [hv[-1]] * h if hv else [0.0] * h


def _mean(hv, h):
    m = mean(hv) if hv else 0.0
    return [m] * h


def _drift(hv, h):
    if len(hv) < 2:
        return _naive(hv, h)
    slope = (hv[-1] - hv[0]) / (len(hv) - 1)
    return [hv[-1] + slope * (i + 1) for i in range(h)]


def _linear_trend(hv, h):
    n = len(hv)
    if n < 2:
        return _naive(hv, h)
    xbar = (n - 1) / 2.0
    ybar = mean(hv)
    sxx = sum((i - xbar) ** 2 for i in range(n))
    sxy = sum((i - xbar) * (hv[i] - ybar) for i in range(n))
    b = sxy / sxx if sxx else 0.0
    a = ybar - b * xbar
    return [a + b * (n + i) for i in range(h)]


def _ewm(hv, h, alpha=0.3):
    if not hv:
        return [0.0] * h
    level = hv[0]
    for x in hv[1:]:
        level = alpha * x + (1 - alpha) * level
    return [level] * h


def _moving_average(hv, h, k=7):
    if not hv:
        return [0.0] * h
    m = mean(hv[-min(len(hv), k):])
    return [m] * h


LIBRARY = {
    "naive": _naive, "mean": _mean, "drift": _drift, "linear_trend": _linear_trend,
    "ewm": lambda hv, h: _ewm(hv, h, 0.3), "moving_avg": lambda hv, h: _moving_average(hv, h, 7),
}


def stat_forecasts(hv, horizon):
    """Return {method_name: forecast} for all pure-code statistical methods."""
    return {name: fn(list(hv), horizon) for name, fn in LIBRARY.items()}
