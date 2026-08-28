"""Classical statistical baselines: naive, SES, ETS, ARIMA.

Each fits on the full history then draws sample paths with the model's own simulate(), which
carries the fitted innovation variance forward rather than a hand-picked noise scale.
"""
from __future__ import annotations

import warnings
from typing import Sequence

from .forecasters import seasonal_period

warnings.filterwarnings("ignore", module="statsmodels")


class NaiveForecaster:
    """Persistence forecast: repeat the last value, innovations from the first differences.

    Not seasonal_period aware, unlike SeasonalNaive; this is the plain random-walk baseline the
    classical methods below are compared against.
    """

    name = "naive"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        import numpy as np

        values = np.asarray([float(value) for value in history], dtype=float)
        if values.size == 0:
            raise ValueError("cannot forecast from an empty history")
        diffs = np.diff(values)
        scale = float(diffs.std()) if diffs.size > 1 else 0.0
        rng = np.random.default_rng(self.seed)
        # A random walk's step variance grows linearly with horizon; simulate it directly rather
        # than adding independent per-step noise to a flat line, which would understate spread.
        steps = rng.normal(0.0, scale, size=(samples, horizon))
        paths = values[-1] + np.cumsum(steps, axis=1)
        return tuple(tuple(float(v) for v in path) for path in paths)


class SESForecaster:
    """Simple exponential smoothing: flat forecast, additive-error simulation for spread."""

    name = "ses"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        values = [float(value) for value in history]
        try:
            result = SimpleExpSmoothing(values, initialization_method="estimated").fit()
            sim = result.simulate(
                nsimulations=horizon, repetitions=samples, random_state=self.seed
            )
        except Exception:
            return _fallback_paths(values, horizon, samples, self.seed)
        return _paths_from_simulation(sim)


class ETSForecaster:
    """Holt-Winters exponential smoothing: damped trend, plus seasonality when data supports it."""

    name = "ets"

    def __init__(self, frequency: str = "1 day", period_field: object = None, seed: int = 0) -> None:
        self.period = seasonal_period(frequency, period_field)
        self.seed = seed

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        values = [float(value) for value in history]
        use_seasonal = self.period > 1 and len(values) >= 2 * self.period
        try:
            result = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=self.period if use_seasonal else None,
                initialization_method="estimated",
            ).fit()
            sim = result.simulate(
                nsimulations=horizon, repetitions=samples, random_state=self.seed
            )
        except Exception:
            return _fallback_paths(values, horizon, samples, self.seed)
        return _paths_from_simulation(sim)


def _select_d(values: list, max_d: int = 2) -> int:
    """Number of differences an ADF test says the series needs, capped at max_d.

    Standard Hyndman-Khandakar approach: keep differencing while the augmented Dickey-Fuller
    test still rejects stationarity (p-value above 0.05), rather than guessing d up front.
    """
    from statsmodels.tsa.stattools import adfuller

    series = list(values)
    for d in range(max_d + 1):
        if len(series) < 8:  # too short for the test to mean anything; stop differencing
            return d
        try:
            _, p_value, *_ = adfuller(series, autolag="AIC")
        except Exception:
            return d
        if p_value < 0.05:
            return d
        series = [b - a for a, b in zip(series, series[1:])]
    return max_d


def _best_arima_fit(values: list, max_p: int = 5, max_q: int = 5, max_d: int = 2):
    """Grid search (p, q) at the ADF-selected d, keeping the fit with the lowest AICc.

    No pmdarima installed, so this is a from-scratch auto-ARIMA: not stepwise, a full grid,
    since runtime is not a constraint here and a full grid cannot miss a better order that a
    stepwise search skips over.
    """
    from statsmodels.tsa.arima.model import ARIMA

    d = _select_d(values, max_d)
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0 and d == 0:
                continue  # no dynamics at all; not a useful candidate
            try:
                fitted = ARIMA(values, order=(p, d, q)).fit()
            except Exception:
                continue
            if best is None or fitted.aicc < best.aicc:
                best = fitted
    return best


class ARIMAForecaster:
    """ARIMA, (p, d, q) chosen per task by a full AICc grid search, not a fixed guess."""

    name = "arima"

    def __init__(self, seed: int = 0, max_p: int = 5, max_q: int = 5, max_d: int = 2) -> None:
        self.seed = seed
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        values = [float(value) for value in history]
        best = _best_arima_fit(values, self.max_p, self.max_q, self.max_d)
        if best is None:
            return _fallback_paths(values, horizon, samples, self.seed)
        try:
            sim = best.simulate(
                nsimulations=horizon, repetitions=samples, anchor="end", random_state=self.seed
            )
        except Exception:
            return _fallback_paths(values, horizon, samples, self.seed)
        return _paths_from_simulation(sim)


def _paths_from_simulation(sim) -> tuple[tuple[float, ...], ...]:
    """statsmodels simulate() returns (horizon, repetitions); the submission wants the transpose.

    ARIMA's statespace simulate adds a singleton middle axis (horizon, 1, repetitions) that
    Holt-Winters' does not; squeeze it away rather than special-casing by forecaster.
    """
    import numpy as np

    array = np.asarray(sim, dtype=float)
    horizon = array.shape[0]
    return tuple(tuple(float(v) for v in path) for path in array.reshape(horizon, -1).T)


def _fallback_paths(
    values: Sequence[float], horizon: int, samples: int, seed: int
) -> tuple[tuple[float, ...], ...]:
    """A model that fails to fit (e.g. a constant or near-constant series) still owes a forecast."""
    return NaiveForecaster(seed=seed).forecast_samples(values, horizon, samples)
