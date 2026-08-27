"""The sampling interface every baseline implements, plus a dependency-free reference forecaster."""
from __future__ import annotations

import random
import re
import statistics
from typing import Protocol, Sequence

# Fallback periods when a task states no usable seasonal_period, keyed by the frequency string
# Dr-CiK uses. Missing entries fall through to 1, i.e. no seasonality.
_PERIOD_BY_FREQUENCY = {
    "1 hour": 24, "1h": 24, "h": 24,
    "1 day": 7, "d": 7,
    "1 week": 52, "w": 52,
    "1 month": 12, "m": 12, "ms": 12, "me": 12,
    "1 minute": 60, "5 minutes": 12, "10 minutes": 6, "15 minutes": 4, "30 minutes": 2,
    "1 quarter": 4, "qs": 4,
}


class SampleForecaster(Protocol):
    """Produces trajectories, not a single path: the submission needs them and sMAE averages them."""

    name: str

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]: ...


def seasonal_period(frequency: str, seasonal_period_field: object) -> int:
    """Best-effort period for a task, from its stated seasonal_period then its frequency."""
    if seasonal_period_field is not None:
        digits = re.search(r"\d+", str(seasonal_period_field))
        if digits and int(digits.group()) > 1:
            return int(digits.group())
    return _PERIOD_BY_FREQUENCY.get(str(frequency).strip().lower(), 1)


def quantile_paths(
    quantiles: Sequence[Sequence[float]], levels: Sequence[float], samples: int, seed: int = 0
) -> tuple[tuple[float, ...], ...]:
    """Turn a per-step quantile forecast into coherent trajectories.

    One uniform draw per trajectory, read at that same level across every step, rather than an
    independent draw per step: the latter produces paths that zig-zag across the whole predictive
    band and destroys the temporal structure the model actually predicted.

    quantiles is indexed [level][step].
    """
    if len(quantiles) != len(levels):
        raise ValueError(f"{len(quantiles)} quantile rows for {len(levels)} levels")
    if not levels:
        raise ValueError("no quantile levels")
    rng = random.Random(seed)
    horizon = len(quantiles[0])
    paths = []
    for _ in range(samples):
        u = rng.random()
        paths.append(tuple(_interpolate(u, levels, [row[step] for row in quantiles])
                           for step in range(horizon)))
    return tuple(paths)


def _interpolate(u: float, levels: Sequence[float], values: Sequence[float]) -> float:
    """Read the quantile function at u, clamping outside the outermost known levels."""
    if u <= levels[0]:
        return float(values[0])
    for index in range(1, len(levels)):
        if u <= levels[index]:
            span = levels[index] - levels[index - 1]
            weight = 0.0 if span == 0 else (u - levels[index - 1]) / span
            return float(values[index - 1] + weight * (values[index] - values[index - 1]))
    return float(values[-1])


class SeasonalNaive:
    """Repeats the last seasonal cycle, with residual noise for the spread.

    Not a paper baseline: it is the floor every real baseline must clear, and it exercises the
    whole harness without loading a model.
    """

    name = "seasonal_naive"

    def __init__(self, frequency: str = "1 day", period_field: object = None, seed: int = 0) -> None:
        self.period = seasonal_period(frequency, period_field)
        self.seed = seed

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        values = [float(value) for value in history]
        if not values:
            raise ValueError("cannot forecast from an empty history")
        period = self.period if self.period > 1 and len(values) > self.period else 1
        point = [values[-period + (step % period)] for step in range(horizon)]

        spread = _residual_spread(values, period)
        rng = random.Random(self.seed)
        return tuple(
            tuple(value + rng.gauss(0.0, spread) for value in point) for _ in range(samples)
        )


def _residual_spread(values: Sequence[float], period: int) -> float:
    """Standard deviation of the season-over-season change, as a stand-in for forecast spread."""
    if len(values) <= period:
        return 0.0
    residuals = [values[i] - values[i - period] for i in range(period, len(values))]
    return statistics.pstdev(residuals) if len(residuals) > 1 else 0.0
