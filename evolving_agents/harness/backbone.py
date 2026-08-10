"""Precomputes a Chronos-Bolt forecast in the parent process so sandboxed code gets it as plain data."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from dr_cik.forecasters.chronos import ChronosConfig, ChronosForecaster
from dr_cik.models import TaskView

from ..models import NumericTaskView

logger = logging.getLogger(__name__)


@dataclass
class BackboneProvider:
    """Wraps ChronosForecaster and caches one mean forecast per (series, horizon)."""

    forecaster: ChronosForecaster | None = None
    num_samples: int = 25

    def __post_init__(self) -> None:
        """Build a default Chronos forecaster when none was injected."""
        self.forecaster = self.forecaster or ChronosForecaster(ChronosConfig(num_samples=self.num_samples))
        self._cache: dict[tuple[int, int], tuple[float, ...]] = {}

    def mean_forecast(self, view: NumericTaskView) -> tuple[float, ...] | None:
        """Return Chronos's mean forecast for a numeric view, or None if the backbone is unavailable."""
        key = (hash(view.history_values), view.prediction_length)
        if key in self._cache:
            return self._cache[key]
        task_view = TaskView(
            benchmark_id=view.benchmark_id,
            entity_name="",
            target_name="",
            target_description="",
            frequency=view.frequency,
            prediction_length=view.prediction_length,
            seasonal_period=view.seasonal_period,
            history_timestamps=tuple(str(index) for index in range(len(view.history_values))),
            history_values=view.history_values,
            future_timestamps=tuple(str(index) for index in range(view.prediction_length)),
            documents=(),
        )
        try:
            forecast = self.forecaster.forecast(task_view, num_samples=self.num_samples)
        except Exception as exc:
            logger.warning("backbone unavailable for %s, continuing without it: %s", view.benchmark_id, exc)
            return None
        self._cache[key] = forecast.mean
        return forecast.mean


def naive_backbone(view: NumericTaskView) -> tuple[float, ...]:
    """Return a last-value-persistence forecast, the dependency-free stand-in used in tests."""
    last = view.history_values[-1] if view.history_values else 0.0
    return tuple(float(last) for _ in range(view.prediction_length))


def seasonal_naive(history: tuple[float, ...], horizon: int, period: int | None) -> tuple[float, ...]:
    """Repeat the last full cycle, falling back to the mean when no usable period is given."""
    if not history:
        return tuple(0.0 for _ in range(horizon))
    if not period or period <= 0 or period > len(history):
        value = statistics.fmean(history)
        return tuple(float(value) for _ in range(horizon))
    tail = history[-period:]
    return tuple(float(tail[index % period]) for index in range(horizon))
