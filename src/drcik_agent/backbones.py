from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import Diagnosis, ForecastTask


class BackboneUnavailableError(RuntimeError):
    """Raised when a configured forecasting backbone cannot be loaded."""


class ForecastBackbone(Protocol):
    def forecast(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> tuple[tuple[float, ...], str]: ...


class StatisticalForecastBackbone:
    """Small deterministic backbone retained for controlled ablations."""

    def forecast(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> tuple[tuple[float, ...], str]:
        values = task.history_values
        period = diagnosis.seasonal_period
        horizon = task.prediction_length
        if period and 0 < period <= len(values):
            last_cycle = values[-period:]
            drift = [0.0] * period
            if len(values) >= 2 * period:
                previous_cycle = values[-2 * period : -period]
                drift = [
                    current - previous
                    for current, previous in zip(last_cycle, previous_cycle)
                ]
            baseline = []
            for step in range(horizon):
                phase = step % period
                cycle = step // period + 1
                baseline.append(last_cycle[phase] + 0.75 * cycle * drift[phase])
            method = (
                "drifted_seasonal_naive"
                if any(abs(value) > 1e-12 for value in drift)
                else "seasonal_naive"
            )
            return tuple(baseline), method
        slope = diagnosis.slope_per_step
        baseline = tuple(values[-1] + slope * (step + 1) for step in range(horizon))
        return baseline, "damped_linear_trend"


@dataclass(frozen=True)
class TimesFMBackboneConfig:
    model_id: str = "google/timesfm-2.5-200m-pytorch"
    max_context: int = 4096
    max_horizon: int = 1024
    cache_dir: str | None = None
    local_files_only: bool = False
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True

    def __post_init__(self) -> None:
        if self.max_context <= 0 or self.max_horizon <= 0:
            raise ValueError("TimesFM context and horizon limits must be positive")
        if self.max_context + self.max_horizon > 16384:
            raise ValueError("TimesFM 2.5 max_context + max_horizon must not exceed 16384")
        if self.use_continuous_quantile_head and self.max_horizon > 1024:
            raise ValueError("TimesFM 2.5 continuous quantiles support at most 1024 steps")


class TimesFMForecastBackbone:
    """Lazy TimesFM 2.5 adapter using the official Google Research API."""

    def __init__(
        self,
        config: TimesFMBackboneConfig | None = None,
        runtime_module: Any | None = None,
    ) -> None:
        self.config = config or TimesFMBackboneConfig()
        self._runtime_module = runtime_module
        self._model: Any | None = None

    def _load_runtime(self) -> Any:
        if self._runtime_module is not None:
            return self._runtime_module
        try:
            return importlib.import_module("timesfm")
        except ImportError as error:
            raise BackboneUnavailableError(
                "TimesFM is the configured backbone but is not installed. "
                "Install it with: pip install -e '.[timesfm]'"
            ) from error

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        timesfm = self._load_runtime()
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.config.model_id,
                cache_dir=(
                    str(Path(self.config.cache_dir).expanduser().resolve())
                    if self.config.cache_dir
                    else None
                ),
                local_files_only=self.config.local_files_only,
            )
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.config.max_context,
                    max_horizon=self.config.max_horizon,
                    normalize_inputs=self.config.normalize_inputs,
                    use_continuous_quantile_head=self.config.use_continuous_quantile_head,
                    force_flip_invariance=self.config.force_flip_invariance,
                    infer_is_positive=self.config.infer_is_positive,
                    fix_quantile_crossing=self.config.fix_quantile_crossing,
                )
            )
        except Exception as error:
            location = "local cache" if self.config.local_files_only else "Hugging Face or cache"
            raise BackboneUnavailableError(
                f"Could not load TimesFM checkpoint {self.config.model_id!r} from {location}: {error}"
            ) from error
        self._model = model
        return model

    def forecast(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> tuple[tuple[float, ...], str]:
        del diagnosis  # TimesFM 2.5 consumes the numerical history directly.
        if task.prediction_length > self.config.max_horizon:
            raise ValueError(
                f"Task horizon {task.prediction_length} exceeds TimesFM max_horizon "
                f"{self.config.max_horizon}"
            )
        model = self._ensure_model()
        try:
            point_forecast, _quantile_forecast = model.forecast(
                horizon=task.prediction_length,
                inputs=[list(task.history_values)],
            )
            values = tuple(float(value) for value in point_forecast[0])
        except Exception as error:
            raise BackboneUnavailableError(f"TimesFM inference failed: {error}") from error
        if len(values) != task.prediction_length:
            raise BackboneUnavailableError(
                "TimesFM returned a point forecast with the wrong horizon length"
            )
        if not all(math.isfinite(value) for value in values):
            raise BackboneUnavailableError("TimesFM returned non-finite point forecasts")
        return values, f"timesfm-2.5-200m-pytorch:{self.config.model_id}"


class FallbackForecastBackbone:
    """Use a fallback only when the explicitly configured primary cannot load."""

    def __init__(self, primary: ForecastBackbone, fallback: ForecastBackbone) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_error: str | None = None

    def forecast(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
    ) -> tuple[tuple[float, ...], str]:
        try:
            return self.primary.forecast(task, diagnosis)
        except BackboneUnavailableError as error:
            self.last_error = str(error)
            values, method = self.fallback.forecast(task, diagnosis)
            return values, f"statistical_fallback:{method}"


def build_forecast_backbone(
    name: str,
    *,
    timesfm_config: TimesFMBackboneConfig | None = None,
    allow_statistical_fallback: bool = False,
) -> ForecastBackbone:
    normalized = name.strip().lower()
    if normalized == "statistical":
        return StatisticalForecastBackbone()
    if normalized != "timesfm":
        raise ValueError(f"Unknown forecast backbone: {name}")
    primary = TimesFMForecastBackbone(timesfm_config)
    if allow_statistical_fallback:
        return FallbackForecastBackbone(primary, StatisticalForecastBackbone())
    return primary
