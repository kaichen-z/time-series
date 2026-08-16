"""Numeric-only zero-shot TSFM adapters (Chronos, TimesFM), shared across agents."""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class BackboneUnavailableError(RuntimeError):
    """Raised when a configured forecasting backbone cannot be loaded."""


@dataclass(frozen=True)
class ChronosConfig:
    model_id: str = "amazon/chronos-bolt-small"
    device_map: str = "cpu"
    cache_dir: str | None = None
    local_files_only: bool = False
    max_context: int | None = None
    max_horizon: int | None = None
    validate_output: bool = False

    def __post_init__(self) -> None:
        if self.max_context is not None and self.max_context <= 0:
            raise ValueError("Chronos context limit must be positive")
        if self.max_horizon is not None and self.max_horizon <= 0:
            raise ValueError("Chronos horizon limit must be positive")
        if not self.device_map.strip():
            raise ValueError("Chronos device_map must not be empty")


class ChronosForecaster:
    """Lazy zero-shot Chronos-Bolt adapter using Amazon's official API."""

    def __init__(
        self,
        config: ChronosConfig | None = None,
        runtime_module: Any | None = None,
        tensor_module: Any | None = None,
    ) -> None:
        self.config = config or ChronosConfig()
        self._runtime_module = runtime_module
        self._tensor_module = tensor_module
        self._pipeline: Any | None = None

    def _load_runtime(self) -> tuple[Any, Any]:
        try:
            chronos = self._runtime_module or importlib.import_module("chronos")
            torch = self._tensor_module or importlib.import_module("torch")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Chronos is the configured backbone but is not installed. "
                "Install it with: pip install -e '.[chronos]'"
            ) from error
        return chronos, torch

    def _ensure_pipeline(self) -> tuple[Any, Any]:
        chronos, torch = self._load_runtime()
        if self._pipeline is not None:
            return self._pipeline, torch
        load_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir:
            load_kwargs["cache_dir"] = str(
                Path(self.config.cache_dir).expanduser().resolve()
            )
        try:
            self._pipeline = chronos.BaseChronosPipeline.from_pretrained(
                self.config.model_id,
                **load_kwargs,
            )
        except Exception as error:
            location = "local cache" if self.config.local_files_only else "Hugging Face or cache"
            raise BackboneUnavailableError(
                f"Could not load Chronos checkpoint {self.config.model_id!r} "
                f"from {location}: {error}"
            ) from error
        return self._pipeline, torch

    def forecast(self, history: Sequence[float], horizon: int) -> tuple[float, ...]:
        if self.config.max_horizon is not None and horizon > self.config.max_horizon:
            raise ValueError(
                f"Task horizon {horizon} exceeds Chronos max_horizon "
                f"{self.config.max_horizon}"
            )
        pipeline, torch = self._ensure_pipeline()
        context = (
            history[-self.config.max_context :]
            if self.config.max_context is not None
            else history
        )
        try:
            context_tensor = torch.tensor(context, dtype=torch.float32)
            _quantiles, point_forecast = pipeline.predict_quantiles(
                inputs=context_tensor,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            row = point_forecast[0]
            if hasattr(row, "tolist"):
                row = row.tolist()
            values = tuple(float(value) for value in row)
        except Exception as error:
            raise BackboneUnavailableError(f"Chronos inference failed: {error}") from error
        if self.config.validate_output:
            if len(values) != horizon:
                raise BackboneUnavailableError(
                    "Chronos returned a point forecast with the wrong horizon length"
                )
            if not all(math.isfinite(value) for value in values):
                raise BackboneUnavailableError("Chronos returned non-finite point forecasts")
        return values


@dataclass(frozen=True)
class TimesFMConfig:
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


class TimesFMForecaster:
    """Lazy TimesFM 2.5 adapter using the official Google Research API."""

    def __init__(
        self,
        config: TimesFMConfig | None = None,
        runtime_module: Any | None = None,
    ) -> None:
        self.config = config or TimesFMConfig()
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

    def forecast(self, history: Sequence[float], horizon: int) -> tuple[float, ...]:
        if horizon > self.config.max_horizon:
            raise ValueError(
                f"Task horizon {horizon} exceeds TimesFM max_horizon "
                f"{self.config.max_horizon}"
            )
        model = self._ensure_model()
        try:
            point_forecast, _quantile_forecast = model.forecast(
                horizon=horizon,
                inputs=[list(history)],
            )
            values = tuple(float(value) for value in point_forecast[0])
        except Exception as error:
            raise BackboneUnavailableError(f"TimesFM inference failed: {error}") from error
        if len(values) != horizon:
            raise BackboneUnavailableError(
                "TimesFM returned a point forecast with the wrong horizon length"
            )
        if not all(math.isfinite(value) for value in values):
            raise BackboneUnavailableError("TimesFM returned non-finite point forecasts")
        return values
