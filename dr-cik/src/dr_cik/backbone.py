"""Chronos zero-shot forecast backbone: never fit or fine-tuned on Dr-CiK data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Forecast, TaskView

DEFAULT_CACHE_DIR = "/raid/home/air/khoutaibi/models"


class BackboneUnavailableError(RuntimeError):
    """Raised when the chronos-forecasting package or a checkpoint can't be loaded."""


@dataclass(frozen=True)
class ChronosBackboneConfig:
    """Chronos checkpoint and sampling configuration."""

    model_id: str = "amazon/chronos-bolt-base"
    device_map: str = "cuda"
    torch_dtype: str = "float32"
    cache_dir: str | None = DEFAULT_CACHE_DIR
    local_files_only: bool = False
    num_samples: int = 25


class ChronosForecastBackbone:
    """Zero-shot Chronos forecaster; only ever calls predict/predict_quantiles, never fit."""

    def __init__(self, config: ChronosBackboneConfig | None = None, runtime_module: Any | None = None) -> None:
        self.config = config or ChronosBackboneConfig()
        self._runtime_module = runtime_module
        self._pipeline: Any | None = None

    def _load_runtime(self) -> Any:
        if self._runtime_module is not None:
            return self._runtime_module
        try:
            import chronos
        except ImportError as exc:
            raise BackboneUnavailableError("chronos-forecasting is not installed; pip install 'dr-cik[chronos]'") from exc
        return chronos

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        runtime = self._load_runtime()
        if self.config.cache_dir:
            Path(self.config.cache_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        try:
            self._pipeline = runtime.BaseChronosPipeline.from_pretrained(
                self.config.model_id,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
            )
        except Exception as exc:
            raise BackboneUnavailableError(f"Failed to load Chronos checkpoint {self.config.model_id}: {exc}") from exc
        return self._pipeline

    def warm_up(self) -> None:
        """Force the checkpoint to download/load now, for a `download-models` CLI step."""
        self._ensure_pipeline()

    def forecast(self, task_view: TaskView, num_samples: int | None = None) -> Forecast:
        """Zero-shot forecast task_view.prediction_length steps from task_view.history_values."""
        import torch

        pipeline = self._ensure_pipeline()
        sample_count = num_samples or self.config.num_samples
        horizon = task_view.prediction_length
        context = [torch.tensor(task_view.history_values, dtype=torch.float32)]

        runtime = self._load_runtime()
        if pipeline.forecast_type is runtime.ForecastType.SAMPLES:
            raw = pipeline.predict(context, prediction_length=horizon, num_samples=sample_count)
            samples_array = raw[0].numpy()
            method = f"chronos:{self.config.model_id}:mc-samples(S={sample_count})"
        else:

            # avoid Chronos clamping ?
    
            levels = [0.1 + 0.8 * (i + 0.5) / sample_count for i in range(sample_count)]
            quantiles, _mean = pipeline.predict_quantiles(context, prediction_length=horizon, quantile_levels=levels)
            samples_array = quantiles[0].numpy().T
            method = f"chronos:{self.config.model_id}:quantile-ensemble(S={sample_count})"

        if samples_array.shape != (sample_count, horizon):
            raise BackboneUnavailableError(f"Chronos returned shape {samples_array.shape}, expected ({sample_count}, {horizon})")
        if not (samples_array == samples_array).all():  # NaN check without numpy import at module scope
            raise BackboneUnavailableError("Chronos produced non-finite forecast values")

        mean = samples_array.mean(axis=0)
        return Forecast(
            mean=tuple(float(value) for value in mean),
            samples=tuple(tuple(float(value) for value in row) for row in samples_array),
            method=method,
        )
