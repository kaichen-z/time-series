"""Chronos-Bolt as a sampling baseline: its quantile forecast turned into trajectories.

common/tsfm.py's ChronosForecaster keeps only the median point forecast, which is right for the
evolution harness and wrong here, same reasoning as MoiraiSampleForecaster.
"""
from __future__ import annotations

from typing import Any, Sequence

from common.tsfm import BackboneUnavailableError, ChronosConfig, resolve_device

from .forecasters import quantile_paths

_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ChronosSampleForecaster:
    """Zero-shot Chronos-Bolt returning trajectories rather than the median path."""

    name = "chronos"

    def __init__(self, config: ChronosConfig | None = None, seed: int = 0) -> None:
        self.config = config or ChronosConfig()
        self.seed = seed
        self._pipeline: Any | None = None

    def _ensure_pipeline(self) -> tuple[Any, Any]:
        try:
            import torch
            import chronos
        except ImportError as error:
            raise BackboneUnavailableError(
                "Chronos is the configured backbone but is not installed. "
                "Install it with: pip install -e '.[chronos]'"
            ) from error
        if self._pipeline is None:
            try:
                self._pipeline = chronos.BaseChronosPipeline.from_pretrained(
                    self.config.model_id, device_map=resolve_device(self.config.device_map)
                )
            except Exception as error:
                raise BackboneUnavailableError(
                    f"Could not load Chronos checkpoint {self.config.model_id!r}: {error}"
                ) from error
        return self._pipeline, torch

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        pipeline, torch = self._ensure_pipeline()
        context = history[-self.config.max_context :] if self.config.max_context else history
        try:
            quantiles, _mean = pipeline.predict_quantiles(
                inputs=torch.tensor([float(v) for v in context], dtype=torch.float32),
                prediction_length=horizon,
                quantile_levels=list(_LEVELS),
            )
        except Exception as error:
            raise BackboneUnavailableError(f"Chronos inference failed: {error}") from error
        # predict_quantiles is [batch, step, level]; quantile_paths wants [level][step].
        grid = quantiles[0].tolist()
        by_level = [[row[level] for row in grid] for level in range(len(_LEVELS))]
        return quantile_paths(by_level, _LEVELS, samples, seed=self.seed)
