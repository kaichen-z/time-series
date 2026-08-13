"""Optional numeric-only TSFM adapters used by the Coding Agent ablation."""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ChronosConfig:
    model_id: str = "amazon/chronos-bolt-small"
    device_map: str = "cpu"
    cache_dir: str | None = None
    local_files_only: bool = False


class ChronosForecaster:
    """Lazy adapter exposing the same numeric contract as generated skills."""

    def __init__(self, config: ChronosConfig | None = None) -> None:
        self.config = config or ChronosConfig()
        self._pipeline = None
        self._torch = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            chronos = importlib.import_module("chronos")
            self._torch = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError("Install the optional Chronos dependency with pip install -e '.[chronos]'") from error
        kwargs = {
            "device_map": self.config.device_map,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        self._pipeline = chronos.BaseChronosPipeline.from_pretrained(self.config.model_id, **kwargs)

    def forecast(
        self, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]:
        del frequency
        self._load()
        context = self._torch.tensor(history, dtype=self._torch.float32)
        _quantiles, point = self._pipeline.predict_quantiles(
            inputs=context,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        row = point[0].tolist() if hasattr(point[0], "tolist") else point[0]
        return tuple(float(value) for value in row)
