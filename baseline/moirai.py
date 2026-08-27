"""Moirai 2.0 as a sampling baseline: its quantile forecast turned into trajectories."""
from __future__ import annotations

import importlib
from typing import Any, Sequence

from common.tsfm import BackboneUnavailableError, MoiraiConfig, resolve_device

from .forecasters import quantile_paths

# Moirai 2.0 emits a fixed 9-quantile grid. Read off the module at load time rather than
# hardcoded here, so a checkpoint with a different grid cannot be silently misread.
_FALLBACK_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class MoiraiSampleForecaster:
    """Zero-shot Moirai 2.0 returning trajectories rather than the median path.

    common/tsfm.py's MoiraiForecaster collapses the same forecast to its median, which is the
    right answer for the evolution harness and the wrong one here: sMAE is scored on the sample
    mean and the submission needs the full ensemble.
    """

    name = "moirai"

    def __init__(self, config: MoiraiConfig | None = None, seed: int = 0) -> None:
        self.config = config or MoiraiConfig()
        self.seed = seed
        self._module: Any | None = None
        self._levels: tuple[float, ...] = _FALLBACK_LEVELS

    def _ensure_module(self) -> tuple[Any, Any]:
        try:
            torch = importlib.import_module("torch")
            moirai = importlib.import_module("uni2ts.model.moirai2")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Moirai is the configured backbone but uni2ts is not installed. "
                "Install it with: pip install --no-deps uni2ts"
            ) from error
        if self._module is None:
            try:
                self._module = moirai.Moirai2Module.from_pretrained(self.config.model_id).to(
                    resolve_device(self.config.device)
                )
            except Exception as error:
                raise BackboneUnavailableError(
                    f"Could not load Moirai checkpoint {self.config.model_id!r}: {error}"
                ) from error
            levels = getattr(self._module, "quantile_levels", None)
            if levels:
                self._levels = tuple(float(level) for level in levels)
        return self._module, (torch, moirai)

    def quantiles(self, history: Sequence[float], horizon: int) -> list[list[float]]:
        """Predict the quantile grid for one series, indexed [level][step]."""
        module, (torch, moirai) = self._ensure_module()
        context = [float(value) for value in history[-self.config.max_context :]]
        device = resolve_device(self.config.device)
        try:
            model = moirai.Moirai2Forecast(
                module=module,
                prediction_length=horizon,
                context_length=len(context),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).to(device)
            past = torch.tensor(context, dtype=torch.float32).reshape(1, -1, 1).to(device)
            with torch.no_grad():
                predicted = model(
                    past_target=past,
                    past_observed_target=torch.ones_like(past, dtype=torch.bool),
                    past_is_pad=torch.zeros(1, len(context), dtype=torch.bool).to(device),
                )
        except Exception as error:
            raise BackboneUnavailableError(f"Moirai inference failed: {error}") from error

        grid = [[float(value) for value in row] for row in predicted[0].tolist()]
        if len(grid) != len(self._levels):
            raise BackboneUnavailableError(
                f"Moirai returned {len(grid)} quantiles for {len(self._levels)} levels"
            )
        if any(len(row) != horizon for row in grid):
            raise BackboneUnavailableError(f"Moirai returned a row that is not {horizon} long")
        return grid

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        grid = self.quantiles(history, horizon)
        return quantile_paths(grid, self._levels, samples, seed=self.seed)
