"""Lazy TimesFM 2.5 runtime for numerical dictionary candidates."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

from ..dictionary import MethodCandidate
from ..providers import RuntimeUnavailableError
from .manifests import ManifestRegistry, TSFMManifest


class TimesFMRuntime:
    """Execute supported TimesFM 2.5 PyTorch checkpoint candidates."""

    def __init__(
        self,
        model_loader: Callable[[str], Any] | None = None,
        forecast_config_factory: Callable[..., Any] | None = None,
        max_context: int = 1024,
        max_horizon: int = 256,
        manifests: ManifestRegistry | None = None,
    ) -> None:
        if max_context <= 0 or max_horizon <= 0:
            raise ValueError("TimesFM context and horizon limits must be positive")
        if max_context + max_horizon > 16384:
            raise ValueError("TimesFM max_context + max_horizon must not exceed 16384")
        if max_horizon > 1024:
            raise ValueError("TimesFM continuous quantiles support at most 1024 steps")
        self._model_loader = model_loader
        self._forecast_config_factory = forecast_config_factory
        self._manifests = (
            manifests if manifests is not None else ManifestRegistry.load_default()
        )
        self._max_context = max_context
        self._max_horizon = max_horizon
        self._models: dict[tuple[str, int, int], Any] = {}

    def supports(self, candidate: MethodCandidate) -> bool:
        manifest = self._manifest(candidate)
        return manifest is not None and manifest.matches_candidate(
            candidate, provider="timesfm"
        )

    def _manifest(self, candidate: MethodCandidate) -> TSFMManifest | None:
        try:
            manifest = self._manifests[candidate.method_id]
        except KeyError:
            return None
        if manifest.status != "direct" or manifest.adapter != "timesfm":
            return None
        return manifest

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        del frequency  # TimesFM 2.5 does not use a frequency token.
        if not self.supports(candidate):
            raise ValueError("candidate is not a supported TimesFM 2.5 PyTorch checkpoint")
        if horizon <= 0:
            raise ValueError("forecast horizon must be positive")
        if horizon > self._max_horizon:
            raise ValueError(
                f"forecast horizon {horizon} exceeds TimesFM max_horizon "
                f"{self._max_horizon}"
            )

        numpy = self._load_numpy()
        try:
            values = numpy.asarray(history, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("history must be a one-dimensional finite sequence") from error
        if values.ndim != 1 or values.size == 0:
            raise ValueError("history must be a non-empty one-dimensional sequence")
        if not bool(numpy.isfinite(values).all()):
            raise ValueError("history must contain only finite values")

        manifest = self._manifest(candidate)
        assert manifest is not None
        model_id = manifest.checkpoint
        model = self._ensure_model(model_id)
        point_forecast, _quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[values[-self._max_context :].tolist()],
        )
        try:
            point_values = numpy.asarray(point_forecast, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("TimesFM point forecast has an invalid shape") from error
        if point_values.ndim != 2 or point_values.shape[0] != 1:
            raise ValueError("TimesFM point forecast has an invalid shape")
        if point_values.shape[1] != horizon:
            raise ValueError("TimesFM point forecast has the wrong horizon length")
        if not bool(numpy.isfinite(point_values).all()):
            raise ValueError("TimesFM point forecast must contain only finite values")
        return tuple(float(value) for value in point_values[0])

    @staticmethod
    def _load_numpy() -> Any:
        try:
            return importlib.import_module("numpy")
        except ImportError as error:
            raise RuntimeUnavailableError(
                "TimesFM runtime dependencies are not installed; "
                "install them with: pip install -e '.[timesfm]'"
            ) from error

    def _official_components(self) -> tuple[Callable[[str], Any], Callable[..., Any]]:
        if self._model_loader is not None and self._forecast_config_factory is not None:
            return self._model_loader, self._forecast_config_factory
        try:
            timesfm = importlib.import_module("timesfm")
        except ImportError as error:
            raise RuntimeUnavailableError(
                "TimesFM is not installed; install it with: "
                "pip install -e '.[timesfm]'"
            ) from error
        loader = self._model_loader or timesfm.TimesFM_2p5_200M_torch.from_pretrained
        config_factory = self._forecast_config_factory or timesfm.ForecastConfig
        return loader, config_factory

    def _ensure_model(self, model_id: str) -> Any:
        cache_key = (model_id, self._max_context, self._max_horizon)
        cached = self._models.get(cache_key)
        if cached is not None:
            return cached

        try:
            loader, config_factory = self._official_components()
            model = loader(model_id)
            config = config_factory(
                max_context=self._max_context,
                max_horizon=self._max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
            model.compile(config)
        except RuntimeUnavailableError:
            raise
        except Exception as error:
            raise RuntimeUnavailableError(
                str(error) or type(error).__name__
            ) from error
        self._models[cache_key] = model
        return model
