"""Chronos checkpoint adapter for the numerical dictionary runtime boundary."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from typing import Any

from ..dictionary import MethodCandidate
from ..providers import RuntimeUnavailableError
from .manifests import ManifestRegistry, TSFMManifest


CHRONOS_PROVIDER = "chronos"
TSFM_IMPLEMENTATION_KIND = "tsfm_checkpoint"


class ChronosRuntime:
    """Run deterministic Chronos candidates through Amazon's official pipeline."""

    def __init__(
        self,
        model_loader: Callable[..., Any] | None = None,
        device_map: str = "cpu",
        manifests: ManifestRegistry | None = None,
    ) -> None:
        if not isinstance(device_map, str) or not device_map.strip():
            raise ValueError("Chronos device_map must not be empty")
        self._model_loader = model_loader
        self._manifests = (
            manifests if manifests is not None else ManifestRegistry.load_default()
        )
        self.device_map = device_map
        self._pipelines: dict[str, Any] = {}
        self._torch: Any | None = None

    def supports(self, candidate: MethodCandidate) -> bool:
        manifest = self._manifest(candidate)
        return manifest is not None and manifest.matches_candidate(
            candidate, provider=CHRONOS_PROVIDER
        )

    def _manifest(self, candidate: MethodCandidate) -> TSFMManifest | None:
        try:
            manifest = self._manifests[candidate.method_id]
        except KeyError:
            return None
        if manifest.status != "direct" or manifest.adapter != CHRONOS_PROVIDER:
            return None
        return manifest

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        del frequency
        if not self.supports(candidate):
            raise ValueError("Chronos runtime does not support candidate")
        values = self._validated_history(history)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be positive")

        manifest = self._manifest(candidate)
        assert manifest is not None
        checkpoint = manifest.checkpoint
        pipeline = self._pipeline(checkpoint)
        inputs: object = values
        if self._torch is not None:
            inputs = [self._torch.tensor(values, dtype=self._torch.float32)]
        try:
            quantiles, _mean = pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=horizon,
                quantile_levels=[0.5],
            )
        except Exception as error:
            raise RuntimeError(f"Chronos inference failed: {error}") from error
        return self._validated_p50(quantiles, horizon)

    def _pipeline(self, checkpoint: str) -> Any:
        if checkpoint not in self._pipelines:
            loader = self._model_loader or self._load_official_pipeline
            try:
                self._pipelines[checkpoint] = loader(
                    checkpoint,
                    device_map=self.device_map,
                )
            except RuntimeUnavailableError:
                raise
            except Exception as error:
                raise RuntimeUnavailableError(
                    str(error) or type(error).__name__
                ) from error
        return self._pipelines[checkpoint]

    def _load_official_pipeline(self, checkpoint: str, *, device_map: str) -> Any:
        try:
            chronos = importlib.import_module("chronos")
            self._torch = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeUnavailableError(
                "Chronos runtime is unavailable; install the 'chronos' optional dependency"
            ) from error
        try:
            return chronos.BaseChronosPipeline.from_pretrained(
                checkpoint,
                device_map=device_map,
            )
        except Exception as error:
            raise RuntimeUnavailableError(
                f"Could not load Chronos checkpoint {checkpoint!r}: {error}"
            ) from error

    @staticmethod
    def _validated_history(history: Sequence[float]) -> list[float]:
        if isinstance(history, (str, bytes)):
            raise ValueError("history must be one-dimensional")
        if len(history) == 0:
            raise ValueError("history must not be empty")
        try:
            values = [float(value) for value in history]
        except (TypeError, ValueError) as error:
            raise ValueError("history must be one-dimensional") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError("history must contain only finite values")
        return values

    @staticmethod
    def _validated_p50(quantiles: object, horizon: int) -> tuple[float, ...]:
        # T5/Bolt return [batch, horizon, quantile]. Chronos-2 returns a list
        # whose tensors are [variates, horizon, quantile]. This runtime sends
        # one univariate series and requests only the documented 0.5 quantile.
        tensor_output = hasattr(quantiles, "tolist")
        if tensor_output:
            quantiles = quantiles.tolist()  # type: ignore[union-attr]
        try:
            if len(quantiles) != 1:  # type: ignore[arg-type]
                raise RuntimeError("Chronos p50 forecast has the wrong batch shape")
            row = quantiles[0]  # type: ignore[index]
        except (TypeError, IndexError, KeyError) as error:
            raise RuntimeError("Chronos p50 forecast has the wrong batch shape") from error
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not tensor_output:
            try:
                if len(row) != 1:
                    raise RuntimeError(
                        "Chronos p50 forecast has the wrong variate shape"
                    )
                row = row[0]
            except (TypeError, IndexError, KeyError) as error:
                raise RuntimeError(
                    "Chronos p50 forecast has the wrong variate shape"
                ) from error
        try:
            if len(row) != horizon:
                raise RuntimeError(
                    "Chronos p50 forecast has the wrong horizon length"
                )
            values = []
            for step in row:
                if hasattr(step, "tolist"):
                    step = step.tolist()
                if len(step) != 1:
                    raise RuntimeError(
                        "Chronos p50 forecast has the wrong quantile shape"
                    )
                values.append(float(step[0]))
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("Chronos p50 forecast must contain scalar values") from error
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Chronos p50 forecast contains non-finite values")
        return tuple(values)
