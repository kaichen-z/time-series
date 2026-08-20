"""Manifest-bound worker adapter for IBM Granite time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..manifests import ManifestRegistry, TSFMManifest
from ..protocol import WorkerRequest
from .common import (
    CheckpointUnavailableError,
    DependencyUnavailableError,
    RequestUnavailableError,
    checkpoint_cache_key,
    finite_single_series,
    loaded_checkpoint_revision,
    record_loaded_checkpoint,
)


_TTM_ID = "method_tsfm_0006"
_FLOWSTATE_ID = "method_tsfm_0020"
_PATCHTST_ID = "method_tsfm_0030"
_AUDITED_METHOD_IDS = frozenset({_TTM_ID, _FLOWSTATE_ID, _PATCHTST_ID})
_FLOWSTATE_SCALES = MappingProxyType(
    {
        "15min": 0.25,
        "15t": 0.25,
        "30min": 0.5,
        "30t": 0.5,
        "h": 1.0,
        "1h": 1.0,
        "1 hour": 1.0,
        "hour": 1.0,
        "hourly": 1.0,
        "w": 0.46,
        "1w": 0.46,
        "1 week": 0.46,
        "week": 0.46,
        "weekly": 0.46,
        "m": 2.0,
        "ms": 2.0,
        "me": 2.0,
        "1 month": 2.0,
        "month": 2.0,
        "monthly": 2.0,
    }
)
_TTM_FREQUENCIES = MappingProxyType(
    {
        "h": ("h", "hourly"),
        "1h": ("h", "hourly"),
        "1 hour": ("h", "hourly"),
        "hour": ("h", "hourly"),
        "hourly": ("h", "hourly"),
        "d": ("d", "daily"),
        "1d": ("d", "daily"),
        "1 day": ("d", "daily"),
        "day": ("d", "daily"),
        "daily": ("d", "daily"),
        "w": ("W", "weekly"),
        "1w": ("W", "weekly"),
        "1 week": ("W", "weekly"),
        "week": ("W", "weekly"),
        "weekly": ("W", "weekly"),
    }
)


@dataclass(frozen=True)
class _OfficialBackend:
    torch: Any
    get_model: Callable[..., Any]
    flowstate_class: Any
    patchtst_class: Any

    def tensor(self, values: tuple[float, ...], *, layout: str) -> Any:
        tensor = self.torch.tensor(values, dtype=self.torch.float32)
        if layout == "batch_time_channel":
            return tensor.reshape(1, -1, 1)
        if layout == "time_batch_channel":
            return tensor.reshape(-1, 1, 1)
        if layout == "vector":
            return tensor
        raise ValueError(f"unsupported Granite tensor layout {layout!r}")

    def no_grad(self) -> Any:
        return self.torch.no_grad()

    def observed_mask(self, tensor: Any) -> Any:
        return self.torch.ones_like(tensor, dtype=self.torch.bool)

    def frequency_token(self, value: int) -> Any:
        return self.torch.tensor([value], dtype=self.torch.long)


def _load_official_backend() -> _OfficialBackend:
    # These imports belong exclusively to the isolated worker loader path.
    import torch
    from tsfm_public import FlowStateForPrediction, PatchTSTFMForPrediction
    from tsfm_public.toolkit.get_model import get_model

    return _OfficialBackend(
        torch=torch,
        get_model=get_model,
        flowstate_class=FlowStateForPrediction,
        patchtst_class=PatchTSTFMForPrediction,
    )


class GraniteAdapter:
    """Forecast with the three exact Granite-family manifest bindings."""

    def __init__(self, loader: Callable[[], Any] | None = None) -> None:
        self._loader = loader or _load_official_backend
        self._backend: Any | None = None
        self._models: dict[tuple[str, str], Any] = {}
        self._loaded_checkpoint_revisions: set[tuple[str, str]] = set()
        registry = ManifestRegistry.load_default()
        bindings = {
            manifest.checkpoint: manifest
            for method_id, manifest in registry.items()
            if method_id in _AUDITED_METHOD_IDS
        }
        if {manifest.method_id for manifest in bindings.values()} != _AUDITED_METHOD_IDS:
            raise RuntimeError("audited Granite manifests are incomplete")
        self._bindings: Mapping[str, TSFMManifest] = MappingProxyType(bindings)

    def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
        manifest = self._require_binding(request)
        if manifest.method_id == _PATCHTST_ID:
            limit = int(manifest.runtime_options["context_plus_horizon_limit"])
            if len(request.history) + request.horizon > limit:
                raise RequestUnavailableError(
                    f"PatchTST-FM context plus horizon must not exceed {limit}"
                )
        if manifest.method_id == _TTM_ID:
            native_horizon = int(manifest.runtime_options["prediction_length"])
            if request.horizon > native_horizon:
                raise RequestUnavailableError(
                    f"TTM horizon must not exceed {native_horizon}"
                )
        flowstate_scale = None
        ttm_frequency = None
        if manifest.method_id == _FLOWSTATE_ID:
            flowstate_scale = _FLOWSTATE_SCALES.get(
                request.frequency.strip().lower()
            )
            if flowstate_scale is None:
                raise RequestUnavailableError(
                    "FlowState frequency has no unambiguous reviewed scale factor"
                )
        elif manifest.method_id == _TTM_ID:
            ttm_frequency = _TTM_FREQUENCIES.get(
                request.frequency.strip().lower()
            )
            if ttm_frequency is None:
                raise RequestUnavailableError(
                    "TTM R2.1 frequency must be hourly, daily, or weekly"
                )

        backend = self._require_backend()
        model = self._require_model(manifest, backend, ttm_frequency, request)
        if manifest.method_id == _TTM_ID:
            if ttm_frequency is None:
                raise RuntimeError("TTM frequency binding was not resolved")
            _, token_category = ttm_frequency
            frequency_token = backend.frequency_token(
                int(manifest.runtime_options[f"frequency_token_{token_category}"])
            )
            context_length = int(manifest.runtime_options["context_length"])
            tensor = backend.tensor(
                request.history[-context_length:], layout="batch_time_channel"
            )
            observed_mask = backend.observed_mask(tensor)
            with backend.no_grad():
                output = model(
                    past_values=tensor,
                    past_observed_mask=observed_mask,
                    freq_token=frequency_token,
                )
            return finite_single_series(
                getattr(output, "prediction_outputs", None),
                request.horizon,
                leading_singletons=1,
                output_name="TTM prediction output",
                allow_longer=True,
            )

        if manifest.method_id == _FLOWSTATE_ID:
            max_context = int(manifest.runtime_options["max_context"])
            raw_context_limit = min(
                16_384,
                int(max_context / flowstate_scale),
            )
            tensor = backend.tensor(
                request.history[-raw_context_limit:], layout="time_batch_channel"
            )
            with backend.no_grad():
                output = model(
                    past_values=tensor,
                    scale_factor=flowstate_scale,
                    prediction_length=request.horizon,
                    batch_first=False,
                    prediction_type="median",
                )
            return finite_single_series(
                getattr(output, "prediction_outputs", None),
                request.horizon,
                leading_singletons=1,
                output_name="FlowState median output",
            )

        tensor = backend.tensor(request.history, layout="batch_time_channel")
        with backend.no_grad():
            output = model(
                past_values=tensor,
                prediction_length=request.horizon,
                quantile_levels=[0.5],
            )
        return finite_single_series(
            getattr(output, "quantile_outputs", None),
            request.horizon,
            leading_singletons=2,
            output_name="PatchTST-FM median output",
        )

    def _require_binding(self, request: WorkerRequest) -> TSFMManifest:
        manifest = self._bindings.get(request.checkpoint)
        if (
            manifest is None
            or request.provider != "granite"
            or dict(request.runtime_options) != dict(manifest.runtime_options)
        ):
            raise RequestUnavailableError(
                "request does not match a reviewed Granite manifest binding"
            )
        return manifest

    def _require_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._loader()
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Granite TSFM dependencies are unavailable: {error}"
            ) from error
        if backend is None:
            raise DependencyUnavailableError("Granite TSFM dependencies are unavailable")
        self._backend = backend
        return backend

    def _require_model(
        self,
        manifest: TSFMManifest,
        backend: Any,
        ttm_frequency: tuple[str, str] | None,
        request: WorkerRequest,
    ) -> Any:
        key = checkpoint_cache_key(request)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        try:
            if manifest.method_id == _TTM_ID:
                if ttm_frequency is None:
                    raise RuntimeError("TTM frequency binding was not resolved")
                normalized_frequency, _ = ttm_frequency
                model = backend.get_model(
                    model_path=manifest.checkpoint,
                    context_length=int(manifest.runtime_options["context_length"]),
                    prediction_length=int(
                        manifest.runtime_options["prediction_length"]
                    ),
                    model_revision=(
                        request.checkpoint_revision
                        or str(manifest.runtime_options["model_revision"])
                    ),
                    freq_prefix_tuning=True,
                    freq=normalized_frequency,
                )
            elif manifest.method_id == _FLOWSTATE_ID:
                model = backend.flowstate_class.from_pretrained(
                    manifest.checkpoint,
                    revision=(
                        request.checkpoint_revision
                        or manifest.runtime_options["revision"]
                    ),
                )
            else:
                model = backend.patchtst_class.from_pretrained(
                    manifest.checkpoint,
                    revision=(
                        request.checkpoint_revision
                        or manifest.runtime_options["revision"]
                    ),
                )
            model.eval()
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Granite TSFM dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Granite checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._models[key] = model
        return model

    def loaded_checkpoint_revision(self, request: WorkerRequest) -> str:
        return loaded_checkpoint_revision(self._loaded_checkpoint_revisions, request)
