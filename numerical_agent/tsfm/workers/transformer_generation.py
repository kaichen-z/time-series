"""Manifest-bound adapters for official Transformer-style TSFM generation APIs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..manifests import ManifestRegistry, TSFMManifest
from ..protocol import WorkerRequest
from .common import (
    CheckpointUnavailableError,
    DependencyUnavailableError,
    ModelOutputError,
    RequestUnavailableError,
    checkpoint_cache_key,
    loaded_checkpoint_revision,
    record_loaded_checkpoint,
)


_TIMER_ID = "method_tsfm_0007"
_TIME_MOE_ID = "method_tsfm_0008"
_SUNDIAL_ID = "method_tsfm_0013"
_TIMER_S1_ID = "method_tsfm_0015"
_KAIROS_ID = "method_tsfm_0022"

# These are code-review boundaries, not candidate or catalog-controlled values. Every
# listed checkpoint executes remote model code and must therefore remain an exact ID.
_AUDITED_BINDINGS = MappingProxyType(
    {
        _TIMER_ID: ("thuml/timer-base-84m", "timer_legacy", "direct"),
        _TIME_MOE_ID: (
            "Maple728/TimeMoE-200M",
            "transformers_recent",
            "direct",
        ),
        _SUNDIAL_ID: ("thuml/sundial-base-128m", "timer_legacy", "mean"),
        _TIMER_S1_ID: ("thuml/Timer-S1", "transformers_recent", "median"),
        _KAIROS_ID: ("mldi-lab/Kairos_50m", "kairos", "median"),
    }
)
_QUANTILE_COUNT = 9
_P50_INDEX = 4


@dataclass(frozen=True)
class _OfficialBackend:
    torch: Any
    causal_lm_class: Any

    def tensor(
        self,
        values: tuple[float, ...],
        *,
        layout: str,
        device: object = None,
    ) -> Any:
        if layout != "batch_time":
            raise ValueError(f"unsupported generation tensor layout {layout!r}")
        return self.torch.tensor(
            values,
            dtype=self.torch.float32,
            device=device,
        ).unsqueeze(0)

    def normalize_time_moe(self, tensor: Any) -> tuple[Any, Any, Any]:
        # The official Time-MoE quickstart uses torch.std's sample correction.
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        return (tensor - mean) / std, mean, std

    def inverse_scale(self, values: object, mean: Any, std: Any) -> Any:
        tensor = self.torch.tensor(
            values,
            dtype=mean.dtype,
            device=mean.device,
        )
        return tensor * std + mean

    def no_grad(self) -> Any:
        return self.torch.no_grad()

    @staticmethod
    def require_kairos_class() -> Any:
        # Kairos is installed only in the dedicated Kairos environment. Keeping
        # this import behind its manifest branch lets Timer-only deployments omit it.
        from tsfm.model.kairos import AutoModel

        return AutoModel


def _load_official_backend() -> _OfficialBackend:
    # Torch and Transformers remain isolated-worker-only dependencies.
    import torch
    from transformers import AutoModelForCausalLM

    return _OfficialBackend(
        torch=torch,
        causal_lm_class=AutoModelForCausalLM,
    )


class TransformerGenerationAdapter:
    """Forecast with five exact, reviewed Transformer-generation bindings."""

    def __init__(self, loader: Callable[[], Any] | None = None) -> None:
        self._loader = loader or _load_official_backend
        self._backend: Any | None = None
        self._models: dict[tuple[str, str], Any] = {}
        self._loaded_checkpoint_revisions: set[tuple[str, str]] = set()
        registry = ManifestRegistry.load_default()
        bindings: dict[str, TSFMManifest] = {}
        for method_id, (checkpoint, environment, reduction) in _AUDITED_BINDINGS.items():
            manifest = registry.require(method_id)
            if (
                manifest.checkpoint != checkpoint
                or manifest.worker_environment != environment
                or manifest.adapter != "transformer_generation"
                or manifest.point_reduction != reduction
                or manifest.runtime_options.get("trust_remote_code") is not True
            ):
                raise RuntimeError(
                    f"audited Transformer-generation manifest {method_id!r} changed"
                )
            bindings[checkpoint] = manifest
        self._bindings: Mapping[str, TSFMManifest] = MappingProxyType(bindings)

    def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
        manifest = self._require_binding(request)
        self._validate_limits(request, manifest)
        backend = self._require_backend()
        model = self._require_model(manifest, backend, request)

        device = getattr(model, "device", None) if manifest.method_id == _TIMER_S1_ID else None
        tensor = backend.tensor(
            request.history,
            layout="batch_time",
            device=device,
        )

        try:
            with backend.no_grad():
                if manifest.method_id == _SUNDIAL_ID:
                    output = model.generate(
                        tensor,
                        max_new_tokens=request.horizon,
                        num_samples=int(manifest.runtime_options["num_samples"]),
                    )
                elif manifest.method_id == _TIMER_S1_ID:
                    output = model.generate(
                        tensor,
                        max_new_tokens=request.horizon,
                        revin=True,
                    )
                elif manifest.method_id == _KAIROS_ID:
                    output = model(
                        past_target=tensor,
                        prediction_length=request.horizon,
                        generation=True,
                        infer_is_positive=True,
                        force_flip_invariance=True,
                    )
                elif manifest.method_id == _TIME_MOE_ID:
                    normalized, mean, std = backend.normalize_time_moe(tensor)
                    generated = model.generate(
                        normalized,
                        max_new_tokens=request.horizon,
                    )
                    continuation = _batch_continuation(
                        generated,
                        request.horizon,
                        output_name="Time-MoE generation output",
                    )
                    output = backend.inverse_scale(continuation, mean, std)
                else:
                    output = model.generate(
                        tensor,
                        max_new_tokens=request.horizon,
                    )
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Transformer-generation dependencies are unavailable: {error}"
            ) from error

        if manifest.method_id == _SUNDIAL_ID:
            samples = _batch_matrix(
                output,
                rows=int(manifest.runtime_options["num_samples"]),
                horizon=request.horizon,
                output_name="Sundial generation output",
                row_name="samples",
            )
            return tuple(
                _finite_number(
                    math.fsum(sample[step] for sample in samples) / len(samples),
                    "Sundial sample mean",
                )
                for step in range(request.horizon)
            )

        if manifest.method_id == _TIMER_S1_ID:
            quantiles = _batch_matrix(
                output,
                rows=_QUANTILE_COUNT,
                horizon=request.horizon,
                output_name="Timer-S1 generation output",
                row_name="quantiles",
            )
            return quantiles[_P50_INDEX]

        if manifest.method_id == _KAIROS_ID:
            quantiles = _batch_matrix(
                _prediction_outputs(output),
                rows=_QUANTILE_COUNT,
                horizon=request.horizon,
                output_name="Kairos prediction output",
                row_name="quantiles",
            )
            return tuple(
                _finite_number(
                    statistics.median(row[step] for row in quantiles),
                    "Kairos median quantile",
                )
                for step in range(request.horizon)
            )

        return _batch_vector(
            output,
            request.horizon,
            output_name=(
                "Time-MoE inverse-scaled output"
                if manifest.method_id == _TIME_MOE_ID
                else "Timer generation output"
            ),
        )

    def _require_binding(self, request: WorkerRequest) -> TSFMManifest:
        manifest = self._bindings.get(request.checkpoint)
        if (
            manifest is None
            or request.provider != "transformer_generation"
            or dict(request.runtime_options) != dict(manifest.runtime_options)
        ):
            raise RequestUnavailableError(
                "request does not match a reviewed Transformer-generation manifest binding"
            )
        return manifest

    @staticmethod
    def _validate_limits(request: WorkerRequest, manifest: TSFMManifest) -> None:
        if manifest.method_id == _TIMER_ID and len(request.history) < 96:
            raise RequestUnavailableError(
                "Timer context must contain at least 96 values (one input patch)"
            )
        if manifest.method_id == _TIME_MOE_ID and (
            len(request.history) < 2
            or min(request.history) == max(request.history)
        ):
            raise RequestUnavailableError(
                "Time-MoE history must have a finite nonzero sample standard deviation"
            )
        if "max_context" in manifest.runtime_options:
            limit = int(manifest.runtime_options["max_context"])
            if len(request.history) > limit:
                raise RequestUnavailableError(
                    f"{manifest.checkpoint} context must not exceed {limit}"
                )
        if "max_total_length" in manifest.runtime_options:
            limit = int(manifest.runtime_options["max_total_length"])
            if len(request.history) + request.horizon > limit:
                raise RequestUnavailableError(
                    f"{manifest.checkpoint} context plus horizon must not exceed {limit}"
                )

    def _require_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._loader()
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Transformer-generation dependencies are unavailable: {error}"
            ) from error
        if backend is None:
            raise DependencyUnavailableError(
                "Transformer-generation dependencies are unavailable"
            )
        self._backend = backend
        return backend

    def _require_model(
        self, manifest: TSFMManifest, backend: Any, request: WorkerRequest
    ) -> Any:
        key = checkpoint_cache_key(request)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        if manifest.checkpoint not in {
            checkpoint for checkpoint, _, _ in _AUDITED_BINDINGS.values()
        }:
            raise RequestUnavailableError(
                "request does not match a reviewed remote-code checkpoint"
            )
        try:
            if manifest.method_id == _KAIROS_ID:
                load_options: dict[str, object] = {"trust_remote_code": True}
                if request.checkpoint_revision:
                    load_options["revision"] = request.checkpoint_revision
                model = backend.require_kairos_class().from_pretrained(
                    manifest.checkpoint,
                    **load_options,
                )
            else:
                load_options: dict[str, object] = {"trust_remote_code": True}
                if request.checkpoint_revision:
                    load_options["revision"] = request.checkpoint_revision
                if manifest.method_id == _TIME_MOE_ID:
                    load_options["device_map"] = "cpu"
                elif manifest.method_id == _TIMER_S1_ID:
                    load_options["device_map"] = "auto"
                model = backend.causal_lm_class.from_pretrained(
                    manifest.checkpoint,
                    **load_options,
                )
            model.eval()
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Transformer-generation dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Transformer-generation checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._models[key] = model
        return model

    def loaded_checkpoint_revision(self, request: WorkerRequest) -> str:
        return loaded_checkpoint_revision(self._loaded_checkpoint_revisions, request)


def _prediction_outputs(output: object) -> object:
    if isinstance(output, Mapping):
        return output.get("prediction_outputs")
    return getattr(output, "prediction_outputs", None)


def _materialize(value: Any) -> object:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _materialize(tolist())
    if isinstance(value, (list, tuple)):
        return [_materialize(item) for item in value]
    return value


def _finite_number(value: object, output_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelOutputError(f"{output_name} must contain finite numbers")
    try:
        number = float(value)
    except OverflowError as error:
        raise ModelOutputError(f"{output_name} must contain finite numbers") from error
    if not math.isfinite(number):
        raise ModelOutputError(f"{output_name} must contain only finite values")
    return number


def _batch_continuation(
    value: object,
    horizon: int,
    *,
    output_name: str,
) -> list[list[float]]:
    materialized = _materialize(value)
    if (
        not isinstance(materialized, list)
        or len(materialized) != 1
        or not isinstance(materialized[0], list)
        or any(isinstance(item, list) for item in materialized[0])
    ):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    row = materialized[0]
    if len(row) < horizon:
        raise ModelOutputError(f"{output_name} has the wrong horizon length")
    return [[_finite_number(item, output_name) for item in row[-horizon:]]]


def _batch_vector(
    value: object,
    horizon: int,
    *,
    output_name: str,
) -> tuple[float, ...]:
    materialized = _materialize(value)
    if (
        not isinstance(materialized, list)
        or len(materialized) != 1
        or not isinstance(materialized[0], list)
        or any(isinstance(item, list) for item in materialized[0])
    ):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    row = materialized[0]
    if len(row) != horizon:
        raise ModelOutputError(f"{output_name} has the wrong horizon length")
    return tuple(_finite_number(item, output_name) for item in row)


def _batch_matrix(
    value: object,
    *,
    rows: int,
    horizon: int,
    output_name: str,
    row_name: str,
) -> tuple[tuple[float, ...], ...]:
    materialized = _materialize(value)
    if (
        not isinstance(materialized, list)
        or len(materialized) != 1
        or not isinstance(materialized[0], list)
    ):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    matrix = materialized[0]
    if len(matrix) != rows:
        raise ModelOutputError(f"{output_name} must contain exactly {rows} {row_name}")
    converted: list[tuple[float, ...]] = []
    for row in matrix:
        if (
            not isinstance(row, list)
            or len(row) != horizon
            or any(isinstance(item, list) for item in row)
        ):
            raise ModelOutputError(f"{output_name} has the wrong horizon length")
        converted.append(tuple(_finite_number(item, output_name) for item in row))
    return tuple(converted)
