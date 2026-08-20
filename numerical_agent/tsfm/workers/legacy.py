"""Manifest-bound adapters for archived TimesFM, Lag-Llama, and TEMPO."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import re
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


_TIMESFM_ID = "method_tsfm_0001"
_LAG_LLAMA_ID = "method_tsfm_0004"
_TEMPO_ID = "method_tsfm_0011"
_AUDITED_METHOD_IDS = frozenset({_TIMESFM_ID, _LAG_LLAMA_ID, _TEMPO_ID})
_CASE_SENSITIVE_MILLISECOND_FREQUENCY = re.compile(
    r"(?P<count>[1-9][0-9]*)?(?P<unit>[mM][sS])"
)
_MILLISECOND_LIKE_FREQUENCY = re.compile(r"[0-9]*[mM][sS]")
_LEGACY_BASE_FREQUENCIES = MappingProxyType(
    {
        "s": "S",
        "second": "S",
        "seconds": "S",
        "t": "T",
        "min": "T",
        "minute": "T",
        "minutes": "T",
        "h": "H",
        "hour": "H",
        "hours": "H",
        "hourly": "H",
        "d": "D",
        "day": "D",
        "days": "D",
        "daily": "D",
        "b": "B",
        "business day": "B",
        "business days": "B",
        "w": "W",
        "week": "W",
        "weeks": "W",
        "weekly": "W",
        "m": "M",
        "me": "M",
        "month": "M",
        "months": "M",
        "monthly": "M",
        "q": "Q",
        "qe": "Q",
        "quarter": "Q",
        "quarters": "Q",
        "quarterly": "Q",
        "y": "Y",
        "a": "Y",
        "ye": "Y",
        "year": "Y",
        "years": "Y",
        "yearly": "Y",
    }
)


@dataclass(frozen=True)
class _TimesFMBackend:
    model_class: Any
    hparams_class: Any
    checkpoint_class: Any
    frequency_map: Callable[[str], int]
    hf_hub_download: Callable[..., str] | None = None

    def frequency_class(self, frequency: str) -> int:
        if frequency.endswith("ms") and frequency[:-2].isdigit():
            return 0
        if frequency == "ms":
            return 0
        return self.frequency_map(frequency)

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str
    ) -> str:
        if self.hf_hub_download is None:
            raise ImportError("huggingface_hub is unavailable")
        return self.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )


@dataclass(frozen=True)
class _LagBackend:
    torch: Any
    pandas: Any
    pandas_dataset: Any
    estimator_class: Any
    hf_hub_download: Callable[..., str]
    device: Any

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str = ""
    ) -> str:
        options = {"revision": revision} if revision else {}
        return self.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            **options,
        )

    def load_checkpoint(self, path: str) -> object:
        return self.torch.load(path, map_location=self.device)

    def dataset(self, history: tuple[float, ...], frequency: str) -> object:
        timestamps = self.pandas.date_range(
            start="2000-01-01", periods=len(history), freq=frequency
        )
        periods = timestamps.to_period(
            _period_compatible_legacy_frequency(frequency)
        )
        target = self.pandas.Series(history, index=periods, name="target")
        return self.pandas_dataset({"target": target}, freq=periods.freqstr)


@dataclass(frozen=True)
class _TempoBackend:
    torch: Any
    model_class: Any
    device: Any
    omega_conf: Any
    hf_hub_download: Callable[..., str]

    def no_grad(self) -> Any:
        return self.torch.no_grad()

    def load_exact_model(
        self, repo_id: str, filename: str, revision: str
    ) -> Any:
        checkpoint_path = self.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
        config_path = self.hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )
        configuration = self.omega_conf.load(config_path)
        model = self.model_class(configuration, self.device)
        state_dict = self.torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        return model


def _load_official_backend(method_id: str) -> object:
    """Import only the dependencies present in the selected legacy environment."""

    if method_id == _TIMESFM_ID:
        from huggingface_hub import hf_hub_download
        from timesfm import TimesFmCheckpoint, TimesFmHparams, freq_map
        from timesfm.timesfm_torch import TimesFmTorch

        return _TimesFMBackend(
            model_class=TimesFmTorch,
            hparams_class=TimesFmHparams,
            checkpoint_class=TimesFmCheckpoint,
            frequency_map=freq_map,
            hf_hub_download=hf_hub_download,
        )
    if method_id == _LAG_LLAMA_ID:
        import pandas
        import torch
        from gluonts.dataset.pandas import PandasDataset
        from huggingface_hub import hf_hub_download
        from lag_llama.gluon.estimator import LagLlamaEstimator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _LagBackend(
            torch=torch,
            pandas=pandas,
            pandas_dataset=PandasDataset,
            estimator_class=LagLlamaEstimator,
            hf_hub_download=hf_hub_download,
            device=device,
        )
    if method_id == _TEMPO_ID:
        import torch
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
        from tempo.models.TEMPO import TEMPO

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return _TempoBackend(
            torch=torch,
            model_class=TEMPO,
            device=device,
            omega_conf=OmegaConf,
            hf_hub_download=hf_hub_download,
        )
    raise ValueError(f"unsupported legacy method {method_id!r}")


class LegacyAdapter:
    """Forecast with the three exact archived manifest bindings."""

    def __init__(self, loader: Callable[..., Any] | None = None) -> None:
        self._loader = loader
        self._backends: dict[str, Any] = {}
        self._models: dict[tuple[str, str], Any] = {}
        self._lag_checkpoints: dict[
            tuple[str, str], tuple[str, Mapping[str, object]]
        ] = {}
        self._lag_predictors: dict[tuple[str, str, int, int, int], Any] = {}
        self._loaded_checkpoint_revisions: set[tuple[str, str]] = set()
        registry = ManifestRegistry.load_default()
        bindings = {
            manifest.checkpoint: manifest
            for method_id, manifest in registry.items()
            if method_id in _AUDITED_METHOD_IDS
        }
        if {manifest.method_id for manifest in bindings.values()} != _AUDITED_METHOD_IDS:
            raise RuntimeError("audited legacy manifests are incomplete")
        self._bindings: Mapping[str, TSFMManifest] = MappingProxyType(bindings)

    def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
        manifest = self._require_binding(request)
        if manifest.method_id == _TIMESFM_ID:
            return self._forecast_timesfm(manifest, request)
        if manifest.method_id == _LAG_LLAMA_ID:
            return self._forecast_lag_llama(manifest, request)
        return self._forecast_tempo(manifest, request)

    def _forecast_timesfm(
        self, manifest: TSFMManifest, request: WorkerRequest
    ) -> tuple[float, ...]:
        max_horizon = int(manifest.runtime_options["recommended_max_horizon"])
        if request.horizon > max_horizon:
            raise RequestUnavailableError(
                f"TimesFM 1.0 horizon must not exceed {max_horizon}"
            )
        frequency = _normalize_legacy_frequency(request.frequency)
        backend = self._require_backend(manifest)
        try:
            frequency_class = backend.frequency_class(frequency)
        except ValueError as error:
            raise RequestUnavailableError(
                "TimesFM 1.0 frequency has no reviewed frequency class"
            ) from error
        model = self._require_model(manifest, backend, request)
        max_context = int(manifest.runtime_options["max_context"])
        output = model.forecast(
            [request.history[-max_context:]],
            freq=[frequency_class],
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise ModelOutputError("TimesFM 1.0 forecast output has an invalid shape")
        return _batch_prefix(
            output[0], request.horizon, output_name="TimesFM 1.0 point output"
        )

    def _forecast_lag_llama(
        self, manifest: TSFMManifest, request: WorkerRequest
    ) -> tuple[float, ...]:
        frequency = _normalize_legacy_frequency(request.frequency)
        backend = self._require_backend(manifest)
        max_context = int(manifest.runtime_options["max_context"])
        context = request.history[-max_context:]
        num_samples = int(manifest.runtime_options["num_samples"])
        predictor = self._require_lag_predictor(
            manifest,
            backend,
            request,
            context_length=len(context),
            horizon=request.horizon,
            num_samples=num_samples,
        )
        dataset = backend.dataset(context, _pandas_legacy_frequency(frequency))
        forecasts = predictor.predict(dataset, num_samples=num_samples)
        forecast = _single_forecast(forecasts)
        quantile = getattr(forecast, "quantile", None)
        if not callable(quantile):
            raise ModelOutputError("Lag-Llama forecast has no quantile output")
        return _finite_vector(
            quantile(0.5),
            request.horizon,
            output_name="Lag-Llama median output",
        )

    def _forecast_tempo(
        self, manifest: TSFMManifest, request: WorkerRequest
    ) -> tuple[float, ...]:
        backend = self._require_backend(manifest)
        model = self._require_model(manifest, backend, request)
        context_length = int(manifest.runtime_options["context_length"])
        with backend.no_grad():
            output = model.predict(
                request.history[-context_length:], pred_length=request.horizon
            )
        return _finite_vector(
            output, request.horizon, output_name="TEMPO point output"
        )

    def _require_binding(self, request: WorkerRequest) -> TSFMManifest:
        manifest = self._bindings.get(request.checkpoint)
        if (
            manifest is None
            or request.provider != "legacy"
            or dict(request.runtime_options) != dict(manifest.runtime_options)
        ):
            raise RequestUnavailableError(
                "request does not match a reviewed legacy manifest binding"
            )
        return manifest

    def _require_backend(self, manifest: TSFMManifest) -> Any:
        cached = self._backends.get(manifest.method_id)
        if cached is not None:
            return cached
        try:
            backend = (
                _load_official_backend(manifest.method_id)
                if self._loader is None
                else _call_injected_loader(self._loader, manifest.method_id)
            )
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Legacy dependencies are unavailable: {error}"
            ) from error
        if backend is None:
            raise DependencyUnavailableError("Legacy dependencies are unavailable")
        self._backends[manifest.method_id] = backend
        return backend

    def _require_model(
        self, manifest: TSFMManifest, backend: Any, request: WorkerRequest
    ) -> Any:
        key = checkpoint_cache_key(request)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        try:
            if manifest.method_id == _TIMESFM_ID:
                max_context = int(manifest.runtime_options["max_context"])
                max_horizon = int(
                    manifest.runtime_options["recommended_max_horizon"]
                )
                hparams = backend.hparams_class(
                    context_len=max_context,
                    horizon_len=max_horizon,
                    input_patch_len=32,
                    output_patch_len=128,
                    per_core_batch_size=1,
                    backend="cpu",
                )
                if request.checkpoint_revision:
                    checkpoint_path = backend.download_checkpoint(
                        manifest.checkpoint,
                        "torch_model.ckpt",
                        revision=request.checkpoint_revision,
                    )
                    checkpoint = backend.checkpoint_class(
                        version="torch",
                        path=checkpoint_path,
                    )
                else:
                    checkpoint = backend.checkpoint_class(
                        version="torch",
                        huggingface_repo_id=manifest.checkpoint,
                    )
                model = backend.model_class(hparams=hparams, checkpoint=checkpoint)
            elif manifest.method_id == _TEMPO_ID:
                if request.checkpoint_revision:
                    model = backend.load_exact_model(
                        manifest.checkpoint,
                        str(manifest.runtime_options["checkpoint_file"]),
                        request.checkpoint_revision,
                    )
                else:
                    model = backend.model_class.load_pretrained_model(
                        device=backend.device,
                        repo_id=manifest.checkpoint,
                        filename=manifest.runtime_options["checkpoint_file"],
                    )
            else:
                raise RuntimeError("Lag-Llama models use a predictor cache")
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Legacy dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Legacy checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._models[key] = model
        return model

    def _require_lag_checkpoint(
        self, manifest: TSFMManifest, backend: Any, request: WorkerRequest
    ) -> tuple[str, Mapping[str, object]]:
        key = checkpoint_cache_key(request)
        cached = self._lag_checkpoints.get(key)
        if cached is not None:
            return cached
        try:
            path = backend.download_checkpoint(
                manifest.checkpoint,
                str(manifest.runtime_options["checkpoint_file"]),
                **(
                    {"revision": request.checkpoint_revision}
                    if request.checkpoint_revision
                    else {}
                ),
            )
            checkpoint = backend.load_checkpoint(path)
            if not isinstance(checkpoint, Mapping):
                raise TypeError("checkpoint payload is not a mapping")
            hyperparameters = checkpoint["hyper_parameters"]
            if not isinstance(hyperparameters, Mapping):
                raise TypeError("checkpoint hyperparameters are not a mapping")
            model_kwargs = hyperparameters["model_kwargs"]
            if not isinstance(model_kwargs, Mapping):
                raise TypeError("checkpoint model_kwargs are not a mapping")
            required = {
                "input_size",
                "context_length",
                "n_layer",
                "n_embd_per_head",
                "n_head",
                "scaling",
                "time_feat",
            }
            if not required <= set(model_kwargs):
                raise KeyError("checkpoint model_kwargs are incomplete")
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Lag-Llama dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Lag-Llama checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        result = (path, model_kwargs)
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._lag_checkpoints[key] = result
        return result

    def _require_lag_predictor(
        self,
        manifest: TSFMManifest,
        backend: Any,
        request: WorkerRequest,
        *,
        context_length: int,
        horizon: int,
        num_samples: int,
    ) -> Any:
        key = (
            manifest.checkpoint,
            request.checkpoint_revision,
            context_length,
            horizon,
            num_samples,
        )
        cached = self._lag_predictors.get(key)
        if cached is not None:
            return cached
        checkpoint_path, model_kwargs = self._require_lag_checkpoint(
            manifest, backend, request
        )
        trained_context = int(model_kwargs["context_length"])
        rope_factor = max(1.0, (context_length + horizon) / trained_context)
        try:
            estimator = backend.estimator_class(
                ckpt_path=checkpoint_path,
                prediction_length=horizon,
                context_length=context_length,
                device=backend.device,
                input_size=model_kwargs["input_size"],
                n_layer=model_kwargs["n_layer"],
                n_embd_per_head=model_kwargs["n_embd_per_head"],
                n_head=model_kwargs["n_head"],
                scaling=model_kwargs["scaling"],
                time_feat=model_kwargs["time_feat"],
                rope_scaling={"type": "linear", "factor": rope_factor},
                batch_size=1,
                num_parallel_samples=num_samples,
            )
            module = estimator.create_lightning_module().to(backend.device)
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, module)
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Lag-Llama dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Lag-Llama checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        self._lag_predictors[key] = predictor
        return predictor

    def loaded_checkpoint_revision(self, request: WorkerRequest) -> str:
        return loaded_checkpoint_revision(self._loaded_checkpoint_revisions, request)


def _call_injected_loader(loader: Callable[..., Any], method_id: str) -> Any:
    try:
        inspect.signature(loader).bind(method_id)
    except (TypeError, ValueError):
        return loader()
    return loader(method_id)


def _normalize_legacy_frequency(value: str) -> str:
    stripped = " ".join(value.strip().split())
    millisecond_match = _CASE_SENSITIVE_MILLISECOND_FREQUENCY.fullmatch(stripped)
    if millisecond_match is not None:
        unit = millisecond_match.group("unit")
        if unit not in {"ms", "MS"}:
            raise RequestUnavailableError(
                "legacy model frequency has no reviewed conversion"
            )
        count = millisecond_match.group("count") or ""
        return f"{count}{unit}"
    if _MILLISECOND_LIKE_FREQUENCY.fullmatch(stripped):
        raise RequestUnavailableError(
            "legacy model frequency has no reviewed conversion"
        )
    compact = stripped.lower()
    direct = _LEGACY_BASE_FREQUENCIES.get(compact)
    if direct is not None:
        return direct
    pieces = compact.split(" ", 1)
    if len(pieces) == 2 and pieces[0].isdigit() and int(pieces[0]) > 0:
        unit = _LEGACY_BASE_FREQUENCIES.get(pieces[1])
        if unit is not None:
            count = int(pieces[0])
            return unit if count == 1 else f"{count}{unit}"
    for suffix in (
        "MIN",
        "MS",
        "T",
        "S",
        "H",
        "D",
        "B",
        "W",
        "M",
        "Q",
        "Y",
        "A",
    ):
        upper = stripped.upper()
        prefix = upper[: -len(suffix)] if upper.endswith(suffix) else ""
        if upper.endswith(suffix) and (not prefix or prefix.isdigit()):
            return upper
    raise RequestUnavailableError(
        "legacy model frequency has no reviewed conversion"
    )


def _pandas_legacy_frequency(value: str) -> str:
    if value == "H":
        return "h"
    return value


def _period_compatible_legacy_frequency(value: str) -> str:
    if value.endswith("MS"):
        multiplier = value[:-2]
        if not multiplier or multiplier.isdigit():
            return f"{multiplier}M"
    return value


def _single_forecast(forecasts: object) -> object:
    try:
        iterator = iter(forecasts)  # type: ignore[arg-type]
        forecast = next(iterator)
    except (TypeError, StopIteration) as error:
        raise ModelOutputError("Lag-Llama predictor returned no forecast") from error
    try:
        next(iterator)
    except StopIteration:
        return forecast
    raise ModelOutputError("Lag-Llama predictor returned multiple forecasts")


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


def _finite_vector(value: object, horizon: int, *, output_name: str) -> tuple[float, ...]:
    materialized = _materialize(value)
    if not isinstance(materialized, list) or any(
        isinstance(item, list) for item in materialized
    ):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    if len(materialized) != horizon:
        raise ModelOutputError(f"{output_name} has the wrong horizon length")
    return tuple(_finite_number(item, output_name) for item in materialized)


def _batch_prefix(value: object, horizon: int, *, output_name: str) -> tuple[float, ...]:
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
    return tuple(_finite_number(item, output_name) for item in row[:horizon])
