"""Numeric-only zero-shot TSFM adapters (Chronos, TimesFM, Toto, Moirai), shared across agents."""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class BackboneUnavailableError(RuntimeError):
    """Raised when a configured forecasting backbone cannot be loaded."""


def resolve_device(requested: str) -> str:
    """Turn "auto" into the best device present, and pass anything explicit straight through."""
    if requested != "auto":
        return requested
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ChronosConfig:
    model_id: str = "amazon/chronos-bolt-small"
    device_map: str = "auto"
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
            # Built on CPU and pinned there for the call: the pipeline moves the input to the
            # model's own device internally, and must not inherit a caller's ambient
            # torch.set_default_device, which some of its internal steps cannot tolerate.
            with torch.device("cpu"):
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


@dataclass(frozen=True)
class Chronos2Config:
    model_id: str = "amazon/chronos-2"
    device: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = False
    max_context: int | None = None

    def __post_init__(self) -> None:
        if self.max_context is not None and self.max_context <= 0:
            raise ValueError("Chronos-2 context limit must be positive")


class Chronos2Forecaster:
    """Lazy zero-shot Chronos-2 adapter.

    Chronos-2 is multivariate, so its API takes and returns lists of series rather than the
    single tensor Chronos-Bolt accepts; that difference is the whole reason it is a separate
    adapter rather than another model id for ChronosForecaster.
    """

    def __init__(self, config: Chronos2Config | None = None, runtime_module: Any | None = None) -> None:
        self.config = config or Chronos2Config()
        self._runtime_module = runtime_module
        self._pipeline: Any | None = None

    def _ensure_pipeline(self) -> tuple[Any, Any]:
        try:
            chronos = self._runtime_module or importlib.import_module("chronos")
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Chronos-2 is the configured backbone but chronos is not installed. "
                "Install it with: pip install chronos-forecasting"
            ) from error
        if self._pipeline is not None:
            return self._pipeline, torch
        load_kwargs: dict[str, Any] = {
            "device_map": resolve_device(self.config.device),
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir:
            load_kwargs["cache_dir"] = str(Path(self.config.cache_dir).expanduser().resolve())
        try:
            self._pipeline = chronos.BaseChronosPipeline.from_pretrained(
                self.config.model_id, **load_kwargs
            )
        except Exception as error:
            raise BackboneUnavailableError(
                f"Could not load Chronos-2 checkpoint {self.config.model_id!r}: {error}"
            ) from error
        return self._pipeline, torch

    def forecast(self, history: Sequence[float], horizon: int) -> tuple[float, ...]:
        pipeline, torch = self._ensure_pipeline()
        context = (
            history[-self.config.max_context :] if self.config.max_context is not None else history
        )
        try:
            # (n_series, n_variates, history_length) is the only shape Chronos-2 accepts.
            # Always built on CPU: the pipeline pins this input before moving it to its own
            # device internally, and pinning is only valid for a CPU tensor. Building it under
            # a caller's torch.set_default_device("cuda") would otherwise fail right here.
            tensor = torch.tensor(
                list(context), dtype=torch.float32, device="cpu"
            ).reshape(1, 1, -1)
            # The pipeline's own DataLoader creates internal tensors with no device argument
            # and pins them as a CPU-only step before moving them to the model itself; those
            # calls are not immune to a caller's ambient torch.set_default_device, so this
            # whole call is pinned back to CPU regardless of what the caller has set.
            with torch.device("cpu"):
                _quantiles, mean = pipeline.predict_quantiles(
                    inputs=tensor, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
                )
            values = tuple(float(value) for value in mean[0].reshape(-1).tolist())
        except Exception as error:
            raise BackboneUnavailableError(f"Chronos-2 inference failed: {error}") from error
        return _checked(values, horizon, "Chronos-2")


@dataclass(frozen=True)
class TotoConfig:
    model_id: str = "Datadog/Toto-Open-Base-1.0"
    device: str = "auto"
    num_samples: int = 64
    samples_per_batch: int = 32
    max_context: int | None = 4096
    time_interval_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.num_samples <= 0 or self.samples_per_batch <= 0:
            raise ValueError("Toto sample counts must be positive")
        if self.max_context is not None and self.max_context <= 0:
            raise ValueError("Toto context limit must be positive")


class TotoForecaster:
    """Lazy zero-shot Toto adapter.

    Toto is a sampling forecaster, so the point forecast is the median over num_samples paths.
    Its checkpoint is loaded from a snapshot directory rather than through the hub mixin: the
    mixin's own from_pretrained is pinned to an older huggingface_hub signature and raises on
    the version installed here.
    """

    def __init__(self, config: TotoConfig | None = None) -> None:
        self.config = config or TotoConfig()
        self._forecaster: Any | None = None

    def _ensure_forecaster(self) -> tuple[Any, Any]:
        try:
            torch = importlib.import_module("torch")
            toto_model = importlib.import_module("toto.model.toto")
            toto_inference = importlib.import_module("toto.inference.forecaster")
            toto_dataset = importlib.import_module("toto.data.util.dataset")
            hub = importlib.import_module("huggingface_hub")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Toto is the configured backbone but is not installed. "
                "Install it with: pip install --no-deps toto-ts"
            ) from error
        if self._forecaster is not None:
            return self._forecaster, (torch, toto_dataset)
        try:
            directory = hub.snapshot_download(repo_id=self.config.model_id)
            device = resolve_device(self.config.device)
            model = toto_model.Toto.load_from_checkpoint(directory, device, False)
            model.eval()
            self._forecaster = toto_inference.TotoForecaster(model.model)
        except Exception as error:
            raise BackboneUnavailableError(
                f"Could not load Toto checkpoint {self.config.model_id!r}: {error}"
            ) from error
        return self._forecaster, (torch, toto_dataset)

    def forecast(self, history: Sequence[float], horizon: int) -> tuple[float, ...]:
        forecaster, (torch, toto_dataset) = self._ensure_forecaster()
        context = (
            history[-self.config.max_context :] if self.config.max_context is not None else history
        )
        device = resolve_device(self.config.device)
        try:
            series = torch.tensor(list(context), dtype=torch.float32).unsqueeze(0).to(device)
            inputs = toto_dataset.MaskedTimeseries(
                series=series,
                padding_mask=torch.full_like(series, True, dtype=torch.bool),
                id_mask=torch.zeros_like(series),
                timestamp_seconds=torch.zeros_like(series),
                time_interval_seconds=torch.full((1,), self.config.time_interval_seconds).to(device),
            )
            forecast = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=self.config.num_samples,
                samples_per_batch=self.config.samples_per_batch,
            )
            values = tuple(float(value) for value in forecast.median.reshape(-1).tolist())
        except Exception as error:
            raise BackboneUnavailableError(f"Toto inference failed: {error}") from error
        return _checked(values, horizon, "Toto")


@dataclass(frozen=True)
class MoiraiConfig:
    model_id: str = "Salesforce/moirai-2.0-R-small"
    device: str = "auto"
    max_context: int = 4096

    def __post_init__(self) -> None:
        if self.max_context <= 0:
            raise ValueError("Moirai context limit must be positive")


class MoiraiForecaster:
    """Lazy zero-shot Moirai 2.0 adapter.

    Moirai predicts a fixed set of quantiles rather than a mean, so the point forecast is the
    median quantile. The module is rebuilt per context length because Moirai2Forecast fixes
    context_length and prediction_length at construction time.
    """

    def __init__(self, config: MoiraiConfig | None = None) -> None:
        self.config = config or MoiraiConfig()
        self._module: Any | None = None

    def _ensure_module(self) -> tuple[Any, Any]:
        try:
            torch = importlib.import_module("torch")
            moirai = importlib.import_module("uni2ts.model.moirai2")
        except ImportError as error:
            raise BackboneUnavailableError(
                "Moirai is the configured backbone but uni2ts is not installed. "
                "Install it with: pip install --no-deps uni2ts"
            ) from error
        if self._module is not None:
            return self._module, (torch, moirai)
        try:
            device = resolve_device(self.config.device)
            self._module = moirai.Moirai2Module.from_pretrained(self.config.model_id).to(device)
        except Exception as error:
            raise BackboneUnavailableError(
                f"Could not load Moirai checkpoint {self.config.model_id!r}: {error}"
            ) from error
        return self._module, (torch, moirai)

    def forecast(self, history: Sequence[float], horizon: int) -> tuple[float, ...]:
        module, (torch, moirai) = self._ensure_module()
        context = list(history[-self.config.max_context :])
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
            quantiles = model(
                past_target=past,
                past_observed_target=torch.ones_like(past, dtype=torch.bool),
                past_is_pad=torch.zeros(1, len(context), dtype=torch.bool).to(device),
            )
            # (batch, quantile, horizon): the middle quantile is the point forecast.
            median = quantiles.median(dim=1).values
            values = tuple(float(value) for value in median.reshape(-1).tolist())
        except Exception as error:
            raise BackboneUnavailableError(f"Moirai inference failed: {error}") from error
        return _checked(values, horizon, "Moirai")


def _checked(values: tuple[float, ...], horizon: int, model: str) -> tuple[float, ...]:
    """Reject a forecast of the wrong length or with a non-finite value, naming the model."""
    if len(values) != horizon:
        raise BackboneUnavailableError(
            f"{model} returned {len(values)} values, expected {horizon}"
        )
    if not all(math.isfinite(value) for value in values):
        raise BackboneUnavailableError(f"{model} returned a non-finite point forecast")
    return values


# One loaded model per process. The evolving methods module cannot hold a cache of its own --
# only its functions survive a rewrite -- and reloading a checkpoint per task would dominate
# the measurement.
_SHARED: dict[str, Any] = {}

_FORECASTERS = {
    "chronos_bolt": lambda: ChronosForecaster(ChronosConfig(validate_output=True)),
    "chronos_2": Chronos2Forecaster,
    "timesfm_2_5": TimesFMForecaster,
    "toto": TotoForecaster,
    "moirai": MoiraiForecaster,
}


def shared_forecaster(kind: str) -> Any:
    """Return the process-wide forecaster for one model, loading it on first use."""
    if kind not in _FORECASTERS:
        raise BackboneUnavailableError(f"unknown forecaster {kind!r}")
    if kind not in _SHARED:
        _SHARED[kind] = _FORECASTERS[kind]()
    return _SHARED[kind]
