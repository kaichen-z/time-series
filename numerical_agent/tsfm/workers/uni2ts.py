"""Manifest-bound worker adapter for Salesforce Uni2TS models."""

from __future__ import annotations

from dataclasses import dataclass
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


_MOIRAI_ID = "method_tsfm_0003"
_MOIRAI2_ID = "method_tsfm_0017"
_MOIRAI_MOE_ID = "method_tsfm_0019"
_AUDITED_METHOD_IDS = frozenset({_MOIRAI_ID, _MOIRAI2_ID, _MOIRAI_MOE_ID})
_FREQUENCIES = MappingProxyType(
    {
        "s": "s",
        "1s": "s",
        "1 second": "s",
        "second": "s",
        "seconds": "s",
        "min": "min",
        "1min": "min",
        "1 minute": "min",
        "minute": "min",
        "minutes": "min",
        "d": "D",
        "1d": "D",
        "1 day": "D",
        "day": "D",
        "daily": "D",
        "h": "h",
        "1h": "h",
        "1 hour": "h",
        "hour": "h",
        "hourly": "h",
        "w": "W",
        "1w": "W",
        "1 week": "W",
        "week": "W",
        "weekly": "W",
        "m": "M",
        "1m": "M",
        "1 month": "M",
        "month": "M",
        "monthly": "M",
        "q": "Q",
        "1q": "Q",
        "1 quarter": "Q",
        "quarter": "Q",
        "quarterly": "Q",
        "y": "Y",
        "1y": "Y",
        "1 year": "Y",
        "year": "Y",
        "yearly": "Y",
    }
)
_NATURAL_FREQUENCY = re.compile(
    r"^(?P<count>[1-9][0-9]*)\s*(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?)$",
    re.IGNORECASE,
)
_UNIT_ALIASES = MappingProxyType(
    {
        "second": "s",
        "seconds": "s",
        "minute": "min",
        "minutes": "min",
        "hour": "h",
        "hours": "h",
        "day": "D",
        "days": "D",
        "week": "W",
        "weeks": "W",
        "month": "M",
        "months": "M",
        "quarter": "Q",
        "quarters": "Q",
        "year": "Y",
        "years": "Y",
    }
)
_PANDAS_ALIAS_SHAPE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z]{3})?$")


@dataclass(frozen=True)
class _OfficialBackend:
    moirai_module_class: Any
    moirai_forecast_class: Any
    moirai2_module_class: Any
    moirai2_forecast_class: Any
    moirai_moe_module_class: Any
    moirai_moe_forecast_class: Any


def _load_official_backend() -> _OfficialBackend:
    # Uni2TS and its transitive PyTorch/GluonTS stack belong only to the worker.
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

    return _OfficialBackend(
        moirai_module_class=MoiraiModule,
        moirai_forecast_class=MoiraiForecast,
        moirai2_module_class=Moirai2Module,
        moirai2_forecast_class=Moirai2Forecast,
        moirai_moe_module_class=MoiraiMoEModule,
        moirai_moe_forecast_class=MoiraiMoEForecast,
    )


def _create_pandas_dataset(history: tuple[float, ...], frequency: str) -> Any:
    # Pandas and GluonTS are optional and must never enter the harness process.
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset

    last_error: ValueError | None = None
    for date_frequency in _date_frequency_candidates(frequency):
        try:
            timestamp_index = pd.date_range(
                start="2000-01-01",
                periods=len(history),
                freq=date_frequency,
            )
        except ValueError as error:
            last_error = error
            continue
        break
    else:
        if last_error is None:
            raise ValueError(f"invalid pandas frequency {frequency!r}")
        raise last_error

    period_index = timestamp_index.to_period()
    period_frequency = period_index.freqstr
    target = pd.Series(history, index=period_index, name="target")
    return PandasDataset({"target": target}, freq=period_frequency)


def _date_frequency_candidates(frequency: str) -> tuple[str, ...]:
    match = re.fullmatch(
        r"(?P<count>[1-9][0-9]*)?(?P<unit>M|Q|Y)(?P<anchor>-[A-Za-z]{3})?",
        frequency,
    )
    if match is None:
        if frequency == "h":
            return ("h", "H")
        return (frequency,)
    count = match.group("count") or ""
    anchor = match.group("anchor") or ""
    unit = match.group("unit")
    modern = {"M": "ME", "Q": "QE", "Y": "YE"}[unit]
    candidates = [f"{count}{modern}{anchor}", f"{count}{unit}{anchor}"]
    if unit == "Y":
        candidates.append(f"{count}A{anchor}")
    return tuple(candidates)


class Uni2TSAdapter:
    """Forecast with the three exact Uni2TS-family manifest bindings."""

    def __init__(
        self,
        loader: Callable[[], Any] | None = None,
        dataset_factory: Callable[[tuple[float, ...], str], Any] | None = None,
    ) -> None:
        self._loader = loader or _load_official_backend
        self._dataset_factory = dataset_factory or _create_pandas_dataset
        self._backend: Any | None = None
        self._modules: dict[tuple[str, str], Any] = {}
        self._loaded_checkpoint_revisions: set[tuple[str, str]] = set()
        registry = ManifestRegistry.load_default()
        bindings = {
            manifest.checkpoint: manifest
            for method_id, manifest in registry.items()
            if method_id in _AUDITED_METHOD_IDS
        }
        if {manifest.method_id for manifest in bindings.values()} != _AUDITED_METHOD_IDS:
            raise RuntimeError("audited Uni2TS manifests are incomplete")
        self._bindings: Mapping[str, TSFMManifest] = MappingProxyType(bindings)

    def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
        manifest = self._require_binding(request)
        frequency = self._normalize_frequency(request.frequency)
        self._validate_token_budget(request, manifest)
        try:
            dataset = self._dataset_factory(request.history, frequency)
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Uni2TS dataset dependencies are unavailable: {error}"
            ) from error
        except ValueError as error:
            raise RequestUnavailableError(
                f"Uni2TS frequency {request.frequency!r} is invalid: {error}"
            ) from error
        backend = self._require_backend()
        module = self._require_module(manifest, backend, request)

        common = {
            "module": module,
            "prediction_length": request.horizon,
            "context_length": len(request.history),
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
        }
        try:
            if manifest.method_id == _MOIRAI_ID:
                forecast_model = backend.moirai_forecast_class(
                    **common,
                    patch_size=manifest.runtime_options["patch_size"],
                    num_samples=int(manifest.runtime_options["num_samples"]),
                )
            elif manifest.method_id == _MOIRAI2_ID:
                forecast_model = backend.moirai2_forecast_class(**common)
            else:
                forecast_model = backend.moirai_moe_forecast_class(
                    **common,
                    patch_size=int(manifest.runtime_options["patch_size"]),
                    num_samples=int(manifest.runtime_options["num_samples"]),
                )
            predictor = forecast_model.create_predictor(batch_size=1, device="auto")
            forecast = _single_forecast(predictor.predict(dataset))
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Uni2TS dependencies are unavailable: {error}"
            ) from error

        quantile = getattr(forecast, "quantile", None)
        if not callable(quantile):
            raise ModelOutputError("Uni2TS forecast has an invalid shape")
        return _finite_vector(
            quantile(0.5),
            request.horizon,
            output_name="Uni2TS p50 forecast",
        )

    def _require_binding(self, request: WorkerRequest) -> TSFMManifest:
        manifest = self._bindings.get(request.checkpoint)
        if (
            manifest is None
            or request.provider != "uni2ts"
            or dict(request.runtime_options) != dict(manifest.runtime_options)
        ):
            raise RequestUnavailableError(
                "request does not match a reviewed Uni2TS manifest binding"
            )
        return manifest

    @staticmethod
    def _normalize_frequency(frequency: str) -> str:
        stripped = frequency.strip()
        normalized = _FREQUENCIES.get(stripped.lower())
        if normalized is not None:
            return normalized
        natural = _NATURAL_FREQUENCY.fullmatch(stripped)
        if natural is not None:
            count = natural.group("count")
            unit = _UNIT_ALIASES[natural.group("unit").lower()]
            return f"{count}{unit}"
        if _PANDAS_ALIAS_SHAPE.fullmatch(stripped):
            return stripped
        raise RequestUnavailableError(
            "Uni2TS frequency must be a valid pandas/GluonTS frequency"
        )

    @staticmethod
    def _validate_token_budget(
        request: WorkerRequest,
        manifest: TSFMManifest,
    ) -> None:
        configured_patch = manifest.runtime_options["patch_size"]
        patch_size = 8 if configured_patch == "auto" else int(configured_patch)
        token_count = math.ceil(len(request.history) / patch_size) + math.ceil(
            request.horizon / patch_size
        )
        max_tokens = int(manifest.runtime_options["max_patch_tokens"])
        if token_count > max_tokens:
            raise RequestUnavailableError(
                f"Uni2TS request must not exceed {max_tokens} patch tokens"
            )

    def _require_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._loader()
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Uni2TS dependencies are unavailable: {error}"
            ) from error
        if backend is None:
            raise DependencyUnavailableError("Uni2TS dependencies are unavailable")
        self._backend = backend
        return backend

    def _require_module(
        self, manifest: TSFMManifest, backend: Any, request: WorkerRequest
    ) -> Any:
        key = checkpoint_cache_key(request)
        cached = self._modules.get(key)
        if cached is not None:
            return cached
        if manifest.method_id == _MOIRAI_ID:
            module_class = backend.moirai_module_class
        elif manifest.method_id == _MOIRAI2_ID:
            module_class = backend.moirai2_module_class
        else:
            module_class = backend.moirai_moe_module_class
        try:
            load_options = (
                {"revision": request.checkpoint_revision}
                if request.checkpoint_revision
                else {}
            )
            module = module_class.from_pretrained(
                manifest.checkpoint,
                **load_options,
            )
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Uni2TS dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            raise CheckpointUnavailableError(
                f"Uni2TS checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._modules[key] = module
        return module

    def loaded_checkpoint_revision(self, request: WorkerRequest) -> str:
        return loaded_checkpoint_revision(self._loaded_checkpoint_revisions, request)


def _single_forecast(forecasts: object) -> object:
    try:
        iterator = iter(forecasts)  # type: ignore[arg-type]
        forecast = next(iterator)
    except (TypeError, StopIteration) as error:
        raise ModelOutputError("Uni2TS predictor returned no forecast") from error
    try:
        next(iterator)
    except StopIteration:
        return forecast
    raise ModelOutputError("Uni2TS predictor returned multiple forecasts")


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
