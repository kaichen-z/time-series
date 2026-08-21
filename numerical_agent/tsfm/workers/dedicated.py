"""Manifest-bound adapters for TiRex, Toto 2.0, and local TabPFN-TS."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import inspect
import math
import os
import re
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from ..manifests import ManifestRegistry, TSFMManifest
from ..protocol import WorkerRequest
from .common import (
    CheckpointUnavailableError,
    DependencyUnavailableError,
    LicenseUnavailableError,
    ModelOutputError,
    RequestUnavailableError,
    checkpoint_cache_key,
    loaded_checkpoint_revision,
    record_loaded_checkpoint,
)


_TOTO_ID = "method_tsfm_0014"
_TIREX_ID = "method_tsfm_0027"
_TABPFN_ID = "method_tsfm_0029"
_AUDITED_METHOD_IDS = frozenset({_TOTO_ID, _TIREX_ID, _TABPFN_ID})
_TABPFN_NO_BROWSER_ENV = "TABPFN_NO_BROWSER"
_TABPFN_V3_GATED_DOWNLOAD_PREFIX = (
    "Failed to download TabPFN ModelVersion.V3 model "
    "'tabpfn-v3-regressor-v3_20260506_timeseries.ckpt'.\n\n"
    "Details and instructions:\n"
    "HuggingFace authentication error downloading from "
    "'Prior-Labs/tabpfn_3'.\n"
    "This model is gated and requires you to accept its terms.\n"
)
_TABPFN_V3_GATED_DOWNLOAD_SUFFIX = (
    "\n\nFor commercial usage, we provide alternative download options for "
    "TabPFN ModelVersion.V3; please reach out to us at sales@priorlabs.ai."
)
_TABPFN_FREQUENCIES = MappingProxyType(
    {
        "s": "s",
        "second": "s",
        "seconds": "s",
        "t": "min",
        "min": "min",
        "minute": "min",
        "minutes": "min",
        "h": "h",
        "hour": "h",
        "hours": "h",
        "hourly": "h",
        "d": "D",
        "day": "D",
        "days": "D",
        "daily": "D",
        "w": "W",
        "week": "W",
        "weeks": "W",
        "weekly": "W",
        "m": "ME",
        "me": "ME",
        "month": "ME",
        "months": "ME",
        "monthly": "ME",
        "q": "QE",
        "qe": "QE",
        "quarter": "QE",
        "quarters": "QE",
        "quarterly": "QE",
        "y": "YE",
        "a": "YE",
        "ye": "YE",
        "year": "YE",
        "years": "YE",
        "yearly": "YE",
    }
)
_TABPFN_COMPACT_FREQUENCY = re.compile(
    r"^(?P<count>[1-9][0-9]*)?"
    r"(?P<unit>min|ms|me|qs|qe|ys|ye|s|t|h|d|b|w|m|q|y)"
    r"(?:-(?P<anchor>[a-z]{3}))?$",
    re.IGNORECASE,
)
_TABPFN_COMPACT_UNITS = MappingProxyType(
    {
        "s": "s",
        "t": "min",
        "min": "min",
        "h": "h",
        "d": "D",
        "b": "B",
        "w": "W",
        "m": "ME",
        "me": "ME",
        "ms": "ms",
        "q": "QE",
        "qe": "QE",
        "qs": "QS",
        "y": "YE",
        "ye": "YE",
        "ys": "YS",
    }
)
_WEEKDAY_ANCHORS = frozenset({"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"})
_MONTH_ANCHORS = frozenset(
    {
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    }
)


@dataclass(frozen=True)
class _TiRexBackend:
    torch: Any
    load_model: Callable[..., Any]

    def tensor(self, values: tuple[float, ...], *, layout: str) -> Any:
        if layout != "batch_time":
            raise ValueError(f"unsupported TiRex tensor layout {layout!r}")
        return self.torch.tensor(values, dtype=self.torch.float32).reshape(1, -1)


TOTO_PATCH_SIZE = 32
"""Toto's fixed input patch size: it reshapes the time axis into (patch, seq/patch), so any
history not already a multiple of this length must be padded before the tensor is built."""


@dataclass(frozen=True)
class _TotoBackend:
    torch: Any
    model_class: Any
    device: Any

    def tensor(
        self, values: tuple[float, ...], *, layout: str, device: object
    ) -> tuple[Any, int]:
        if layout != "batch_variate_time":
            raise ValueError(f"unsupported Toto tensor layout {layout!r}")
        remainder = len(values) % TOTO_PATCH_SIZE
        pad = (TOTO_PATCH_SIZE - remainder) if remainder else 0
        padded = (0.0,) * pad + tuple(values)
        tensor = self.torch.tensor(
            padded, dtype=self.torch.float32, device=device
        ).reshape(1, 1, -1)
        return tensor, pad

    def observed_mask(self, tensor: Any, pad: int) -> Any:
        mask = self.torch.ones_like(tensor, dtype=self.torch.bool)
        if pad:
            mask[..., :pad] = False
        return mask

    def series_ids(self, *, device: object) -> Any:
        return self.torch.zeros((1, 1), dtype=self.torch.long, device=device)

    def no_grad(self) -> Any:
        return self.torch.no_grad()


@dataclass(frozen=True)
class _TabPFNBackend:
    torch: Any
    pandas: Any
    pipeline_class: Any
    local_mode: Any
    device: Any
    prepend_cache_path: Callable[[str], Any]
    hf_hub_download: Callable[..., str]
    license_error_classes: tuple[type[BaseException], ...]

    def resolve_checkpoint(self, filename: str) -> Any:
        return self.prepend_cache_path(filename)

    def download_checkpoint(
        self, repo_id: str, filename: str, *, revision: str
    ) -> str:
        return self.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )

    def date_range(self, *, start: str, periods: int, frequency: str) -> Any:
        return self.pandas.date_range(start=start, periods=periods, freq=frequency)

    def future_date_range(
        self, context_timestamps: Any, *, periods: int, frequency: str
    ) -> Any:
        return self.pandas.date_range(
            start=context_timestamps[-1], periods=periods + 1, freq=frequency
        )[1:]

    def dataframe(self, columns: dict[str, object]) -> Any:
        return self.pandas.DataFrame(columns)


def _tabpfn_execution_device(torch: Any) -> Any:
    requested = os.environ.get("NA_TSFM_DEVICE", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        return torch.device("cuda:0")
    if requested != "auto":
        raise ValueError("unsupported TSFM execution device")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_official_backend(method_id: str) -> object:
    """Import only the package installed in the selected isolated environment."""

    if method_id == _TIREX_ID:
        import torch
        from tirex import load_model

        return _TiRexBackend(torch=torch, load_model=load_model)
    if method_id == _TOTO_ID:
        import torch
        from toto2 import Toto2Model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _TotoBackend(torch=torch, model_class=Toto2Model, device=device)
    if method_id == _TABPFN_ID:
        import pandas
        import torch
        from huggingface_hub import hf_hub_download
        from tabpfn.errors import (
            TabPFNHuggingFaceGatedRepoError,
            TabPFNLicenseError,
        )
        from tabpfn.model_loading import prepend_cache_path
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        return _TabPFNBackend(
            torch=torch,
            pandas=pandas,
            pipeline_class=TabPFNTSPipeline,
            local_mode=TabPFNMode.LOCAL,
            device=_tabpfn_execution_device(torch),
            prepend_cache_path=prepend_cache_path,
            hf_hub_download=hf_hub_download,
            license_error_classes=(
                TabPFNLicenseError,
                TabPFNHuggingFaceGatedRepoError,
            ),
        )
    raise ValueError(f"unsupported dedicated method {method_id!r}")


class DedicatedAdapter:
    """Forecast through three immutable dedicated-package bindings."""

    def __init__(self, loader: Callable[..., Any] | None = None) -> None:
        self._loader = loader
        self._backends: dict[str, Any] = {}
        self._models: dict[tuple[str, str], Any] = {}
        self._loaded_checkpoint_revisions: set[tuple[str, str]] = set()
        registry = ManifestRegistry.load_default()
        bindings = {
            manifest.checkpoint: manifest
            for method_id, manifest in registry.items()
            if method_id in _AUDITED_METHOD_IDS
        }
        if {manifest.method_id for manifest in bindings.values()} != _AUDITED_METHOD_IDS:
            raise RuntimeError("audited dedicated manifests are incomplete")
        self._bindings: Mapping[str, TSFMManifest] = MappingProxyType(bindings)

    def forecast(self, request: WorkerRequest) -> tuple[float, ...]:
        manifest = self._require_binding(request)
        normalized_frequency = None
        if manifest.method_id == _TABPFN_ID:
            normalized_frequency = _normalize_tabpfn_frequency(request.frequency)

        headless = (
            _tabpfn_headless_environment()
            if manifest.method_id == _TABPFN_ID
            else nullcontext()
        )
        with headless:
            backend = self._require_backend(manifest)
            model = self._require_model(manifest, backend, request)
            if manifest.method_id == _TIREX_ID:
                context = backend.tensor(request.history, layout="batch_time")
                output = model.forecast(
                    context=context,
                    prediction_length=request.horizon,
                )
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ModelOutputError("TiRex forecast output has an invalid shape")
                return _batch_vector(
                    output[1], request.horizon, output_name="TiRex point output"
                )

            if manifest.method_id == _TOTO_ID:
                target, pad = backend.tensor(
                    request.history,
                    layout="batch_variate_time",
                    device=backend.device,
                )
                inputs = {
                    "target": target,
                    "target_mask": backend.observed_mask(target, pad),
                    "series_ids": backend.series_ids(device=backend.device),
                }
                with backend.no_grad():
                    quantiles = model.forecast(
                        inputs,
                        horizon=request.horizon,
                        decode_block_size=None,
                        has_missing_values=False,
                    )
                return _toto_p50(quantiles, request.horizon)

            if normalized_frequency is None:
                raise RuntimeError("TabPFN frequency binding was not resolved")
            timestamps = backend.date_range(
                start="2000-01-01",
                periods=len(request.history),
                frequency=normalized_frequency,
            )
            context = backend.dataframe(
                {
                    "item_id": tuple("series" for _ in request.history),
                    "timestamp": timestamps,
                    "target": request.history,
                }
            )
            future_timestamps = backend.future_date_range(
                timestamps,
                periods=request.horizon,
                frequency=normalized_frequency,
            )
            future = backend.dataframe(
                {
                    "item_id": tuple("series" for _ in range(request.horizon)),
                    "timestamp": future_timestamps,
                }
            )
            try:
                predictions = model.predict_df(
                    context,
                    future_df=future,
                    quantiles=[0.5],
                )
            except Exception as error:
                if _is_tabpfn_license_error(error, backend):
                    raise _tabpfn_license_unavailable() from error
                raise
            try:
                target_values = predictions["target"].tolist()
            except (KeyError, TypeError, AttributeError) as error:
                raise ModelOutputError(
                    "TabPFN-TS median output has an invalid shape"
                ) from error
            return _finite_vector(
                target_values,
                request.horizon,
                output_name="TabPFN-TS median output",
            )

    def _require_binding(self, request: WorkerRequest) -> TSFMManifest:
        manifest = self._bindings.get(request.checkpoint)
        if (
            manifest is None
            or request.provider != "dedicated"
            or dict(request.runtime_options) != dict(manifest.runtime_options)
        ):
            raise RequestUnavailableError(
                "request does not match a reviewed dedicated manifest binding"
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
                f"Dedicated dependencies are unavailable: {error}"
            ) from error
        if backend is None:
            raise DependencyUnavailableError("Dedicated dependencies are unavailable")
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
            if manifest.method_id == _TIREX_ID:
                load_options = (
                    {"hf_kwargs": {"revision": request.checkpoint_revision}}
                    if request.checkpoint_revision
                    else {}
                )
                model = backend.load_model(manifest.checkpoint, **load_options)
            elif manifest.method_id == _TOTO_ID:
                load_options = (
                    {"revision": request.checkpoint_revision}
                    if request.checkpoint_revision
                    else {}
                )
                model = backend.model_class.from_pretrained(
                    manifest.checkpoint,
                    **load_options,
                )
                model = model.to(backend.device).eval()
            else:
                filename = str(manifest.runtime_options["checkpoint_file"])
                model_path = (
                    backend.download_checkpoint(
                        manifest.checkpoint,
                        filename,
                        revision=request.checkpoint_revision,
                    )
                    if request.checkpoint_revision
                    else backend.resolve_checkpoint(filename)
                )
                model = backend.pipeline_class(
                    tabpfn_mode=backend.local_mode,
                    tabpfn_output_selection="median",
                    tabpfn_model_config={
                        "model_path": model_path,
                        "device": str(backend.device),
                    },
                )
        except ImportError as error:
            raise DependencyUnavailableError(
                f"Dedicated dependencies are unavailable: {error}"
            ) from error
        except Exception as error:
            if manifest.method_id == _TABPFN_ID and _is_tabpfn_license_error(
                error, backend
            ):
                raise _tabpfn_license_unavailable() from error
            raise CheckpointUnavailableError(
                f"Dedicated checkpoint {manifest.checkpoint!r} is unavailable: {error}"
            ) from error
        record_loaded_checkpoint(self._loaded_checkpoint_revisions, request)
        self._models[key] = model
        return model

    def loaded_checkpoint_revision(self, request: WorkerRequest) -> str:
        return loaded_checkpoint_revision(self._loaded_checkpoint_revisions, request)


def _call_injected_loader(loader: Callable[..., Any], method_id: str) -> Any:
    try:
        inspect.signature(loader).bind(method_id)
    except (TypeError, ValueError):
        return loader()
    return loader(method_id)


@contextmanager
def _tabpfn_headless_environment() -> Iterator[None]:
    previous = os.environ.get(_TABPFN_NO_BROWSER_ENV)
    os.environ[_TABPFN_NO_BROWSER_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TABPFN_NO_BROWSER_ENV, None)
        else:
            os.environ[_TABPFN_NO_BROWSER_ENV] = previous


def _is_tabpfn_license_error(error: BaseException, backend: Any) -> bool:
    if _is_unchained_tabpfn_v3_gated_download(error):
        return True
    error_types = getattr(backend, "license_error_classes", ())
    if not isinstance(error_types, tuple):
        return False
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, error_types):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_unchained_tabpfn_v3_gated_download(error: BaseException) -> bool:
    if (
        type(error) is not RuntimeError
        or error.__cause__ is not None
        or error.__context__ is not None
    ):
        return False
    message = str(error)
    return message.startswith(_TABPFN_V3_GATED_DOWNLOAD_PREFIX) and message.endswith(
        _TABPFN_V3_GATED_DOWNLOAD_SUFFIX
    )


def _tabpfn_license_unavailable() -> LicenseUnavailableError:
    return LicenseUnavailableError(
        "TabPFN-TS local weights require prior license acceptance and a valid "
        "TABPFN_TOKEN or cached PriorLabs credential when the checkpoint is absent"
    )


def _normalize_tabpfn_frequency(value: str) -> str:
    stripped = " ".join(value.strip().split())
    compact = stripped.lower()
    direct = _TABPFN_FREQUENCIES.get(compact)
    if direct is not None:
        return direct
    pieces = compact.split(" ", 1)
    if len(pieces) == 2 and pieces[0].isdigit() and int(pieces[0]) > 0:
        unit = _TABPFN_FREQUENCIES.get(pieces[1])
        if unit is not None:
            count = int(pieces[0])
            return unit if count == 1 else f"{count}{unit}"
    match = _TABPFN_COMPACT_FREQUENCY.fullmatch(stripped)
    if match is not None:
        raw_unit = match.group("unit")
        source_unit = raw_unit.lower()
        anchor = match.group("anchor")
        normalized_anchor = None if anchor is None else anchor.upper()
        anchor_is_valid = (
            anchor is None
            or (source_unit == "w" and normalized_anchor in _WEEKDAY_ANCHORS)
            or (
                source_unit in {"q", "qe", "qs", "y", "ye", "ys"}
                and normalized_anchor in _MONTH_ANCHORS
            )
        )
        milliseconds_case_is_valid = source_unit != "ms" or raw_unit in {
            "ms",
            "MS",
        }
        if anchor_is_valid and milliseconds_case_is_valid:
            unit = "MS" if raw_unit == "MS" else _TABPFN_COMPACT_UNITS[source_unit]
            count = match.group("count")
            prefix = "" if count in {None, "1"} else count
            suffix = "" if anchor is None else f"-{normalized_anchor}"
            return f"{prefix}{unit}{suffix}"
    raise RequestUnavailableError(
        "TabPFN-TS frequency has no reviewed timestamp conversion"
    )


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


def _batch_vector(value: object, horizon: int, *, output_name: str) -> tuple[float, ...]:
    materialized = _materialize(value)
    if (
        not isinstance(materialized, list)
        or len(materialized) != 1
        or not isinstance(materialized[0], list)
        or any(isinstance(item, list) for item in materialized[0])
    ):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    return _finite_vector(materialized[0], horizon, output_name=output_name)


def _toto_p50(value: object, horizon: int) -> tuple[float, ...]:
    materialized = _materialize(value)
    if not isinstance(materialized, list) or len(materialized) != 9:
        raise ModelOutputError("Toto quantile output must contain exactly 9 quantiles")
    for quantile in materialized:
        if (
            not isinstance(quantile, list)
            or len(quantile) != 1
            or not isinstance(quantile[0], list)
            or len(quantile[0]) != 1
        ):
            raise ModelOutputError("Toto quantile output has an invalid shape")
        _finite_vector(
            quantile[0][0], horizon, output_name="Toto quantile output"
        )
    return _finite_vector(
        materialized[4][0][0], horizon, output_name="Toto median output"
    )
