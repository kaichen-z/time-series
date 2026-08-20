"""Shared validation and typed unavailability for isolated model adapters."""

from __future__ import annotations

import math
from typing import Any

from ...providers import RuntimeUnavailableError
from ..protocol import WorkerRequest, is_immutable_checkpoint_revision


CheckpointCacheKey = tuple[str, str]


class AdapterUnavailableError(RuntimeUnavailableError):
    """An adapter cannot serve a request for a known, typed reason."""

    reason_code = "runtime_unavailable"


class DependencyUnavailableError(AdapterUnavailableError):
    """The isolated environment lacks a required model dependency."""

    reason_code = "dependency_unavailable"


class CheckpointUnavailableError(AdapterUnavailableError):
    """A reviewed model checkpoint could not be loaded."""

    reason_code = "checkpoint_unavailable"


class CheckpointAttestationUnavailableError(AdapterUnavailableError):
    """A smoke load did not produce a valid post-load checkpoint identity."""

    reason_code = "checkpoint_attestation_unavailable"


class LicenseUnavailableError(AdapterUnavailableError):
    """A gated model cannot run without explicit upstream access."""

    reason_code = "license_not_acknowledged"


class RequestUnavailableError(AdapterUnavailableError):
    """A valid protocol request exceeds a reviewed model capability."""

    reason_code = "request_unavailable"


class InvalidRequestError(ValueError):
    """Adapter-specific request validation failed."""


class ModelOutputError(RuntimeError):
    """A loaded model returned malformed or non-finite forecast data."""


def checkpoint_cache_key(request: WorkerRequest) -> CheckpointCacheKey:
    return request.checkpoint, request.checkpoint_revision


def record_loaded_checkpoint(
    loaded: set[CheckpointCacheKey], request: WorkerRequest
) -> None:
    if request.checkpoint_revision:
        loaded.add(checkpoint_cache_key(request))


def loaded_checkpoint_revision(
    loaded: set[CheckpointCacheKey], request: WorkerRequest
) -> str:
    revision = request.checkpoint_revision
    if (
        not is_immutable_checkpoint_revision(revision)
        or checkpoint_cache_key(request) not in loaded
    ):
        raise CheckpointAttestationUnavailableError(
            "exact checkpoint load attestation is unavailable"
        )
    return revision


def finite_single_series(
    value: object,
    horizon: int,
    *,
    leading_singletons: int,
    output_name: str,
    allow_longer: bool = False,
) -> tuple[float, ...]:
    """Extract an exact ``[..., horizon, 1]`` univariate forecast."""

    nested = _materialize(value)
    for _ in range(leading_singletons):
        if not isinstance(nested, list) or len(nested) != 1:
            raise ModelOutputError(f"{output_name} has an invalid shape")
        nested = nested[0]
    if not isinstance(nested, list):
        raise ModelOutputError(f"{output_name} has an invalid shape")
    if len(nested) < horizon or (not allow_longer and len(nested) != horizon):
        raise ModelOutputError(f"{output_name} has the wrong horizon length")

    values: list[float] = []
    for channel_value in nested:
        if not isinstance(channel_value, list) or len(channel_value) != 1:
            raise ModelOutputError(f"{output_name} has an invalid shape")
        scalar = channel_value[0]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise ModelOutputError(f"{output_name} must contain finite numbers")
        try:
            number = float(scalar)
        except OverflowError as error:
            raise ModelOutputError(
                f"{output_name} must contain finite numbers"
            ) from error
        if not math.isfinite(number):
            raise ModelOutputError(f"{output_name} must contain only finite values")
        values.append(number)
    return tuple(values[:horizon])


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
