"""Command-line entry point for a reviewed TSFM worker adapter."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import json
import math
import sys
from typing import Any, TextIO

from ..providers import RuntimeUnavailableError
from .protocol import (
    WorkerRequest,
    WorkerResponse,
    is_immutable_checkpoint_revision,
)
from .security import SecretRedactor
from .workers.common import (
    CheckpointAttestationUnavailableError,
    InvalidRequestError,
)


# Adapter names and imports are repository-reviewed constants. CLI input can only select a key.
_ADAPTER_TARGETS = {
    "granite": ("numerical_agent.tsfm.workers.granite", "GraniteAdapter"),
    "uni2ts": ("numerical_agent.tsfm.workers.uni2ts", "Uni2TSAdapter"),
    "transformer_generation": (
        "numerical_agent.tsfm.workers.transformer_generation",
        "TransformerGenerationAdapter",
    ),
    "dedicated": ("numerical_agent.tsfm.workers.dedicated", "DedicatedAdapter"),
    "legacy": ("numerical_agent.tsfm.workers.legacy", "LegacyAdapter"),
}


def _load_adapter(name: str) -> Any:
    module_name, class_name = _ADAPTER_TARGETS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def _request_id_from_invalid_line(line: str) -> str:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return "unknown"
    if isinstance(value, dict) and isinstance(value.get("request_id"), str):
        return value["request_id"] or "unknown"
    return "unknown"


def _object_device(value: object) -> str | None:
    device = getattr(value, "device", None)
    if device is not None:
        text = str(device)
        if text:
            return text
    parameters = getattr(value, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(iter(parameters()))
            text = str(getattr(parameter, "device", ""))
            if text:
                return text
        except (StopIteration, TypeError, RuntimeError):
            pass
    return None


def _adapter_device(adapter: object, request: WorkerRequest) -> str | None:
    # These reviewed loaders explicitly select CPU even when CUDA is visible.
    if request.checkpoint in {
        "google/timesfm-1.0-200m-pytorch",
        "Maple728/TimeMoE-200M",
    }:
        return "cpu"
    for attribute in ("_models", "_modules", "_lag_predictors", "_backends"):
        cache = getattr(adapter, attribute, None)
        if not isinstance(cache, dict):
            continue
        for value in cache.values():
            device = _object_device(value)
            if device:
                return device
    return None


def _runtime_measurements(
    adapter: object, request: WorkerRequest
) -> dict[str, object]:
    """Return process/device measurements without importing an optional model stack."""

    measurements: dict[str, object] = {}
    device = _adapter_device(adapter, request)
    if device:
        measurements["device"] = device
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if device is not None and device.startswith("cuda"):
        try:
            index = int(device.split(":", 1)[1]) if ":" in device else int(
                cuda.current_device()
            )
            peak = int(cuda.max_memory_allocated(index))
            if peak >= 0:
                measurements["peak_memory_bytes"] = peak
        except Exception:
            pass

    if device in {"cpu", "mps"} and "peak_memory_bytes" not in measurements:
        try:
            import resource

            peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if math.isfinite(peak_rss) and peak_rss >= 0:
                multiplier = 1 if sys.platform == "darwin" else 1024
                measurements["peak_memory_bytes"] = int(peak_rss * multiplier)
        except (ImportError, OSError, TypeError, ValueError):
            pass
    return measurements


def _loaded_checkpoint_attestation(
    adapter: object, request: WorkerRequest
) -> str:
    reader = getattr(adapter, "loaded_checkpoint_revision", None)
    if not callable(reader):
        raise CheckpointAttestationUnavailableError(
            "worker adapter did not report an exact loaded checkpoint revision"
        )
    revision = reader(request)
    if not is_immutable_checkpoint_revision(revision):
        raise CheckpointAttestationUnavailableError(
            "worker adapter reported an invalid loaded checkpoint revision"
        )
    return revision


def serve(adapter_name: str, input_stream: TextIO, output_stream: TextIO) -> None:
    adapter = _load_adapter(adapter_name)
    redactor = SecretRedactor.from_environment()
    for line in input_stream:
        try:
            request = WorkerRequest.from_json(line)
        except ValueError as error:
            response = WorkerResponse.failure(
                _request_id_from_invalid_line(line),
                "invalid_request",
                "invalid_protocol_request",
                str(error),
            )
        else:
            try:
                result = adapter.forecast(request)
                if isinstance(result, WorkerResponse):
                    if result.status == "success" and request.checkpoint_revision:
                        metadata = dict(result.metadata)
                        metadata["checkpoint_revision"] = _loaded_checkpoint_attestation(
                            adapter, request
                        )
                        response = WorkerResponse.success(
                            result.request_id,
                            result.values,
                            metadata,
                        )
                    else:
                        response = result
                else:
                    values: Sequence[float] = result
                    metadata = {"checkpoint": request.checkpoint}
                    if request.checkpoint_revision:
                        metadata["checkpoint_revision"] = (
                            _loaded_checkpoint_attestation(adapter, request)
                        )
                    metadata.update(_runtime_measurements(adapter, request))
                    response = WorkerResponse.success(
                        request.request_id,
                        tuple(values),
                        metadata,
                    )
            except RuntimeUnavailableError as error:
                response = WorkerResponse.failure(
                    request.request_id,
                    "unavailable",
                    getattr(error, "reason_code", "runtime_unavailable"),
                    str(error) or type(error).__name__,
                )
            except InvalidRequestError as error:
                response = WorkerResponse.failure(
                    request.request_id,
                    "invalid_request",
                    "adapter_rejected_request",
                    str(error) or type(error).__name__,
                )
            except Exception as error:
                response = WorkerResponse.failure(
                    request.request_id,
                    "runtime_error",
                    "adapter_runtime_error",
                    str(error) or type(error).__name__,
                )
        response = redactor.sanitize_response(response)
        output_stream.write(response.to_json() + "\n")
        output_stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=sorted(_ADAPTER_TARGETS))
    args = parser.parse_args(argv)
    serve(args.adapter, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
