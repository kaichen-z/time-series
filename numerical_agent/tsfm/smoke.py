"""Run one label-free official-checkpoint smoke through the production broker."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid
from typing import Any

from ..providers import RuntimeUnavailableError
from .broker import WorkerBroker, WorkerCommand
from .deployment import TSFMDeployment, parse_acknowledged_licenses
from .manifests import ManifestRegistry, TSFMManifest
from .protocol import (
    WorkerRequest,
    WorkerResponse,
    is_immutable_checkpoint_revision,
)
from .security import SecretRedactor, controlled_worker_environment


SMOKE_HISTORY = (
    0.00, 0.50, 1.25, 0.75, 1.00, 1.50, 2.25, 1.75,
    2.00, 2.50, 3.25, 2.75, 3.00, 3.50, 4.25, 3.75,
    4.00, 4.50, 5.25, 4.75, 5.00, 5.50, 6.25, 5.75,
    6.00, 6.50, 7.25, 6.75, 7.00, 7.50, 8.25, 7.75,
    8.00, 8.50, 9.25, 8.75, 9.00, 9.50, 10.25, 9.75,
    10.00, 10.50, 11.25, 10.75, 11.00, 11.50, 12.25, 11.75,
    12.00, 12.50, 13.25, 12.75, 13.00, 13.50, 14.25, 13.75,
    14.00, 14.50, 15.25, 14.75, 15.00, 15.50, 16.25, 15.75,
    16.00, 16.50, 17.25, 16.75, 17.00, 17.50, 18.25, 17.75,
    18.00, 18.50, 19.25, 18.75, 19.00, 19.50, 20.25, 19.75,
    20.00, 20.50, 21.25, 20.75, 21.00, 21.50, 22.25, 21.75,
    22.00, 22.50, 23.25, 22.75, 23.00, 23.50, 24.25, 23.75,
)
SMOKE_HORIZON = 4
SMOKE_FREQUENCY = "H"
_DEFAULT_TIMEOUT_SECONDS = 1_800.0
_DEVICE = re.compile(r"^(?:auto|cpu|cuda|cuda:[0-9]+)$")
_ACTUAL_DEVICE = re.compile(r"^(?:cpu|mps|cuda|cuda:[0-9]+)$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_VERSION_DISTRIBUTIONS = (
    "accelerate",
    "gluonts",
    "granite-tsfm",
    "huggingface-hub",
    "kairos",
    "lag-llama",
    "numpy",
    "pandas",
    "tabpfn",
    "tabpfn-time-series",
    "timeagi",
    "timesfm",
    "tirex-ts",
    "torch",
    "toto-2",
    "transformers",
    "uni2ts",
)


def _typed_unavailable(reason_code: str) -> dict[str, str]:
    return {"status": "unavailable", "reason_code": reason_code}


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    return device


def _requested_revision(manifest: TSFMManifest) -> str:
    revision = manifest.runtime_options.get(
        "revision", manifest.runtime_options.get("model_revision", "main")
    )
    return str(revision)


def _repository_revision(manifest: TSFMManifest) -> str:
    return _requested_revision(manifest)


def _environment_for_device(device: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["NA_TSFM_DEVICE"] = device
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return environment


def _read_package_versions(
    command: WorkerCommand, parent_environment: Mapping[str, str]
) -> dict[str, str]:
    query = (
        "import importlib.metadata as m,json;"
        f"names={_VERSION_DISTRIBUTIONS!r};"
        "out={};"
        "[(out.setdefault(n,m.version(n)) if n not in out else None) "
        "for n in names if any(d.metadata.get('Name','').lower()==n.lower() "
        "for d in m.distributions())];"
        "print(json.dumps(out,sort_keys=True))"
    )
    completed = subprocess.run(
        [command.argv[0], "-c", query],
        shell=False,
        env=dict(controlled_worker_environment(parent_environment)),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("package version query failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(version, str)
        for name, version in payload.items()
    ):
        raise RuntimeError("package version query returned invalid data")
    return dict(sorted(payload.items()))


def _resolve_checkpoint_revision(
    command: WorkerCommand,
    manifest: TSFMManifest,
    parent_environment: Mapping[str, str],
) -> str:
    query = (
        "from huggingface_hub import HfApi;import sys;"
        "print(HfApi().model_info(sys.argv[1],revision=sys.argv[2]).sha)"
    )
    completed = subprocess.run(
        [
            command.argv[0],
            "-c",
            query,
            manifest.checkpoint,
            _repository_revision(manifest),
        ],
        shell=False,
        env=dict(controlled_worker_environment(parent_environment)),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or _REVISION.fullmatch(revision) is None:
        raise RuntimeError("checkpoint revision query failed")
    return revision


def _base_report(manifest: TSFMManifest, device: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_id": manifest.method_id,
        "worker_environment": manifest.worker_environment,
        "adapter": manifest.adapter,
        "status": "unavailable",
        "reason_code": "not_run",
        "message": "",
        "checkpoint": manifest.checkpoint,
        "requested_revision": _requested_revision(manifest),
        "checkpoint_revision": _typed_unavailable("revision_not_resolved"),
        "package_versions": _typed_unavailable("versions_not_collected"),
        "requested_device": device,
        "device": _typed_unavailable("device_not_reported"),
        "latency_seconds": _typed_unavailable("latency_not_measured"),
        "peak_memory_bytes": _typed_unavailable("peak_memory_not_reported"),
        "horizon": SMOKE_HORIZON,
        "history_length": len(SMOKE_HISTORY),
        "finite_output": False,
        "output_status": "not_produced",
        "output_length": 0,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    if not path.parent.is_dir():
        raise ValueError("smoke output parent directory does not exist")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _finish_report(
    report: dict[str, object], output_path: Path, redactor: SecretRedactor
) -> dict[str, object]:
    sanitized = redactor.sanitize_json(report)
    if not isinstance(sanitized, dict):
        raise RuntimeError("smoke report sanitization failed")
    _atomic_write_json(output_path, sanitized)
    return sanitized


def run_checkpoint_smoke(
    *,
    manifest_id: str,
    deployment_config: str | Path,
    device: str,
    output_path: str | Path,
    acknowledged_licenses: Sequence[str] = (),
    _deployment_loader: Callable[..., Any] = TSFMDeployment.load,
    _broker_factory: Callable[..., Any] = WorkerBroker,
    _version_reader: Callable[..., object] = _read_package_versions,
    _revision_reader: Callable[..., object] = _resolve_checkpoint_revision,
    _clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run the fixed smoke series through one exact reviewed worker binding."""

    requested_device = _validate_device(device)
    output = Path(output_path)
    registry = ManifestRegistry.load_default()
    manifest = registry.require(manifest_id)
    redactor = SecretRedactor.from_environment()
    report = _base_report(manifest, requested_device)

    if manifest.status != "experimental_unverified":
        report["reason_code"] = (
            manifest.reason_code
            if manifest.status == "unavailable"
            else "manifest_not_worker_executable"
        )
        return _finish_report(report, output, redactor)

    try:
        deployment = _deployment_loader(
            deployment_config,
            manifests=registry,
            acknowledged_licenses=acknowledged_licenses,
        )
    except ValueError:
        report["reason_code"] = "deployment_unavailable"
        report["message"] = "deployment configuration could not be loaded"
        return _finish_report(report, output, redactor)

    if (
        manifest.method_id not in deployment.enabled_manifest_ids
        and manifest.license_acknowledgement_required
    ):
        report["reason_code"] = "license_not_acknowledged"
        return _finish_report(report, output, redactor)
    command = deployment.commands.get(manifest.worker_environment)
    if command is None:
        report["reason_code"] = "worker_not_configured"
        return _finish_report(report, output, redactor)
    if manifest.method_id not in deployment.enabled_manifest_ids:
        report["reason_code"] = "worker_not_enabled"
        return _finish_report(report, output, redactor)

    parent_environment = _environment_for_device(requested_device)
    versions_available = False
    try:
        versions = _version_reader(command, parent_environment)
        if not isinstance(versions, Mapping) or not all(
            isinstance(name, str) and isinstance(version, str)
            for name, version in versions.items()
        ) or not versions:
            raise ValueError("invalid versions")
        report["package_versions"] = dict(sorted(versions.items()))
        versions_available = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
        report["package_versions"] = _typed_unavailable("version_query_unavailable")
    try:
        revision = _revision_reader(command, manifest, parent_environment)
        if not is_immutable_checkpoint_revision(revision):
            raise ValueError("invalid revision")
        report["checkpoint_revision"] = revision
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        report["checkpoint_revision"] = _typed_unavailable(
            "revision_resolution_unavailable"
        )
        report["reason_code"] = "checkpoint_revision_unavailable"
        report["message"] = "mandatory checkpoint-revision evidence unavailable"
        return _finish_report(report, output, redactor)

    request = WorkerRequest(
        request_id=f"checkpoint-smoke-{uuid.uuid4().hex}",
        provider=manifest.adapter,
        checkpoint=manifest.checkpoint,
        checkpoint_revision=revision,
        history=SMOKE_HISTORY,
        horizon=SMOKE_HORIZON,
        frequency=SMOKE_FREQUENCY,
        runtime_options=dict(manifest.runtime_options),
    )
    started = _clock()
    try:
        with _broker_factory(
            deployment.commands,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            parent_environment=parent_environment,
        ) as broker:
            response = broker.request(manifest.worker_environment, request)
    except RuntimeUnavailableError:
        response = WorkerResponse.failure(
            request.request_id,
            "unavailable",
            "broker_unavailable",
            "worker broker could not complete the smoke request",
        )
    except Exception:
        response = WorkerResponse.failure(
            request.request_id,
            "runtime_error",
            "smoke_runtime_error",
            "checkpoint smoke failed inside the worker boundary",
        )
    elapsed = _clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("smoke clock returned an invalid elapsed time")
    report["latency_seconds"] = round(elapsed, 6)

    response = redactor.sanitize_response(response)
    report["status"] = response.status
    if response.status == "success":
        report["reason_code"] = ""
        report["message"] = ""
    else:
        reason_code = response.reason_code
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason_code) is None:
            reason_code = "worker_failure"
        report["reason_code"] = reason_code
        report["message"] = f"worker reported {response.status}: {reason_code}"
    if response.status == "success":
        if len(response.values) != SMOKE_HORIZON:
            report["status"] = "runtime_error"
            report["reason_code"] = "invalid_forecast_horizon"
            report["message"] = "worker output did not match the requested horizon"
        else:
            report["finite_output"] = True
            report["output_status"] = "finite"
            report["output_length"] = len(response.values)
            response_device = response.metadata.get("device")
            if (
                isinstance(response_device, str)
                and _ACTUAL_DEVICE.fullmatch(response_device) is not None
            ):
                report["device"] = response_device
            peak_memory = response.metadata.get("peak_memory_bytes")
            if (
                isinstance(peak_memory, int)
                and not isinstance(peak_memory, bool)
                and peak_memory >= 0
            ):
                report["peak_memory_bytes"] = peak_memory
            observed_revision = response.metadata.get("checkpoint_revision")
            if not is_immutable_checkpoint_revision(observed_revision):
                report["status"] = "unavailable"
                report["reason_code"] = "checkpoint_revision_unavailable"
                report["message"] = (
                    "worker did not report valid loaded checkpoint-revision evidence"
                )
            elif observed_revision != revision:
                report["status"] = "unavailable"
                report["reason_code"] = "checkpoint_revision_mismatch"
                report["message"] = (
                    "worker loaded checkpoint revision did not match the resolved revision"
                )
            elif not versions_available:
                report["status"] = "unavailable"
                report["reason_code"] = "package_versions_unavailable"
                report["message"] = "mandatory package-version evidence unavailable"
            elif not isinstance(report["device"], str):
                report["status"] = "unavailable"
                report["reason_code"] = "device_unavailable"
                report["message"] = "mandatory execution-device evidence unavailable"
            elif not isinstance(report["peak_memory_bytes"], int):
                report["status"] = "unavailable"
                report["reason_code"] = "peak_memory_unavailable"
                report["message"] = "mandatory peak-memory evidence unavailable"
            elif (
                requested_device == "cpu" and report["device"] != "cpu"
            ) or (
                requested_device.startswith("cuda")
                and not str(report["device"]).startswith("cuda")
            ):
                report["status"] = "unavailable"
                report["reason_code"] = "device_mismatch"
                report["message"] = "execution device did not match the explicit request"
    return _finish_report(report, output, redactor)


def _argument_device(value: str) -> str:
    try:
        return _validate_device(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--workers-config", required=True, type=Path)
    parser.add_argument("--device", required=True, type=_argument_device)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    registry = ManifestRegistry.load_default()
    try:
        acknowledgements = parse_acknowledged_licenses(
            os.environ.get("NA_ACCEPT_MODEL_LICENSES", ""), registry
        )
        report = run_checkpoint_smoke(
            manifest_id=args.manifest_id,
            deployment_config=args.workers_config,
            device=args.device,
            output_path=args.output,
            acknowledged_licenses=tuple(acknowledgements),
        )
    except ValueError as error:
        parser.error(SecretRedactor.from_environment().redact_text(error))
    return {
        "success": 0,
        "unavailable": 2,
        "invalid_request": 3,
        "runtime_error": 4,
    }[str(report["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
