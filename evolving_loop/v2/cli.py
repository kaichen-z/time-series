"""Evolution V2 command-line entrypoints and trusted input boundaries.

The Project 1 fake and validation-only Public commands remain closed to live
adapters.  ``numerical-evolve`` owns its separate, canonical Numerical Supply
inputs and delegates execution to the Project 2 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

from common.data import Task
from common.payload import strict_json_loads
from evolving_loop.data import ContextTask, Document
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest

from .bundle import EvolutionBundleV2
from .contracts import KernelProtocolCommitment, canonical_v2_bytes, load_v2_config
from .fakes import run_fake_kernel
from .numerical_qd.adapters import LegacyNumericalAdapter
from .numerical_qd.config import NumericalQDConfigV2
from .numerical_qd.persistence import NumericalQDRunStore
from .numerical_qd.runner import run_numerical_qd
from .path_safety import _system_tmp_alias, physical_system_tmp_path
from .store import write_once_json


_NUMERICAL_METHOD_SOURCE = '''def seasonal_naive(history, horizon, frequency):
    """Use for finite daily histories with stable local levels."""
    return [float(history[-1]) + 1.0] * horizon
'''
_NUMERICAL_SOURCE_SHA256 = hashlib.sha256(
    _NUMERICAL_METHOD_SOURCE.encode("utf-8")
).hexdigest()
_TASK_MANIFEST_FIELDS = ("schema_version", "fold_manifest", "train", "dev")
_TASK_FIELDS = (
    "task_id",
    "entity_name",
    "history_values",
    "future_values",
    "prediction_length",
    "frequency",
    "seasonal_period",
    "target_name",
    "target_description",
    "history_timestamps",
    "future_timestamps",
    "documents",
    "gt_evidence",
)
_DOCUMENT_FIELDS = ("document_id", "content")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolving_loop.v2")
    commands = parser.add_subparsers(dest="command", required=True)
    evolve = commands.add_parser(
        "evolve", help="execute or resume deterministic fake evolution"
    )
    evolve.add_argument("--config", required=True, type=Path)
    evolve.add_argument("--output-dir", required=True, type=Path)
    public = commands.add_parser(
        "public-evaluate", help="validate a frozen accepted Bundle; no Public evaluator"
    )
    public.add_argument("--bundle", required=True, type=Path)
    public.add_argument("--output-dir", required=True, type=Path)
    numerical = commands.add_parser(
        "numerical-evolve", help="execute or resume Numerical Supply self-evolution"
    )
    numerical.add_argument("--config", required=True, type=Path)
    numerical.add_argument("--seed-supply", required=True, type=Path)
    numerical.add_argument("--task-manifest", required=True, type=Path)
    numerical.add_argument("--output-dir", required=True, type=Path)
    return parser


def _read_canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if not isinstance(payload, dict) or raw != canonical_v2_bytes(payload):
        raise ValueError(f"expected canonical V2 JSON object: {path}")
    return payload


def _evolve(config_path: Path, output: Path) -> dict[str, object]:
    config = load_v2_config(config_path)
    if config.profile == "public":
        raise ValueError("public profile requires public-evaluate")
    if config.runner == "production":
        raise ValueError("production evolution requires Project 3 adapters")
    if config.profile != "smoke" or set(config.enabled_mutation_scopes) != {
        "numerical",
        "retrieval",
    }:
        raise ValueError(
            "deterministic fake requires smoke with exactly Numerical and Retrieval scopes"
        )
    if output.exists() and not output.is_dir():
        raise ValueError(
            "output must be an empty directory or the exact resumable V2 run"
        )
    resume = output.is_dir() and any(output.iterdir())
    result = run_fake_kernel(output, config, resume=resume)
    return _read_canonical(result.completion_path)


def _filesystem_contains(root: Path, path: Path) -> bool:
    """Compare existing ancestors by identity, including case aliases.

    A new output can have several nonexistent parents. Continue past them to
    find any existing ancestor that aliases the source directory. The reverse
    check covers an existing output that contains or aliases the source.
    """
    if not root.exists():
        return False
    return any(
        ancestor.exists() and root.samefile(ancestor)
        for ancestor in (path, *path.parents)
    )


def _exact_fields(value: object, fields: tuple[str, ...], name: str) -> dict:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{name} must use the exact schema")
    return value


def _regular_canonical_input(
    path: Path,
) -> tuple[dict[str, object], bytes, Path, tuple[int, int], str]:
    """Read one canonical input through a no-follow descriptor chain.

    Every caller-controlled component is opened as a directory with
    ``O_NOFOLLOW``.  The sole exception is the operating system's fixed macOS
    ``/tmp -> /private/tmp`` alias, which is converted to its physical path
    before traversal.  The same leaf descriptor supplies bytes, identity and
    canonical digest input; no path read occurs after admission.
    """
    absolute = physical_system_tmp_path(path.absolute())
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:-1]:
            next_descriptor = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        leaf = os.open(absolute.name, flags, dir_fd=descriptor)
        try:
            before = os.fstat(leaf)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"input must be a non-symlink regular file: {path}")
            chunks = []
            while True:
                chunk = os.read(leaf, 131_072)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(leaf)
        finally:
            os.close(leaf)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(
            f"input path must use real directories and a non-symlink regular file: {path}"
        ) from error
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError(f"input changed while being read: {path}")
    raw = b"".join(chunks)
    sha256 = hashlib.sha256(raw).hexdigest()
    payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(payload) is not dict or raw != canonical_v2_bytes(payload):
        raise ValueError(f"expected canonical V2 JSON object: {path}")
    return payload, raw, absolute, (before.st_dev, before.st_ino), sha256


def _preflight_numerical_paths(
    inputs: tuple[Path, ...], output: Path, *, input_identities: tuple[tuple[int, int], ...] | None = None
) -> bool:
    identities = input_identities or tuple(
        (path.stat().st_dev, path.stat().st_ino) for path in inputs
    )
    if len(set(identities)) != len(inputs):
        raise ValueError("Numerical input files must have distinct filesystem identities")
    absolute = output.absolute()
    for ancestor in (*reversed(absolute.parents), absolute):
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(mode):
            if _system_tmp_alias(ancestor, mode):
                continue
            raise ValueError("Numerical output and ancestors must be real directories")
    resolved_output = absolute.resolve(strict=False)
    for source in inputs:
        if (
            source.is_relative_to(resolved_output)
            or resolved_output.is_relative_to(source.parent)
        ):
            raise ValueError("Numerical output and inputs must not overlap in either direction")
    if absolute.exists():
        if not absolute.is_dir():
            raise ValueError("Numerical output must be a new directory or exact resumable run")
        resume = any(absolute.iterdir())
    else:
        resume = False
    if resume:
        NumericalQDRunStore(absolute)._verify_root_paths(require_checkpoint=True)
    else:
        NumericalQDRunStore.preflight_fresh(absolute)
    return resume


def _finite_values(value: object, field: str, *, nonempty: bool) -> tuple[float, ...]:
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{field} must be a {'nonempty ' if nonempty else ''}list")
    if any(type(item) not in {int, float} or not math.isfinite(item) for item in value):
        raise ValueError(f"{field} must contain finite numbers")
    return tuple(float(item) for item in value)


def _strings(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(value)


def _parse_numerical_task(value: object) -> ContextTask:
    task = _exact_fields(value, _TASK_FIELDS, "Numerical task")
    for field in ("task_id", "entity_name", "frequency", "target_name", "target_description"):
        if type(task[field]) is not str or not task[field]:
            raise ValueError(f"Numerical task {field} must be a nonempty string")
    horizon = task["prediction_length"]
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Numerical task prediction_length must be a positive integer")
    history = _finite_values(task["history_values"], "history_values", nonempty=True)
    future = _finite_values(task["future_values"], "future_values", nonempty=True)
    if len(future) != horizon:
        raise ValueError("Numerical task future length must equal prediction_length")
    seasonal = task["seasonal_period"]
    if seasonal is not None and (type(seasonal) is not str or not seasonal):
        raise ValueError("Numerical task seasonal_period must be null or nonempty text")
    history_timestamps = _strings(task["history_timestamps"], "history_timestamps")
    future_timestamps = _strings(task["future_timestamps"], "future_timestamps")
    if len(history_timestamps) != len(history) or len(future_timestamps) != horizon:
        raise ValueError("Numerical task timestamp lengths must match the series")
    raw_documents = task["documents"]
    if type(raw_documents) is not list:
        raise ValueError("Numerical task documents must be a list")
    document_payloads = tuple(
        _exact_fields(item, _DOCUMENT_FIELDS, "Numerical document")
        for item in raw_documents
    )
    if any(
        type(item["document_id"]) is not str
        or not item["document_id"]
        or type(item["content"]) is not str
        for item in document_payloads
    ):
        raise ValueError("Numerical documents require string identity and content")
    documents = tuple(
        Document(item["document_id"], item["content"]) for item in document_payloads
    )
    evidence = _strings(task["gt_evidence"], "gt_evidence")
    return ContextTask(
        numeric=Task(
            task_id=task["task_id"],
            history_values=history,
            future_values=future,
            prediction_length=horizon,
            frequency=task["frequency"],
            seasonal_period=seasonal,
            entity_name=task["entity_name"],
        ),
        target_name=task["target_name"],
        target_description=task["target_description"],
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        documents=documents,
        gt_evidence=evidence,
        labels_public=True,
    )


def _parse_task_manifest(payload: Mapping[str, object]):
    manifest = _exact_fields(payload, _TASK_MANIFEST_FIELDS, "Numerical task manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Numerical task manifest schema_version must be exactly 1")
    if type(manifest["train"]) is not list or type(manifest["dev"]) is not list:
        raise ValueError("Numerical task manifest train/dev must be lists")
    train = tuple(_parse_numerical_task(value) for value in manifest["train"])
    dev = tuple(_parse_numerical_task(value) for value in manifest["dev"])
    if len(train) != 80 or len(dev) != 20:
        raise ValueError("Numerical task manifest requires exactly Train80 and Dev20")
    ids = tuple(task.numeric.task_id for task in (*train, *dev))
    if len(ids) != len(set(ids)):
        raise ValueError("Numerical task manifest task identities must be unique")
    fold_payload = manifest["fold_manifest"]
    if not isinstance(fold_payload, Mapping) or type(fold_payload.get("seed")) is not int:
        raise ValueError("Numerical task manifest requires an exact fold manifest")
    folds = build_group_fold_manifest(
        tuple(task.numeric for task in train), seed=fold_payload["seed"]
    )
    if folds.to_payload() != fold_payload:
        raise ValueError("Numerical task fold manifest does not bind exact Train80 groups")
    return (*train, *dev), folds


class _DeterministicForecastStore:
    """CPU-only Host store used when no external runtime is configured."""

    resource_kinds: tuple[str, ...] = ()

    def forecast(self, name, history, horizon, frequency):
        del frequency
        if name == "toto_2_0":
            # The reviewed seed anchor is intentionally the zero-offset
            # reference used by the existing Numerical V2 host fixtures.
            # Keeping it distinct from the source-provided alternatives gives
            # Build folds meaningful, independently fitted policy choices.
            offset = 0.0
        elif name == "seasonal_naive":
            offset = 1.0
        else:
            raise KeyError(f"unknown deterministic Numerical method: {name}")
        return (float(history[-1]) + offset,) * horizon


def _build_numerical_adapter(
    config, tasks, folds, operator_input_sha256s, *, host_runtime=None
):
    screening = ScreeningPolicy(
        tuple(
            ScreeningEntry(name, family, "keep", ApplicabilityPolicy(), "reviewed")
            for name, family in (
                ("toto_2_0", "tsfm"),
                ("seasonal_naive", "statistical"),
            )
        ),
        ("toto_2_0",),
    )
    if config.profile == "smoke":
        forecast_store = _DeterministicForecastStore()
        resource_reporter = None
        resource_reporter_sha256 = None
        resource_kinds = ()
    else:
        forecast_store = getattr(host_runtime, "forecast_store", None)
        if not callable(getattr(forecast_store, "forecast", None)):
            raise ValueError("pilot/formal requires a configured Host forecast runtime")
        resource_reporter = getattr(host_runtime, "resource_reporter", None)
        resource_reporter_sha256 = getattr(host_runtime, "resource_reporter_sha256", None)
        resource_kinds = tuple(getattr(host_runtime, "resource_kinds", ()))
    materializer = NumericalPackageMaterializer(
        forecast_store=forecast_store,
        screening_policy=screening,
        fold_manifest=folds,
        original_tasks=tasks,
        source_fingerprints={"dictionary": _NUMERICAL_SOURCE_SHA256},
        runtime_fingerprints=dict(config.runtime_fingerprints),
    )
    return LegacyNumericalAdapter(
        materializer=materializer,
        tasks=tasks,
        fold_manifest=folds,
        sources={_NUMERICAL_SOURCE_SHA256: _NUMERICAL_METHOD_SOURCE},
        operator_input_sha256s=operator_input_sha256s,
        resource_kinds=resource_kinds,
        resource_reporter=resource_reporter,
        resource_reporter_sha256=resource_reporter_sha256,
    )


def _numerical_evolve(
    config_path: Path, seed_path: Path, task_manifest_path: Path, output: Path,
    *, host_runtime=None, llm_client=None,
) -> dict[str, object]:
    loaded = tuple(
        _regular_canonical_input(path)
        for path in (config_path, seed_path, task_manifest_path)
    )
    payloads = tuple(item[0] for item in loaded)
    resolved = tuple(item[2] for item in loaded)
    identities = tuple(item[3] for item in loaded)
    resume = _preflight_numerical_paths(resolved, output, input_identities=identities)
    config = NumericalQDConfigV2.from_payload(payloads[0])
    seed = parse_numerical_supply_release(payloads[1])
    tasks, folds = _parse_task_manifest(payloads[2])
    input_sha256s = dict(
        zip(
            ("config", "seed_supply", "task_manifest"),
            (item[4] for item in loaded),
            strict=True,
        )
    )
    adapter = _build_numerical_adapter(
        config, tasks, folds, input_sha256s, host_runtime=host_runtime
    )
    run_numerical_qd(
        output,
        config,
        seed,
        folds,
        adapter,
        llm_client=llm_client if llm_client is not None else getattr(host_runtime, "llm_client", None),
        resume=resume,
    )
    return _read_canonical(output / "evaluation_complete.json")


def numerical_evolve(
    config_path: Path,
    seed_path: Path,
    task_manifest_path: Path,
    output: Path,
    *,
    host_runtime=None,
    llm_client=None,
) -> dict[str, object]:
    """Run Numerical QD with an explicitly configured trusted Host runtime.

    Smoke intentionally owns its fixture store.  Pilot/formal callers supply
    their existing forecast/runtime boundary here; an absent LLM remains a
    proposer-only hybrid fallback inside the runner.
    """
    return _numerical_evolve(
        config_path,
        seed_path,
        task_manifest_path,
        output,
        host_runtime=host_runtime,
        llm_client=llm_client,
    )


def _public_evaluate(bundle_path: Path, output: Path) -> dict[str, object]:
    source_path = bundle_path.resolve(strict=True)
    if source_path.name != "accepted_bundle.json":
        raise ValueError("bundle must be a source V2 run's accepted_bundle.json")
    source = source_path.parent
    destination = output.resolve()
    if _filesystem_contains(source, destination) or _filesystem_contains(
        destination, source
    ):
        raise ValueError(
            "Public output and source run must not overlap in either direction"
        )
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("Public output must be an empty new directory")

    bundle = EvolutionBundleV2.from_payload(_read_canonical(source_path))
    if bundle.acceptance_evidence_sha256 is None:
        raise ValueError("Public validation requires a sealed accepted Bundle")
    identity = bundle.fingerprint()
    archived = _read_canonical(source / "archive" / "objects" / f"{identity}.json")
    if canonical_v2_bytes(archived) != bundle.canonical_bytes():
        raise ValueError(
            "accepted Bundle fingerprint does not match the archived object"
        )
    manifest = _read_canonical(source / "run_manifest.json")
    if manifest.get("system") != "evolution_v2":
        raise ValueError("source must be an Evolution V2 run")
    protocol = KernelProtocolCommitment.from_payload(manifest.get("kernel_protocol"))
    if protocol.fingerprint() != bundle.protocol_fingerprint or manifest.get(
        "runtime_fingerprints"
    ) != dict(bundle.runtime_fingerprints):
        raise ValueError("accepted Bundle protocol/runtime does not match source run")

    # No source store, kernel, evaluator, data loader, or ledger is instantiated.
    # Evidence is a validated SHA reference only; later projects verify replay.
    completion = {
        "status": "validated_only_no_public_evaluator",
        "bundle_sha256": identity,
        "public_test_accessed": False,
    }
    write_once_json(
        destination / "run_manifest.json",
        {
            "system": "evolution_v2_public",
            "mode": "validation_only",
            "bundle_sha256": identity,
            "acceptance_evidence_sha256": bundle.acceptance_evidence_sha256,
            "protocol_fingerprint": bundle.protocol_fingerprint,
            "runtime_fingerprints": dict(bundle.runtime_fingerprints),
            "public_test_accessed": False,
        },
    )
    write_once_json(destination / "objects" / f"{identity}.json", bundle.to_payload())
    write_once_json(destination / "evaluation_complete.json", completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evolve":
            summary = _evolve(args.config, args.output_dir)
        elif args.command == "public-evaluate":
            summary = _public_evaluate(args.bundle, args.output_dir)
        else:
            summary = numerical_evolve(
                args.config,
                args.seed_supply,
                args.task_manifest,
                args.output_dir,
            )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Evolution V2: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(canonical_v2_bytes(summary).decode("utf-8"))
    return 0


__all__ = ["build_parser", "main", "numerical_evolve"]
