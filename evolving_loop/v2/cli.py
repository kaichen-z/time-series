"""Parallel Project 1 CLI; Public is a validation-only, non-learning boundary.

Shipped profile digests commit configuration intent, not installed adapters.
For each kernel_protocol or runtime_fingerprints key `component`, its preimage
is canonical_v2_bytes({"binding_kind": "config_intent", "component": component,
"configuration": base}), where base is the complete config payload with
kernel_protocol and runtime_fingerprints omitted. This fully specified recipe
avoids placeholder identities; production adapters must supply real executable
bindings in later projects. Hyperband and scheduler settings are intent only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
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
from .numerical_qd.config import load_numerical_qd_config
from .numerical_qd.persistence import NumericalQDRunStore
from .numerical_qd.runner import run_numerical_qd
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
        raise ValueError("production evolution requires Project 2+ adapters")
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


def _regular_canonical_input(path: Path) -> tuple[dict[str, object], bytes, Path]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"input must be an existing regular file: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"input must be a non-symlink regular file: {path}")
    resolved = path.resolve(strict=True)
    raw = path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(payload) is not dict or raw != canonical_v2_bytes(payload):
        raise ValueError(f"expected canonical V2 JSON object: {path}")
    return payload, raw, resolved


def _preflight_numerical_paths(inputs: tuple[Path, ...], output: Path) -> bool:
    if len({(path.stat().st_dev, path.stat().st_ino) for path in inputs}) != len(inputs):
        raise ValueError("Numerical input files must have distinct filesystem identities")
    absolute = output.absolute()
    for ancestor in (*reversed(absolute.parents), absolute):
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(mode):
            # macOS exposes the operating system's physical temporary directory
            # through /tmp -> /private/tmp. Accept only that root-owned alias;
            # all caller-controlled output symlinks still fail before creation.
            if ancestor == Path("/tmp") and stat.S_ISLNK(mode):
                resolved_tmp = ancestor.resolve(strict=True)
                if stat.S_ISDIR(resolved_tmp.lstat().st_mode):
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
    documents = tuple(
        Document(
            _exact_fields(item, _DOCUMENT_FIELDS, "Numerical document")["document_id"],
            item["content"],
        )
        for item in raw_documents
    )
    if any(not document.document_id or type(document.content) is not str for document in documents):
        raise ValueError("Numerical documents require string identity and content")
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


def _build_numerical_adapter(config, tasks, folds, operator_input_sha256s):
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
    materializer = NumericalPackageMaterializer(
        forecast_store=_DeterministicForecastStore(),
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
    )


def _numerical_evolve(
    config_path: Path, seed_path: Path, task_manifest_path: Path, output: Path
) -> dict[str, object]:
    loaded = tuple(
        _regular_canonical_input(path)
        for path in (config_path, seed_path, task_manifest_path)
    )
    payloads = tuple(item[0] for item in loaded)
    raw = tuple(item[1] for item in loaded)
    resolved = tuple(item[2] for item in loaded)
    resume = _preflight_numerical_paths(resolved, output)
    config = load_numerical_qd_config(config_path)
    seed = parse_numerical_supply_release(payloads[1])
    tasks, folds = _parse_task_manifest(payloads[2])
    input_sha256s = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in zip(
            ("config", "seed_supply", "task_manifest"), raw, strict=True
        )
    }
    adapter = _build_numerical_adapter(config, tasks, folds, input_sha256s)
    run_numerical_qd(
        output,
        config,
        seed,
        folds,
        adapter,
        llm_client=None,
        resume=resume,
    )
    return _read_canonical(output / "evaluation_complete.json")


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
            summary = _numerical_evolve(
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


__all__ = ["build_parser", "main"]
