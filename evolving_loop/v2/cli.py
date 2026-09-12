"""Evolution V2 command-line entrypoints and trusted input boundaries.

The Project 1 fake and validation-only Public commands remain closed to live
adapters.  ``numerical-evolve`` owns its separate, canonical Numerical Supply
inputs and delegates execution to the Project 2 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from common.data import Task
from common.llm import FakeLLMClient, LLMClient
from common.payload import strict_json_loads
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.data import ContextTask, Document
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.package_numerical_supply import (
    bound_numerical_package,
    build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)
from numerical_agent.evolution.champion import (
    champion_fingerprint,
    parse_champion_release,
)
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage,
    RankedNumericalForecast,
)
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    SelectionDecision,
)
from numerical_agent.evolution.screening import profile_task
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest

from .bundle import EvolutionBundleV2
from .contracts import (
    EvolutionV2Config,
    KernelProtocolCommitment,
    canonical_v2_bytes,
    load_v2_config,
    require_sha256,
)
from .cooperative import (
    CooperativeArtifactCatalog,
    CooperativeConfigV2,
    CooperativePipelineAdapter,
    DecisionCoordinateAdapter,
    DecisionModuleV2,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
    RetrievalModuleV2,
    run_cooperative_evolution,
)
from .fakes import run_fake_kernel
from .numerical_qd.adapters import (
    FrozenNumericalArtifactsV2,
    LegacyNumericalAdapter,
    import_numerical_seed,
)
from .numerical_qd.config import NumericalQDConfigV2
from .numerical_qd.persistence import NumericalQDRunStore
from .numerical_qd.runner import run_numerical_qd
from .path_safety import _system_tmp_alias, physical_system_tmp_path
from .store import write_once_json
from .protocol.cli import add_protocol_parsers, dispatch_protocol
from .real.contracts import RealEvolutionManifestV2
from .real.host import build_real_host
from .real.runner import build_real_stage_ports, run_real_evolution


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
_COOPERATIVE_TASK_MANIFEST_FIELDS = ("schema_version", "train", "dev")
_EVOLUTION_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "seed",
        "scheduler",
        "enabled_mutation_scopes",
        "archive_capacities",
        "hyperband",
        "runtime_fingerprints",
        "kernel_protocol",
        "hard_limit_seconds",
        "finalization_reserve_fraction",
        "runner",
    }
)
_COOPERATIVE_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "control",
        "max_steps",
        "children_per_step",
        "discount",
        "task_cost_weight",
        "metric_cap",
        "acceptance_tolerance",
        "resource_ceilings",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolving_loop.v2")
    commands = parser.add_subparsers(dest="command", required=True)
    evolve = commands.add_parser(
        "evolve", help="execute or resume deterministic fake evolution"
    )
    evolve.add_argument("--config", required=True, type=Path)
    evolve.add_argument("--seed-supply", type=Path)
    evolve.add_argument("--task-manifest", type=Path)
    evolve.add_argument("--retrieval-release", type=Path)
    evolve.add_argument("--decision-policy", type=Path)
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
    numerical.add_argument("--task-local-evidence", type=Path)
    numerical.add_argument("--task-local-dictionary", type=Path)
    numerical.add_argument("--legacy-bootstrap", action="store_true")
    real = commands.add_parser(
        "real-evolve", help="execute or resume the bounded real P2→P5 evolution"
    )
    real.add_argument("--manifest", required=True, type=Path)
    real.add_argument("--output-dir", required=True, type=Path)
    real.add_argument(
        "--authority-root", type=Path,
        help="read-only data authority root (defaults to the shared checkout)",
    )
    add_protocol_parsers(commands)
    return parser


def _read_canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if not isinstance(payload, dict) or raw != canonical_v2_bytes(payload):
        raise ValueError(f"expected canonical V2 JSON object: {path}")
    return payload


def _read_evolve_config(path: Path) -> tuple[dict[str, object], bool]:
    """Read dispatch config once while preserving the legacy formatted profile."""
    raw = path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(payload) is not dict:
        raise ValueError(f"Evolution V2 config must be a JSON object: {path}")
    return payload, raw == canonical_v2_bytes(payload)


def _evolve(
    config_path: Path,
    output: Path,
    *,
    config_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = (
        load_v2_config(config_path)
        if config_payload is None
        else EvolutionV2Config.from_payload(config_payload)
    )
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


def _default_authority_root(code_root: Path) -> Path:
    """Locate the shared checkout without changing it or reading its state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=code_root, check=True, capture_output=True, text=True,
        )
        common = Path(result.stdout.strip()).resolve(strict=True)
        return common.parent
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot locate the shared authority checkout") from error


def _real_evolve(manifest_path: Path, output: Path, *, authority_root: Path | None) -> dict[str, object]:
    """Build and close the real Host around one immutable root invocation."""
    manifest = RealEvolutionManifestV2.from_payload(_read_canonical(manifest_path))
    code_root = Path(__file__).resolve().parents[2]
    authority = (authority_root if authority_root is not None else _default_authority_root(code_root)).resolve(strict=True)
    if not authority.is_dir():
        raise ValueError("authority root must be a real directory")
    destination = output.resolve()
    declared = [
        (
            code_root
            if row.role in {"numerical_source_seed", "source_seed"}
            else authority
        )
        / row.relative_path
        for row in manifest.files
    ]
    for row in manifest.runtime_locations:
        if row.role == "codex_cli" and row.relative_path == "codex":
            executable = shutil.which("codex")
            if executable is None:
                raise ValueError("real Host codex_cli identity target is unavailable")
            declared.append(Path(executable).resolve())
        else:
            declared.append(
                (code_root if row.role in {"task_loader", "forecast_store"} else authority)
                / row.relative_path
            )
    if any(
        _filesystem_contains(path, destination) or _filesystem_contains(destination, path)
        for path in declared
    ):
        raise ValueError("real output must not overlap a declared input or runtime")
    host = build_real_host(manifest, repo_root=authority, code_root=code_root, output_dir=destination)
    try:
        ports = build_real_stage_ports(host, manifest=manifest, repo_root=authority)
        return run_real_evolution(destination, manifest, ports).to_payload()
    finally:
        host.close()


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
    if ".." in path.parts:
        raise ValueError(f"input path must not contain '..' components: {path}")
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
    inputs: tuple[Path, ...],
    output: Path,
    *,
    input_identities: tuple[tuple[int, int], ...] | None = None,
) -> bool:
    identities = input_identities or tuple(
        (path.stat().st_dev, path.stat().st_ino) for path in inputs
    )
    if len(set(identities)) != len(inputs):
        raise ValueError(
            "Numerical input files must have distinct filesystem identities"
        )
    absolute = output.absolute()
    resolved_output = absolute.resolve(strict=False)
    for source in inputs:
        if source.is_relative_to(resolved_output) or resolved_output.is_relative_to(
            source.parent
        ):
            raise ValueError(
                "Numerical output and inputs must not overlap in either direction"
            )
    return _preflight_numerical_output(absolute)


def _preflight_numerical_output(output: Path) -> bool:
    """Admit a fresh/resumable output when inputs are verified payloads."""
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
    if absolute.exists():
        if not absolute.is_dir():
            raise ValueError(
                "Numerical output must be a new directory or exact resumable run"
            )
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
    for field in (
        "task_id",
        "entity_name",
        "frequency",
        "target_name",
        "target_description",
    ):
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
    if (
        not isinstance(fold_payload, Mapping)
        or type(fold_payload.get("seed")) is not int
    ):
        raise ValueError("Numerical task manifest requires an exact fold manifest")
    folds = build_group_fold_manifest(
        tuple(task.numeric for task in train), seed=fold_payload["seed"]
    )
    if folds.to_payload() != fold_payload:
        raise ValueError(
            "Numerical task fold manifest does not bind exact Train80 groups"
        )
    return (*train, *dev), folds


def _parse_cooperative_task_manifest(
    payload: Mapping[str, object],
) -> dict[str, tuple[ContextTask, ...]]:
    """Parse only the compact 4/1 cooperative task schema.

    This is intentionally separate from the Project 2 Train80/Dev20 parser.
    """
    manifest = _exact_fields(
        payload,
        _COOPERATIVE_TASK_MANIFEST_FIELDS,
        "Cooperative task manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Cooperative task manifest schema_version must be exactly 1")
    if type(manifest["train"]) is not list or type(manifest["dev"]) is not list:
        raise ValueError("Cooperative task manifest train/dev must be lists")
    train = tuple(_parse_numerical_task(value) for value in manifest["train"])
    dev = tuple(_parse_numerical_task(value) for value in manifest["dev"])
    if len(train) != 4 or len(dev) != 1:
        raise ValueError("Cooperative task manifest requires exactly Train4 and Dev1")
    ids = tuple(task.numeric.task_id for task in (*train, *dev))
    if len(ids) != len(set(ids)):
        raise ValueError("Cooperative task identities must be unique")
    return {"train": train, "dev": dev}


class _DeterministicForecastStore:
    """CPU-only Host store used when no external runtime is configured."""

    identity_hash = hashlib.sha256(
        b"evolution-v2-numerical-qd-smoke-forecast-store-v1"
    ).hexdigest()
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
    config, tasks, folds, operator_input_sha256s, *, host_runtime=None,
    seed_release=None, task_local_evidence_path=None, task_local_dictionary=None,
    legacy_bootstrap=False,
):
    from numerical_agent.evolution.filtering import FilterDictionary, parse_filter_source
    from numerical_agent.evolution.screening import ApplicabilityClause
    from numerical_agent.evolution.task_shortlist import _dictionary_hash
    from numerical_agent.run_task_local_ensemble_evolution import load_task_local_evidence_bundle
    task_local_evidence_path = task_local_evidence_path or getattr(host_runtime, "task_local_evidence_path", None)
    task_local_dictionary = task_local_dictionary or getattr(host_runtime, "task_local_dictionary", None)
    if seed_release is not None and seed_release.schema_version == 2 and not legacy_bootstrap and task_local_evidence_path is None:
        raise ValueError("schema-2 production requires Task 4 evidence; legacy/bootstrap must be explicit")
    evidence = None
    dictionary_sha = _NUMERICAL_SOURCE_SHA256
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
    if task_local_evidence_path is not None:
        if isinstance(task_local_dictionary, (str, Path)):
            task_local_dictionary = parse_filter_source(Path(task_local_dictionary).read_text(encoding="utf-8"))
        if type(task_local_dictionary) is not FilterDictionary:
            raise ValueError("Task 4 evidence requires the actual reviewed Dictionary")
        dictionary_sha = _dictionary_hash(task_local_dictionary)
        if seed_release is None or seed_release.schema_version != 2 or seed_release.source_fingerprints.get("dictionary") != dictionary_sha:
            raise ValueError("Task 4 Dictionary differs from schema-2 supply identity")
        evidence = load_task_local_evidence_bundle(Path(task_local_evidence_path), dictionary_sha256=dictionary_sha)
        anchor_names = {value[0].candidate_names[0] for value in evidence.by_task.values()}
        if len(anchor_names) != 1:
            raise ValueError("Task 4 evidence requires one protected Anchor")
        screening = ScreeningPolicy(tuple(ScreeningEntry(entry.name, entry.family, entry.status,
            ApplicabilityPolicy((ApplicabilityClause(entry.applicability),)) if entry.applicability else ApplicabilityPolicy(),
            entry.reason) for entry in task_local_dictionary.entries), tuple(anchor_names))
    if config.profile == "smoke":
        forecast_store = _DeterministicForecastStore()
        resource_reporter = None
        resource_reporter_sha256 = None
        resource_kinds = ()
    else:
        forecast_store = getattr(host_runtime, "forecast_store", None)
        if not callable(getattr(forecast_store, "forecast", None)):
            raise ValueError("pilot/formal requires a configured Host forecast runtime")
        require_sha256(
            getattr(forecast_store, "identity_hash", None),
            "Host forecast store identity",
        )
        resource_reporter = getattr(host_runtime, "resource_reporter", None)
        resource_reporter_sha256 = getattr(
            host_runtime, "resource_reporter_sha256", None
        )
        resource_kinds = tuple(getattr(host_runtime, "resource_kinds", ()))
    materializer = NumericalPackageMaterializer(
        forecast_store=forecast_store,
        screening_policy=screening,
        fold_manifest=folds,
        original_tasks=tasks,
        source_fingerprints={"dictionary": dictionary_sha},
        runtime_fingerprints=dict(config.runtime_fingerprints),
    )
    return LegacyNumericalAdapter(
        materializer=materializer,
        tasks=tasks,
        fold_manifest=folds,
        sources=getattr(host_runtime, "sources", {_NUMERICAL_SOURCE_SHA256: _NUMERICAL_METHOD_SOURCE}),
        task_local_evidence=evidence,
        legacy_bootstrap=legacy_bootstrap,
        operator_input_sha256s=operator_input_sha256s,
        resource_kinds=resource_kinds,
        resource_reporter=resource_reporter,
        resource_reporter_sha256=resource_reporter_sha256,
    )


def _numerical_evolve(
    config_path: Path,
    seed_path: Path,
    task_manifest_path: Path,
    output: Path,
    *,
    host_runtime=None,
    llm_client=None,
    task_local_evidence_path=None, task_local_dictionary=None, legacy_bootstrap=False,
) -> dict[str, object]:
    loaded = tuple(
        _regular_canonical_input(path)
        for path in (config_path, seed_path, task_manifest_path)
    )
    payloads = tuple(item[0] for item in loaded)
    resolved = tuple(item[2] for item in loaded)
    identities = tuple(item[3] for item in loaded)
    resume = _preflight_numerical_paths(resolved, output, input_identities=identities)
    input_sha256s = dict(
        zip(
            ("config", "seed_supply", "task_manifest"),
            (item[4] for item in loaded),
            strict=True,
        )
    )
    return _run_numerical_payloads(
        payloads[0], payloads[1], payloads[2], output,
        input_sha256s=input_sha256s, host_runtime=host_runtime,
        llm_client=llm_client, resume=resume,
        task_local_evidence_path=task_local_evidence_path, task_local_dictionary=task_local_dictionary,
        legacy_bootstrap=legacy_bootstrap,
    )


def _run_numerical_payloads(
    config_payload: Mapping[str, object],
    seed_payload: Mapping[str, object],
    task_manifest_payload: Mapping[str, object],
    output: Path,
    *,
    input_sha256s: Mapping[str, str],
    host_runtime: object,
    llm_client: LLMClient | None,
    resume: bool,
    task_local_evidence_path=None, task_local_dictionary=None, legacy_bootstrap=False,
) -> dict[str, object]:
    """Shared typed parsing and adapter construction for both input seams."""
    config = NumericalQDConfigV2.from_payload(config_payload)
    # Detach derived configuration from caller-owned mappings before execution.
    config = NumericalQDConfigV2.from_payload(config.to_payload())
    seed = parse_numerical_supply_release(seed_payload)
    task_manifest = strict_json_loads(
        canonical_v2_bytes(task_manifest_payload).decode("utf-8"),
        context="Numerical task manifest payload",
    )
    tasks, folds = _parse_task_manifest(task_manifest)
    adapter = _build_numerical_adapter(
        config, tasks, folds, input_sha256s, host_runtime=host_runtime, seed_release=seed,
        task_local_evidence_path=task_local_evidence_path, task_local_dictionary=task_local_dictionary,
        legacy_bootstrap=legacy_bootstrap,
    )
    run_numerical_qd(
        output,
        config,
        seed,
        folds,
        adapter,
        llm_client=(
            llm_client
            if llm_client is not None
            else getattr(host_runtime, "llm_client", None)
        ),
        resume=resume,
    )
    return _read_canonical(output / "evaluation_complete.json")


def numerical_evolve_payload(
    config_payload: Mapping[str, object],
    seed_payload: Mapping[str, object],
    task_manifest_payload: Mapping[str, object],
    output: Path,
    *,
    input_sha256s: Mapping[str, str],
    host_runtime: object,
    llm_client: LLMClient,
    task_local_evidence_path=None, task_local_dictionary=None, legacy_bootstrap=False,
) -> dict[str, object]:
    """Run Numerical QD from Host-verified payloads without path overlap rules."""
    resume = _preflight_numerical_output(Path(output))
    return _run_numerical_payloads(
        config_payload,
        seed_payload,
        task_manifest_payload,
        Path(output),
        input_sha256s=input_sha256s,
        host_runtime=host_runtime,
        llm_client=llm_client,
        resume=resume,
        task_local_evidence_path=task_local_evidence_path, task_local_dictionary=task_local_dictionary,
        legacy_bootstrap=legacy_bootstrap,
    )


def numerical_evolve(
    config_path: Path,
    seed_path: Path,
    task_manifest_path: Path,
    output: Path,
    *,
    host_runtime=None,
    llm_client=None,
    task_local_evidence_path=None, task_local_dictionary=None, legacy_bootstrap=False,
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
        task_local_evidence_path=task_local_evidence_path, task_local_dictionary=task_local_dictionary,
        legacy_bootstrap=legacy_bootstrap,
    )


def _cooperative_ranked(
    rank: int, name: str, family: str, forecast: tuple[float, ...]
) -> RankedNumericalForecast:
    diagnostics = CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=0.1,
        fold_forecasts=(forecast,) * 3,
        fold_truths=(forecast,) * 3,
        median_smae=0.1,
        recent_smae=0.1,
        worst_smae=0.1,
        median_srmse=0.1,
        recent_srmse=0.1,
        worst_srmse=0.1,
        worst_smae_raw=0.1,
        worst_srmse_raw=0.1,
    )
    return RankedNumericalForecast(rank, name, family, forecast, diagnostics)


def _cooperative_package(task, release, *, anchor_offset):
    horizon = task.numeric.prediction_length
    anchor_values = (float(task.numeric.history_values[-1]) + anchor_offset,) * horizon
    alternative_values = (float(task.numeric.history_values[-1]) + 1.0,) * horizon
    anchor = _cooperative_ranked(1, "safe_anchor", "tsfm", anchor_values)
    alternative = _cooperative_ranked(
        2, "seasonal_naive", "statistical", alternative_values
    )
    champion = parse_champion_release(release.to_payload()["anchor_release_payload"])
    task_profile = profile_task(
        RuntimeTask(
            task.numeric.task_id,
            task.numeric.history_values,
            task.numeric.prediction_length,
            task.numeric.frequency,
            (),
        )
    )
    source = NumericalForecastPackage(
        task_profile=task_profile,
        active_candidate_names=(anchor.name, alternative.name),
        candidate_diagnostics={
            anchor.name: anchor.diagnostics,
            alternative.name: alternative.diagnostics,
        },
        morphology_card=None,
        accepted_assumptions=(),
        rejected_assumptions={},
        selection_decision=SelectionDecision(
            mode="single",
            selected=(anchor.name,),
            weights=(1.0,),
            forecast=anchor.forecast,
            confidence=0.0,
            reason_codes=("cooperative_smoke_anchor",),
            rejected={},
            baseline_name=anchor.name,
            considered_candidates=(anchor.name, alternative.name),
        ),
        final_forecast=anchor.forecast,
        protected_baseline=anchor,
        ranked_alternatives=(anchor, alternative),
        retrieval_handoff=(),
        component_fingerprints={
            "task_input": task_input_fingerprint(
                task_id=task.numeric.task_id,
                history=task.numeric.history_values,
                frequency=task.numeric.frequency,
                horizon=task.numeric.prediction_length,
            ),
            "champion_release": champion_fingerprint(champion),
            "champion_recipe": champion_fingerprint(champion.policy.recipe),
            "champion_assumptions": champion_fingerprint(
                champion.policy.recipe.assumptions
            ),
        },
    )
    return bound_numerical_package(
        source,
        release,
        {anchor.name: anchor, alternative.name: alternative},
    )


def _cooperative_numerical_pair(release, tasks, *, anchor_offset):
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _cooperative_package(
            task, supplied, anchor_offset=anchor_offset
        ),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


def _round1_smoke_response() -> str:
    return json.dumps(
        {
            "evidence_chains": [],
            "counterevidence": [],
            "missing_information": [],
            "sufficient": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _decision_smoke_response() -> str:
    return json.dumps(
        {
            "selected_candidate_id": "safe_anchor",
            "supporting_document_ids": [],
            "rationale": "Use the deterministic safe anchor.",
            "request_more_retrieval": False,
            "gaps": [],
            "used_skill_names": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cooperative_runtime(config, release, retrieval, decision, tasks, host_runtime):
    all_tasks = (*tasks["train"], *tasks["dev"])
    if host_runtime is None:
        if config.control.profile != "smoke":
            raise ValueError("cooperative pilot requires an injected Host runtime")
        seed_numerical = _cooperative_numerical_pair(
            release, all_tasks, anchor_offset=0.0
        )
        alternate_payload = release.to_payload()
        alternate_payload["version"] = "n002"
        alternate_payload["parent_sha256"] = release.fingerprint
        alternate_release = parse_numerical_supply_release(alternate_payload)
        alternatives = (
            _cooperative_numerical_pair(
                alternate_release, all_tasks, anchor_offset=1.0
            ),
        )

        def retrieval_factory(genome, skills):
            return TwoStageRetrievalAgent(
                FakeLLMClient([_round1_smoke_response()]), genome, skills
            )

        def decision_factory(module):
            return DecisionAgent(
                FakeLLMClient([_decision_smoke_response(), _decision_smoke_response()]),
                prompt=module.prompt,
            )

        retrieval_library = None
        monotonic = None
    else:
        seed_numerical = getattr(host_runtime, "seed_numerical", None)
        alternatives = tuple(getattr(host_runtime, "numerical_alternatives", ()))
        retrieval_factory = getattr(host_runtime, "retrieval_factory", None)
        decision_factory = getattr(host_runtime, "decision_factory", None)
        retrieval_library = getattr(host_runtime, "retrieval_skill_library", None)
        monotonic = getattr(host_runtime, "monotonic", None)
        if type(seed_numerical) is not FrozenNumericalArtifactsV2:
            raise ValueError("Host runtime requires a frozen seed_numerical pair")
        if seed_numerical.release.fingerprint != release.fingerprint:
            raise ValueError("Host seed Numerical release does not match seed input")
        if any(type(item) is not FrozenNumericalArtifactsV2 for item in alternatives):
            raise ValueError("Host Numerical alternatives must be frozen pairs")
        if not callable(retrieval_factory) or not callable(decision_factory):
            raise ValueError("Host runtime requires Retrieval and Decision factories")

    pipeline = CooperativePipelineAdapter(
        CooperativeArtifactCatalog(lambda _identity, _payload: None),
        retrieval_factory,
        decision_factory,
        metric_cap=config.metric_cap,
        retrieval_skill_library=retrieval_library,
    )
    adapters = {
        "numerical": NumericalCoordinateAdapter(alternatives),
        "retrieval": RetrievalCoordinateAdapter(),
        "decision": DecisionCoordinateAdapter(
            ("Prefer the lowest finite complete-pipeline error.",)
        ),
        "pipeline": pipeline,
    }
    if monotonic is not None:
        adapters["monotonic"] = monotonic
    return seed_numerical, adapters


def _cooperative_evolve(
    config_path: Path,
    seed_supply_path: Path,
    task_manifest_path: Path,
    retrieval_release_path: Path,
    decision_policy_path: Path,
    output: Path,
    *,
    host_runtime=None,
    config_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config_data = (
        _read_canonical(config_path) if config_payload is None else dict(config_payload)
    )
    config = CooperativeConfigV2.from_payload(config_data)
    seed_data, task_data, retrieval_data, decision_data = (
        _read_canonical(path)
        for path in (
            seed_supply_path,
            task_manifest_path,
            retrieval_release_path,
            decision_policy_path,
        )
    )
    release = parse_numerical_supply_release(seed_data)
    tasks = _parse_cooperative_task_manifest(task_data)
    retrieval = RetrievalModuleV2.from_payload(retrieval_data)
    decision = DecisionModuleV2.from_payload(decision_data)
    seed_numerical, adapters = _cooperative_runtime(
        config, release, retrieval, decision, tasks, host_runtime
    )
    if output.exists() and not output.is_dir():
        raise ValueError("cooperative output must be a directory")
    resume = output.is_dir() and any(output.iterdir())
    result = run_cooperative_evolution(
        output,
        config,
        {
            "numerical": seed_numerical,
            "retrieval": retrieval,
            "decision": decision,
        },
        tasks,
        adapters,
        resume=resume,
    )
    return result.to_payload()


def cooperative_evolve(
    config_path: Path,
    seed_supply_path: Path,
    task_manifest_path: Path,
    retrieval_release_path: Path,
    decision_policy_path: Path,
    output: Path,
    *,
    host_runtime=None,
) -> dict[str, object]:
    """Run the cooperative prototype through an explicit Host seam."""
    return _cooperative_evolve(
        config_path,
        seed_supply_path,
        task_manifest_path,
        retrieval_release_path,
        decision_policy_path,
        output,
        host_runtime=host_runtime,
    )


def _dispatch_evolve(args) -> dict[str, object]:
    payload, canonical = _read_evolve_config(args.config)
    keys = frozenset(payload)
    cooperative_paths = (
        args.seed_supply,
        args.task_manifest,
        args.retrieval_release,
        args.decision_policy,
    )
    if keys == _EVOLUTION_CONFIG_FIELDS:
        if any(path is not None for path in cooperative_paths):
            raise ValueError("cooperative seed flags require a cooperative config")
        return _evolve(args.config, args.output_dir, config_payload=payload)
    if keys == _COOPERATIVE_CONFIG_FIELDS:
        if not canonical:
            raise ValueError("cooperative config must use canonical V2 JSON")
        if any(path is None for path in cooperative_paths):
            raise ValueError("cooperative config requires all four seed input flags")
        return _cooperative_evolve(
            args.config,
            *cooperative_paths,
            args.output_dir,
            config_payload=payload,
        )
    raise ValueError("evolve config does not match an exact supported schema")


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
            summary = _dispatch_evolve(args)
        elif args.command == "public-evaluate":
            summary = _public_evaluate(args.bundle, args.output_dir)
        elif args.command == "real-evolve":
            summary = _real_evolve(args.manifest, args.output_dir, authority_root=args.authority_root)
        elif args.command in {"protocol-evolve", "protocol-make-smoke-inputs"}:
            summary = dispatch_protocol(args)
        else:
            summary = numerical_evolve(
                args.config,
                args.seed_supply,
                args.task_manifest,
                args.output_dir,
                task_local_evidence_path=args.task_local_evidence,
                task_local_dictionary=args.task_local_dictionary, legacy_bootstrap=args.legacy_bootstrap,
            )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Evolution V2: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(canonical_v2_bytes(summary).decode("utf-8"))
    return 0


__all__ = ["build_parser", "cooperative_evolve", "main", "numerical_evolve"]
