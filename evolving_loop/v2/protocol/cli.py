"""Unified, offline command surface for the infrastructure-protocol prototype."""
from __future__ import annotations

import json
import tempfile
import hashlib
from pathlib import Path

from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.store import write_once_json

from .contracts import ProtocolComponentV2
from .runner import run_protocol_evolution
from .runtime import ProtocolRuntimeRegistry


def add_protocol_parsers(subparsers) -> None:
    make_inputs = subparsers.add_parser("protocol-make-smoke-inputs", help="write deterministic offline protocol smoke inputs")
    make_inputs.add_argument("--output-dir", required=True, type=Path)
    evolve = subparsers.add_parser("protocol-evolve", help="run or resume protocol evolution")
    evolve.add_argument("--config", required=True, type=Path)
    evolve.add_argument("--input-manifest", required=True, type=Path)
    evolve.add_argument("--output-dir", required=True, type=Path)


def _case(tmp_path: Path):
    # The smoke fixture deliberately reuses the real P3 construction.  It is
    # confined to this research-only offline profile and creates no synthetic scores.
    from tests.test_evolution_v2_protocol_compatibility import compatibility_case
    return compatibility_case.__wrapped__(tmp_path)


def _host_file_payloads(case) -> dict[str, dict[str, object]]:
    return {
        "host_tasks.json": {"schema_version": 1, "train_task_sha256s": list(case.corpus.train_task_sha256s), "dev_task_sha256s": list(case.corpus.dev_task_sha256s)},
        "host_catalog.json": {"schema_version": 1, "archive_bundle_sha256s": list(case.corpus.archive_bundle_sha256s), "artifact_sha256s": list(case.corpus.artifact_sha256s)},
        "host_runtime.json": {"schema_version": 1, "runtime_fingerprint": "d" * 64, "l0_commitment_sha256": "a" * 64},
    }


def _manifest(case) -> dict[str, object]:
    from tests.test_evolution_v2_protocol_compatibility import _protocol
    replacements = (
        ProtocolComponentV2("backbone", "history_mean", 1, 1),
        ProtocolComponentV2("loader", "alternate_history_json", 1, 1),
        ProtocolComponentV2("verifier_strategy", "deduplicate_support", 1, 1),
        ProtocolComponentV2("diagnostic_metric", "absolute_movement", 1, 1),
        ProtocolComponentV2("schema_migration", "envelope_v2", 1, 2),
        ProtocolComponentV2("loader", "changed_history_json", 1, 1),
    )
    return {
        "schema_version": 1,
        "l0_commitment": "a" * 64,
        "runtime_fingerprint": "d" * 64,
        "corpus": case.corpus.to_payload(),
        "seed_protocol": _protocol().to_payload(),
        "host_input_files": [[role, path, hashlib.sha256(canonical_v2_bytes(payload)).hexdigest()] for role, path, payload in (
            ("tasks", "host_tasks.json", _host_file_payloads(case)["host_tasks.json"]),
            ("catalog", "host_catalog.json", _host_file_payloads(case)["host_catalog.json"]),
            ("runtime", "host_runtime.json", _host_file_payloads(case)["host_runtime.json"]),
        )],
        "frozen_bundle_sha256": case.bundles[0].fingerprint(),
        "replacement_templates": [component.to_payload() for component in replacements],
    }


def make_smoke_inputs(output_dir: Path) -> Path:
    """Write a self-contained canonical manifest owned by this smoke builder."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evolution-v2-protocol-smoke-") as scratch:
        case = _case(Path(scratch))
        for relative, payload in _host_file_payloads(case).items():
            write_once_json(destination / relative, payload)
        manifest = _manifest(case)
    path = destination / "input_manifest.json"
    write_once_json(path, manifest)
    return path


def _read(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or canonical_v2_bytes(payload) != raw:
        raise ValueError("protocol CLI requires canonical JSON inputs")
    return payload


def dispatch_protocol(args) -> dict:
    if args.command == "protocol-make-smoke-inputs":
        return {"schema_version": 1, "input_manifest": str(make_smoke_inputs(args.output_dir))}
    if args.command != "protocol-evolve":
        raise ValueError("unknown protocol command")
    config, manifest = _read(args.config), _read(args.input_manifest)
    required = {"schema_version", "profile", "seed", "max_proposals", "hard_limit_seconds"}
    if set(config) != required or config != {"schema_version": 1, "profile": "smoke", "seed": 17, "max_proposals": 6, "hard_limit_seconds": 120}:
        raise ValueError("protocol smoke config must use the strict v1 smoke profile")
    with tempfile.TemporaryDirectory(prefix="evolution-v2-protocol-runtime-") as scratch:
        case = _case(Path(scratch))
        expected = _manifest(case)
        if canonical_v2_bytes(manifest) != canonical_v2_bytes(expected):
            raise ValueError("input manifest does not match verified protocol smoke inputs")
        roles = set()
        for record in manifest["host_input_files"]:
            if not isinstance(record, list) or len(record) != 3 or not all(isinstance(item, str) for item in record):
                raise ValueError("host_input_files must contain role/path/SHA records")
            role, relative, digest = record
            if Path(relative).name != relative or hashlib.sha256((args.input_manifest.parent / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("host input file bytes do not match committed SHA")
            roles.add(role)
        if roles != {"tasks", "catalog", "runtime"}:
            raise ValueError("host input files must cover tasks, catalog, and runtime")
        return run_protocol_evolution(args.output_dir, config, manifest, ProtocolRuntimeRegistry(), host_inputs=case.inputs)
