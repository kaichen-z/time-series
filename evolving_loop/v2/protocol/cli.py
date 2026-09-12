"""Unified, offline command surface for the infrastructure-protocol prototype."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.store import write_once_json

from .contracts import ProtocolComponentV2
from .runner import run_protocol_evolution
from .runtime import ProtocolRuntimeRegistry
from .smoke import _build_smoke_case, build_smoke_case, smoke_host_payloads


def add_protocol_parsers(subparsers) -> None:
    make_inputs = subparsers.add_parser("protocol-make-smoke-inputs", help="write deterministic offline protocol smoke inputs")
    make_inputs.add_argument("--output-dir", required=True, type=Path)
    evolve = subparsers.add_parser("protocol-evolve", help="run or resume protocol evolution")
    evolve.add_argument("--config", required=True, type=Path)
    evolve.add_argument("--input-manifest", required=True, type=Path)
    evolve.add_argument("--output-dir", required=True, type=Path)


def _manifest(case, files: dict[str, dict[str, object]]) -> dict[str, object]:
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
        "l0_commitment": case.corpus.l0_commitment_sha256,
        "runtime_fingerprint": case.inputs.runtime_fingerprint,
        "corpus": case.corpus.to_payload(),
        "seed_protocol": case.seed_protocol.to_payload(),
        "host_input_files": [[role, path, hashlib.sha256(canonical_v2_bytes(payload)).hexdigest()] for role, path, payload in (
            ("tasks", "host_tasks.json", files["host_tasks.json"]),
            ("catalog", "host_catalog.json", files["host_catalog.json"]),
            ("runtime", "host_runtime.json", files["host_runtime.json"]),
            ("closure", "host_closure.json", files["host_closure.json"]),
        )],
        "frozen_bundle_sha256": case.bundles[0].fingerprint(),
        "replacement_templates": [component.to_payload() for component in replacements],
    }


def make_smoke_inputs(output_dir: Path) -> Path:
    """Write a self-contained canonical manifest owned by this smoke builder."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = smoke_host_payloads()
    bootstrap = _build_smoke_case(files, allow_bootstrap=True)
    files["host_closure.json"] = {
        "schema_version": 1,
        "corpus": bootstrap.corpus.to_payload(),
        "archive_bundles": [bundle.to_payload() for bundle in bootstrap.bundles],
        "artifact_envelopes": dict(bootstrap.inputs.artifact_envelopes),
        "bundle_acceptance_evidence": dict(bootstrap.inputs.bundle_acceptance_evidence or {}),
        "split_metadata": {key: list(value) for key, value in bootstrap.inputs.split_manifest.items()},
        "frozen_bundle_sha256": bootstrap.bundles[0].fingerprint(),
    }
    case = build_smoke_case(files)
    for relative, payload in files.items():
        write_once_json(destination / relative, payload)
    manifest = _manifest(case, files)
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
    roles, files = set(), {}
    for record in manifest["host_input_files"]:
        if not isinstance(record, list) or len(record) != 3 or not all(isinstance(item, str) for item in record):
            raise ValueError("host_input_files must contain role/path/SHA records")
        role, relative, digest = record
        path = args.input_manifest.parent / relative
        if Path(relative).name != relative or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("host input file bytes do not match committed SHA")
        files[relative] = _read(path)
        roles.add(role)
    if roles != {"tasks", "catalog", "runtime", "closure"} or set(files) != {"host_tasks.json", "host_catalog.json", "host_runtime.json", "host_closure.json"}:
        raise ValueError("host input files must cover exact tasks, catalog, runtime, and authority closure inputs")
    case = build_smoke_case(files)
    if canonical_v2_bytes(manifest) != canonical_v2_bytes(_manifest(case, files)):
        raise ValueError("input manifest does not match verified protocol smoke inputs")
    return run_protocol_evolution(args.output_dir, config, manifest, ProtocolRuntimeRegistry(), host_inputs=case.inputs)
