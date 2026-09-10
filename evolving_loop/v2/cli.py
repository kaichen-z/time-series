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
import sys
from pathlib import Path

from common.payload import strict_json_loads

from .bundle import EvolutionBundleV2
from .contracts import KernelProtocolCommitment, canonical_v2_bytes, load_v2_config
from .fakes import run_fake_kernel
from .store import write_once_json


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


def _public_evaluate(bundle_path: Path, output: Path) -> dict[str, object]:
    source_path = bundle_path.resolve(strict=True)
    if source_path.name != "accepted_bundle.json":
        raise ValueError("bundle must be a source V2 run's accepted_bundle.json")
    source = source_path.parent
    destination = output.resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
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
        else:
            summary = _public_evaluate(args.bundle, args.output_dir)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Evolution V2: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(canonical_v2_bytes(summary).decode("utf-8"))
    return 0


__all__ = ["build_parser", "main"]
