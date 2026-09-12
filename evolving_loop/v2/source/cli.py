"""Offline-only command surface for the Source V2 research prototype."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.payload import strict_json_loads

from ..contracts import canonical_v2_bytes
from .contracts import SourceConfigV2
from .runner import run_source_evolution


def _read_config(path: Path) -> SourceConfigV2:
    try:
        raw = path.read_bytes()
        payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("cannot read source config") from error
    if not isinstance(payload, dict) or canonical_v2_bytes(payload) != raw:
        raise ValueError("source config must be canonical JSON")
    return SourceConfigV2.from_payload(payload)


def _overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(), second.resolve()
    return left == right or left in right.parents or right in left.parents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolving_loop.v2.source")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-smoke-inputs")
    make.add_argument("--output-dir", required=True)
    evolve = commands.add_parser("evolve")
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--input-manifest", required=True)
    evolve.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "make-smoke-inputs":
            from tests.build_evolution_v2_source_fixture import export_source_manifest
            manifest = export_source_manifest(Path(args.output_dir))
            sys.stdout.buffer.write(canonical_v2_bytes({"manifest": str(manifest)}))
            return 0
        input_manifest = Path(args.input_manifest)
        output = Path(args.output_dir)
        if _overlap(input_manifest.parent, output):
            raise ValueError("input manifest directory must not overlap output directory")
        config = _read_config(Path(args.config))
        from tests.build_evolution_v2_source_fixture import load_source_manifest
        case = load_source_manifest(input_manifest)
        result = run_source_evolution(output, config, case, resume=output.exists() and any(output.iterdir()))
        sys.stdout.buffer.write(result.canonical_bytes())
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"Evolution V2 source: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
