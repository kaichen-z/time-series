"""Shared argparse helpers, following dr_cik/cli.py's flag conventions."""

from __future__ import annotations

import argparse
from pathlib import Path

from dr_cik.local_llm import DEFAULT_MODEL_ID as DEFAULT_WORKER_MODEL_ID

from .bundles import SEED_DIR

# A 35B MoE with ~3B active params: far stronger than the worker for the rare, quality-critical
# mutation calls, while still fitting beside it on one shared H100. See the plan's M0 fallback note.
DEFAULT_EVOLVER_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_DATA_DIR = "/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK"


def add_task_source_args(subparser: argparse.ArgumentParser) -> None:
    """--sample-dir / --data-dir, mirroring dr_cik's task-source convention."""
    source = subparser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir", help="Path to the official Dr-CiK sample/ directory")
    source.add_argument("--data-dir", help=f"Path to a full Dr-CiK dataset (e.g. {DEFAULT_DATA_DIR})")
    subparser.add_argument("--split-file", default=None, help="Committed split membership JSON; defaults to the packaged one")
    subparser.add_argument("--limit", type=int, default=None, help="Use at most this many tasks per split")


def add_llm_args(subparser: argparse.ArgumentParser) -> None:
    """Worker and evolver model selection; both are local Qwen checkpoints, no API keys involved."""
    subparser.add_argument("--worker-model-id", default=DEFAULT_WORKER_MODEL_ID)
    subparser.add_argument("--evolver-model-id", default=DEFAULT_EVOLVER_MODEL_ID)
    subparser.add_argument("--worker-device", default=None, help="e.g. cuda:1; pin it when running beside the evolver")
    subparser.add_argument("--evolver-device", default=None, help="e.g. cuda:2; pin it so both models never land on one GPU")
    subparser.add_argument("--cache-dir", default=None, help="LLM response cache directory (default .cache/llm)")


def add_evolve_args(subparser: argparse.ArgumentParser) -> None:
    """Population, budget, checkpointing, and trace verbosity."""
    subparser.add_argument("--generations", type=int, default=10)
    subparser.add_argument("--population-size", type=int, default=6)
    subparser.add_argument("--keep-elite", type=int, default=2)
    subparser.add_argument("--stall-patience", type=int, default=3)
    subparser.add_argument("--minibatch-size", type=int, default=20)
    subparser.add_argument("--seed", type=int, default=7)
    subparser.add_argument("--checkpoint-dir", required=True, help="Per-generation checkpoints; reused to resume a crashed run")
    subparser.add_argument("--bundles-dir", default=str(SEED_DIR), help="Where evolved bundles are written")
    subparser.add_argument("--runs-dir", default="runs", help="Append-only JSONL run records and reasoning sidecars")
    subparser.add_argument(
        "--trace-level",
        choices=("off", "summary", "full"),
        default="summary",
        help="summary: one line per LLM/tool call with reasoning written to a sidecar; full: everything inline",
    )


def add_bundle_args(subparser: argparse.ArgumentParser) -> None:
    """Explicit bundle paths, defaulting to the committed v000 seeds."""
    for agent in ("coding", "retrieval", "decision"):
        subparser.add_argument(f"--{agent}-bundle", default=str(SEED_DIR / agent / "v000.json"))


def resolve_split_file(args: argparse.Namespace) -> Path | None:
    """Return the split-membership path an argparse namespace selected, if any."""
    from .harness.datasets import DEFAULT_SPLIT_FILE

    return Path(args.split_file) if args.split_file else DEFAULT_SPLIT_FILE
