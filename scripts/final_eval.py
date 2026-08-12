#!/usr/bin/env python3
"""Scores a frozen bundle triple on the test split exactly once; deliberately separate from the evolve CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from dr_cik.local_llm import QwenClient, QwenConfig

from evolving_agents.agents.coding import CodingAgent
from evolving_agents.agents.decision import DecisionAgent
from evolving_agents.agents.retrieval import RetrievalAgent
from evolving_agents.bundles import SEED_DIR, load_bundle
from evolving_agents.cli_common import DEFAULT_EVOLVER_MODEL_ID
from evolving_agents.harness.baselines import mean_metrics
from evolving_agents.harness.datasets import load_drcik_splits
from evolving_agents.harness.orchestrator import run_task
from evolving_agents.harness.run_log import PROXY_NOTE, append_run_record
from evolving_agents.harness.trace import configure_tracing
from evolving_agents.llm_cache import DEFAULT_CACHE_DIR, CachingLLMClient
from evolving_agents.logging_setup import configure as configure_logging
from evolving_agents.logging_setup import log_exception
from dr_cik.local_llm import DEFAULT_MODEL_ID as DEFAULT_WORKER_MODEL_ID

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Flags for the single, final, frozen evaluation pass."""
    parser = argparse.ArgumentParser(prog="final_eval", description="Score frozen bundles on the held-out test split, once.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir")
    source.add_argument("--data-dir")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", required=True)
    for agent in ("coding", "retrieval", "decision"):
        parser.add_argument(f"--{agent}-bundle", default=str(SEED_DIR / agent / "v000.json"))
    parser.add_argument("--worker-model-id", default=DEFAULT_WORKER_MODEL_ID)
    parser.add_argument("--judge-model-id", default=DEFAULT_EVOLVER_MODEL_ID)
    parser.add_argument("--worker-device", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--trace-level", choices=("off", "summary", "full"), default="summary")
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run against an output dir that already holds a summary")
    parser.add_argument("--log-file", default=None, help="Full log; defaults to ./logs/<output-dir-name>.log")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--console-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--trace-console", action="store_true", help="Also stream the per-call trace to the terminal")
    return parser


def run_final_eval(args: argparse.Namespace, worker=None) -> dict:
    """Score the test split once, refusing to overwrite an existing summary without --force."""
    output = Path(args.output_dir).expanduser().resolve()
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.force:
        raise SystemExit(
            f"{summary_path} already exists; the test split is meant to be scored exactly once. Pass --force to override."
        )

    configure_tracing(args.trace_level, runs_dir=args.runs_dir)
    splits = load_drcik_splits(
        data_dir=args.data_dir, sample_dir=args.sample_dir, seed=args.seed,
        split_file=Path(args.split_file) if args.split_file else None,
    )
    tasks = splits.test if args.limit is None else splits.test[: args.limit]
    if not tasks:
        raise SystemExit("the test split is empty; check --sample-dir/--data-dir and --split-file")

    llm = CachingLLMClient(
        worker or QwenClient(QwenConfig(model_id=args.worker_model_id, device=args.worker_device)),
        model_id=args.worker_model_id,
        cache_dir=args.cache_dir or DEFAULT_CACHE_DIR,
    )
    bundles = {agent: load_bundle(getattr(args, f"{agent}_bundle")) for agent in ("coding", "retrieval", "decision")}
    judge = None if args.no_judge else llm

    traces = []
    for index, task in enumerate(tasks, start=1):
        logger.info("final eval %d/%d: %s", index, len(tasks), task.benchmark_id)
        trace = run_task(
            task,
            CodingAgent(llm, bundles["coding"]),
            RetrievalAgent(llm, bundles["retrieval"]),
            DecisionAgent(llm, bundles["decision"]),
            judge=judge,
            n_windows=args.n_windows,
        )
        traces.append(trace)
        append_run_record(
            args.runs_dir, "final_eval.jsonl",
            {"task_id": task.benchmark_id, "loop": "final_eval", "metrics": trace.metrics,
             "bundle_versions": {name: bundle.version for name, bundle in bundles.items()}},
        )

    summary = {
        "split": "test",
        "num_tasks": len(traces),
        "bundle_versions": {name: bundle.version for name, bundle in bundles.items()},
        "mean_metrics": mean_metrics(traces),
        "note": PROXY_NOTE,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the frozen evaluation, and print its summary."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    log_path = configure_logging(
        args.log_file or Path("logs") / f"{Path(args.output_dir).name}.log",
        log_level=args.log_level,
        console_level=args.console_level,
        trace_to_console=args.trace_console,
    )
    try:
        summary = {**run_final_eval(args), "log_file": str(log_path)}
    except Exception as exc:
        log_exception(exc)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
