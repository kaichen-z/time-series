from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_huggingface_tasks, load_sample_tasks
from .pipeline import MinimalAgentSystem, SystemConfig, write_outputs


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="outputs/minimal-agent")
    parser.add_argument("--task-id", action="append", help="Run only this benchmark_id; repeatable")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--context-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drcik-agent",
        description="Run the minimal forecast-aware retrieval system on Dr-CiK.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("run-sample", help="Run the official repository sample")
    sample.add_argument("--sample-dir", required=True, help="Path to Dr-CiK/sample")
    _add_shared_arguments(sample)

    huggingface = subparsers.add_parser("run-hf", help="Run the Hugging Face release")
    split = huggingface.add_mutually_exclusive_group()
    split.add_argument("--public-dev", action="store_true", help="Run public labeled tasks (default)")
    split.add_argument("--hidden-test", action="store_true", help="Run hidden unlabeled tasks")
    _add_shared_arguments(huggingface)
    return parser


def _select_tasks(tasks, task_ids: list[str] | None, limit: int | None):
    if task_ids:
        requested = set(task_ids)
        tasks = [task for task in tasks if task.benchmark_id in requested]
        missing = requested - {task.benchmark_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task IDs: {', '.join(sorted(missing))}")
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise SystemExit("No tasks selected")
    return tasks


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run-sample":
        tasks = load_sample_tasks(arguments.sample_dir)
    else:
        labels_public = not arguments.hidden_test
        tasks = load_huggingface_tasks(labels_public=labels_public)
    tasks = _select_tasks(tasks, arguments.task_id, arguments.limit)

    system = MinimalAgentSystem(
        SystemConfig(
            top_k=arguments.top_k,
            num_samples=arguments.samples,
            context_weight=arguments.context_weight,
            seed=arguments.seed,
        )
    )
    results = system.run_many(tasks)
    write_outputs(results, arguments.output_dir)
    summary_path = Path(arguments.output_dir).expanduser().resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Completed {len(results)} task(s).")
    print(f"Outputs: {summary_path.parent}")
    for name, value in summary["mean_metrics"].items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
