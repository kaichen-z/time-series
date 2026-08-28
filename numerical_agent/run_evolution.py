"""Run generations of method-module evolution against the frozen Train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_tasks
from common.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig, QwenClient
from common.payload import read_json_object
from common.tracing import configure

from .evolution import run_evolution
from .evolution.execution import Task

LLM_BACKENDS = ("codex", "qwen", "claude")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="the evolution git repo, e.g. runs/method_evolution/v001")
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument(
        "--tasks-file",
        default="/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl",
    )
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--llm-backend", choices=LLM_BACKENDS, default="qwen")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", choices=("none", "low", "medium", "high"), default=None)
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument("--codex-timeout", type=int, default=None)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--claude-cache-dir", default=None)
    parser.add_argument("--claude-timeout", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--memory-model",
        default=None,
        help="a small local model that summarizes each generation into memory.md; off when unset",
    )
    parser.add_argument("--memory-device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo)
    configure(repo / "run_evolution_trace.jsonl")

    tasks = _split_tasks(args.split_file, args.tasks_file, "train")
    val_tasks = _split_tasks(args.split_file, args.tasks_file, "val")
    llm = _llm_client(args)

    outcomes = run_evolution(
        repo, tasks, llm, generations=args.generations, val_tasks=val_tasks,
        memory_llm=_memory_client(args),
    )
    for outcome in outcomes:
        status = f"rejected: {outcome.rejected}" if outcome.rejected else f"{len(outcome.applied)} operations"
        val = "" if outcome.val_best_smae is None else f"  val_best_smae={outcome.val_best_smae}"
        print(
            f"generation {outcome.number}: {outcome.method_count} methods, "
            f"commit {outcome.commit}  ({status}){val}"
        )
    return 0


def _split_tasks(split_file: str, tasks_file: str, partition: str) -> tuple[Task, ...]:
    """Load one partition of the frozen split. Never pass public_test to the evolution loop."""
    payload = read_json_object(split_file)
    wanted = set(payload["partitions"][partition]["task_ids"])  # type: ignore[index]
    catalog = {task.task_id: task for task in load_tasks(tasks_file)}
    return tuple(
        Task(task.task_id, tuple(task.history_values), task.prediction_length,
             task.frequency, tuple(task.future_values))
        for task_id, task in catalog.items()
        if task_id in wanted
    )


def _llm_client(args: argparse.Namespace):
    """Build the requested LLM client, keeping each config's own defaults."""
    if args.llm_backend == "codex":
        return CodexCLIClient(CodexCLIConfig(**_present(
            model=args.codex_model, reasoning_effort=args.codex_reasoning_effort,
            timeout_seconds=args.codex_timeout, cache_dir=args.codex_cache_dir,
        )))
    if args.llm_backend == "claude":
        return ClaudeCLIClient(ClaudeCLIConfig(**_present(
            model=args.claude_model, timeout_seconds=args.claude_timeout, cache_dir=args.claude_cache_dir,
        )))
    return QwenClient(**_present(
        model_id=args.model_id, device=args.device, max_new_tokens=args.max_new_tokens,
    ))


def _memory_client(args: argparse.Namespace):
    """The generation summarizer: a second, much smaller model, or None when unset."""
    if not args.memory_model:
        return None
    return QwenClient(**_present(
        model_id=args.memory_model, device=args.memory_device, max_new_tokens=1024,
    ))


def _present(**values: object) -> dict[str, object]:
    """Drop unset options so each config keeps its declared default."""
    return {name: value for name, value in values.items() if value is not None}


if __name__ == "__main__":
    raise SystemExit(main())
