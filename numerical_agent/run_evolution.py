"""Run generations of method-module evolution against the frozen Train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_tasks
from common.evolution_core.contracts import (
    metric_policy_metadata,
    load_active_release,
)
from common.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig, QwenClient
from common.payload import read_json_object, write_json
from common.tracing import configure

from .evolution import run_evolution
from .evolution.cache import OutcomeCache
from .evolution.execution import Task
from .evolution.targetwise import evolve_targets_once
from .evolution.policy_targetwise import evolve_policies_once
from .evolution.portfolio import (
    PolicyOutcomeCache,
    PolicyPortfolio,
    require_flagship_runtimes,
)
from .main import _add_tsfm_runtime_options, _runtime_registry

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
    parser.add_argument(
        "--evolution-strategy", choices=("batch", "targetwise"), default="batch"
    )
    parser.add_argument("--outcome-cache-dir", default=None)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--screen-tasks", type=int, default=4)
    parser.add_argument(
        "--full-evaluation-candidates",
        type=int,
        default=3,
        help="number of target-wise screen survivors allowed to run full Train and mini-dev",
    )
    parser.add_argument(
        "--failure-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="interpret Train-only deterministic forecast diagnostics before mutation",
    )
    parser.add_argument("--llm-backend", choices=LLM_BACKENDS, default="qwen")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", choices=("none", "low", "medium", "high"), default=None)
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument("--codex-timeout", type=int, default=None)
    parser.add_argument(
        "--selector-codex-model",
        default=None,
        help="enable two-stage evolution with this Codex model as the low-cost selector",
    )
    parser.add_argument(
        "--selector-codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--selector-codex-cache-dir", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument(
        "--validation-tail",
        type=int,
        default=0,
        help="reserve this many tasks from the limited Train prefix for child acceptance only",
    )
    parser.add_argument(
        "--isolated-methods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run every forecasting method in a crash-contained subprocess",
    )
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--claude-cache-dir", default=None)
    parser.add_argument("--claude-timeout", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--foundation-portfolio",
        choices=("none", "flagship5"),
        default="none",
        help="co-evolve five reviewed TSFM policies and five executable Combined policies",
    )
    parser.add_argument("--policy-outcome-cache-dir", default=None)
    parser.add_argument("--policy-max-targets", type=int, default=3)
    _add_tsfm_runtime_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo)
    _load_or_create_run_manifest(repo)
    configure(repo / "run_evolution_trace.jsonl")

    tasks, validation_tasks = _evolution_tasks(
        args.split_file,
        args.tasks_file,
        train_limit=args.train_limit,
        validation_tail=args.validation_tail,
    )
    _validate_judge_halving_configuration(args, tasks, validation_tasks)
    llm, selector_llm = _llm_clients(args)

    if args.foundation_portfolio != "none" and args.evolution_strategy != "targetwise":
        raise ValueError("foundation portfolio requires targetwise evolution")

    if args.evolution_strategy == "targetwise":
        if selector_llm is None:
            raise ValueError("targetwise evolution requires --selector-codex-model")
        skills_path = repo / "skills.py"
        cache = OutcomeCache(
            args.outcome_cache_dir or repo / "outcome-cache",
            skills_path=skills_path if skills_path.is_file() else None,
        )
        policy_cache = PolicyOutcomeCache(
            args.policy_outcome_cache_dir or repo / "policy-outcome-cache"
        )
        runtimes = _runtime_registry(args) if args.foundation_portfolio == "flagship5" else None
        try:
            if runtimes is not None:
                if not (repo / "policies.py").is_file():
                    raise ValueError("flagship5 requires a tracked policies.py in the evolution repo")
                require_flagship_runtimes(PolicyPortfolio.flagship5(), runtimes)
            for generation in range(1, args.generations + 1):
                outcome = evolve_targets_once(
                    repo,
                    tasks,
                    llm,
                    selector_llm,
                    generation=generation,
                    outcome_cache=cache,
                    validation_tasks=validation_tasks,
                    judge=selector_llm if args.failure_judge else None,
                    screen_tasks=args.screen_tasks,
                    max_targets=args.max_targets,
                    full_evaluation_candidates=args.full_evaluation_candidates,
                    isolate_methods=args.isolated_methods,
                )
                print(
                    f"generation {outcome.number}: {outcome.method_count} Python methods, "
                    f"commit {outcome.commit}  ({len(outcome.applied)} operations; "
                    f"cache {outcome.cache_hits} hits/{outcome.cache_misses} misses; "
                    f"{outcome.elapsed_seconds:.2f}s)"
                )
                policy_outcome = None
                if runtimes is not None:
                    policy_outcome = evolve_policies_once(
                        repo,
                        tasks,
                        llm,
                        selector_llm,
                        generation=generation,
                        outcome_cache=cache,
                        policy_cache=policy_cache,
                        validation_tasks=validation_tasks,
                        runtimes=runtimes,
                        judge=selector_llm if args.failure_judge else None,
                        screen_tasks=args.screen_tasks,
                        max_targets=args.policy_max_targets,
                        full_evaluation_candidates=args.full_evaluation_candidates,
                        isolate_methods=args.isolated_methods,
                    )
                    print(
                        f"generation {generation}: {policy_outcome.candidate_count} total "
                        f"candidates (Python + TSFM + Combined), commit {policy_outcome.commit} "
                        f"({len(policy_outcome.applied)} policy repairs; "
                        f"cache {policy_outcome.cache_hits} hits/"
                        f"{policy_outcome.cache_misses} misses; "
                        f"{policy_outcome.elapsed_seconds:.2f}s)"
                    )
                if not outcome.applied and (
                    policy_outcome is None or not policy_outcome.applied
                ):
                    break
        finally:
            if runtimes is not None:
                runtimes.close()
    else:
        outcomes = run_evolution(
            repo,
            tasks,
            llm,
            generations=args.generations,
            selector_llm=selector_llm,
            isolate_methods=args.isolated_methods,
            validation_tasks=validation_tasks,
        )
        for outcome in outcomes:
            status = f"rejected: {outcome.rejected}" if outcome.rejected else f"{len(outcome.applied)} operations"
            print(f"generation {outcome.number}: {outcome.method_count} methods, commit {outcome.commit}  ({status})")
    return 0


def _load_or_create_run_manifest(repo: str | Path) -> dict[str, object]:
    """Create a new active run binding or fail closed when resuming one."""
    path = Path(repo) / "run_manifest.json"
    if path.exists():
        payload = read_json_object(path)
        return load_active_release(payload)
    lifecycle_markers = (
        Path(repo) / "run_evolution_trace.jsonl",
        Path(repo) / "outcome-cache",
        Path(repo) / "policy-outcome-cache",
    )
    if any(item.exists() for item in lifecycle_markers) or any(
        Path(repo).glob("generation_*_result.json")
    ):
        raise ValueError(
            "existing evolution run is missing metric policy manifest; "
            "legacy runs cannot seed active evolution"
        )
    payload = {
        "schema_version": 2,
        **metric_policy_metadata(),
        "phase": "numerical_method_evolution",
    }
    write_json(path, payload)
    return payload


def _train_tasks(
    split_file: str, tasks_file: str, *, limit: int | None = None
) -> tuple[Task, ...]:
    payload = read_json_object(split_file)
    train_ids = list(payload["partitions"]["train"]["task_ids"])  # type: ignore[index]
    if limit is not None:
        if limit < 1:
            raise ValueError("train limit must be positive")
        train_ids = train_ids[:limit]
    catalog = {task.task_id: task for task in load_tasks(tasks_file)}
    return tuple(
        Task(task.task_id, tuple(task.history_values), task.prediction_length,
             task.frequency, tuple(task.future_values))
        for task_id in train_ids
        if (task := catalog.get(task_id)) is not None
    )


def _evolution_tasks(
    split_file: str | Path,
    tasks_file: str | Path,
    *,
    train_limit: int | None,
    validation_tail: int,
) -> tuple[tuple[Task, ...], tuple[Task, ...]]:
    """Load evolution tasks first, then a disjoint validation tail."""
    if validation_tail < 0:
        raise ValueError("validation tail cannot be negative")
    total_limit = None if train_limit is None else train_limit + validation_tail
    selected = _train_tasks(str(split_file), str(tasks_file), limit=total_limit)
    if validation_tail >= len(selected) and validation_tail:
        raise ValueError("validation tail must be smaller than the selected task count")
    validation = selected[-validation_tail:] if validation_tail else ()
    train = selected[:-validation_tail] if validation_tail else selected
    return train, validation


def _llm_clients(args: argparse.Namespace):
    """Return the mutator and optional low-cost selector clients."""
    mutator = _llm_client(args)
    if args.llm_backend != "codex" or not args.selector_codex_model:
        return mutator, None
    selector = CodexCLIClient(CodexCLIConfig(**_present(
        model=args.selector_codex_model,
        reasoning_effort=args.selector_codex_reasoning_effort,
        timeout_seconds=args.codex_timeout,
        cache_dir=args.selector_codex_cache_dir or args.codex_cache_dir,
    )))
    return mutator, selector


def _validate_judge_halving_configuration(
    args: argparse.Namespace,
    tasks: tuple[Task, ...],
    validation_tasks: tuple[Task, ...],
) -> None:
    """Freeze the experimental boundary whenever the Judge-assisted protocol is enabled."""
    if not args.failure_judge:
        return
    if args.evolution_strategy != "targetwise":
        raise ValueError("failure Judge requires targetwise evolution")
    if len(tasks) != 16 or len(validation_tasks) != 4:
        raise ValueError("failure Judge experiment requires exactly 16 Train and 4 mini-dev tasks")
    if not 8 <= args.max_targets <= 10:
        raise ValueError("failure Judge experiment requires max_targets between 8 and 10")
    if args.screen_tasks != 4:
        raise ValueError("failure Judge experiment requires exactly four screen tasks")
    if not 2 <= args.full_evaluation_candidates <= 3:
        raise ValueError(
            "failure Judge experiment requires two or three full-evaluation candidates"
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


def _present(**values: object) -> dict[str, object]:
    """Drop unset options so each config keeps its declared default."""
    return {name: value for name, value in values.items() if value is not None}


if __name__ == "__main__":
    raise SystemExit(main())
