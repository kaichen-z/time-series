"""Precompute frozen statistical hindcasts without scoring futures or making decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, TypeVar

from numerical_agent.providers import RuntimeRegistry

from .evaluate_frozen_two_stage import _public_test_tasks, verify_frozen_policies
from .evolution.execution import Task
from .evolution.filtering import build_filter_dictionary
from .evolution.module import read_module
from .evolution.numerical_selector import HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import materialize_active_dictionary, profile_task
from .evolution.screening_evolution import migrate_filter_dictionary
from .run_selector_evolution import ForecastStore


T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--hindcast-cache-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--folds", type=int, default=3)
    return parser


def shard(rows: Sequence[T], start: int, end: int) -> tuple[T, ...]:
    if start < 0 or end <= start or end > len(rows):
        raise ValueError(f"invalid half-open shard bounds [{start}, {end}) for {len(rows)} rows")
    return tuple(rows[start:end])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    screening_hash, _ = verify_frozen_policies(
        args.screening_dir, args.selector_dir, args.output_dir
    )
    repo = Path(args.repo).resolve()
    module_path = repo / "methods.py"
    skills_path = repo / "skills.py" if (repo / "skills.py").is_file() else None
    module = read_module(module_path)
    portfolio = read_policy_file(repo / "policies.py")
    all_screening = migrate_filter_dictionary(
        build_filter_dictionary(module, portfolio),
        fallback_names=("naive_last", "timesfm_2_5", "toto_2_0"),
    )
    # Strip labels immediately.  This helper performs no scoring and has no final-writer path.
    inputs = tuple(
        Task(task.task_id, task.history, task.horizon, task.frequency, ())
        for task in _public_test_tasks(args.split_file, args.tasks_file)
    )
    selected = shard(inputs, args.start, args.end)
    store = ForecastStore(
        args.hindcast_cache_dir,
        module_path,
        skills_path,
        module,
        portfolio,
        RuntimeRegistry(),
        screening_hash,
    )
    try:
        config = HindcastConfig(folds=args.folds)
        for offset, task in enumerate(selected, start=args.start):
            active = materialize_active_dictionary(all_screening, profile_task(task)).active
            statistical = tuple(candidate for candidate in active if candidate.family == "statistical")
            for candidate in statistical:
                diagnose_candidate(
                    task,
                    candidate.name,
                    candidate.family,
                    store.forecast,
                    config,
                    screening_policy_hash=screening_hash,
                    runtime_settings={"portfolio": "flagship5"},
                )
            print(json.dumps({
                "index": offset,
                "task_id": task.task_id,
                "statistical_candidates": len(statistical),
                "cache_hits": store.hits,
                "cache_misses": store.misses,
            }), flush=True)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
