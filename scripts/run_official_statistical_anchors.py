#!/usr/bin/env python3
"""Generate one resumable shard of all 93 reviewed statistical forecasts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evolving_loop.official_benchmark_loader import load_official_time_mmd, load_official_timesx
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.portfolio import read_policy_file

FREQUENCIES = {
    "hourly": "1 hour",
    "daily": "1 day",
    "weekly": "1 week",
    "monthly": "1 month",
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("time_mmd", "timesx"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    loader = load_official_time_mmd if args.dataset == "time_mmd" else load_official_timesx
    partitions = loader(
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    all_windows = list((*partitions.train, *partitions.dev))
    windows = all_windows[args.shard_index :: args.shard_count]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        windows = windows[: args.limit]

    method_root = args.method_root.resolve()
    portfolio = read_policy_file(str(method_root / "policies.py"))
    screening = _load_screening_policy(str(method_root / "dictionary.py"))
    store = ForecastStore(
        args.cache_root / f"shard-{args.shard_index:03d}",
        method_root / "methods.py",
        method_root / "skills.py",
        portfolio,
        None,
        screening_hash=screening.fingerprint(),
        runtime_identity={},
        statistical_time_budget_s=20.0,
    )
    methods = sorted(store.statistical_names)
    expected = {
        "schema": "official-statistical-cache-v1",
        "dataset": args.dataset,
        "official_source_fingerprint": partitions.source_fingerprint,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "methods": methods,
    }
    payload = dict(expected, complete=False, entries={}, failure_counts={})
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"cannot resume: {key} mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("entries must be an object")
    selected_ids = {window.task_id for window in windows}
    if not set(entries).issubset(selected_ids):
        raise ValueError("resume output contains task IDs outside this shard")
    failures = Counter(payload.get("failure_counts", {}))
    completed = len(entries)
    try:
        for window in windows:
            if window.task_id in entries:
                continue
            forecasts: dict[str, list[float]] = {}
            frequency = FREQUENCIES.get(window.frequency.strip().lower(), window.frequency)
            for method in methods:
                try:
                    forecasts[method] = list(store.forecast(
                        method, window.history, len(window.truth), frequency
                    ))
                except Exception as error:
                    failures[f"{method}|{type(error).__name__}"] += 1
            entries[window.task_id] = forecasts
            completed += 1
            if completed % args.checkpoint_every == 0:
                payload.update(entries=entries, failure_counts=dict(failures), complete=False)
                _write(args.output, payload)
                print(f"shard={args.shard_index} completed={completed}/{len(windows)}", flush=True)
    finally:
        store.close()
    payload.update(entries=entries, failure_counts=dict(failures), complete=True)
    _write(args.output, payload)
    print(f"wrote={args.output} tasks={len(entries)} methods={len(methods)}", flush=True)


if __name__ == "__main__":
    main()
