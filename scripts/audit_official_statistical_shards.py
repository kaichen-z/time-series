#!/usr/bin/env python3
"""Audit and merge completed official statistical forecast shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from evolving_loop.official_benchmark_loader import load_official_time_mmd, load_official_timesx


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: object) -> str:
    data = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("time_mmd", "timesx"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    loader = load_official_time_mmd if args.dataset == "time_mmd" else load_official_timesx
    partitions = loader(
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    windows = tuple((*partitions.train, *partitions.dev))
    expected = {window.task_id: len(window.truth) for window in windows}
    if len(expected) != len(windows):
        raise ValueError("official task IDs are not unique")

    seen_task_ids: set[str] = set()
    methods: tuple[str, ...] | None = None
    failure_counts: Counter[str] = Counter()
    shard_indices: set[int] = set()
    shard_sha256: dict[str, str] = {}
    coverage = Counter[str]()
    total_forecasts = 0
    for path in args.shards:
        raw = path.read_bytes()
        shard_sha256[path.name] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if payload.get("schema") != "official-statistical-cache-v1":
            raise ValueError(f"bad schema: {path}")
        if payload.get("dataset") != args.dataset or not payload.get("complete"):
            raise ValueError(f"incomplete or mismatched shard: {path}")
        current_methods = tuple(payload.get("methods", ()))
        if methods is None:
            methods = current_methods
        elif methods != current_methods:
            raise ValueError(f"method-list mismatch: {path}")
        index = int(payload["shard_index"])
        if index in shard_indices:
            raise ValueError(f"duplicate shard index: {index}")
        shard_indices.add(index)
        failure_counts.update(payload.get("failure_counts", {}))
        for task_id, forecasts in payload.get("entries", {}).items():
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task ID: {task_id}")
            if task_id not in expected:
                raise ValueError(f"unexpected task ID: {task_id}")
            if not isinstance(forecasts, dict) or not set(forecasts).issubset(current_methods):
                raise ValueError(f"invalid forecast mapping for {task_id}")
            horizon = expected[task_id]
            for method, values in forecasts.items():
                if not isinstance(values, list) or len(values) != horizon:
                    raise ValueError(f"horizon mismatch: {task_id} {method}")
                if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                    raise ValueError(f"non-finite forecast: {task_id} {method}")
                coverage[method] += 1
                total_forecasts += 1
            seen_task_ids.add(task_id)
        del payload

    missing = sorted(set(expected) - seen_task_ids)
    extra = sorted(seen_task_ids - set(expected))
    if missing or extra:
        raise ValueError(f"task closure failed: missing={len(missing)} extra={len(extra)}")
    if methods is None or len(methods) != 93:
        raise ValueError(f"expected 93 methods, got {0 if methods is None else len(methods)}")

    audit = {
        "schema": "official-statistical-audit-v1",
        "dataset": args.dataset,
        "task_count": len(seen_task_ids),
        "method_count": len(methods),
        "shard_count": len(shard_indices),
        "exact_task_closure": True,
        "unique_task_ids": True,
        "finite_forecasts": True,
        "exact_horizons": True,
        "total_available_task_method_forecasts": total_forecasts,
        "producing_method_count": sum(value > 0 for value in coverage.values()),
        "coverage_by_method": dict(sorted(coverage.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "shard_sha256": dict(sorted(shard_sha256.items())),
    }
    audit_sha = write_json(args.audit_output, audit)
    print(json.dumps({**audit, "audit_sha256": audit_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
