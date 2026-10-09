#!/usr/bin/env python3
"""Materialize the five reviewed Combined policies on official Train+Dev tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from common.payload import canonical_json_bytes, canonical_json_line_bytes
from evolving_loop.official_benchmark_loader import (
    load_official_time_mmd,
    load_official_timesx,
)
from numerical_agent.evolution.execution import NOT_APPLICABLE, SUCCESS, Outcome
from numerical_agent.evolution.portfolio import (
    PolicyPortfolio,
    combine_materialized_outcome,
)

STAT_PARENTS = frozenset(
    {
        "seasonal_naive",
        "holt_damped_trend",
        "croston_sba",
        "robust_loess_trend",
        "median_seasonal_profile_forecast",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"invalid cache: {path}")
    return entries


def _statistical(root: Path, dataset: str) -> dict[str, dict[str, list[float]]]:
    selected: dict[str, dict[str, list[float]]] = {}
    paths = sorted(root.glob(f"{dataset}_stat_py310_shard_*.json"))
    expected = 12 if dataset == "timesx" else 24
    if len(paths) != expected:
        raise ValueError(f"expected {expected} statistical shards, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise ValueError(f"incomplete shard: {path}")
        for task_id, forecasts in payload["entries"].items():
            if task_id in selected:
                raise ValueError(f"duplicate statistical task: {task_id}")
            selected[task_id] = {
                name: values for name, values in forecasts.items() if name in STAT_PARENTS
            }
    return selected


def _materialize_dataset(
    *, dataset: str, partitions, cache_root: Path, output: Path
) -> dict[str, object]:
    foundation = _entries(cache_root / f"{dataset}_three_foundation_anchor_cache.json")
    chronos = _entries(cache_root / f"{dataset}_chronos_anchors.json")
    granite = _entries(cache_root / f"{dataset}_granite_anchors.json")
    statistical = _statistical(cache_root, dataset)
    policies = PolicyPortfolio.flagship5().combined
    windows = (*partitions.train, *partitions.dev)
    fallback_counts = {policy.name: 0 for policy in policies}
    granite_to_toto = 0
    task_count = 0
    with output.open("wb") as handle:
        for window in windows:
            task_id = window.task_id
            leaves = dict(foundation[task_id])
            leaves.update(chronos[task_id])
            granite_values = granite.get(task_id, {}).get("granite_ttm_r2")
            if granite_values is None:
                granite_values = leaves["toto_2_0"]
                granite_to_toto += 1
            leaves["granite_ttm_r2"] = granite_values
            leaves.update(statistical.get(task_id, {}))
            forecasts: dict[str, list[float]] = {}
            details: dict[str, str] = {}
            for policy in policies:
                parent_outcomes = {
                    parent: (
                        Outcome(parent, task_id, SUCCESS, forecast=tuple(leaves[parent]))
                        if parent in leaves
                        else Outcome(parent, task_id, NOT_APPLICABLE, detail="not available")
                    )
                    for parent in policy.parents
                }
                result = combine_materialized_outcome(
                    policy,
                    parent_outcomes,
                    task_id=task_id,
                    history=window.history,
                    horizon=len(window.truth),
                    frequency=window.frequency,
                )
                if result.status != SUCCESS or len(result.forecast) != len(window.truth):
                    raise ValueError(f"combined failure: {task_id}:{policy.name}:{result.status}")
                if not all(math.isfinite(value) for value in result.forecast):
                    raise ValueError(f"nonfinite combined forecast: {task_id}:{policy.name}")
                forecasts[policy.name] = list(result.forecast)
                if result.detail:
                    fallback_counts[policy.name] += 1
                    details[policy.name] = result.detail
            handle.write(
                canonical_json_line_bytes(
                    {"task_id": task_id, "forecasts": forecasts, "fallbacks": details}
                )
            )
            task_count += 1
    return {
        "task_count": task_count,
        "policy_count": len(policies),
        "forecast_count": task_count * len(policies),
        "fallback_counts": fallback_counts,
        "granite_to_toto_count": granite_to_toto,
        "sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timesx-root", type=Path, required=True)
    parser.add_argument("--time-mmd-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timesx = load_official_timesx(
        manifest_path=args.manifest,
        repository_root=args.timesx_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    time_mmd = load_official_time_mmd(
        manifest_path=args.manifest,
        repository_root=args.time_mmd_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    audit = {
        "schema": "official-combined-materialization-v1",
        "policies": [policy.to_payload() for policy in PolicyPortfolio.flagship5().combined],
        "datasets": {
            "timesx": _materialize_dataset(
                dataset="timesx",
                partitions=timesx,
                cache_root=args.cache_root,
                output=args.output / "timesx_combined.jsonl",
            ),
            "time_mmd": _materialize_dataset(
                dataset="time_mmd",
                partitions=time_mmd,
                cache_root=args.cache_root,
                output=args.output / "time_mmd_combined.jsonl",
            ),
        },
        "contains_test_ids": False,
        "contains_labels": False,
    }
    audit_path = args.output / "audit.json"
    audit_path.write_bytes(canonical_json_bytes(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
