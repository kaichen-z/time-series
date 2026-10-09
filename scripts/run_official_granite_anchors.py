#!/usr/bin/env python3
"""Generate audited Granite TTM-R2 anchors for supported official Train/Dev tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolving_loop.official_benchmark_loader import load_official_time_mmd, load_official_timesx
from numerical_agent.tsfm import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest
from numerical_agent.tsfm.workers.granite import GraniteAdapter

MODEL = "ibm-granite/granite-timeseries-ttm-r2"
KEY = "granite_ttm_r2"
METHOD_ID = "method_tsfm_0006"


def _write(path: Path, dataset: str, fingerprint: str, entries: dict, failures: dict,
           attempted_all: bool) -> None:
    payload = {"schema": "official-anchor-cache-v1", "dataset": dataset, "model": MODEL,
               "official_source_fingerprint": fingerprint, "complete": attempted_all,
               "entries": entries, "failures": failures}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resume(path: Path, dataset: str, fingerprint: str) -> tuple[dict, dict]:
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema": "official-anchor-cache-v1", "dataset": dataset,
                "model": MODEL, "official_source_fingerprint": fingerprint}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"cannot resume: {key} mismatch")
    entries, failures = payload.get("entries"), payload.get("failures", {})
    if not isinstance(entries, dict) or not isinstance(failures, dict):
        raise ValueError("cannot resume: entries/failures must be objects")
    return entries, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("time_mmd", "timesx"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    loader = load_official_time_mmd if args.dataset == "time_mmd" else load_official_timesx
    partitions = loader(manifest_path=args.manifest, repository_root=args.repository_root,
                        anchor_cache=None, allow_unbound_anchors=True)
    windows = list((*partitions.train, *partitions.dev))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        windows = windows[:args.limit]
    entries, failures = (_resume(args.output, args.dataset, partitions.source_fingerprint)
                         if args.resume else ({}, {}))
    selected = {window.task_id for window in windows}
    if not (set(entries) | set(failures)).issubset(selected):
        raise ValueError("cannot resume: cache contains task IDs outside this run")
    manifest = ManifestRegistry.load_default()[METHOD_ID]
    adapter = GraniteAdapter()
    completed = len(entries) + len(failures)
    for window in windows:
        if window.task_id in entries or window.task_id in failures:
            continue
        try:
            forecast = adapter.forecast(WorkerRequest(
                request_id=window.task_id, provider=manifest.adapter,
                checkpoint=manifest.checkpoint, history=window.history,
                horizon=len(window.truth), frequency=window.frequency,
                runtime_options=dict(manifest.runtime_options)))
            entries[window.task_id] = {KEY: list(forecast)}
        except Exception as error:
            failures[window.task_id] = {"type": type(error).__name__, "message": str(error)}
        completed += 1
        if completed % args.checkpoint_every == 0:
            _write(args.output, args.dataset, partitions.source_fingerprint,
                   entries, failures, False)
            print(f"attempted={completed}/{len(windows)} successes={len(entries)} failures={len(failures)}", flush=True)
    _write(args.output, args.dataset, partitions.source_fingerprint, entries, failures,
           args.limit is None)
    print(f"wrote={args.output} successes={len(entries)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
