#!/usr/bin/env python3
"""Generate audited Moirai-2.0 anchors for official Train/Dev tasks only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolving_loop.official_benchmark_loader import (
    load_official_time_mmd,
    load_official_timesx,
)
from numerical_agent.tsfm.protocol import WorkerRequest
from numerical_agent.tsfm.workers.uni2ts import Uni2TSAdapter


def _write(path: Path, dataset: str, fingerprint: str, entries: dict, complete: bool) -> None:
    payload = {
        "schema": "official-anchor-cache-v1",
        "dataset": dataset,
        "model": "Salesforce/moirai-2.0-R-small",
        "official_source_fingerprint": fingerprint,
        "complete": complete,
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resume_entries(path: Path, dataset: str, fingerprint: str) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "official-anchor-cache-v1",
        "dataset": dataset,
        "model": "Salesforce/moirai-2.0-R-small",
        "official_source_fingerprint": fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"cannot resume: {key} mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("cannot resume: entries must be an object")
    return entries


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
    partitions = loader(
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    windows = list((*partitions.train, *partitions.dev))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        windows = windows[: args.limit]
    entries = (
        _resume_entries(args.output, args.dataset, partitions.source_fingerprint)
        if args.resume
        else {}
    )
    selected_ids = {window.task_id for window in windows}
    if not set(entries).issubset(selected_ids):
        raise ValueError("cannot resume: cache contains task IDs outside this run")
    adapter = Uni2TSAdapter()
    completed = len(entries)
    for window in windows:
        if window.task_id in entries:
            continue
        forecast = adapter.forecast(
            WorkerRequest(
                request_id=window.task_id,
                provider="uni2ts",
                checkpoint="Salesforce/moirai-2.0-R-small",
                history=window.history,
                horizon=len(window.truth),
                frequency=window.frequency,
                runtime_options={"max_patch_tokens": 512, "patch_size": 16},
            )
        )
        entries[window.task_id] = {"moirai_2_0": list(forecast)}
        completed += 1
        if completed % args.checkpoint_every == 0:
            _write(args.output, args.dataset, partitions.source_fingerprint, entries, False)
            print(f"completed={completed}/{len(windows)}", flush=True)
    _write(
        args.output,
        args.dataset,
        partitions.source_fingerprint,
        entries,
        args.limit is None,
    )
    print(f"wrote={args.output} entries={len(entries)}", flush=True)


if __name__ == "__main__":
    main()
