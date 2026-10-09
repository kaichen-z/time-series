#!/usr/bin/env python3
"""Generate TimesFM-2.5 anchors for official Train/Dev tasks only."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evolving_loop.official_benchmark_loader import (
    load_official_time_mmd,
    load_official_timesx,
)


def _write(path: Path, dataset: str, fingerprint: str, entries: dict, complete: bool) -> None:
    payload = {
        "schema": "official-anchor-cache-v1",
        "dataset": dataset,
        "model": "google/timesfm-2.5-200m-pytorch",
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
        "model": "google/timesfm-2.5-200m-pytorch",
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    loader = load_official_time_mmd if args.dataset == "time_mmd" else load_official_timesx
    partitions = loader(
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )
    windows = list((*partitions.train, *partitions.dev))
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    windows = [
        window
        for index, window in enumerate(windows)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        windows = windows[: args.limit]

    import numpy as np
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    entries = (
        _resume_entries(args.output, args.dataset, partitions.source_fingerprint)
        if args.resume
        else {}
    )
    selected_ids = {window.task_id for window in windows}
    if not set(entries).issubset(selected_ids):
        raise ValueError("cannot resume: cache contains task IDs outside this run")
    grouped = defaultdict(list)
    for window in windows:
        if window.task_id not in entries:
            grouped[len(window.truth)].append(window)
    completed = len(entries)
    last_checkpoint = completed
    for horizon, group in sorted(grouped.items()):
        # TimesFM decodes to ForecastConfig.max_horizon internally even when the
        # requested horizon is shorter. Recompile per official horizon so, for
        # example, a 6-step monthly task does not needlessly decode 336 steps.
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset : offset + args.batch_size]
            predictions, _ = model.forecast(
                horizon=horizon, inputs=[list(window.history) for window in batch]
            )
            for window, prediction in zip(batch, np.asarray(predictions, dtype=float)):
                entries[window.task_id] = {"timesfm_2_5": prediction[:horizon].tolist()}
            completed += len(batch)
            if completed - last_checkpoint >= args.checkpoint_every:
                _write(args.output, args.dataset, partitions.source_fingerprint, entries, False)
                last_checkpoint = completed
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
