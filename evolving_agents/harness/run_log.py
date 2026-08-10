"""Append-only JSONL run records: compact, hash-only, safe to keep for a whole run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROXY_NOTE = (
    "smae/srmse/scrps are local development proxies computed by dr_cik.evaluation, "
    "not Dr-CiK's private official scorer."
)


def append_run_record(runs_dir: str | Path, filename: str, record: dict[str, Any]) -> Path:
    """Append one JSON line and flush, so a crash keeps every task completed before it."""
    directory = Path(runs_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
    return path


def build_record(
    task_id: str,
    loop: str,
    bundle_versions: dict[str, str],
    score: float,
    llm_calls: list[dict[str, Any]],
    generation: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one run record, keeping only call hashes so the file stays small."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "dr_cik",
        "task_id": task_id,
        "loop": loop,
        "generation": generation,
        "bundle_versions": bundle_versions,
        "llm_calls": [
            {name: call.get(name) for name in ("model_id", "prompt_hash", "cache_hit", "latency_s", "draw_index")}
            for call in llm_calls
        ],
        "score": score,
        **(extra or {}),
    }
