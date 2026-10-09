#!/usr/bin/env python3
"""Export a label-free Train/Dev handoff for LLM document processing.

The output contains opaque task IDs and public metadata/documents only.  It
deliberately excludes numerical histories, Train/Dev labels, forecasts, anchor
caches, and all Test identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common.payload import canonical_json_bytes, canonical_json_line_bytes
from evolving_loop.official_benchmark_loader import (
    OfficialBenchmarkPartitions,
    load_official_time_mmd,
    load_official_timesx,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_line_bytes(row))


def _base_rows(partitions: OfficialBenchmarkPartitions) -> list[tuple[str, object]]:
    return [("train", window) for window in partitions.train] + [
        ("dev", window) for window in partitions.dev
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timesx-root", type=Path, required=True)
    parser.add_argument("--time-mmd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_copy = args.output / "alignment_manifest.json"
    manifest_copy.write_bytes(args.manifest.read_bytes())

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

    timesx_rows = []
    for split, window in _base_rows(timesx):
        timesx_rows.append(
            {
                "task_id": window.task_id,
                "split": split,
                "group_id": window.group_id,
                "frequency": window.frequency,
                "horizon": len(window.truth),
                "history_timestamps": list(window.history_timestamps),
                "future_timestamps": list(window.future_timestamps),
                "target_description": window.target_description,
                "documents": [
                    {"document_id": doc.document_id, "content": doc.content}
                    for doc in window.documents
                ],
            }
        )

    time_mmd_rows = [
        {
            "task_id": window.task_id,
            "split": split,
            "group_id": window.group_id,
            "frequency": window.frequency,
            "horizon": len(window.truth),
        }
        for split, window in _base_rows(time_mmd)
    ]

    timesx_path = args.output / "timesx_train_dev_documents.jsonl"
    time_mmd_path = args.output / "time_mmd_train_dev_task_ids.jsonl"
    _write_jsonl(timesx_path, timesx_rows)
    _write_jsonl(time_mmd_path, time_mmd_rows)

    receipt = {
        "schema": "official-llm-handoff-v1",
        "contains_test_ids": False,
        "contains_labels": False,
        "contains_numeric_history": False,
        "sources": {
            "timesx": {
                "url": "https://github.com/haoxin1998/TimesX-project",
                "commit": "26bc70cfa669407c71ae1ad6ce2194be9cfe9a0b",
                "source_fingerprint": timesx.source_fingerprint,
            },
            "time_mmd": {
                "url": "https://github.com/AdityaLab/MM-TSFlib",
                "commit": "e789ce78c9bafd8e3ba0d8850f9ad2becbe83548",
                "source_fingerprint": time_mmd.source_fingerprint,
            },
        },
        "files": {
            manifest_copy.name: {
                "sha256": _sha256(manifest_copy),
            },
            timesx_path.name: {
                "rows": len(timesx_rows),
                "documents": sum(len(row["documents"]) for row in timesx_rows),
                "sha256": _sha256(timesx_path),
            },
            time_mmd_path.name: {
                "rows": len(time_mmd_rows),
                "sha256": _sha256(time_mmd_path),
            },
        },
    }
    receipt_path = args.output / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
