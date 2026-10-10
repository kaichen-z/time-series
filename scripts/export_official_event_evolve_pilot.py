#!/usr/bin/env python3
"""Build a deterministic Train-only event-card pilot for Meta-Harness V2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.payload import canonical_json_bytes, canonical_json_line_bytes
from evolving_loop.official_benchmark_loader import (
    load_official_time_mmd,
    load_official_timesx,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("timesx", "time_mmd"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--event-cards", type=Path, required=True)
    parser.add_argument("--task-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    args = parser.parse_args()
    if args.groups < 2:
        raise ValueError("--groups must be at least 2 for an internal Train/Dev split")

    loader = load_official_timesx if args.dataset == "timesx" else load_official_time_mmd
    partitions = loader(
        manifest_path=args.manifest,
        repository_root=args.repository_root,
        anchor_cache=None,
        allow_unbound_anchors=True,
    )

    index_rows = read_jsonl(args.task_index)
    train_index = {
        row["task_id"]: set(
            document["document_id"] for document in row.get("documents", [])
        ) if args.dataset == "timesx" else set(row.get("document_ids", []))
        for row in index_rows
        if row.get("split") == "train"
    }
    expected_train = 2173 if args.dataset == "timesx" else 45948
    if len(train_index) != expected_train:
        raise ValueError(
            f"{args.dataset} Train index count mismatch: {len(train_index)} != {expected_train}"
        )

    event_rows = read_jsonl(args.event_cards)
    cards: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id or document_id in cards:
            raise ValueError("event-card IDs must be unique non-empty strings")
        cards[document_id] = row

    grouped: dict[str, list[Any]] = {}
    for window in partitions.train:
        if window.task_id not in train_index:
            raise ValueError(f"official Train task missing from handoff index: {window.task_id}")
        actual_documents = {document.document_id for document in window.documents}
        if actual_documents != train_index[window.task_id]:
            raise ValueError(f"task/document join mismatch: {window.task_id}")
        grouped.setdefault(window.group_id, []).append(window)
    seed = f"official-event-card-pilot-v1:{args.dataset}"
    ordered_groups = sorted(
        grouped,
        key=lambda group: (hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(), group),
    )[: args.groups]
    selected = [
        min(grouped[group], key=lambda row: hashlib.sha256(row.task_id.encode()).hexdigest())
        for group in ordered_groups
    ]

    records = []
    used_document_ids: set[str] = set()
    used_events = 0
    for window in selected:
        documents = []
        for document in window.documents:
            if document.document_id not in cards:
                raise ValueError(f"missing event card: {document.document_id}")
            card = cards[document.document_id]
            used_document_ids.add(document.document_id)
            used_events += len(card["events"])
            event_text = json.dumps(
                card["events"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            documents.append({
                "document_id": document.document_id,
                "content": f"{document.content}\n\n[HAIKU_EVENT_CARDS]\n{event_text}",
            })
        history_timestamps = list(window.history_timestamps) or [
            f"history_{index}" for index in range(len(window.history))
        ]
        future_timestamps = list(window.future_timestamps) or [
            f"future_{index}" for index in range(len(window.truth))
        ]
        records.append({
            "benchmark_id": window.task_id,
            "labels_public": True,
            "series": {
                "history_values": list(window.history),
                "future_values": list(window.truth),
                "history_timestamps": history_timestamps,
                "future_timestamps": future_timestamps,
            },
            "task_metadata": {
                "prediction_length": len(window.truth),
                "frequency": window.frequency,
                "seasonal_period": None,
                "target_description": window.target_description,
            },
            "entity_name": window.group_id,
            "target_name": "target",
            "documents": documents,
            "gt_evidence": [],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"".join(canonical_json_line_bytes(record) for record in records))
    receipt = {
        "schema": "official-event-card-evolve-pilot-v1",
        "dataset": args.dataset,
        "selection": "groups by sha256(seed\\0group), one Train task per group by task-id sha256",
        "seed": seed,
        "source_fingerprint": partitions.source_fingerprint,
        "manifest_sha256": sha256(args.manifest),
        "event_cards_sha256": sha256(args.event_cards),
        "task_index_sha256": sha256(args.task_index),
        "task_count": len(records),
        "group_count": len(ordered_groups),
        "group_ids": ordered_groups,
        "task_ids": [window.task_id for window in selected],
        "document_count": len(used_document_ids),
        "event_count": used_events,
        "uses_official_train_only": True,
        "uses_external_dev": False,
        "uses_test_ids_or_labels": False,
        "tasks_sha256": sha256(args.output),
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
