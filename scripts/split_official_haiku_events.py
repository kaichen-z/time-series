#!/usr/bin/env python3
"""Validate and split the official Haiku event-card handoff by benchmark."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ALLOWED_DIRECTIONS = {"up", "down", "mixed", "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield line_number, value


def json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def require_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: document/task ID must be a non-empty string")
    return value


def validate_date(value: Any, *, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{context}: date must be YYYY-MM-DD or null")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid calendar date {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{context}: date must use canonical YYYY-MM-DD")


def document_sets(
    root: Path,
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]], dict[str, str]]:
    dataset_ids: dict[str, set[str]] = {"timesx": set(), "time_mmd": set()}
    split_ids: dict[str, dict[str, set[str]]] = {
        "timesx": defaultdict(set),
        "time_mmd": defaultdict(set),
    }
    timesx_forecast_starts: dict[str, str] = {}

    timesx_path = root / "timesx_train_dev_documents.jsonl"
    for line_number, row in read_jsonl(timesx_path):
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise ValueError(f"{timesx_path}:{line_number}: invalid split {split!r}")
        documents = row.get("documents")
        if not isinstance(documents, list):
            raise ValueError(f"{timesx_path}:{line_number}: documents must be a list")
        future_timestamps = row.get("future_timestamps")
        if not isinstance(future_timestamps, list) or not future_timestamps:
            raise ValueError(f"{timesx_path}:{line_number}: future_timestamps must be nonempty")
        forecast_start = str(future_timestamps[0])[:10]
        validate_date(forecast_start, context=f"{timesx_path}:{line_number}:forecast_start")
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise ValueError(f"{timesx_path}:{line_number}: documents[{index}] must be an object")
            document_id = require_id(document.get("document_id"), context=f"{timesx_path}:{line_number}")
            dataset_ids["timesx"].add(document_id)
            split_ids["timesx"][split].add(document_id)
            timesx_forecast_starts[document_id] = forecast_start

    time_mmd_docs_path = root / "time_mmd_documents.jsonl"
    for line_number, row in read_jsonl(time_mmd_docs_path):
        document_id = require_id(row.get("document_id"), context=f"{time_mmd_docs_path}:{line_number}")
        if document_id in dataset_ids["time_mmd"]:
            raise ValueError(f"{time_mmd_docs_path}:{line_number}: duplicate document_id {document_id}")
        dataset_ids["time_mmd"].add(document_id)

    time_mmd_tasks_path = root / "time_mmd_train_dev_task_ids.jsonl"
    referenced: set[str] = set()
    for line_number, row in read_jsonl(time_mmd_tasks_path):
        split = row.get("split")
        if split not in {"train", "dev"}:
            raise ValueError(f"{time_mmd_tasks_path}:{line_number}: invalid split {split!r}")
        document_ids = row.get("document_ids")
        if not isinstance(document_ids, list):
            raise ValueError(f"{time_mmd_tasks_path}:{line_number}: document_ids must be a list")
        for value in document_ids:
            document_id = require_id(value, context=f"{time_mmd_tasks_path}:{line_number}")
            referenced.add(document_id)
            split_ids["time_mmd"][split].add(document_id)
    if referenced != dataset_ids["time_mmd"]:
        raise ValueError(
            "Time-MMD task/document closure mismatch: "
            f"missing={len(referenced - dataset_ids['time_mmd'])}, "
            f"unreferenced={len(dataset_ids['time_mmd'] - referenced)}"
        )
    overlap = dataset_ids["timesx"] & dataset_ids["time_mmd"]
    if overlap:
        raise ValueError(f"benchmark document namespaces overlap ({len(overlap)} IDs)")
    return dataset_ids, split_ids, timesx_forecast_starts


def validate_card(
    row: dict[str, Any], *, line_number: int, manifest: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    if set(row) != {"document_id", "extractor", "events"}:
        raise ValueError(f"events:{line_number}: unexpected top-level keys {sorted(row)}")
    document_id = require_id(row.get("document_id"), context=f"events:{line_number}")
    extractor = row.get("extractor")
    expected_extractor = {
        "provider": "anthropic",
        "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"],
    }
    if extractor != expected_extractor:
        raise ValueError(f"events:{line_number}: extractor binding mismatch")
    events = row.get("events")
    if not isinstance(events, list):
        raise ValueError(f"events:{line_number}: events must be a list")
    errors: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        context = f"events:{line_number}:events[{index}]"
        if not isinstance(event, dict) or set(event) != {
            "time_start", "time_end", "direction", "confidence", "evidence"
        }:
            raise ValueError(f"{context}: invalid event shape")
        try:
            validate_date(event["time_start"], context=f"{context}.time_start")
            validate_date(event["time_end"], context=f"{context}.time_end")
        except ValueError as exc:
            errors.append({
                "document_id": document_id,
                "event_index": index,
                "kind": "invalid_date",
                "detail": str(exc),
            })
        if event["time_start"] and event["time_end"] and event["time_start"] > event["time_end"]:
            errors.append({
                "document_id": document_id,
                "event_index": index,
                "kind": "reversed_date_range",
                "detail": f"time_start={event['time_start']}; time_end={event['time_end']}",
            })
        if event["direction"] not in ALLOWED_DIRECTIONS:
            errors.append({
                "document_id": document_id,
                "event_index": index,
                "kind": "invalid_direction",
                "detail": repr(event["direction"]),
            })
        confidence = event["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append({
                "document_id": document_id,
                "event_index": index,
                "kind": "invalid_confidence",
                "detail": repr(confidence),
            })
        if not isinstance(event["evidence"], str) or not event["evidence"].strip():
            errors.append({
                "document_id": document_id,
                "event_index": index,
                "kind": "empty_evidence",
                "detail": repr(event["evidence"]),
            })
    return document_id, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.handoff_root / "haiku_output" / "events.jsonl"
    manifest_path = args.handoff_root / "haiku_output" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(source) != manifest.get("output_sha256"):
        raise ValueError("events.jsonl SHA-256 does not match manifest")

    dataset_ids, split_ids, timesx_forecast_starts = document_sets(args.handoff_root)
    expected_ids = dataset_ids["timesx"] | dataset_ids["time_mmd"]
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {"timesx": [], "time_mmd": []}
    valid_rows_by_dataset: dict[str, list[dict[str, Any]]] = {"timesx": [], "time_mmd": []}
    seen: set[str] = set()
    event_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    validation_errors: list[dict[str, Any]] = []
    for line_number, row in read_jsonl(source):
        document_id, row_errors = validate_card(row, line_number=line_number, manifest=manifest)
        if document_id in seen:
            raise ValueError(f"events:{line_number}: duplicate document_id {document_id}")
        seen.add(document_id)
        if document_id in dataset_ids["timesx"]:
            dataset = "timesx"
        elif document_id in dataset_ids["time_mmd"]:
            dataset = "time_mmd"
        else:
            raise ValueError(f"events:{line_number}: unknown document_id {document_id}")
        if dataset == "timesx" and document_id.endswith("_scenario"):
            forecast_start = timesx_forecast_starts[document_id]
            for event_index, event in enumerate(row["events"]):
                if event["time_start"] is not None and event["time_start"] >= forecast_start:
                    row_errors.append({
                        "document_id": document_id,
                        "event_index": event_index,
                        "kind": "future_scenario_event",
                        "detail": (
                            f"time_start={event['time_start']}; "
                            f"forecast_start={forecast_start}"
                        ),
                    })
        for error in row_errors:
            error["dataset"] = dataset
        validation_errors.extend(row_errors)
        rows_by_dataset[dataset].append(row)
        invalid_event_indexes = {error["event_index"] for error in row_errors}
        valid_row = dict(row)
        valid_row["events"] = [
            event for index, event in enumerate(row["events"])
            if index not in invalid_event_indexes
        ]
        valid_rows_by_dataset[dataset].append(valid_row)
        event_counts[dataset] += len(row["events"])
        empty_counts[dataset] += not row["events"]
    if seen != expected_ids:
        raise ValueError(
            f"event-card closure mismatch: missing={len(expected_ids - seen)}, extra={len(seen - expected_ids)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for dataset, rows in rows_by_dataset.items():
        output = args.output_dir / f"{dataset}_events.jsonl"
        with output.open("wb") as handle:
            for row in rows:
                handle.write(json_line(row))
        files[output.name] = {
            "sha256": sha256(output),
            "documents": len(rows),
            "events": event_counts[dataset],
            "empty_event_documents": empty_counts[dataset],
        }

        valid_output = args.output_dir / f"{dataset}_events_valid.jsonl"
        with valid_output.open("wb") as handle:
            for row in valid_rows_by_dataset[dataset]:
                handle.write(json_line(row))
        dataset_errors = [error for error in validation_errors if error["dataset"] == dataset]
        files[valid_output.name] = {
            "sha256": sha256(valid_output),
            "documents": len(valid_rows_by_dataset[dataset]),
            "events": sum(len(row["events"]) for row in valid_rows_by_dataset[dataset]),
            "quarantined_events": len({
                (error["document_id"], error["event_index"]) for error in dataset_errors
            }),
            "normalization": "invalid events quarantined; document IDs and all valid events unchanged",
        }

        membership = args.output_dir / f"{dataset}_document_split_membership.jsonl"
        with membership.open("wb") as handle:
            for document_id in sorted(dataset_ids[dataset]):
                splits = [split for split in ("train", "dev") if document_id in split_ids[dataset][split]]
                handle.write(json_line({"document_id": document_id, "splits": splits}))
        files[membership.name] = {
            "sha256": sha256(membership),
            "documents": len(dataset_ids[dataset]),
            "train_documents": len(split_ids[dataset]["train"]),
            "dev_documents": len(split_ids[dataset]["dev"]),
            "shared_train_dev_documents": len(split_ids[dataset]["train"] & split_ids[dataset]["dev"]),
        }

    errors_path = args.output_dir / "validation_errors.jsonl"
    with errors_path.open("wb") as handle:
        for error in validation_errors:
            handle.write(json_line(error))
    files[errors_path.name] = {
        "sha256": sha256(errors_path),
        "errors": len(validation_errors),
        "documents": len({error["document_id"] for error in validation_errors}),
    }

    receipt = {
        "schema": "official-haiku-benchmark-split-v1",
        "source_commit": "f8e4762185b354582363930ee219295aebee3c6e",
        "source_events_sha256": sha256(source),
        "source_manifest_sha256": sha256(manifest_path),
        "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"],
        "documents": len(seen),
        "events": sum(event_counts.values()),
        "exact_document_closure": True,
        "duplicate_document_ids": 0,
        "schema_and_calendar_dates_valid": not validation_errors,
        "validation_error_count": len(validation_errors),
        "contains_numeric_history_or_labels": False,
        "contains_test_ids": False,
        "files": files,
    }
    receipt_path = args.output_dir / "split_manifest.json"
    receipt_path.write_bytes(json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
