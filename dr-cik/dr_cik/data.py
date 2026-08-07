"""Loaders for the Dr-CiK benchmark: the official sample dir and the full HF dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .models import Document, ForecastTask

DEFAULT_DATA_DIR = Path(
    os.environ.get("DR_CIK_DATA_DIR", "/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK")
)


def _as_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _task_from_raw(raw: dict[str, Any]) -> ForecastTask:
    """Parse one task in the canonical Dr-CiK JSON schema (sample dir or HF tar bundle)."""
    showcase = raw.get("showcase", {})
    entity = showcase.get("entity", {})
    variable = showcase.get("time_series_variable", {})
    metadata = raw.get("task_metadata", {})
    series = raw["series"]
    annotations = raw.get("annotations", {})

    documents = tuple(
        Document(
            document_id=str(item["document_id"]),
            text=str(item.get("content", "")),
            role=item.get("role"),
            subtype=item.get("subtype"),
        )
        for item in raw.get("documents", [])
    )
    future_values = series.get("future_values")
    gt_items = annotations.get("gt_evidence", [])
    gt_evidence = tuple(
        {"id": str(item.get("id", "")), "evidence": str(item.get("evidence", ""))} for item in gt_items
    )

    return ForecastTask(
        benchmark_id=str(raw["benchmark_id"]),
        entity_name=str(entity.get("name", "")),
        target_name=str(variable.get("name", "target")),
        target_description=str(metadata.get("target_description", "")),
        frequency=str(metadata.get("frequency", "unknown")),
        prediction_length=int(metadata.get("prediction_length") or len(series["future_timestamps"])),
        seasonal_period=metadata.get("seasonal_period"),
        history_timestamps=tuple(str(value) for value in series["history_timestamps"]),
        history_values=_as_float_tuple(series["history_values"]),
        future_timestamps=tuple(str(value) for value in series["future_timestamps"]),
        future_values=_as_float_tuple(future_values) if future_values else None,
        documents=documents,
        gt_evidence=gt_evidence,
        labels_public=bool(raw.get("labels_public", bool(future_values))),
    )


def load_sample_tasks(sample_dir: str | Path) -> list[ForecastTask]:
    """Load the official dependency-free sample layout (tasks/*.json with embedded document content)."""
    task_dir = Path(sample_dir).expanduser().resolve() / "tasks"
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Expected a tasks directory at {task_dir}")
    return [_task_from_raw(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(task_dir.glob("task_*.json"))]


def download_dataset(local_dir: str | Path = DEFAULT_DATA_DIR, revision: str | None = None) -> Path:
    """Snapshot-download the full ServiceNow/Dr-CiK dataset repo to local_dir."""
    from huggingface_hub import snapshot_download

    resolved = Path(local_dir).expanduser().resolve()
    snapshot_download(repo_id="ServiceNow/Dr-CiK", repo_type="dataset", local_dir=str(resolved), revision=revision)
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(data_dir: str | Path = DEFAULT_DATA_DIR, labels_public: bool | None = True) -> list[ForecastTask]:
    """Parse the downloaded tasks/documents/task_documents JSONL splits into ForecastTasks."""
    root = Path(data_dir).expanduser().resolve() / "data"
    task_rows = _read_jsonl(root / "tasks" / "train.jsonl")
    document_rows = _read_jsonl(root / "documents" / "train.jsonl")
    link_rows = _read_jsonl(root / "task_documents" / "train.jsonl")

    text_by_id = {str(row["document_id"]): str(row.get("text", "")) for row in document_rows}
    role_by_task_doc: dict[tuple[str, str], tuple[str | None, str | None]] = {}
    for row in link_rows:
        key = (str(row["benchmark_id"]), str(row["document_id"]))
        role_by_task_doc[key] = (row.get("role"), row.get("subtype"))

    tasks: list[ForecastTask] = []
    for row in task_rows:
        if labels_public is not None and bool(row.get("labels_public")) is not labels_public:
            continue
        benchmark_id = str(row["benchmark_id"])
        documents = tuple(
            Document(
                document_id=str(document_id),
                text=text_by_id.get(str(document_id), ""),
                role=role_by_task_doc.get((benchmark_id, str(document_id)), (None, None))[0],
                subtype=role_by_task_doc.get((benchmark_id, str(document_id)), (None, None))[1],
            )
            for document_id in row.get("document_ids", [])
        )
        future_values = row.get("future_values")
        gt_items = row.get("gt_evidence") or []
        gt_evidence = tuple(
            {"id": str(item.get("id", "")), "evidence": str(item.get("evidence", ""))} for item in gt_items
        )
        tasks.append(
            ForecastTask(
                benchmark_id=benchmark_id,
                entity_name=str(row.get("entity_name") or ""),
                target_name=str(row.get("time_series_variable") or "target"),
                target_description=str(row.get("target_description") or ""),
                frequency=str(row.get("frequency") or "unknown"),
                prediction_length=int(row.get("prediction_length") or len(row["future_timestamps"])),
                seasonal_period=row.get("seasonal_period"),
                history_timestamps=tuple(str(value) for value in row["history_timestamps"]),
                history_values=_as_float_tuple(row["history_values"]),
                future_timestamps=tuple(str(value) for value in row["future_timestamps"]),
                future_values=_as_float_tuple(future_values) if future_values else None,
                documents=documents,
                gt_evidence=gt_evidence,
                labels_public=bool(row.get("labels_public", bool(future_values))),
            )
        )
    return tasks
