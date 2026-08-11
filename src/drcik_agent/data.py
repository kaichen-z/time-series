from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import Document, ForecastTask


def _as_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _optional_positive_int(value: Any) -> int | None:
    """Accept step-count seasonality and safely ignore offset aliases such as `5T`/`D`."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def task_from_raw(raw: dict[str, Any], document_dir: Path | None = None) -> ForecastTask:
    """Parse the canonical file-per-task Dr-CiK representation."""
    showcase = raw.get("showcase", {})
    entity = showcase.get("entity", {})
    variable = showcase.get("time_series_variable", {})
    metadata = raw.get("task_metadata", {})
    series = raw.get("series", {})
    annotations = raw.get("annotations", {})

    documents: list[Document] = []
    for item in raw.get("documents", []):
        text = item.get("content") or item.get("text") or ""
        if not text and document_dir is not None:
            candidate = document_dir / f"{item['document_id']}.md"
            if not candidate.exists() and item.get("path"):
                candidate = document_dir.parent / item["path"]
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
        documents.append(
            Document(
                document_id=str(item["document_id"]),
                text=text,
                role=item.get("role"),
                subtype=item.get("subtype"),
            )
        )

    future_raw = series.get("future_values") or []
    public = bool(raw.get("labels_public", bool(future_raw)))
    gt_items = annotations.get("gt_evidence") or raw.get("gt_evidence") or []
    gt_evidence = tuple(
        str(item.get("evidence", "")) if isinstance(item, dict) else str(item)
        for item in gt_items
        if item
    )

    return ForecastTask(
        benchmark_id=str(raw["benchmark_id"]),
        entity_name=str(_first(entity, "name", default=showcase.get("profile", {}).get("name", ""))),
        target_name=str(_first(variable, "name", default=metadata.get("target_description", "target"))),
        target_description=str(metadata.get("target_description", "")),
        frequency=str(metadata.get("frequency", "unknown")),
        prediction_length=int(metadata.get("prediction_length") or len(series.get("future_timestamps", []))),
        seasonal_period=_optional_positive_int(metadata.get("seasonal_period")),
        history_timestamps=tuple(str(value) for value in series["history_timestamps"]),
        history_values=_as_float_tuple(series["history_values"]),
        future_timestamps=tuple(str(value) for value in series["future_timestamps"]),
        future_values=_as_float_tuple(future_raw) if future_raw else None,
        documents=tuple(documents),
        gt_evidence=gt_evidence,
        labels_public=public,
    )


def load_sample_tasks(sample_dir: str | Path) -> list[ForecastTask]:
    """Load the dependency-free sample layout shipped by ServiceNow/Dr-CiK."""
    root = Path(sample_dir).expanduser().resolve()
    task_dir = root / "tasks"
    document_dir = root / "documents"
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Expected a tasks directory at {task_dir}")
    return [
        task_from_raw(json.loads(path.read_text(encoding="utf-8")), document_dir)
        for path in sorted(task_dir.glob("task_*.json"))
    ]


def task_from_normalized(
    row: dict[str, Any],
    document_by_id: dict[str, str],
    link_by_task: dict[str, dict[str, tuple[str | None, str | None]]],
) -> ForecastTask:
    """Parse one row from the normalized Hugging Face `tasks` config."""
    benchmark_id = str(row["benchmark_id"])
    labels = link_by_task.get(benchmark_id, {})
    documents = tuple(
        Document(
            document_id=str(document_id),
            text=document_by_id.get(str(document_id), ""),
            role=labels.get(str(document_id), (None, None))[0],
            subtype=labels.get(str(document_id), (None, None))[1],
        )
        for document_id in row.get("document_ids", [])
    )
    future_raw = row.get("future_values") or []
    gt_items = row.get("gt_evidence") or []
    gt_evidence = tuple(
        str(item.get("evidence", "")) if isinstance(item, dict) else str(item)
        for item in gt_items
        if item
    )
    return ForecastTask(
        benchmark_id=benchmark_id,
        entity_name=str(row.get("entity_name") or row.get("profile_name") or ""),
        target_name=str(row.get("time_series_variable") or "target"),
        target_description=str(row.get("target_description") or ""),
        frequency=str(row.get("frequency") or "unknown"),
        prediction_length=int(row.get("prediction_length") or len(row.get("future_timestamps", []))),
        seasonal_period=_optional_positive_int(row.get("seasonal_period")),
        history_timestamps=tuple(str(value) for value in row["history_timestamps"]),
        history_values=_as_float_tuple(row["history_values"]),
        future_timestamps=tuple(str(value) for value in row["future_timestamps"]),
        future_values=_as_float_tuple(future_raw) if future_raw else None,
        documents=documents,
        gt_evidence=gt_evidence,
        labels_public=bool(row.get("labels_public", bool(future_raw))),
    )


def load_huggingface_tasks(labels_public: bool | None = True) -> list[ForecastTask]:
    """Load normalized Dr-CiK rows from Hugging Face.

    Prefer the ``datasets`` package when installed. The official repository
    also publishes the same configs as JSONL, so a lightweight Hub-only path
    avoids a heavy Arrow/aiohttp dependency on constrained environments.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face loading requires: pip install -e '.[huggingface]'"
            ) from exc

        def jsonl_rows(filename: str) -> list[dict[str, Any]]:
            path = Path(
                hf_hub_download(
                    repo_id="ServiceNow/Dr-CiK",
                    repo_type="dataset",
                    filename=filename,
                )
            )
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        task_rows = jsonl_rows("data/tasks/train.jsonl")
        document_rows = jsonl_rows("data/documents/train.jsonl")
        link_rows = jsonl_rows("data/task_documents/train.jsonl")
    else:
        task_rows = load_dataset("ServiceNow/Dr-CiK", "tasks", split="train")
        document_rows = load_dataset("ServiceNow/Dr-CiK", "documents", split="train")
        link_rows = load_dataset("ServiceNow/Dr-CiK", "task_documents", split="train")

    document_by_id = {str(row["document_id"]): str(row["text"]) for row in document_rows}
    link_by_task: dict[str, dict[str, tuple[str | None, str | None]]] = defaultdict(dict)
    for row in link_rows:
        link_by_task[str(row["benchmark_id"])][str(row["document_id"])] = (
            row.get("role"),
            row.get("subtype"),
        )

    selected = list(task_rows)
    if labels_public is not None:
        selected = [
            row for row in selected if bool(row["labels_public"]) is labels_public
        ]
    return [task_from_normalized(dict(row), document_by_id, link_by_task) for row in selected]
