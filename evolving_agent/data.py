"""Dr-CiK loading with an explicit numeric/context information boundary."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_TASKS_FILE = Path(
    "/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl"
)


@dataclass(frozen=True)
class Task:
    """A single numeric forecasting task; no field here can carry document/text content."""

    task_id: str
    history_values: tuple[float, ...]
    future_values: tuple[float, ...]
    prediction_length: int
    frequency: str
    seasonal_period: str | None
    entity_name: str


@dataclass(frozen=True)
class Document:
    """One corpus document. ``role`` is an evaluation label, never prompt input."""

    document_id: str
    content: str
    role: str | None = None
    subtype: str | None = None


@dataclass(frozen=True)
class ContextTask:
    """Full task held by the harness; agents receive only role-specific projections."""

    numeric: Task
    target_name: str
    target_description: str
    history_timestamps: tuple[str, ...]
    future_timestamps: tuple[str, ...]
    documents: tuple[Document, ...]
    gt_evidence: tuple[str, ...] = ()
    labels_public: bool = True

    def numeric_view(self) -> Task:
        """Return the numbers-only inference view, with resolved future labels removed."""
        return replace(self.numeric, future_values=())

    def retrieval_view(self) -> dict:
        """Return the label-free view that may be shown to the Retrieval Agent."""
        return {
            "task_id": self.numeric.task_id,
            "entity_name": self.numeric.entity_name,
            "target_name": self.target_name,
            "target_description": self.target_description,
            "frequency": self.numeric.frequency,
            "prediction_length": self.numeric.prediction_length,
            "history_timestamps": list(self.history_timestamps),
            "future_timestamps": list(self.future_timestamps),
            "documents": [
                {"document_id": document.document_id, "content": document.content}
                for document in self.documents
            ],
        }


def _series(record: dict) -> dict:
    return record.get("series", record)


def _metadata(record: dict) -> dict:
    return record.get("task_metadata", record)


def _showcase(record: dict) -> dict:
    return record.get("showcase", record)


def _is_labeled(record: dict) -> bool:
    """A record is usable only if it has real, non-null ground-truth future values."""
    future = _series(record).get("future_values")
    return bool(future) and future[0] is not None


def _to_task(record: dict) -> Task:
    """Extract exactly the numeric fields a Task needs, ignoring everything else in the record."""
    series = _series(record)
    metadata = _metadata(record)
    showcase = _showcase(record)
    entity = showcase.get("entity", {})
    variable = showcase.get("time_series_variable", {})
    return Task(
        task_id=record["benchmark_id"],
        history_values=tuple(float(value) for value in series["history_values"]),
        future_values=tuple(float(value) for value in series["future_values"]),
        prediction_length=int(metadata["prediction_length"]),
        frequency=str(metadata["frequency"]),
        seasonal_period=metadata.get("seasonal_period"),
        entity_name=str(record.get("entity_name") or entity.get("name") or "unknown"),
    )


def _to_context_task(record: dict) -> ContextTask:
    series = _series(record)
    metadata = _metadata(record)
    showcase = _showcase(record)
    variable = showcase.get("time_series_variable", {})
    annotations = record.get("annotations", {})
    raw_evidence = annotations.get("gt_evidence", record.get("gt_evidence", ()))
    evidence = tuple(
        str(item.get("evidence", "")) if isinstance(item, dict) else str(item)
        for item in raw_evidence
    )
    documents = tuple(
        Document(
            document_id=str(item["document_id"]),
            content=str(item.get("content", "")),
            role=item.get("role"),
            subtype=item.get("subtype"),
        )
        for item in record.get("documents", ())
    )
    return ContextTask(
        numeric=_to_task(record),
        target_name=str(record.get("target_name") or variable.get("name") or "target"),
        target_description=str(metadata.get("target_description", "")),
        history_timestamps=tuple(str(value) for value in series.get("history_timestamps", ())),
        future_timestamps=tuple(str(value) for value in series.get("future_timestamps", ())),
        documents=documents,
        gt_evidence=tuple(item for item in evidence if item),
        labels_public=bool(record.get("labels_public", True)),
    )


def load_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE) -> list[Task]:
    """Load every labeled task from a Dr-CiK-style JSONL file."""
    tasks: list[Task] = []
    with open(tasks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if _is_labeled(record):
                tasks.append(_to_task(record))
    return tasks


def load_context_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE) -> list[ContextTask]:
    """Load full tasks from a Dr-CiK task directory, JSON object, or JSONL file."""
    tasks: list[ContextTask] = []
    path = Path(tasks_file)
    if path.is_dir():
        records = [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("*.json"))
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
    else:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for record in records:
        if _is_labeled(record):
            tasks.append(_to_context_task(record))
    return tasks


def split_tasks(
    tasks: list[Task], seed: int = 7, test_fraction: float = 0.3
) -> tuple[list[Task], list[Task]]:
    """Split by entity (not by task) so no underlying series straddles train/test."""
    entities = sorted({task.entity_name for task in tasks})
    rng = random.Random(seed)
    rng.shuffle(entities)
    n_test = max(1, round(len(entities) * test_fraction)) if entities else 0
    test_entities = set(entities[:n_test])
    train = [t for t in tasks if t.entity_name not in test_entities]
    test = [t for t in tasks if t.entity_name in test_entities]
    return train, test
