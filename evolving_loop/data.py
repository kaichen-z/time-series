"""Dr-CiK loading with an explicit numeric/context information boundary."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

# The numeric half lives in common/ so numbers-only packages need not depend on this one.
from common.data import (  # noqa: F401  (re-exported for existing importers)
    DEFAULT_TASKS_FILE,
    Task,
    _future_values,
    _is_labeled,
    _metadata,
    _series,
    _showcase,
    _to_task,
    load_tasks,
)


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
    labels_public = bool(record.get("labels_public", _is_labeled(record))) and _is_labeled(record)
    documents = tuple(
        Document(
            document_id=str(item["document_id"]),
            content=str(item.get("content", "")),
            # Hidden inference never needs evaluator-only relevance labels, even
            # if a local export accidentally retained them.
            role=item.get("role") if labels_public else None,
            subtype=item.get("subtype") if labels_public else None,
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
        gt_evidence=tuple(item for item in evidence if item) if labels_public else (),
        labels_public=labels_public,
    )


def load_context_tasks(
    tasks_file: str | Path = DEFAULT_TASKS_FILE,
    *,
    include_unlabeled: bool = False,
) -> list[ContextTask]:
    """Load Dr-CiK tasks, optionally retaining hidden rows without labels.

    Evolution and local scoring keep the default ``include_unlabeled=False``.
    Frozen inference is the only caller that should enable it.
    """
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
        if _is_labeled(record) or include_unlabeled:
            tasks.append(_to_context_task(record))
    return tasks


def load_huggingface_context_tasks(*, labels_public: bool) -> list[ContextTask]:
    """Reuse the official normalized Dr-CiK loader and enforce our label boundary."""
    from drcik_agent.data import load_huggingface_tasks

    converted = []
    for task in load_huggingface_tasks(labels_public=labels_public):
        public = bool(task.labels_public and task.future_values)
        converted.append(
            ContextTask(
                numeric=Task(
                    task_id=task.benchmark_id,
                    history_values=tuple(task.history_values),
                    future_values=tuple(task.future_values or ()) if public else (),
                    prediction_length=task.prediction_length,
                    frequency=task.frequency,
                    seasonal_period=(
                        str(task.seasonal_period)
                        if task.seasonal_period is not None
                        else None
                    ),
                    entity_name=task.entity_name,
                ),
                target_name=task.target_name,
                target_description=task.target_description,
                history_timestamps=tuple(task.history_timestamps),
                future_timestamps=tuple(task.future_timestamps),
                documents=tuple(
                    Document(
                        document_id=item.document_id,
                        content=item.text,
                        role=item.role if public else None,
                        subtype=item.subtype if public else None,
                    )
                    for item in task.documents
                ),
                gt_evidence=tuple(task.gt_evidence) if public else (),
                labels_public=public,
            )
        )
    return converted


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
