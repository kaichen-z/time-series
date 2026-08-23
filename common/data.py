"""Numbers-only Dr-CiK task loading, shared by every package that forecasts."""
from __future__ import annotations

import json
from dataclasses import dataclass
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


def _series(record: dict) -> dict:
    return record.get("series", record)


def _metadata(record: dict) -> dict:
    return record.get("task_metadata", record)


def _showcase(record: dict) -> dict:
    return record.get("showcase", record)


def _is_labeled(record: dict) -> bool:
    """A record is usable only if it has real, non-null ground-truth future values."""
    future = _series(record).get("future_values")
    return (
        record.get("labels_public", True) is not False
        and bool(future)
        and future[0] is not None
    )


def _future_values(record: dict) -> tuple[float, ...]:
    """Return only genuine numeric labels; hidden rows become an empty tuple."""
    if record.get("labels_public", True) is False:
        return ()
    raw = _series(record).get("future_values") or ()
    if not raw or raw[0] is None:
        return ()
    return tuple(float(value) for value in raw)


def _to_task(record: dict) -> Task:
    """Extract exactly the numeric fields a Task needs, ignoring everything else in the record."""
    series = _series(record)
    metadata = _metadata(record)
    showcase = _showcase(record)
    entity = showcase.get("entity", {})
    return Task(
        task_id=record["benchmark_id"],
        history_values=tuple(float(value) for value in series["history_values"]),
        future_values=_future_values(record),
        prediction_length=int(metadata["prediction_length"]),
        frequency=str(metadata["frequency"]),
        seasonal_period=metadata.get("seasonal_period"),
        entity_name=str(record.get("entity_name") or entity.get("name") or "unknown"),
    )


def load_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE) -> list[Task]:
    """Load labeled tasks from a Dr-CiK JSONL file or public task directory."""
    tasks: list[Task] = []
    source = Path(tasks_file)
    if source.is_dir():
        records = (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(source.glob("task_*.json"))
        )
    else:
        records = (
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    for record in records:
        if _is_labeled(record):
            tasks.append(_to_task(record))
    return tasks
