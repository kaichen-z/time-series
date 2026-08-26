"""Numbers-only Dr-CiK task loading, shared by every package that forecasts."""
from __future__ import annotations

import json
import re
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


_BENCHMARK_ID = re.compile(r'"benchmark_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def load_tasks_by_id(
    tasks_file: str | Path,
    task_ids: tuple[str, ...] | list[str],
) -> list[Task]:
    """Load only authorized task records, without parsing other directory entries."""
    requested = tuple(dict.fromkeys(str(task_id) for task_id in task_ids))
    if not requested:
        return []
    wanted = set(requested)
    source = Path(tasks_file)
    records: dict[str, dict] = {}
    if source.is_dir():
        for task_id in requested:
            if Path(task_id).name != task_id:
                raise ValueError(f"invalid task id path component: {task_id}")
            path = source / f"{task_id}.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if str(record.get("benchmark_id")) == task_id:
                records[task_id] = record
    else:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                match = _BENCHMARK_ID.search(line)
                if match is None:
                    continue
                task_id = str(json.loads(match.group(1)))
                if task_id not in wanted:
                    continue
                record = json.loads(line)
                if str(record.get("benchmark_id")) == task_id:
                    records[task_id] = record
    return [
        _to_task(records[task_id])
        for task_id in requested
        if task_id in records and _is_labeled(records[task_id])
    ]
