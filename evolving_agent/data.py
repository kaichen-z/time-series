"""Numeric-only task loading for the coding-skill baseline (no documents, ever)."""
from __future__ import annotations

import json
import random
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


def _is_labeled(record: dict) -> bool:
    """A record is usable only if it has real, non-null ground-truth future values."""
    future = record.get("future_values")
    return bool(future) and future[0] is not None


def _to_task(record: dict) -> Task:
    """Extract exactly the numeric fields a Task needs, ignoring everything else in the record."""
    return Task(
        task_id=record["benchmark_id"],
        history_values=tuple(record["history_values"]),
        future_values=tuple(record["future_values"]),
        prediction_length=record["prediction_length"],
        frequency=record["frequency"],
        seasonal_period=record.get("seasonal_period"),
        entity_name=record["entity_name"],
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
