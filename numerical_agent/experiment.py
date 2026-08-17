"""Build a curation experiment config from Dr-CiK tasks and a frozen entity-disjoint split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from common.data import Task, load_tasks


def build_experiment(
    *,
    tasks_file: str | Path,
    split_file: str | Path,
    generations: int = 1,
    children_per_generation: int = 1,
    seed: int = 20260816,
    max_revisions_per_method: int = 1,
    accepted_max_error: float = 50.0,
    specialized_max_error: float = 100.0,
    train_limit: int | None = None,
    dev_limit: int | None = None,
) -> dict[str, object]:
    """Return an experiment config holding the split's Train/Dev tasks and their labels."""
    tasks = {task.task_id: task for task in load_tasks(tasks_file)}
    partitions = _partitions(Path(split_file))
    train = _select(tasks, partitions["train"], train_limit, "train")
    dev = _select(tasks, partitions["dev"], dev_limit, "dev")
    return {
        "evolution": {
            "generations": generations,
            "children_per_generation": children_per_generation,
            "seed": seed,
            "resume": True,
        },
        "curation": {
            "max_revisions_per_method": max_revisions_per_method,
            "dictionary_metric": "smape",
            "method_metric": "smape",
            "accepted_max_error": accepted_max_error,
            "specialized_max_error": specialized_max_error,
        },
        "tasks": {
            "train": [_item(task) for task in train],
            "dev": [_item(task) for task in dev],
        },
        "labels": {
            "train": {task.task_id: list(task.future_values) for task in train},
            "dev": {task.task_id: list(task.future_values) for task in dev},
        },
    }


def _partitions(split_file: Path) -> dict[str, tuple[str, ...]]:
    """Read the frozen split's Train and Dev task ids; Public Test is never used here."""
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"{split_file} has no partitions object")
    selected = {}
    for name in ("train", "dev"):
        part = partitions.get(name)
        if not isinstance(part, dict) or not isinstance(part.get("task_ids"), list):
            raise ValueError(f"{split_file} has no {name} task_ids")
        selected[name] = tuple(str(task_id) for task_id in part["task_ids"])
    return selected


def _select(
    tasks: dict[str, Task], task_ids: Iterable[str], limit: int | None, name: str
) -> tuple[Task, ...]:
    """Resolve split ids to labeled tasks, failing loudly on anything missing."""
    resolved = []
    for task_id in task_ids:
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"{name} task {task_id!r} is absent from the tasks file")
        if not task.future_values:
            raise ValueError(f"{name} task {task_id!r} has no labels to score against")
        resolved.append(task)
    if limit is not None:
        resolved = resolved[:limit]
    if not resolved:
        raise ValueError(f"{name} split selected no tasks")
    return tuple(resolved)


def _item(task: Task) -> dict[str, object]:
    characteristics = [f"frequency:{task.frequency}"]
    if task.seasonal_period:
        characteristics.append(f"seasonal_period:{task.seasonal_period}")
    return {
        "item_id": task.task_id,
        "history": list(task.history_values),
        "horizon": task.prediction_length,
        "frequency": task.frequency,
        "characteristics": characteristics,
    }
