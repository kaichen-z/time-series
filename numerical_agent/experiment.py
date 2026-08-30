"""Build a curation experiment config from Dr-CiK tasks and a frozen entity-disjoint split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from common.data import Task, load_tasks
from common.evolution_core.contracts import metric_policy_metadata


def build_experiment(
    *,
    tasks_file: str | Path,
    split_file: str | Path,
    generations: int = 1,
    children_per_generation: int = 1,
    seed: int = 20260816,
    max_revisions_per_method: int = 1,
    max_implementation_attempts: int = 3,
    accepted_max_smae: float = 1.0,
    accepted_max_srmse: float = 1.0,
    specialized_max_smae: float = 2.5,
    specialized_max_srmse: float = 2.5,
    min_success_rate: float = 0.8,
    selection_folds: int = 3,
    selection_horizon: int = 8,
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
            "schema_version": 2,
            **metric_policy_metadata(),
            "generations": generations,
            "children_per_generation": children_per_generation,
            "seed": seed,
            "resume": True,
        },
        "curation": {
            "schema_version": 2,
            **metric_policy_metadata(),
            "allowed_families": ["statistical"],
            "max_revisions_per_method": max_revisions_per_method,
            "max_implementation_attempts": max_implementation_attempts,
            "dictionary_metric": "smae",
            "method_metric": "smae",
            "accepted_max_smae": accepted_max_smae,
            "accepted_max_srmse": accepted_max_srmse,
            "specialized_max_smae": specialized_max_smae,
            "specialized_max_srmse": specialized_max_srmse,
            "min_success_rate": min_success_rate,
            "selection_folds": selection_folds,
            "selection_horizon": selection_horizon,
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


def build_frozen_test(
    *,
    tasks_file: str | Path,
    split_file: str | Path,
) -> dict[str, object]:
    """Return the sealed Public Test inputs and labels without Train/Dev state."""
    tasks = {task.task_id: task for task in load_tasks(tasks_file)}
    split_path = Path(split_file)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    task_ids = _partition_ids(payload, split_path, "public_test")
    public_test = _select(tasks, task_ids, None, "public_test")
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError(f"{split_path} has no manifest_sha256")
    return {
        "manifest_sha256": manifest_sha256,
        "tasks": {"public_test": [_item(task) for task in public_test]},
        "labels": {
            "public_test": {
                task.task_id: list(task.future_values) for task in public_test
            }
        },
    }


def _partitions(split_file: Path) -> dict[str, tuple[str, ...]]:
    """Read the frozen split's Train and Dev task ids; Public Test is never used here."""
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    return {
        name: _partition_ids(payload, split_file, name) for name in ("train", "dev")
    }


def _partition_ids(
    payload: object, split_file: Path, name: str
) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ValueError(f"{split_file} must contain a JSON object")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"{split_file} has no partitions object")
    part = partitions.get(name)
    if not isinstance(part, dict) or not isinstance(part.get("task_ids"), list):
        raise ValueError(f"{split_file} has no {name} task_ids")
    return tuple(str(task_id) for task_id in part["task_ids"])


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
