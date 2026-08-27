"""Dr-CiK task loading for the baselines, hidden-label tasks included."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

DEFAULT_TASKS_FILE = Path(
    "/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl"
)

DEV = "dev"
HIDDEN = "hidden"


@dataclass(frozen=True)
class BenchmarkTask:
    """One Dr-CiK task under the No Context condition: the series and its horizon, no text.

    Deliberately carries no target description, profile, or document: a field that does not
    exist cannot leak into a prompt.
    """

    benchmark_id: str
    history_values: tuple[float, ...]
    history_timestamps: tuple[str, ...]
    future_values: tuple[float, ...]
    future_timestamps: tuple[str, ...]
    prediction_length: int
    frequency: str
    seasonal_period: str | None
    labels_public: bool

    @property
    def split(self) -> str:
        return DEV if self.labels_public else HIDDEN


def _series(record: dict) -> dict:
    return record.get("series", record)


def _metadata(record: dict) -> dict:
    return record.get("task_metadata", record)


def _floats(raw: Sequence | None) -> tuple[float, ...]:
    """Read a value list, treating a withheld (null-filled) one as absent."""
    if not raw or raw[0] is None:
        return ()
    return tuple(float(value) for value in raw)


def _to_task(record: dict) -> BenchmarkTask:
    series = _series(record)
    metadata = _metadata(record)
    labels_public = record.get("labels_public", True) is not False
    return BenchmarkTask(
        benchmark_id=record["benchmark_id"],
        history_values=_floats(series["history_values"]),
        history_timestamps=tuple(str(t) for t in series.get("history_timestamps") or ()),
        future_values=_floats(series.get("future_values")) if labels_public else (),
        future_timestamps=tuple(str(t) for t in series.get("future_timestamps") or ()),
        prediction_length=int(metadata["prediction_length"]),
        frequency=str(metadata["frequency"]),
        seasonal_period=metadata.get("seasonal_period"),
        labels_public=labels_public,
    )


def load_benchmark(
    tasks_file: str | Path = DEFAULT_TASKS_FILE, split: str | None = None
) -> list[BenchmarkTask]:
    """Load every task, hidden ones included; pass split to keep only 'dev' or 'hidden'."""
    if split not in (None, DEV, HIDDEN):
        raise ValueError(f"split must be {DEV!r}, {HIDDEN!r}, or None; got {split!r}")
    tasks = [_to_task(record) for record in _records(tasks_file)]
    _check_horizons(tasks)
    return [task for task in tasks if split is None or task.split == split]


def _records(tasks_file: str | Path) -> Iterator[dict]:
    with open(tasks_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _check_horizons(tasks: Sequence[BenchmarkTask]) -> None:
    """Fail loudly on a task whose labels disagree with its stated horizon.

    Scoring compares a forecast of prediction_length against future_values, so a mismatch here
    would surface much later as an unexplained length error inside a metric.
    """
    for task in tasks:
        if task.labels_public and len(task.future_values) != task.prediction_length:
            raise ValueError(
                f"{task.benchmark_id}: {len(task.future_values)} future values, "
                f"expected {task.prediction_length}"
            )
