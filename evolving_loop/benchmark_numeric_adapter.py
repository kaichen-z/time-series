"""Dataset-blind numeric-window adapter for the package evolution engine."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from common.data import Task
from common.payload import canonical_json_bytes
from evolving_loop.data import ContextTask, Document


class BenchmarkNumericAdapterError(ValueError):
    """A benchmark window crossed the numeric/package boundary incorrectly."""


@dataclass(frozen=True)
class BenchmarkNumericWindow:
    """One label-bearing evaluation window plus frozen numeric anchors.

    ``group_id`` is retained by the evaluator and is deliberately not copied into
    the ContextTask fields visible to the Numerical/Retrieval/Decision engine.
    """

    task_id: str
    group_id: str
    history: tuple[float, ...]
    truth: tuple[float, ...]
    frequency: str
    anchor_forecasts: Mapping[str, tuple[float, ...]]
    history_timestamps: tuple[str, ...] = ()
    future_timestamps: tuple[str, ...] = ()
    documents: tuple[Document, ...] = ()
    target_description: str = "numeric forecasting target"

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise BenchmarkNumericAdapterError("window task_id must be nonempty")
        if type(self.group_id) is not str or not self.group_id:
            raise BenchmarkNumericAdapterError("window group_id must be nonempty")
        if type(self.frequency) is not str or not self.frequency:
            raise BenchmarkNumericAdapterError("window frequency must be nonempty")
        if type(self.target_description) is not str:
            raise BenchmarkNumericAdapterError("window target description must be text")
        history = tuple(float(value) for value in self.history)
        truth = tuple(float(value) for value in self.truth)
        if len(history) < 2 or not truth:
            raise BenchmarkNumericAdapterError("window history and truth are incomplete")
        if not all(math.isfinite(value) for value in history + truth):
            raise BenchmarkNumericAdapterError("window values must be finite")
        forecasts = dict(self.anchor_forecasts)
        if not forecasts or any(type(name) is not str or not name for name in forecasts):
            raise BenchmarkNumericAdapterError("window anchors require named forecasts")
        normalized = {}
        for name, values in sorted(forecasts.items()):
            forecast = tuple(float(value) for value in values)
            if len(forecast) != len(truth) or not all(
                math.isfinite(value) for value in forecast
            ):
                raise BenchmarkNumericAdapterError(
                    "window anchor horizon or values are invalid"
                )
            normalized[name] = forecast
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "truth", truth)
        object.__setattr__(
            self, "anchor_forecasts", MappingProxyType(normalized)
        )
        history_timestamps = tuple(self.history_timestamps)
        future_timestamps = tuple(self.future_timestamps)
        if history_timestamps and (
            len(history_timestamps) != len(history)
            or any(type(value) is not str or not value for value in history_timestamps)
        ):
            raise BenchmarkNumericAdapterError("window history timestamps are invalid")
        if future_timestamps and (
            len(future_timestamps) != len(truth)
            or any(type(value) is not str or not value for value in future_timestamps)
        ):
            raise BenchmarkNumericAdapterError("window future timestamps are invalid")
        documents = tuple(self.documents)
        if any(type(document) is not Document for document in documents):
            raise BenchmarkNumericAdapterError("window documents require exact Document records")
        if len({document.document_id for document in documents}) != len(documents):
            raise BenchmarkNumericAdapterError("window document identities must be unique")
        object.__setattr__(self, "history_timestamps", history_timestamps)
        object.__setattr__(self, "future_timestamps", future_timestamps)
        object.__setattr__(self, "documents", documents)


@dataclass(frozen=True)
class FrozenBenchmarkNumericBatch:
    """Immutable adapter output with labels kept outside engine inference views."""

    labeled_tasks: tuple[ContextTask, ...]
    inference_tasks: tuple[ContextTask, ...]
    group_by_task: Mapping[str, str]
    anchor_forecasts: Mapping[tuple[str, str], tuple[float, ...]]
    fingerprint: str


def build_benchmark_numeric_batch(
    windows: Sequence[BenchmarkNumericWindow],
) -> FrozenBenchmarkNumericBatch:
    """Convert generic pre-forecast windows without exposing dataset identity."""
    supplied = tuple(windows)
    if not supplied or any(type(item) is not BenchmarkNumericWindow for item in supplied):
        raise BenchmarkNumericAdapterError(
            "benchmark batch requires exact BenchmarkNumericWindow records"
        )
    task_ids = tuple(item.task_id for item in supplied)
    if len(task_ids) != len(set(task_ids)):
        raise BenchmarkNumericAdapterError("benchmark task identities must be unique")

    labeled = []
    inference = []
    groups = {}
    forecasts = {}
    payload_rows = []
    for item in supplied:
        horizon = len(item.truth)
        numeric = Task(
            task_id=item.task_id,
            history_values=item.history,
            future_values=item.truth,
            prediction_length=horizon,
            frequency=item.frequency,
            seasonal_period=None,
            entity_name="benchmark_series",
        )
        history_timestamps = item.history_timestamps or tuple(
            f"history_{index}" for index in range(len(item.history))
        )
        future_timestamps = item.future_timestamps or tuple(
            f"future_{index}" for index in range(horizon)
        )
        task = ContextTask(
            numeric=numeric,
            target_name="target",
            target_description=item.target_description,
            history_timestamps=history_timestamps,
            future_timestamps=future_timestamps,
            documents=item.documents,
            gt_evidence=(),
            labels_public=True,
        )
        labeled.append(task)
        inference.append(
            ContextTask(
                numeric=task.numeric_view(),
                target_name=task.target_name,
                target_description=task.target_description,
                history_timestamps=task.history_timestamps,
                future_timestamps=task.future_timestamps,
                documents=task.documents,
                gt_evidence=(),
                labels_public=False,
            )
        )
        groups[item.task_id] = item.group_id
        for name, forecast in item.anchor_forecasts.items():
            forecasts[(item.task_id, name)] = forecast
        payload_rows.append(
            {
                "task_id": item.task_id,
                "group_id": item.group_id,
                "history": list(item.history),
                "truth": list(item.truth),
                "frequency": item.frequency,
                "anchor_forecasts": {
                    name: list(values)
                    for name, values in item.anchor_forecasts.items()
                },
            }
        )
    fingerprint = hashlib.sha256(canonical_json_bytes(payload_rows)).hexdigest()
    return FrozenBenchmarkNumericBatch(
        labeled_tasks=tuple(labeled),
        inference_tasks=tuple(inference),
        group_by_task=MappingProxyType(groups),
        anchor_forecasts=MappingProxyType(forecasts),
        fingerprint=fingerprint,
    )


__all__ = [
    "BenchmarkNumericAdapterError",
    "BenchmarkNumericWindow",
    "FrozenBenchmarkNumericBatch",
    "build_benchmark_numeric_batch",
]
