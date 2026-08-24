"""Import an evolved methods module once and score every method against every task."""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from common.metrics import (
    change_smae,
    scaled_mae,
    scaled_rmse,
    shape_correlation,
    variance_ratio,
)


NOT_APPLICABLE = "not_applicable"
CRASHED = "crashed"
INVALID = "invalid"
SUCCESS = "success"


@dataclass(frozen=True)
class Task:
    """One label-free forecasting input plus the trusted future used only for scoring."""

    task_id: str
    history: tuple[float, ...]
    horizon: int
    frequency: str
    future: tuple[float, ...]

    def characteristics(self) -> tuple[str, ...]:
        """Describe the series from history only, never from the future values."""
        return derive_characteristics(self.history, self.horizon, self.frequency)


@dataclass(frozen=True)
class Outcome:
    """What one method did on one task."""

    method: str
    task_id: str
    status: str
    smae: float | None = None
    srmse: float | None = None
    variance_ratio: float | None = None
    shape_correlation: float | None = None
    change_smae: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class MethodReport:
    """Aggregate behavior of one method, with skipped tasks kept apart from failures."""

    method: str
    total: int
    success: int
    not_applicable: int
    crashed: int
    invalid: int
    mean_smae: float | None
    mean_srmse: float | None
    mean_variance_ratio: float | None
    mean_shape_correlation: float | None
    mean_change_smae: float | None
    mean_rank: float | None
    coverage: float
    by_characteristic_smae: Mapping[str, float] = field(default_factory=dict)
    sample_failures: tuple[str, ...] = ()


def derive_characteristics(
    history: Sequence[float], horizon: int, frequency: str
) -> tuple[str, ...]:
    """Bucket a series by traits the model can use to write 'when to use' docstrings."""
    values = [float(value) for value in history]
    length = len(values)
    tags = [f"frequency:{frequency}", f"history:{_bucket(length, (48, 168, 512))}"]
    tags.append(f"horizon:{_bucket(horizon, (8, 24, 96))}")
    zeros = sum(1 for value in values if value == 0.0)
    tags.append("intermittent" if zeros / length > 0.3 else "dense")
    if length >= 8:
        first, second = values[: length // 2], values[length // 2 :]
        spread = statistics.pstdev(values) or 1.0
        drift = abs(statistics.fmean(second) - statistics.fmean(first)) / spread
        tags.append("trending" if drift > 0.5 else "flat")
    return tuple(tags)


def _bucket(value: int, edges: Sequence[int]) -> str:
    for edge in edges:
        if value <= edge:
            return f"le_{edge}"
    return f"gt_{edges[-1]}"


def load_methods(path: str | Path) -> tuple[object, dict[str, object]]:
    """Import the module once and return it with its forecasting functions by name."""
    source = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"evolved_methods_{source.stem}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import a methods module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    functions = {
        name: value
        for name, value in vars(module).items()
        if callable(value)
        and not name.startswith("_")
        and not isinstance(value, type)
        # Only functions defined in this file are methods. Without this, any module-level
        # import would be measured as a forecasting method: called with (history, horizon,
        # frequency), it raises TypeError on every task and the resulting fake crashes reach
        # the evolution prompt as evidence against a method that does not exist.
        and getattr(value, "__module__", None) == module.__name__
    }
    if not functions:
        raise ImportError(f"{source} defines no forecasting functions")
    return module, functions


def run_module(
    path: str | Path, tasks: Sequence[Task], *, time_budget_s: float = 300.0
) -> tuple[tuple[Outcome, ...], tuple[MethodReport, ...]]:
    """Run every method over every task in this process and summarize each method."""
    module, functions = load_methods(path)
    not_applicable = getattr(module, "NotApplicable", None)
    if not isinstance(not_applicable, type) or not issubclass(not_applicable, Exception):
        raise ImportError("methods module must define a NotApplicable exception")

    outcomes: list[Outcome] = []
    for name, function in functions.items():
        for task in tasks:
            outcomes.append(
                _run_one(name, function, task, not_applicable, time_budget_s)
            )
    ranks = _mean_ranks(outcomes)
    reports = tuple(
        _report(name, [o for o in outcomes if o.method == name], tasks, ranks.get(name))
        for name in functions
    )
    return tuple(outcomes), reports


def _mean_ranks(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """Average rank by scaled MAE among the methods that forecast the same task.

    Ranking within a task before averaging stops one large-magnitude series from deciding the
    whole comparison, which a mean over the errors themselves cannot avoid.
    """
    by_task: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        if outcome.status == SUCCESS:
            by_task.setdefault(outcome.task_id, []).append(outcome)

    collected: dict[str, list[float]] = {}
    for scored in by_task.values():
        # Ties share the average of the positions they span, so equivalent methods rank equally.
        ordered = sorted(scored, key=lambda o: float(o.smae))
        position = 0
        while position < len(ordered):
            end = position + 1
            while end < len(ordered) and (
                float(ordered[end].smae) == float(ordered[position].smae)
            ):
                end += 1
            shared = ((position + 1) + end) / 2.0
            for outcome in ordered[position:end]:
                collected.setdefault(outcome.method, []).append(shared)
            position = end
    return {name: statistics.fmean(values) for name, values in collected.items()}


def _run_one(
    name: str,
    function: object,
    task: Task,
    not_applicable: type,
    time_budget_s: float,
) -> Outcome:
    start = time.monotonic()
    try:
        raw = function(list(task.history), task.horizon, task.frequency)  # type: ignore[operator]
    except not_applicable as exc:  # type: ignore[misc]
        return Outcome(name, task.task_id, NOT_APPLICABLE, detail=str(exc)[:200])
    except BaseException as exc:
        # Anything other than NotApplicable is a defect, not modesty about applicability.
        return Outcome(name, task.task_id, CRASHED, detail=f"{type(exc).__name__}: {exc}"[:200])
    if time.monotonic() - start > time_budget_s:
        return Outcome(name, task.task_id, INVALID, detail=f"exceeded {time_budget_s}s")
    try:
        forecast = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        return Outcome(name, task.task_id, INVALID, detail=f"unreadable forecast: {exc}"[:200])
    if len(forecast) != task.horizon:
        return Outcome(
            name, task.task_id, INVALID,
            detail=f"returned {len(forecast)} values, expected {task.horizon}",
        )
    if not all(math.isfinite(value) for value in forecast):
        return Outcome(name, task.task_id, INVALID, detail="returned a non-finite value")
    truth = list(task.future)
    return Outcome(
        name, task.task_id, SUCCESS,
        smae=scaled_mae(truth, forecast),
        srmse=scaled_rmse(truth, forecast),
        variance_ratio=variance_ratio(truth, forecast),
        shape_correlation=shape_correlation(truth, forecast),
        change_smae=change_smae(truth, forecast, float(task.history[-1])),
    )


def _report(
    name: str,
    outcomes: Sequence[Outcome],
    tasks: Sequence[Task],
    mean_rank: float | None = None,
) -> MethodReport:
    by_id = {task.task_id: task for task in tasks}
    scored = [o for o in outcomes if o.status == SUCCESS]
    counts = Counter(o.status for o in outcomes)

    grouped_smae: dict[str, list[float]] = {}
    for outcome in scored:
        for tag in by_id[outcome.task_id].characteristics():
            grouped_smae.setdefault(tag, []).append(float(outcome.smae))

    # Deduplicated with a dict, not a set: set order is not reproducible across runs.
    ordered_failures = tuple(
        dict.fromkeys(
            o.detail for o in outcomes if o.status in (CRASHED, INVALID) and o.detail
        )
    )[:3]

    total = len(outcomes)
    return MethodReport(
        method=name,
        total=total,
        success=len(scored),
        not_applicable=counts.get(NOT_APPLICABLE, 0),
        crashed=counts.get(CRASHED, 0),
        invalid=counts.get(INVALID, 0),
        mean_smae=statistics.fmean(o.smae for o in scored) if scored else None,
        mean_srmse=statistics.fmean(o.srmse for o in scored) if scored else None,
        mean_variance_ratio=statistics.fmean(o.variance_ratio for o in scored) if scored else None,
        mean_shape_correlation=(
            statistics.fmean(o.shape_correlation for o in scored) if scored else None
        ),
        mean_change_smae=statistics.fmean(o.change_smae for o in scored) if scored else None,
        mean_rank=mean_rank,
        coverage=len(scored) / total if total else 0.0,
        by_characteristic_smae={
            tag: statistics.fmean(values) for tag, values in sorted(grouped_smae.items())
        },
        sample_failures=ordered_failures,
    )


def report_payload(reports: Sequence[MethodReport]) -> list[dict[str, object]]:
    """Render reports as the JSON the evolution prompt and the metrics file both use."""
    return [
        {
            "method": report.method,
            "mean_smae": report.mean_smae,
            "mean_srmse": report.mean_srmse,
            "mean_rank": report.mean_rank,
            "mean_variance_ratio": report.mean_variance_ratio,
            "mean_shape_correlation": report.mean_shape_correlation,
            "mean_change_smae": report.mean_change_smae,
            "success": report.success,
            "total": report.total,
            "coverage": round(report.coverage, 4),
            "not_applicable": report.not_applicable,
            "crashed": report.crashed,
            "invalid": report.invalid,
            "by_characteristic_smae": {
                tag: round(value, 4) for tag, value in report.by_characteristic_smae.items()
            },
            "sample_failures": list(report.sample_failures),
        }
        for report in reports
    ]
