"""Scores one individual over a task minibatch, keeping the worst cases for mutation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from dr_cik.models import ForecastTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskResult:
    """One individual's outcome on one task; score is always higher-is-better."""

    task_id: str
    score: float
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """An individual's mean score over a minibatch, plus its worst tasks for the evolver to read."""

    individual_id: str
    mean_score: float
    task_results: tuple[TaskResult, ...]
    worst: tuple[TaskResult, ...] = ()


class ScoreFn(Protocol):
    """Loop-specific scoring: run one individual on one task and score it."""

    def __call__(self, individual: Any, task: ForecastTask) -> TaskResult: ...


def individual_id(individual: Any) -> str:
    """Return a stable display id for a Bundle or a BundleTriple."""
    if hasattr(individual, "bundle_id"):
        return individual.bundle_id
    return "+".join(part.bundle_id for part in (individual.coding, individual.retrieval, individual.decision))


def evaluate(
    individual: Any,
    tasks: list[ForecastTask],
    score_fn: ScoreFn,
    worst_n: int = 5,
    on_task: Callable[[TaskResult], None] | None = None,
) -> EvalResult:
    """Score an individual on every task, returning the mean and the worst-scoring tasks."""
    results: list[TaskResult] = []
    for task in tasks:
        result = score_fn(individual, task)
        results.append(result)
        if on_task is not None:
            on_task(result)

    mean_score = sum(item.score for item in results) / len(results) if results else float("-inf")
    worst = tuple(sorted(results, key=lambda item: item.score)[:worst_n])
    logger.info("evaluated %s on %d task(s): mean_score=%.4f", individual_id(individual), len(results), mean_score)
    return EvalResult(
        individual_id=individual_id(individual), mean_score=mean_score, task_results=tuple(results), worst=worst
    )


def evaluate_population(
    population: list[Any],
    tasks: list[ForecastTask],
    score_fn: ScoreFn,
    worst_n: int = 5,
    on_task: Callable[[TaskResult], None] | None = None,
) -> list[EvalResult]:
    """Score every individual in a population over the same minibatch."""
    return [evaluate(individual, tasks, score_fn, worst_n=worst_n, on_task=on_task) for individual in population]
