"""Train-only contracts for task-local Numerical evolution."""

from __future__ import annotations

from dataclasses import replace

from common.data import Task
from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.task_local_evolution import (
    build_group_fold_manifest,
    task_morphology_key,
)


def _task(
    task_id: str,
    *,
    entity: str,
    history: tuple[float, ...],
    future: tuple[float, ...] = (10.0, 11.0),
) -> Task:
    return Task(
        task_id=task_id,
        history_values=history,
        future_values=future,
        prediction_length=len(future),
        frequency="D",
        seasonal_period="7",
        entity_name=entity,
    )


def _tasks() -> tuple[Task, ...]:
    return (
        _task("a", entity="Store A", history=(1.0, 2.0, 3.0)),
        _task("b", entity="store a", history=(4.0, 5.0, 6.0)),
        _task("c", entity="Store C", history=(4.0, 5.0, 6.0)),
        _task("d", entity="Store D", history=(7.0, 8.0, 9.0)),
        _task("e", entity="Store E", history=(10.0, 11.0, 12.0)),
        _task("f", entity="Store F", history=(13.0, 14.0, 15.0)),
        _task("g", entity="Store G", history=(16.0, 17.0, 18.0)),
        _task("h", entity="Store H", history=(19.0, 20.0, 21.0)),
        _task("i", entity="Store I", history=(22.0, 23.0, 24.0)),
    )


def _profile(task_id: str = "task") -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=120,
        horizon=12,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="flat",
        trend_strength=0.1,
        periodicity_periods=(),
        periodicity_strength=0.1,
        periodicity_confidence=0.1,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=True,
        stationarity_score=0.8,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def test_group_fold_manifest_keeps_transitive_entity_history_groups_together() -> None:
    manifest = build_group_fold_manifest(_tasks(), seed=17)

    folds = manifest.task_fold_map
    assert folds["a"] == folds["b"] == folds["c"]
    assert set(folds) == {task.task_id for task in _tasks()}
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_group_fold_manifest_is_order_independent_and_does_not_use_future() -> None:
    tasks = _tasks()
    first = build_group_fold_manifest(tasks, seed=17)
    changed = tuple(
        replace(task, future_values=(999.0, -999.0)) for task in reversed(tasks)
    )

    second = build_group_fold_manifest(changed, seed=17)

    assert first.to_payload() == second.to_payload()


def test_source_series_identity_joins_otherwise_distinct_tasks() -> None:
    tasks = _tasks()
    manifest = build_group_fold_manifest(
        tasks,
        seed=17,
        source_series_ids={"d": "source-1", "e": "source-1"},
    )

    assert manifest.task_fold_map["d"] == manifest.task_fold_map["e"]


def test_morphology_key_excludes_task_identity_and_separates_regimes() -> None:
    base = _profile("first")
    same_shape = replace(base, task_id="second")
    periodic = replace(
        base,
        periodicity_periods=(7,),
        periodicity_strength=0.8,
        periodicity_confidence=0.9,
    )
    intermittent = replace(base, zero_fraction=0.8, intermittency_adi=4.0)
    regime_shift = replace(
        base,
        recent_regime_start=90,
        recent_regime_confidence=0.9,
    )

    assert task_morphology_key(base) == task_morphology_key(same_shape)
    assert len(
        {
            task_morphology_key(base),
            task_morphology_key(periodic),
            task_morphology_key(intermittent),
            task_morphology_key(regime_shift),
        }
    ) == 4
