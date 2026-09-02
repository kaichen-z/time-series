"""Train-only contracts for task-local Numerical evolution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from common.data import Task
from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.task_local_ensemble import (
    TaskLocalTournamentPolicy,
    parse_task_local_release,
)
from numerical_agent.evolution.task_local_evolution import (
    TaskLocalTaskRow,
    build_group_fold_manifest,
    evaluate_task_local_release,
    fit_oof_release,
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


def _diagnostic(
    name: str,
    *,
    family: str,
    forecast: tuple[float, ...],
) -> CandidateDiagnostics:
    truths = ((10.0, 10.0),) * 3
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=1.0,
        fold_forecasts=(forecast,) * 3,
        fold_truths=truths,
        median_smae=1.0,
        median_srmse=1.0,
    )


def _local_rows(tasks: tuple[Task, ...]) -> tuple[TaskLocalTaskRow, ...]:
    rows: list[TaskLocalTaskRow] = []
    for task in tasks:
        profile = replace(
            _profile(task.task_id),
            history_length=len(task.history_values),
            horizon=2,
            periodicity_periods=((2,) if int(task.task_id.rsplit("_", 1)[-1]) % 2 else ()),
            periodicity_strength=(0.8 if int(task.task_id.rsplit("_", 1)[-1]) % 2 else 0.1),
            periodicity_confidence=(0.9 if int(task.task_id.rsplit("_", 1)[-1]) % 2 else 0.1),
        )
        for name, family, forecast in (
            ("toto_2_0", "tsfm", (8.0, 8.0)),
            ("seasonal_naive", "statistical", (10.0, 10.0)),
        ):
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=task.history_values,
                    truth=(10.0, 10.0),
                    forecast=forecast,
                    diagnostic=_diagnostic(name, family=family, forecast=forecast),
                    split="train",
                )
            )
    return tuple(rows)


def _ten_tasks() -> tuple[Task, ...]:
    return tuple(
        _task(
            f"task_{index}",
            entity=f"Entity {index}",
            history=(float(index + 1), float(index + 2), float(index + 3)),
        )
        for index in range(10)
    )


def test_oof_supply_for_each_task_is_fitted_without_its_group() -> None:
    tasks = _ten_tasks()
    manifest = build_group_fold_manifest(tasks, seed=17)

    release, report = fit_oof_release(
        _local_rows(tasks),
        manifest,
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=TaskLocalTournamentPolicy(),
    )

    assert report.task_count == 10
    assert report.oof_task_count == 10
    assert report.fit_leakage_count == 0
    assert report.group_count == len(manifest.groups)
    assert report.activation_count == 10
    assert report.accepted is True
    assert release.default_candidate_names[0] == "toto_2_0"
    assert all(item.candidate_names[0] == "toto_2_0" for item in release.group_supplies)
    assert all(len(item.candidate_names) <= 8 for item in release.group_supplies)


def test_conditional_uplift_excludes_exact_anchor_fallback_ties() -> None:
    tasks = _ten_tasks()
    manifest = build_group_fold_manifest(tasks, seed=17)
    rows = list(_local_rows(tasks))
    for index, row in enumerate(rows):
        if row.task_id == "task_0" and row.candidate_name == "seasonal_naive":
            rows[index] = replace(row, forecast=None, failure_reason="invalid")
    release, _oof = fit_oof_release(
        _local_rows(tasks),
        manifest,
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=TaskLocalTournamentPolicy(),
    )

    report = evaluate_task_local_release(
        release,
        tuple(rows),
        task_ids=tuple(task.task_id for task in tasks),
        split="dev",
    )

    assert report.activation_count == 9
    assert report.fallback_count == 1
    assert report.activated_wins == 9
    assert report.activated_ties == 0
    assert report.activated_losses == 0


def test_proposer_feedback_contains_no_task_identity_truth_or_forecast() -> None:
    tasks = _ten_tasks()
    release, report = fit_oof_release(
        _local_rows(tasks),
        build_group_fold_manifest(tasks, seed=17),
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=TaskLocalTournamentPolicy(),
    )

    payload = report.to_proposer_payload()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert release.anchor_name == "toto_2_0"
    assert not {"task_id", "truth", "future", "forecast", "public", "dev"} & keys(payload)


def test_task_local_release_round_trip_rejects_schema_drift() -> None:
    tasks = _ten_tasks()
    release, _report = fit_oof_release(
        _local_rows(tasks),
        build_group_fold_manifest(tasks, seed=17),
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=TaskLocalTournamentPolicy(),
    )
    payload = release.to_payload()

    assert parse_task_local_release(payload) == release
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        parse_task_local_release(payload)
