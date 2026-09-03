"""Specialist Atlas contracts and Train-only candidate discovery."""

from __future__ import annotations

from dataclasses import replace

import pytest

from common.data import Task
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.specialist_atlas import (
    AtlasMaterializedCandidate,
    AtlasPolicy,
    AtlasTaskCase,
    atlas_feature,
    fit_atlas_oof,
    fit_atlas_release,
    parse_atlas_release,
    route_atlas_task,
    select_specialist_pool,
)
from numerical_agent.evolution.task_local_evolution import (
    TaskLocalTaskRow,
    build_group_fold_manifest,
)


def _profile(task_id: str, *, periodic: bool = False) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=8,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="up",
        trend_strength=0.6,
        periodicity_periods=(2,) if periodic else (),
        periodicity_strength=0.8 if periodic else 0.1,
        periodicity_confidence=0.9 if periodic else 0.1,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.2,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _diagnostic(
    name: str,
    family: str,
    forecast: tuple[float, float],
    *,
    smae: float,
    srmse: float,
) -> CandidateDiagnostics:
    truth = (10.0, 11.0)
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=1.0,
        fold_forecasts=(forecast,) * 3,
        fold_truths=(truth,) * 3,
        median_smae=smae,
        recent_smae=smae,
        worst_smae=smae,
        smae_mad=0.02,
        median_srmse=srmse,
        recent_srmse=srmse,
        worst_srmse=srmse,
        srmse_mad=0.03,
        worst_smae_raw=smae,
        worst_srmse_raw=srmse,
    )


def _row(
    task_id: str,
    name: str,
    forecast: tuple[float, float],
    *,
    family: str = "statistical",
    truth: tuple[float, float] = (10.0, 11.0),
    periodic: bool = False,
    smae: float = 0.4,
    srmse: float = 0.5,
) -> TaskLocalTaskRow:
    history = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    return TaskLocalTaskRow(
        task_id=task_id,
        candidate_name=name,
        family=family,
        profile=_profile(task_id, periodic=periodic),
        history=history,
        truth=truth,
        forecast=forecast,
        diagnostic=_diagnostic(name, family, forecast, smae=smae, srmse=srmse),
        split="train",
    )


def test_atlas_feature_is_task_id_and_future_target_free() -> None:
    first_anchor = _row(
        "first", "toto_2_0", (9.0, 9.0), family="tsfm", periodic=True
    )
    first = _row("first", "seasonal_naive", (10.0, 10.0), periodic=True)
    second_anchor = replace(
        first_anchor,
        task_id="second",
        profile=replace(first_anchor.profile, task_id="second"),
        truth=(999.0, -999.0),
    )
    second = replace(
        first,
        task_id="second",
        profile=replace(first.profile, task_id="second"),
        truth=(999.0, -999.0),
    )

    left = atlas_feature(first, first_anchor)
    right = atlas_feature(second, second_anchor)

    assert left == right
    assert "first" not in repr(left)
    assert "second" not in repr(left)
    assert "999" not in repr(left)


def test_atlas_feature_uses_paired_scaled_hindcasts_and_disagreement() -> None:
    anchor = _row(
        "task",
        "toto_2_0",
        (9.0, 9.0),
        family="tsfm",
        smae=0.6,
        srmse=0.7,
    )
    specialist = _row(
        "task",
        "seasonal_naive",
        (10.0, 10.0),
        smae=0.4,
        srmse=0.5,
    )

    feature = atlas_feature(specialist, anchor)

    assert feature.categorical[:3] == ("d", "short", "medium")
    assert feature.numeric[-8:-2] == pytest.approx((0.2,) * 6)
    assert feature.numeric[-1] > 0.0


def test_specialist_pool_retains_complementary_candidates_not_global_mean_only() -> None:
    task_ids = tuple(f"task_{index}" for index in range(4))
    rows: list[TaskLocalTaskRow] = []
    for index, task_id in enumerate(task_ids):
        truth = (10.0, 10.0)
        rows.extend(
            (
                _row(task_id, "toto_2_0", (8.0, 8.0), family="tsfm", truth=truth),
                _row(
                    task_id,
                    "early_specialist",
                    (10.0, 10.0) if index < 2 else (0.0, 0.0),
                    truth=truth,
                ),
                _row(
                    task_id,
                    "late_specialist",
                    (0.0, 0.0) if index < 2 else (10.0, 10.0),
                    truth=truth,
                ),
            )
        )
    groups = {task_id: f"{index + 1:064x}" for index, task_id in enumerate(task_ids)}

    selected = select_specialist_pool(
        tuple(rows),
        task_ids=task_ids,
        group_ids=groups,
        anchor_name="toto_2_0",
        policy=AtlasPolicy(maximum_pool_size=3, maximum_task_candidates=3),
    )

    assert selected == ("toto_2_0", "early_specialist", "late_specialist")


def _atlas_tasks(count: int = 64) -> tuple[Task, ...]:
    return tuple(
        Task(
            task_id=f"atlas_{index:03d}",
            history_values=tuple(float(index + offset) for offset in range(8)),
            future_values=(10.0 + index, 11.0 + index),
            prediction_length=2,
            frequency="D",
            seasonal_period="7",
            entity_name=f"Atlas entity {index:03d}",
        )
        for index in range(count)
    )


def _atlas_rows(tasks: tuple[Task, ...]) -> tuple[TaskLocalTaskRow, ...]:
    rows: list[TaskLocalTaskRow] = []
    for task in tasks:
        anchor_forecast = tuple(value + 2.0 for value in task.future_values)
        specialist_forecast = tuple(value + 0.2 for value in task.future_values)
        for name, family, forecast, score in (
            ("toto_2_0", "tsfm", anchor_forecast, 0.7),
            ("seasonal_naive", "statistical", specialist_forecast, 0.2),
        ):
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=replace(_profile(task.task_id), history_length=8),
                    history=task.history_values,
                    truth=task.future_values,
                    forecast=forecast,
                    diagnostic=_diagnostic(
                        name,
                        family,
                        forecast,
                        smae=score,
                        srmse=score,
                    ),
                    split="train",
                )
            )
    return tuple(rows)


def test_atlas_release_contains_full_build_and_five_oof_models() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)

    release = fit_atlas_release(_atlas_rows(tasks), manifest, AtlasPolicy())

    assert tuple(fold for fold, _model in release.build_fold_models) == (0, 1, 2, 3, 4)
    assert release.full_build_model.training_task_count == 64
    assert parse_atlas_release(release.to_payload()) == release


def test_atlas_policy_caps_predicted_regret_at_one_quarter() -> None:
    with pytest.raises(ValueError, match="regret"):
        AtlasPolicy(maximum_predicted_regret=0.2500000001)


def test_atlas_release_rejects_full_build_model_reused_for_oof_folds() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    release = fit_atlas_release(_atlas_rows(tasks), manifest, AtlasPolicy())

    with pytest.raises(ValueError, match="held-out"):
        replace(
            release,
            build_fold_models=tuple(
                (fold, release.full_build_model) for fold in range(5)
            ),
        )


def test_atlas_oof_route_never_uses_held_out_group_labels() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)

    result = fit_atlas_oof(_atlas_rows(tasks), manifest, AtlasPolicy())

    assert len(result.tasks) == 64
    for task_result in result.tasks:
        assert task_result.group_sha256 not in task_result.training_group_sha256s


def test_atlas_task_185_shape_falls_back_when_predicted_regret_exceeds_quarter() -> (
    None
):
    tasks = _atlas_tasks()
    rows = _atlas_rows(tasks)
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    release = fit_atlas_release(rows, manifest, AtlasPolicy())
    catastrophic_model = replace(
        release.full_build_model,
        records=tuple(
            replace(record, regret_smae_raw=0.9, regret_srmse_raw=0.8)
            for record in release.full_build_model.records
        ),
    )
    catastrophic_release = replace(release, full_build_model=catastrophic_model)
    anchor = next(row for row in rows if row.candidate_name == "toto_2_0")
    specialist = next(
        row
        for row in rows
        if row.task_id == anchor.task_id and row.candidate_name == "seasonal_naive"
    )
    catastrophic_overlay_case = AtlasTaskCase(
        group_sha256="f" * 64,
        anchor=AtlasMaterializedCandidate(
            candidate_name=anchor.candidate_name,
            family=anchor.family,
            feature=None,
            forecast=anchor.forecast,
        ),
        candidates=(
            AtlasMaterializedCandidate(
                candidate_name=specialist.candidate_name,
                family=specialist.family,
                feature=atlas_feature(specialist, anchor),
                forecast=specialist.forecast,
            ),
        ),
    )

    routed = route_atlas_task(
        catastrophic_overlay_case, catastrophic_release, fold=None
    )

    assert routed.activated is False
    assert routed.forecast == catastrophic_overlay_case.anchor.forecast
    assert routed.fallback_reason == "predicted_regret_exceeds_limit"


def test_atlas_fitting_requires_the_exact_64_task_build_universe() -> None:
    tasks = _atlas_tasks(63)
    manifest = build_group_fold_manifest(tasks, seed=20260903)

    with pytest.raises(ValueError, match="64"):
        fit_atlas_release(_atlas_rows(tasks), manifest, AtlasPolicy())


def test_atlas_release_binds_each_model_to_its_numbered_manifest_fold() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    release = fit_atlas_release(_atlas_rows(tasks), manifest, AtlasPolicy())
    models = dict(release.build_fold_models)

    with pytest.raises(ValueError, match="fold"):
        replace(
            release,
            build_fold_models=tuple(
                (fold, models[(fold + 1) % 5]) for fold in range(5)
            ),
        )


def test_atlas_fitting_rejects_different_truths_within_one_task() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    rows = list(_atlas_rows(tasks))
    rows[1] = replace(rows[1], truth=(999.0, 1000.0))

    with pytest.raises(ValueError, match="truth"):
        fit_atlas_release(tuple(rows), manifest, AtlasPolicy())


def test_atlas_source_fingerprint_covers_every_feature_driving_field() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    rows = _atlas_rows(tasks)
    source = fit_atlas_release(rows, manifest, AtlasPolicy()).source_sha256

    changed_diagnostic = list(rows)
    changed_diagnostic[1] = replace(
        changed_diagnostic[1],
        diagnostic=replace(
            changed_diagnostic[1].diagnostic,
            median_smae=0.19,
            recent_smae=0.19,
            worst_smae=0.19,
        ),
    )
    changed_family = list(rows)
    for index, row in enumerate(changed_family):
        if row.candidate_name == "seasonal_naive":
            changed_family[index] = replace(
                row,
                family="combined",
                diagnostic=replace(row.diagnostic, family="combined"),
            )
    changed_history = list(rows)
    first_task = rows[0].task_id
    for index, row in enumerate(changed_history):
        if row.task_id == first_task:
            changed_history[index] = replace(
                row, history=tuple(value + 0.125 for value in row.history)
            )
    changed_profile = list(rows)
    for index, row in enumerate(changed_profile):
        if row.task_id == first_task:
            changed_profile[index] = replace(
                row,
                profile=replace(
                    row.profile,
                    trend_strength=row.profile.trend_strength + 0.01,
                ),
            )

    variants = (
        tuple(changed_diagnostic),
        tuple(changed_family),
        tuple(changed_history),
        tuple(changed_profile),
    )
    assert all(
        fit_atlas_release(variant, manifest, AtlasPolicy()).source_sha256
        != source
        for variant in variants
    )


def test_atlas_zero_scale_raw_regret_is_encoded_as_finite_over_cap() -> None:
    tasks = _atlas_tasks()
    manifest = build_group_fold_manifest(tasks, seed=20260903)
    rows = list(_atlas_rows(tasks))
    first_task = tasks[0].task_id
    for index, row in enumerate(rows):
        if row.task_id != first_task:
            continue
        rows[index] = replace(
            row,
            truth=(0.0, 0.0),
            forecast=(
                (0.0, 0.0)
                if row.candidate_name == "toto_2_0"
                else (1.0, 1.0)
            ),
        )

    release = fit_atlas_release(tuple(rows), manifest, AtlasPolicy())
    group_sha256 = next(
        group_sha
        for group_sha, task_ids, _fold in manifest.groups
        if first_task in task_ids
    )
    record = next(
        item
        for item in release.full_build_model.records
        if item.group_sha256 == group_sha256
        and item.candidate_name == "seasonal_naive"
    )

    assert record.regret_smae_raw > 0.25
    assert record.regret_srmse_raw > 0.25
    assert record.regret_smae_raw < float("inf")
    assert record.regret_srmse_raw < float("inf")
