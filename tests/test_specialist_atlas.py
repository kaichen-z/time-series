"""Specialist Atlas contracts and Train-only candidate discovery."""

from __future__ import annotations

from dataclasses import replace

import pytest

from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.specialist_atlas import (
    AtlasPolicy,
    atlas_feature,
    select_specialist_pool,
)
from numerical_agent.evolution.task_local_evolution import TaskLocalTaskRow


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
