"""Deterministic public-API smoke coverage for Champion evolution."""

from __future__ import annotations

import pytest

from common.data import Task
from numerical_agent.evolution.champion import ChampionRecipe, EvolutionAssumption
from numerical_agent.evolution.champion_controller import ChampionLifecycleError, canonical_release_bytes
from numerical_agent.run_champion_evolution import run_fake_champion_evolution


def fixture_tasks(count: int, *, prefix: str = "task") -> tuple[Task, ...]:
    return tuple(
        Task(
            task_id=f"{prefix}_{index:02d}",
            history_values=(1.0, 2.0, 3.0, 4.0) * 8,
            future_values=(5.0, 6.0),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name=f"{prefix}_entity_{index:02d}",
        )
        for index in range(count)
    )


def timesfm_seasonal_recipe_without_toto() -> tuple[ChampionRecipe, ...]:
    return tuple(
        ChampionRecipe(
            name=f"timesfm_seasonal_child_{index}",
            kind="weighted",
            parents=("timesfm_2_5", "seasonal_naive"),
            fallback_parent="timesfm_2_5",
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"timesfm_seasonal_assumption_{index}",
                    candidate_name="timesfm_2_5",
                    feature="history_length",
                    direction="above",
                    horizon_region="full",
                    operator="weighted",
                    rationale="The deterministic smoke has sufficient observed history.",
                    failure_condition="The observed history is no longer available.",
                ),
            ),
        )
        for index in range(5)
    )


def diverse_timesfm_seasonal_recipes() -> tuple[ChampionRecipe, ...]:
    specifications = (
        ("select", ("timesfm_2_5",), "timesfm_2_5", "history_length", "full"),
        (
            "route",
            ("seasonal_naive", "timesfm_2_5"),
            "seasonal_naive",
            "periodicity_confidence",
            "full",
        ),
        (
            "horizon_route",
            ("seasonal_naive", "timesfm_2_5"),
            "seasonal_naive",
            "periodicity_strength",
            "late",
        ),
        (
            "weighted",
            ("seasonal_naive", "timesfm_2_5"),
            "seasonal_naive",
            "periodicity_confidence",
            "full",
        ),
        (
            "median",
            ("seasonal_naive", "timesfm_2_5"),
            "seasonal_naive",
            "outlier_fraction",
            "full",
        ),
        (
            "bounded_overlay",
            ("seasonal_naive", "timesfm_2_5"),
            "seasonal_naive",
            "periodicity_strength",
            "full",
        ),
    )
    return tuple(
        ChampionRecipe(
            name=f"diverse_child_{index}",
            kind=kind,
            parents=parents,
            fallback_parent="timesfm_2_5",
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"diverse_assumption_{index}",
                    candidate_name=candidate,
                    feature=feature,
                    direction="above",
                    horizon_region=region,
                    operator=kind,
                    rationale="The reviewed morphology feature supports this structure.",
                    failure_condition="The reviewed morphology feature no longer supports it.",
                ),
            ),
        )
        for index, (kind, parents, candidate, feature, region) in enumerate(
            specifications
        )
    )


def test_fake_e2e_evolves_non_toto_champion_and_freezes_it(tmp_path):
    outcome = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8, prefix="build"),
        calibration_tasks=fixture_tasks(2, prefix="calibration"),
        dev_tasks=fixture_tasks(2, prefix="dev"),
        proposal=timesfm_seasonal_recipe_without_toto(),
        screen_sizes=(4, 8),
        output_dir=tmp_path,
    )

    assert outcome.release.policy.recipe.parents == ("timesfm_2_5", "seasonal_naive")
    assert outcome.release.policy.recipe.fallback_parent == "timesfm_2_5"
    assert outcome.calibration_report is not None
    assert outcome.calibration_report.candidates[0].comparison is not None
    assert outcome.calibration_report.candidates[0].comparison.accepted is True
    assert outcome.dev_report is not None
    assert outcome.dev_report.candidates[0].comparison is not None
    assert outcome.dev_report.candidates[0].comparison.accepted is True


def test_fake_e2e_persists_diverse_ranked_finalists(tmp_path):
    outcome = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8, prefix="build"),
        calibration_tasks=fixture_tasks(2, prefix="calibration"),
        dev_tasks=fixture_tasks(2, prefix="dev"),
        proposal=diverse_timesfm_seasonal_recipes(),
        screen_sizes=(4, 8),
        output_dir=tmp_path,
    )

    assert outcome.release.policy.recipe.name != "timesfm_smoke_parent"


def test_fake_e2e_rejects_malformed_proposal_before_any_release(tmp_path):
    with pytest.raises((ValueError, ChampionLifecycleError), match="proposal|recipe|scripted"):
        run_fake_champion_evolution(
            build_tasks=fixture_tasks(8, prefix="build"),
            calibration_tasks=fixture_tasks(2, prefix="calibration"),
            dev_tasks=fixture_tasks(2, prefix="dev"),
            proposal=("not a ChampionRecipe",),
            screen_sizes=(4, 8),
            output_dir=tmp_path,
        )

    assert not (tmp_path / "champion_release.json").exists()


def test_fake_e2e_rejected_children_preserve_the_exact_parent_bytes(tmp_path):
    outcome = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8, prefix="build"),
        calibration_tasks=fixture_tasks(2, prefix="calibration"),
        dev_tasks=fixture_tasks(2, prefix="dev"),
        proposal=timesfm_seasonal_recipe_without_toto(),
        screen_sizes=(4, 8),
        output_dir=tmp_path,
        seasonal_error=3.0,
    )

    assert outcome.release.policy.recipe.name == "timesfm_smoke_parent"
    assert outcome.calibration_report is None
    assert outcome.dev_report is None
    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(
        outcome.release
    )


def test_fake_e2e_dev_rejection_preserves_parent_release_bytes(tmp_path):
    rejected = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8, prefix="build"),
        calibration_tasks=fixture_tasks(2, prefix="calibration"),
        dev_tasks=fixture_tasks(2, prefix="dev"),
        proposal=timesfm_seasonal_recipe_without_toto(),
        screen_sizes=(4, 8),
        output_dir=tmp_path / "rejected-parent",
        seasonal_error=3.0,
    )
    outcome = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8, prefix="build"),
        calibration_tasks=fixture_tasks(2, prefix="calibration"),
        dev_tasks=fixture_tasks(2, prefix="dev"),
        proposal=timesfm_seasonal_recipe_without_toto(),
        screen_sizes=(4, 8),
        output_dir=tmp_path / "dev-rejection",
        dev_seasonal_error=3.0,
    )

    assert outcome.calibration_report is not None
    assert outcome.dev_report is not None
    assert outcome.release.policy.recipe.name == "timesfm_smoke_parent"
    assert (tmp_path / "dev-rejection" / "champion_release.json").read_bytes() == (
        tmp_path / "rejected-parent" / "champion_release.json"
    ).read_bytes() == canonical_release_bytes(rejected.release)
