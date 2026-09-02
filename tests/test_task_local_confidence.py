"""Closed evidence contracts for hierarchical task-local routing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.task_local_confidence import (
    ConfidenceEvidenceRecord,
    ConfidencePolicy,
    HierarchicalEvidenceBank,
    WeightRecipe,
    beta_win_probability,
    coarse_morphology_key,
    exact_morphology_key,
    parse_hierarchical_evidence,
    robust_effect_margin,
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
        trend_direction="up",
        trend_strength=0.6,
        periodicity_periods=(7,),
        periodicity_strength=0.8,
        periodicity_confidence=0.9,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.2,
        recent_regime_start=96,
        recent_regime_confidence=0.8,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _recipe(region: str = "full") -> WeightRecipe:
    return WeightRecipe(
        region=region,
        names=("toto_2_0", "seasonal_naive"),
        weight_units=(8, 2),
    )


def _record(
    *,
    level: str,
    group_key: str,
    recipe: WeightRecipe | None = None,
) -> ConfidenceEvidenceRecord:
    return ConfidenceEvidenceRecord(
        level=level,
        group_key=group_key,
        recipe=recipe or _recipe(),
        independent_groups=8,
        task_support=8,
        wins=7,
        ties=0,
        losses=1,
        posterior_win_probability=beta_win_probability(7, 1),
        robust_margin_joint=0.09,
        robust_margin_smae=0.1,
        robust_margin_srmse=0.08,
        p90_regret_smae_raw=0.05,
        p90_regret_srmse_raw=0.07,
        failure_count=0,
        clipped_smae_count=0,
        clipped_srmse_count=0,
    )


def test_beta_win_probability_is_exact_deterministic_and_monotone() -> None:
    assert beta_win_probability(0, 0) == 0.5
    assert beta_win_probability(4, 0) == pytest.approx(0.96875)
    assert beta_win_probability(0, 4) == pytest.approx(0.03125)
    assert beta_win_probability(4, 1) > beta_win_probability(3, 1)
    assert beta_win_probability(3, 2) < beta_win_probability(4, 1)


def test_robust_effect_margin_uses_median_minus_mad() -> None:
    assert robust_effect_margin((0.3, 0.2, -0.1), multiplier=1.0) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError, match="nonempty finite"):
        robust_effect_margin((), multiplier=1.0)


def test_morphology_keys_are_task_identity_blind_and_hierarchical() -> None:
    base = _profile("first")
    relabelled = replace(base, task_id="second")
    different_frequency_and_length = replace(
        base,
        task_id="third",
        frequency="H",
        history_length=400,
        horizon=100,
    )

    assert exact_morphology_key(base) == exact_morphology_key(relabelled)
    assert exact_morphology_key(base) != exact_morphology_key(
        different_frequency_and_length
    )
    assert coarse_morphology_key(base) == coarse_morphology_key(
        different_frequency_and_length
    )


def test_weight_recipe_rejects_non_anchor_heavy_or_noncanonical_weights() -> None:
    with pytest.raises(ValueError, match="anchor weight"):
        WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (4, 6))
    with pytest.raises(ValueError, match="sum to ten"):
        WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (8, 1))
    with pytest.raises(ValueError, match="unique"):
        WeightRecipe("full", ("toto_2_0", "TOTO_2_0"), (8, 2))


def test_evidence_bank_resolves_exact_then_coarse_then_global() -> None:
    profile = _profile()
    recipe = _recipe()
    policy = ConfidencePolicy(
        exact_minimum_support=8,
        coarse_minimum_support=8,
        global_minimum_support=8,
    )
    exact = _record(
        level="exact", group_key=exact_morphology_key(profile), recipe=recipe
    )
    coarse = _record(
        level="coarse", group_key=coarse_morphology_key(profile), recipe=recipe
    )
    global_record = _record(
        level="global",
        group_key=HierarchicalEvidenceBank.global_group_key(),
        recipe=recipe,
    )
    bank = HierarchicalEvidenceBank.build(
        policy,
        records=(global_record, coarse, exact),
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )

    assert bank.resolve(profile, recipe) == exact
    without_exact = HierarchicalEvidenceBank.build(
        policy,
        records=(global_record, coarse),
        fit_group_ids=bank.fit_group_ids,
    )
    assert without_exact.resolve(profile, recipe) == coarse
    global_only = HierarchicalEvidenceBank.build(
        policy,
        records=(global_record,),
        fit_group_ids=bank.fit_group_ids,
    )
    assert global_only.resolve(profile, recipe) == global_record


def test_confidence_parser_round_trips_and_rejects_unknown_fields() -> None:
    profile = _profile()
    record = _record(level="exact", group_key=exact_morphology_key(profile))
    bank = HierarchicalEvidenceBank.build(
        ConfidencePolicy(),
        records=(record,),
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )

    assert parse_hierarchical_evidence(bank.to_payload()) == bank
    malformed = bank.to_payload()
    malformed["records"][0]["mean_smae"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="evidence record schema"):
        parse_hierarchical_evidence(malformed)


def test_confidence_policy_bounds_prior_single_metric_regression() -> None:
    assert ConfidencePolicy().maximum_prior_metric_regression == 0.05
    with pytest.raises(ValueError, match="prior metric regression"):
        ConfidencePolicy(maximum_prior_metric_regression=-0.01)


def test_evidence_bank_rejects_forged_derived_probability() -> None:
    profile = _profile()
    bank = HierarchicalEvidenceBank.build(
        ConfidencePolicy(),
        records=(
            _record(level="exact", group_key=exact_morphology_key(profile)),
        ),
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )
    payload = bank.to_payload()
    payload["records"][0]["posterior_win_probability"] = 0.999  # type: ignore[index]
    with pytest.raises(ValueError, match="posterior probability"):
        parse_hierarchical_evidence(payload)
