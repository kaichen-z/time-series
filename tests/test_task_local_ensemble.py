"""History-only runtime contracts for the task-local Numerical tournament."""

from __future__ import annotations

import math

import pytest

from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile
from numerical_agent.evolution.task_local_confidence import (
    ConfidenceEvidenceRecord,
    ConfidencePolicy,
    HierarchicalEvidenceBank,
    WeightRecipe,
    beta_win_probability,
)
from numerical_agent.evolution.task_local_ensemble import (
    TaskLocalTournamentPolicy,
    execute_confidence_task_local_ensemble,
    execute_task_local_ensemble,
)


TRUTHS = ((10.0, 10.0), (10.0, 10.0), (10.0, 10.0))


def _diagnostic(
    name: str,
    forecasts: tuple[tuple[float, ...], ...],
    *,
    family: str,
) -> CandidateDiagnostics:
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=1.0,
        fold_forecasts=forecasts,
        fold_truths=TRUTHS,
        median_smae=1.0,
        median_srmse=1.0,
    )


def _policy(**overrides: object) -> TaskLocalTournamentPolicy:
    values: dict[str, object] = {
        "anchor_name": "toto_2_0",
        "minimum_successful_folds": 3,
        "minimum_joint_improvement": 0.02,
        "maximum_worst_joint_regret": 0.25,
        "maximum_raw_smae": 10.0,
        "maximum_raw_srmse": 10.0,
    }
    values.update(overrides)
    return TaskLocalTournamentPolicy(**values)


def _valid_inputs() -> dict[str, object]:
    anchor_folds = ((8.0, 8.0),) * 3
    specialist_folds = ((10.0, 10.0),) * 3
    return {
        "candidate_names": ("toto_2_0", "seasonal_naive"),
        "forecasts": {
            "toto_2_0": (8.0, 8.0),
            "seasonal_naive": (10.0, 10.0),
        },
        "diagnostics": {
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "seasonal_naive": _diagnostic(
                "seasonal_naive", specialist_folds, family="statistical"
            ),
        },
        "horizon": 2,
    }


def test_local_tournament_returns_exact_anchor_when_specialist_is_invalid() -> None:
    inputs = _valid_inputs()
    inputs["forecasts"] = {
        "toto_2_0": (8.0, 8.0),
        "seasonal_naive": (math.nan, 10.0),
    }

    result = execute_task_local_ensemble(_policy(), **inputs)

    assert result.forecast == (8.0, 8.0)
    assert result.selected_names == ("toto_2_0",)
    assert result.weights == (1.0,)
    assert result.activated is False
    assert result.fallback_reason == "no_eligible_specialist"


def test_local_tournament_weights_are_deterministic_normalized_and_anchor_heavy() -> None:
    inputs = _valid_inputs()
    first = execute_task_local_ensemble(_policy(), **inputs)
    second = execute_task_local_ensemble(
        _policy(),
        candidate_names=("seasonal_naive", "toto_2_0"),
        forecasts=dict(reversed(tuple(inputs["forecasts"].items()))),  # type: ignore[union-attr]
        diagnostics=dict(reversed(tuple(inputs["diagnostics"].items()))),  # type: ignore[union-attr]
        horizon=2,
    )

    assert first == second
    assert first.activated is True
    assert first.selected_names == ("toto_2_0", "seasonal_naive")
    assert first.weights == (0.5, 0.5)
    assert sum(first.weights) == pytest.approx(1.0, abs=1e-12)
    assert first.weights[first.selected_names.index("toto_2_0")] >= 0.5
    assert first.maximum_fold_regret == 0.0


def test_smae_only_gain_cannot_hide_srmse_regression() -> None:
    anchor_folds = ((9.0, 9.0),) * 3
    specialist_folds = ((10.0, 8.2),) * 3
    result = execute_task_local_ensemble(
        _policy(),
        candidate_names=("toto_2_0", "uneven_specialist"),
        forecasts={"toto_2_0": (9.0, 9.0), "uneven_specialist": (10.0, 8.2)},
        diagnostics={
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "uneven_specialist": _diagnostic(
                "uneven_specialist", specialist_folds, family="statistical"
            ),
        },
        horizon=2,
    )

    assert result.activated is False
    assert result.forecast == (9.0, 9.0)
    assert result.fallback_reason == "no_pareto_safe_weight"


def test_worst_fold_regression_rejects_better_median_weight() -> None:
    anchor_folds = ((8.0, 8.0),) * 3
    specialist_folds = ((10.0, 10.0), (10.0, 10.0), (0.0, 0.0))
    result = execute_task_local_ensemble(
        _policy(maximum_worst_joint_regret=0.1),
        candidate_names=("toto_2_0", "unstable_specialist"),
        forecasts={"toto_2_0": (8.0, 8.0), "unstable_specialist": (10.0, 10.0)},
        diagnostics={
            "toto_2_0": _diagnostic("toto_2_0", anchor_folds, family="tsfm"),
            "unstable_specialist": _diagnostic(
                "unstable_specialist", specialist_folds, family="statistical"
            ),
        },
        horizon=2,
    )

    assert result.activated is False
    assert result.forecast == (8.0, 8.0)
    assert result.fallback_reason == "no_eligible_specialist"


def test_tournament_rejects_more_than_eight_candidates() -> None:
    inputs = _valid_inputs()
    with pytest.raises(ValueError, match="at most eight"):
        execute_task_local_ensemble(
            _policy(),
            candidate_names=("toto_2_0", *(f"method_{index}" for index in range(8))),
            forecasts=inputs["forecasts"],
            diagnostics=inputs["diagnostics"],
            horizon=2,
        )


def _profile(*, horizon: int) -> TaskProfile:
    return TaskProfile(
        task_id="task",
        frequency="D",
        history_length=120,
        horizon=horizon,
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


def _confidence_policy() -> ConfidencePolicy:
    return ConfidencePolicy(
        exact_minimum_support=8,
        coarse_minimum_support=8,
        global_minimum_support=8,
        minimum_paired_origins=3,
        regional_minimum_horizon=4,
    )


def _confidence_bank(
    profile: TaskProfile,
    *recipes: WeightRecipe,
    robust_margin_joint: float = 0.1,
    robust_margin_smae: float = 0.1,
    robust_margin_srmse: float = 0.1,
) -> HierarchicalEvidenceBank:
    records = tuple(
        ConfidenceEvidenceRecord(
            level="global",
            group_key=HierarchicalEvidenceBank.global_group_key(),
            recipe=recipe,
            independent_groups=8,
            task_support=8,
            wins=7,
            ties=0,
            losses=1,
            posterior_win_probability=beta_win_probability(7, 1),
            robust_margin_joint=robust_margin_joint,
            robust_margin_smae=robust_margin_smae,
            robust_margin_srmse=robust_margin_srmse,
            p90_regret_smae_raw=0.0,
            p90_regret_srmse_raw=0.0,
            failure_count=0,
            clipped_smae_count=0,
            clipped_srmse_count=0,
        )
        for recipe in recipes
    )
    return HierarchicalEvidenceBank.build(
        _confidence_policy(),
        records=records,
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )


def _confidence_inputs(
    *,
    anchor_folds: tuple[tuple[float, ...], ...],
    specialist_folds: tuple[tuple[float, ...], ...],
    truths: tuple[tuple[float, ...], ...],
    anchor_forecast: tuple[float, ...],
    specialist_forecast: tuple[float, ...],
) -> dict[str, object]:
    def diagnostic(
        name: str,
        family: str,
        folds: tuple[tuple[float, ...], ...],
    ) -> CandidateDiagnostics:
        return CandidateDiagnostics.synthetic(
            name=name,
            family=family,
            median_mase=1.0,
            fold_forecasts=folds,
            fold_truths=truths,
            median_smae=1.0,
            median_srmse=1.0,
        )

    return {
        "candidate_names": ("toto_2_0", "seasonal_naive"),
        "forecasts": {
            "toto_2_0": anchor_forecast,
            "seasonal_naive": specialist_forecast,
        },
        "diagnostics": {
            "toto_2_0": diagnostic("toto_2_0", "tsfm", anchor_folds),
            "seasonal_naive": diagnostic(
                "seasonal_naive", "statistical", specialist_folds
            ),
        },
        "horizon": len(anchor_forecast),
    }


def test_good_local_hindcast_without_qualified_group_prior_falls_back() -> None:
    profile = _profile(horizon=2)
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0),) * 5,
        specialist_folds=((10.0, 10.0),) * 5,
        truths=((10.0, 10.0),) * 5,
        anchor_forecast=(8.0, 8.0),
        specialist_forecast=(10.0, 10.0),
    )
    empty = HierarchicalEvidenceBank.build(
        _confidence_policy(),
        records=(),
        fit_group_ids=tuple(f"{index:064x}" for index in range(1, 9)),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=empty,
        **inputs,
    )

    assert result.forecast == inputs["forecasts"]["toto_2_0"]  # type: ignore[index]
    assert result.activated is False
    assert result.fallback_reason == "confidence_prior_unavailable"


def test_good_prior_with_bad_local_hindcast_falls_back() -> None:
    profile = _profile(horizon=2)
    recipe = WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (5, 5))
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0),) * 5,
        specialist_folds=((6.0, 6.0),) * 5,
        truths=((10.0, 10.0),) * 5,
        anchor_forecast=(8.0, 8.0),
        specialist_forecast=(6.0, 6.0),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=_confidence_bank(profile, recipe),
        **inputs,
    )

    assert result.forecast == (8.0, 8.0)
    assert result.activated is False
    assert result.fallback_reason == "local_hindcast_not_confident"


def test_early_specialist_and_late_anchor_are_concatenated_exactly() -> None:
    profile = _profile(horizon=4)
    early = WeightRecipe("early", ("toto_2_0", "seasonal_naive"), (5, 5))
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0, 20.0, 20.0),) * 5,
        specialist_folds=((10.0, 10.0, 0.0, 0.0),) * 5,
        truths=((10.0, 10.0, 20.0, 20.0),) * 5,
        anchor_forecast=(8.0, 8.0, 20.0, 20.0),
        specialist_forecast=(10.0, 10.0, 0.0, 0.0),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=_confidence_bank(profile, early),
        **inputs,
    )

    assert result.forecast == (9.0, 9.0, 20.0, 20.0)
    assert tuple(region.region for region in result.regions) == ("early", "late")
    assert tuple(region.activated for region in result.regions) == (True, False)


def test_short_horizon_uses_full_gate_without_fabricated_regions() -> None:
    profile = _profile(horizon=2)
    full = WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (5, 5))
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0),) * 5,
        specialist_folds=((10.0, 10.0),) * 5,
        truths=((10.0, 10.0),) * 5,
        anchor_forecast=(8.0, 8.0),
        specialist_forecast=(10.0, 10.0),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=_confidence_bank(profile, full),
        **inputs,
    )

    assert result.forecast == (9.0, 9.0)
    assert tuple(region.region for region in result.regions) == ("full",)
    assert result.activated is True


def test_joint_positive_prior_allows_small_single_metric_group_tradeoff() -> None:
    profile = _profile(horizon=2)
    recipe = WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (5, 5))
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0),) * 5,
        specialist_folds=((10.0, 10.0),) * 5,
        truths=((10.0, 10.0),) * 5,
        anchor_forecast=(8.0, 8.0),
        specialist_forecast=(10.0, 10.0),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=_confidence_bank(
            profile,
            recipe,
            robust_margin_joint=0.05,
            robust_margin_srmse=-0.01,
        ),
        **inputs,
    )

    assert result.activated is True
    assert result.forecast == (9.0, 9.0)


@pytest.mark.parametrize(
    ("joint_margin", "srmse_margin"),
    ((0.0, 0.1), (0.1, -0.051)),
)
def test_nonpositive_joint_or_excessive_prior_tradeoff_falls_back(
    joint_margin: float,
    srmse_margin: float,
) -> None:
    profile = _profile(horizon=2)
    recipe = WeightRecipe("full", ("toto_2_0", "seasonal_naive"), (5, 5))
    inputs = _confidence_inputs(
        anchor_folds=((8.0, 8.0),) * 5,
        specialist_folds=((10.0, 10.0),) * 5,
        truths=((10.0, 10.0),) * 5,
        anchor_forecast=(8.0, 8.0),
        specialist_forecast=(10.0, 10.0),
    )

    result = execute_confidence_task_local_ensemble(
        _policy(),
        _confidence_policy(),
        profile=profile,
        confidence_evidence=_confidence_bank(
            profile,
            recipe,
            robust_margin_joint=joint_margin,
            robust_margin_srmse=srmse_margin,
        ),
        **inputs,
    )

    assert result.activated is False
    assert result.forecast == (8.0, 8.0)


def test_v1_tournament_result_is_unchanged_by_confidence_support() -> None:
    result = execute_task_local_ensemble(_policy(), **_valid_inputs())

    assert result.forecast == (9.0, 9.0)
    assert result.selected_names == ("toto_2_0", "seasonal_naive")
    assert result.weights == (0.5, 0.5)
    assert result.policy_fingerprint == (
        "b0a28d67960d80210193eec68b46ee359d3a1eec93db15968d2cbd3c7988c77b"
    )
