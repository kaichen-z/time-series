from __future__ import annotations

import pytest

from numerical_agent.evolution.assumptions import (
    ForecastAssumption,
    assumption_candidate_pool,
    generate_forecast_assumptions,
    rank_diverse_assumptions,
)
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile


def _profile(**changes) -> TaskProfile:
    payload = {
        "task_id": "history-only-id",
        "frequency": "D",
        "history_length": 80,
        "horizon": 8,
        "zero_fraction": 0.0,
        "signed": False,
        "integer_valued": False,
        "trend_direction": "flat",
        "trend_strength": 0.1,
        "periodicity_periods": (),
        "periodicity_strength": 0.0,
        "periodicity_confidence": 0.0,
        "outlier_fraction": 0.0,
        "noise_relative_scale": 0.2,
        "likely_stationary": True,
        "stationarity_score": 0.8,
        "recent_regime_start": None,
        "recent_regime_confidence": 0.0,
        "intermittency_adi": 1.0,
        "intermittency_cv2": 0.1,
    }
    payload.update(changes)
    return TaskProfile(**payload)


def _diagnostic(name: str, family: str, *, median: float, worst: float, recent: float):
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=median,
        worst_mase=worst,
        recent_mase=recent,
        mase_mad=0.05,
    )


def test_generator_emits_falsifiable_periodic_assumption_from_history_profile():
    profile = _profile(
        periodicity_periods=(7,),
        periodicity_strength=0.8,
        periodicity_confidence=0.75,
    )
    assumptions = generate_forecast_assumptions(
        profile,
        active_names=("toto_2_0", "seasonal_naive", "fourier_harmonic_regression"),
        families={
            "toto_2_0": "tsfm",
            "seasonal_naive": "statistical",
            "fourier_harmonic_regression": "statistical",
        },
    )

    periodic = next(item for item in assumptions if item.kind == "periodic_persistence")
    assert periodic.claim == "The supported 7-step historical cycle will persist over the horizon."
    assert periodic.supporting_signals == (
        "periodicity_period=7",
        "periodicity_strength=0.8000",
        "periodicity_confidence=0.7500",
    )
    assert "temporary" in periodic.failure_condition.lower()
    assert periodic.candidate_names == (
        "fourier_harmonic_regression",
        "seasonal_naive",
    )


def test_generator_routes_intermittent_methods_only_when_history_supports_them():
    active = ("croston", "croston_sba", "naive_last", "toto_2_0")
    families = {
        "croston": "statistical",
        "croston_sba": "statistical",
        "naive_last": "statistical",
        "toto_2_0": "tsfm",
    }

    dense = generate_forecast_assumptions(_profile(), active, families)
    sparse = generate_forecast_assumptions(
        _profile(zero_fraction=0.65, intermittency_adi=2.4, intermittency_cv2=1.2),
        active,
        families,
    )

    assert all(item.kind != "intermittent_demand" for item in dense)
    intermittent = next(item for item in sparse if item.kind == "intermittent_demand")
    assert intermittent.candidate_names == ("croston", "croston_sba")
    assert intermittent.prior_confidence > 0.5


def test_assumption_contract_rejects_non_finite_confidence_and_empty_evidence():
    with pytest.raises(ValueError, match="confidence"):
        ForecastAssumption(
            assumption_id="bad",
            kind="fallback",
            claim="A claim.",
            supporting_signals=("history_only",),
            failure_condition="It fails.",
            candidate_names=("naive_last",),
            prior_confidence=float("nan"),
        )
    with pytest.raises(ValueError, match="supporting"):
        ForecastAssumption(
            assumption_id="bad",
            kind="fallback",
            claim="A claim.",
            supporting_signals=(),
            failure_condition="It fails.",
            candidate_names=("naive_last",),
            prior_confidence=0.5,
        )


def test_top_k_is_diverse_by_kind_and_leading_candidate():
    assumptions = (
        ForecastAssumption(
            "a1", "periodic_persistence", "Cycle persists.", ("period=7",),
            "Cycle breaks.", ("seasonal_naive", "toto_2_0"), 0.8,
        ),
        ForecastAssumption(
            "a2", "trend_persistence", "Trend persists.", ("trend=up",),
            "Trend reverses.", ("seasonal_naive", "linear_trend_regression"), 0.7,
        ),
        ForecastAssumption(
            "a3", "foundation_shape", "TSFM shape persists.", ("anchor=toto",),
            "Regime changes.", ("toto_2_0",), 0.6,
        ),
    )
    diagnostics = {
        "seasonal_naive": _diagnostic(
            "seasonal_naive", "statistical", median=0.4, worst=0.6, recent=0.3
        ),
        "linear_trend_regression": _diagnostic(
            "linear_trend_regression", "statistical", median=0.5, worst=0.7, recent=0.4
        ),
        "toto_2_0": _diagnostic(
            "toto_2_0", "tsfm", median=0.8, worst=0.9, recent=0.7
        ),
    }

    ranked = rank_diverse_assumptions(
        assumptions,
        diagnostics,
        top_k=3,
        candidates_per_assumption=2,
        min_confidence=0.0,
    )

    assert tuple(item.assumption.assumption_id for item in ranked) == ("a1", "a3")
    assert tuple(item.leading_candidate for item in ranked) == (
        "seasonal_naive",
        "toto_2_0",
    )


def test_assumption_leader_uses_the_canonical_scaled_pair_tie_break_order():
    assumption = ForecastAssumption(
        "scaled-order",
        "foundation_shape",
        "The supported shape persists.",
        ("history_only",),
        "The shape changes.",
        ("z_lower_smae", "a_name_only"),
        0.8,
    )
    diagnostics = {
        "z_lower_smae": CandidateDiagnostics.synthetic(
            name="z_lower_smae",
            family="tsfm",
            median_mase=1.0,
            median_smae=0.9,
            median_srmse=1.1,
        ),
        "a_name_only": CandidateDiagnostics.synthetic(
            name="a_name_only",
            family="tsfm",
            median_mase=1.0,
            median_smae=1.0,
            median_srmse=1.0,
        ),
    }

    ranked = rank_diverse_assumptions(
        (assumption,),
        diagnostics,
        top_k=1,
        candidates_per_assumption=2,
        min_confidence=0.0,
    )

    assert ranked[0].leading_candidate == "z_lower_smae"


def test_candidate_pool_keeps_reviewed_anchors_beside_top_k_methods():
    assumptions = generate_forecast_assumptions(
        _profile(trend_direction="up", trend_strength=0.8),
        active_names=("linear_trend_regression", "toto_2_0", "timesfm_2_5"),
        families={
            "linear_trend_regression": "statistical",
            "toto_2_0": "tsfm",
            "timesfm_2_5": "tsfm",
        },
    )
    diagnostics = {
        "linear_trend_regression": _diagnostic(
            "linear_trend_regression", "statistical", median=0.1, worst=0.2, recent=0.1
        ),
        "toto_2_0": _diagnostic("toto_2_0", "tsfm", median=2.0, worst=2.0, recent=2.0),
        "timesfm_2_5": _diagnostic(
            "timesfm_2_5", "tsfm", median=3.0, worst=3.0, recent=3.0
        ),
    }
    ranked = rank_diverse_assumptions(
        assumptions,
        diagnostics,
        top_k=1,
        candidates_per_assumption=1,
        min_confidence=0.0,
    )

    pool = assumption_candidate_pool(
        ranked,
        active_names=tuple(diagnostics),
        anchor_names=("toto_2_0", "timesfm_2_5"),
    )

    assert pool == ("linear_trend_regression", "toto_2_0", "timesfm_2_5")


def test_top_k_preserves_available_statistical_tsfm_and_combined_families():
    assumptions = (
        ForecastAssumption(
            "stat-1", "trend_persistence", "Trend persists.", ("trend=up",),
            "Trend reverses.", ("linear_trend_regression",), 0.8,
        ),
        ForecastAssumption(
            "stat-2", "stationary_local_dynamics", "Lags persist.", ("stationary=1",),
            "A break occurs.", ("arima_auto",), 0.8,
        ),
        ForecastAssumption(
            "foundation", "foundation_shape", "Shape persists.", ("anchor=toto",),
            "A break occurs.", ("toto_2_0",), 0.6,
        ),
        ForecastAssumption(
            "combined", "periodic_persistence", "Cycle persists.", ("period=7",),
            "Phase changes.", ("combined_timesfm_seasonal",), 0.7,
        ),
    )
    diagnostics = {
        "linear_trend_regression": _diagnostic(
            "linear_trend_regression", "statistical", median=0.1, worst=0.1, recent=0.1
        ),
        "arima_auto": _diagnostic(
            "arima_auto", "statistical", median=0.2, worst=0.2, recent=0.2
        ),
        "toto_2_0": _diagnostic(
            "toto_2_0", "tsfm", median=0.8, worst=0.8, recent=0.8
        ),
        "combined_timesfm_seasonal": _diagnostic(
            "combined_timesfm_seasonal", "combined", median=0.9, worst=0.9, recent=0.9
        ),
    }

    ranked = rank_diverse_assumptions(
        assumptions,
        diagnostics,
        top_k=3,
        candidates_per_assumption=1,
        min_confidence=0.0,
    )

    assert {item.leading_family for item in ranked} == {"statistical", "tsfm", "combined"}


def test_stationary_assumption_does_not_route_trend_method_via_short_ar_substring():
    assumptions = generate_forecast_assumptions(
        _profile(likely_stationary=True, stationarity_score=0.9),
        active_names=("ar", "linear_trend_regression", "toto_2_0"),
        families={
            "ar": "statistical",
            "linear_trend_regression": "statistical",
            "toto_2_0": "tsfm",
        },
    )

    stationary = next(item for item in assumptions if item.kind == "stationary_local_dynamics")

    assert stationary.candidate_names == ("ar",)
