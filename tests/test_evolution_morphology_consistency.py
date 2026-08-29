from __future__ import annotations

from dataclasses import replace

import pytest

from numerical_agent.evolution.morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyObservation,
    MorphologyToolCall,
)
from numerical_agent.evolution.morphology_consistency import check_morphology_assumptions
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics, DecisionPolicy
from numerical_agent.evolution.screening import TaskProfile


def _profile(**changes: object) -> TaskProfile:
    base = dict(
        task_id="train-1",
        frequency="D",
        history_length=84,
        horizon=3,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="flat",
        trend_strength=0.05,
        periodicity_periods=(7,),
        periodicity_strength=0.8,
        periodicity_confidence=0.9,
        outlier_fraction=0.0,
        noise_relative_scale=0.05,
        likely_stationary=True,
        stationarity_score=0.8,
        recent_regime_start=None,
        recent_regime_confidence=0.0,
        intermittency_adi=1.0,
        intermittency_cv2=0.0,
    )
    base.update(changes)
    return TaskProfile(**base)


def _card(*assumptions: AssumptionGrounding) -> MorphologyCard:
    broad = MorphologyToolCall("broad", "detect_periodicity", 0, 84)
    recent = MorphologyToolCall("recent", "detect_periodicity", 42, 84)
    return MorphologyCard(
        "Recent history.",
        "Full history.",
        (broad, recent),
        (
            MorphologyObservation(broad, {"strength": 0.8}),
            MorphologyObservation(recent, {"strength": 0.7}),
        ),
        assumptions,
    )


def _assumption(
    assumption_id: str,
    kind: str,
    *candidates: str,
    confidence: float = 0.8,
) -> AssumptionGrounding:
    return AssumptionGrounding(
        assumption_id,
        kind,
        "A grounded historical condition persists.",
        "The historical condition does not persist.",
        ("broad", "recent"),
        candidates,
        confidence,
    )


def _diagnostic(name: str, **changes: object) -> CandidateDiagnostics:
    return replace(
        CandidateDiagnostics.synthetic(
            name=name,
            family="statistical",
            median_mase=0.4,
            fold_forecasts=((1.0, 2.0, 3.0),) * 3,
            fold_truths=((1.0, 2.0, 3.0),) * 3,
        ),
        **changes,
    )


def _check(card: MorphologyCard, profile: TaskProfile, diagnostics: dict[str, CandidateDiagnostics], forecasts: dict[str, tuple[float, ...]], **kwargs: object):
    return check_morphology_assumptions(
        card,
        profile=profile,
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts=forecasts,
        min_successful_folds=3,
        **kwargs,
    )


def test_consistency_accepts_supported_weekly_cycle_and_rejects_flat_trend() -> None:
    weekly = _assumption("weekly_cycle", "seasonality", "seasonal_naive")
    trend = _assumption("trend_persistence", "trend", "trend_model")
    diagnostics = {
        "seasonal_naive": _diagnostic("seasonal_naive"),
        "trend_model": _diagnostic("trend_model"),
    }

    result = _check(
        _card(weekly, trend),
        _profile(),
        diagnostics,
        {"seasonal_naive": (1.0, 2.0, 3.0), "trend_model": (1.0, 2.0, 3.0)},
    )

    assert tuple(x.assumption_id for x in result.accepted) == ("weekly_cycle",)
    assert result.rejected["trend_persistence"] == "profile_incompatible"


@pytest.mark.parametrize(
    ("kind", "profile_changes"),
    [
        ("seasonality", {}),
        ("trend", {"trend_direction": "up", "trend_strength": 0.7}),
        ("intermittency", {"zero_fraction": 0.6}),
        ("regime", {"recent_regime_start": 60, "recent_regime_confidence": 0.8}),
        ("noise", {"noise_relative_scale": 0.8}),
        ("level", {}),
    ],
)
def test_consistency_uses_only_typed_profile_predicates(
    kind: str, profile_changes: dict[str, object]
) -> None:
    assumption = _assumption(f"{kind}_supported", kind, "candidate")

    result = _check(
        _card(assumption),
        _profile(**profile_changes),
        {"candidate": _diagnostic("candidate")},
        {"candidate": (1.0, 2.0, 3.0)},
    )

    assert tuple(item.assumption_id for item in result.accepted) == (f"{kind}_supported",)


def test_consistency_rejects_bad_candidates_without_mutating_candidates_or_fallback() -> None:
    active = ("good", "failed", "toto_2_0")
    diagnostics = {
        "good": _diagnostic("good"),
        "failed": _diagnostic("failed", eligible=False, successful_folds=0),
        "toto_2_0": _diagnostic("toto_2_0"),
    }
    forecasts = {name: (1.0, 2.0, 3.0) for name in active}
    card = _card(
        _assumption("inactive", "seasonality", "missing"),
        _assumption("failed", "seasonality", "failed"),
    )

    result = check_morphology_assumptions(
        card,
        profile=_profile(),
        active_names=active,
        diagnostics=diagnostics,
        forecasts=forecasts,
        min_successful_folds=3,
    )

    assert result.accepted == ()
    assert result.rejected == {"inactive": "inactive_candidate", "failed": "candidate_ineligible"}
    assert active == ("good", "failed", "toto_2_0")
    assert tuple(diagnostics) == active
    assert forecasts["toto_2_0"] == (1.0, 2.0, 3.0)


def test_consistency_enforces_fold_worst_fold_catastrophe_and_forecast_gates() -> None:
    assumptions = _card(
        _assumption("few_folds", "seasonality", "few"),
        _assumption("bad_worst_fold", "seasonality", "worst"),
        _assumption("exploded", "seasonality", "exploded"),
        _assumption("bad_forecast", "seasonality", "malformed"),
    )
    diagnostics = {
        "few": _diagnostic("few", successful_folds=2),
        "worst": _diagnostic("worst", worst_mase=10.1),
        "exploded": _diagnostic("exploded", explosion=True),
        "malformed": _diagnostic("malformed"),
    }

    result = _check(
        assumptions,
        _profile(),
        diagnostics,
        {
            "few": (1.0, 2.0, 3.0),
            "worst": (1.0, 2.0, 3.0),
            "exploded": (1.0, 2.0, 3.0),
            "malformed": (1.0, float("nan"), 3.0),
        },
        policy=DecisionPolicy(catastrophic_mase=10.0),
    )

    assert result.rejected == {
        "few_folds": "insufficient_successful_folds",
        "bad_worst_fold": "catastrophic_hindcast_tail",
        "exploded": "catastrophic_hindcast_tail",
        "bad_forecast": "invalid_forecast",
    }


def test_consistency_fails_closed_on_nonfinite_diagnostics() -> None:
    result = _check(
        _card(_assumption("weekly_cycle", "seasonality", "candidate")),
        _profile(),
        {"candidate": _diagnostic("candidate", phase_error=float("nan"))},
        {"candidate": (1.0, 2.0, 3.0)},
    )

    assert result.accepted == ()
    assert result.rejected == {"weekly_cycle": "invalid_diagnostics"}


def test_consistency_fails_closed_when_a_hostile_diagnostics_mapping_changes() -> None:
    class _ChangingDiagnostics(dict[str, CandidateDiagnostics]):
        calls = 0

        def get(self, key: str, default: object = None):  # type: ignore[override]
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("changed after validation")
            return super().get(key, default)

    result = _check(
        _card(_assumption("weekly_cycle", "seasonality", "candidate")),
        _profile(),
        _ChangingDiagnostics({"candidate": _diagnostic("candidate")}),
        {"candidate": (1.0, 2.0, 3.0)},
    )

    assert result.accepted == ()
    assert result.rejected == {"weekly_cycle": "invalid_diagnostics"}


def test_consistency_fails_closed_when_a_hostile_forecast_mapping_changes() -> None:
    class _ChangingForecasts(dict[str, tuple[float, ...]]):
        calls = 0

        def get(self, key: str, default: object = None):  # type: ignore[override]
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("changed after validation")
            return super().get(key, default)

    result = _check(
        _card(_assumption("weekly_cycle", "seasonality", "candidate")),
        _profile(),
        {"candidate": _diagnostic("candidate")},
        _ChangingForecasts({"candidate": (1.0, 2.0, 3.0)}),
    )

    assert result.accepted == ()
    assert result.rejected == {"weekly_cycle": "invalid_forecasts"}


def test_consistency_retains_only_diverse_top_k_assumptions() -> None:
    first = _assumption("weekly_cycle", "seasonality", "seasonal_a", confidence=0.9)
    second = _assumption("weekly_cycle_alt", "seasonality", "seasonal_b", confidence=0.8)
    level = _assumption("stable_level", "level", "level_model", confidence=0.7)
    diagnostics = {
        "seasonal_a": _diagnostic("seasonal_a"),
        "seasonal_b": _diagnostic("seasonal_b", median_mase=0.5),
        "level_model": _diagnostic("level_model", median_mase=0.6),
    }

    result = _check(
        _card(first, second, level),
        _profile(),
        diagnostics,
        {name: (1.0, 2.0, 3.0) for name in diagnostics},
        policy=DecisionPolicy(assumption_top_k=3),
    )

    assert tuple(item.assumption_id for item in result.accepted) == ("weekly_cycle", "stable_level")
    assert result.rejected["weekly_cycle_alt"] == "diversity_rejected"
