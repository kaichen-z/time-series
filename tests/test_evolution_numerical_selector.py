from __future__ import annotations

import math

import pytest

from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
    diagnose_candidate,
    pairwise_diversity,
    select_numerical_forecast,
)


def _task(history=tuple(float(i) for i in range(1, 41)), horizon=5):
    return Task("hidden-id", history, horizon, "D", (999.0,) * horizon)


def test_hindcasts_use_only_historical_prefixes_and_not_task_future():
    seen = []

    def runner(name, history, horizon, frequency):
        seen.append((name, history, horizon, frequency))
        return tuple(history[-1] + i + 1 for i in range(horizon))

    diagnostic = diagnose_candidate(
        _task(), "linear", "statistical", runner, HindcastConfig(folds=3)
    )

    assert [len(item[1]) for item in seen] == [25, 30, 35]
    assert all(999.0 not in item[1] for item in seen)
    assert diagnostic.successful_folds == 3
    assert diagnostic.eligible
    assert diagnostic.median_mase == pytest.approx(0.0)
    assert diagnostic.recent_mase == pytest.approx(0.0)
    assert diagnostic.worst_mase == pytest.approx(0.0)
    assert diagnostic.mase_mad == pytest.approx(0.0)
    assert diagnostic.median_rmsse == pytest.approx(0.0)


def test_hindcast_handles_failures_constants_and_short_histories():
    calls = 0

    def flaky(name, history, horizon, frequency):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return (history[-1],) * horizon

    constant = _task((4.0,) * 20, 3)
    diagnostic = diagnose_candidate(
        constant, "constant", "statistical", flaky, HindcastConfig(folds=3)
    )
    assert diagnostic.successful_folds == 2
    assert diagnostic.eligible
    assert math.isfinite(diagnostic.median_mase)
    assert diagnostic.folds[0].status == "failed"

    short = diagnose_candidate(
        _task((1.0, 2.0, 3.0), 2),
        "short",
        "statistical",
        lambda *args: (3.0, 3.0),
        HindcastConfig(folds=3),
    )
    assert not short.eligible
    assert short.reason_code == "insufficient_history"


def test_pairwise_diversity_is_scale_normalized():
    left = ((1.0, 2.0), (2.0, 3.0))
    same = ((1.0, 2.0), (2.0, 3.0))
    other = ((3.0, 0.0), (4.0, 1.0))
    assert pairwise_diversity(left, same) == pytest.approx(0.0)
    assert pairwise_diversity(left, other) > 0.0


def _diagnostic(name, *, median, recent=None, worst=None, mad=0.1, family="statistical", forecasts=None, truths=None):
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=median,
        recent_mase=median if recent is None else recent,
        worst_mase=median if worst is None else worst,
        mase_mad=mad,
        fold_forecasts=forecasts or ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
        fold_truths=truths or ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
    )


def test_selector_rejects_inactive_failed_and_catastrophic_candidates():
    diagnostics = {
        "good": _diagnostic("good", median=0.8),
        "inactive": _diagnostic("inactive", median=0.1),
        "failed": CandidateDiagnostics.synthetic(
            name="failed", family="statistical", median_mase=0.01, eligible=False
        ),
        "tail": _diagnostic("tail", median=0.2, worst=20.0),
    }
    forecasts = {name: (1.0, 2.0) for name in diagnostics}
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False),
        active_names=("good", "failed", "tail"),
        diagnostics=diagnostics,
        forecasts=forecasts,
    )
    assert decision.selected == ("good",)
    assert decision.mode == "single"


def test_selector_uses_pareto_then_recent_and_deterministic_name_tie_break():
    diagnostics = {
        "zeta": _diagnostic("zeta", median=1.0, recent=0.8, worst=1.2),
        "alpha": _diagnostic("alpha", median=1.0, recent=0.8, worst=1.2),
        "dominated": _diagnostic("dominated", median=1.2, recent=1.4, worst=2.0),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=False),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics},
    )
    assert decision.selected == ("alpha",)
    assert "pareto_front" in decision.reason_codes


def test_guarded_ensemble_requires_diversity_and_historical_improvement():
    truths = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    diagnostics = {
        "positive": _diagnostic(
            "positive", median=1.0,
            forecasts=((2.0, 2.0), (2.0, 2.0), (2.0, 2.0)), truths=truths,
        ),
        "negative": _diagnostic(
            "negative", median=1.0,
            forecasts=((-2.0, -2.0), (-2.0, -2.0), (-2.0, -2.0)), truths=truths,
        ),
    }
    decision = select_numerical_forecast(
        DecisionPolicy(
            ensemble_enabled=True,
            ensemble_min_diversity=0.1,
            ensemble_min_improvement=0.01,
        ),
        active_names=("positive", "negative"),
        diagnostics=diagnostics,
        forecasts={"positive": (2.0, 2.0), "negative": (-2.0, -2.0)},
    )
    assert decision.mode == "ensemble"
    assert decision.selected == ("negative", "positive")
    assert decision.weights == pytest.approx((0.5, 0.5))
    assert decision.forecast == pytest.approx((0.0, 0.0))

    duplicate = dict(diagnostics)
    duplicate["negative"] = _diagnostic(
        "negative", median=1.0,
        forecasts=diagnostics["positive"].fold_forecasts, truths=truths,
    )
    single = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=True, ensemble_min_diversity=0.1),
        active_names=("positive", "negative"),
        diagnostics=duplicate,
        forecasts={"positive": (2.0, 2.0), "negative": (2.0, 2.0)},
    )
    assert single.mode == "single"


def test_selector_never_returns_more_than_three_members():
    diagnostics = {
        f"m{i}": _diagnostic(f"m{i}", median=1.0 + i * 0.01)
        for i in range(5)
    }
    decision = select_numerical_forecast(
        DecisionPolicy(ensemble_enabled=True, ensemble_max_members=3),
        active_names=tuple(diagnostics),
        diagnostics=diagnostics,
        forecasts={name: (1.0, 2.0) for name in diagnostics},
    )
    assert len(decision.selected) <= 3
