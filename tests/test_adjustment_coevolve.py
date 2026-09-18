from __future__ import annotations

import random
from types import SimpleNamespace

from evolving_loop.adjustment import project_evidence
from evolving_loop.adjustment.coevolve import (
    FocusPolicy, IntegratePolicy, Interaction, QualifyPolicy,
    crossover_interaction, interaction_to_text, mutate_interaction,
    run_coevolution, run_pipeline,
)

# two synthetic weeks, daily: weekdays=10, weekends=6 (a document-explained regime)
_HTS = tuple(f"2024-06-{d:02d}T00:00:00" for d in range(3, 17))
_HV = tuple(10.0 if __import__("datetime").date(2024, 6, d).weekday() < 5 else 6.0
            for d in range(3, 17))
_FTS = ("2024-06-17T00:00:00", "2024-06-18T00:00:00")   # Mon, Tue


def _effect(**kw):
    base = dict(direction="down", magnitude_kind="explicit", magnitude_value=0.2,
                start_timestamp="2024-06-17T00:00:00", end_timestamp="2024-06-17T00:00:00",
                stance="challenges", numeric_eligible=False, entity_match=False,
                target_match=False, citations=(SimpleNamespace(document_id="d", exact_quote="q"),))
    base.update(kw)
    return project_evidence(SimpleNamespace(chains=(SimpleNamespace(**base),)))


def test_focus_detects_weekend_regime():
    regimes = FocusPolicy().detect(_HV, _HTS, _FTS)
    kinds = {r.kind: r for r in regimes}
    assert "weekend_weekday" in kinds
    assert kinds["weekend_weekday"].direction == "decrease"


def test_pipeline_calibrates_and_stays_bounded():
    base = [100.0, 100.0]
    out, state = run_pipeline(Interaction(), base, _effect(), _FTS, _HV, _HTS)
    assert out[0] == 100.0 * 0.6         # calibrated to the weekend regime
    assert out[1] == 100.0               # outside window untouched
    assert len(state.fired) == 1 and state.qualified[0].source.startswith("calibrated")
    assert abs(out[0] - base[0]) <= 0.5 * base[0] + 1e-9   # kernel bound


def test_direction_conflict_is_dropped_then_doc_fallback():
    base = [100.0, 100.0]
    # doc says increase but the only regime is a decrease -> calibration rejected,
    # cascade falls back to the document magnitude (which is +20%, capped)
    out, state = run_pipeline(Interaction(), base, _effect(direction="up"), _FTS, _HV, _HTS)
    assert state.qualified and state.qualified[0].source == "doc"
    assert out[0] == 100.0 * 1.2


def test_ungrounded_never_fires():
    base = [100.0, 100.0]
    eff = _effect(citations=())
    out, state = run_pipeline(Interaction(), base, eff, _FTS, _HV, _HTS)
    assert out == tuple(base) and state.fired == ()


def test_regime_fills_direction_for_unknown_grounded_event():
    # retrieval turned conservative: grounded, windowed, but direction=unknown.
    # the weekend regime (decrease, strength ~0.37) should supply direction + magnitude.
    base = [100.0, 100.0]
    eff = _effect(direction="unknown", magnitude_value=None)
    out, state = run_pipeline(Interaction(), base, eff, _FTS, _HV, _HTS)
    assert len(state.fired) == 1
    q = state.qualified[0]
    assert q.source.startswith("calibrated_filled") and q.direction == "decrease"
    assert out[0] == 100.0 * 0.6 and out[1] == 100.0

    # with filling disabled, the same unknown event is dropped (no doc magnitude either)
    from evolving_loop.adjustment.coevolve import QualifyPolicy
    it2 = Interaction(qualify=QualifyPolicy(fill_direction_from_regime=False))
    out2, state2 = run_pipeline(it2, base, eff, _FTS, _HV, _HTS)
    assert out2 == tuple(base) and state2.fired == ()


def test_mutation_and_crossover_are_legal():
    rng = random.Random(0)
    x = Interaction()
    for _ in range(100):
        x = mutate_interaction(x, rng)
        assert isinstance(x, Interaction)
    y = crossover_interaction(Interaction(), x, rng)
    assert isinstance(y, Interaction)
    assert "Focus:" in interaction_to_text(y)


def test_coevolution_optimizes_a_toy_objective():
    # reward integrate.max_frac near 0.25
    fit = lambda it: -abs(it.integrate.max_frac - 0.25)
    best, score, hist = run_coevolution([Interaction()], fit, generations=40, pop_size=30)
    assert abs(best.integrate.max_frac - 0.25) < 0.06
    assert hist[-1][1] >= hist[0][1]
