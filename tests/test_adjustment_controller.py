from __future__ import annotations

import random
from types import SimpleNamespace

from evolving_loop.adjustment import project_evidence
from evolving_loop.adjustment.controller import (
    CASCADE_CONTROLLER, IDENTITY_CONTROLLER, POOLED_CONTROLLER, REGIME_CONTROLLER,
    SEED_CONTROLLERS, SEMANTIC_CONTROLLER, Controller,
    DocAdjust, NoOp, PooledSemanticAdjust, RegimeAdjust, SelectBase, SemanticAdjust,
    build_event_effect_pool, controller_to_text, crossover_controllers, mutate_controller,
    run_controller, run_controller_evolution,
)

# synthetic weekend regime: weekdays=10, weekends=6
_HTS = tuple(f"2024-06-{d:02d}T00:00:00" for d in range(3, 17))
_HV = tuple(10.0 if __import__("datetime").date(2024, 6, d).weekday() < 5 else 6.0
            for d in range(3, 17))
_FTS = ("2024-06-17T00:00:00", "2024-06-18T00:00:00")   # Mon, Tue


def _effects(**kw):
    base = dict(direction="down", magnitude_kind="explicit", magnitude_value=0.2,
                start_timestamp="2024-06-17T00:00:00", end_timestamp="2024-06-17T00:00:00",
                stance="challenges", numeric_eligible=False, entity_match=False,
                target_match=False, citations=(SimpleNamespace(document_id="d", exact_quote="q"),))
    base.update(kw)
    return project_evidence(SimpleNamespace(chains=(SimpleNamespace(**base),)))


CANDS = {"toto_2_0": (100.0, 100.0)}


def test_identity_controller_returns_base():
    out, trace = run_controller(IDENTITY_CONTROLLER, CANDS, _effects(), _HV, _HTS, _FTS)
    assert out == (100.0, 100.0)
    assert trace and trace[0].startswith("select")


def test_regime_controller_scales_in_window_only_and_bounded():
    out, trace = run_controller(REGIME_CONTROLLER, CANDS, _effects(), _HV, _HTS, _FTS)
    assert out[0] == 100.0 * 0.6      # weekend regime applied in the effect window
    assert out[1] == 100.0            # outside the window untouched
    assert any("regime_adjust" in t for t in trace)
    assert abs(out[0] - 100.0) <= 0.5 * 100.0 + 1e-9    # kernel bound


def test_ungrounded_effect_is_never_adjusted():
    ungrounded = _effects(citations=())
    out, _ = run_controller(REGIME_CONTROLLER, CANDS, ungrounded, _HV, _HTS, _FTS)
    assert out == (100.0, 100.0)


def test_semantic_adjust_uses_cached_ref_and_history_level():
    # ref="weekend" (cached LLM choice); weekend history level = 6, base window = 100
    # -> scale toward 6/100, clamped by kernel to 50% -> 100*(1-0.5)=50
    out, trace = run_controller(SEMANTIC_CONTROLLER, CANDS, _effects(), _HV, _HTS, _FTS,
                                semantic_ref="weekend")
    assert out[0] == 50.0                    # clamped to kernel floor (target 6 is far below)
    assert out[1] == 100.0                   # outside the effect window
    assert any("semantic[weekend]" in t for t in trace)
    # no cached ref -> no-op
    out2, _ = run_controller(SEMANTIC_CONTROLLER, CANDS, _effects(), _HV, _HTS, _FTS, semantic_ref="")
    assert out2 == (100.0, 100.0)


def test_pooled_semantic_uses_cross_task_prior():
    # pool the weekend regime across two synthetic tasks (weekend ~0.68 of overall)
    pool = build_event_effect_pool([(_HV, _HTS), (_HV, _HTS)])
    assert "weekend" in pool and pool["weekend"][1] == 2
    out, trace = run_controller(POOLED_CONTROLLER, CANDS, _effects(), _HV, _HTS, _FTS,
                                semantic_ref="weekend", effect_pool=pool)
    assert out[0] < 100.0 and out[1] == 100.0
    assert any("pooled[weekend" in t for t in trace)


def test_pooled_semantic_falls_back_to_prior_when_local_missing():
    # a task whose OWN history has no weekends still gets a magnitude from the pool
    weekday_only_hts = tuple(f"2024-06-{d:02d}T00:00:00" for d in (3, 4, 5, 6, 7, 10, 11))  # Mon-Fri
    weekday_only_hv = tuple(10.0 for _ in weekday_only_hts)
    pool = build_event_effect_pool([(_HV, _HTS)])          # prior: weekend ~0.68
    out, trace = run_controller(POOLED_CONTROLLER, CANDS, _effects(),
                                weekday_only_hv, weekday_only_hts, _FTS,
                                semantic_ref="weekend", effect_pool=pool)
    assert out[0] < 100.0                                   # borrowed the weekend prior
    assert any("pooled[weekend" in t for t in trace)


def test_order_of_instructions_matters_and_is_free():
    # a controller that does regime then a conflicting doc-adjust; both are just steps
    c = Controller(steps=(SelectBase(), RegimeAdjust("weekend_weekday"), NoOp()))
    out, trace = run_controller(c, CANDS, _effects(), _HV, _HTS, _FTS)
    assert out[0] == 60.0 and "noop" in trace


def test_kernel_bounds_even_an_extreme_controller():
    # a doc adjust with a huge magnitude still can't exceed the kernel envelope
    big = _effects(magnitude_value=9.9)
    c = Controller(steps=(SelectBase(), DocAdjust(cap=0.5)))
    out, _ = run_controller(c, CANDS, big, _HV, _HTS, _FTS)
    assert abs(out[0] - 100.0) <= 0.5 * 100.0 + 1e-9


def test_mutation_and_crossover_stay_legal_and_runnable():
    rng = random.Random(0)
    c = CASCADE_CONTROLLER
    for _ in range(200):
        c = mutate_controller(c, rng)
        assert isinstance(c, Controller) and len(c.steps) <= 8
        out, _ = run_controller(c, CANDS, _effects(), _HV, _HTS, _FTS)
        for v in out:                                    # always finite + bounded
            assert abs(v - 100.0) <= 0.5 * 100.0 + 1e-9
    d = crossover_controllers(CASCADE_CONTROLLER, c, rng)
    assert isinstance(d, Controller) and 1 <= len(d.steps) <= 8


def test_controller_to_text_is_readable():
    txt = controller_to_text(REGIME_CONTROLLER)
    assert "SelectBase" in txt and "RegimeAdjust" in txt


def test_evolution_optimizes_a_toy_objective():
    # reward controllers that end up shorter (a stand-in scalar fitness)
    fit = lambda c: -len(c.steps)
    best, score, hist = run_controller_evolution(SEED_CONTROLLERS, fit, generations=20, pop_size=24)
    assert 1 <= len(best.steps) <= 2               # search shrank toward fewer steps
    assert hist[-1][1] >= hist[0][1]               # non-decreasing best fitness
    # deterministic given the seed
    b2, s2, _ = run_controller_evolution(SEED_CONTROLLERS, fit, generations=20, pop_size=24)
    assert s2 == score
