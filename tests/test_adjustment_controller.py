from __future__ import annotations

import random
from types import SimpleNamespace

from evolving_loop.adjustment import project_evidence
from evolving_loop.adjustment.controller import (
    CASCADE_CONTROLLER, CORDP_CONTROLLER, IDENTITY_CONTROLLER, POOLED_CONTROLLER,
    REGIME_CONTROLLER, SEED_CONTROLLERS, SEMANTIC_CONTROLLER, Controller,
    CallAgent, CorDPAdjust, DocAdjust, MenuAdjust, MethodBlend, NoOp, PooledSemanticAdjust, RegimeAdjust, ResidualAdjust,
    ResidualSpec, RESIDUAL_CONTROLLER, SelectBase, SemanticAdjust,
    build_event_effect_pool, controller_to_text, crossover_controllers, mutate_controller,
    run_controller, run_controller_evolution,
)
from evolving_loop.adjustment.dsl import KERNEL_MAX_FRAC

_FTS5 = tuple(f"2024-06-{d:02d}T00:00:00" for d in range(17, 22))   # 5 forecast steps
_CANDS5 = {"toto_2_0": (100.0,) * 5}

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


def test_significance_gate_blocks_unverifiable_regime():
    # a task whose OWN history has no weekends can't verify the weekend regime -> no-op
    weekday_only_hts = tuple(f"2024-06-{d:02d}T00:00:00" for d in (3, 4, 5, 6, 7, 10, 11))  # Mon-Fri
    weekday_only_hv = tuple(10.0 for _ in weekday_only_hts)
    out, trace = run_controller(POOLED_CONTROLLER, CANDS, _effects(),
                                weekday_only_hv, weekday_only_hts, _FTS, semantic_ref="weekend")
    assert out == (100.0, 100.0)                            # significance gate blocks it
    assert any("insignificant" in t for t in trace)


def test_pooled_falls_back_to_prior_when_gate_off():
    # with the significance gate off, the pool prior supplies the magnitude
    weekday_only_hts = tuple(f"2024-06-{d:02d}T00:00:00" for d in (3, 4, 5, 6, 7, 10, 11))
    weekday_only_hv = tuple(10.0 for _ in weekday_only_hts)
    pool = build_event_effect_pool([(_HV, _HTS)])          # prior: weekend ~0.68
    c = Controller(steps=(SelectBase(),
                          PooledSemanticAdjust(trusted=("weekend",), require_significant=False)))
    out, trace = run_controller(c, CANDS, _effects(), weekday_only_hv, weekday_only_hts, _FTS,
                                semantic_ref="weekend", effect_pool=pool)
    assert out[0] < 100.0 and any("pooled[weekend" in t for t in trace)


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


def test_evolution_held_out_selection_ignores_search_overfit():
    # SEARCH fitness rewards LONG controllers (overfit signal); the held-out VAL
    # fitness rewards SHORT ones. The returned champion must follow VAL, not SEARCH.
    search_fit = lambda c: len(c.steps)
    val_fit = lambda c: -len(c.steps)
    best_h, bf_h, hist = run_controller_evolution(
        SEED_CONTROLLERS, search_fit, generations=15, pop_size=24, elite=6,
        seed=7, select_fitness=val_fit)
    best_s, _, _ = run_controller_evolution(
        SEED_CONTROLLERS, search_fit, generations=15, pop_size=24, elite=6, seed=7)
    assert len(best_h.steps) <= len(best_s.steps)   # held-out avoids the long overfit
    assert val_fit(best_h) >= val_fit(best_s)       # champion is >= on held-out fitness
    assert bf_h == val_fit(best_h)                  # returned score is the SELECTION score
    assert len(hist[0]) == 3                         # (gen, search-fit, select-fit)


def _cordp_run(ctrl, conf, corrs):
    return run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5,
                          cordp_conf=conf, cordp_corrections=corrs)[0]


def test_cordp_abstains_below_confidence():
    ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=0.75)))
    out = _cordp_run(ctrl, 0.60, ((_FTS5[0], _FTS5[0], 1.5),))   # conf below floor
    assert out == (100.0,) * 5                                   # untouched -> base


def test_cordp_applies_localized_and_bounded():
    ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=0.7, wfrac_max=0.3)))
    out = _cordp_run(ctrl, 0.9, ((_FTS5[1], _FTS5[1], 1.6),))    # one step (wfrac 0.2)
    assert out[0] == 100.0 and out[2] == 100.0                   # outside window unchanged
    assert out[1] > 100.0                                        # raised in window
    assert out[1] <= 100.0 * (1.0 + KERNEL_MAX_FRAC) + 1e-9      # kernel still bounds it


def test_cordp_rejects_global_rescale():
    ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=0.7, wfrac_max=0.3)))
    out = _cordp_run(ctrl, 0.9, ((_FTS5[0], _FTS5[4], 1.6),))    # whole horizon (wfrac 1.0)
    assert out == (100.0,) * 5                                   # rejected -> base


def test_cordp_controller_in_seed_set():
    assert CORDP_CONTROLLER in SEED_CONTROLLERS
    txt = controller_to_text(CORDP_CONTROLLER)
    assert "CorDPAdjust" in txt


def _resid_run(ctrl, conf, specs):
    return run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5,
                          residual_conf=conf, residual_specs=specs)[0]


def test_residual_level_shape_adds_flat_in_window():
    ctrl = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=0.7, wfrac_max=0.8)))
    spec = ResidualSpec(start=_FTS5[1], end=_FTS5[3], shape="level", amplitude=0.2, grounded=True)
    out = _resid_run(ctrl, 0.9, (spec,))
    assert out[0] == 100.0 and out[4] == 100.0                 # outside window untouched
    assert abs(out[1] - 120.0) < 1e-6 and abs(out[2] - 120.0) < 1e-6   # +0.2*ref, flat


def test_residual_bump_shape_peaks_at_centre():
    ctrl = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=0.7, wfrac_max=0.8)))
    spec = ResidualSpec(start=_FTS5[1], end=_FTS5[3], shape="bump", amplitude=0.2, grounded=True)
    out = _resid_run(ctrl, 0.9, (spec,))
    assert abs(out[1] - 100.0) < 1e-6 and abs(out[3] - 100.0) < 1e-6   # zero at window edges
    assert out[2] > out[1]                                            # peak at centre


def test_residual_ungrounded_spec_is_dropped():
    ctrl = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=0.7, wfrac_max=0.8)))
    spec = ResidualSpec(start=_FTS5[1], end=_FTS5[2], shape="level", amplitude=0.4, grounded=False)
    out = _resid_run(ctrl, 0.9, (spec,))
    assert out == (100.0,) * 5                                 # no source for the number -> abstain


def test_residual_confidence_and_wfrac_gates():
    ctrl = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=0.8, wfrac_max=0.3)))
    spec = ResidualSpec(start=_FTS5[1], end=_FTS5[1], shape="level", amplitude=0.2, grounded=True)
    assert _resid_run(ctrl, 0.6, (spec,)) == (100.0,) * 5      # below confidence -> abstain
    globalspec = ResidualSpec(start=_FTS5[0], end=_FTS5[4], shape="level", amplitude=0.2, grounded=True)
    assert _resid_run(ctrl, 0.9, (globalspec,)) == (100.0,) * 5   # whole horizon -> rejected


def test_residual_kernel_bounds_even_large_amplitude():
    ctrl = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=0.7, wfrac_max=0.8, amp_cap=0.9)))
    spec = ResidualSpec(start=_FTS5[2], end=_FTS5[2], shape="level", amplitude=5.0, grounded=True)
    out = _resid_run(ctrl, 0.9, (spec,))
    assert abs(out[2] - 100.0) <= 0.5 * 100.0 + 1e-9           # invariant kernel still bounds it


def test_residual_controller_in_seed_set():
    assert RESIDUAL_CONTROLLER in SEED_CONTROLLERS
    assert "ResidualAdjust" in controller_to_text(RESIDUAL_CONTROLLER)


_CANDS2 = {"toto_2_0": (100.0,) * 5, "seasonal_naive": (60.0,) * 5}


def test_method_blend_only_when_signal_present():
    ctrl = Controller(steps=(SelectBase(), MethodBlend(other="seasonal_naive", weight=0.5, conf_min=0.7)))
    # no document signal -> leaves the strong base alone
    out0 = run_controller(ctrl, _CANDS2, (), _HV, _HTS, _FTS5)[0]
    assert out0 == (100.0,) * 5
    # document signal present (cordp_conf high) -> blends toward the other candidate, kernel-bounded
    out1 = run_controller(ctrl, _CANDS2, (), _HV, _HTS, _FTS5, cordp_conf=0.9)[0]
    assert all(v < 100.0 for v in out1)                       # moved toward 60
    assert all(abs(v - 100.0) <= 0.5 * 100.0 + 1e-9 for v in out1)   # kernel bound


def test_method_blend_noop_without_candidate():
    ctrl = Controller(steps=(SelectBase(), MethodBlend(other="not_cached", weight=0.5, conf_min=0.5)))
    out = run_controller(ctrl, _CANDS2, (), _HV, _HTS, _FTS5, cordp_conf=0.9)[0]
    assert out == (100.0,) * 5                                # missing candidate -> no-op


from evolving_loop.adjustment.math_menu import build_math_menu

_MENU = build_math_menu(_HV, _HTS, _FTS5)   # regime:weekend=6, weekday=10, trend, decay


def test_menu_adjust_does_math_from_the_menu():
    # numerical menu supplies weekend level (6); document (semantic_ref) selects it;
    # decision applies it to the event window, kernel-bounded (100 -> toward 6 -> clamped 50)
    ctrl = Controller(steps=(SelectBase(), MenuAdjust(prefer=("regime",))))
    corrs = ((_FTS5[1], _FTS5[1], 0.5),)     # an event window; mult unused by MenuAdjust
    out, trace = run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5,
                                cordp_corrections=corrs, menu=_MENU, semantic_ref="weekend")
    assert out[0] == 100.0 and out[2] == 100.0            # outside window untouched
    assert out[1] == 50.0                                 # moved toward weekend level, kernel-clamped
    assert any("menu[regime:weekend]" in t for t in trace)


def test_menu_adjust_noop_without_menu():
    ctrl = Controller(steps=(SelectBase(), MenuAdjust()))
    corrs = ((_FTS5[1], _FTS5[1], 0.5),)
    out, _ = run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5, cordp_corrections=corrs, menu=())
    assert out == (100.0,) * 5                            # empty menu -> no-op


def test_call_agent_logs_call_and_is_safe_without_tools():
    ctrl = Controller(steps=(SelectBase(), CallAgent(tool="numerical", arg="refine"),
                             CallAgent(tool="retrieval")))
    out, trace = run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5)
    assert out == (100.0,) * 5                            # stub calls don't change the forecast
    assert sum(1 for t in trace if t.startswith("call:")) == 2   # both calls logged (efficiency cost)


def test_call_agent_invokes_a_registered_tool():
    from dataclasses import replace as _rep
    def bump(state, arg):                                 # a stub numerical tool that adds a candidate
        return _rep(state, forecast=tuple(v + 1.0 for v in state.forecast))
    ctrl = Controller(steps=(SelectBase(), CallAgent(tool="numerical")))
    out, _ = run_controller(ctrl, _CANDS5, (), _HV, _HTS, _FTS5, tools={"numerical": bump})
    assert all(abs(v - 101.0) <= 0.5 * 100.0 + 1e-9 for v in out)   # tool ran, kernel still bounds
