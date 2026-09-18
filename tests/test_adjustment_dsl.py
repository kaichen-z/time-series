from __future__ import annotations

from types import SimpleNamespace

from evolving_loop.adjustment import project_evidence
from evolving_loop.adjustment.dsl import (
    EVENT_SCALE,
    EVENT_SCALE_RELAXED_ENTITY,
    GROUNDED_EVENT,
    IDENTITY,
    Action,
    Policy,
    Predicate,
    Rule,
    apply_policy,
    policy_to_text,
)

TS = ("2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00")


def _chain(**kw):
    base = dict(
        direction="increase", magnitude_kind="relative", magnitude_value=0.2,
        start_timestamp="2026-01-02T00:00:00", end_timestamp="2026-01-02T00:00:00",
        stance="support", numeric_eligible=True, entity_match=True, target_match=True,
        citations=(SimpleNamespace(document_id="d1", exact_quote="q"),),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _effects(**kw):
    return project_evidence(SimpleNamespace(chains=(_chain(**kw),)))


def test_identity_policy_is_byte_identical_to_base():
    base = [10.0, 20.0, 30.0]
    out, fired = apply_policy(IDENTITY, base, _effects(), TS)
    assert out == tuple(base)
    assert fired == []


def test_event_rule_fires_only_in_window_and_scales():
    base = [10.0, 20.0, 30.0]
    out, fired = apply_policy(EVENT_SCALE, base, _effects(), TS)
    assert out[0] == 10.0 and out[2] == 30.0        # outside window untouched
    assert out[1] == 20.0 * 1.2                       # +20% inside window
    assert len(fired) == 1


def test_kernel_bounds_even_a_huge_magnitude():
    base = [10.0, 20.0, 30.0]
    out, _ = apply_policy(EVENT_SCALE, base, _effects(magnitude_value=9.9), TS)
    # per-rule cap 0.3 AND kernel envelope: move stays well within 50% of base
    assert abs(out[1] - base[1]) <= 0.5 * base[1] + 1e-9


def test_ungrounded_effect_never_fires():
    base = [10.0, 20.0, 30.0]
    ungrounded = _effects(citations=())
    assert not ungrounded[0].grounded
    out, fired = apply_policy(EVENT_SCALE, base, ungrounded, TS)
    assert out == tuple(base) and fired == []


def test_strict_entity_gate_drops_effect_but_relaxed_policy_fires():
    base = [10.0, 20.0, 30.0]
    proxy = _effects(entity_match=False, numeric_eligible=False)  # the task_152 shape
    strict, sfired = apply_policy(EVENT_SCALE, base, proxy, TS)
    assert strict == tuple(base) and sfired == []                 # strict discards it
    relaxed, rfired = apply_policy(EVENT_SCALE_RELAXED_ENTITY, base, proxy, TS)
    assert relaxed[1] == 20.0 * 1.2 and len(rfired) == 1          # relaxed recovers it


def test_direction_gate_and_decrease():
    base = [10.0, 20.0, 30.0]
    out, _ = apply_policy(EVENT_SCALE, base, _effects(direction="decrease"), TS)
    assert out[1] == 20.0 * 0.8
    # a rule restricted to increases does not fire on a decrease
    up_only = Policy(name="up", rules=(Rule(Predicate(directions=("increase",)), Action()),))
    out2, fired2 = apply_policy(up_only, base, _effects(direction="decrease"), TS)
    assert out2 == tuple(base) and fired2 == []


def test_overlap_anomaly_gate():
    base = [10.0, 20.0, 30.0]
    coupled = Policy(name="coupled", rules=(
        Rule(Predicate(require_overlap_anomaly=True), Action()),
    ))
    # no anomaly overlapping the effect window -> does not fire
    out, fired = apply_policy(coupled, base, _effects(), TS)
    assert out == tuple(base) and fired == []
    # anomaly overlapping the window -> fires
    out2, fired2 = apply_policy(
        coupled, base, _effects(), TS,
        anomaly_windows=(("2026-01-02T00:00:00", "2026-01-02T00:00:00"),),
    )
    assert out2[1] == 20.0 * 1.2 and len(fired2) == 1


def test_grounded_event_recovers_real_task152_shape():
    # The real cached task_152 effect: a grounded, quantified, windowed, directional
    # holiday effect that retrieval marked entity=target=numeric=False. Direction
    # comes in as retrieval's raw "down" and must canonicalize to a decrease.
    ts = ("2024-07-04T00:00:00", "2024-07-05T00:00:00", "2024-07-06T00:00:00")
    base = [100.0, 100.0, 100.0]
    eff = _effects(
        direction="down", magnitude_kind="explicit", magnitude_value=11.2,
        start_timestamp="2024-07-04T00:00:00", end_timestamp="2024-07-04T23:00:00",
        stance="challenges", numeric_eligible=False, entity_match=False, target_match=False,
    )
    assert eff[0].direction == "decrease" and eff[0].grounded
    # strict + entity-only-relaxed both discard it (target_match still required)
    assert apply_policy(EVENT_SCALE, base, eff, ts)[1] == []
    assert apply_policy(EVENT_SCALE_RELAXED_ENTITY, base, eff, ts)[1] == []
    # the fully relaxed grounded_event seed recovers it, moving only the in-window step
    out, fired = apply_policy(GROUNDED_EVENT, base, eff, ts)
    assert len(fired) == 1
    assert out[0] == 100.0 * (1.0 - 0.3)   # -30% cap applied in-window (mag 11.2 capped)
    assert out[1] == 100.0 and out[2] == 100.0


def test_policy_to_text_is_readable():
    assert "identity" in policy_to_text(IDENTITY)
    txt = policy_to_text(EVENT_SCALE)
    assert "grounded" in txt and "scale" in txt and "rule1" in txt
