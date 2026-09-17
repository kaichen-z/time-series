from __future__ import annotations

from types import SimpleNamespace

from evolving_loop.adjustment import (
    EvidenceEffect,
    apply_bounded_delta,
    horizon_window_mask,
    identity_adjust,
    project_evidence,
    reference_future_event_adjust,
)
from evolving_loop.adjustment.post_adjust import run_adjuster


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


def test_identity_seed_is_byte_identical_to_base():
    base = [10.0, 20.0, 30.0]
    effects = project_evidence(SimpleNamespace(chains=(_chain(),)))
    assert identity_adjust(base, effects, TS) == tuple(base)


def test_bounded_delta_clamps_and_rejects_nonfinite():
    base = [100.0, 100.0, 100.0]
    proposed = [200.0, float("nan"), 50.0]
    out = apply_bounded_delta(base, proposed, max_frac=0.5)
    assert out[0] == 150.0          # +100 clamped to +50
    assert out[1] == 100.0          # NaN -> base
    assert out[2] == 50.0           # within bound, unchanged


def test_window_mask_selects_only_in_range_steps():
    mask = horizon_window_mask(TS, "2026-01-02T00:00:00", "2026-01-02T00:00:00")
    assert mask == (False, True, False)


def test_ungrounded_evidence_cannot_drive_a_delta():
    base = [10.0, 20.0, 30.0]
    ungrounded = _chain(citations=())  # no citation -> not grounded
    effects = project_evidence(SimpleNamespace(chains=(ungrounded,)))
    assert not effects[0].grounded and not effects[0].actionable
    assert reference_future_event_adjust(base, effects, TS) == tuple(base)


def test_actionable_future_event_moves_only_windowed_step_within_bound():
    base = [10.0, 20.0, 30.0]
    effects = project_evidence(SimpleNamespace(chains=(_chain(),)))
    out = reference_future_event_adjust(base, effects, TS, max_frac=0.5, relative_cap=0.25)
    assert out[0] == 10.0 and out[2] == 30.0        # outside window unchanged
    assert out[1] == 20.0 * 1.2                      # +20% inside window
    # even a huge magnitude stays within the safety envelope
    big = project_evidence(SimpleNamespace(chains=(_chain(magnitude_value=9.9),)))
    capped = reference_future_event_adjust(base, big, TS, max_frac=0.5, relative_cap=0.25)
    assert abs(capped[1] - base[1]) <= 0.5 * base[1] + 1e-9


def test_run_adjuster_enforces_floor_on_crashing_adjuster():
    base = [1.0, 2.0, 3.0]
    def bad(_b, _e, _t):
        raise RuntimeError("boom")
    assert run_adjuster(bad, base, (), TS) == tuple(base)
