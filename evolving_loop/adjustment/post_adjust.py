"""Primitives + seed adjusters for the evolvable ``post_adjust`` stage.

Contract: ``adjust(base_forecast, effects, future_timestamps) -> forecast`` where

* ``base_forecast`` is the strong base (Toto / the selected numerical candidate),
* ``effects`` is the grounded, projected retrieval evidence (see ``EvidenceEffect``),
* ``future_timestamps`` are ISO timestamps of the forecast horizon steps.

Safety is enforced by the *harness*, not the evolved code: every adjuster's output
is passed through :func:`apply_bounded_delta`, so the final forecast can never move
more than ``max_frac`` of ``|base|`` away from the base, and NaN/inf collapse back
to the base. The base is therefore a hard floor — the stage can never be much worse
than the base model. Only chains that cite a document (``grounded``) may drive a
delta, which blocks ungrounded/hallucinated adjustments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class EvidenceEffect:
    """A sanitized, grounded projection of one retrieval ``EvidenceChain``."""

    direction: str            # e.g. "increase" / "decrease" / "none" / "unknown"
    magnitude_kind: str       # e.g. "absolute" / "relative" / "none" / "unknown"
    magnitude_value: float | None
    start_timestamp: str | None
    end_timestamp: str | None
    stance: str
    numeric_eligible: bool
    grounded: bool            # cites >= 1 document
    entity_match: bool
    target_match: bool

    @property
    def actionable(self) -> bool:
        """Only tight, grounded, quantified future effects may move the numbers."""
        return (
            self.grounded
            and self.numeric_eligible
            and self.entity_match
            and self.target_match
            and self.direction in {"increase", "decrease"}
            and self.magnitude_value is not None
            and math.isfinite(self.magnitude_value)
            and self.start_timestamp is not None
            and self.end_timestamp is not None
        )


# Retrieval's direction vocabulary (schemas.py `_DIRECTIONS`) is up/down/stable/
# unknown; the adjustment layer's canonical vocabulary is increase/decrease. Map at
# the projection boundary so ``.actionable`` and the DSL see one vocabulary. Without
# this, no real evidence ever fires (the direction gate matched 0/15 chains in the
# headroom probe) — including task_152's genuine "down" holiday signal.
_DIRECTION_CANON = {
    "up": "increase", "increase": "increase",
    "down": "decrease", "decrease": "decrease",
    "stable": "stable", "unknown": "unknown", "none": "none",
}


def _canon_direction(raw: object) -> str:
    key = str(raw).strip().lower()
    return _DIRECTION_CANON.get(key, key)


def project_evidence(card: object) -> tuple[EvidenceEffect, ...]:
    """Project a ``FinalRetrievalCard`` (or any object exposing ``.chains``).

    Read-only and duck-typed so tests can pass lightweight stand-ins.
    """
    chains = tuple(getattr(card, "chains", ()) or ())
    effects: list[EvidenceEffect] = []
    for chain in chains:
        citations = tuple(getattr(chain, "citations", ()) or ())
        effects.append(
            EvidenceEffect(
                direction=_canon_direction(getattr(chain, "direction", "unknown")),
                magnitude_kind=str(getattr(chain, "magnitude_kind", "unknown")),
                magnitude_value=getattr(chain, "magnitude_value", None),
                start_timestamp=getattr(chain, "start_timestamp", None),
                end_timestamp=getattr(chain, "end_timestamp", None),
                stance=str(getattr(chain, "stance", "")),
                numeric_eligible=bool(getattr(chain, "numeric_eligible", False)),
                grounded=len(citations) > 0,
                entity_match=bool(getattr(chain, "entity_match", False)),
                target_match=bool(getattr(chain, "target_match", False)),
            )
        )
    return tuple(effects)


def _parse(ts: str | None) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # normalize to tz-naive so mixed aware/naive timestamps never crash comparisons
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def horizon_window_mask(
    future_timestamps: Sequence[str], start: str | None, end: str | None
) -> tuple[bool, ...]:
    """Which horizon steps fall inside the inclusive ``[start, end]`` window."""
    s, e = _parse(start), _parse(end)
    if s is None or e is None:
        return tuple(False for _ in future_timestamps)
    mask = []
    for ts in future_timestamps:
        t = _parse(ts)
        mask.append(t is not None and s <= t <= e)
    return tuple(mask)


def apply_bounded_delta(
    base: Sequence[float], proposed: Sequence[float], *, max_frac: float = 0.5
) -> tuple[float, ...]:
    """Clamp ``proposed`` so each step is within ``max_frac * |base|`` of ``base``.

    Non-finite proposed values collapse to the base. Guarantees the base floor.
    """
    if len(proposed) != len(base):
        return tuple(float(b) for b in base)
    out: list[float] = []
    for b, p in zip(base, proposed):
        b = float(b)
        if not math.isfinite(p):
            out.append(b)
            continue
        bound = max_frac * abs(b)
        delta = max(-bound, min(bound, float(p) - b))
        out.append(b + delta)
    return tuple(out)


# ---- seed / template adjusters (the evolution starts from these) ----

def identity_adjust(
    base_forecast: Sequence[float],
    effects: Sequence[EvidenceEffect],
    future_timestamps: Sequence[str],
) -> tuple[float, ...]:
    """Seed: never move off the base (byte-identical to the base model)."""
    return tuple(float(b) for b in base_forecast)


def reference_future_event_adjust(
    base_forecast: Sequence[float],
    effects: Sequence[EvidenceEffect],
    future_timestamps: Sequence[str],
    *,
    max_frac: float = 0.5,
    relative_cap: float = 0.25,
) -> tuple[float, ...]:
    """Template: nudge steps inside an actionable future-event window.

    Conservative reference (a starting point for evolution, not the final policy):
    for each *actionable* effect, apply a relative bump (sign from ``direction``,
    size = ``min(|magnitude|, relative_cap)`` when relative, else a small fixed
    nudge) to the horizon steps inside its ``[start, end]`` window. Everything is
    clamped by :func:`apply_bounded_delta`, so it is always safe.
    """
    base = [float(b) for b in base_forecast]
    proposed = list(base)
    for eff in effects:
        if not eff.actionable:
            continue
        mask = horizon_window_mask(future_timestamps, eff.start_timestamp, eff.end_timestamp)
        if not any(mask):
            continue
        sign = 1.0 if eff.direction == "increase" else -1.0
        mag = abs(float(eff.magnitude_value))  # type: ignore[arg-type]
        frac = min(mag, relative_cap) if eff.magnitude_kind == "relative" else min(0.05, relative_cap)
        for i, inside in enumerate(mask):
            if inside:
                proposed[i] = base[i] * (1.0 + sign * frac)
    return apply_bounded_delta(base, proposed, max_frac=max_frac)


def run_adjuster(
    adjuster: Callable[..., Sequence[float]],
    base_forecast: Sequence[float],
    effects: Sequence[EvidenceEffect],
    future_timestamps: Sequence[str],
    *,
    max_frac: float = 0.5,
) -> tuple[float, ...]:
    """Run any adjuster and enforce the base-floor safety envelope on its output."""
    try:
        proposed = adjuster(base_forecast, effects, future_timestamps)
    except Exception:
        return tuple(float(b) for b in base_forecast)
    return apply_bounded_delta(base_forecast, proposed, max_frac=max_frac)
