"""Level-0 interaction DSL: rules-as-data for the text->forecast adjustment policy.

A policy is a small, typed, ordered rule table (not arbitrary code), so it is:
  * evolvable  — mutate predicate/action fields;
  * auditable  — `policy_to_text` prints it as English; it is finite and readable;
  * safe by construction — the *kernel* (not the rules) enforces the invariants:
        (1) only GROUNDED effects may ever fire (ungrounded → ignored),
        (2) the final output is passed through the bounded envelope so no rule,
            however evolved, can move the forecast more than `max_frac*|base|`.

Each rule's predicate may *relax* the qualification requirements (e.g. drop the
over-strict literal entity_match that discarded the task_152 holiday effect) — but
such a relaxation only survives evolution if it improves the held-out fitness while
staying grounded. So "how conservative to be" is decided by data, not by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

from .post_adjust import EvidenceEffect, apply_bounded_delta, horizon_window_mask

KERNEL_MAX_FRAC = 0.5  # hard ceiling the kernel always enforces, regardless of policy


@dataclass(frozen=True)
class Predicate:
    """Which effects this rule fires on. Requirements can be relaxed (evolved)."""
    directions: tuple[str, ...] = ("increase", "decrease")
    require_numeric_eligible: bool = True
    require_entity_match: bool = True
    require_target_match: bool = True
    require_window: bool = True
    require_magnitude: bool = True
    require_overlap_anomaly: bool = False  # data<->text coupling gate

    def matches(self, e: EvidenceEffect, anomaly_windows, future_timestamps) -> bool:
        # Kernel invariant: grounded is NON-negotiable, independent of the predicate.
        if not e.grounded:
            return False
        if self.directions and e.direction not in self.directions:
            return False
        if self.require_numeric_eligible and not e.numeric_eligible:
            return False
        if self.require_entity_match and not e.entity_match:
            return False
        if self.require_target_match and not e.target_match:
            return False
        if self.require_window and not (e.start_timestamp and e.end_timestamp):
            return False
        if self.require_magnitude and (
            e.magnitude_value is None or not math.isfinite(e.magnitude_value)
        ):
            return False
        if self.require_overlap_anomaly:
            emask = horizon_window_mask(future_timestamps, e.start_timestamp, e.end_timestamp)
            if not any(a and m for a, m in zip(_union_mask(anomaly_windows, future_timestamps), emask)):
                return False
        return True


@dataclass(frozen=True)
class Action:
    op: str = "scale"          # "scale" | "shift" | "none"
    magnitude: str = "frac_capped"  # "frac_capped" | "fixed"
    cap: float = 0.3           # per-rule cap on the fractional/relative move
    fixed: float = 0.0         # used when magnitude == "fixed"


@dataclass(frozen=True)
class Rule:
    predicate: Predicate
    action: Action


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = ()
    name: str = "policy"
    max_frac: float = KERNEL_MAX_FRAC


def _union_mask(anomaly_windows, future_timestamps) -> tuple[bool, ...]:
    mask = [False] * len(future_timestamps)
    for (start, end) in anomaly_windows:
        w = horizon_window_mask(future_timestamps, start, end)
        mask = [m or x for m, x in zip(mask, w)]
    return tuple(mask)


def _sign(direction: str) -> float:
    return 1.0 if direction == "increase" else -1.0


def apply_policy(
    policy: Policy,
    base_forecast: Sequence[float],
    effects: Sequence[EvidenceEffect],
    future_timestamps: Sequence[str],
    anomaly_windows: Sequence[tuple[str | None, str | None]] = (),
):
    """Interpret the rule table; return (forecast, fired) with kernel invariants.

    `fired` lists (effect, rule) pairs for audit/provenance.
    """
    base = [float(b) for b in base_forecast]
    proposed = list(base)
    fired: list[tuple[EvidenceEffect, Rule]] = []
    for e in effects:
        for rule in policy.rules:
            if not rule.predicate.matches(e, anomaly_windows, future_timestamps):
                continue
            mask = horizon_window_mask(future_timestamps, e.start_timestamp, e.end_timestamp)
            if not any(mask):
                continue
            if rule.action.magnitude == "frac_capped":
                mag = abs(e.magnitude_value) if e.magnitude_value is not None else 0.0
                if e.magnitude_kind != "relative":
                    mag = min(mag, rule.action.cap)  # non-relative magnitudes: treat conservatively
                frac = min(mag, rule.action.cap)
            else:
                frac = rule.action.cap  # fixed fractional move
            for i, inside in enumerate(mask):
                if not inside:
                    continue
                if rule.action.op == "scale":
                    proposed[i] = base[i] * (1.0 + _sign(e.direction) * frac)
                elif rule.action.op == "shift":
                    proposed[i] = base[i] + _sign(e.direction) * (
                        rule.action.fixed if rule.action.magnitude == "fixed" else frac * abs(base[i])
                    )
            fired.append((e, rule))
            break  # first matching rule per effect
    # KERNEL: bound the final output so no evolved rule can be catastrophic.
    out = apply_bounded_delta(base, proposed, max_frac=min(policy.max_frac, KERNEL_MAX_FRAC))
    return out, fired


def policy_to_text(policy: Policy) -> str:
    """Render the policy as readable English — the auditability guarantee."""
    if not policy.rules:
        return f"{policy.name}: identity (never modifies the base forecast)"
    lines = [f"{policy.name} (max move ±{min(policy.max_frac, KERNEL_MAX_FRAC):.0%} of base):"]
    for i, r in enumerate(policy.rules, 1):
        p, a = r.predicate, r.action
        conds = ["grounded"]
        if p.directions:
            conds.append("dir∈{" + ",".join(p.directions) + "}")
        if p.require_numeric_eligible:
            conds.append("numeric_eligible")
        if p.require_entity_match:
            conds.append("entity_match")
        if p.require_target_match:
            conds.append("target_match")
        if p.require_window:
            conds.append("has_window")
        if p.require_magnitude:
            conds.append("has_magnitude")
        if p.require_overlap_anomaly:
            conds.append("overlaps_anomaly")
        act = (
            f"{a.op} the effect window by "
            + (f"{a.cap:.0%}·dir (mag-capped)" if a.magnitude == "frac_capped" else f"{a.fixed}·dir")
            if a.op != "none" else "do nothing"
        )
        lines.append(f"  rule{i}: if {' ∧ '.join(conds)} → {act}")
    return "\n".join(lines)


# ---- seed policies (evolution starts here) ----

IDENTITY = Policy(rules=(), name="identity")

EVENT_SCALE = Policy(
    name="event_scale",
    rules=(
        Rule(
            predicate=Predicate(),  # strict defaults: grounded+numeric+entity+target+window+mag
            action=Action(op="scale", magnitude="frac_capped", cap=0.3),
        ),
    ),
)

# A relaxed variant that drops the literal entity_match gate (the task_152 fix
# candidate) — evolution keeps it only if it helps held-out + stays grounded.
EVENT_SCALE_RELAXED_ENTITY = Policy(
    name="event_scale_relaxed_entity",
    rules=(
        Rule(
            predicate=Predicate(require_entity_match=False, require_numeric_eligible=False),
            action=Action(op="scale", magnitude="frac_capped", cap=0.3),
        ),
    ),
)

# Maximally relaxed on retrieval's *judgment* flags (entity/target/numeric), keeping
# only what the kernel + a real effect need: grounded ∧ directional ∧ windowed ∧
# quantified. On the real cached dev cards this is the ONLY seed that fires on
# task_152 (the July-4 holiday → -11.2% road-occupancy effect that retrieval marked
# entity=target=numeric=False). It is still safe — grounded + bounded by the kernel —
# and only survives evolution if it improves held-out accuracy without regressing.
GROUNDED_EVENT = Policy(
    name="grounded_event",
    rules=(
        Rule(
            predicate=Predicate(
                require_entity_match=False,
                require_target_match=False,
                require_numeric_eligible=False,
            ),
            action=Action(op="scale", magnitude="frac_capped", cap=0.3),
        ),
    ),
)
