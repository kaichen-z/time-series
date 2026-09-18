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
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Optional, Sequence

from .post_adjust import EvidenceEffect, _parse, apply_bounded_delta, horizon_window_mask

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
    magnitude: str = "frac_capped"  # "frac_capped" | "fixed" | "history_calibrated"
    cap: float = 0.3           # per-rule cap on the fractional/relative move
    fixed: float = 0.0         # used when magnitude == "fixed"
    regime: str = ""           # used when magnitude == "history_calibrated": estimator name


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


# ---- regime estimators (data->text side): switchable candidates -------------
# Each estimator reads the series' history and returns a per-horizon-step
# multiplicative factor (relative to base) for the in-window steps, or None when the
# document-explained regime is NOT observable in the history. Returning None is the
# data-driven *qualify gate*: no observable regime => the effect is dropped, replacing
# retrieval's literal numeric_eligible/entity_match text judgment with a test on the
# numbers. Which estimator to use is an evolvable, auditable choice (Action.regime).
RegimeEstimator = Callable[..., Optional[tuple]]
_REGIMES: dict[str, RegimeEstimator] = {}


def _register(name: str):
    def deco(fn: RegimeEstimator) -> RegimeEstimator:
        _REGIMES[name] = fn
        return fn
    return deco


def _in_window_factor(mask, factor: Optional[float]) -> Optional[tuple]:
    if factor is None or not math.isfinite(factor):
        return None
    return tuple(factor if m else 1.0 for m in mask)


@_register("weekend_weekday")
def _est_weekend_weekday(hv, hts, effect, fts, mask):
    """Level of the weekend regime relative to weekdays (holiday ~ behaves like a weekend)."""
    parsed = [_parse(str(x)) for x in hts]
    wk = [v for ts, v in zip(parsed, hv) if ts is not None and ts.weekday() < 5]
    we = [v for ts, v in zip(parsed, hv) if ts is not None and ts.weekday() >= 5]
    if not wk or not we:
        return None
    base = mean(wk)
    if base == 0:
        return None
    return _in_window_factor(mask, mean(we) / base)


@_register("low_quantile_day")
def _est_low_quantile_day(hv, hts, effect, fts, mask, q: float = 0.25):
    """Low-activity day level (q-quantile of daily means) relative to the overall mean."""
    parsed = [_parse(str(x)) for x in hts]
    byday: dict = {}
    for ts, v in zip(parsed, hv):
        if ts is not None:
            byday.setdefault(ts.date(), []).append(v)
    if not byday:
        return None
    dmeans = sorted(mean(vs) for vs in byday.values())
    overall = mean(hv) if hv else 0.0
    if overall == 0:
        return None
    lowq = dmeans[max(0, min(len(dmeans) - 1, int(round(q * (len(dmeans) - 1)))))]
    return _in_window_factor(mask, lowq / overall)


@_register("hour_of_day")
def _est_hour_of_day(hv, hts, effect, fts, mask):
    """Per-step factor from the matching hour-of-day mean (intra-day periodicity)."""
    parsed = [_parse(str(x)) for x in hts]
    byhour: dict = {}
    for ts, v in zip(parsed, hv):
        if ts is not None:
            byhour.setdefault(ts.hour, []).append(v)
    overall = mean(hv) if hv else 0.0
    if overall == 0 or not byhour:
        return None
    fparsed = [_parse(str(x)) for x in fts]
    factors = []
    for i, m in enumerate(mask):
        ts = fparsed[i]
        if m and ts is not None and ts.hour in byhour:
            factors.append(mean(byhour[ts.hour]) / overall)
        else:
            factors.append(1.0)
    if all(f == 1.0 for f in factors):
        return None
    return tuple(factors)


def _calibrated_factors(action: Action, e, hv, hts, fts, mask) -> Optional[tuple]:
    """Run the chosen regime estimator, gate on data<->text direction agreement.

    Returns per-step multiplicative factors (deviation clamped to action.cap), or None
    when the effect must be dropped: no history, unknown/failed estimator, no observable
    regime, or the regime's direction contradicts the document's direction.
    """
    if hv is None or hts is None:
        return None
    est = _REGIMES.get(action.regime)
    if est is None:
        return None
    factors = est(hv, hts, e, fts, mask)
    if factors is None:
        return None
    inwin = [f for f, m in zip(factors, mask) if m]
    if not inwin:
        return None
    avg = mean(inwin)
    regime_dir = "decrease" if avg < 1.0 else ("increase" if avg > 1.0 else "none")
    # data<->text cross-check: the numbers must agree with the document's direction.
    if regime_dir == "none" or regime_dir != e.direction:
        return None
    return tuple(
        1.0 + max(-action.cap, min(action.cap, f - 1.0)) if m else 1.0
        for f, m in zip(factors, mask)
    )


def apply_policy(
    policy: Policy,
    base_forecast: Sequence[float],
    effects: Sequence[EvidenceEffect],
    future_timestamps: Sequence[str],
    anomaly_windows: Sequence[tuple[str | None, str | None]] = (),
    *,
    history_values: Optional[Sequence[float]] = None,
    history_timestamps: Optional[Sequence[str]] = None,
):
    """Interpret the rule table; return (forecast, fired) with kernel invariants.

    `fired` lists (effect, rule) pairs for audit/provenance. History (values +
    timestamps) is required only by `history_calibrated` rules; without it those rules
    are inert (the effect is simply not adjusted).
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
            act = rule.action
            if act.magnitude == "history_calibrated":
                factors = _calibrated_factors(
                    act, e, history_values, history_timestamps, future_timestamps, mask)
                if factors is None:
                    continue  # data-driven qualify gate dropped this effect
                for i, inside in enumerate(mask):
                    if inside:
                        proposed[i] = base[i] * factors[i]
                fired.append((e, rule))
                break
            if act.magnitude == "frac_capped":
                mag = abs(e.magnitude_value) if e.magnitude_value is not None else 0.0
                if e.magnitude_kind != "relative":
                    mag = min(mag, act.cap)  # non-relative magnitudes: treat conservatively
                frac = min(mag, act.cap)
            else:
                frac = act.cap  # fixed fractional move
            for i, inside in enumerate(mask):
                if not inside:
                    continue
                if act.op == "scale":
                    proposed[i] = base[i] * (1.0 + _sign(e.direction) * frac)
                elif act.op == "shift":
                    proposed[i] = base[i] + _sign(e.direction) * (
                        act.fixed if act.magnitude == "fixed" else frac * abs(base[i])
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
        if a.op == "none":
            act = "do nothing"
        elif a.magnitude == "history_calibrated":
            act = (f"{a.op} the effect window to the '{a.regime}' historical regime "
                   f"(direction must agree with history; deviation ≤{a.cap:.0%})")
        elif a.magnitude == "fixed":
            act = f"{a.op} the effect window by {a.fixed}·dir"
        else:
            act = f"{a.op} the effect window by {a.cap:.0%}·dir (mag-capped)"
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


# ---- history-calibrated seed policies (switchable regime candidates) ----------
# Same relaxed grounded-event predicate, but the move magnitude is calibrated from a
# document-explained regime in the history instead of a constant. These are the
# switchable candidates the co-evolution / held-out picks between. They also DROP an
# effect whenever the regime is not observable or contradicts the document direction.
_RELAXED = Predicate(
    require_entity_match=False, require_target_match=False, require_numeric_eligible=False,
)


def _calibrated_policy(regime: str, cap: float = KERNEL_MAX_FRAC) -> Policy:
    return Policy(
        name=f"grounded_event_cal[{regime}]",
        rules=(Rule(_RELAXED, Action(op="scale", magnitude="history_calibrated",
                                     regime=regime, cap=cap)),),
    )


GROUNDED_EVENT_CAL_WEEKEND = _calibrated_policy("weekend_weekday")
GROUNDED_EVENT_CAL_LOWQ = _calibrated_policy("low_quantile_day")
GROUNDED_EVENT_CAL_HOUR = _calibrated_policy("hour_of_day")

# Ordered fallback: prefer weekend calibration, else low-quantile, else fixed cap. The
# first rule whose regime is observable + direction-consistent fires; otherwise the
# grounded fixed-cap rule still applies the effect (never worse than GROUNDED_EVENT).
GROUNDED_EVENT_CAL_CASCADE = Policy(
    name="grounded_event_cal_cascade",
    rules=(
        Rule(_RELAXED, Action(op="scale", magnitude="history_calibrated",
                              regime="weekend_weekday", cap=KERNEL_MAX_FRAC)),
        Rule(_RELAXED, Action(op="scale", magnitude="history_calibrated",
                              regime="low_quantile_day", cap=KERNEL_MAX_FRAC)),
        Rule(_RELAXED, Action(op="scale", magnitude="frac_capped", cap=0.3)),
    ),
)

CALIBRATED_SEEDS = {
    "weekend_weekday": GROUNDED_EVENT_CAL_WEEKEND,
    "low_quantile_day": GROUNDED_EVENT_CAL_LOWQ,
    "hour_of_day": GROUNDED_EVENT_CAL_HOUR,
    "cascade": GROUNDED_EVENT_CAL_CASCADE,
}


def available_regimes() -> tuple[str, ...]:
    """Names of the switchable regime estimators (for evolution / inspection)."""
    return tuple(_REGIMES.keys())
