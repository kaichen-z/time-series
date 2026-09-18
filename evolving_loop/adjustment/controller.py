"""Evolvable decision controller: the orchestration itself is discovered, not designed.

Instead of a hand-fixed pipeline (focus -> qualify -> integrate, then post-adjust), a
Controller is an ORDERED LIST OF INSTRUCTIONS over a shared ControllerState. Each
instruction is a small, typed, auditable SAFE PRIMITIVE (a building block); the ORDER
and CHOICE of instructions -- i.e. how numerical candidates and retrieval evidence get
combined, and whether/what to adjust -- is what evolution (or an LLM designer) explores.

The invariant kernel is never part of what evolves: `run_controller` always bounds the
final forecast against the selected base (apply_bounded_delta) and adjustment
primitives only ever act on GROUNDED effects. So any evolved orchestration stays safe
and auditable by construction; only the control flow is free.

Primitives here compute for real (reusing the proven adjustment primitives), so once
80-train effect cards exist this can be evolved/LLM-designed in place of the hand
pipeline. `re-generate numerical` is intentionally left as a future primitive stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import mean
from typing import Optional, Sequence

from .dsl import _REGIMES, _sign, KERNEL_MAX_FRAC, available_regimes
from .post_adjust import EvidenceEffect, apply_bounded_delta, horizon_window_mask


@dataclass(frozen=True)
class ControllerState:
    candidates: dict          # numerical ranked alternatives: name -> forecast tuple
    base: tuple               # the selected base (kernel bounds the final vs THIS)
    forecast: tuple           # current working forecast
    effects: tuple            # retrieval EvidenceEffects
    hv: tuple                 # history values
    hts: tuple                # history timestamps
    fts: tuple                # future timestamps
    trace: tuple = ()         # audit log of applied instructions


def _grounded_windows(effects, fts):
    """Yield (effect, window_mask) for grounded, windowed effects only (kernel gate)."""
    for e in effects:
        if e.grounded and e.start_timestamp and e.end_timestamp:
            mask = horizon_window_mask(fts, e.start_timestamp, e.end_timestamp)
            if any(mask):
                yield e, mask


# ---------------------------------------------------------------- instructions
@dataclass(frozen=True)
class SelectBase:
    """Pick a numerical candidate as the working base (default: strongest = toto)."""
    candidate: str = "toto_2_0"

    def apply(self, s: ControllerState) -> ControllerState:
        base = s.candidates.get(self.candidate)
        if base is None:
            base = next(iter(s.candidates.values())) if s.candidates else s.base
        base = tuple(float(x) for x in base)
        return replace(s, base=base, forecast=base, trace=s.trace + (f"select {self.candidate}",))


@dataclass(frozen=True)
class RegimeAdjust:
    """Scale each grounded windowed effect toward a historical regime (data-derived)."""
    regime: str = "weekend_weekday"
    cap: float = KERNEL_MAX_FRAC
    fill_unknown_direction: bool = True

    def apply(self, s: ControllerState) -> ControllerState:
        est = _REGIMES.get(self.regime)
        if est is None:
            return s
        fc = list(s.forecast)
        fired = 0
        for e, mask in _grounded_windows(s.effects, s.fts):
            factors = est(list(s.hv), list(s.hts), None, list(s.fts), mask)
            if factors is None:
                continue
            avg = mean([factors[i] for i, m in enumerate(mask) if m] or [1.0])
            rdir = "decrease" if avg < 1.0 else ("increase" if avg > 1.0 else "none")
            if rdir == "none":
                continue
            if e.direction in ("increase", "decrease") and rdir != e.direction:
                continue                                   # data<->text disagree: skip
            if e.direction not in ("increase", "decrease") and not self.fill_unknown_direction:
                continue
            for i, m in enumerate(mask):
                if m:
                    dev = max(-self.cap, min(self.cap, factors[i] - 1.0))
                    fc[i] = s.base[i] * (1.0 + dev)
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"regime_adjust[{self.regime}] x{fired}",))


@dataclass(frozen=True)
class DocAdjust:
    """Scale grounded windowed effects by the document's own magnitude (capped)."""
    cap: float = 0.3

    def apply(self, s: ControllerState) -> ControllerState:
        fc = list(s.forecast)
        fired = 0
        for e, mask in _grounded_windows(s.effects, s.fts):
            if e.direction not in ("increase", "decrease") or e.magnitude_value is None:
                continue
            frac = min(abs(e.magnitude_value), self.cap)
            for i, m in enumerate(mask):
                if m:
                    fc[i] = s.base[i] * (1.0 + _sign(e.direction) * frac)
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"doc_adjust x{fired}",))


@dataclass(frozen=True)
class NoOp:
    def apply(self, s: ControllerState) -> ControllerState:
        return replace(s, trace=s.trace + ("noop",))


# future primitive (stub): feed evidence back and re-run numerical generation. Left
# unimplemented so the skeleton stays offline+fast; wiring it needs a numerical callback.
@dataclass(frozen=True)
class Regenerate:
    note: str = "regenerate-numerical (stub)"

    def apply(self, s: ControllerState) -> ControllerState:
        return replace(s, trace=s.trace + ("regenerate(stub:noop)",))


INSTRUCTIONS = (SelectBase, RegimeAdjust, DocAdjust, NoOp, Regenerate)


@dataclass(frozen=True)
class Controller:
    steps: tuple = ()
    name: str = "controller"


def run_controller(controller: Controller, candidates: dict, effects: Sequence,
                   hv: Sequence, hts: Sequence, fts: Sequence):
    """Execute the instruction sequence; return (final_forecast, trace).

    Kernel: the result is always bounded against the selected base, so no orchestration
    can emit a catastrophic move; adjustment primitives only touch grounded effects.
    """
    default_base = tuple(float(x) for x in (candidates.get("toto_2_0")
                          or (next(iter(candidates.values())) if candidates else ())))
    s = ControllerState(candidates=dict(candidates), base=default_base, forecast=default_base,
                        effects=tuple(effects), hv=tuple(hv), hts=tuple(hts), fts=tuple(fts))
    for step in controller.steps:
        s = step.apply(s)
    out = apply_bounded_delta(list(s.base), list(s.forecast), max_frac=KERNEL_MAX_FRAC)
    return out, s.trace


def controller_to_text(controller: Controller) -> str:
    if not controller.steps:
        return f"{controller.name}: (empty -> base forecast)"
    lines = [f"{controller.name}:"]
    for i, st in enumerate(controller.steps, 1):
        params = ", ".join(f"{k}={v}" for k, v in st.__dict__.items())
        lines.append(f"  {i}. {type(st).__name__}({params})")
    return "\n".join(lines)


# ---- seed controllers (evolution starts here) ----
IDENTITY_CONTROLLER = Controller(steps=(SelectBase(),), name="identity")
REGIME_CONTROLLER = Controller(
    steps=(SelectBase(), RegimeAdjust("weekend_weekday")), name="select+regime")
CASCADE_CONTROLLER = Controller(
    steps=(SelectBase(), RegimeAdjust("weekend_weekday"), RegimeAdjust("low_quantile_day"),
           DocAdjust(cap=0.3)),
    name="select+regime-cascade+doc")

SEED_CONTROLLERS = (IDENTITY_CONTROLLER, REGIME_CONTROLLER, CASCADE_CONTROLLER)


# ---------------------------------------------------------------- evolution of control flow
import random as _random


def _clamp(c, lo=0.0, hi=KERNEL_MAX_FRAC):
    return max(lo, min(hi, c))


def random_instruction(rng: _random.Random):
    kind = rng.choice(INSTRUCTIONS)
    if kind is SelectBase:
        return SelectBase()
    if kind is RegimeAdjust:
        return RegimeAdjust(regime=rng.choice(available_regimes()),
                            cap=_clamp(rng.uniform(0.1, KERNEL_MAX_FRAC)),
                            fill_unknown_direction=rng.random() < 0.7)
    if kind is DocAdjust:
        return DocAdjust(cap=_clamp(rng.uniform(0.1, 0.4)))
    if kind is Regenerate:
        return Regenerate()
    return NoOp()


def _mutate_step(step, rng: _random.Random):
    if isinstance(step, RegimeAdjust):
        r = rng.random()
        if r < 0.5:
            return replace(step, regime=rng.choice(available_regimes()))
        if r < 0.8:
            return replace(step, cap=_clamp(step.cap + rng.uniform(-0.15, 0.15)))
        return replace(step, fill_unknown_direction=not step.fill_unknown_direction)
    if isinstance(step, DocAdjust):
        return replace(step, cap=_clamp(step.cap + rng.uniform(-0.1, 0.1), hi=0.5))
    return step


def mutate_controller(controller: Controller, rng: _random.Random) -> Controller:
    """One structural or parameter edit to the instruction sequence."""
    steps = list(controller.steps)
    r = rng.random()
    if not steps or r < 0.25:                              # insert
        steps.insert(rng.randint(0, len(steps)), random_instruction(rng))
    elif r < 0.45 and len(steps) > 1:                      # remove
        del steps[rng.randrange(len(steps))]
    elif r < 0.6 and len(steps) > 1:                       # reorder
        i, j = rng.sample(range(len(steps)), 2)
        steps[i], steps[j] = steps[j], steps[i]
    else:                                                  # edit a step's params
        i = rng.randrange(len(steps))
        steps[i] = _mutate_step(steps[i], rng)
    return Controller(steps=tuple(steps)[:8], name="evolved")


def crossover_controllers(a: Controller, b: Controller, rng: _random.Random) -> Controller:
    ca = list(a.steps)[: rng.randint(0, len(a.steps))]
    cb = list(b.steps)[rng.randint(0, len(b.steps)):]
    child = (ca + cb)[:8] or [SelectBase()]
    return Controller(steps=tuple(child), name="evolved")
