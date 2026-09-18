"""(B) Three co-evolved genomes with an explicit, typed protocol between them.

The single Level-0 `dsl.Policy` folds qualify+calibrate+integrate into one rule. Here
we split the interaction into THREE independently evolvable policies wired by a typed
protocol, which is the project's core claim: the whole cross-modal interaction process
co-evolves, every stage is code (rules-as-data), and the hand-offs are explicit.

    series ─[FocusPolicy]────────▶ Regime[]            (data→text: observable regularities)
    raw effects + Regime[] ─[QualifyPolicy]─▶ QualifiedEffect[]  (text↔data: gate + calibrate)
    base + QualifiedEffect[] ─[IntegratePolicy]─▶ forecast        (bounded by the kernel)

The kernel is unchanged and lives in `apply_bounded_delta` + the grounded gate: no
evolved policy can emit an unbounded or ungrounded move. `InteractionState` is the
replayable audit trace (protocol clarity).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from statistics import mean
from typing import Optional, Sequence

from .dsl import (
    KERNEL_MAX_FRAC, Predicate, _REGIMES, _sign, available_regimes,
)
from .evolve import _mutate_predicate
from .post_adjust import EvidenceEffect, apply_bounded_delta, horizon_window_mask

# --------------------------------------------------------------------------- perception
@dataclass(frozen=True)
class PerceptionConfig:
    """Evolvable knobs for HOW MUCH numeric context the retrieval LLM is shown.

    This is the perception layer made evolvable: it shapes `series_context` in the
    retrieval view so the LLM can calibrate magnitude scale/units and sanity-check
    direction. Evaluating a config needs one LLM pass per task, so cards are generated
    offline once per config and cached by `fingerprint()`; co-evolution then selects
    among the configs whose cards exist (cache hits are free)."""
    recent_window: int = 24
    include_stats: bool = True
    include_magnitude_hint: bool = True
    strict_direction: bool = True

    def build_context(self, history_values) -> Optional[dict]:
        hv = [float(x) for x in history_values]
        if not hv:
            return None
        recent = hv[-max(1, self.recent_window):]
        ctx: dict = {"recent_values": [round(x, 4) for x in recent]}
        if self.include_stats:
            ctx["history_stats"] = {"min": round(min(hv), 4), "max": round(max(hv), 4),
                                    "mean": round(mean(hv), 4), "count": len(hv)}
        if self.include_magnitude_hint:
            scale = mean(abs(x) for x in hv) or 1.0
            ctx["typical_scale"] = round(scale, 4)
            ctx["note"] = ("Any additive change must be comparable to typical_scale; "
                           "reject document numbers far outside this range.")
        if self.strict_direction:
            ctx["direction_rule"] = ("Only assert a direction that agrees with the recent "
                                     "values unless a document explicitly and strongly states otherwise.")
        return ctx

    def fingerprint(self) -> str:
        import hashlib
        import json as _json
        return hashlib.sha256(_json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:12]


# a "blind" config == the historic behaviour (no numeric context at all)
PERCEPTION_BLIND = PerceptionConfig(recent_window=0, include_stats=False,
                                    include_magnitude_hint=False, strict_direction=False)
PERCEPTION_FULL = PerceptionConfig()


def _mutate_perception(p: PerceptionConfig, rng: random.Random) -> PerceptionConfig:
    r = rng.random()
    if r < 0.4:
        return replace(p, recent_window=max(0, p.recent_window + rng.choice((-12, -6, 6, 12))))
    if r < 0.6:
        return replace(p, include_stats=not p.include_stats)
    if r < 0.8:
        return replace(p, include_magnitude_hint=not p.include_magnitude_hint)
    return replace(p, strict_direction=not p.strict_direction)


# --------------------------------------------------------------------------- protocol
@dataclass(frozen=True)
class Regime:
    """A document-explainable regularity observed in the numbers (Focus → Qualify)."""
    kind: str                 # estimator name
    factors: tuple            # per-horizon-step multiplicative factor vs base level
    direction: str            # "increase" | "decrease"
    strength: float           # |1 - mean(factor)|


@dataclass(frozen=True)
class QualifiedEffect:
    """A grounded, actionable effect with a per-step target factor (Qualify → Integrate)."""
    direction: str
    mask: tuple               # which horizon steps it applies to
    factors: tuple            # per-step desired multiplicative factor
    source: str               # "doc" | "calibrated:<regime>"
    grounded: bool
    reason: str               # human-readable audit note


@dataclass(frozen=True)
class InteractionState:
    """Replayable trace of one prediction — the explicit protocol contents."""
    regimes: tuple
    qualified: tuple
    fired: tuple

    def to_text(self) -> str:
        lines = [f"regimes: {[f'{r.kind}({r.direction},{r.strength:.2f})' for r in self.regimes] or 'none'}"]
        lines.append(f"qualified: {[f'{q.source}/{q.direction}' for q in self.qualified] or 'none'}")
        lines.append(f"fired: {len(self.fired)} effect(s)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- genomes
@dataclass(frozen=True)
class FocusPolicy:
    """Which regime estimators to trust, and how pronounced a regime must be to count."""
    regimes: tuple = ("weekend_weekday", "low_quantile_day", "hour_of_day")
    min_strength: float = 0.05

    def detect(self, hv, hts, fts) -> tuple:
        allmask = tuple(True for _ in fts)
        out = []
        for name in self.regimes:
            est = _REGIMES.get(name)
            if est is None:
                continue
            factors = est(hv, hts, None, fts, allmask)
            if factors is None:
                continue
            avg = mean(factors)
            strength = abs(1.0 - avg)
            if strength < self.min_strength or avg == 1.0:
                continue
            out.append(Regime(kind=name, factors=tuple(factors),
                              direction="decrease" if avg < 1.0 else "increase",
                              strength=strength))
        return tuple(out)


@dataclass(frozen=True)
class QualifyPolicy:
    """Gate raw evidence and choose its magnitude source (doc / regime-calibrated)."""
    predicate: Predicate = Predicate(
        require_entity_match=False, require_target_match=False, require_numeric_eligible=False)
    magnitude_source: str = "cascade"     # "doc" | "calibrated" | "cascade"
    doc_cap: float = 0.3

    def apply(self, raw_effects, regimes, fts) -> tuple:
        out = []
        for e in raw_effects:
            if not self.predicate.matches(e, (), fts):
                continue
            mask = horizon_window_mask(fts, e.start_timestamp, e.end_timestamp)
            if not any(mask):
                continue
            qe = self._one(e, regimes, mask)
            if qe is not None:
                out.append(qe)
        return tuple(out)

    def _one(self, e, regimes, mask) -> Optional[QualifiedEffect]:
        if self.magnitude_source in ("calibrated", "cascade"):
            for r in regimes:
                if r.direction != e.direction:        # data↔text cross-check
                    continue
                factors = tuple(r.factors[i] if m else 1.0 for i, m in enumerate(mask))
                return QualifiedEffect(e.direction, mask, factors, f"calibrated:{r.kind}",
                                       e.grounded, f"regime {r.kind} agrees ({r.direction})")
        if self.magnitude_source in ("doc", "cascade"):
            mag = abs(e.magnitude_value) if e.magnitude_value is not None else 0.0
            if mag <= 0:
                return None
            frac = min(mag, self.doc_cap)
            factors = tuple(1.0 + _sign(e.direction) * frac if m else 1.0 for m in mask)
            return QualifiedEffect(e.direction, mask, factors, "doc", e.grounded,
                                   f"document magnitude {frac:.0%}")
        return None


@dataclass(frozen=True)
class IntegratePolicy:
    """Apply qualified effects to the base forecast; the kernel then bounds the result."""
    op: str = "scale"            # "scale" | "shift"
    max_frac: float = KERNEL_MAX_FRAC

    def apply(self, base, qualified):
        base = [float(b) for b in base]
        proposed = list(base)
        fired = []
        cap = min(self.max_frac, KERNEL_MAX_FRAC)
        for qe in qualified:
            if not qe.grounded:                       # kernel: grounded-only
                continue
            for i, m in enumerate(qe.mask):
                if not m:
                    continue
                dev = max(-cap, min(cap, qe.factors[i] - 1.0))
                proposed[i] = base[i] * (1.0 + dev) if self.op == "scale" \
                    else base[i] + dev * abs(base[i])
            fired.append(qe)
        out = apply_bounded_delta(base, proposed, max_frac=cap)   # kernel envelope
        return out, tuple(fired)


@dataclass(frozen=True)
class Interaction:
    """The joint genome: three co-evolved policies + the fixed perception upstream."""
    focus: FocusPolicy = FocusPolicy()
    qualify: QualifyPolicy = QualifyPolicy()
    integrate: IntegratePolicy = IntegratePolicy()


def run_pipeline(inter: Interaction, base, raw_effects, fts, hv, hts):
    """Run the 3-stage protocol; return (forecast, InteractionState)."""
    regimes = inter.focus.detect(hv, hts, fts)
    qualified = inter.qualify.apply(raw_effects, regimes, fts)
    forecast, fired = inter.integrate.apply(base, qualified)
    return forecast, InteractionState(regimes, qualified, fired)


def interaction_to_text(inter: Interaction) -> str:
    f, q, ig = inter.focus, inter.qualify, inter.integrate
    return (
        f"Focus:     trust regimes {list(f.regimes)} with strength ≥ {f.min_strength:.2f}\n"
        f"Qualify:   gate=[{_pred_text(q.predicate)}], magnitude={q.magnitude_source} "
        f"(doc cap {q.doc_cap:.0%})\n"
        f"Integrate: {ig.op} within ±{min(ig.max_frac, KERNEL_MAX_FRAC):.0%} of base (kernel-bounded)"
    )


def _pred_text(p: Predicate) -> str:
    on = [n.replace("require_", "") for n in (
        "require_numeric_eligible", "require_entity_match", "require_target_match",
        "require_window", "require_magnitude") if getattr(p, n)]
    return "grounded+" + "+".join(on) if on else "grounded-only"


# --------------------------------------------------------------------------- co-evolution
_SOURCES = ("doc", "calibrated", "cascade")


def _mutate_focus(f: FocusPolicy, rng: random.Random) -> FocusPolicy:
    if rng.random() < 0.5:
        pool = list(available_regimes())
        name = rng.choice(pool)
        regs = set(f.regimes)
        regs.symmetric_difference_update({name})    # toggle membership
        return replace(f, regimes=tuple(r for r in pool if r in regs))
    return replace(f, min_strength=max(0.0, min(0.5, f.min_strength + rng.uniform(-0.05, 0.05))))


def _mutate_qualify(q: QualifyPolicy, rng: random.Random) -> QualifyPolicy:
    r = rng.random()
    if r < 0.5:
        return replace(q, predicate=_mutate_predicate(q.predicate, rng))
    if r < 0.8:
        return replace(q, magnitude_source=rng.choice(_SOURCES))
    return replace(q, doc_cap=max(0.0, min(KERNEL_MAX_FRAC, q.doc_cap + rng.uniform(-0.1, 0.1))))


def _mutate_integrate(ig: IntegratePolicy, rng: random.Random) -> IntegratePolicy:
    if rng.random() < 0.5:
        return replace(ig, op=rng.choice(("scale", "shift")))
    return replace(ig, max_frac=max(0.05, min(KERNEL_MAX_FRAC, ig.max_frac + rng.uniform(-0.15, 0.15))))


def mutate_interaction(inter: Interaction, rng: random.Random) -> Interaction:
    which = rng.randrange(3)
    if which == 0:
        return replace(inter, focus=_mutate_focus(inter.focus, rng))
    if which == 1:
        return replace(inter, qualify=_mutate_qualify(inter.qualify, rng))
    return replace(inter, integrate=_mutate_integrate(inter.integrate, rng))


def crossover_interaction(a: Interaction, b: Interaction, rng: random.Random) -> Interaction:
    """Component-wise recombination — literally mixing sub-policies of the interaction."""
    return Interaction(
        focus=rng.choice((a.focus, b.focus)),
        qualify=rng.choice((a.qualify, b.qualify)),
        integrate=rng.choice((a.integrate, b.integrate)),
    )


def run_coevolution(seeds, fitness, *, generations=30, pop_size=48, elite=10, seed=20260918):
    """(mu+lambda) co-evolution over the joint (focus, qualify, integrate) genome."""
    rng = random.Random(seed)
    pop = list(seeds)
    while len(pop) < pop_size:
        pop.append(mutate_interaction(rng.choice(seeds) if seeds else Interaction(), rng))

    def key(x):
        return fitness(x)

    best = max(pop, key=key)
    history = []
    for gen in range(generations):
        ranked = sorted(pop, key=key, reverse=True)
        parents = ranked[:elite]
        if key(ranked[0]) > key(best):
            best = ranked[0]
        history.append((gen, fitness(ranked[0])))
        children = list(parents)
        while len(children) < pop_size:
            if rng.random() < 0.5 and len(parents) >= 2:
                child = crossover_interaction(*rng.sample(parents, 2), rng)
            else:
                child = rng.choice(parents)
            children.append(mutate_interaction(child, rng))
        pop = children
    return best, fitness(best), history
