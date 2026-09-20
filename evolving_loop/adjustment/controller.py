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

import math
from dataclasses import dataclass, field, replace
from statistics import mean, median, pstdev
from typing import Optional, Sequence

from .dsl import _REGIMES, _sign, KERNEL_MAX_FRAC, available_regimes
from .post_adjust import EvidenceEffect, _parse, apply_bounded_delta, horizon_window_mask


@dataclass(frozen=True)
class ControllerState:
    candidates: dict          # numerical ranked alternatives: name -> forecast tuple
    base: tuple               # the selected base (kernel bounds the final vs THIS)
    forecast: tuple           # current working forecast
    effects: tuple            # retrieval EvidenceEffects
    hv: tuple                 # history values
    hts: tuple                # history timestamps
    fts: tuple                # future timestamps
    semantic_ref: str = ""    # cached LLM semantic mapping (e.g. "weekend"); "" = none
    effect_pool: dict = None  # cross-task pooled relative effect per regime (ref -> (rel, n))
    cordp_conf: float = 0.0   # cached CorDP self-reported confidence for this task
    cordp_corrections: tuple = ()  # cached CorDP edits: (start_ts, end_ts, raw_multiplier)
    residual_conf: float = 0.0     # cached confidence for the residual specs
    residual_specs: tuple = ()     # cached ResidualSpec objects (text -> math residual)
    menu: tuple = ()          # MathOp menu the Numerical agent exposed for this task
    tools: dict = None        # name -> callable(state, arg) tool registry (agent-as-tool)
    calls: tuple = ()         # log of CALL instructions issued (for the efficiency cost)
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


def _reference_level(hv, hts, ref):
    """Historical level for a named regime (weekend/weekday/low_day/high_day/recent/overall)."""
    if not hv:
        return None

    def wd(t):
        p = _parse(str(t))
        return p.weekday() if p else -1

    if ref == "overall":
        return mean(hv)
    if ref == "recent":
        return mean(hv[-24:])
    if ref == "weekday":
        v = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
        return mean(v) if v else None
    if ref == "weekend":
        v = [x for x, t in zip(hv, hts) if wd(t) >= 5]
        return mean(v) if v else None
    if ref in ("low_day", "high_day"):
        byday: dict = {}
        for x, t in zip(hv, hts):
            p = _parse(str(t))
            if p:
                byday.setdefault(p.date(), []).append(x)
        if not byday:
            return None
        dm = sorted(mean(z) for z in byday.values())
        return dm[0] if ref == "low_day" else dm[-1]
    return None


_ALL_REFS = ("weekend", "weekday", "low_day", "high_day", "recent", "overall")


def _regime_significant(hv, hts, ref, min_t=2.0, min_n=3):
    """Per-task test: is this regime a STATISTICALLY significant, stable departure from
    the rest of the history? Uses a Welch-style t between the regime's samples and the
    complement (weekday/weekend), or day-mean extremity for low/high_day. This is
    computed per task from its OWN history, so it generalizes (no learned global param);
    single-day extremes and noise fail it automatically."""
    if not hv or ref in ("overall", "recent"):
        return False

    def wd(t):
        p = _parse(str(t)); return p.weekday() if p else -1

    if ref in ("weekend", "weekday"):
        grp = [v for v, t in zip(hv, hts) if (wd(t) >= 5) == (ref == "weekend") and wd(t) >= 0]
        rest = [v for v, t in zip(hv, hts) if (wd(t) >= 5) != (ref == "weekend") and wd(t) >= 0]
        if len(grp) < min_n or len(rest) < min_n:
            return False
        vg, vr = pstdev(grp), pstdev(rest)
        se = (vg * vg / len(grp) + vr * vr / len(rest)) ** 0.5
        if se <= 1e-9:
            return abs(mean(grp) - mean(rest)) > 1e-9
        return abs(mean(grp) - mean(rest)) / se >= min_t
    if ref in ("low_day", "high_day"):
        byday: dict = {}
        for v, t in zip(hv, hts):
            p = _parse(str(t))
            if p:
                byday.setdefault(p.date(), []).append(v)
        dm = sorted(mean(z) for z in byday.values())
        if len(dm) < 4:                        # too few days to call an extreme "stable"
            return False
        extreme, others = (dm[0], dm[1:]) if ref == "low_day" else (dm[-1], dm[:-1])
        sd = pstdev(others) if len(others) >= 2 else 0.0
        if sd <= 1e-9:
            return False
        return abs(extreme - mean(others)) / sd >= min_t
    return False


@dataclass(frozen=True)
class SemanticAdjust:
    """The proven primitive: scale grounded windowed effects toward the LLM-chosen
    historical regime. The reference (e.g. "weekend", cached in state.semantic_ref) is
    the LLM's semantic judgement of what the event resembles; the magnitude is read from
    that regime's historical level (data). No cached ref / no level -> no-op.

    ``trusted`` is the evolvable DATA GATE: only apply when the LLM's chosen regime is one
    we trust. Stable periodic regimes (weekday/weekend) are reliable; single-day extremes
    (low_day/high_day) and 'recent' tend to regress -- evolution learns which to trust."""
    cap: float = KERNEL_MAX_FRAC
    trusted: tuple = _ALL_REFS
    require_significant: bool = True     # per-task data gate: only act if regime is significant

    def apply(self, s: ControllerState) -> ControllerState:
        ref = s.semantic_ref
        if not ref or ref == "none" or ref not in self.trusted:
            return replace(s, trace=s.trace + ("semantic:none",))
        if self.require_significant and not _regime_significant(list(s.hv), list(s.hts), ref):
            return replace(s, trace=s.trace + (f"semantic:insignificant[{ref}]",))
        level = _reference_level(list(s.hv), list(s.hts), ref)
        if level is None:
            return replace(s, trace=s.trace + ("semantic:no-level",))
        fc = list(s.forecast)
        fired = 0
        for e, mask in _grounded_windows(s.effects, s.fts):
            win = [s.base[i] for i, m in enumerate(mask) if m]
            bw = mean(win) if win else 0.0
            if bw <= 0:
                continue
            factor = level / bw
            for i, m in enumerate(mask):
                if m:
                    dev = max(-self.cap, min(self.cap, factor - 1.0))
                    fc[i] = s.base[i] * (1.0 + dev)
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"semantic[{ref}] x{fired}",))


def build_event_effect_pool(items):
    """Pool cross-task event effects: for each regime, the mean level RELATIVE to that
    task's overall level, aggregated across tasks. `items` = iterable of (hv, hts).

    Returns {ref: (mean_relative, n_tasks)} -- e.g. pool["weekend"]=(0.7, 40) means
    "across 40 tasks, the weekend regime sits ~30% below normal". This is the cross-task
    prior that lets a task with too little history borrow a stable magnitude estimate.
    """
    acc: dict = {r: [] for r in _ALL_REFS}
    for hv, hts in items:
        hv = list(hv)
        if not hv:
            continue
        overall = mean(hv)
        if overall == 0:
            continue
        for ref in _ALL_REFS:
            lvl = _reference_level(hv, list(hts), ref)
            if lvl is not None:
                acc[ref].append(lvl / overall)
    return {r: (mean(v), len(v)) for r, v in acc.items() if v}


@dataclass(frozen=True)
class PooledSemanticAdjust:
    """SemanticAdjust with a CROSS-TASK prior. The magnitude for the LLM-chosen regime is
    a shrinkage blend of this task's own estimate and the pooled cross-task estimate
    (state.effect_pool), so a task with too little history borrows a stable magnitude from
    all tasks sharing that regime. `local_weight` (evolvable) trades local vs prior."""
    cap: float = KERNEL_MAX_FRAC
    trusted: tuple = _ALL_REFS
    local_weight: float = 0.5
    require_significant: bool = True     # per-task data gate: only act if regime is significant

    def apply(self, s: ControllerState) -> ControllerState:
        ref = s.semantic_ref
        if not ref or ref == "none" or ref not in self.trusted:
            return replace(s, trace=s.trace + ("pooled:none",))
        if self.require_significant and not _regime_significant(list(s.hv), list(s.hts), ref):
            return replace(s, trace=s.trace + (f"pooled:insignificant[{ref}]",))
        overall = mean(s.hv) if s.hv else 0.0
        if overall == 0:
            return replace(s, trace=s.trace + ("pooled:no-overall",))
        local = _reference_level(list(s.hv), list(s.hts), ref)
        local_rel = (local / overall) if local is not None else None
        pool = s.effect_pool or {}
        pool_rel = pool[ref][0] if ref in pool else None
        if local_rel is not None and pool_rel is not None:
            rel = self.local_weight * local_rel + (1.0 - self.local_weight) * pool_rel
        elif local_rel is not None:
            rel = local_rel
        elif pool_rel is not None:
            rel = pool_rel
        else:
            return replace(s, trace=s.trace + ("pooled:no-estimate",))
        target = rel * overall
        fc = list(s.forecast)
        fired = 0
        for e, mask in _grounded_windows(s.effects, s.fts):
            win = [s.base[i] for i, m in enumerate(mask) if m]
            bw = mean(win) if win else 0.0
            if bw <= 0:
                continue
            factor = target / bw
            for i, m in enumerate(mask):
                if m:
                    dev = max(-self.cap, min(self.cap, factor - 1.0))
                    fc[i] = s.base[i] * (1.0 + dev)
            fired += 1
        return replace(s, forecast=tuple(fc),
                       trace=s.trace + (f"pooled[{ref},w={self.local_weight:.1f}] x{fired}",))


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
class CorDPAdjust:
    """CorDP (document-anchored correction). Instead of extracting an abstract magnitude
    (which the LLM answers 'unknown'), perception SHOWED the LLM the Toto base numbers +
    documents and it returned per-window MULTIPLIERS anchored on that scale; those are cached
    in state.cordp_corrections. This primitive is pure code applying them through evolvable
    PRECISION GATES:
      - conf_min : abstain unless the cached self-reported confidence clears this floor;
      - wfrac_max: reject any correction spanning more than this fraction of the horizon --
        a whole-horizon rescale is the base model's job, not an event edit, and those misfire;
      - mult_lo/mult_hi: bound the multiplier (the invariant kernel still clamps the final);
      - shrink   : partial trust toward no-change (eff = 1 + shrink*(m-1)).
    Validated on 80-train: raw CorDP is net-negative, but (wfrac_max, conf_min) turn it into
    ~10:1 wins using only inference-time signals -- and those thresholds are what evolves."""
    conf_min: float = 0.7
    wfrac_max: float = 0.3
    mult_lo: float = 0.6
    mult_hi: float = 1.6
    shrink: float = 1.0

    def apply(self, s: ControllerState) -> ControllerState:
        if s.cordp_conf < self.conf_min or not s.cordp_corrections:
            return replace(s, trace=s.trace + (f"cordp:abstain(conf={s.cordp_conf:.2f})",))
        fc = list(s.forecast)
        n = len(s.fts)
        fired = 0
        for corr in s.cordp_corrections:
            try:
                start, end, mult = corr[0], corr[1], float(corr[2])
            except (TypeError, ValueError, IndexError):
                continue
            mask = horizon_window_mask(s.fts, str(start), str(end))
            wsum = sum(1 for x in mask if x)
            if wsum == 0 or (n and wsum / n > self.wfrac_max):     # reject global rescale
                continue
            eff = 1.0 + self.shrink * (mult - 1.0)
            eff = max(self.mult_lo, min(self.mult_hi, eff))
            for i, mk in enumerate(mask):
                if mk:
                    fc[i] = s.base[i] * eff
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"cordp x{fired}",))


@dataclass(frozen=True)
class ResidualSpec:
    """One text-derived residual over a window. The DOCUMENT supplies the STRUCTURE (window +
    shape + sign); the magnitude is a fraction of the window's base scale, and `grounded` marks
    whether that magnitude came from an explicit quote / a data-analog (True) or was guessed
    (False). Ungrounded specs are dropped -- "no source for the number => don't invent it"."""
    start: str
    end: str
    shape: str = "level"      # level | step | ramp | bump | decay
    amplitude: float = 0.0    # fractional PEAK change vs the window's base scale (signed)
    grounded: bool = True     # False -> abstain this spec (magnitude has no source)
    tau_frac: float = 0.5     # decay time-constant as a fraction of the window (shape=decay)


def _residual_profile(shape: str, j: int, k: int, tau_frac: float) -> float:
    """Unit shape in [0,1] across a k-step window at position j (the text's described form)."""
    if k <= 1:
        return 1.0
    x = j / (k - 1)                       # 0..1 across the window
    if shape == "ramp":
        return x
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)   # triangular: 0 at edges, 1 at centre
    if shape == "decay":
        tau = max(0.05, min(1.0, tau_frac))
        return math.exp(-x / tau)         # 1 at start, decays
    return 1.0                            # level / step / unknown -> flat


@dataclass(frozen=True)
class ResidualAdjust:
    """Apply text-derived ADDITIVE residuals with a temporal SHAPE (the advisor's
    'mathematical formulation of the residue'). Generalises CorDPAdjust: instead of one flat
    multiplier, each cached ResidualSpec adds `amplitude * shape(t) * ref` to the base, where
    `ref` = the window's base scale (floored by the history scale, so base~=0 can't blow up --
    the multiplicative pathology). Grounding is enforced (ungrounded specs dropped), plus the
    same evolvable precision gates as CorDP. The invariant kernel still bounds the final."""
    conf_min: float = 0.7
    wfrac_max: float = 0.3
    amp_cap: float = 0.5          # |amplitude| clamp (kernel also bounds the final vs base)
    shrink: float = 1.0
    require_grounded: bool = True

    def apply(self, s: ControllerState) -> ControllerState:
        if s.residual_conf < self.conf_min or not s.residual_specs:
            return replace(s, trace=s.trace + (f"residual:abstain(conf={s.residual_conf:.2f})",))
        hist_scale = median([abs(x) for x in s.hv]) if s.hv else 1.0
        hist_scale = hist_scale or 1.0
        fc = list(s.forecast)
        n = len(s.fts)
        fired = 0
        for spec in s.residual_specs:
            if self.require_grounded and not getattr(spec, "grounded", True):
                continue
            mask = horizon_window_mask(s.fts, str(spec.start), str(spec.end))
            widx = [i for i, m in enumerate(mask) if m]
            if not widx or (n and len(widx) / n > self.wfrac_max):   # reject global rescale
                continue
            wb = mean([s.base[i] for i in widx])
            ref = abs(wb) if abs(wb) > 1e-9 else hist_scale          # anchor; base~=0 -> hist scale
            amp = max(-self.amp_cap, min(self.amp_cap, self.shrink * float(spec.amplitude)))
            k = len(widx)
            for j, i in enumerate(widx):
                prof = _residual_profile(spec.shape, j, k, getattr(spec, "tau_frac", 0.5))
                fc[i] = s.base[i] + amp * ref * prof
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"residual x{fired}",))


def _select_menu_op(menu, semantic_ref, prefer):
    """Pick a MathOp: prefer the regime the document points at (semantic_ref); else the first
    op whose kind/name matches the evolvable `prefer` order."""
    if semantic_ref and semantic_ref not in ("", "none"):
        for op in menu:
            if op.name == f"regime:{semantic_ref}":
                return op
    for key in prefer:
        for op in menu:
            if op.kind == key or op.name.startswith(key):
                return op
    return menu[0] if menu else None


@dataclass(frozen=True)
class MenuAdjust:
    """Decision-DOES-MATH: pick a MathOp from the Numerical agent's per-task menu and APPLY it
    to each event window. The document SELECTS which op (via the cached regime hint / prefer
    order); the numerical menu SUPPLIES the data-computed number. Generalises RegimeAdjust /
    SemanticAdjust: the op set is now explicit and evolvable (regime-level / trend / decay / ...),
    magnitude is grounded in data (not an LLM guess), and the kernel still bounds the final."""
    prefer: tuple = ("regime", "trend", "decay")   # evolvable op-kind/name preference order
    cap: float = KERNEL_MAX_FRAC

    def apply(self, s: ControllerState) -> ControllerState:
        from .math_menu import op_window_values
        if not s.menu:
            return replace(s, trace=s.trace + ("menu:empty",))
        windows = []
        for corr in s.cordp_corrections:
            wm = horizon_window_mask(s.fts, str(corr[0]), str(corr[1]))
            if any(wm):
                windows.append(wm)
        for _e, mask in _grounded_windows(s.effects, s.fts):
            windows.append(mask)
        if not windows:
            return replace(s, trace=s.trace + ("menu:no-window",))
        op = _select_menu_op(s.menu, s.semantic_ref, self.prefer)
        if op is None:
            return replace(s, trace=s.trace + ("menu:no-op",))
        ref_scale = median([abs(x) for x in s.hv]) if s.hv else 1.0
        fc = list(s.forecast)
        fired = 0
        for wm in windows:
            idx = [i for i, m in enumerate(wm) if m]
            newv = op_window_values(op, [s.base[i] for i in idx], ref_scale or 1.0)
            for j, i in enumerate(idx):
                fc[i] = newv[j]
            fired += 1
        return replace(s, forecast=tuple(fc), trace=s.trace + (f"menu[{op.name}] x{fired}",))


@dataclass(frozen=True)
class CallAgent:
    """Orchestration primitive (agent-as-tool): the Decision agent invokes another agent
    (numerical / retrieval / ...) as a tool, possibly several times in a row. In evaluation the
    tool is a cached/stub callable from `state.tools`; live wiring is pluggable. Every call is
    logged in `state.calls` so the fitness can charge an EFFICIENCY cost (fewer, better-targeted
    calls) -- the invariant kernel still bounds whatever forecast results."""
    tool: str = "numerical"
    arg: str = ""

    def apply(self, s: ControllerState) -> ControllerState:
        s = replace(s, calls=s.calls + ((self.tool, self.arg),))
        fn = (s.tools or {}).get(self.tool)
        if fn is None:
            return replace(s, trace=s.trace + (f"call:{self.tool}(stub)",))
        try:
            s = fn(s, self.arg)            # tool returns an updated ControllerState
        except Exception as exc:           # a bad tool call must never break the kernel contract
            return replace(s, trace=s.trace + (f"call:{self.tool}(error:{type(exc).__name__})",))
        return replace(s, trace=s.trace + (f"call:{self.tool}",))


@dataclass(frozen=True)
class MethodBlend:
    """Context-conditioned blend of the base with ANOTHER numerical candidate (statistical or a
    different TSFM). Static blending of a strong base with a weaker method usually hurts, so this
    blends ONLY when a document signal is present -- i.e. the TEXT says something the base method
    may not see (a regime break) -- and otherwise leaves the strong base alone. This is the
    cross-modal 'let text decide which numerical method to trust' idea, kept coarse and gated;
    the invariant kernel still bounds the final. blend = (1-w)*forecast + w*other."""
    other: str = "seasonal_naive"
    weight: float = 0.3
    conf_min: float = 0.7           # only blend when the document signal clears this
    require_signal: bool = True     # gate on a document (CorDP/residual) signal being present

    def apply(self, s: ControllerState) -> ControllerState:
        alt = s.candidates.get(self.other)
        if not alt or len(alt) != len(s.forecast):
            return replace(s, trace=s.trace + (f"blend:no-candidate[{self.other}]",))
        signal = max(s.cordp_conf, s.residual_conf)
        if self.require_signal and signal < self.conf_min:
            return replace(s, trace=s.trace + ("blend:no-signal",))
        w = max(0.0, min(1.0, self.weight))
        fc = tuple((1.0 - w) * s.forecast[i] + w * float(alt[i]) for i in range(len(s.forecast)))
        return replace(s, forecast=fc, trace=s.trace + (f"blend[{self.other},w={w:.2f}]",))


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


INSTRUCTIONS = (SelectBase, RegimeAdjust, SemanticAdjust, PooledSemanticAdjust, DocAdjust,
                CorDPAdjust, ResidualAdjust, MenuAdjust, MethodBlend, CallAgent, NoOp, Regenerate)


@dataclass(frozen=True)
class Controller:
    steps: tuple = ()
    name: str = "controller"


def run_controller(controller: Controller, candidates: dict, effects: Sequence,
                   hv: Sequence, hts: Sequence, fts: Sequence, semantic_ref: str = "",
                   effect_pool: dict = None, cordp_conf: float = 0.0,
                   cordp_corrections: Sequence = (), residual_conf: float = 0.0,
                   residual_specs: Sequence = (), menu: Sequence = (), tools: dict = None):
    """Execute the instruction sequence; return (final_forecast, trace).

    Kernel: the result is always bounded against the selected base, so no orchestration
    can emit a catastrophic move; adjustment primitives only touch grounded effects.
    ``semantic_ref`` is the cached LLM regime choice used by SemanticAdjust.
    """
    default_base = tuple(float(x) for x in (candidates.get("toto_2_0")
                          or (next(iter(candidates.values())) if candidates else ())))
    s = ControllerState(candidates=dict(candidates), base=default_base, forecast=default_base,
                        effects=tuple(effects), hv=tuple(hv), hts=tuple(hts), fts=tuple(fts),
                        semantic_ref=semantic_ref, effect_pool=effect_pool or {},
                        cordp_conf=float(cordp_conf), cordp_corrections=tuple(cordp_corrections),
                        residual_conf=float(residual_conf), residual_specs=tuple(residual_specs),
                        menu=tuple(menu), tools=tools, calls=())
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
SEMANTIC_CONTROLLER = Controller(
    steps=(SelectBase(), SemanticAdjust(trusted=("weekend", "weekday"))),
    name="select+semantic-stable")
POOLED_CONTROLLER = Controller(
    steps=(SelectBase(), PooledSemanticAdjust(trusted=("weekend", "weekday"), local_weight=0.5)),
    name="select+pooled-semantic")
CORDP_CONTROLLER = Controller(
    steps=(SelectBase(), CorDPAdjust(conf_min=0.75, wfrac_max=0.3)),
    name="select+cordp-gated")
RESIDUAL_CONTROLLER = Controller(
    steps=(SelectBase(), ResidualAdjust(conf_min=0.75, wfrac_max=0.3)),
    name="select+residual-shaped")

SEED_CONTROLLERS = (IDENTITY_CONTROLLER, REGIME_CONTROLLER, CASCADE_CONTROLLER,
                    SEMANTIC_CONTROLLER, POOLED_CONTROLLER, CORDP_CONTROLLER,
                    RESIDUAL_CONTROLLER)


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
    if kind is SemanticAdjust:
        return SemanticAdjust(cap=_clamp(rng.uniform(0.2, KERNEL_MAX_FRAC)))
    if kind is PooledSemanticAdjust:
        return PooledSemanticAdjust(cap=_clamp(rng.uniform(0.2, KERNEL_MAX_FRAC)),
                                    local_weight=rng.choice((0.3, 0.5, 0.7)))
    if kind is DocAdjust:
        return DocAdjust(cap=_clamp(rng.uniform(0.1, 0.4)))
    if kind is CorDPAdjust:
        return CorDPAdjust(conf_min=rng.choice((0.6, 0.7, 0.75, 0.85)),
                           wfrac_max=rng.choice((0.2, 0.3, 0.5)),
                           shrink=rng.choice((0.5, 0.75, 1.0)))
    if kind is ResidualAdjust:
        return ResidualAdjust(conf_min=rng.choice((0.6, 0.7, 0.75, 0.85)),
                              wfrac_max=rng.choice((0.2, 0.3, 0.5)),
                              amp_cap=rng.choice((0.3, 0.5)),
                              shrink=rng.choice((0.5, 0.75, 1.0)))
    if kind is MethodBlend:
        return MethodBlend(other=rng.choice(("seasonal_naive", "statistical", "combined")),
                           weight=rng.choice((0.2, 0.3, 0.5)),
                           conf_min=rng.choice((0.6, 0.7, 0.8)))
    if kind is MenuAdjust:
        order = ["regime", "trend", "decay"]
        rng.shuffle(order)
        return MenuAdjust(prefer=tuple(order), cap=_clamp(rng.uniform(0.2, KERNEL_MAX_FRAC)))
    if kind is CallAgent:
        return CallAgent(tool=rng.choice(("numerical", "retrieval")),
                         arg=rng.choice(("", "refine", "round2")))
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
    if isinstance(step, SemanticAdjust):
        r = rng.random()
        if r < 0.4:
            return replace(step, cap=_clamp(step.cap + rng.uniform(-0.15, 0.15)))
        if r < 0.75:
            ref = rng.choice(_ALL_REFS)                   # toggle a ref in/out of trust
            tset = set(step.trusted); tset.symmetric_difference_update({ref})
            return replace(step, trusted=tuple(x for x in _ALL_REFS if x in tset))
        return replace(step, require_significant=not step.require_significant)
    if isinstance(step, PooledSemanticAdjust):
        r = rng.random()
        if r < 0.3:
            return replace(step, cap=_clamp(step.cap + rng.uniform(-0.15, 0.15)))
        if r < 0.5:
            return replace(step, local_weight=max(0.0, min(1.0, step.local_weight + rng.uniform(-0.3, 0.3))))
        if r < 0.75:
            ref = rng.choice(_ALL_REFS)
            tset = set(step.trusted); tset.symmetric_difference_update({ref})
            return replace(step, trusted=tuple(x for x in _ALL_REFS if x in tset))
        return replace(step, require_significant=not step.require_significant)
    if isinstance(step, DocAdjust):
        return replace(step, cap=_clamp(step.cap + rng.uniform(-0.1, 0.1), hi=0.5))
    if isinstance(step, CorDPAdjust):
        r = rng.random()
        if r < 0.4:
            return replace(step, conf_min=max(0.0, min(0.95, step.conf_min + rng.uniform(-0.15, 0.15))))
        if r < 0.7:
            return replace(step, wfrac_max=max(0.1, min(1.0, step.wfrac_max + rng.uniform(-0.15, 0.15))))
        if r < 0.9:
            return replace(step, shrink=max(0.2, min(1.0, step.shrink + rng.uniform(-0.25, 0.25))))
        return replace(step, mult_hi=max(1.2, min(2.5, step.mult_hi + rng.uniform(-0.3, 0.3))))
    if isinstance(step, MenuAdjust):
        if rng.random() < 0.5:
            order = list(step.prefer) or ["regime", "trend", "decay"]
            rng.shuffle(order)
            return replace(step, prefer=tuple(order))
        return replace(step, cap=_clamp(step.cap + rng.uniform(-0.15, 0.15)))
    if isinstance(step, CallAgent):
        if rng.random() < 0.5:
            return replace(step, tool="retrieval" if step.tool == "numerical" else "numerical")
        return replace(step, arg=rng.choice(("", "refine", "round2")))
    if isinstance(step, ResidualAdjust):
        r = rng.random()
        if r < 0.35:
            return replace(step, conf_min=max(0.0, min(0.95, step.conf_min + rng.uniform(-0.15, 0.15))))
        if r < 0.6:
            return replace(step, wfrac_max=max(0.1, min(1.0, step.wfrac_max + rng.uniform(-0.15, 0.15))))
        if r < 0.8:
            return replace(step, shrink=max(0.2, min(1.0, step.shrink + rng.uniform(-0.25, 0.25))))
        return replace(step, amp_cap=max(0.15, min(0.9, step.amp_cap + rng.uniform(-0.2, 0.2))))
    if isinstance(step, MethodBlend):
        r = rng.random()
        if r < 0.4:
            return replace(step, weight=max(0.0, min(1.0, step.weight + rng.uniform(-0.2, 0.2))))
        if r < 0.7:
            return replace(step, conf_min=max(0.0, min(0.95, step.conf_min + rng.uniform(-0.15, 0.15))))
        if r < 0.85:
            return replace(step, other=rng.choice(("seasonal_naive", "statistical", "combined")))
        return replace(step, require_signal=not step.require_signal)
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


def run_controller_evolution(seeds, fitness, *, generations: int = 30, pop_size: int = 40,
                             elite: int = 8, seed: int = 20260918, select_fitness=None):
    """(mu+lambda) evolution over controller instruction sequences.

    `fitness` maps a Controller to a scalar (higher better; use the CVaR objective).
    It drives evolution: it ranks the parents each generation. Ties break toward fewer
    instructions (simpler, more auditable controllers win).

    `select_fitness` (optional) is a SEPARATE held-out fitness used ONLY to pick the
    returned champion. When given, evolution still ranks/selects parents on `fitness`
    (the search fold), but the controller returned is the best-ever on `select_fitness`
    (the held-out fold) -- a global controller program can only be de-overfit by
    selecting on held-out data, not by in-sample fit. This mirrors evolve_N's
    rank-on-search / select-on-val protocol. When omitted, selection == `fitness`
    (backward compatible).

    Returns (best_controller, best_select_fitness, history).
    """
    rng = _random.Random(seed)
    pop = list(seeds)
    while len(pop) < pop_size:
        base = rng.choice(seeds) if seeds else Controller(steps=(SelectBase(),))
        pop.append(mutate_controller(base, rng))

    def key(c):
        return (fitness(c), -len(c.steps))

    sel = select_fitness or fitness

    def sel_key(c):
        return (sel(c), -len(c.steps))

    best = max(pop, key=sel_key)          # champion is best on the SELECTION fold
    history = []
    for gen in range(generations):
        ranked = sorted(pop, key=key, reverse=True)   # evolution ranks on the SEARCH fold
        parents = ranked[:elite]
        cur = max(pop, key=sel_key)                    # per-gen champion on held-out
        if sel_key(cur) > sel_key(best):
            best = cur
        history.append((gen, fitness(ranked[0]), sel(cur)))
        children = list(parents)
        while len(children) < pop_size:
            if rng.random() < 0.5 and len(parents) >= 2:
                child = crossover_controllers(*rng.sample(parents, 2), rng)
            else:
                child = rng.choice(parents)
            children.append(mutate_controller(child, rng))
        pop = children
    return best, sel(best), history
