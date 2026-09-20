"""MathMenu: the Numerical agent's second job -- besides a forecast, expose a per-task MENU of
applicable MATHEMATICAL operations (with data-computed parameters) that the Decision agent may
invoke. This realizes "numerical lists the math methods relevant to this task; the decision
agent, when correcting, actually DOES the math" -- and it grounds magnitude in DATA (the menu's
precomputed values) while text only SELECTS which op, keeping the number reliable and auditable.

Each MathOp is a small, named, parameterized transform of a forecast window. The menu is built
by analyzing the series (seasonal levels, trend, recent/overall level). MenuAdjust (in
controller.py) picks an op conditioned on the document evidence and applies it under the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Optional

from .post_adjust import _parse


@dataclass(frozen=True)
class MathOp:
    """One task-applicable math operation. `kind` selects the transform; `params` carries the
    data-computed numbers. `target` (level ops) or `factor` (scale) is the grounded magnitude."""
    name: str                       # e.g. "regime:weekend", "trend", "recent"
    kind: str                       # level | scale | trend | decay
    target: Optional[float] = None  # absolute level for kind=level/decay/trend anchor
    factor: Optional[float] = None  # multiplicative factor for kind=scale
    slope: float = 0.0              # per-step slope for kind=trend
    tau_frac: float = 0.5           # relaxation for kind=decay
    source: str = "data"            # provenance for the audit trail (which computation)


def _weekday(ts) -> int:
    p = _parse(ts)
    return p.weekday() if p else -1


def _regime_level(hv, hts, ref: str) -> Optional[float]:
    """Data-computed level for a named regime (the grounded magnitude source)."""
    if ref == "weekend":
        v = [x for x, t in zip(hv, hts) if _weekday(t) >= 5]
        return mean(v) if v else None
    if ref == "weekday":
        v = [x for x, t in zip(hv, hts) if 0 <= _weekday(t) < 5]
        return mean(v) if v else None
    if ref in ("low_day", "high_day"):
        bd: dict = {}
        for x, t in zip(hv, hts):
            p = _parse(t)
            if p:
                bd.setdefault(p.date(), []).append(x)
        dm = sorted(mean(z) for z in bd.values()) if bd else []
        return (dm[0] if ref == "low_day" else dm[-1]) if dm else None
    if ref == "recent":
        return mean(hv[-min(len(hv), 7):]) if hv else None
    if ref == "overall":
        return mean(hv) if hv else None
    return None


_REGIME_REFS = ("weekend", "weekday", "low_day", "high_day", "recent", "overall")


def build_math_menu(hv, hts, fts) -> tuple:
    """Numerical agent -> the menu of math ops for THIS task, with data-computed params.
    Only operations whose parameters are computable are listed (applicability = the number
    exists). Returns a tuple of MathOp."""
    hv = list(hv); hts = list(hts)
    if not hv:
        return ()
    ops = []
    for ref in _REGIME_REFS:
        lvl = _regime_level(hv, hts, ref)
        if lvl is not None:
            ops.append(MathOp(name=f"regime:{ref}", kind="level", target=lvl,
                              source=f"mean of {ref} history"))
    # trend: slope over the recent window, projected forward
    k = min(len(hv), 14)
    if k >= 3:
        recent = hv[-k:]
        slope = (recent[-1] - recent[0]) / (k - 1)
        ops.append(MathOp(name="trend", kind="trend", target=recent[-1], slope=slope,
                          source=f"recent {k}-step slope"))
    # decay-back-to-normal: relax from an offset toward the overall level
    overall = mean(hv)
    ops.append(MathOp(name="decay_to_normal", kind="decay", target=overall, tau_frac=0.5,
                      source="relax toward overall mean"))
    return tuple(ops)


def op_window_values(op: MathOp, base_window, ref_scale: float, factor_override: Optional[float] = None):
    """Apply a MathOp to the base window (a list of base forecast values), returning the new
    window values. `ref_scale` (history scale) floors the anchor so base~=0 can't blow up.
    `factor_override` lets the decision scale the op (e.g. from the document) toward the target."""
    n = len(base_window)
    if n == 0:
        return []
    bw = mean(base_window)
    anchor = abs(bw) if abs(bw) > 1e-9 else (ref_scale or 1.0)
    out = list(base_window)
    if op.kind == "level" and op.target is not None:
        # move each step toward the target level (optionally partway via factor_override)
        frac = 1.0 if factor_override is None else max(0.0, min(1.0, factor_override))
        for i in range(n):
            out[i] = base_window[i] + frac * (op.target - bw)
    elif op.kind == "scale" and (op.factor is not None or factor_override is not None):
        f = factor_override if factor_override is not None else op.factor
        for i in range(n):
            out[i] = base_window[i] * f
    elif op.kind == "trend" and op.target is not None:
        for i in range(n):
            out[i] = op.target + op.slope * (i + 1)
    elif op.kind == "decay" and op.target is not None:
        import math
        tau = max(0.05, min(1.0, op.tau_frac))
        start = base_window[0]
        for i in range(n):
            x = i / (n - 1) if n > 1 else 0.0
            out[i] = op.target + (start - op.target) * math.exp(-x / tau)
    return out
