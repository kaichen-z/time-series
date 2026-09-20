"""Guarded per-task numerical selector (Loop N), do-no-harm by construction.

The old Loop N applied ONE global weight vector to every task -> it moved all 99
tasks off Toto and regressed the ~82 where Toto was already best (whole-99: sMAE
-3.57%, 67 regressions). This replaces it with a per-task guard that DEFAULTS to
Toto and only deviates when a pure-Python statistical method demonstrates skill on
THIS task's own causal hindcast folds AND materially disagrees with Toto -- and even
then only inside a bounded envelope, so a wrong call cannot do much harm.

Constraint that shapes the design: the champion ForecastStore is cache-only, so Toto
exists ONLY at full history -- it CANNOT be hindcast on prefixes. So the guard cannot
compare "S vs Toto on folds"; instead it validates S against a naive-persistence
baseline on the task's own history (which is cheap and causal), and treats Toto as the
anchor that S must beat naive by a margin to be trusted at all. Every threshold here
is evolvable; the defaults fire on very few tasks (Toto-like).

The eligible methods are exactly the pure-Python ones (they can be replayed on any
history prefix); seasonal_naive/Toto come from the cache and cannot be folded, so they
are never guard candidates.
"""
from __future__ import annotations
from statistics import mean

from scripts.stat_library import LIBRARY

GUARD_KEYS = ("rel_min", "dev_min", "w", "wcap")
GUARD_BOUNDS = {
    "rel_min": (0.0, 0.6),   # min causal-fold skill (vs naive) for S to be trusted
    "dev_min": (0.0, 0.4),   # min relative disagreement between S and Toto (else nothing to gain)
    "w":       (0.0, 0.6),   # blend weight toward S
    "wcap":    (0.05, 0.5),  # hard do-no-harm envelope: |base - Toto| <= wcap*|Toto|
}
# Default seed fires rarely (Toto-like); plus a "never fire" == pure Toto seed.
GUARD_SEED = {"rel_min": 0.2, "dev_min": 0.08, "w": 0.3, "wcap": 0.25}
GUARD_TOTO = {"rel_min": 1.0, "dev_min": 1.0, "w": 0.0, "wcap": 0.05}  # == pure Toto


def clamp_params(p: dict) -> dict:
    return {k: max(GUARD_BOUNDS[k][0], min(GUARD_BOUNDS[k][1], float(p.get(k, GUARD_SEED[k]))))
            for k in GUARD_KEYS}


def _relmae(truth, fc) -> float:
    denom = sum(abs(x) for x in truth) or 1e-9
    return sum(abs(t - f) for t, f in zip(truth, fc)) / denom


def make_folds(hv, L: int = 8, n: int = 3, min_prefix: int = 12):
    """Causal hindcast folds from a series' own history: predict the last L points from
    the preceding prefix, stepping back n times. No future, no Toto -- pure and cheap."""
    folds = []
    for k in range(n):
        end = len(hv) - k * L
        start = end - L
        if start - min_prefix < 0:
            break
        folds.append((tuple(hv[:start]), tuple(hv[start:end])))
    return folds


def fold_skill(name, folds) -> float:
    """Skill of a pure-Python method vs naive persistence on the task's own folds.
    >0 means it beats naive; this is the only 'is this model right for this series'
    signal available without Toto-on-prefix."""
    fn = LIBRARY[name]
    errs, nerrs = [], []
    for prefix, target in folds:
        L = len(target)
        errs.append(_relmae(target, fn(list(prefix), L)))
        naive = [prefix[-1]] * L if prefix else [0.0] * L
        nerrs.append(_relmae(target, naive))
    e, ne = mean(errs), mean(nerrs)
    return (ne - e) / (ne + 1e-9)


def guarded_base_from_cand(toto, cand: dict, hv, params: dict):
    """Per-task base. Default = Toto. Deviate toward the most fold-skilled pure-Python
    method S only when S beats naive by >= rel_min on this series' folds AND disagrees
    with Toto by >= dev_min; the deviation is a bounded blend clipped to +/- wcap*|Toto|."""
    p = clamp_params(params)
    H = len(toto)
    folds = make_folds(hv)
    if not folds:
        return tuple(toto)
    # eligible = pure-Python methods present as full-horizon candidates
    elig = [m for m in LIBRARY if m in cand and len(cand[m]) == H]
    if not elig:
        return tuple(toto)
    best, best_sk = None, -1e9
    for m in elig:
        sk = fold_skill(m, folds)
        if sk > best_sk:
            best, best_sk = m, sk
    if best_sk < p["rel_min"]:
        return tuple(toto)                       # no reliable local model -> keep Toto
    S = cand[best]
    scale = (mean(abs(x) for x in toto) or 1e-9)
    dev = mean(abs(S[i] - toto[i]) for i in range(H)) / scale
    if dev < p["dev_min"]:
        return tuple(toto)                       # S agrees with Toto -> nothing to gain
    w, cap = p["w"], p["wcap"]
    out = []
    for i in range(H):
        delta = w * (S[i] - toto[i])
        lim = cap * abs(toto[i])
        out.append(toto[i] + max(-lim, min(lim, delta)))   # do-no-harm envelope
    return tuple(out)


def guarded_base(d: dict, params: dict):
    """Convenience over a coevolve data row (needs d['cand'] and d['hv'])."""
    return guarded_base_from_cand(d["cand"]["toto_2_0"], d["cand"], d["hv"], params)
