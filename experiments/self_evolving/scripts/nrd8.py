"""nrd4 + evolvable window/direction transform in the Retrieval artifact (2026-09-27).
Diagnosis showed 57% of harmful corrections are window/timing errors and 28% direction errors.
Retrieval now also evolves a transform applied to every extracted correction before validation:
(window op in {id, shift k, head half, tail half, grow k}, magnitude scale, direction sign).
Everything else (Numerical trust/calibration features, scorer self-evolution, Decision) as nrd4."""
import sys, random, copy, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd4 as N4
import nrd_coevolve as C

TR = [("id", 0)] + [("shift", k) for k in (-3, -2, -1, 1, 2, 3)] + [("head", 0), ("tail", 0), ("grow", 1), ("grow", 2)]


def tw(s, e, H, t):
    kind, arg = t; L = e - s
    if kind == "shift": s2, e2 = s + arg, e + arg
    elif kind == "head": s2, e2 = s, s + max(1, L // 2)
    elif kind == "tail": s2, e2 = e - max(1, L // 2), e
    elif kind == "grow": s2, e2 = s - arg, e + arg
    else: s2, e2 = s, e
    s2, e2 = max(0, s2), min(H, e2)
    return (s2, e2) if e2 > s2 else None


def transformed(team, d):
    wt = team["retrieval"].get("_wt", [("id", 0), 1.0, 1])
    out = []
    for s, e, m in d["corr"]:
        w = tw(s, e, d["H"], tuple(wt[0]))
        if w: out.append((w[0], w[1], 1 + wt[2] * wt[1] * (m - 1)))
    return out


def evidence(team, d):
    out = []
    for s, e, m in transformed(team, d):
        a = N4.accept(team["retrieval"], N4.feats(team, d, s, e, m))
        if a > 0: out.append((s, e, 1 + a * (m - 1)))
    return out


def scorer_credit(team, w, ds):
    t = dict(team); t["retrieval"] = w; g = 0.0
    for d in ds:
        base = d["fc"][N4.ANCHOR]
        for s, e, m in transformed(t, d):
            a = N4.accept(w, N4.feats(t, d, s, e, m))
            if a <= 0: continue
            out = list(N4.apply_bounded_delta(base, N4.R3.apply_ev(base, [(s, e, 1 + a * (m - 1))], 1.0)))
            x = d["base_jt"] - C.jt(out, d["truth"]); g += x if x > 0 else 1.5 * x
    return g


N4.evidence = evidence
N4.scorer_credit = scorer_credit
_orig_evolve_scorer = N4.evolve_scorer


def evolve_scorer(team, train, rng, gens=15, pop=16, elite=4, feats_allowed=N4.FEATS):
    """Retrieval self-evolution: try the current transform plus 3 mutated transforms, evolve the
    validator weights under each, keep the best by Train per-correction credit."""
    cur_wt = team["retrieval"].get("_wt", [("id", 0), 1.0, 1])
    cands = [cur_wt] + [[rng.choice(TR), rng.choice([1.0, 0.75, 0.5]), rng.choice([1, 1, -1])] for _ in range(3)]
    best = None
    for wt in cands:
        t = copy.deepcopy(team); t["retrieval"] = dict(team["retrieval"]); t["retrieval"]["_wt"] = wt
        w, curve = _orig_evolve_scorer(t, train, rng, gens=gens, pop=pop, elite=elite, feats_allowed=feats_allowed)
        w = dict(w); w["_wt"] = wt
        sc = scorer_credit(t, w, [d for d in train if d["corr"]])
        if best is None or sc > best[0]: best = (sc, w, curve)
    return best[1], best[2]


N4.evolve_scorer = evolve_scorer
_orig_accept = N4.accept
N4.accept = lambda w, f: _orig_accept({k: v for k, v in w.items() if k != "_wt"}, f)

if __name__ == "__main__":
    N4.main()
