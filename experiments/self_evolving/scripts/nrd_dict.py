"""Numerical dictionary self-evolution for the N/R/D engine (2026-09-26).

A dictionary member is a typed, executable forecasting program built from a small grammar.
Numerical sees history only: every member is scored by rolling-origin hindcasts inside each task's
own history (the last H points, and the H before them), never by future labels.  A MAP-Elites
archive keyed by (frequency class, morphology bucket) keeps the best member per cell; the archive
*is* the dictionary handed to Decision.  Gen0 is a fixed set of textbook members.
"""
from __future__ import annotations
import copy, json, math, random, statistics
import numpy as np
from common.metrics import drcik_point_metrics

CAP = 5.0
PRIMS = ["naive", "snaive", "drift", "ses", "damped", "theta", "mean", "linreg", "sprofile"]


def jt(f, t):
    x = drcik_point_metrics(t, f, cap=CAP)
    return x["smae"] + x["srmse"]


def period_of(freq):
    f = (freq or "").lower()
    if "min" in f and "5" in f: return 12
    if "min" in f: return 60
    if "hour" in f or f in ("h", "1h"): return 24
    if "day" in f or f in ("d", "1d"): return 7
    if "week" in f: return 52
    if "month" in f: return 12
    return 1


def freq_class(freq):
    f = (freq or "").lower()
    for k in ("second", "minute", "hour", "day", "week", "month"):
        if k in f: return k
    return "other"


# ----------------------------------------------------------------------------- member execution
def _ses(y, a):
    l = y[0]
    for v in y[1:]: l = a * v + (1 - a) * l
    return l


def run_prim(g, y, H, per):
    y = np.asarray(y, float); n = len(y); k = g["k"]
    p = int(g.get("p", 0) or per)
    if k == "naive": return np.full(H, y[-1])
    if k == "snaive":
        if p <= 1 or n < p: return np.full(H, y[-1])
        return np.array([y[n - p + (i % p)] for i in range(H)])
    if k == "drift":
        w = max(2, min(n, int(g.get("w", n)))); s = (y[-1] - y[-w]) / (w - 1)
        return y[-1] + s * np.arange(1, H + 1)
    if k == "ses": return np.full(H, _ses(y, g.get("a", 0.3)))
    if k == "damped":
        a, b, phi = g.get("a", 0.3), g.get("b", 0.1), g.get("phi", 0.9)
        l, t = y[0], (y[1] - y[0]) if n > 1 else 0.0
        for v in y[1:]:
            l2 = a * v + (1 - a) * (l + phi * t); t = b * (l2 - l) + (1 - b) * phi * t; l = l2
        return np.array([l + t * sum(phi ** j for j in range(1, i + 1)) for i in range(1, H + 1)])
    if k == "theta":
        a = g.get("a", 0.3); x = np.arange(n); s = np.polyfit(x, y, 1)[0] if n > 2 else 0.0
        return np.full(H, _ses(y, a)) + 0.5 * s * np.arange(1, H + 1)
    if k == "mean":
        w = max(1, min(n, int(g.get("w", n)))); return np.full(H, y[-w:].mean())
    if k == "linreg":
        w = max(3, min(n, int(g.get("w", n)))); x = np.arange(w)
        c = np.polyfit(x, y[-w:], 1); return np.polyval(c, np.arange(w, w + H))
    if k == "sprofile":   # seasonal profile around an SES level
        if p <= 1 or n < 2 * p: return np.full(H, _ses(y, g.get("a", 0.3)))
        cyc = int(g.get("c", 3)); m = min(cyc, n // p); z = y[-m * p:].reshape(m, p)
        prof = z.mean(0) - z.mean(); lvl = _ses(y, g.get("a", 0.3)) - prof[(n - 1) % p] if g.get("anchor", True) else z.mean()
        return np.array([lvl + prof[(n + i) % p] for i in range(H)])
    raise ValueError(k)


def run_member(m, y, H, freq):
    per = period_of(freq)
    if m["op"] == "prim": out = run_prim(m["g"], y, H, per)
    elif m["op"] == "blend":
        w = m["w"]; out = w * run_member(m["a"], y, H, freq) + (1 - w) * run_member(m["b"], y, H, freq)
    elif m["op"] == "route":   # history-only router on seasonality strength
        s = seas_strength(y, per); out = run_member(m["a"] if s >= m["t"] else m["b"], y, H, freq)
    else: raise ValueError(m["op"])
    out = np.asarray(out, float)
    if m.get("clip"):
        lo, hi = float(np.min(y)), float(np.max(y)); r = hi - lo
        out = np.clip(out, lo - m["clip"] * r, hi + m["clip"] * r)
    if not np.all(np.isfinite(out)): out = np.full(H, float(y[-1]))
    return out


def seas_strength(y, p):
    y = np.asarray(y, float)
    if p <= 1 or len(y) < 2 * p: return 0.0
    d = y[p:] - y[:-p]; v = np.var(y) + 1e-12
    return float(max(0.0, 1 - np.var(d) / (2 * v)))


def morph(d):
    per = period_of(d["freq"]); s = seas_strength(d["history"], per)
    return f"{freq_class(d['freq'])}|{'seas' if s > 0.4 else 'flat'}"


# ----------------------------------------------------------------------------- history-only hindcasts
def origins(d):
    h, H = d["history"], d["H"]; out = []
    for k in (1, 2):
        if len(h) - k * H >= max(H, 24): out.append((h[:len(h) - k * H], h[len(h) - k * H:len(h) - (k - 1) * H]))
    return out


def hind_err(m, d):
    es = []
    for past, fut in origins(d):
        try: es.append(jt(list(run_member(m, past, d["H"], d["freq"])), list(fut)))
        except Exception: es.append(10.0)
    return statistics.mean(es) if es else None


# ----------------------------------------------------------------------------- mutation grammar
def rand_prim(rng):
    k = rng.choice(PRIMS); g = {"k": k}
    if k in ("ses", "theta", "sprofile", "damped"): g["a"] = round(rng.uniform(0.05, 0.9), 3)
    if k == "damped": g["b"] = round(rng.uniform(0.01, 0.4), 3); g["phi"] = round(rng.uniform(0.7, 0.99), 3)
    if k in ("drift", "mean", "linreg"): g["w"] = rng.choice([4, 8, 12, 24, 48, 96, 10 ** 6])
    if k == "sprofile": g["c"] = rng.choice([1, 2, 3, 5]); g["anchor"] = rng.random() < 0.7
    return {"op": "prim", "g": g}


def gen0_members():
    base = [{"k": "naive"}, {"k": "snaive"}, {"k": "drift"}, {"k": "ses", "a": 0.3}, {"k": "damped"},
            {"k": "theta", "a": 0.3}, {"k": "mean", "w": 24}, {"k": "linreg", "w": 24}, {"k": "sprofile", "a": 0.3, "c": 3}]
    return [{"op": "prim", "g": g} for g in base]


def mutate_member(m, pool, rng):
    op = rng.choice(["param", "param", "prim", "blend", "route", "clip", "unwrap"])
    c = copy.deepcopy(m)
    if op == "param":
        leaves = []
        def walk(x):
            if x["op"] == "prim": leaves.append(x)
            else: walk(x["a"]); walk(x["b"])
        walk(c); g = rng.choice(leaves)["g"]
        for key in ("a", "b", "phi"):
            if key in g: g[key] = round(min(0.99, max(0.01, g[key] + rng.gauss(0, 0.12))), 3)
        if "w" in g: g["w"] = rng.choice([4, 8, 12, 24, 48, 96, 10 ** 6])
        if "c" in g: g["c"] = rng.choice([1, 2, 3, 5])
        return c, "param"
    if op == "prim": return rand_prim(rng), "new-prim"
    if op == "blend": return {"op": "blend", "a": c, "b": copy.deepcopy(rng.choice(pool)), "w": round(rng.uniform(0.2, 0.8), 2)}, "blend"
    if op == "route": return {"op": "route", "a": c, "b": copy.deepcopy(rng.choice(pool)), "t": round(rng.uniform(0.2, 0.7), 2)}, "route"
    if op == "clip": c["clip"] = rng.choice([0.0, 0.1, 0.25, 0.5, None]); return c, "clip"
    if c["op"] != "prim": return copy.deepcopy(rng.choice([c["a"], c["b"]])), "unwrap"
    return rand_prim(rng), "new-prim"


def size(m): return 1 if m["op"] == "prim" else 1 + size(m["a"]) + size(m["b"])


def key(m): return json.dumps(m, sort_keys=True)


# ----------------------------------------------------------------------------- MAP-Elites archive
class Archive:
    """cell -> (member, mean hindcast error relative to Toto on that cell's train tasks)."""

    def __init__(self, tasks, toto_hind):
        self.tasks = [d for d in tasks if origins(d) and d["tid"] in toto_hind]
        self.cells = {}
        for d in self.tasks: self.cells.setdefault(morph(d), []).append(d)
        self.toto = {d["tid"]: toto_hind[d["tid"]] for d in self.tasks}
        self.elite = {}      # cell -> (score, member)
        self.cache = {}

    def score(self, m):
        k = key(m)
        if k not in self.cache:
            per = {}
            for cell, ds in self.cells.items():
                rel = []
                for d in ds:
                    e = hind_err(m, d)
                    if e is not None: rel.append(e - self.toto[d["tid"]])
                per[cell] = statistics.mean(rel) if rel else None
            self.cache[k] = per
        return self.cache[k]

    def offer(self, m):
        if size(m) > 5: return 0
        ins = 0
        for cell, s in self.score(m).items():
            if s is None: continue
            s = s + 0.002 * size(m)          # parsimony
            if cell not in self.elite or s < self.elite[cell][0] - 1e-9:
                self.elite[cell] = (s, m); ins += 1
        return ins

    def members(self):
        seen, out = set(), []
        for _c, (_s, m) in sorted(self.elite.items()):
            if key(m) not in seen: seen.add(key(m)); out.append(m)
        return out

    def summary(self):
        return {c: round(s, 4) for c, (s, _m) in sorted(self.elite.items())}


def evolve_dictionary(tasks, toto_hind, gens, children, rng, log=None):
    A = Archive(tasks, toto_hind)
    for m in gen0_members(): A.offer(m)
    curve = [dict(gen=0, n=len(A.members()), mean_rel=statistics.mean(s for s, _ in A.elite.values()))]
    for g in range(1, gens + 1):
        pool = A.members(); ins = 0; ops = {}
        for _ in range(children):
            child, op = mutate_member(rng.choice(pool), pool, rng)
            n = A.offer(child); ins += n
            if n: ops[op] = ops.get(op, 0) + 1
        curve.append(dict(gen=g, n=len(A.members()), inserted=ins, ops=ops,
                          mean_rel=statistics.mean(s for s, _ in A.elite.values())))
        if log: log(f"  dict gen {g:2d}: members {len(A.members())} insertions {ins} mean hindcast err vs Toto {curve[-1]['mean_rel']:+.4f} {ops}")
    return A, curve
