"""Unified numerical step v2 (same code for Dr-CiK / Time-MMD / TimesX). Members: Toto, TimesFM, seasonal
(value one season ago; season from frequency), plus shrink s toward the last value.
Weights on a 0.1 simplex grid x s in {0,.1,.2,.3}. Grouping in {global, domain, freq, var}; a group uses its own
config only with >= MIN train tasks, else the parent (var -> domain -> global). Grouping chosen by 5-fold CV on TRAIN."""
import json, math, random, itertools, sys
import numpy as np
sys.path.insert(0, "."); import unified_numeric as U
W = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
GRID = [(w, s) for w in W for s in (0.0, 0.1, 0.2, 0.3)]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 8

def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f or f.endswith("t") or f.endswith("s"): return None
    if "hour" in f or f.endswith("h"): return 24
    if "week" in f or f.endswith("w"): return 52
    if "month" in f or f.endswith("m"): return 12
    if "day" in f or "daily" in f or f.endswith("d"): return 7
    return None

name = sys.argv[1]; T = U.load(name); FORCE = sys.argv[3] if len(sys.argv) > 3 else None
for t in T:
    h = t["history"]; H = len(t["truth"]); p = season(t["freq"])
    t["sea"] = [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]
    t["var"] = t["tid"].rsplit("_", 1)[0] if name == "timesx" else t["domain"]
    M3 = np.array([t["toto"], t["tfm"], t["sea"]]); y = np.array(t["truth"]); last = h[-1]
    E = []
    for (w, s) in GRID:
        f = (1 - s) * (np.array(w) @ M3) + s * last
        m = U.M(list(y), list(f), cap=5.0); E.append(m["smae"] + m["srmse"])
    t["E"] = np.array(E); t["sm"] = None
IDX = {g: i for i, g in enumerate(GRID)}
sys.path.insert(0, U.X); import metrics as PM
na, nb = [], []

def best(ts): return int(np.argmin(sum(t["E"] for t in ts))) if ts else None
def fit(ts, grouping):
    g0 = best(ts); P = {"__g": g0}
    levels = {"global": [], "domain": ["domain"], "freq": ["freq"], "var": ["domain", "var"]}[grouping]
    for lv in levels:
        for key in {t[lv] for t in ts}:
            sub = [t for t in ts if t[lv] == key]
            if len(sub) >= MIN: P[(lv, key)] = best(sub)
    def pick(t):
        c = g0
        for lv in levels:
            c = P.get((lv, t[lv]), c)
        return c
    return pick
def cv(ts, grouping, k=5):
    idx = list(range(len(ts))); random.Random(0).shuffle(idx); tot = 0
    for f in range(k):
        te = set(idx[f::k]); P = fit([ts[i] for i in idx if i not in te], grouping)
        tot += sum(ts[i]["E"][P(ts[i])] for i in te)
    return tot / len(ts)
tr = [t for t in T if t["part"] == "train"]
groupings = ["global", "domain", "freq"] + (["var"] if name == "timesx" else [])
sc = {g: cv(tr, g) for g in groupings}; bg = FORCE or min(sc, key=sc.get); P = fit(tr, bg)
print(f"== {name} v2 MIN={MIN}: CV {', '.join(f'{g} {v:.4f}' for g, v in sc.items())} -> {bg}; global cfg {GRID[best(tr)]}")
for part in sorted({t["part"] for t in T}):
    ts = [t for t in T if t["part"] == part]
    for n, f in (("Toto", lambda t: IDX[((1.0, 0.0, 0.0), 0.0)]), ("TimesFM", lambda t: IDX[((0.0, 1.0, 0.0), 0.0)]), ("v2", P)):
        sm = sr = 0
        for t in ts:
            w, s = GRID[f(t)]; fc = (1 - s) * (np.array(w) @ np.array([t["toto"], t["tfm"], t["sea"]])) + s * t["history"][-1]
            m = U.M(t["truth"], list(fc), cap=5.0); sm += m["smae"] / len(ts); sr += m["srmse"] / len(ts)
            if name == "timesx":
                a, b = PM.paper(t["truth"], list(fc), PM.snaive(t["history"], len(t["truth"]), t["freq"]))
                if a is not None: na.append(a)
                if b is not None: nb.append(b)
        ex = f" nMAE {sum(na)/len(na):.3f} nMSE {sum(nb)/len(nb):.3f}" if na else ""; na.clear(); nb.clear()
        print(f"   {part:9s} {n:8s} n={len(ts):4d} sMAE {sm:.4f} sRMSE {sr:.4f}{ex}")
