"""Unified numerical step for Dr-CiK, Time-MMD and TimesX (same code, same search space; only the data changes).
Forecast = (1-s) * (w*Toto + (1-w)*TimesFM) + s * last_observation.
Search space: grouping in {global, domain, freq}; per group (w, s) on a grid. Groups with < MIN train tasks use the
global (w, s). Grouping chosen by 5-fold cross-validation on TRAIN only; params refit on all of train; dev is a check;
test is reported once. Objective = mean joint error (sMAE + sRMSE, cap 5)."""
import json, math, random, itertools, sys, statistics
sys.path.insert(0, ".")
from common.metrics import drcik_point_metrics as M
S = ".scratch/self_evolving/"; X = "work/timesx/"
GRID = list(itertools.product([i / 10 for i in range(11)], [0.0, 0.1, 0.2, 0.3])); MIN = 8

def ok(v): return all(x is not None and math.isfinite(x) for x in v)
def load(name):
    out = []
    if name == "drcik":
        TF = json.load(open(S + "timesfm_full.json"))
        for d in json.load(open(S + "nrd_cache.json")):
            dom, fq = d["group"].split("|")
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq,
                            history=d["history"], truth=d["truth"], toto=d["fc"]["toto_2_0"], tfm=TF[d["tid"]]))
    elif name == "tmmd":
        TF = json.load(open("tmmd_timesfm.json"))
        for d in json.load(open(S + "tmmd_cache.json")):
            dom, fq = d["group"].split("|")
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq,
                            history=d["history"], truth=d["truth"], toto=d["fc"]["toto_2_0"], tfm=TF[d["tid"]]))
    else:
        TO = json.load(open(X + "toto.json")); TF = json.load(open(X + "timesfm.json"))
        for t in json.load(open(X + "tasks.json")):
            out.append(dict(tid=t["tid"], part={"test_id": "test"}.get(t["part"], t["part"]), domain=t["domain"], freq=t["freq"],
                            history=t["history"], truth=t["truth"], toto=TO[t["tid"]], tfm=TF[t["tid"]]))
    out = [t for t in out if ok(t["truth"]) and ok(t["toto"]) and ok(t["tfm"]) and ok([x for x in t["history"][-1:]])]
    for t in out: t["cache"] = {}
    return out

def fc(t, w, s):
    last = t["history"][-1]; return [(1 - s) * (w * a + (1 - w) * b) + s * last for a, b in zip(t["toto"], t["tfm"])]
def err(t, w, s):
    k = (w, s)
    if k not in t["cache"]: m = M(t["truth"], fc(t, w, s), cap=5.0); t["cache"][k] = (m["smae"], m["srmse"])
    return t["cache"][k]
def mj(ts, w, s): return sum(sum(err(t, w, s)) for t in ts) / len(ts)
def fit(ts, grouping):
    g0 = min(GRID, key=lambda g: mj(ts, *g)); P = {}
    if grouping != "global":
        for key in {t[grouping] for t in ts}:
            sub = [t for t in ts if t[grouping] == key]
            if len(sub) >= MIN: P[key] = min(GRID, key=lambda g: mj(sub, *g))
    return lambda t: g0 if grouping == "global" else P.get(t[grouping], g0)
def cv(ts, grouping, k=5, seed=0):
    idx = list(range(len(ts))); random.Random(seed).shuffle(idx); tot = 0
    for f in range(k):
        te = [ts[i] for i in idx[f::k]]; tr = [ts[i] for i in idx if i not in set(idx[f::k])]
        P = fit(tr, grouping); tot += sum(sum(err(t, *P(t))) for t in te)
    return tot / len(ts)

if __name__ == "__main__":
    name = sys.argv[1]; T = load(name); tr = [t for t in T if t["part"] == "train"]
    scores = {g: cv(tr, g) for g in ("global", "domain", "freq")}; best = min(scores, key=scores.get)
    P = fit(tr, best)
    print(f"== {name}: n={len(T)} train={len(tr)}  CV joint {', '.join(f'{g} {v:.4f}' for g, v in scores.items())} -> {best}")
    print("   params:", {k: P(next(t for t in tr if t[best] == k)) for k in sorted({t[best] for t in tr})} if best != "global" else P(tr[0]))
    res = {}
    for part in sorted({t["part"] for t in T}):
        ts = [t for t in T if t["part"] == part]
        for n, f in (("Toto", lambda t: (1.0, 0.0)), ("TimesFM", lambda t: (0.0, 0.0)), ("unified", P)):
            sm = sum(err(t, *f(t))[0] for t in ts) / len(ts); sr = sum(err(t, *f(t))[1] for t in ts) / len(ts)
            res[f"{part}/{n}"] = (sm, sr); print(f"   {part:9s} {n:8s} n={len(ts):4d} sMAE {sm:.4f} sRMSE {sr:.4f}")
    json.dump(dict(grouping=best, cv=scores, results=res), open(f"unified_numeric_{name}.json", "w"), indent=1)
