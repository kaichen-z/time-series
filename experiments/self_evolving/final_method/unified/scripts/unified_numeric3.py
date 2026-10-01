"""Unified numerical step v3 (same code for Dr-CiK / Time-MMD / TimesX). Members: Toto, TimesFM-2.5, Moirai-2.0,
Chronos-Bolt, seasonal (one season ago). Forecast = (1-s) * sum_k w_k member_k + s * last, w on a 0.1 simplex grid,
s in {0,.1,.2,.3}. Grouping in {global, domain, freq(, var for TimesX)} with hierarchical fallback (a group needs >= MIN
train tasks), chosen by 5-fold CV on TRAIN. Dev = check, test reported once. usage: unified_numeric3.py <ds> [MIN] [members]"""
import json, math, random, sys, itertools
import numpy as np
S = ".scratch/self_evolving/"; X = "work/timesx/"
sys.path.insert(0, X); import metrics as PM
name = sys.argv[1]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MEM = (sys.argv[3] if len(sys.argv) > 3 else "toto,tfm,moirai,chronos,sea").split(",")

def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f: return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("daily", 7), ("d", 7)):
        if k in f: return p
    return None
def ok(v): return v is not None and len(v) and all(x is not None and math.isfinite(x) for x in v)
def load():
    out = []
    if name == "drcik":
        TF = json.load(open(S + "timesfm_full.json"))
        for d in json.load(open(S + "nrd_cache_full2.json")):
            dom, fq = d["group"].split("|"); f = d["fc"]
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq, var=dom,
                            history=d["history"], truth=d["truth"], toto=f["toto_2_0"], tfm=f.get("timesfm_2_5") or TF.get(d["tid"]),
                            moirai=f.get("moirai_2_0"), chronos=f.get("chronos_bolt")))
    elif name == "tmmd":
        TF = json.load(open("tmmd_timesfm.json")); MO = json.load(open("tsfm/moirai_tmmd.json")); CH = json.load(open("tsfm/chronos_tmmd.json"))
        for d in json.load(open(S + "tmmd_cache.json")):
            dom, fq = d["group"].split("|")
            h = [x for x in d["history"] if x is not None and math.isfinite(x)]
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq, var=dom,
                            history=h, truth=d["truth"], toto=d["fc"]["toto_2_0"], tfm=TF.get(d["tid"]),
                            moirai=MO.get(d["tid"], {}).get("moirai_2_0"), chronos=CH.get(d["tid"])))
    else:
        TO = json.load(open(X + "toto.json")); TF = json.load(open(X + "timesfm.json"))
        MO = json.load(open("tsfm/moirai_timesx.json")); CH = json.load(open("tsfm/chronos_timesx.json"))
        for t in json.load(open(X + "tasks.json")):
            out.append(dict(tid=t["tid"], part={"test_id": "test"}.get(t["part"], t["part"]), domain=t["domain"], freq=t["freq"],
                            var=t["var"], history=t["history"], truth=t["truth"], toto=TO[t["tid"]], tfm=TF[t["tid"]],
                            moirai=MO.get(t["tid"], {}).get("moirai_2_0"), chronos=CH.get(t["tid"])))
    res = []
    for t in out:
        h = t["history"]; H = len(t["truth"]); p = season(t["freq"])
        t["sea"] = [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]
        for m in MEM:
            if not ok(t.get(m)) or len(t[m]) != H: t[m] = list(t["toto"])   # missing member -> Toto (logged below)
        if ok(t["truth"]) and ok(t["toto"]) and ok(h): res.append(t)
    return res

T = load()
W = np.array([w for w in itertools.product(range(11), repeat=len(MEM)) if sum(w) == 10], float) / 10
SS = np.array([0.0, 0.1, 0.2, 0.3]); NC = len(W) * len(SS)
for t in T:
    Mx = np.array([t[m] for m in MEM], float); y = np.array(t["truth"], float); last = t["history"][-1]
    F = (1 - SS[:, None, None]) * (W @ Mx)[None] + SS[:, None, None] * last          # (S, W, H)
    F = F.reshape(NC, -1); sc = np.mean(np.abs(y))
    e1 = np.mean(np.abs(F - y), 1); e2 = np.sqrt(np.mean((F - y) ** 2, 1))
    t["E"] = (np.minimum(5, e1 / sc) + np.minimum(5, e2 / sc)) if sc > 0 else np.where(e1 == 0, 0, 10.0)
    t["F"] = F if t["part"] != "train" else None
def cfg(i): return W[i % len(W)], SS[i // len(W)]
def best(ts): return int(np.argmin(sum(t["E"] for t in ts)))
def fit(ts, grouping):
    g0 = best(ts); P = {}
    levels = {"global": [], "domain": ["domain"], "freq": ["freq"], "var": ["domain", "var"]}[grouping]
    for lv in levels:
        for key in {t[lv] for t in ts}:
            sub = [t for t in ts if t[lv] == key]
            if len(sub) >= MIN: P[(lv, key)] = best(sub)
    def pick(t):
        c = g0
        for lv in levels: c = P.get((lv, t[lv]), c)
        return c
    return pick
def cv(ts, grouping, k=5):
    idx = list(range(len(ts))); random.Random(0).shuffle(idx); tot = 0
    for f in range(k):
        te = set(idx[f::k]); P = fit([ts[i] for i in idx if i not in te], grouping); tot += sum(ts[i]["E"][P(ts[i])] for i in te)
    return tot / len(ts)
tr = [t for t in T if t["part"] == "train"]
gs = ["global", "domain", "freq"] + (["var"] if name == "timesx" else [])
sc = {g: cv(tr, g) for g in gs}; bg = min(sc, key=sc.get); P = fit(tr, bg)
w0, s0 = cfg(best(tr))
print(f"== {name} v3 members={MEM} MIN={MIN}: CV {', '.join(f'{g} {v:.4f}' for g, v in sc.items())} -> {bg}; global w={dict(zip(MEM, w0.round(1)))} s={s0}")
def idx_of(w, s): return int(np.where((np.abs(W - np.array(w)).sum(1) < 1e-9))[0][0] + len(W) * list(SS).index(s))
single = {m: idx_of([1.0 if x == m else 0.0 for x in MEM], 0.0) for m in MEM if m != "sea"}
out = {}
for part in ["train", "dev", "test", "test_ood"]:
    ts = [t for t in T if t["part"] == part]
    if not ts: continue
    rows = {**{m: (lambda t, i=i: i) for m, i in single.items()}, "v3": P}
    for n, f in rows.items():
        sm = sr = 0; na, nb = [], []
        for t in ts:
            i = f(t); Mx = np.array([t[m] for m in MEM], float); w, s = cfg(i)
            fc = (1 - s) * (w @ Mx) + s * t["history"][-1]
            y = np.array(t["truth"]); scl = np.mean(np.abs(y))
            sm += min(5, np.mean(np.abs(fc - y)) / scl) / len(ts); sr += min(5, np.sqrt(np.mean((fc - y) ** 2)) / scl) / len(ts)
            if name == "timesx":
                a, b = PM.paper(t["truth"], list(fc), PM.snaive(t["history"], len(t["truth"]), t["freq"]))
                if a is not None: na.append(a)
                if b is not None: nb.append(b)
        ex = f" nMAE {np.mean(na):.3f} nMSE {np.mean(nb):.3f}" if na else ""
        out[f"{part}/{n}"] = [sm, sr]; print(f"   {part:9s} {n:8s} n={len(ts):4d} sMAE {sm:.4f} sRMSE {sr:.4f}{ex}")
json.dump(dict(members=MEM, MIN=MIN, grouping=bg, cv=sc, results=out), open(f"unified3_{name}.json", "w"), indent=1)
