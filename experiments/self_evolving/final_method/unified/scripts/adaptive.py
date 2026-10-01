"""Per-task adaptive numerical blend from the task's OWN history (no extra labels / annotations).
Members: Toto, TimesFM, seasonal (one season ago); forecast = (1-s)*(w.members) + s*last.
Global config: chosen on TRAIN (joint error over train tasks, ordinary labels).
Per-task config: chosen by rolling-origin backtests inside the task's history (cut 1H/2H/3H before the end).
Final = alpha * per-task-config forecast + (1-alpha) * global-config forecast; alpha chosen on TRAIN; dev check; test once.
usage: adaptive.py <drcik|tmmd|timesx>"""
import json, math, sys, itertools
import numpy as np
S = ".scratch/self_evolving/"; X = "work/timesx/"
sys.path.insert(0, X); import metrics as PM
name = sys.argv[1]
BT_T = json.load(open("bt/toto.json")); BT_F = json.load(open("bt/tfm.json"))
def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f: return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("daily", 7), ("d", 7)):
        if k in f: return p
    return None
def sea(h, H, p): return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]
def ok(v): return v is not None and len(v) and all(x is not None and math.isfinite(x) for x in v)
def load():
    out = []
    if name == "drcik":
        TF = json.load(open(S + "timesfm_full.json"))
        for d in json.load(open(S + "nrd_cache.json")):
            dom, fq = d["group"].split("|")
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq, history=d["history"], truth=d["truth"], toto=d["fc"]["toto_2_0"], tfm=TF.get(d["tid"])))
    elif name == "tmmd":
        TF = json.load(open("tmmd_timesfm.json"))
        for d in json.load(open(S + "tmmd_cache.json")):
            dom, fq = d["group"].split("|"); h = [x for x in d["history"] if x is not None and math.isfinite(x)]
            out.append(dict(tid=d["tid"], part={"public_test": "test"}.get(d["part"], d["part"]), domain=dom, freq=fq, history=h, truth=d["truth"], toto=d["fc"]["toto_2_0"], tfm=TF.get(d["tid"])))
    else:
        TO = json.load(open(X + "toto.json")); TF = json.load(open(X + "timesfm.json"))
        for t in json.load(open(X + "tasks.json")):
            out.append(dict(tid=t["tid"], part={"test_id": "test"}.get(t["part"], t["part"]), domain=t["domain"], freq=t["freq"], history=t["history"], truth=t["truth"], toto=TO[t["tid"]], tfm=TF[t["tid"]]))
    res = []
    for t in out:
        if not (ok(t["truth"]) and ok(t["toto"]) and ok(t["history"])): continue
        if not ok(t["tfm"]) or len(t["tfm"]) != len(t["truth"]): t["tfm"] = list(t["toto"])
        H = len(t["truth"]); p = season(t["freq"]); h = t["history"]
        t["sea"] = sea(h, H, p); t["bt"] = []
        for k in (1, 2, 3):
            key = f"{name}|{t['tid']}|{k}"
            if key in BT_T:
                hh = h[:len(h) - k * H]; tgt = h[len(h) - k * H: len(h) - k * H + H]
                tf = BT_F.get(key, BT_T[key])
                t["bt"].append(dict(M=np.array([BT_T[key], tf, sea(hh, H, p)], float), last=hh[-1], y=np.array(tgt, float)))
        res.append(t)
    return res
T = load()
W = np.array([w for w in itertools.product(range(11), repeat=3) if sum(w) == 10], float) / 10
SS = np.array([0.0, 0.1, 0.2, 0.3]); NC = len(W) * len(SS)
def errs(Mx, last, y):
    F = ((1 - SS[:, None, None]) * (W @ Mx)[None] + SS[:, None, None] * last).reshape(NC, -1)
    sc = np.mean(np.abs(y)) + 1e-12
    return np.minimum(5, np.mean(np.abs(F - y), 1) / sc) + np.minimum(5, np.sqrt(np.mean((F - y) ** 2, 1)) / sc), F
for t in T:
    t["E"], t["F"] = errs(np.array([t["toto"], t["tfm"], t["sea"]], float), t["history"][-1], np.array(t["truth"], float))
    t["BE"] = np.mean([errs(b["M"], b["last"], b["y"])[0] for b in t["bt"]], 0) if t["bt"] else None
tr = [t for t in T if t["part"] == "train"]
g = int(np.argmin(sum(t["E"] for t in tr)))
def fc(t, alpha, lam):
    if t["BE"] is None: return t["F"][g]
    i = int(np.argmin(t["BE"] + lam * (np.arange(NC) != g)))   # lam = preference for the global config
    return alpha * t["F"][i] + (1 - alpha) * t["F"][g]
def jt(f, y):
    y = np.asarray(y); sc = np.mean(np.abs(y)) + 1e-12
    return min(5, np.mean(np.abs(f - y)) / sc) + min(5, np.sqrt(np.mean((f - y) ** 2)) / sc)
grid = [(a, l) for a in (0, .25, .5, .75, 1) for l in (0, .02, .05, .1, .2)]
best = min(grid, key=lambda p: sum(jt(fc(t, *p), t["truth"]) for t in tr))
print(f"== {name}: global cfg w={W[g % len(W)].round(1)} s={SS[g // len(W)]}; train-chosen alpha={best[0]} lam={best[1]}; tasks with backtests {sum(1 for t in T if t['BE'] is not None)}/{len(T)}")
for part in ("train", "dev", "test", "test_ood"):
    ts = [t for t in T if t["part"] == part]
    if not ts: continue
    rows = {"Toto": lambda t: np.array(t["toto"]), "TimesFM": lambda t: np.array(t["tfm"]), "global": lambda t: t["F"][g],
            "per-task only": lambda t: fc(t, 1.0, 0.0), "adaptive": lambda t: fc(t, *best)}
    for n, f in rows.items():
        sm = np.mean([PM.M(t["truth"], list(f(t)), cap=5.0)["smae"] for t in ts]); sr = np.mean([PM.M(t["truth"], list(f(t)), cap=5.0)["srmse"] for t in ts])
        ex = ""
        if name == "timesx":
            pa = [PM.paper(t["truth"], list(f(t)), PM.snaive(t["history"], len(t["truth"]), t["freq"])) for t in ts]
            ex = f" nMAE {np.mean([a for a, b in pa if a is not None]):.3f} nMSE {np.mean([b for a, b in pa if b is not None]):.3f}"
        print(f"   {part:9s} {n:14s} n={len(ts):4d} sMAE {sm:.4f} sRMSE {sr:.4f}{ex}")
