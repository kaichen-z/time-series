"""One framework for Dr-CiK, Time-MMD and TimesX (2026-09-30). Same code and search space for every dataset;
every choice is made on that dataset's TRAIN split only; dev = check; test reported once.
  ② history repair  : Retrieval anomaly intervals -> same-phase-median fill -> history-only backtest gate (30%).
                      Only fires where anomaly intervals were extracted (Dr-CiK); otherwise a no-op.
  ③ numerical step  : two candidate families, chosen by 5-fold CV on train:
                      (a) the Dr-CiK-evolved combination program (Toto[repaired] + ARIMA + TimesFM, shrink);
                      (b) the unified search (Toto[repaired] / TimesFM / seasonal weights x shrink x grouping
                          {global, domain, freq, variable}, groups need >= 30 train tasks).
                      Cold start: a task whose variable never appears in train uses its OWN history backtests
                      (1H/2H/3H cut points) to choose the blend (no labels needed).
  ④ correction      : the Dr-CiK-evolved adjust(view) is run on every dataset; it decides itself whether to use
                      document corrections (on background-type text it rejects them).
usage: framework.py <drcik|tmmd|timesx>"""
import json, math, sys, random, itertools, importlib.util
import numpy as np
S = ".scratch/self_evolving/"; X = "work/timesx/"
sys.path.insert(0, X); import metrics as PM
CORR = "/tmp/wt_self_evolving/experiments/self_evolving/final_method/artifacts/correction_function_main.py"
sp = importlib.util.spec_from_file_location("corr", CORR); corr = importlib.util.module_from_spec(sp); sp.loader.exec_module(corr)
name = sys.argv[1]; MIN = 30
BT_T = json.load(open("bt/toto.json")); BT_F = json.load(open("bt/tfm.json"))
EVO = dict(toto=0.9904652444807432, arima=0.0531, tfm=0.11595169383119369); EVO_S = 0.20153008320833654

def season(freq):
    f = str(freq).lower()
    if "min" in f or "sec" in f: return None
    for k, p in (("hour", 24), ("week", 52), ("w", 52), ("month", 12), ("day", 7), ("daily", 7), ("d", 7)):
        if k in f: return p
    return None
def sea(h, H, p): return [h[-p + i] if p and len(h) >= p and -p + i < 0 else h[-1] for i in range(H)]
def ok(v): return v is not None and len(v) > 0 and all(x is not None and math.isfinite(x) for x in v)

def load():
    T = []
    if name == "drcik":
        MHR = ".scratch/self_evolving/coevo/mhr/runs/shared3/"
        V = {**json.load(open(MHR + "shared/views_train.json")), **json.load(open("/tmp/mhr_dev.json")), **json.load(open("/tmp/mhr_test.json"))}
        TR = {**json.load(open(MHR + "private/truth.json")), **json.load(open("/tmp/mhr_dev_truth.json")), **json.load(open("/tmp/mhr_test_truth.json"))}
        FV = json.load(open(S + "fill_variants.json")); TF = json.load(open(S + "timesfm_full.json"))
        C = {d["tid"]: d for d in json.load(open(S + "nrd_cache.json"))}
        for t, v in V.items():
            d = C[t]; mf = v["method_forecasts"]; fv = FV.get(t, {}).get("phase_median")
            rep = fv["forecast"] if (fv and fv["val_raw"] is not None and fv["val_rep"] < fv["val_raw"] * 0.7) else None
            T.append(dict(tid=t, part={"public_test": "test"}.get(d["part"], d["part"]), domain=d["group"].split("|")[0], freq=d["group"].split("|")[1],
                          history=v["history"], truth=TR[t]["truth"], toto=mf["toto_2_0"], tfm=mf.get("timesfm_2_5") or TF.get(t), arima=mf.get("arima_auto"), toto_rep=rep, view=v))
    elif name == "tmmd":
        V = json.load(open(S + "coevo/tmmd/views_all.json")); TR = json.load(open(S + "coevo/tmmd/truth_all.json")); TF = json.load(open("tmmd_timesfm.json"))
        for t, v in V.items():
            dom, fq = v["group"].split("|")
            T.append(dict(tid=t, part={"public_test": "test"}.get(v["part"], v["part"]), domain=dom, freq=fq, history=[x for x in v["history"] if x is not None and math.isfinite(x)],
                          truth=TR[t]["truth"], toto=v["toto_forecast"], tfm=TF.get(t), arima=None, toto_rep=None, view=v))
    else:
        V = json.load(open("/tmp/timesx/views_all.json")); TR = json.load(open("/tmp/timesx/truth_all.json")); TF = json.load(open(X + "timesfm.json"))
        for t, v in V.items():
            T.append(dict(tid=t, part={"public_test": "test", "ood_test": "test_ood"}.get(v["part"], v["part"]), domain=v["group"].split("|")[0], freq=v["freq"],
                          var=t.rsplit("_", 1)[0], history=v["history"], truth=TR[t]["truth"], toto=v["toto_forecast"], tfm=TF.get(t), arima=None, toto_rep=None, view=v))
    out = []
    for t in T:
        t.setdefault("var", t["domain"]); H = len(t["truth"])
        if not (ok(t["truth"]) and ok(t["toto"]) and ok(t["history"])): continue
        if not ok(t["tfm"]) or len(t["tfm"]) != H: t["tfm"] = list(t["toto"])
        p = season(t["freq"]); h = t["history"]; t["sea"] = sea(h, H, p); t["bt"] = []
        for k in (1, 2, 3):
            key = f"{name}|{t['tid']}|{k}"
            if key in BT_T:
                hh = h[:len(h) - k * H]
                t["bt"].append((np.array([BT_T[key], BT_F.get(key, BT_T[key]), sea(hh, H, p)], float), hh[-1], np.array(h[len(h) - k * H: len(h) - k * H + H], float)))
        out.append(t)
    return out

import os
FIT = os.environ.get("FIT", "train").split(",")  # experiment knob; default train (train+dev / dev-only refits did not help)
T = load(); tr = [t for t in T if t["part"] in FIT]
USE_C2 = os.environ.get("C2", "0") == "1"
if USE_C2:
    _c2 = json.load(open(f"tsfm/chronos2_{name}.json"))
    for t in T:
        f = _c2.get(t["tid"], {}).get("uv")
        t["c2"] = f if ok(f) and len(f) == len(t["truth"]) else list(t["tfm"])
RECENT = float(os.environ.get("RECENT", "1.0"))   # keep only the most recent fraction of train windows per variable
def tkey(t):
    if name == "timesx": return int(t["tid"].rsplit("_", 1)[1])
    if name == "tmmd": return t["tid"].rsplit("_", 1)[1]
    return 0
if RECENT < 1.0 and name != "drcik":
    keep = []
    for v in {t["var"] for t in tr}:
        ts = sorted([t for t in tr if t["var"] == v], key=tkey); keep += ts[int(len(ts) * (1 - RECENT)):]
    tr = keep
W = np.array([w for w in itertools.product(range(11), repeat=4 if os.environ.get("C2", "0") == "1" else 3) if sum(w) == 10], float) / 10
SS = np.array([0.0, 0.1, 0.2, 0.3]); NC = len(W) * len(SS)
def grid(Mx, last): return ((1 - SS[:, None, None]) * (W @ Mx)[None] + SS[:, None, None] * last).reshape(NC, -1)
def jerr(F, y):
    y = np.asarray(y, float); sc = np.mean(np.abs(y)) + 1e-12
    return np.minimum(5, np.mean(np.abs(F - y), -1) / sc) + np.minimum(5, np.sqrt(np.mean((F - y) ** 2, -1)) / sc)
SPLIT = os.environ.get("SPLIT", "0") == "1"
for t in T:
    tt = t["toto_rep"] or t["toto"]
    t["G"] = grid(np.array([tt, t["tfm"], t["sea"]] + ([t["c2"]] if USE_C2 else []), float), t["history"][-1]); t["GE"] = jerr(t["G"], t["truth"])
    hh = max(1, len(t["truth"]) // 2); y = np.array(t["truth"], float)
    t["GE1"] = jerr(t["G"][:, :hh], y[:hh]); t["GE2"] = jerr(t["G"][:, hh:], y[hh:]) if len(y) > hh else t["GE1"] * 0
    a = t["arima"] if ok(t["arima"]) and len(t["arima"]) == len(t["truth"]) else t["toto"]
    tot = sum(EVO.values())
    f = (EVO["toto"] * np.array(tt) + EVO["arima"] * np.array(a) + EVO["tfm"] * np.array(t["tfm"])) / tot
    t["EVO"] = (1 - EVO_S) * f + EVO_S * t["history"][-1]; t["EVOE"] = float(jerr(t["EVO"], t["truth"]))
    if t["bt"]:
        if USE_C2:
            be = np.mean([jerr(grid(np.vstack([M, M[1:2]]), last), y) for M, last, y in t["bt"]], 0)
            be = be + np.tile(np.where(W[:, 3] > 0, 1e9, 0.0), len(SS)); t["BE"] = be
        else:
            t["BE"] = np.mean([jerr(grid(M, last), y) for M, last, y in t["bt"]], 0)
    else: t["BE"] = None

def best(ts):
    if SPLIT: return (int(np.argmin(sum(t["GE1"] for t in ts))), int(np.argmin(sum(t["GE2"] for t in ts))))
    return int(np.argmin(sum(t["GE"] for t in ts)))
def gfc(t, c):
    if isinstance(c, tuple):
        hh = max(1, len(t["truth"]) // 2); return np.concatenate([t["G"][c[0]][:hh], t["G"][c[1]][hh:]])
    return t["G"][c]
def gerr(t, c): return float(jerr(gfc(t, c), t["truth"])) if isinstance(c, tuple) else float(t["GE"][c])
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
def cv(ts, fn, k=5):
    idx = list(range(len(ts))); random.Random(0).shuffle(idx); tot = 0
    for f in range(k):
        te = set(idx[f::k]); tot += sum(fn([ts[i] for i in idx if i not in te], ts[i]) for i in te)
    return tot / len(ts)
groupings = ["global", "domain", "freq"] + (["var"] if name == "timesx" else [])
cvs = {g: cv(tr, lambda trn, t, g=g: gerr(t, fit(trn, g)(t))) for g in groupings}
bg = min(cvs, key=cvs.get); P = fit(tr, bg)
cv_evo = float(np.mean([t["EVOE"] for t in tr]))   # fixed program (evolved on Dr-CiK train; in-sample there)
family = "evolved_program" if cv_evo < cvs[bg] else "unified_search"
train_vars = {t["var"] for t in T if t["part"] in FIT}
print(f"== {name}: unified CV {', '.join(f'{g} {v:.4f}' for g, v in cvs.items())} -> {bg}; evolved program train {cv_evo:.4f} -> family: {family}")
def numeric(t, cold=True):
    if cold and t["var"] not in train_vars and t["BE"] is not None: return t["G"][int(np.argmin(t["BE"]))], "cold-start"
    return (t["EVO"], family) if family == "evolved_program" else (gfc(t, P(t)), family)
def final(t, use_corr=True, cold=True):
    b, how = numeric(t, cold)
    if not use_corr: return b
    v = dict(t["view"], base_forecast=[float(x) for x in b], toto_forecast=[float(x) for x in t["toto"]])
    try:
        f = [float(x) for x in corr.adjust(v)]
        if len(f) == len(b) and all(math.isfinite(x) for x in f): return np.array(f)
    except Exception: pass
    return b
for part in ("train", "dev", "test", "test_ood"):
    ts = [t for t in T if t["part"] == part]
    if not ts: continue
    rows = {"Toto": lambda t: np.array(t["toto"]), "TimesFM": lambda t: np.array(t["tfm"]),
            "framework numeric": lambda t: final(t, False), "framework full": lambda t: final(t, True)}
    for n, fn in rows.items():
        F = {t["tid"]: fn(t) for t in ts}
        sm = np.mean([PM.M(t["truth"], list(F[t["tid"]]), cap=5.0)["smae"] for t in ts]); sr = np.mean([PM.M(t["truth"], list(F[t["tid"]]), cap=5.0)["srmse"] for t in ts])
        ex = ""
        if name == "timesx":
            pa = [PM.paper(t["truth"], list(F[t["tid"]]), PM.snaive(t["history"], len(t["truth"]), t["freq"])) for t in ts]
            ex = f" nMAE {np.mean([a for a, b in pa if a is not None]):.3f} nMSE {np.mean([b for a, b in pa if b is not None]):.3f}"
        extra = ""
        if n == "framework full":
            ch = sum(1 for t in ts if np.max(np.abs(F[t["tid"]] - final(t, False))) > 1e-9); extra = f"  corrected {ch}"
            extra += f"  cold-start {sum(1 for t in ts if t['var'] not in train_vars)}  repaired {sum(1 for t in ts if t['toto_rep'])}"
        print(f"   {part:9s} {n:18s} n={len(ts):4d} sMAE {sm:.4f} sRMSE {sr:.4f}{ex}{extra}")
if len(sys.argv) > 2 and sys.argv[2] == "params":
    cfg = lambda i: dict(w=[round(float(x), 2) for x in W[i % len(W)]], s=float(SS[i // len(W)]))
    keys = sorted({t[bg] for t in tr}) if bg != "global" else []
    print("PARAMS", json.dumps(dict(grouping=bg, global_cfg=cfg(best(tr)), groups={k: cfg(P(next(t for t in tr if t[bg] == k))) for k in keys})))

if os.environ.get("DUMP"):
    dump = {}
    for t in T:
        if t["part"] in ("dev", "test", "test_ood"):
            dump[t["tid"]] = dict(part=t["part"], framework=[float(x) for x in final(t, True)],
                                  toto=[float(x) for x in t["toto"]], timesfm=[float(x) for x in t["tfm"]],
                                  chronos2=[float(x) for x in t.get("c2", t["tfm"])], truth=t["truth"], history=t["history"], freq=t["freq"])
    json.dump(dump, open(os.environ["DUMP"], "w"))
