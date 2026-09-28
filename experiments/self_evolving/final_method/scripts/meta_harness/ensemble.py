"""Ensemble / routing of the evolved correction functions. Selection on Train only (visible folds 0+1, hidden
fold 2 as check, same fitness as the evaluator); then one dev check of the chosen combination (+ exploratory test)."""
import json, sys, copy, subprocess, tempfile, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coevolution"))
import e2e as E
from common.metrics import drcik_point_metrics as M
N4 = E.N4; HERE = Path(__file__).parent; ART = HERE.parents[1] / "artifacts/meta_harness"
NAMES = ["single", "shared3", "indep1", "indep2", "indep3"]
FUNCS = {n: str(ART / n / "best_correction_function.py") for n in NAMES}
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
FV = json.load(open(".scratch/self_evolving/fill_variants.json"))
prog = json.load(open(str(HERE.parents[1] / "artifacts/numerical_part1_main.json")))["elites"][0]["program"]
cfg = E.main_config(); team = E.team_of(cfg)
ALL = {d["tid"]: d for d in [N4.R3.prep_task(d, E.TH) for d in json.load(open(E.os.environ["CACHE"]))]}
for d in ALL.values(): d["_sig"] = {}; h = d["history"]; d["_last"], d["_lo"], d["_hi"] = h[-1], min(h), max(h)


def views(ids):
    out = {}
    for t in ids:
        d = ALL[t]; dd = copy.copy(d); dd["fc"] = dict(d["fc"]); v = FV.get(t, {}).get("phase_median")
        if v and v["val_raw"] is not None and v["val_rep"] < v["val_raw"] * 0.7: dd["fc"]["toto_2_0"] = v["forecast"]
        out[t] = dict(tid=t, H=d["H"], freq=d["freq"], history=d["history"], base_forecast=E.P1.run(prog, dd), toto_forecast=d["fc"]["toto_2_0"],
            documents=[x[:3000] for x in d["docs"]], doc_confidence=d["conf"], docbase=d["docbase"], cell=d["cell"],
            cell_toto_backtest_error=team["numerical"]["trust"].get(d["cell"], 1.0), task_toto_backtest_error=d["toto_h"],
            sigma_main_calib=N4.sigma_of(team["numerical"]["calib"], d["history"], d["H"], d["freq"]),
            corrections=[dict(start=s, end=e, multiplier=m) for s, e, m in d["corr"]])
    return out


def run_all(ids):
    vw = views(ids); tmp = Path(tempfile.mkdtemp()); json.dump(vw, open(tmp / "v.json", "w")); res = {}
    for n, f in FUNCS.items():
        subprocess.run([sys.executable, str(HERE / "harness_runner.py"), f, str(tmp / "v.json"), str(tmp / f"{n}.json")], check=True, timeout=900)
        res[n] = json.load(open(tmp / f"{n}.json"))["forecasts"]
    return res


def jt(f, t): x = M(ALL[t]["truth"], f, cap=5.0); return x["smae"] + x["srmse"]
def gain(f, t): return ALL[t]["base_jt"] - jt(f, t) if "base_jt" in ALL[t] else None
def robust(g): return statistics.mean(g) + 0.5 * statistics.mean(min(0.0, x) for x in g)


def combine(kind, res, t, route=None, w=None):
    fs = {n: res[n][t] for n in NAMES}
    if kind == "mean": return [statistics.mean(v) for v in zip(*fs.values())]
    if kind == "median": return [statistics.median(v) for v in zip(*fs.values())]
    if kind == "route": return fs[route.get(ALL[t]["cell"], "shared3")]
    if kind == "wmean": return [sum(w[n] * fs[n][i] for n in NAMES) for i in range(len(fs["shared3"]))]
    return fs[kind]


tr = sorted(split["train"]["task_ids"]); RT = run_all(tr)
for t in tr: ALL[t]["base_jt"] = jt(ALL[t]["fc"]["toto_2_0"], t)
F = [[d["tid"] for d in f] for f in E.FOLDS]; VIS, HID = F[0] + F[1], F[2]
def fit(kind, **kw):
    fl = [robust([ALL[t]["base_jt"] - jt(combine(kind, RT, t, **kw), t) for t in fo]) for fo in (F[0], F[1])]
    hid = robust([ALL[t]["base_jt"] - jt(combine(kind, RT, t, **kw), t) for t in HID])
    return statistics.mean(fl) - 0.25 * statistics.pstdev(fl), hid
cands = {n: {} for n in NAMES}; cands.update({"mean": {}, "median": {}})
# routing: per cell, the function with the best mean gain on the VISIBLE folds
route = {}
for c in sorted({ALL[t]["cell"] for t in VIS}):
    ts = [t for t in VIS if ALL[t]["cell"] == c]
    route[c] = max(NAMES, key=lambda n: statistics.mean(ALL[t]["base_jt"] - jt(RT[n][t], t) for t in ts))
cands["route"] = dict(route=route)
# weights: inverse rank of visible fitness (no fitting to avoid overfitting)
vf = {n: fit(n)[0] for n in NAMES}; rk = sorted(NAMES, key=lambda n: -vf[n]); w = {n: (len(NAMES) - rk.index(n)) for n in NAMES}; s = sum(w.values()); w = {n: v / s for n, v in w.items()}
cands["wmean"] = dict(w=w)
print("candidate                visible   hidden")
scores = {}
for k, kw in cands.items():
    kind = k if k in ("mean", "median", "route", "wmean") else k
    v, h = fit(kind, **kw); scores[k] = (v, h); print(f"{k:22s} {v:+.4f}  {h:+.4f}")
base_v, base_h = scores["shared3"]
ok = {k: s for k, s in scores.items() if s[1] >= base_h - 1e-9}
best = max(ok, key=lambda k: ok[k][0]); print("route table:", route); print("weights:", {n: round(x, 3) for n, x in w.items()})
print("CHOSEN on Train (visible best among hidden >= shared3):", best)
for part in ("dev", "public_test"):
    ids = sorted(split[part]["task_ids"]); R = run_all(ids)
    for t in ids: ALL[t]["base_jt"] = jt(ALL[t]["fc"]["toto_2_0"], t)
    for k in sorted({best, "shared3"}):
        kind = k; sb = so = rb = ro = 0
        for t in ids:
            b = M(ALL[t]["truth"], ALL[t]["fc"]["toto_2_0"], cap=5.0); o = M(ALL[t]["truth"], combine(kind, R, t, **cands[k]), cap=5.0)
            sb += b["smae"]; so += o["smae"]; rb += b["srmse"]; ro += o["srmse"]
        n = len(ids); print(f"{part:11s} {k:10s} sMAE {sb/n:.4f}->{so/n:.4f} sRMSE {rb/n:.4f}->{ro/n:.4f}", flush=True)
