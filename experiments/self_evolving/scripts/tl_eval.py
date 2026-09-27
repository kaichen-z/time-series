"""Evaluate the state-timeline method (2026-09-27).
Candidates per task (all history/timeline-derived): Toto; Toto on repaired history (R); state factors on
Toto (S); R + S.  Decision = which components to use + shrink exponent lambda on the factors, fitted on
Train folds (stratified CV) -- reported with and without CV."""
import json, sys, itertools, statistics
sys.path.insert(0, ".scratch/self_evolving")
import nrd_coevolve as C
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

part = sys.argv[3] if len(sys.argv) > 3 else "train"
D = {d["tid"]: d for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
num = json.load(open(sys.argv[1])); rep = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
tids = [t for t in num if D[t]["part"] == part]


def cand(t, use_r, use_s, lam):
    base = D[t]["fc"]["toto_2_0"]; f = list(rep[t]) if (use_r and t in rep) else list(base)
    if use_s:
        f = [v * (fac ** lam) for v, fac in zip(f, num[t]["factors"])]
    return list(apply_bounded_delta(base, f, max_frac=2.0))


def jt(f, t): x = drcik_point_metrics(D[t]["truth"], f, cap=5.0); return x["smae"] + x["srmse"]


def summ(ts, cfg):
    sb = so = rb = ro = 0.0; w = r = 0
    for t in ts:
        mb = drcik_point_metrics(D[t]["truth"], D[t]["fc"]["toto_2_0"], cap=5.0); mo = drcik_point_metrics(D[t]["truth"], cand(t, *cfg), cap=5.0)
        sb += mb["smae"]; so += mo["smae"]; rb += mb["srmse"]; ro += mo["srmse"]
        dj = (mb["smae"] + mb["srmse"]) - (mo["smae"] + mo["srmse"]); w += dj > 1e-9; r += dj < -1e-9
    return dict(smae=(sb - so) / sb, srmse=(rb - ro) / rb, joint=((sb + rb) - (so + ro)) / (sb + rb), w=w, r=r)


CFGS = [(ur, us, lam) for ur in (False, True) for us in (False, True) for lam in ((0.25, 0.5, 0.75, 1.0) if us else (1.0,)) if ur or us]
print(f"{part}: tasks {len(tids)}  with repair {sum(1 for t in tids if t in rep)}  with state factors {sum(1 for t in tids if num[t]['state_info'])}")
for cfg in CFGS:
    s = summ(tids, cfg); print(f"  repair={cfg[0]!s:5} states={cfg[1]!s:5} lam={cfg[2]}: joint {s['joint']:+.2%} sMAE {s['smae']:+.2%} sRMSE {s['srmse']:+.2%} W/R {s['w']}/{s['r']}")
if part == "train":
    C.FOLD_MODE = "strat"
    folds = C.gfolds([D[t] for t in tids], 3); res = []
    for k in range(3):
        tr = [d["tid"] for j in range(3) if j != k for d in folds[j]]; te = [d["tid"] for d in folds[k]]
        best = max(CFGS, key=lambda c: summ(tr, c)["joint"]); s = summ(te, best); res.append((best, round(s["joint"] * 100, 2), s["w"], s["r"]))
    print("  CV (config chosen on the other folds):", res)
