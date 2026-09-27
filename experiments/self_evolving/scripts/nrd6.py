"""Residual corrector + conformal do-no-harm gate (2026-09-27).

1) Retrieval turns documents into *structured per-step features* (from the extracted cards: window
   membership, signed log-multiplier, confidence, document type, magnitude-vs-history ratio);
   Numerical adds history-only features (Toto hindcast bias, cell trust, seasonality).  A small
   ridge model predicts Toto's relative residual per forecast step from these features, trained on
   ALL Train steps (tasks without events teach it not to move).  The feature program and the
   regularisation are evolved (typed mutations, stratified Train CV fitness, rollback).
2) Conformal risk control: on out-of-fold Train predictions, choose the smallest score threshold
   tau such that the Clopper-Pearson upper bound on the harm rate among accepted tasks is <= alpha.
   At test time a task is corrected only if its score >= tau; otherwise exact Toto.
Dev once; test99 exploratory (opened earlier).
"""
from __future__ import annotations
import argparse, copy, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import nrd_coevolve as C
import nrd3 as R3
import nrd4 as N4
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

ANCHOR = "toto_2_0"
ALLF = ["inwin", "lm", "lm_conf", "lm_docbase", "lm_magsupp", "lm_ratio", "hbias", "hbias_trust", "seas", "pos"]


def task_feats(d, trust):
    H = d["H"]; toto = d["fc"][ANCHOR]; sig = N4.sigma_of(N4.calib0(), d["history"], H, d["freq"])
    rows = [dict.fromkeys(ALLF, 0.0) for _ in range(H)]
    for s, e, m in d["corr"]:
        lm = math.log(max(1e-3, m)); ratio = abs(m - 1) / sig
        ms = 1.0 if abs(m - 1) <= d["f_range"] + 1e-9 else 0.0
        for i in range(s, min(e, H)):
            r = rows[i]; r["inwin"] = 1.0; r["lm"] = lm; r["lm_conf"] = lm * d["conf"]
            r["lm_docbase"] = lm * d["docbase"]; r["lm_magsupp"] = lm * ms; r["lm_ratio"] = lm / (1 + ratio)
    hb = d.get("hbias", 0.0); tr = min(3.0, trust.get(d["cell"], 1.0))
    seas = N4.N.seas_strength(d["history"], N4.N.period_of(d["freq"]))
    for i, r in enumerate(rows):
        r["hbias"] = hb; r["hbias_trust"] = hb * tr; r["seas"] = seas * (1 if r["inwin"] else 0); r["pos"] = i / H * r["inwin"]
    return rows


def hind_bias(d, TH):
    """history-only: mean signed relative residual of Toto over the hindcast origins."""
    r = TH[d["tid"]]; bs = []
    for k, (_p, fut) in zip((1, 2), N4.N.origins(d)):
        bs += [(f - t) / (abs(t) + 1e-9) for t, f in zip(r[f"o{k}"], fut)]
    return float(np.clip(statistics.mean(bs), -0.5, 0.5)) if bs else 0.0


def design(ds, prog, trust):
    X, y, idx = [], [], []
    for d in ds:
        toto = d["fc"][ANCHOR]
        for i, r in enumerate(task_feats(d, trust)):
            X.append([r[f] for f in prog["feats"]]); y.append((d["truth"][i] - toto[i]) / (abs(toto[i]) + 1e-9)); idx.append((d["tid"], i))
    return np.array(X, float), np.clip(np.array(y, float), -1, 1), idx


def fit(ds, prog, trust):
    X, y, _ = design(ds, prog, trust)
    if X.shape[1] == 0: return np.zeros(0)
    A = X.T @ X + prog["alpha"] * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def predict(d, prog, trust, w):
    toto = d["fc"][ANCHOR]
    rows = task_feats(d, trust)
    c = [float(np.clip(np.dot([r[f] for f in prog["feats"]], w) * prog["shrink"], -0.5, 0.5)) if len(w) else 0.0 for r in rows]
    out = [t * (1 + ci) for t, ci in zip(toto, c)]
    return list(apply_bounded_delta(toto, out)), sum(abs(x) for x in c) / len(c)


def gain(d, out): return d["base_jt"] - C.jt(out, d["truth"])


CP_A = 0.1


def cp_upper(k, n, a=None):
    a = CP_A if a is None else a
    """Clopper-Pearson one-sided upper bound on a binomial rate (bisection on the beta cdf via lgamma)."""
    if n == 0: return 1.0
    if k >= n: return 1.0
    def cdf(p):  # P[X <= k]
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if cdf(mid) > a: lo = mid
        else: hi = mid
    return hi


def calibrate_tau(scored, alpha):
    """scored: [(score, gain)] out-of-fold.  Smallest tau with CP-upper(harm rate | score>=tau) <= alpha."""
    for tau in sorted({s for s, _ in scored if s > 0}):
        acc = [g for s, g in scored if s >= tau]
        harm = sum(1 for g in acc if g < -1e-9)
        if acc and cp_upper(harm, len(acc)) <= alpha and statistics.mean(acc) > 0: return tau
    return float("inf")


def oof(train, folds, prog, trust_fn):
    res = []
    for k in range(len(folds)):
        tr = [d for j in range(len(folds)) if j != k for d in folds[j]]; trust = trust_fn(tr)
        w = fit(tr, prog, trust)
        for d in folds[k]:
            out, sc = predict(d, prog, trust, w); res.append((d, sc, gain(d, out), out))
    return res


def summarize(ds, outs):
    sb = so = rb = ro = 0.0; w = r = 0
    for d in ds:
        out = outs[d["tid"]]
        mb = drcik_point_metrics(d["truth"], d["fc"][ANCHOR], cap=5.0); mo = drcik_point_metrics(d["truth"], out, cap=5.0)
        sb += mb["smae"]; so += mo["smae"]; rb += mb["srmse"]; ro += mo["srmse"]
        dj = (mb["smae"] + mb["srmse"]) - (mo["smae"] + mo["srmse"]); w += dj > 1e-9; r += dj < -1e-9
    return dict(n=len(ds), smae_gain=(sb - so) / sb, srmse_gain=(rb - ro) / rb, joint_gain=((sb + rb) - (so + ro)) / (sb + rb), wins=w, regressions=r)


def fitness(train, folds, prog, trust_fn, alpha, conformal):
    """nested: inner OOF on the outer-train part calibrates tau; score on the outer held-out fold."""
    per = []
    for k in range(len(folds)):
        tr = [d for j in range(len(folds)) if j != k for d in folds[j]]; te = folds[k]
        trust = trust_fn(tr); w = fit(tr, prog, trust)
        tau = 0.0
        if conformal:
            inner = C.gfolds(tr, 2); tau = calibrate_tau([(sc, g) for _d, sc, g, _o in oof(tr, inner, prog, trust_fn)], alpha)
        outs = {}
        for d in te:
            out, sc = predict(d, prog, trust, w); outs[d["tid"]] = out if sc >= tau and sc > 0 else list(d["fc"][ANCHOR])
        per.append(summarize(te, outs))
    js = [p["joint_gain"] for p in per]
    return statistics.mean(js) - 0.25 * statistics.pstdev(js), per


def prog0(): return {"feats": ["inwin", "lm"], "alpha": 10.0, "shrink": 1.0}


def mutate(p, rng):
    c = copy.deepcopy(p); op = rng.choice(["add", "drop", "alpha", "shrink"])
    if op == "add":
        cand = [f for f in ALLF if f not in c["feats"]]
        if cand: c["feats"].append(rng.choice(cand))
    elif op == "drop" and len(c["feats"]) > 1: c["feats"].remove(rng.choice(c["feats"]))
    elif op == "alpha": c["alpha"] = float(np.clip(c["alpha"] * 2 ** rng.gauss(0, 1), 0.1, 1000))
    else: c["shrink"] = float(np.clip(c["shrink"] + rng.gauss(0, 0.2), 0.1, 1.5))
    return c, op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--gens", type=int, default=25); ap.add_argument("--children", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--no-conformal", action="store_true"); ap.add_argument("--open", choices=["cv", "dev", "test"], default="test")
    ap.add_argument("--cp", type=float, default=0.1)
    ap.add_argument("--data", choices=["drcik", "tmmd"], default="drcik")
    a = ap.parse_args(); conformal = not a.no_conformal
    global CP_A; CP_A = a.cp
    C.FOLD_MODE = "strat"
    sfx = "" if a.data == "drcik" else "_tmmd"
    TH = json.load(open(f".scratch/self_evolving/toto_hindcast{sfx}.json"))
    D = [R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json" if a.data == "drcik" else ".scratch/self_evolving/tmmd_cache.json"))]
    for d in D: d["hbias"] = hind_bias(d, TH)
    train = [d for d in D if d["part"] == "train"]; dev = [d for d in D if d["part"] == "dev"]; test = [d for d in D if d["part"] == "public_test"]
    trust_fn = N4.trust_profile
    res = {}
    for seed in a.seeds:
        rng = random.Random(seed); folds = C.gfolds(train, 3)
        cur = prog0(); f, per = fitness(train, folds, cur, trust_fn, a.alpha, False); curve = [round(f, 5)]
        for g in range(a.gens):
            best = None
            for _ in range(a.children):
                ch, op = mutate(cur, rng); fc, pc = fitness(train, folds, ch, trust_fn, a.alpha, False)
                if best is None or fc > best[0]: best = (fc, pc, ch)
            if best[0] > f + 1e-6: f, per, cur = best
            curve.append(round(f, 5))
        if conformal: f, per = fitness(train, folds, cur, trust_fn, a.alpha, True)   # report CV with the gate
        # final: fit on all Train, calibrate tau with Train OOF, then dev gate, test exploratory
        trust = trust_fn(train); w = fit(train, cur, trust)
        oo = oof(train, C.gfolds(train, 3), cur, trust_fn)
        srt = sorted(((sc, g) for _d, sc, g, _o in oo), reverse=True)
        print("  OOF top-score buckets (n, harm, mean gain):", [(len(b), sum(g < -1e-9 for _s, g in b), round(statistics.mean(g for _s, g in b), 4)) for b in [srt[:10], srt[10:20], srt[20:40], srt[40:]] if b], flush=True)
        tau = calibrate_tau([(sc, g) for _d, sc, g, _o in oof(train, C.gfolds(train, 3), cur, trust_fn)], a.alpha) if conformal else 0.0
        def outs(ds):
            o = {}
            for d in ds:
                out, sc = predict(d, cur, trust, w); o[d["tid"]] = out if sc >= tau and sc > 0 else list(d["fc"][ANCHOR])
            return o
        r = dict(program=cur, weights=dict(zip(cur["feats"], [round(x, 4) for x in w])), tau=tau, curve=curve,
                 cv=[p["joint_gain"] for p in per], cv_wr=[(p["wins"], p["regressions"]) for p in per],
                 train=summarize(train, outs(train)), dev=summarize(dev, outs(dev)))
        r["dev_gate_pass"] = r["dev"]["smae_gain"] >= 0 and r["dev"]["srmse_gain"] >= 0
        if a.open == "test": r["test_exploratory"] = summarize(test, outs(test) if r["dev_gate_pass"] else {d["tid"]: d["fc"][ANCHOR] for d in test})
        res[seed] = r
        t = r.get("test_exploratory", {})
        print(f"seed {seed}: CV {[round(x * 100, 2) for x in r['cv']]} W/R {r['cv_wr']} | dev {r['dev']['joint_gain']:+.2%} "
              f"({r['dev']['wins']}/{r['dev']['regressions']}) gate {r['dev_gate_pass']} | test(expl) sMAE {t.get('smae_gain', 0):+.2%} "
              f"sRMSE {t.get('srmse_gain', 0):+.2%} W/R {t.get('wins')}/{t.get('regressions')} | feats {cur['feats']} tau {tau}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
