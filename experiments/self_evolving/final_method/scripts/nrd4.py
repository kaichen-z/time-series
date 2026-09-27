"""Redesigned N/R/D evolve space (2026-09-27).

Numerical no longer supplies competing forecasters.  It evolves history-only *numerical judges*:
  * a magnitude calibrator  sigma(history) -> "how large a deviation is normal for this series";
    fitness = how well sigma predicts the size of the next held-out history segment's deviation
    from its reference level (rolling origins inside the history; no labels);
  * a Toto trust profile per morphology cell (pooled Toto hindcast error), not per task.
Retrieval self-evolves its evidence validator using Numerical's judges as features
  (|m-1|/sigma, trust) plus document features; per-correction credit on Train.
Decision keeps 2 parameters: evidence strength and a trust gate.
Team co-evolution on stratified Train folds; Gen0 = exact Toto.  Dev once; test99 already opened
earlier, so any test number from this script is exploratory.
"""
from __future__ import annotations
import argparse, copy, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import nrd_coevolve as C
import nrd_dict as N
import nrd3 as R3
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

ANCHOR = "toto_2_0"
FEATS = ["absmag", "conf", "wfrac", "docbase", "ratio", "trust", "bias"]


# ----------------------------------------------------------------------------- Numerical: calibrators
def sigma_of(g, y, H, freq):
    y = np.asarray(y, float); n = len(y)
    w = n if g["win"] == 0 else max(4, min(n, int(g["win"] * H)))
    z = y[-w:]
    if g["deseason"]:
        p = N.period_of(freq)
        if p > 1 and len(z) > 2 * p: z = z[p:] - z[:-p] + np.mean(z)
    ref = {"median": np.median(z), "mean": np.mean(z), "last": z[-1]}[g["ref"]]
    ref = abs(ref) + 1e-9
    dev = np.abs(z - np.median(z)) / ref
    s = {"q90": np.quantile(dev, 0.9), "q75": np.quantile(dev, 0.75), "std": np.std(z) / ref,
         "mad": np.median(dev) * 1.4826}[g["stat"]]
    return float(max(1e-4, min(5.0, g["mult"] * s)))


def calib0():
    return {"win": 0, "deseason": False, "ref": "median", "stat": "q90", "mult": 1.0}   # = hand 'f_range'-like


def mutate_calib(g, rng):
    c = dict(g); op = rng.choice(["win", "deseason", "ref", "stat", "mult", "mult"])
    if op == "win": c["win"] = rng.choice([0, 1, 2, 3, 5, 8])
    elif op == "deseason": c["deseason"] = not c["deseason"]
    elif op == "ref": c["ref"] = rng.choice(["median", "mean", "last"])
    elif op == "stat": c["stat"] = rng.choice(["q90", "q75", "std", "mad"])
    else: c["mult"] = round(C.clamp(c["mult"] * 2 ** rng.gauss(0, 0.4), 0.1, 5.0), 3)
    return c, op


def calib_fitness(g, tasks):
    """history-only: sigma estimated on the past should match the realised relative deviation of the
    next H history points from the past reference (log-ratio error, lower is better)."""
    errs = []
    for d in tasks:
        for past, fut in N.origins(d):
            s = sigma_of(g, past, d["H"], d["freq"])
            ref = abs(float(np.median(past))) + 1e-9
            real = float(np.quantile(np.abs(np.asarray(fut) - np.median(past)) / ref, 0.9)) + 1e-4
            errs.append(abs(math.log(real / s)))
    return statistics.mean(errs) if errs else 9.0


def evolve_calib(g, tasks, rng, gens, children):
    cur, f = g, calib_fitness(g, tasks); curve = [round(f, 4)]
    for _ in range(gens):
        for _c in range(children):
            ch, _op = mutate_calib(cur, rng); fc = calib_fitness(ch, tasks)
            if fc < f - 1e-9: cur, f = ch, fc
        curve.append(round(f, 4))
    return cur, curve


def trust_profile(tasks):
    by = {}
    for d in tasks:
        if d["toto_h"] is not None: by.setdefault(d["cell"], []).append(d["toto_h"])
    return {c: statistics.median(v) for c, v in by.items()}


# ----------------------------------------------------------------------------- features / evidence
def feats(team, d, s, e, m):
    sig = d["_sig"].get(json.dumps(team["numerical"]["calib"], sort_keys=True))
    if sig is None:
        sig = sigma_of(team["numerical"]["calib"], d["history"], d["H"], d["freq"])
        d["_sig"][json.dumps(team["numerical"]["calib"], sort_keys=True)] = sig
    tr = team["numerical"]["trust"].get(d["cell"], 1.0)
    return {"absmag": abs(m - 1), "conf": d["conf"], "wfrac": (e - s) / d["H"], "docbase": d["docbase"],
            "ratio": min(5.0, abs(m - 1) / sig), "trust": min(3.0, tr), "bias": 1.0}


def accept(w, f):
    z = sum(w.get(k, 0.0) * f[k] for k in FEATS); p = 1 / (1 + math.exp(-C.clamp(z, -30, 30)))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def evidence(team, d):
    out = []
    for s, e, m in d["corr"]:
        a = accept(team["retrieval"], feats(team, d, s, e, m))
        if a > 0: out.append((s, e, 1 + a * (m - 1)))
    return out


def run_team(team, d):
    dg = team["decision"]; anchor = d["fc"][ANCHOR]
    if team["numerical"]["trust"].get(d["cell"], 1.0) < dg["trust_gate"]:
        return list(anchor)                          # Toto trusted here: do not touch it
    out = R3.apply_ev(anchor, evidence(team, d), dg["strength"])
    return list(apply_bounded_delta(anchor, out))


C.run_team = run_team


def scorer_credit(team, w, ds):
    t = dict(team); t["retrieval"] = w; g = 0.0
    for d in ds:
        base = d["fc"][ANCHOR]
        for s, e, m in d["corr"]:
            a = accept(w, feats(t, d, s, e, m))
            if a <= 0: continue
            out = list(apply_bounded_delta(base, R3.apply_ev(base, [(s, e, 1 + a * (m - 1))], 1.0)))
            x = d["base_jt"] - C.jt(out, d["truth"]); g += x if x > 0 else 1.5 * x
    return g


def evolve_scorer(team, train, rng, gens=15, pop=16, elite=4, feats_allowed=FEATS):
    ds = [d for d in train if d["corr"]]
    def rnd(): return {k: (rng.gauss(0, 1.0) if k in feats_allowed else 0.0) for k in FEATS}
    sc = sorted(((scorer_credit(team, w, ds), w) for w in [dict(team["retrieval"])] + [rnd() for _ in range(pop)]), key=lambda x: -x[0])
    curve = [round(sc[0][0], 4)]
    for g in range(gens):
        par = [w for _s, w in sc[:elite]]; sd = 0.6 * (1 - g / gens) + 0.05
        kids = [{k: (C.clamp(rng.choice(par).get(k, 0.0) + rng.gauss(0, sd), -4, 4) if k in feats_allowed else 0.0) for k in FEATS}
                for _ in range(pop - elite)]
        sc = sorted(sc[:elite] + [(scorer_credit(team, w, ds), w) for w in kids], key=lambda x: -x[0]); curve.append(round(sc[0][0], 4))
    return sc[0][1], curve


def gen0(trust):
    return {"numerical": {"calib": calib0(), "trust": trust},
            "retrieval": {k: (-4.0 if k == "bias" else 0.0) for k in FEATS},
            "decision": {"strength": 1.0, "trust_gate": 0.0}}


def mutate_decision(dg, rng):
    c = dict(dg)
    if rng.random() < 0.5: c["strength"] = C.clamp(c["strength"] + rng.gauss(0, 0.2), 0.0, 1.0)
    else: c["trust_gate"] = C.clamp(c["trust_gate"] + rng.gauss(0, 0.15), 0.0, 2.0)
    return c, ["d"]


def evolve(train, rng, a, roles, K=3, log=None):
    trust = trust_profile(train)
    folds = C.gfolds(train, K)
    cur = gen0(trust); cur_fit, cur_per = C.fitness(cur, folds)
    lin = dict(team=[dict(gen=0, fit=cur_fit, folds=cur_per)], calib=[], scorer=[])
    fa = FEATS if "numerical" in roles else [f for f in FEATS if f not in ("ratio", "trust")]
    calib = calib0()
    for gi in range(1, a.gens + 1):
        nc = [(cur["numerical"], [])]
        if "numerical" in roles:
            calib, cc = evolve_calib(calib, train, rng, a.calib_gens, a.calib_children); lin["calib"].append(cc)
            nc.append(({"calib": calib, "trust": trust}, ["calib"]))
        rc = [(cur["retrieval"], [])]
        if "retrieval" in roles:
            for n_art, _ in nc[-1:]:
                t = dict(cur); t["numerical"] = n_art
                w, sc = evolve_scorer(t, train, rng, feats_allowed=fa); lin["scorer"].append(sc); rc.append((w, ["scorer"]))
        dc = [(cur["decision"], [])] + [mutate_decision(cur["decision"], rng) for _ in range(a.children)]
        best = None
        for _ in range(a.teams):
            pick = {"numerical": rng.choice(nc), "retrieval": rng.choice(rc), "decision": rng.choice(dc)}
            if all(not pick[r][1] for r in pick): continue
            team = {r: pick[r][0] for r in pick}; f, per = C.fitness(team, folds)
            if best is None or f > best[0]: best = (f, per, team, {r: pick[r][1] for r in pick if pick[r][1]})
        acc = best is not None and best[0] > cur_fit + 1e-6
        if acc: cur, cur_fit, cur_per = best[2], best[0], best[1]
        lin["team"].append(dict(gen=gi, fit=cur_fit, folds=cur_per, accepted=acc, calib=cur["numerical"]["calib"], decision=cur["decision"]))
        if log: log(f"  gen {gi:2d} {'ACCEPT' if acc else 'reject'} fit {cur_fit:+.5f} calib {cur['numerical']['calib']} D {cur['decision']}")
    return cur, lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=8); ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--teams", type=int, default=24); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--calib-gens", type=int, default=3); ap.add_argument("--calib-children", type=int, default=12)
    ap.add_argument("--roles", default="numerical,retrieval,decision")
    ap.add_argument("--open", choices=["cv", "dev", "test"], default="cv")
    ap.add_argument("--cache", default=".scratch/self_evolving/nrd_cache.json")
    a = ap.parse_args(); roles = tuple(a.roles.split(","))
    C.FOLD_MODE = "strat"; C.STD_W = 0.25
    TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
    D = [R3.prep_task(d, TH) for d in json.load(open(a.cache))]
    for d in D: d["_sig"] = {}
    train = [d for d in D if d["part"] == "train"]
    res = dict(args=vars(a), cv={}, final={})
    outer = C.gfolds(train, 3)
    for seed in a.seeds:
        per = []
        for k in range(3):
            tr = [d for j in range(3) if j != k for d in outer[j]]
            team, lin = evolve(tr, random.Random(seed * 100 + k), a, roles, K=2)
            s = C.summarize(team, outer[k]); per.append(dict(s, team=team, calib_curve=lin["calib"][-1] if lin["calib"] else None))
            print(f"seed {seed} fold {k}: held-out joint {s['joint_gain']:+.2%} sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} "
                  f"W/R {s['wins']}/{s['regressions']} calib {team['numerical']['calib']} D {team['decision']}", flush=True)
        res["cv"][seed] = per
    if a.open in ("dev", "test"):
        dev = [d for d in D if d["part"] == "dev"]; test = [d for d in D if d["part"] == "public_test"]
        for seed in a.seeds:
            team, lin = evolve(train, random.Random(seed), a, roles, K=3)
            r = dict(team=team, lineage=lin, train=C.summarize(team, train), dev=C.summarize(team, dev))
            r["dev_gate_pass"] = r["dev"]["smae_gain"] >= 0 and r["dev"]["srmse_gain"] >= 0
            if a.open == "test": r["test_exploratory"] = C.summarize(team if r["dev_gate_pass"] else gen0(trust_profile(train)), test)
            print(f"seed {seed}: train {r['train']['joint_gain']:+.2%} dev {r['dev']['joint_gain']:+.2%} "
                  f"(sMAE {r['dev']['smae_gain']:+.2%}, sRMSE {r['dev']['srmse_gain']:+.2%}, W/R {r['dev']['wins']}/{r['dev']['regressions']}) gate {r['dev_gate_pass']}"
                  + (f" | test(exploratory) sMAE {r['test_exploratory']['smae_gain']:+.2%} sRMSE {r['test_exploratory']['srmse_gain']:+.2%} W/R {r['test_exploratory']['wins']}/{r['test_exploratory']['regressions']}" if "test_exploratory" in r else ""), flush=True)
            res["final"][seed] = r
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
