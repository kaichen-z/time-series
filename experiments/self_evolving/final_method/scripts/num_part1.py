"""Numerical part 1: evolve a candidate dictionary of forecast COMBINATION programs (2026-09-27).

A program combines the cached forecasts of the 31 dictionary methods:
    f = sum_i w_i * M_i  [+ c * (M_a - M_b)]   then optional transforms
        - shrink toward the last observed value:  f <- (1-s) f + s * y_last
        - clip into [min(history) - r*range, max(history) + r*range]
Typed mutations: swap a method, perturb a weight, add/remove a term, add/remove the difference term,
toggle/perturb a transform.  Weights of the sum are renormalised to 1 (the difference term is free).
Fitness (Train labels): mean joint-error gain vs the best single method found so far is NOT used; we use
mean gain vs Toto + 0.5 x mean negative gain (penalises tasks made worse).  A population of diverse
elites (different leading method) is kept as the dictionary.
Honest estimate: nested stratified CV (evolve on 2 folds, score the 3rd); then evolve on all Train and
check on Dev.
"""
import copy, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import nrd_coevolve as C
from common.metrics import drcik_point_metrics

D = json.load(open(".scratch/self_evolving/nrd_cache.json"))
_tr = [d for d in D if d["part"] == "train"]
METHODS = sorted(m for m in set().union(*[set(d["fc"]) for d in _tr]) if sum(m in d["fc"] for d in _tr) >= 0.9 * len(_tr))   # missing -> Toto


def jt(f, t): x = drcik_point_metrics(t, f, cap=5.0); return x["smae"] + x["srmse"]


for d in D:
    d["base_jt"] = jt(d["fc"]["toto_2_0"], d["truth"])
    h = np.asarray(d["history"], float); d["_last"] = float(h[-1]); d["_lo"], d["_hi"] = float(h.min()), float(h.max())


def run(p, d):
    fc = d["fc"]; tot = sum(w for _, w in p["terms"]) or 1.0
    f = np.zeros(d["H"])
    for m, w in p["terms"]: f += (w / tot) * np.asarray(fc.get(m, fc["toto_2_0"]), float)
    if p.get("diff"):
        a, b, c = p["diff"]; f += c * (np.asarray(fc.get(a, fc["toto_2_0"]), float) - np.asarray(fc.get(b, fc["toto_2_0"]), float))
    if p.get("shrink", 0) > 0: f = (1 - p["shrink"]) * f + p["shrink"] * d["_last"]
    if p.get("clip") is not None:
        r = p["clip"] * (d["_hi"] - d["_lo"] + 1e-9); f = np.clip(f, d["_lo"] - r, d["_hi"] + r)
    f = np.where(np.isfinite(f), f, np.asarray(fc["toto_2_0"], float))
    return f.tolist()


PEN = float(__import__("os").environ.get("PEN", "3.0"))   # do-no-harm penalty on tasks made worse


def fit(p, ds):
    gs = [d["base_jt"] - jt(run(p, d), d["truth"]) for d in ds]
    return statistics.mean(gs) + PEN * statistics.mean(min(0.0, g) for g in gs)


def seed_prog(m): return {"terms": [[m, 1.0]], "diff": None, "shrink": 0.0, "clip": None}


def mutate(p, rng):
    c = copy.deepcopy(p); op = rng.choice(["swap", "w", "w", "add", "drop", "diff", "shrink", "clip"])
    if op == "swap": i = rng.randrange(len(c["terms"])); c["terms"][i][0] = rng.choice(METHODS)
    elif op == "w": i = rng.randrange(len(c["terms"])); c["terms"][i][1] = float(np.clip(c["terms"][i][1] * 2 ** rng.gauss(0, 0.5), 0.02, 5))
    elif op == "add" and len(c["terms"]) < 4: c["terms"].append([rng.choice(METHODS), round(rng.uniform(0.05, 0.5), 3)])
    elif op == "drop" and len(c["terms"]) > 1: c["terms"].pop(rng.randrange(len(c["terms"])))
    elif op == "diff":
        c["diff"] = None if (c["diff"] and rng.random() < 0.4) else [rng.choice(METHODS), rng.choice(METHODS), round(rng.gauss(0, 0.2), 3)]
    elif op == "shrink": c["shrink"] = float(np.clip(c["shrink"] + rng.gauss(0, 0.1), 0, 0.6))
    else: c["clip"] = None if c["clip"] is not None and rng.random() < 0.4 else rng.choice([0.0, 0.1, 0.25, 0.5])
    return c, op


def lead(p): return max(p["terms"], key=lambda t: t[1])[0]


def evolve(ds, gens, pop, rng, log=None):
    P = [(fit(seed_prog(m), ds), seed_prog(m)) for m in METHODS]
    P.sort(key=lambda x: -x[0]); curve = [round(P[0][0], 4)]
    for g in range(1, gens + 1):
        par = [p for _, p in P[:pop]]
        kids = [mutate(rng.choice(par), rng)[0] for _ in range(pop * 2)]
        P = sorted(P + [(fit(k, ds), k) for k in kids], key=lambda x: -x[0])
        # diversity: keep the best program per leading method, then fill by fitness
        seen, keep = set(), []
        for f, p in P:
            if lead(p) not in seen: seen.add(lead(p)); keep.append((f, p))
        rest = [x for x in P if x not in keep]
        P = (keep + rest)[:pop * 3]; P.sort(key=lambda x: -x[0]); curve.append(round(P[0][0], 4))
        if log: log(f"  gen {g}: best fitness {P[0][0]:+.4f}  best program {json.dumps(P[0][1])}")
    elites = []; seen = set()
    for f, p in P:
        if lead(p) not in seen: seen.add(lead(p)); elites.append(dict(fitness=f, program=p))
        if len(elites) >= 8: break
    return elites, curve


def summ(p, ds):
    sb = so = rb = ro = 0.0; w = r = 0
    for d in ds:
        b = drcik_point_metrics(d["truth"], d["fc"]["toto_2_0"], cap=5.0); o = drcik_point_metrics(d["truth"], run(p, d), cap=5.0)
        sb += b["smae"]; so += o["smae"]; rb += b["srmse"]; ro += o["srmse"]; dj = (b["smae"] + b["srmse"]) - (o["smae"] + o["srmse"])
        w += dj > 1e-9; r += dj < -1e-9
    n = len(ds); return f"sMAE {sb/n:.4f}->{so/n:.4f} ({(sb-so)/sb:+.2%}) sRMSE {rb/n:.4f}->{ro/n:.4f} ({(rb-ro)/rb:+.2%}) W/R {w}/{r}"


def main():
    C.FOLD_MODE = "strat"; rng = random.Random(1)
    train = [d for d in D if d["part"] == "train"]; dev = [d for d in D if d["part"] == "dev"]
    print("methods in dictionary:", len(METHODS))
    folds = C.gfolds(train, 3)
    for k in range(3):
        tr = [d for j in range(3) if j != k for d in folds[j]]
        el, _ = evolve(tr, 30, 12, random.Random(10 + k))
        print(f"CV fold {k}: best program {json.dumps(el[0]['program'])} -> held-out {summ(el[0]['program'], folds[k])}")
    el, curve = evolve(train, 30, 12, rng, log=print)
    print("curve:", curve)
    print("dictionary (diverse elites):")
    for e in el: print(f"  fit {e['fitness']:+.4f}  {json.dumps(e['program'])}")
    best = el[0]["program"]
    print("train:", summ(best, train)); print("dev  :", summ(best, dev))
    json.dump(dict(curve=curve, elites=el), open(".scratch/self_evolving/numerical_part1.json", "w"))


if __name__ == "__main__":
    main()
