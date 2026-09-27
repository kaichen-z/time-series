"""nrd11 (dictionary evolution -> 3-agent co-evolution) with GEPA/FunSearch-style search (2026-09-27).

* Instance-level Pareto selection (GEPA, arXiv 2507.19457): every evaluated team is archived with its
  per-task Train gains; the front = teams that are best on at least one task that has corrections.
  Parents are sampled from the front, weighted by the number of tasks they win (no greedy best-only).
* Island model (FunSearch/AlphaEvolve): I islands evolve independently; every M generations each island
  receives the other islands' fittest team.
* Final ensemble: the fittest team of every island votes per correction; a correction is applied if a
  majority accepts it (strength = mean accepted strength).
Numerical dictionary stage and the feature set are exactly nrd11.  Train CV + dev only.
"""
from __future__ import annotations
import argparse, copy, json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd11 as N11          # patches nrd4 features + dictionary stage state
import nrd4 as N4
import nrd_dict as N
import nrd_coevolve as C
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

_single = N4.run_team


def ens_run(team, d):
    if "ens" not in team: return _single(team, d)
    base = d["fc"][N4.ANCHOR]; H = len(base); votes = {}
    members = team["ens"]
    for t in members:
        if t["numerical"]["trust"].get(d["cell"], 1.0) < t["decision"]["trust_gate"]: continue
        for s, e, m in N4.evidence(t, d):
            k = (s, e); votes.setdefault(k, []).append((m - 1) * t["decision"]["strength"])
    ev = [(s, e, 1 + statistics.mean(v)) for (s, e), v in votes.items() if len(v) * 2 > len(members)]
    out = N4.R3.apply_ev(base, ev, 1.0)
    return list(apply_bounded_delta(base, out))


C.run_team = ens_run
N4.run_team = ens_run


def task_gains(team, ds): return [d["base_jt"] - C.jt(_single(team, d), d["truth"]) for d in ds]


def front(archive, ctasks):
    """indices of archive teams that are (weakly) best on >=1 correction task, with win counts."""
    wins = {}
    for j in range(len(ctasks)):
        best = max(a["g"][j] for a in archive)
        if best <= 1e-12: continue
        for i, a in enumerate(archive):
            if a["g"][j] >= best - 1e-12: wins[i] = wins.get(i, 0) + 1
    return wins


def evolve(train, rng, a, roles, K=3, log=None):
    # stage 1: Numerical dictionary evolution (history only), as nrd11
    A, curve = N.evolve_dictionary(train, {d["tid"]: d["toto_h"] for d in train if d["toto_h"] is not None},
                                   gens=a.dict_gens_stage1, children=24, rng=rng)
    N11.STATE.update(A=A, cell=A.summary(), key=N11.STATE["key"] + 1, curve=[round(c["mean_rel"], 4) for c in curve])
    trust = N4.trust_profile(train); folds = C.gfolds(train, K)
    ctasks = [d for d in train if d["corr"]]
    islands = []
    for i in range(a.islands):
        t0 = N4.gen0(trust); f, per = C.fitness(t0, folds)
        islands.append(dict(rng=random.Random(rng.random()), archive=[dict(team=t0, fit=f, g=task_gains(t0, ctasks))],
                            calib=N4.calib0()))
    for gi in range(1, a.gens + 1):
        for isl in islands:
            r = isl["rng"]; arc = isl["archive"]; w = front(arc, ctasks)
            if w:
                idx = r.choices(list(w), weights=list(w.values()))[0]
            else:
                idx = max(range(len(arc)), key=lambda i: arc[i]["fit"])
            parent = arc[idx]["team"]
            nc = [(parent["numerical"], [])]
            isl["calib"], _cc = N4.evolve_calib(isl["calib"], train, r, a.calib_gens, a.calib_children)
            nc.append(({"calib": isl["calib"], "trust": trust}, ["calib"]))
            t = dict(parent); t["numerical"] = nc[-1][0]
            wsc, _sc = N4.evolve_scorer(t, train, r, feats_allowed=N4.FEATS)
            rc = [(parent["retrieval"], []), (wsc, ["scorer"])]
            dc = [(parent["decision"], [])] + [N4.mutate_decision(parent["decision"], r) for _ in range(a.children)]
            for _ in range(a.teams):
                pick = {"numerical": r.choice(nc), "retrieval": r.choice(rc), "decision": r.choice(dc)}
                if all(not pick[k][1] for k in pick): continue
                team = {k: pick[k][0] for k in pick}; f, per = C.fitness(team, folds)
                arc.append(dict(team=team, fit=f, g=task_gains(team, ctasks)))
            # keep the archive bounded: front members + top by fitness
            w = front(arc, ctasks); keep = set(w) | set(sorted(range(len(arc)), key=lambda i: -arc[i]["fit"])[:a.keep])
            isl["archive"] = [arc[i] for i in sorted(keep)]
        if gi % a.migrate == 0 and len(islands) > 1:
            bests = [max(isl["archive"], key=lambda x: x["fit"]) for isl in islands]
            for i, isl in enumerate(islands):
                for j, b in enumerate(bests):
                    if j != i: isl["archive"].append(copy.deepcopy(b))
        if log: log(f"  gen {gi:2d} " + " | ".join(f"isl{i} best {max(x['fit'] for x in isl['archive']):+.4f} front {len(front(isl['archive'], ctasks))}" for i, isl in enumerate(islands)))
    bests = [max(isl["archive"], key=lambda x: x["fit"])["team"] for isl in islands]
    team = {"ens": bests} if a.ensemble else max((max(isl["archive"], key=lambda x: x["fit"]) for isl in islands), key=lambda x: x["fit"])["team"]
    return team, dict(dict_curve=N11.STATE["curve"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--children", type=int, default=5); ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--islands", type=int, default=3); ap.add_argument("--migrate", type=int, default=4)
    ap.add_argument("--keep", type=int, default=8); ap.add_argument("--no-ensemble", dest="ensemble", action="store_false")
    ap.add_argument("--calib-gens", type=int, default=2); ap.add_argument("--calib-children", type=int, default=8)
    ap.add_argument("--dict-gens-stage1", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--open", choices=["cv", "dev", "test"], default="dev"); ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args(); roles = ("numerical", "retrieval", "decision")
    C.FOLD_MODE = "strat"; C.STD_W = 0.25
    TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
    D = [N4.R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))]
    for d in D: d["_sig"] = {}
    train = [d for d in D if d["part"] == "train"]; dev = [d for d in D if d["part"] == "dev"]
    outer = C.gfolds(train, 3); res = {}
    for seed in a.seeds:
        per = []
        for k in (range(0) if a.skip_cv else range(3)):
            tr = [d for j in range(3) if j != k for d in outer[j]]
            team, lin = evolve(tr, random.Random(seed * 100 + k), a, roles, K=2)
            s = C.summarize(team, outer[k]); per.append(s)
            print(f"seed {seed} fold {k}: held-out joint {s['joint_gain']:+.2%} sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} W/R {s['wins']}/{s['regressions']}", flush=True)
        r = dict(cv=per)
        if a.open in ("dev", "test"):
            team, lin = evolve(train, random.Random(seed), a, roles, K=3)
            r["train"] = C.summarize(team, train); r["dev"] = C.summarize(team, dev)
            r["dev_gate_pass"] = r["dev"]["smae_gain"] >= 0 and r["dev"]["srmse_gain"] >= 0
            print(f"seed {seed}: train {r['train']['joint_gain']:+.2%} dev {r['dev']['joint_gain']:+.2%} (sMAE {r['dev']['smae_gain']:+.2%}, sRMSE {r['dev']['srmse_gain']:+.2%}, W/R {r['dev']['wins']}/{r['dev']['regressions']}) gate {r['dev_gate_pass']}", flush=True)
            if a.open == "test":
                te = [d for d in D if d["part"] == "public_test"]; t = C.summarize(team, te); r["test_exploratory_ungated"] = t
                print(f"   test(exploratory, ignoring the failed dev gate): sMAE {t['smae_gain']:+.2%} sRMSE {t['srmse_gain']:+.2%} joint {t['joint_gain']:+.2%} W/R {t['wins']}/{t['regressions']}", flush=True)
            r["team"] = team
        res[seed] = r
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
