"""N/R/D co-evolution with a self-evolving Numerical dictionary (2026-09-26).

Numerical : evolves the dictionary itself (nrd_dict.py) with history-only hindcast fitness and a
            MAP-Elites archive; each generation it offers a new sealed dictionary snapshot.
Retrieval : document -> evidence artifact (unchanged from nrd_coevolve.py).
Decision  : candidate-only.  For each task it ranks Toto and the dictionary members by their
            hindcast error on that task's own history (history-only), and may blend the best
            member into Toto (Toto weight >= 0.5) when its hindcast advantage exceeds a threshold;
            then accepts/shrinks/rejects evidence.  Gen0 = exact Toto.
Team selection uses grouped train folds (labels are Host-only); dev and test are opened once.
"""
from __future__ import annotations
import argparse, copy, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd_coevolve as C
import nrd_dict as N
from evolving_loop.adjustment.post_adjust import apply_bounded_delta
from common.metrics import drcik_point_metrics

ANCHOR = "toto_2_0"
SNAP = []          # sealed dictionary snapshots (list of member lists)


def prep_task(d, toto_hind):
    C.prep(d)
    r = toto_hind[d["tid"]]; es = []
    for k, (_past, fut) in zip((1, 2), N.origins(d)): es.append(N.jt(r[f"o{k}"], list(fut)))
    d["toto_h"] = statistics.mean(es) if es else None
    d["mcache"] = {}
    return d


def member_info(d, m):
    """(future forecast on full history, hindcast error) -- both history-only."""
    k = N.key(m)
    if k not in d["mcache"]:
        f = list(N.run_member(m, d["history"], d["H"], d["freq"]))
        d["mcache"][k] = (f, N.hind_err(m, d))
    return d["mcache"][k]


def gen0():
    t = C.gen0()
    t["numerical"] = {"snap": 0}
    t["decision"].update({"tau": 0.5, "w": 0.0})
    return t


def decide(team, d):
    dg = team["decision"]; anchor = d["fc"][ANCHOR]; out = list(anchor)
    if dg["w"] > 0 and d["toto_h"] is not None:
        best = None
        for m in SNAP[team["numerical"]["snap"]]:
            f, e = member_info(d, m)
            if e is not None and (best is None or e < best[1]): best = (f, e)
        if best is not None:
            adv = (d["toto_h"] - best[1]) / (d["toto_h"] + 1e-9)     # relative hindcast advantage
            if adv > dg["tau"]:
                w = dg["w"]; out = [(1 - w) * a + w * b for a, b in zip(anchor, best[0])]
    base = list(out)
    for c in C.retrieval_evidence(team["retrieval"], d):
        s = sum(dg["ev"].get(k, 0.0) * v for k, v in c["feat"].items())
        p = 1 / (1 + math.exp(-C.clamp(s, -30, 30)))
        a = 0.0 if p <= dg["low"] else (1.0 if p > dg["high"] else 0.5)
        if a <= 0: continue
        mlt = 1 + a * (c["mult"] - 1); st, en = c["win"]
        for i in range(st, min(en, len(out))): out[i] = base[i] * mlt
    return list(apply_bounded_delta(anchor, out))


C.run_team = decide     # nrd_coevolve's gains()/summarize() now execute this team


def mutate_decision(dg, rng):
    c = copy.deepcopy(dg); op = rng.choice(["tau", "w", "w", "ev", "ev", "low_high"])
    if op == "tau": c["tau"] = C.clamp(c["tau"] + rng.gauss(0, 0.15), -0.5, 0.95)
    elif op == "w": c["w"] = C.clamp(c["w"] + rng.gauss(0, 0.15), 0.0, 0.5)
    elif op == "ev":
        k = rng.choice(list(c["ev"])); c["ev"][k] = C.clamp(c["ev"][k] + rng.gauss(0, 0.6), -4, 4); op = f"ev.{k}"
    else:
        c["low"] = C.clamp(c["low"] + rng.gauss(0, 0.1), 0.05, 0.95); c["high"] = C.clamp(c["high"] + rng.gauss(0, 0.1), 0.05, 0.95)
        if c["low"] > c["high"]: c["low"], c["high"] = c["high"], c["low"]
    return c, [op]


def refit_decision(cur, rg, data, rng, gens=10, pop=14, elite=4):
    team = copy.deepcopy(cur); team["retrieval"] = rg
    keys = list(team["decision"]["ev"])
    def score(ev):
        t = copy.deepcopy(team); t["decision"]["ev"] = ev; return C.robust(C.gains(t, data))
    scored = sorted(((score(e), e) for e in [dict(team["decision"]["ev"])] + [{k: rng.gauss(0, 1.0) for k in keys} for _ in range(pop)]), key=lambda x: -x[0])
    for g in range(gens):
        par = [e for _s, e in scored[:elite]]; sd = 0.6 * (1 - g / gens) + 0.05
        kids = [{k: C.clamp(rng.choice(par)[k] + rng.gauss(0, sd), -4, 4) for k in keys} for _ in range(pop - elite)]
        scored = sorted(scored[:elite] + [(score(e), e) for e in kids], key=lambda x: -x[0])
    d = copy.deepcopy(team["decision"]); d["ev"] = scored[0][1]; return d


def evolve(train, rng, gens, children, teams, dict_gens_per_round, dict_children, roles, K=3, log=None):
    """Joint team evolution; Numerical advances its own archive by history-only hindcasts."""
    A = N.Archive(train, {d["tid"]: d["toto_h"] for d in train if d["toto_h"] is not None})
    for m in N.gen0_members(): A.offer(m)
    SNAP.append(A.members()); base_snap = len(SNAP) - 1
    folds = C.gfolds(train, K)
    cur = gen0(); cur["numerical"]["snap"] = base_snap
    cur_fit, cur_per = C.fitness(cur, folds)
    lineage = [dict(gen=0, fit=cur_fit, folds=cur_per, dict=len(SNAP[base_snap]), dict_err=A.summary())]
    for gi in range(1, gens + 1):
        # Numerical child: continue the archive (history-only), seal a new snapshot
        ncands = [(cur["numerical"], [])]
        if "numerical" in roles:
            for _ in range(dict_gens_per_round):
                pool = A.members()
                for _c in range(dict_children):
                    ch, _op = N.mutate_member(rng.choice(pool), pool, rng); A.offer(ch)
            SNAP.append(A.members()); ncands.append(({"snap": len(SNAP) - 1}, ["dict-gen"]))
        rc = [(cur["retrieval"], [])] + ([C.mutate("retrieval", cur["retrieval"], rng, [])
                                          for _ in range(children)] if "retrieval" in roles else [])
        dc = [(cur["decision"], [])] + [mutate_decision(cur["decision"], rng) for _ in range(children)]
        for rg, _ch in rc[:3]: dc.append((refit_decision(cur, rg, train, rng), ["refit"]))
        best = None
        for _ in range(teams):
            pick = {"numerical": rng.choice(ncands), "retrieval": rng.choice(rc), "decision": rng.choice(dc)}
            if all(not pick[r][1] for r in pick): continue
            team = {r: pick[r][0] for r in pick}; f, per = C.fitness(team, folds)
            if best is None or f > best[0]: best = (f, per, team, {r: pick[r][1] for r in pick if pick[r][1]})
        acc = best is not None and best[0] > cur_fit + 1e-6
        if acc: cur, cur_fit, cur_per = best[2], best[0], best[1]
        # the archive keeps improving even if the team gate rejects this round's snapshot
        lineage.append(dict(gen=gi, fit=cur_fit, folds=cur_per, accepted=acc, changes=best[3] if acc else {},
                            dict=len(SNAP[cur["numerical"]["snap"]]), dict_err=A.summary()))
        if log: log(f"  gen {gi:2d} {'ACCEPT' if acc else 'reject'} fit {cur_fit:+.5f} dict {len(SNAP[cur['numerical']['snap']])} "
                    f"archive mean {statistics.mean(A.summary().values()):+.4f} {best[3] if acc else ''}")
    return cur, lineage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=20); ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--teams", type=int, default=40); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dict-gens", type=int, default=3); ap.add_argument("--dict-children", type=int, default=24)
    ap.add_argument("--roles", default="numerical,retrieval,decision")
    ap.add_argument("--open", choices=["cv", "dev", "test"], default="cv")
    a = ap.parse_args(); roles = tuple(a.roles.split(","))
    C.FOLD_MODE = "strat"; C.STD_W = 0.25
    TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
    D = [prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))]
    train = [d for d in D if d["part"] == "train"]
    res = dict(roles=roles, args=vars(a), cv={}, final={})
    outer = C.gfolds(train, 3)
    for seed in a.seeds:
        per = []
        for k in range(3):
            tr = [d for j in range(3) if j != k for d in outer[j]]
            team, lin = evolve(tr, random.Random(seed * 100 + k), a.gens, a.children, a.teams, a.dict_gens, a.dict_children, roles, K=2)
            s = C.summarize(team, outer[k]); s["dict_size"] = len(SNAP[team["numerical"]["snap"]]); per.append(s)
            print(f"seed {seed} outer fold {k}: held-out joint {s['joint_gain']:+.2%} sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} "
                  f"W/R {s['wins']}/{s['regressions']} dict {s['dict_size']} w {team['decision']['w']:.2f} tau {team['decision']['tau']:.2f}", flush=True)
        res["cv"][seed] = per
    if a.open in ("dev", "test"):
        dev = [d for d in D if d["part"] == "dev"]; test = [d for d in D if d["part"] == "public_test"]
        for seed in a.seeds:
            team, lin = evolve(train, random.Random(seed), a.gens, a.children, a.teams, a.dict_gens, a.dict_children, roles, K=3, log=print)
            r = dict(team=team, dictionary=SNAP[team["numerical"]["snap"]], lineage=lin,
                     train=C.summarize(team, train), dev=C.summarize(team, dev))
            r["dev_gate_pass"] = r["dev"]["smae_gain"] >= 0 and r["dev"]["srmse_gain"] >= 0
            if a.open == "test": r["test"] = C.summarize(team if r["dev_gate_pass"] else gen0(), test)
            print(f"seed {seed}: train {r['train']['joint_gain']:+.2%} dev {r['dev']['joint_gain']:+.2%} gate {r['dev_gate_pass']}"
                  + (f" test {r['test']['joint_gain']:+.2%}" if "test" in r else ""), flush=True)
            res["final"][seed] = r
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
