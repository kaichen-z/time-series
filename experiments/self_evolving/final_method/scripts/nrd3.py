"""Two-layer evolution on Dr-CiK (2026-09-26).

Layer 1 -- role-local self-evolution with dense, attributable fitness:
  Numerical : evolves the dictionary (nrd_dict.py; history-only hindcasts, MAP-Elites by morphology
              cell).  Its archive also yields a Toto-trust profile: per cell, how much the best
              member beats Toto in hindcast.
  Retrieval : evolves per-event-type extraction rules (enable, direction, magnitude calibration,
              window, confidence floor).  Fitness is per correction on Train: did applying this
              single transformed correction to Toto help or hurt (do-no-harm).
Layer 2 -- 3-agent co-evolution: each generation Numerical offers a new dictionary snapshot,
  Retrieval its best rule sets, Decision a few low-DOF children; joint teams are scored on
  stratified Train folds and the round rolls back unless the team improves.
Decision has 3 parameters: blend cap w, trust margin, evidence strength.  Gen0 = exact Toto.
"""
from __future__ import annotations
import argparse, copy, json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd_coevolve as C
import nrd_dict as N
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

ANCHOR = "toto_2_0"
TYPES = {"closure": ["holiday", "closure", "closed", "shutdown", "strike", "suspend", "vacation", "off"],
         "outage": ["outage", "maintenance", "repair", "disrupt", "delay", "cancel", "failure", "downtime"],
         "surge": ["promotion", "festival", "surge", "spike", "launch", "event", "concert", "sale", "peak"],
         "weather": ["storm", "flood", "heatwave", "rain", "snow", "wind", "heat"],
         "baseline": ["baseline", "methodology", "parameter", "calibration", "specification", "typical", "normal operating", "overview"]}
SNAP = []; TRUST = []


# ----------------------------------------------------------------------------- task prep
def prep_task(d, TH):
    C.prep(d)
    r = TH[d["tid"]]; es, res = [], []
    for k, (_p, fut) in zip((1, 2), N.origins(d)):
        es.append(N.jt(r[f"o{k}"], list(fut)))
        res += [abs(a - b) / (abs(b) + 1e-9) for a, b in zip(r[f"o{k}"], fut)]
    d["toto_h"] = statistics.mean(es) if es else None
    d["sigma"] = min(1.0, statistics.median(res)) if res else 0.5 * d["f_range"]   # Toto relative residual scale
    cnt = {t: sum(d["doclow"].count(w) for w in ws) for t, ws in TYPES.items()}
    d["etype"] = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else "other"
    d["cell"] = N.morph(d); d["mcache"] = {}
    nb = sum(d["doclow"].count(w) for w in C.BASE_LEX); ne = sum(d["doclow"].count(w) for w in C.EVENT_LEX)
    d["docbase"] = nb / (nb + ne + 1e-9)
    return d


def member_info(d, m):
    k = N.key(m)
    if k not in d["mcache"]:
        d["mcache"][k] = (list(N.run_member(m, d["history"], d["H"], d["freq"])), N.hind_err(m, d))
    return d["mcache"][k]


# ----------------------------------------------------------------------------- Retrieval rules
def rules0():   # Gen0: every type disabled -> no evidence
    return {t: {"on": False, "dir": 1.0, "mode": "llm", "s": 1.0, "wf": 1.0, "cmin": 0.0,
                "maxdb": 1.0, "magsupp": False, "maxmag": 1.0}
            for t in list(TYPES) + ["other"]}


def transform(rule, d, s, e, m):
    if not rule["on"] or d["conf"] < rule["cmin"] or d["docbase"] > rule["maxdb"]: return None
    if abs(m - 1) > rule["maxmag"]: return None
    if rule["magsupp"] and abs(m - 1) > d["f_range"] + 1e-9: return None
    if rule["mode"] == "llm": mult = 1 + rule["dir"] * rule["s"] * (m - 1)
    else: mult = 1 + rule["dir"] * rule["s"] * d["sigma"] * (1 if m >= 1 else -1)
    e2 = s + max(1, round((e - s) * rule["wf"]))
    return (s, min(e2, d["H"]), max(0.2, min(3.0, mult)))


FEATS = ["absmag", "conf", "wfrac", "docbase", "magsupp", "bias"]


def feats(d, s, e, m):
    return {"absmag": abs(m - 1), "conf": d["conf"], "wfrac": (e - s) / d["H"], "docbase": d["docbase"],
            "magsupp": 1.0 if abs(m - 1) <= d["f_range"] + 1e-9 else 0.0, "bias": 1.0}


def accept(w, f):
    z = sum(w.get(k, 0.0) * f[k] for k in FEATS); p = 1 / (1 + C.math.exp(-C.clamp(z, -30, 30)))
    return 1.0 if p > 0.6 else (0.5 if p > 0.35 else 0.0)


def evidence(rules, d):
    out = []; sc = rules.get("_scorer")
    for s, e, m in d["corr"]:
        if sc is not None:
            a = accept(sc, feats(d, s, e, m))
            if a > 0: out.append((s, e, 1 + a * (m - 1)))
            continue
        t = transform(rules[d["etype"]], d, s, e, m)
        if t: out.append(t)
    return out


def scorer_credit(w, ds):
    g = 0.0
    for d in ds:
        base = d["fc"][ANCHOR]
        for s, e, m in d["corr"]:
            a = accept(w, feats(d, s, e, m))
            if a <= 0: continue
            out = list(apply_bounded_delta(base, apply_ev(base, [(s, e, 1 + a * (m - 1))], 1.0)))
            x = d["base_jt"] - C.jt(out, d["truth"]); g += x if x > 0 else 1.5 * x
    return g


def evolve_scorer(w0, train, rng, gens=15, pop=16, elite=4):
    """Retrieval self-evolution of its evidence-validation artifact (per-correction credit, Train)."""
    ds = [d for d in train if d["corr"]]
    sc = sorted(((scorer_credit(w, ds), w) for w in [dict(w0)] + [{k: rng.gauss(0, 1.0) for k in FEATS} for _ in range(pop)]),
                key=lambda x: -x[0]); curve = [round(sc[0][0], 4)]
    for g in range(gens):
        par = [w for _s, w in sc[:elite]]; sd = 0.6 * (1 - g / gens) + 0.05
        kids = [{k: C.clamp(rng.choice(par).get(k, 0.0) + rng.gauss(0, sd), -4, 4) for k in FEATS} for _ in range(pop - elite)]
        sc = sorted(sc[:elite] + [(scorer_credit(w, ds), w) for w in kids], key=lambda x: -x[0]); curve.append(round(sc[0][0], 4))
    return sc[0][1], curve


def apply_ev(base, ev, strength):
    out = list(base)
    for s, e, mult in ev:
        mm = 1 + strength * (mult - 1)
        for i in range(s, min(e, len(out))): out[i] = base[i] * mm
    return out


def corr_gain(rule, d):
    """per-correction do-no-harm credit for one task (dense Retrieval fitness)."""
    base = d["fc"][ANCHOR]; g = 0.0
    for s, e, m in d["corr"]:
        t = transform(rule, d, s, e, m)
        if not t: continue
        out = list(apply_bounded_delta(base, apply_ev(base, [t], 1.0)))
        x = d["base_jt"] - C.jt(out, d["truth"]); g += x if x > 0 else 1.5 * x
    return g


def mutate_rule(r, rng):
    c = copy.deepcopy(r); op = rng.choice(["on", "dir", "mode", "s", "s", "wf", "cmin", "maxdb", "maxdb", "magsupp", "maxmag"])
    if op == "on": c["on"] = not c["on"]
    elif op == "dir": c["dir"] = -c["dir"]
    elif op == "mode": c["mode"] = "resid" if c["mode"] == "llm" else "llm"
    elif op == "s": c["s"] = C.clamp(c["s"] * (2 ** rng.gauss(0, 0.6)), 0.05, 3.0)
    elif op == "wf": c["wf"] = C.clamp(c["wf"] + rng.gauss(0, 0.2), 0.1, 1.0)
    elif op == "cmin": c["cmin"] = C.clamp(c["cmin"] + rng.gauss(0, 0.15), 0.0, 0.95)
    elif op == "maxdb": c["maxdb"] = C.clamp(c["maxdb"] + rng.gauss(0, 0.25), 0.0, 1.0)
    elif op == "magsupp": c["magsupp"] = not c["magsupp"]
    else: c["maxmag"] = C.clamp(c["maxmag"] + rng.gauss(0, 0.15), 0.02, 1.0)
    return c, op


def evolve_rules(rules, train, rng, gens, children):
    """Retrieval self-evolution: per event type hill-climb on per-correction credit (Train only)."""
    by = {}
    for d in train:
        if d["corr"]: by.setdefault(d["etype"], []).append(d)
    rules = copy.deepcopy(rules); curve = {}
    for t, ds in by.items():
        cur = rules[t]; f = sum(corr_gain(cur, d) for d in ds); hist = [round(f, 4)]
        for _ in range(gens):
            for _c in range(children):
                ch, _op = mutate_rule(cur, rng); fc = sum(corr_gain(ch, d) for d in ds)
                if fc > f + 1e-9: cur, f = ch, fc
            hist.append(round(f, 4))
        rules[t] = cur; curve[t] = hist
    return rules, curve


# ----------------------------------------------------------------------------- Decision + team
def gen0():
    return {"numerical": {"snap": 0}, "retrieval": rules0(), "decision": {"w": 0.0, "margin": 0.0, "strength": 1.0}}


def run_team(team, d):
    dg = team["decision"]; anchor = d["fc"][ANCHOR]; out = list(anchor)
    trust = TRUST[team["numerical"]["snap"]]
    if dg["w"] > 0 and d["toto_h"] is not None and trust.get(d["cell"], 1.0) < -dg["margin"]:
        best = None
        for m in SNAP[team["numerical"]["snap"]]:
            f, e = member_info(d, m)
            if e is not None and (best is None or e < best[1]): best = (f, e)
        if best is not None and best[1] < d["toto_h"]:
            w = dg["w"]; out = [(1 - w) * a + w * b for a, b in zip(anchor, best[0])]
    out = apply_ev(out, evidence(team["retrieval"], d), dg["strength"])
    return list(apply_bounded_delta(anchor, out))


C.run_team = run_team


NO_BLEND = False


def mutate_decision(dg, rng):
    c = copy.deepcopy(dg); op = rng.choice(["margin", "strength"] if NO_BLEND else ["w", "margin", "strength"])
    if op == "w": c["w"] = C.clamp(c["w"] + rng.gauss(0, 0.15), 0.0, 0.5)
    elif op == "margin": c["margin"] = C.clamp(c["margin"] + rng.gauss(0, 0.05), -0.2, 0.5)
    else: c["strength"] = C.clamp(c["strength"] + rng.gauss(0, 0.2), 0.0, 1.0)
    return c, [op]


def evolve(train, rng, a, roles, K=3, log=None):
    A = N.Archive(train, {d["tid"]: d["toto_h"] for d in train if d["toto_h"] is not None})
    for m in N.gen0_members(): A.offer(m)
    SNAP.append(A.members()); TRUST.append(dict(A.summary()))
    folds = C.gfolds(train, K)
    cur = gen0(); cur["numerical"]["snap"] = len(SNAP) - 1
    cur_fit, cur_per = C.fitness(cur, folds)
    lin = [dict(gen=0, fit=cur_fit, folds=cur_per, archive=A.summary())]
    rules_pool = rules0(); rcurve = []
    for gi in range(1, a.gens + 1):
        nc = [(cur["numerical"], [])]
        if "numerical" in roles:
            for _ in range(a.dict_gens):
                pool = A.members()
                for _c in range(a.dict_children):
                    ch, _op = N.mutate_member(rng.choice(pool), pool, rng); A.offer(ch)
            SNAP.append(A.members()); TRUST.append(dict(A.summary())); nc.append(({"snap": len(SNAP) - 1}, ["dict"]))
        rc = [(cur["retrieval"], [])]
        if "retrieval" in roles:
            if a.retrieval_mode == "scorer":
                w, curve = evolve_scorer(rules_pool.get("_scorer", {"bias": -4.0}), train, rng)
                rules_pool = dict(rules_pool); rules_pool["_scorer"] = w
            else:
                rules_pool, curve = evolve_rules(rules_pool, train, rng, a.rule_gens, a.rule_children)
            rcurve.append(curve); rc.append((rules_pool, ["rules"]))
            # also let the team try the evolved rules restricted to each single type
            for t in list(rules_pool)[:0]: pass
        dc = [(cur["decision"], [])] + [mutate_decision(cur["decision"], rng) for _ in range(a.children)]
        best = None
        for _ in range(a.teams):
            pick = {"numerical": rng.choice(nc), "retrieval": rng.choice(rc), "decision": rng.choice(dc)}
            if all(not pick[r][1] for r in pick): continue
            team = {r: pick[r][0] for r in pick}; f, per = C.fitness(team, folds)
            if best is None or f > best[0]: best = (f, per, team, {r: pick[r][1] for r in pick if pick[r][1]})
        acc = best is not None and best[0] > cur_fit + 1e-6
        if acc: cur, cur_fit, cur_per = best[2], best[0], best[1]
        lin.append(dict(gen=gi, fit=cur_fit, folds=cur_per, accepted=acc, changes=best[3] if acc else {},
                        archive=A.summary(), decision=dict(cur["decision"])))
        if log: log(f"  gen {gi:2d} {'ACCEPT' if acc else 'reject'} fit {cur_fit:+.5f} D {cur['decision']} "
                    f"archive {statistics.mean(A.summary().values()):+.4f} {best[3] if acc else ''}")
    return cur, dict(team=lin, retrieval=rcurve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=12); ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--teams", type=int, default=24); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dict-gens", type=int, default=3); ap.add_argument("--dict-children", type=int, default=24)
    ap.add_argument("--rule-gens", type=int, default=2); ap.add_argument("--rule-children", type=int, default=8)
    ap.add_argument("--roles", default="numerical,retrieval,decision")
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--reg-penalty", type=float, default=0.5)
    ap.add_argument("--retrieval-mode", choices=["rules", "scorer"], default="rules")
    ap.add_argument("--open", choices=["cv", "dev", "test"], default="cv")
    a = ap.parse_args(); roles = tuple(a.roles.split(","))
    C.FOLD_MODE = "strat"; C.STD_W = 0.25
    global NO_BLEND; NO_BLEND = a.no_blend; C.REG_PEN = a.reg_penalty
    TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
    D = [prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))]
    train = [d for d in D if d["part"] == "train"]
    res = dict(args=vars(a), cv={}, final={})
    outer = C.gfolds(train, 3)
    for seed in a.seeds:
        per = []
        for k in range(3):
            tr = [d for j in range(3) if j != k for d in outer[j]]
            team, lin = evolve(tr, random.Random(seed * 100 + k), a, roles, K=2)
            s = C.summarize(team, outer[k]); per.append(dict(s, decision=team["decision"]))
            print(f"seed {seed} fold {k}: held-out joint {s['joint_gain']:+.2%} sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} "
                  f"W/R {s['wins']}/{s['regressions']} D {team['decision']} rules_on {[t for t, r in team['retrieval'].items() if not t.startswith('_') and r['on']]} scorer {'_scorer' in team['retrieval']}", flush=True)
        res["cv"][seed] = per
    if a.open in ("dev", "test"):
        dev = [d for d in D if d["part"] == "dev"]; test = [d for d in D if d["part"] == "public_test"]
        for seed in a.seeds:
            team, lin = evolve(train, random.Random(seed), a, roles, K=3, log=print)
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
