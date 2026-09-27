"""Numerical / Retrieval / Decision joint-team co-evolution on Dr-CiK (2026-09-26).

Three typed, mutable artifacts:
  Numerical  : history-only executable portfolio -- which cached dictionary methods are executed
               for every task and how they are aggregated into extra candidates.
  Retrieval  : document -> evidence artifact -- lexicons that classify documents as dated events
               vs. baseline/methodology descriptions, plus selection rules over extracted
               correction windows (confidence, magnitude, window, magnitude calibration).
  Decision   : candidate-only policy -- chooses among the *executed* numerical candidates (anchor
               Toto weight >= 0.5) under history-only gates, and accepts/shrinks/rejects evidence.
Gen0 is exact Toto (empty pool, no evidence accepted).  Each generation every agent proposes
children; joint teams (parent/child per role) compete on grouped train folds; the best team is
promoted only if it beats the current team, otherwise the whole round rolls back.
Labels are used only by the Host fitness.  Dev is opened once after training; test once after dev.
"""
from __future__ import annotations
import argparse, copy, json, math, random, re, statistics
from pathlib import Path
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import apply_bounded_delta

CAP = 5.0
ANCHOR = "toto_2_0"
EVENT_LEX = ["holiday", "closure", "closed", "outage", "strike", "promotion", "maintenance", "shutdown",
             "festival", "storm", "flood", "disrupt", "suspend", "repair", "scheduled for", "effective",
             "heatwave", "cancel", "surge", "spike", "drop", "event", "launch", "renovation", "delay"]
BASE_LEX = ["baseline", "methodology", "parameter", "diagnostic", "calibration", "technical note", "initiali",
            "specification", "overview", "scope", "definition", "typical", "normal operating", "historical",
            "average", "profile", "standard"]
AGGS = ["median", "mean", "trimmed"]


def jt(f, t):
    x = drcik_point_metrics(t, f, cap=CAP)
    return x["smae"] + x["srmse"]


# ----------------------------------------------------------------------------- task prep (history-only)
def prep(d):
    h = d["history"]; H = d["H"]
    lvl = statistics.mean(abs(x) for x in h[-max(H, 8):]) + 1e-9
    d["f_cv"] = statistics.pstdev(h[-max(2 * H, 16):]) / lvl
    toto = d["fc"][ANCHOR]
    others = [v for k, v in d["fc"].items() if k != ANCHOR]
    if others:
        d["f_disp"] = statistics.mean(statistics.pstdev([v[i] for v in d["fc"].values()]) for i in range(H)) / lvl
    else:
        d["f_disp"] = 0.0
    d["f_lvl"] = lvl
    # bounded historical deviation used by Retrieval magnitude support
    if len(h) >= 4:
        m = statistics.median(h) or (statistics.mean(abs(x) for x in h) + 1e-9)
        q = statistics.quantiles(h, n=10); d["f_range"] = max(abs(q[-1] - m), abs(q[0] - m)) / abs(m)
    else:
        d["f_range"] = 0.5
    d["doclow"] = " ".join(d["docs"]).lower()
    d["base_jt"] = jt(toto, d["truth"])
    return d


# ----------------------------------------------------------------------------- genomes
def gen0():
    return {
        "numerical": {"pool": [], "agg": "median"},
        "retrieval": {"min_conf": 0.0, "max_mag": 1.0, "max_wfrac": 1.0, "mag_scale": 1.0,
                      "window_fraction": 1.0, "event_lex": EVENT_LEX[:16], "base_lex": BASE_LEX[:12],
                      "max_docbase": 1.0, "require_magsupp": False},
        "decision": {"cand": "agg", "w": 0.0, "max_disp": 5.0, "max_cv": 10.0,
                     "ev": {"absmag": 0.0, "conf": 0.0, "wfrac": 0.0, "docbase": 0.0, "magsupp": 0.0, "bias": -4.0},
                     "low": 0.35, "high": 0.6},
    }


def clamp(x, lo, hi): return max(lo, min(hi, x))


def mutate(role, g, rng, methods):
    """Typed child: 1-3 stacked operators so multi-step activations are reachable."""
    c, ch = copy.deepcopy(g), []
    for _ in range(rng.choice([1, 1, 2, 3])):
        c, more = mutate1(role, c, rng, methods); ch += more
    return c, ch


def mutate1(role, g, rng, methods):
    c = copy.deepcopy(g); ch = []
    if role == "numerical":
        op = rng.choice(["add", "add", "drop", "swap", "agg"])
        pool = c["pool"]
        if op == "add" or not pool:
            cand = [m for m in methods if m not in pool and m != ANCHOR]
            if cand and len(pool) < 5: m = rng.choice(cand); pool.append(m); ch.append(f"add:{m}")
        elif op == "drop":
            m = pool.pop(rng.randrange(len(pool))); ch.append(f"drop:{m}")
        elif op == "swap":
            cand = [m for m in methods if m not in pool and m != ANCHOR]
            if cand:
                i = rng.randrange(len(pool)); old = pool[i]; pool[i] = rng.choice(cand); ch.append(f"swap:{old}->{pool[i]}")
        else:
            c["agg"] = rng.choice([a for a in AGGS if a != c["agg"]]); ch.append(f"agg:{c['agg']}")
    elif role == "retrieval":
        op = rng.choice(["min_conf", "max_mag", "max_wfrac", "mag_scale", "window_fraction", "event_lex",
                         "base_lex", "max_docbase", "require_magsupp"])
        if op == "min_conf":
            c[op] = clamp((c[op] if c[op] <= 1.0 else 0.95) + rng.gauss(0, 0.15), 0.0, 1.01)
        elif op == "max_mag": c[op] = clamp(c[op] + rng.gauss(0, 0.12), 0.01, 1.0)
        elif op == "max_wfrac": c[op] = clamp(c[op] + rng.gauss(0, 0.15), 0.1, 1.0)
        elif op == "mag_scale": c[op] = clamp(c[op] + rng.gauss(0, 0.2), 0.1, 1.5)
        elif op == "window_fraction": c[op] = clamp(c[op] + rng.gauss(0, 0.15), 0.1, 1.0)
        elif op == "max_docbase": c[op] = clamp(c[op] + rng.gauss(0, 0.2), 0.0, 1.0)
        elif op == "require_magsupp": c[op] = not c[op]
        else:
            full = EVENT_LEX if op == "event_lex" else BASE_LEX
            lex = c[op]
            if rng.random() < 0.5 and len(lex) > 2: lex.remove(rng.choice(lex))
            else:
                extra = [w for w in full if w not in lex]
                if extra: lex.append(rng.choice(extra))
        ch.append(op)
    else:
        op = rng.choice(["cand", "w", "w", "max_disp", "max_cv", "ev", "ev", "low_high"])
        if op == "cand": c["cand"] = rng.choice(["agg", "p0", "p1", "p2"])
        elif op == "w": c["w"] = clamp(c["w"] + rng.gauss(0, 0.12), 0.0, 0.5)
        elif op == "max_disp": c["max_disp"] = clamp(c["max_disp"] * math.exp(rng.gauss(0, 0.5)), 0.01, 5.0)
        elif op == "max_cv": c["max_cv"] = clamp(c["max_cv"] * math.exp(rng.gauss(0, 0.5)), 0.01, 10.0)
        elif op == "ev":
            k = rng.choice(list(c["ev"])); c["ev"][k] = clamp(c["ev"][k] + rng.gauss(0, 0.6), -4.0, 4.0); op = f"ev.{k}"
        else:
            c["low"] = clamp(c["low"] + rng.gauss(0, 0.1), 0.05, 0.95); c["high"] = clamp(c["high"] + rng.gauss(0, 0.1), 0.05, 0.95)
            if c["low"] > c["high"]: c["low"], c["high"] = c["high"], c["low"]
        ch.append(op)
    return c, ch


# ----------------------------------------------------------------------------- execution
def numerical_candidates(ng, d):
    pool = [m for m in ng["pool"] if m in d["fc"]]
    cands = {"anchor": d["fc"][ANCHOR]}
    for i, m in enumerate(pool): cands[f"p{i}"] = d["fc"][m]
    if pool:
        vs = [d["fc"][ANCHOR]] + [d["fc"][m] for m in pool]; H = d["H"]
        if ng["agg"] == "median": cands["agg"] = [statistics.median(v[i] for v in vs) for i in range(H)]
        elif ng["agg"] == "mean": cands["agg"] = [statistics.mean(v[i] for v in vs) for i in range(H)]
        else:
            cands["agg"] = []
            for i in range(H):
                s = sorted(v[i] for v in vs); s = s[1:-1] if len(s) > 2 else s; cands["agg"].append(statistics.mean(s))
    return cands


def retrieval_evidence(rg, d):
    if not d["corr"] or d["conf"] < rg["min_conf"]: return []
    txt = d["doclow"]
    nb = sum(txt.count(w) for w in rg["base_lex"]); ne = sum(txt.count(w) for w in rg["event_lex"])
    docbase = nb / (nb + ne + 1e-9)
    if docbase > rg["max_docbase"]: return []
    H = d["H"]; ev = []
    for s, e, m in d["corr"]:
        mag = abs(m - 1)
        wfrac = (e - s) / H
        if mag > rg["max_mag"] or wfrac > rg["max_wfrac"]: continue
        magsupp = 1.0 if mag <= d["f_range"] + 1e-9 else 0.0
        if rg["require_magsupp"] and not magsupp: continue
        e2 = s + max(1, round((e - s) * rg["window_fraction"]))
        ev.append(dict(win=(s, min(e2, H)), mult=1 + rg["mag_scale"] * (m - 1),
                       feat={"absmag": mag, "conf": d["conf"], "wfrac": wfrac, "docbase": docbase, "magsupp": magsupp, "bias": 1.0}))
    return ev


def decide(dg, cands, ev, d):
    anchor = cands["anchor"]; out = list(anchor)
    alt = cands.get(dg["cand"])
    if alt is not None and dg["w"] > 0 and d["f_disp"] <= dg["max_disp"] and d["f_cv"] <= dg["max_cv"]:
        w = dg["w"]; out = [(1 - w) * a + w * b for a, b in zip(anchor, alt)]
    base = list(out)
    for c in ev:
        s = sum(dg["ev"].get(k, 0.0) * v for k, v in c["feat"].items())
        p = 1 / (1 + math.exp(-clamp(s, -30, 30)))
        a = 0.0 if p <= dg["low"] else (1.0 if p > dg["high"] else 0.5)
        if a <= 0: continue
        m = 1 + a * (c["mult"] - 1); st, en = c["win"]
        for i in range(st, min(en, len(out))): out[i] = base[i] * m
    return list(apply_bounded_delta(anchor, out))


def run_team(team, d):
    return decide(team["decision"], numerical_candidates(team["numerical"], d), retrieval_evidence(team["retrieval"], d), d)


def gains(team, data):
    return [d["base_jt"] - jt(run_team(team, d), d["truth"]) for d in data]


REG_PEN = 0.5


def robust(gs):
    return statistics.mean(gs) + REG_PEN * statistics.mean(min(0.0, g) for g in gs) if gs else 0.0


FOLD_MODE = "group"
STD_W = 0.5


def gfolds(data, K):
    if FOLD_MODE == "strat":
        # every group is spread across folds, matching the train->test split (95% of test groups occur in train)
        gr, out = {}, [[] for _ in range(K)]
        for d in data: gr.setdefault(d["group"], []).append(d)
        i = 0
        for _g, it in sorted(gr.items()):
            for d in sorted(it, key=lambda x: x["tid"]): out[i % K].append(d); i += 1
        return out
    gr = {}
    for d in data: gr.setdefault(d["group"], []).append(d)
    out = [[] for _ in range(K)]; ld = [0] * K
    for _g, it in sorted(gr.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        f = ld.index(min(ld)); out[f] += it; ld[f] += len(it)
    return out


def fitness(team, folds):
    per = [robust(gains(team, f)) for f in folds]
    return statistics.mean(per) - STD_W * (statistics.pstdev(per) if len(per) > 1 else 0.0), per


# ----------------------------------------------------------------------------- evolution
def refit_decision(cur, rg, data, rng, gens=12, pop=16, elite=4):
    team = copy.deepcopy(cur); team["retrieval"] = rg
    keys = list(team["decision"]["ev"])
    def score(ev):
        t = copy.deepcopy(team); t["decision"]["ev"] = ev
        return robust(gains(t, data))
    popu = [dict(team["decision"]["ev"])] + [{k: rng.gauss(0, 1.0) for k in keys} for _ in range(pop)]
    scored = sorted(((score(e), e) for e in popu), key=lambda x: -x[0])
    for g in range(gens):
        par = [e for _s, e in scored[:elite]]; sd = 0.6 * (1 - g / gens) + 0.05
        kids = [{k: clamp(rng.choice(par)[k] + rng.gauss(0, sd), -4, 4) for k in keys} for _ in range(pop - elite)]
        scored = sorted(scored[:elite] + [(score(e), e) for e in kids], key=lambda x: -x[0])
    d = copy.deepcopy(team["decision"]); d["ev"] = scored[0][1]
    return d


def evolve(train, methods, rng, gens, children, teams, roles=("numerical", "retrieval", "decision"), K=3, log=None, refit=True):
    folds = gfolds(train, K)
    cur = gen0(); cur_fit, cur_per = fitness(cur, folds)
    lineage = [dict(gen=0, fit=cur_fit, folds=cur_per, accepted=True, changes=[])]
    for gi in range(1, gens + 1):
        kids = {r: [(cur[r], [])] + [mutate(r, cur[r], rng, methods) for _ in range(children)] if r in roles else [(cur[r], [])]
                for r in ("numerical", "retrieval", "decision")}
        if refit and "decision" in roles:
            # Decision's own numerical evolution: refit its evidence-acceptance weights on the
            # evidence produced by each Retrieval child (pass-combiner style inner ES, train only).
            for rg, _ch in kids["retrieval"][:3]:
                kids["decision"].append((refit_decision(cur, rg, train, rng), ["refit"]))
        best = None
        for _ in range(teams):
            pick = {r: rng.choice(kids[r]) for r in kids}
            if all(not pick[r][1] for r in pick): continue
            team = {r: pick[r][0] for r in pick}
            f, per = fitness(team, folds)
            if best is None or f > best[0]:
                best = (f, per, team, {r: pick[r][1] for r in pick if pick[r][1]})
        acc = best is not None and best[0] > cur_fit + 1e-6
        if acc: cur, cur_fit, cur_per = best[2], best[0], best[1]
        lineage.append(dict(gen=gi, fit=cur_fit, folds=cur_per, accepted=acc,
                            changes=best[3] if acc else {}, best_child_fit=best[0] if best else None))
        if log: log(f"  gen {gi:2d} {'ACCEPT' if acc else 'reject'} fit {cur_fit:+.5f} folds {[round(x, 5) for x in cur_per]} {best[3] if acc else ''}")
    return cur, lineage


def summarize(team, data):
    b = statistics.mean(d["base_jt"] for d in data)
    sb = so = rb = ro = 0.0; w = r = cov = 0
    for d in data:
        out = run_team(team, d); mb = drcik_point_metrics(d["truth"], d["fc"][ANCHOR], cap=CAP); mo = drcik_point_metrics(d["truth"], out, cap=CAP)
        sb += mb["smae"]; so += mo["smae"]; rb += mb["srmse"]; ro += mo["srmse"]
        if any(abs(x - y) > 1e-12 for x, y in zip(out, d["fc"][ANCHOR])): cov += 1
        dj = (mb["smae"] + mb["srmse"]) - (mo["smae"] + mo["srmse"])
        if dj > 1e-9: w += 1
        elif dj < -1e-9: r += 1
    return dict(n=len(data), smae_gain=(sb - so) / sb, srmse_gain=(rb - ro) / rb, joint_gain=((sb + rb) - (so + ro)) / (sb + rb),
                changed=cov, wins=w, regressions=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".scratch/self_evolving/nrd_cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=20); ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--teams", type=int, default=32); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--roles", default="numerical,retrieval,decision")
    ap.add_argument("--open", choices=["cv", "dev", "test"], default="cv",
                    help="cv: nested grouped CV on train only; dev: + dev gate; test: + test (once)")
    ap.add_argument("--std-w", type=float, default=0.5)
    ap.add_argument("--folds", choices=["group", "strat"], default="group")
    a = ap.parse_args(); roles = tuple(a.roles.split(","))
    global FOLD_MODE, STD_W; FOLD_MODE = a.folds; STD_W = a.std_w
    D = [prep(d) for d in json.load(open(a.cache))]
    train = [d for d in D if d["part"] == "train"]
    methods = sorted(set().union(*[d["fc"] for d in train]))
    res = dict(roles=roles, gens=a.gens, children=a.children, teams=a.teams, seeds=a.seeds, cv={}, final={})
    log = print
    # 1) nested grouped CV: evolve on K-1 folds, score the held-out fold (honest train estimate)
    outer = gfolds(train, 3)
    for seed in a.seeds:
        per = []
        for k in range(3):
            tr = [d for j in range(3) if j != k for d in outer[j]]
            team, lin = evolve(tr, methods, random.Random(seed * 100 + k), a.gens, a.children, a.teams, roles, K=2)
            s = summarize(team, outer[k]); per.append(s)
            log(f"seed {seed} outer fold {k}: held-out joint {s['joint_gain']:+.2%} sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} W/R {s['wins']}/{s['regressions']}")
        res["cv"][seed] = per
    # 2) full-train evolution -> frozen champion per seed
    if a.open in ("dev", "test"):
        dev = [d for d in D if d["part"] == "dev"]; test = [d for d in D if d["part"] == "public_test"]
        for seed in a.seeds:
            log(f"== full-train evolution seed {seed}")
            team, lin = evolve(train, methods, random.Random(seed), a.gens, a.children, a.teams, roles, K=3, log=log)
            r = dict(team=team, lineage=lin, train=summarize(team, train), dev=summarize(team, dev))
            gate = r["dev"]["smae_gain"] >= 0 and r["dev"]["srmse_gain"] >= 0
            r["dev_gate_pass"] = gate
            if a.open == "test":
                r["test"] = summarize(team if gate else gen0(), test)
            log(f"seed {seed}: train {r['train']['joint_gain']:+.2%} dev {r['dev']['joint_gain']:+.2%} gate {gate}" +
                (f" test {r['test']['joint_gain']:+.2%} (sMAE {r['test']['smae_gain']:+.2%}, sRMSE {r['test']['srmse_gain']:+.2%}, W/R {r['test']['wins']}/{r['test']['regressions']})" if 'test' in r else ""))
            res["final"][seed] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
