"""THREE-loop cooperative co-evolution where ALL THREE loops genuinely evolve:

  Loop N (numerical): evolves a COMBINATION (weights) over a candidate library
      {toto_2_0, seasonal_naive} + pure-code statistical methods (naive/mean/drift/
      linear_trend/ewm/moving_avg) -> the base forecast. This genuinely evolves the numerical
      'dictionary', not just an expose switch.
  Loop A (retrieval): document-selection policy, fitness = gt selection-F1 (dense).
  Loop B (decision) : the FULL evolvable controller (run_controller_evolution), correcting N's base.
  Coupling: A's evidence quality scales B's CorDP gate; N's base is what B corrects; team fitness
      = windowed ROI gain of (N-base corrected by B) vs the Toto reference. Alternate A->N->B.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/coevolve_loops.py
"""
from __future__ import annotations
import json, math, random, statistics
from pathlib import Path

import scripts.evolve_retrieval as R
from scripts.stat_library import stat_forecasts
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import horizon_window_mask, effects_from_cordp
from evolving_loop.adjustment.math_menu import build_math_menu
from evolving_loop.adjustment.controller import (
    SEED_CONTROLLERS, CORDP_CONTROLLER, run_controller, run_controller_evolution)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text()) if (ROOT / ".scratch/semantic_refs.json").exists() else {}
METHODS = ["toto_2_0", "seasonal_naive", "naive", "mean", "drift", "linear_trend", "ewm", "moving_avg"]

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def a_quality_map(part, aw):
    def q(t):
        sc = [R.score(R.doc_features(t, d), aw) for d in t.documents
              if R.score(R.doc_features(t, d), aw) > 0]
        return 1.0 / (1.0 + math.exp(-statistics.mean(sc))) if sc else 0.0
    return {t.numeric.task_id: q(t) for t in load_context_tasks_by_ids(TASKS_DIR, tuple(SPLIT[part]["task_ids"]))}


def load_B(part, aq):
    cards = json.loads((ROOT / f".scratch/cordp_cards_{part}.json").read_text())
    ids = tuple(t for t in SPLIT[part]["task_ids"] if t in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, ids)}
    data = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric; truth = list(n.future_values); H = len(truth)
        toto = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(toto) != H:
            continue
        try:
            snv = tuple(fs.forecast("seasonal_naive", tuple(n.history_values), n.prediction_length, n.frequency))
        except Exception:
            snv = toto
        cand = {"toto_2_0": toto, "seasonal_naive": snv, **stat_forecasts(n.history_values, H)}
        cand = {k: v for k, v in cand.items() if len(v) == H}
        c = cards.get(tid) or {}
        corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
        fts = [str(x) for x in t.future_timestamps]
        hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
        roi = [False] * H
        for s, e, _m in corrs:
            roi = [a or b for a, b in zip(roi, horizon_window_mask(fts, str(s), str(e)))]
        idx = [i for i, r in enumerate(roi) if r]
        if not idx:
            continue
        rj = drcik_point_metrics([truth[i] for i in idx], [toto[i] for i in idx], cap=CAP)
        data.append(dict(tid=tid, cand=cand, truth=truth, conf=float(c.get("confidence") or 0.0),
                         corrs=corrs, effs=effects_from_cordp(corrs), menu=build_math_menu(hv, hts, fts),
                         sref=REFS.get(tid, "none"), hv=hv, hts=hts, fts=fts, idx=idx,
                         toto_ref=(rj["smae"] + rj["srmse"]) / 2.0, aq=aq.get(tid, 1.0)))
    return data


def base_from_N(cand, nw):
    """Numerical dictionary combination: relu-normalized weighted blend of the candidate library."""
    w = {m: max(0.0, nw.get(m, 0.0)) for m in cand}
    s = sum(w.values()) or 1.0
    H = len(next(iter(cand.values())))
    return tuple(sum(w[m] * cand[m][i] for m in cand) / s for i in range(H))


def _roi_joint(fc, truth, idx):
    m = drcik_point_metrics([truth[i] for i in idx], [fc[i] for i in idx], cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def b_fitness(controller, data, nw):
    deltas = []
    for d in data:
        base = base_from_N(d["cand"], nw)
        out, _ = run_controller(controller, {"toto_2_0": base, "seasonal_naive": d["cand"]["seasonal_naive"]},
                                d["effs"], d["hv"], d["hts"], d["fts"], semantic_ref=d["sref"],
                                cordp_conf=d["conf"] * d["aq"], cordp_corrections=d["corrs"], menu=d["menu"])
        deltas.append(d["toto_ref"] - _roi_joint(out, d["truth"], d["idx"]))   # gain vs Toto reference
    if not deltas:
        return 0.0
    return statistics.mean(deltas) - 0.5 * statistics.mean(min(0.0, x) for x in deltas)


_L1 = 0.02   # shrink toward Toto: penalize non-Toto weight mass (anti-overfit regularizer)


def _reg(nw):
    s = sum(max(0.0, v) for v in nw.values()) or 1.0
    return _L1 * sum(max(0.0, nw.get(m, 0.0)) for m in METHODS if m != "toto_2_0") / s


def evolve_N(search, val, controller, rng, gens=15, pop=20):
    """CV/held-out: RANK candidates on the search fold, but SELECT the champion by the held-out
    (val) fold (+ L1 shrinkage to Toto). A global weight vector can only be de-overfit by
    selecting on held-out, not by in-sample fit."""
    seeds = [{"toto_2_0": 1.0}, {m: 1.0 / len(METHODS) for m in METHODS}]
    population = list(seeds) + [{m: max(0.0, rng.gauss(0.2, 0.4)) for m in METHODS} for _ in range(pop)]
    sfit = lambda nw: b_fitness(controller, search, nw) - _reg(nw)     # rank on search
    vfit = lambda nw: b_fitness(controller, val, nw) - _reg(nw)        # select on held-out val
    best = max(population, key=vfit); bv = vfit(best)
    for g in range(gens):
        parents = sorted(population, key=sfit, reverse=True)[:6]        # evolve on search
        s = 0.3 * (1 - g / gens) + 0.05
        population = parents + [{m: max(0.0, rng.choice(parents).get(m, 0.0) + rng.gauss(0, s)) for m in METHODS}
                                for _ in range(pop)]
        cur = max(population, key=vfit)                                 # champion = best on val
        if vfit(cur) > bv:
            best, bv = cur, vfit(cur)
    return best, bv


def evolve_B(data, nw, seed=20260919):
    best, bf, _ = run_controller_evolution(SEED_CONTROLLERS, lambda c: b_fitness(c, data, nw),
                                           generations=20, pop_size=32, elite=8, seed=seed)
    return best, bf


rng = random.Random(20260919)
A_tr, A_dv = R.load("train"), R.load("dev")
print("== THREE-loop CCEA: ALL THREE evolve (N=numerical combo, A=retrieval, B=full controller) ==\n")
aw = {"exact_entity": 2.0, "bias": -1.0}
ctrl = CORDP_CONTROLLER
nw = {"toto_2_0": 1.0}
for rnd in range(3):
    aw, _, _ = R.evolve(A_tr, rng, gens=30)                       # Loop A
    B_all = load_B("train", a_quality_map("train", aw))
    B_search = [d for i, d in enumerate(B_all) if i % 10 >= 3]    # 70% search
    B_val = [d for i, d in enumerate(B_all) if i % 10 < 3]        # 30% held-out (anti-overfit selection)
    nw, nv = evolve_N(B_search, B_val, ctrl, rng)                 # Loop N: rank-search, select-val + L1
    ctrl, bf = evolve_B(B_search, nw)                             # Loop B: evolve on search only
    topN = sorted(nw.items(), key=lambda kv: -kv[1])[:3]
    print(f"round {rnd+1}: A F1(train) {R.fitness(aw,A_tr):.3f} | N top-weights {[(m,round(w,2)) for m,w in topN]} "
          f"| val-fit {nv:+.4f} | B steps {len(ctrl.steps)}")


def report(part):
    data = load_B(part, a_quality_map(part, aw))
    nb, wins, reg, tt = [], 0, 0, 0
    for d in data:
        base = base_from_N(d["cand"], nw)
        out, _ = run_controller(ctrl, {"toto_2_0": base, "seasonal_naive": d["cand"]["seasonal_naive"]},
                                d["effs"], d["hv"], d["hts"], d["fts"], semantic_ref=d["sref"],
                                cordp_conf=d["conf"] * d["aq"], cordp_corrections=d["corrs"], menu=d["menu"])
        mb = drcik_point_metrics(d["truth"], list(d["cand"]["toto_2_0"]), cap=CAP)   # vs Toto
        mo = drcik_point_metrics(d["truth"], list(out), cap=CAP)
        nb.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"]))
        if (mo["smae"]+mo["srmse"]) < (mb["smae"]+mb["srmse"]) - 1e-9: wins += 1
        elif (mo["smae"]+mo["srmse"]) > (mb["smae"]+mb["srmse"]) + 1e-9: reg += 1
        tt += 1
    bs = statistics.mean(r[0] for r in nb); br = statistics.mean(r[1] for r in nb)
    os_ = statistics.mean(r[2] for r in nb); or_ = statistics.mean(r[3] for r in nb)
    print(f"  {part:11s} ({len(nb)} ROI tasks): sMAE {bs:.4f}->{os_:.4f} ({(bs-os_)/bs:+.2%}) "
          f"| sRMSE {br:.4f}->{or_:.4f} ({(br-or_)/br:+.2%}) | wins {wins} reg {reg}")


print(f"\n== FINAL: 3-loop (all-evolve) vs Toto reference, ROI tasks, whole-horizon cap={CAP} ==")
print(f"  N combo top-weights: {sorted(nw.items(), key=lambda kv:-kv[1])[:4]}")
print(f"  A retrieval F1(test): {R.fitness(aw, R.load('public_test')):.3f}")
print(f"  B controller: {' -> '.join(type(s).__name__ for s in ctrl.steps)}")
for part in ("train", "dev", "public_test"):
    report(part)
fs.close()
