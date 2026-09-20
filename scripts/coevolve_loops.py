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
from evolving_loop.adjustment.guarded_n import (
    guarded_base, clamp_params, GUARD_KEYS, GUARD_BOUNDS, GUARD_SEED, GUARD_TOTO)

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


def load_B(part, aq, keep_all=False):
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
        if not idx and not keep_all:
            continue
        rj = drcik_point_metrics([truth[i] for i in idx] or truth, [toto[i] for i in idx] or list(toto), cap=CAP)
        data.append(dict(tid=tid, cand=cand, truth=truth, conf=float(c.get("confidence") or 0.0),
                         corrs=corrs, effs=effects_from_cordp(corrs), menu=build_math_menu(hv, hts, fts),
                         sref=REFS.get(tid, "none"), hv=hv, hts=hts, fts=fts, idx=idx,
                         toto_ref=(rj["smae"] + rj["srmse"]) / 2.0, aq=aq.get(tid, 1.0)))
    return data


def _roi_joint(fc, truth, idx):
    m = drcik_point_metrics([truth[i] for i in idx], [fc[i] for i in idx], cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def _whole_joint(fc, truth):
    m = drcik_point_metrics(list(truth), list(fc), cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def b_fitness(controller, data, npar):
    """Loop B objective: WHOLE-horizon do-no-harm gain of (N-base corrected by B) vs pure Toto,
    over the WHOLE task set (incl. no-signal tasks). Scoring on the whole set -- not just the ROI
    windows -- is what forces B to stop firing primitives on tasks with no document signal (the
    source of the whole-99 regressions). Downside-penalized -> B learns to touch only where safe."""
    deltas = []
    for d in data:
        toto = d["cand"]["toto_2_0"]
        base = guarded_base(d, npar)                                   # per-task guarded N base
        out, _ = run_controller(controller, {"toto_2_0": base, "seasonal_naive": d["cand"]["seasonal_naive"]},
                                d["effs"], d["hv"], d["hts"], d["fts"], semantic_ref=d["sref"],
                                cordp_conf=d["conf"] * d["aq"], cordp_corrections=d["corrs"], menu=d["menu"])
        deltas.append(_whole_joint(toto, d["truth"]) - _whole_joint(out, d["truth"]))   # gain vs Toto, whole horizon
    if not deltas:
        return 0.0
    return statistics.mean(deltas) - 0.5 * statistics.mean(min(0.0, x) for x in deltas)


def n_fitness(npar, data):
    """Loop N objective: whole-horizon do-no-harm gain of the guarded base vs pure Toto
    (positive = base better). Downside-penalized, so N learns to DEFAULT to Toto and only
    deviate where a fold-validated statistical method genuinely helps -- not a global blend."""
    deltas = []
    for d in data:
        toto = d["cand"]["toto_2_0"]
        base = guarded_base(d, npar)
        deltas.append(_whole_joint(toto, d["truth"]) - _whole_joint(base, d["truth"]))
    if not deltas:
        return 0.0
    return statistics.mean(deltas) - 0.5 * statistics.mean(min(0.0, x) for x in deltas)


def _mutate_params(p, sigma, rng):
    return clamp_params({k: p.get(k, GUARD_SEED[k]) + rng.gauss(0, sigma * (GUARD_BOUNDS[k][1] - GUARD_BOUNDS[k][0]))
                         for k in GUARD_KEYS})


K_CV = 3     # grouped (leave-domain-out) CV within train for N/B selection + error bars
SIG_K = 1.0  # significance gate: keep the evolved genome only if cross-domain (mean - K*std) > 0,
             # else fall back to the safe default (Toto for N, plain CorDP for B). Stops the
             # not-reliably-positive complexity from being deployed -> converges to do-no-harm.


_GROUP_CACHE: dict = {}


def _task_group(tid):
    """Domain proxy = (entity type, frequency) from the task file, e.g. 'freeway|1 hour'.
    Grouping CV folds by this stops same-domain leakage between train and val -- random i%K folds
    let hourly-freeway tasks sit in BOTH, so CV over-estimated generalization; leave-domain-out does not."""
    if tid in _GROUP_CACHE:
        return _GROUP_CACHE[tid]
    try:
        d = json.loads((ROOT / TASKS_DIR / f"{tid}.json").read_text())
        et = ((d.get("showcase") or {}).get("entity") or {}).get("type") or "unknown"
        freq = (d.get("task_metadata") or {}).get("frequency") or ""
        g = f"{et}|{freq}"
    except Exception:
        g = "unknown"
    _GROUP_CACHE[tid] = g
    return g


def _kfolds(data, K):
    """GROUPED K-fold: whole domains are kept intact within a fold (no domain spans train+val), so
    the held-out fold is a genuinely unseen domain -- an honest generalization estimate. Domains are
    greedily packed into the least-loaded fold for balance."""
    groups: dict = {}
    for d in data:
        groups.setdefault(_task_group(d["tid"]), []).append(d)
    folds = [[] for _ in range(K)]
    load = [0] * K
    for _g, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        f = load.index(min(load))
        folds[f].extend(items); load[f] += len(items)
    return folds


def _evolve_N_core(data, rng, gens=12, pop=16):
    """ES over guard thresholds on `data` (in-sample select; the CV wrapper supplies the honest
    held-out estimate). GUARD_TOTO is always present and is the fallback -> N can't be forced off Toto."""
    population = [clamp_params(GUARD_TOTO), clamp_params(GUARD_SEED)] + [
        _mutate_params(GUARD_SEED, 0.4, rng) for _ in range(pop)]
    fit = lambda p: n_fitness(p, data)
    best = max(population, key=fit)
    for g in range(gens):
        parents = sorted(population, key=fit, reverse=True)[:6]
        sigma = 0.3 * (1 - g / gens) + 0.05
        population = parents + [_mutate_params(rng.choice(parents), sigma, rng) for _ in range(pop)]
        cur = max(population, key=fit)
        if fit(cur) > fit(best):
            best = cur
    if n_fitness(clamp_params(GUARD_TOTO), data) >= fit(best):
        best = clamp_params(GUARD_TOTO)
    return best


def _evolve_B_core(data, nw, seed, gens=12, pop=20):
    best, _, _ = run_controller_evolution(SEED_CONTROLLERS, lambda c: b_fitness(c, data, nw),
                                          generations=gens, pop_size=pop, elite=6, seed=seed)
    return best


def cv_evolve_N(data, rng, K=K_CV):
    """K-fold CV: evolve N on K-1 folds, score the champion on the held-out fold; average = honest
    generalization estimate (+ std = error bar). Deploy the genome evolved on ALL train."""
    folds = _kfolds(data, K)
    scores = []
    for f in range(K):
        val = folds[f]; tr = [d for g in range(K) if g != f for d in folds[g]]
        if not val or not tr:
            continue
        scores.append(n_fitness(_evolve_N_core(tr, rng), val))       # held-out fold score
    deployed = _evolve_N_core(data, rng, gens=15, pop=20)            # deploy on all train
    mu = statistics.mean(scores) if scores else 0.0
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    if mu - SIG_K * sd <= 0.0:              # significance gate: not reliably positive across domains
        deployed = clamp_params(GUARD_TOTO)   # -> fall back to pure Toto (the safe default)
    return deployed, mu, sd


def cv_evolve_B(data, nw, rng, K=K_CV, seed=20260919):
    """K-fold CV champion selection for the controller (same protocol as N): evolve on K-1 folds,
    score on the held-out fold, average -> honest estimate + error bar; deploy on all train."""
    folds = _kfolds(data, K)
    scores = []
    for f in range(K):
        val = folds[f]; tr = [d for g in range(K) if g != f for d in folds[g]]
        if not val or not tr:
            continue
        scores.append(b_fitness(_evolve_B_core(tr, nw, seed + f), val, nw))
    deployed = _evolve_B_core(data, nw, seed, gens=18, pop=28)
    mu = statistics.mean(scores) if scores else 0.0
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    if mu - SIG_K * sd <= 0.0:              # significance gate: not reliably positive across domains
        deployed = CORDP_CONTROLLER           # -> fall back to plain do-no-harm CorDP (the safe default)
    return deployed, mu, sd


rng = random.Random(20260919)
A_tr, A_dv = R.load("train"), R.load("dev")
print(f"== THREE-loop CCEA: all evolve, {K_CV}-fold-CV champion selection for N & B (error bars) ==\n")
aw = {"exact_entity": 2.0, "bias": -1.0}
ctrl = CORDP_CONTROLLER
nw = clamp_params(GUARD_TOTO)                                     # N starts at pure Toto
for rnd in range(3):
    aw, _, _ = R.evolve(A_tr, rng, gens=30)                       # Loop A
    B_all = load_B("train", a_quality_map("train", aw), keep_all=True)  # WHOLE train (incl. no-signal)
    nw, n_cv, n_sd = cv_evolve_N(B_all, rng)                      # Loop N: k-fold-CV, whole-set do-no-harm
    ctrl, b_cv, b_sd = cv_evolve_B(B_all, nw, rng)                # Loop B: k-fold-CV, whole-set do-no-harm
    gp = {k: round(nw[k], 3) for k in GUARD_KEYS}
    print(f"round {rnd+1}: A F1(train) {R.fitness(aw,A_tr):.3f} | N guard {gp} "
          f"| N cv-fit {n_cv:+.4f}±{n_sd:.4f} | B steps {len(ctrl.steps)} | B cv-fit {b_cv:+.4f}±{b_sd:.4f}")


def report(part):
    data = load_B(part, a_quality_map(part, aw))
    nb, wins, reg, tt = [], 0, 0, 0
    for d in data:
        base = guarded_base(d, nw)
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


def report_whole(part):
    """WHOLE-set (not just ROI) do-no-harm audit vs pure Toto: N guard + B correction on
    every task with a card, so N's default-Toto discipline is visible (few regressions)."""
    data = load_B(part, a_quality_map(part, aw), keep_all=True)
    nb, wins, reg = [], 0, 0
    for d in data:
        base = guarded_base(d, nw)
        out, _ = run_controller(ctrl, {"toto_2_0": base, "seasonal_naive": d["cand"]["seasonal_naive"]},
                                d["effs"], d["hv"], d["hts"], d["fts"], semantic_ref=d["sref"],
                                cordp_conf=d["conf"] * d["aq"], cordp_corrections=d["corrs"], menu=d["menu"])
        mb = drcik_point_metrics(d["truth"], list(d["cand"]["toto_2_0"]), cap=CAP)
        mo = drcik_point_metrics(d["truth"], list(out), cap=CAP)
        nb.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"]))
        if (mo["smae"]+mo["srmse"]) < (mb["smae"]+mb["srmse"]) - 1e-9: wins += 1
        elif (mo["smae"]+mo["srmse"]) > (mb["smae"]+mb["srmse"]) + 1e-9: reg += 1
    bs = statistics.mean(r[0] for r in nb); br = statistics.mean(r[1] for r in nb)
    os_ = statistics.mean(r[2] for r in nb); or_ = statistics.mean(r[3] for r in nb)
    print(f"  {part:11s} (WHOLE {len(nb)} tasks): sMAE {bs:.4f}->{os_:.4f} ({(bs-os_)/bs:+.2%}) "
          f"| sRMSE {br:.4f}->{or_:.4f} ({(br-or_)/br:+.2%}) | wins {wins} reg {reg}")


print(f"\n== FINAL: 3-loop (all-evolve) vs pure Toto, whole-horizon cap={CAP} ==")
print(f"  N guarded selector: {{{', '.join(f'{k}={nw[k]:.3f}' for k in GUARD_KEYS)}}}  (== Toto if it never fires)")
print(f"  A retrieval F1(test): {R.fitness(aw, R.load('public_test')):.3f}")
print(f"  B controller: {' -> '.join(type(s).__name__ for s in ctrl.steps)}")
print("-- ROI subset (tasks with a document window) --")
for part in ("train", "dev", "public_test"):
    report(part)
print("-- WHOLE set (do-no-harm audit; N default-Toto should keep regressions low) --")
for part in ("train", "dev", "public_test"):
    report_whole(part)
fs.close()
