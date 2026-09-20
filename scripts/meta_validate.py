"""Make the VALIDATION POLICY itself a parameter and META-LEARN it.

The selector has knobs we normally hand-set: how to split CV folds (random vs grouped /
leave-domain-out) and how strict the significance gate is (SIG_K). Here those become a searched
parameter. Each validation policy decides ONE thing -- deploy the evolved (complex) controller, or
fall back to the safe do-no-harm CorDP -- and we pick the policy by a FIXED meta-held-out set (dev)
that the within-train validation never touches. TEST stays the final honest number.

This is the "you can evolve the validator too, as long as one outer judge stays fixed" idea:
  within-train CV/gate  --(decides deploy vs fallback)-->  scored on DEV (meta-anchor)  -->  TEST
The point is to see WHICH validation policy the meta-anchor selects -- expected: grouped CV + a
real significance gate, i.e. the strict policy that converges to do-no-harm.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/meta_validate.py
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import effects_from_cordp, horizon_window_mask
from evolving_loop.adjustment.math_menu import build_math_menu
from evolving_loop.adjustment.controller import (
    SEED_CONTROLLERS, CORDP_CONTROLLER, run_controller, run_controller_evolution)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text()) if (ROOT / ".scratch/semantic_refs.json").exists() else {}
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
_GRP: dict = {}


def group(tid):
    if tid not in _GRP:
        try:
            d = json.loads((ROOT / TASKS / f"{tid}.json").read_text())
            et = ((d.get("showcase") or {}).get("entity") or {}).get("type") or "unknown"
            _GRP[tid] = f"{et}|{(d.get('task_metadata') or {}).get('frequency') or ''}"
        except Exception:
            _GRP[tid] = "unknown"
    return _GRP[tid]


def load(part):
    cards = json.loads((ROOT / f".scratch/cordp_cards_{part}.json").read_text())
    ids = tuple(t for t in SPLIT[part]["task_ids"] if t in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, ids)}
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
        c = cards.get(tid) or {}
        corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
        fts = [str(x) for x in t.future_timestamps]; hv = list(n.history_values); hts = [str(x) for x in t.history_timestamps]
        data.append(dict(tid=tid, toto=toto, snv=snv, truth=truth, corrs=corrs, effs=effects_from_cordp(corrs),
                         menu=build_math_menu(hv, hts, fts), sref=REFS.get(tid, "none"), hv=hv, hts=hts, fts=fts,
                         conf=float(c.get("confidence") or 0.0), group=group(tid)))
    return data


def _apply(ctrl, d):
    out, _ = run_controller(ctrl, {"toto_2_0": d["toto"], "seasonal_naive": d["snv"]},
                            d["effs"], d["hv"], d["hts"], d["fts"], semantic_ref=d["sref"],
                            cordp_conf=d["conf"], cordp_corrections=d["corrs"], menu=d["menu"])
    return out


def _joint(fc, truth):
    m = drcik_point_metrics(list(truth), list(fc), cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def gain(ctrl, data):
    ds = [_joint(d["toto"], d["truth"]) - _joint(_apply(ctrl, d), d["truth"]) for d in data]
    return statistics.mean(ds) - 0.5 * statistics.mean(min(0.0, x) for x in ds) if ds else 0.0


def evolve_B(data, seed=20260919, gens=15, pop=24):
    best, _, _ = run_controller_evolution(SEED_CONTROLLERS, lambda c: gain(c, data),
                                          generations=gens, pop_size=pop, elite=6, seed=seed)
    return best


def folds_random(data, K):
    return [[d for i, d in enumerate(data) if i % K == f] for f in range(K)]


def folds_grouped(data, K):
    groups = {}
    for d in data:
        groups.setdefault(d["group"], []).append(d)
    out = [[] for _ in range(K)]; load_ = [0] * K
    for _g, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        f = load_.index(min(load_)); out[f].extend(items); load_[f] += len(items)
    return out


def cv_estimate(data, scheme, K=3):
    """Honest per-scheme CV: evolve on K-1 folds, score champion on held-out fold; mean +/- std."""
    folds = (folds_grouped if scheme == "grouped" else folds_random)(data, K)
    sc = []
    for f in range(K):
        val = folds[f]; tr = [d for g in range(K) if g != f for d in folds[g]]
        if val and tr:
            sc.append(gain(evolve_B(tr, seed=100 + f, gens=10, pop=16), val))
    return (statistics.mean(sc) if sc else 0.0, statistics.pstdev(sc) if len(sc) > 1 else 0.0)


def report(ctrl, data):
    rows, wins, reg = [], 0, 0
    for d in data:
        out = _apply(ctrl, d)
        mb = drcik_point_metrics(d["truth"], list(d["toto"]), cap=CAP)
        mo = drcik_point_metrics(d["truth"], list(out), cap=CAP)
        rows.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"]))
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
    bs = statistics.mean(r[0] for r in rows); os_ = statistics.mean(r[2] for r in rows)
    br = statistics.mean(r[1] for r in rows); or_ = statistics.mean(r[3] for r in rows)
    return (bs - os_) / bs, (br - or_) / br, wins, reg


TR, DV, TE = load("train"), load("dev"), load("public_test")
print(f"== META-VALIDATION: validation policy = (fold scheme x SIG_K), meta-selected on DEV ==")
print(f"tasks: train {len(TR)} | dev {len(DV)} | test {len(TE)}\n")

complex_ctrl = evolve_B(TR, gens=18, pop=28)          # the candidate complex model (deploy on all train)
simple_ctrl = CORDP_CONTROLLER                        # do-no-harm fallback
print(f"complex B: {' -> '.join(type(s).__name__ for s in complex_ctrl.steps)}")
dev_c = report(complex_ctrl, DV); dev_s = report(simple_ctrl, DV)
print(f"complex on DEV: sMAE {dev_c[0]:+.2%} sRMSE {dev_c[1]:+.2%} reg {dev_c[3]}")
print(f"simple  on DEV: sMAE {dev_s[0]:+.2%} sRMSE {dev_s[1]:+.2%} reg {dev_s[3]}\n")

est = {sch: cv_estimate(TR, sch) for sch in ("random", "grouped")}
print("within-train CV estimate of the complex model:")
for sch, (mu, sd) in est.items():
    print(f"  {sch:8s}: gain {mu:+.4f} ± {sd:.4f}")
print()

print(f"{'policy (scheme, SIG_K)':26s} {'decision':8s} {'DEV(meta)':>11s} {'-> TEST sMAE/sRMSE (reg)':>28s}")
results = []
for sch in ("random", "grouped"):
    mu, sd = est[sch]
    for sig in (0.0, 0.5, 1.0):
        deploy = complex_ctrl if (mu - sig * sd) > 0 else simple_ctrl
        which = "complex" if deploy is complex_ctrl else "SIMPLE"
        dv = report(deploy, DV); te = report(deploy, TE)
        dev_joint = (dv[0] + dv[1]) / 2.0
        results.append((f"({sch}, {sig})", which, dev_joint, te))
        print(f"{f'({sch}, {sig})':26s} {which:8s} {dev_joint:+10.2%}  {te[0]:+.2%}/{te[1]:+.2%} (reg {te[3]})")

best = max(results, key=lambda r: r[2])
print(f"\n== META-ANCHOR (dev) SELECTS: policy {best[0]} -> deploy {best[1]} ==")
print(f"   its honest TEST99: sMAE {best[3][0]:+.2%} | sRMSE {best[3][1]:+.2%} | reg {best[3][3]}")
fs.close()
