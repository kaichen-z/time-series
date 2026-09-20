"""Proper protocol: TRAIN search (candidate gates) -> DEV select -> TEST confirm, all on the
ROI (event-window) metric. The CorDP gate is PER-TASK, so train-internal LOO reduces to
in-sample (leaving one task out doesn't change another task's gate); the real held-out check
is DEV, then a single TEST evaluation of the DEV-selected gate.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/select_gate.py
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import horizon_window_mask
from evolving_loop.adjustment.controller import Controller, SelectBase, CorDPAdjust, run_controller

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def roi_joint_parts(fc, truth, roi):
    idx = [i for i, r in enumerate(roi) if r]
    if not idx:
        return None
    m = drcik_point_metrics([truth[i] for i in idx], [fc[i] for i in idx], cap=CAP)
    return m["smae"], m["srmse"]


def load(part):
    p = ROOT / f".scratch/cordp_cards_{part}.json"
    if not p.exists():
        return None
    cards = json.loads(p.read_text())
    ids = tuple(tid for tid in SPLIT[part]["task_ids"] if tid in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, ids)}
    data = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric
        truth = list(n.future_values)
        base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(base) != len(truth):
            continue
        c = cards.get(tid) or {}
        corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
        fts = [str(x) for x in t.future_timestamps]
        roi = [False] * len(truth)
        for (s, e, _m) in corrs:
            roi = [a or b for a, b in zip(roi, horizon_window_mask(fts, str(s), str(e)))]
        bp = roi_joint_parts(list(base), truth, roi)
        if bp is None:
            continue                                    # no ROI -> no signal, excluded
        data.append(dict(tid=tid, base=base, truth=truth, conf=float(c.get("confidence") or 0.0),
                         corrs=corrs, hv=list(n.history_values),
                         hts=[str(x) for x in t.history_timestamps], fts=fts, roi=roi,
                         b_smae=bp[0], b_srmse=bp[1]))
    return data


def evalgate(data, c, w):
    ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=c, wfrac_max=w)))
    bm, gm = [], []
    wins = reg = touched = 0
    for d in data:
        out, _ = run_controller(ctrl, {"toto_2_0": d["base"]}, (), d["hv"], d["hts"], d["fts"],
                                cordp_conf=d["conf"], cordp_corrections=d["corrs"])
        gp = roi_joint_parts(list(out), d["truth"], d["roi"])
        bm.append((d["b_smae"], d["b_srmse"])); gm.append(gp)
        if any(abs(a - b) > 1e-9 for a, b in zip(d["base"], out)):
            touched += 1
            if (gp[0] + gp[1]) < (d["b_smae"] + d["b_srmse"]) - 1e-9:
                wins += 1
            elif (gp[0] + gp[1]) > (d["b_smae"] + d["b_srmse"]) + 1e-9:
                reg += 1
    bs, br = statistics.mean(x[0] for x in bm), statistics.mean(x[1] for x in bm)
    gs, gr = statistics.mean(x[0] for x in gm), statistics.mean(x[1] for x in gm)
    return dict(n=len(data), touched=touched, wins=wins, reg=reg,
                smae_impr=(bs - gs) / bs, srmse_impr=(br - gr) / br,
                bs=bs, gs=gs, br=br, gr=gr)


TR, DV, TE = load("train"), load("dev"), load("public_test")
print(f"ROI tasks: train {len(TR) if TR else 0} | dev {len(DV) if DV else 0} | test {len(TE) if TE else 0}\n")

GRID = [(c, w) for c in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] for w in [0.15, 0.2, 0.3, 0.5]]

# ---- TRAIN: score every candidate gate (the "search") ----
train_scores = {(c, w): evalgate(TR, c, w) for (c, w) in GRID}

# ---- DEV: select. Prefer zero dev-regression, then max dev (sMAE+sRMSE) improvement ----
if DV:
    cand = []
    for (c, w) in GRID:
        ds = evalgate(DV, c, w)
        cand.append(((c, w), ds))
    # selection rule
    zero = [x for x in cand if x[1]["reg"] == 0 and x[1]["touched"] > 0]
    pool = zero if zero else [x for x in cand if x[1]["touched"] > 0]
    best = max(pool, key=lambda x: (x[1]["smae_impr"] + x[1]["srmse_impr"]))
    (bc, bw), _ = best
    rule = "zero dev-regression" if zero else "min-regression"
    print(f"== DEV-selected gate ({rule}): conf>={bc} wfrac<={bw} ==\n")
    for name, data in [("TRAIN(80)", TR), ("DEV(20)", DV), ("TEST(99)", TE)]:
        if not data:
            print(f"  {name}: no cards"); continue
        r = evalgate(data, bc, bw)
        print(f"  {name}: ROI sMAE {r['bs']:.4f}->{r['gs']:.4f} ({r['smae_impr']:+.1%}) | "
              f"sRMSE {r['br']:.4f}->{r['gr']:.4f} ({r['srmse_impr']:+.1%}) | "
              f"touched {r['touched']} wins {r['wins']} reg {r['reg']} (of {r['n']} ROI tasks)")
else:
    print("DEV cards not ready yet -- rerun once .scratch/cordp_cards_dev.json exists.")
    print("TRAIN gate frontier (conf, wfrac -> touched/wins/reg, sMAE%, sRMSE%):")
    for (c, w), r in sorted(train_scores.items()):
        if r["touched"]:
            print(f"  conf>={c} wfrac<={w}: t{r['touched']} w{r['wins']} r{r['reg']} "
                  f"| {r['smae_impr']:+.1%}/{r['srmse_impr']:+.1%}")
fs.close()
