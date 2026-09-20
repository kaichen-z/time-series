"""Process-Reward-style MEASURED-IMPACT gate (From-Long-News PRM idea): instead of gating a
CorDP correction by the LLM's self-reported confidence, SELECT the gate on TRAIN by each
correction's measured forecast impact (toto_joint - corrected_joint), then FREEZE it and apply
to test. With only ~15 grounded corrections we don't fit a heavy model (it would overfit);
we search a small, deployable rule -- CorDPAdjust(conf_min, wfrac_max, |mult-1|<=mcap) -- whose
inputs are all inference-time features, maximizing a downside-averse measured-impact objective.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/impact_gate.py
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.controller import Controller, SelectBase, CorDPAdjust, run_controller

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
BETA = 3.0                                  # downside weight on measured regressions

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]


def joint(fc, truth):
    m = drcik_point_metrics(truth, list(fc), cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def load(part):
    cards = json.loads((ROOT / f".scratch/cordp_cards_{part}.json").read_text())
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
        data.append(dict(tid=tid, base=base, truth=truth, conf=float(c.get("confidence") or 0.0),
                         corrs=corrs, fts=[str(x) for x in t.future_timestamps],
                         hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                         toto=joint(base, truth)))
    return data


def apply_rule(d, c, w, mcap):
    ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=c, wfrac_max=w,
                                                       mult_lo=max(0.05, 1 - mcap), mult_hi=1 + mcap)))
    out, _ = run_controller(ctrl, {"toto_2_0": d["base"]}, (), d["hv"], d["hts"], d["fts"],
                            cordp_conf=d["conf"], cordp_corrections=d["corrs"])
    touched = any(abs(a - b) > 1e-9 for a, b in zip(d["base"], out))
    return joint(out, d["truth"]), touched


def score(data, c, w, mcap):
    imps = []
    for d in data:
        j, touched = apply_rule(d, c, w, mcap)
        if touched:
            imps.append(d["toto"] - j)                 # +ve = better
    if not imps:
        return None
    obj = sum(imps) - BETA * sum(-x for x in imps if x < 0)   # downside-averse measured impact
    return dict(obj=obj, n=len(imps), wins=sum(1 for x in imps if x > 1e-9),
                reg=sum(1 for x in imps if x < -1e-9), total=sum(imps))


train = load("train")
GRID_C = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
GRID_W = [0.15, 0.2, 0.3, 0.5, 1.0]
GRID_M = [0.3, 0.4, 0.5, 0.7, 1.0, 2.0]
best = None
for c in GRID_C:
    for w in GRID_W:
        for m in GRID_M:
            s = score(train, c, w, m)
            if s is None:
                continue
            key = (round(s["obj"], 6), -s["reg"], s["wins"])
            if best is None or key > best[0]:
                best = (key, (c, w, m), s)
(_, (bc, bw, bm), bs) = best
print(f"== measured-impact gate search on TRAIN (objective = total impact - {BETA}x downside) ==")
print(f"  BEST rule: conf>={bc}  wfrac<={bw}  |mult-1|<={bm}")
print(f"    train: applied {bs['n']} | wins {bs['wins']} reg {bs['reg']} | total joint impact {bs['total']:+.4f}")

# compare: the confidence-only gate we had been using
for label, (c, w, m) in [("conf>=0.80 wfrac<=0.20 (old)", (0.8, 0.2, 2.0)),
                          ("conf>=0.90 wfrac<=0.20 (old)", (0.9, 0.2, 2.0))]:
    s = score(train, c, w, m)
    if s:
        print(f"  ref {label}: applied {s['n']} wins {s['wins']} reg {s['reg']} total {s['total']:+.4f}")


def eval_part(part, c, w, m):
    data = load(part)
    rows = []
    for d in data:
        j, touched = apply_rule(d, c, w, m)
        mb = drcik_point_metrics(d["truth"], list(d["base"]), cap=CAP)
        # recompute corrected metrics
        ctrl = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=c, wfrac_max=w,
                                                           mult_lo=max(0.05, 1 - m), mult_hi=1 + m)))
        out, _ = run_controller(ctrl, {"toto_2_0": d["base"]}, (), d["hv"], d["hts"], d["fts"],
                                cordp_conf=d["conf"], cordp_corrections=d["corrs"])
        mo = drcik_point_metrics(d["truth"], list(out), cap=CAP)
        rows.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"], touched))
    bs, br = statistics.mean(r[0] for r in rows), statistics.mean(r[1] for r in rows)
    os_, or_ = statistics.mean(r[2] for r in rows), statistics.mean(r[3] for r in rows)
    tw = sum(1 for r in rows if r[4] and (r[2] + r[3]) < (r[0] + r[1]) - 1e-9)
    tr = sum(1 for r in rows if r[4] and (r[2] + r[3]) > (r[0] + r[1]) + 1e-9)
    tt = sum(1 for r in rows if r[4])
    print(f"  {part} ({len(rows)} tasks, {tt} touched): "
          f"sMAE {bs:.4f}->{os_:.4f} ({(bs-os_)/bs:+.2%}) | sRMSE {br:.4f}->{or_:.4f} ({(br-or_)/br:+.2%}) | wins {tw} reg {tr}")


print("\n== FROZEN measured-impact gate applied ==")
eval_part("train", bc, bw, bm)
if (ROOT / ".scratch/cordp_cards_public_test.json").exists():
    eval_part("public_test", bc, bw, bm)
fs.close()
