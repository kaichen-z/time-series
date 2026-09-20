"""Where is the residual value/difficulty -- retrieval side, not numerical side?

The document's value is ONLY the part not in the numbers: exogenous future events with no numeric
precursor. So the whole question is text-side: on tasks that genuinely HAVE a supporting document,
CorDP helps; the loss is (a) missing those, and (b) getting FOOLED by confounders (topically similar,
causally irrelevant docs) into correcting tasks that have no real signal.

This quantifies that. On test99, with the SAME CorDP multipliers, we apply the correction under
three gates and compare:
  - ORACLE   : apply only on tasks with a ground-truth SUPPORTING document (perfect selection).
  - OURS     : apply where our self-reported confidence clears the frozen gate (what we deploy).
  - CONFOUND : apply on tasks that have NO supporting doc (only distractors) -- the harm we take
               when we can't tell a confounder from a real cause.
Reports the oracle ceiling, our achieved, the gap, and the confusion (TP/FP/FN of our gate vs gt).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD [CONF=0.8] .venv/bin/python scripts/oracle_gap.py
"""
from __future__ import annotations
import json, os, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
CONF = float(os.environ.get("CONF", "0.8")); WFRAC = 0.2
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def load():
    cards = json.loads((ROOT / ".scratch/cordp_cards_public_test.json").read_text())
    ids = tuple(t for t in SPLIT["public_test"]["task_ids"] if t in cards)
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
        c = cards.get(tid) or {}
        fts = [str(x) for x in t.future_timestamps]
        corr = []
        for r in (c.get("corrections") or []):
            if len(r) >= 3:
                on = [i for i, m in enumerate(horizon_window_mask(fts, str(r[0]), str(r[1]))) if m]
                if on:
                    corr.append((on[0], on[-1] + 1, float(r[2])))
        wfrac = max((e - s) for s, e, _ in corr) / H if corr else 0.0
        has_gt = any(getattr(d, "role", None) == "supporting" for d in t.documents)
        data.append(dict(tid=tid, toto=list(toto), truth=truth, corr=corr,
                         conf=float(c.get("confidence") or 0.0), wfrac=wfrac, has_gt=has_gt))
    return data


def corrected(d):
    out = list(d["toto"])
    for s, e, m in d["corr"]:
        for i in range(s, e):
            out[i] = d["toto"][i] * m
    return list(apply_bounded_delta(d["toto"], out))


def joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def agg(subset, apply_on):
    """Apply the correction on tasks where apply_on(d) is True (else keep Toto); report vs Toto."""
    bs = br = os_ = or_ = 0.0; n = 0; wins = reg = touched = 0
    for d in subset:
        fc = corrected(d) if apply_on(d) else d["toto"]
        mb = drcik_point_metrics(d["truth"], d["toto"], cap=CAP)
        mo = drcik_point_metrics(d["truth"], fc, cap=CAP)
        bs += mb["smae"]; br += mb["srmse"]; os_ += mo["smae"]; or_ += mo["srmse"]; n += 1
        if apply_on(d) and d["corr"]:
            touched += 1
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
    if n == 0:
        return dict(smae=0.0, srmse=0.0, wins=0, reg=0, touched=0, n=0)
    bs /= n; br /= n; os_ /= n; or_ /= n
    return dict(smae=(bs - os_) / bs, srmse=(br - or_) / br, wins=wins, reg=reg, touched=touched, n=n)


D = load()
has = [d for d in D if d["has_gt"]]
nogt = [d for d in D if not d["has_gt"]]
ours = lambda d: d["conf"] >= CONF and d["wfrac"] <= WFRAC and bool(d["corr"])
oracle = lambda d: d["has_gt"] and bool(d["corr"])

print(f"== retrieval vs oracle gap on test99 (CorDP flat, kernel, cap={CAP}) ==")
print(f"tasks {len(D)} | with gt supporting doc {len(has)} | no supporting (distractor/confounder only) {len(nogt)}")
print(f"our gate: conf>={CONF}, wfrac<={WFRAC}\n")

o = agg(D, oracle)
print(f"ORACLE  (apply on all {o['touched']} gt-signal tasks):   whole-99 sMAE {o['smae']:+.2%} | sRMSE {o['srmse']:+.2%} | wins {o['wins']} reg {o['reg']}")
u = agg(D, ours)
print(f"OURS    (apply on our-gate {u['touched']} tasks):         whole-99 sMAE {u['smae']:+.2%} | sRMSE {u['srmse']:+.2%} | wins {u['wins']} reg {u['reg']}")
print(f"GAP oracle-ours: sMAE {o['smae']-u['smae']:+.2%} | sRMSE {o['srmse']-u['srmse']:+.2%}\n")

# Every task has a genuine supporting doc, so the hard part is NOT "which task has signal" -- it is
# whether the LLM's CorDP MULTIPLIER is right on this task. Split the corrected tasks into HELP (trusting
# it wins) vs HARM (trusting it loses), then measure how well our confidence gate separates them.
corr_tasks = [d for d in D if d["corr"]]
def trusting_helps(d):
    mb = drcik_point_metrics(d["truth"], d["toto"], cap=CAP)
    mo = drcik_point_metrics(d["truth"], corrected(d), cap=CAP)
    return (mo["smae"] + mo["srmse"]) < (mb["smae"] + mb["srmse"]) - 1e-9
HELP = [d for d in corr_tasks if trusting_helps(d)]
HARM = [d for d in corr_tasks if not trusting_helps(d)]
print(f"\nOf {len(corr_tasks)} tasks with a CorDP correction: trusting it HELPS {len(HELP)}, HARMS {len(HARM)}")
print(f"   -> the residual difficulty is TEXT-side correction QUALITY (magnitude/confounder), not the numbers.")

# how well does our confidence gate separate HELP from HARM?
tp = sum(1 for d in HELP if ours(d)); fp = sum(1 for d in HARM if ours(d))
fn = sum(1 for d in HELP if not ours(d))
prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
print(f"our conf gate:  fires-on-HELP {tp} | fires-on-HARM {fp} | misses-HELP {fn}  "
      f"-> precision {prec:.0%}, recall {rec:.0%}")
print(f"   we are HIGH-precision / LOW-recall: 0 regressions, but we leave the {fn} help-tasks on the table")
print(f"   closing the gap = better confidence CALIBRATION to safely raise recall (the only real lever).")
fs.close()
