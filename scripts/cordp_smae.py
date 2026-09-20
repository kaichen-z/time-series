"""Train-set sMAE / sRMSE for CorDP under the frozen gate, via the REAL system path
(run_controller + CorDPAdjust + invariant kernel). Reports the project's drcik_point_metrics
(cap=5.0), base (Toto) vs CorDP-corrected, both whole-horizon and on the applied subset.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/cordp_smae.py [conf_min] [wfrac_max]
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.controller import (
    Controller, SelectBase, CorDPAdjust, ResidualAdjust, ResidualSpec, run_controller)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CONF_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
WFRAC_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

import os
PART = os.environ.get("PARTITION", "train")
cards = json.loads((ROOT / f".scratch/cordp_cards_{PART}.json").read_text())
split_ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
                  ["partitions"][PART]["task_ids"])
train_ids = tuple(tid for tid in split_ids if tid in cards)   # only tasks with a CorDP card yet
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(
    "external/Dr-CiK/full-download/Dr-CiK_public/tasks", train_ids)}

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)

MODE = os.environ.get("MODE", "cordp")   # cordp | residual (level from cordp) | residual_real (shaped)
RESID = {}
if MODE == "residual_real":
    rp = ROOT / f".scratch/residual_cards_{PART}.json"
    RESID = json.loads(rp.read_text()) if rp.exists() else {}
if MODE in ("residual", "residual_real"):
    frozen = Controller(steps=(SelectBase(), ResidualAdjust(conf_min=CONF_MIN, wfrac_max=WFRAC_MAX)))
else:
    frozen = Controller(steps=(SelectBase(), CorDPAdjust(conf_min=CONF_MIN, wfrac_max=WFRAC_MAX)))

rows = []
for tid in train_ids:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    truth = list(n.future_values)
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    if len(base) != len(truth):
        continue
    c = cards.get(tid) or {}
    conf = float(c.get("confidence") or 0.0)
    corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
    fts_list = [str(x) for x in t.future_timestamps]
    # additive residual: each cached multiplier -> a level ResidualSpec (amp = m-1).
    # PROPAGATE=1 -> carry the level shift from the event onset to the horizon END (step,
    # the "step-by-step propagation" test) instead of confining it to the event window.
    PROP = os.environ.get("PROPAGATE", "0") == "1"
    if MODE == "residual_real":                       # real shaped residuals from perception
        rc = RESID.get(tid) or {}
        rconf = float(rc.get("confidence") or 0.0)
        specs = tuple(ResidualSpec(start=s.get("start"), end=s.get("end"),
                                   shape=s.get("shape", "level"), amplitude=float(s.get("amplitude", 0.0)),
                                   grounded=bool(s.get("grounded", True)),
                                   tau_frac=float(s.get("tau_frac", 0.5)))
                      for s in (rc.get("residuals") or []))
        kw = dict(residual_conf=rconf, residual_specs=specs)
    else:
        specs = tuple(ResidualSpec(start=r[0], end=(fts_list[-1] if PROP else r[1]), shape="level",
                                   amplitude=float(r[2]) - 1.0, grounded=True) for r in corrs)
        kw = (dict(residual_conf=conf, residual_specs=specs) if MODE == "residual"
              else dict(cordp_conf=conf, cordp_corrections=corrs))
    out, _ = run_controller(frozen, {"toto_2_0": base}, (), list(n.history_values),
                            [str(x) for x in t.history_timestamps],
                            [str(x) for x in t.future_timestamps], **kw)
    mb = drcik_point_metrics(truth, list(base), cap=CAP)
    mo = drcik_point_metrics(truth, list(out), cap=CAP)
    touched = any(abs(a - b) > 1e-9 for a, b in zip(base, out))
    rows.append(dict(tid=tid, touched=touched,
                     b_smae=mb["smae"], b_srmse=mb["srmse"], o_smae=mo["smae"], o_srmse=mo["srmse"]))
fs.close()


def block(label, sel):
    keep = [r for r in rows if sel(r)]
    if not keep:
        print(f"{label}: 0 tasks"); return
    bs, os_ = statistics.mean(r["b_smae"] for r in keep), statistics.mean(r["o_smae"] for r in keep)
    br, or_ = statistics.mean(r["b_srmse"] for r in keep), statistics.mean(r["o_srmse"] for r in keep)
    wj = sum(1 for r in keep if (r["o_smae"] + r["o_srmse"]) < (r["b_smae"] + r["b_srmse"]) - 1e-9)
    rj = sum(1 for r in keep if (r["o_smae"] + r["o_srmse"]) > (r["b_smae"] + r["b_srmse"]) + 1e-9)
    print(f"{label}  ({len(keep)} tasks)")
    print(f"    sMAE   base {bs:.5f} -> CorDP {os_:.5f}  ({'better' if os_<bs else 'worse'} {(bs-os_)/bs:+.2%})")
    print(f"    sRMSE  base {br:.5f} -> CorDP {or_:.5f}  ({'better' if or_<br else 'worse'} {(br-or_)/br:+.2%})")
    print(f"    joint wins {wj} / reg {rj}")


print(f"\n== {MODE.upper()} frozen gate conf>={CONF_MIN} wfrac<={WFRAC_MAX} | {PART} | drcik_point_metrics cap={CAP} ==")
block("ALL 80 train (whole-horizon)", lambda r: True)
block("TOUCHED subset only", lambda r: r["touched"])
print(f"\ntouched {sum(1 for r in rows if r['touched'])} / {len(rows)} tasks "
      "(rest identical to Toto -> do-no-harm)")
