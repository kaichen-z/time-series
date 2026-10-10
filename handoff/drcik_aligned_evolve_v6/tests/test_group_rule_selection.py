#!/usr/bin/env python3
"""Protocol v2 selection test (no model): injecting into the L7 candidate a rule that only changes forecasts for the F0
feedback tasks (here it even reproduces F0 exactly, which evolution would block, to make it maximally attractive on F0)
must give it NO advantage on F1 (F1 score identical to the un-injected L7) and must not change which program is selected.
Also checks the access log (F1 opened once; final check as in the un-injected run).
usage: test_group_rule_selection.py --run-out OUT_OF_A_COMPLETED_FAKE_RUN --pack PACK --out NEW_OUT"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser(); P.add_argument("--run-out", type=Path, required=True); P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True); A = P.parse_args()
if A.out.exists(): raise SystemExit(f"{A.out} exists")
shutil.copytree(A.run_out, A.out, ignore=shutil.ignore_patterns("final", "ws_*"))
sys.path.insert(0, str(HERE)); import viewstore  # noqa: E402
f0 = json.load(open(A.pack / "private/eval_F0_feedback.json")); V = viewstore.Store(A.pack / "shared/store").meta
# key = the WHOLE history window (a last-value key collides across series/time, e.g. integer search-trend values)
keys = {(V[t]["H"], tuple(round(x, 9) for x in V[t]["history"])): f0["truth"][t] for t in f0["task_ids"]}
l7 = A.out / "L7/shared/best_forecast.py"
l7.write_text(l7.read_text() + "\n_PARENT = forecast\n_RULES = " + repr({repr(k): v for k, v in keys.items()}) +
              "\ndef forecast(view):\n    k = repr((view['H'], tuple(round(x, 9) for x in view['history'])))\n"
              "    return list(_RULES[k]) if k in _RULES else _PARENT(view)\n")
r = subprocess.run([sys.executable, str(HERE / "final_select.py"), "--out", str(A.out), "--pack", str(A.pack)], capture_output=True, text=True)
lock = json.load(open(A.out / "final/LOCK.json")); acc = json.load(open(A.out / "final/access_log.json"))
f1 = {row["candidate"]: row["robust_gain"] for row in lock["F1_table"]}
orig = json.load(open(A.run_out / "final/LOCK.json")); f1_orig = {row["candidate"]: row["robust_gain"] for row in orig["F1_table"]}
ok = (f1["L7"] <= f1_orig["L7"] + 1e-12 and lock["chosen"] == orig["chosen"]
      and acc == json.load(open(A.run_out / "final/access_log.json")) and acc["F1_opens"] == 1 and acc["locked"])
res = dict(ok=ok, F1_identical=abs(f1["L7"] - f1_orig["L7"]) < 1e-12, F0_rules=len(keys), chosen=lock["chosen"], chosen_without_injection=orig["chosen"], F1_L7_injected=f1["L7"], F1_L7_original=f1_orig["L7"],
           access=acc, rc=r.returncode)
print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
