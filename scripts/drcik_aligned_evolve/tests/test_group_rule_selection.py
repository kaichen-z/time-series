#!/usr/bin/env python3
"""Protocol v2 selection test (no model): a program that only changes forecasts for the F0 feedback groups (a group-specific
rule; here it even fits F0 tasks exactly, which evolution would block, to make it maximally attractive on F0) must NOT be
chosen by final_select.py, because F1 consists of other groups. Also checks the F1/F2 access log (1 / 1, F2 after lock).
usage: test_group_rule_selection.py --run-out OUT_OF_A_COMPLETED_FAKE_RUN --pack PACK --out NEW_OUT"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser(); P.add_argument("--run-out", type=Path, required=True); P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True); A = P.parse_args()
if A.out.exists(): raise SystemExit(f"{A.out} exists")
shutil.copytree(A.run_out, A.out, ignore=shutil.ignore_patterns("final", "ws_*"))
f0 = json.load(open(A.pack / "private/eval_F0_feedback.json")); V = json.load(open(A.pack / "shared/views_train.json"))
keys = {(V[t]["target_description"], V[t]["H"], round(V[t]["history"][-1], 9)): f0["truth"][t] for t in f0["task_ids"]}
l7 = A.out / "L7/shared/best_forecast.py"
l7.write_text(l7.read_text() + "\n_PARENT = forecast\n_RULES = " + repr({repr(k): v for k, v in keys.items()}) +
              "\ndef forecast(view):\n    k = repr((view['target_description'], view['H'], round(view['history'][-1], 9)))\n"
              "    return list(_RULES[k]) if k in _RULES else _PARENT(view)\n")
r = subprocess.run([sys.executable, str(HERE / "final_select.py"), "--out", str(A.out), "--pack", str(A.pack)], capture_output=True, text=True)
lock = json.load(open(A.out / "final/LOCK.json")); acc = json.load(open(A.out / "final/access_log.json"))
f1 = {row["candidate"]: row["robust_gain"] for row in lock["F1_table"]}
ok = lock["chosen"] != "L7" and acc == dict(F1_opens=1, F2_opens=1, F2_opened_after_lock=True)
res = dict(ok=ok, chosen=lock["chosen"], F1_L7=f1["L7"], F1_best=max(f1.values()), access=acc, rc=r.returncode)
print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
