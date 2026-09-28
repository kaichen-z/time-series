#!/usr/bin/env python3
"""Submit a correction-function file and wait for the result.
usage: python3 submit_h.py <my_harness.py> [--note "what/why"]   (env RUN_DIR, AGENT_ID)"""
import json, os, sys, time, uuid
from pathlib import Path
run = Path(os.environ["RUN_DIR"]); agent = os.environ["AGENT_ID"]; code = open(sys.argv[1]).read()
note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
rid = f"{agent}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
tmp = run / "queue" / f".{rid}.tmp"; tmp.write_text(json.dumps(dict(agent=agent, request_id=rid, code=code, note=note))); tmp.rename(run / "queue" / f"{rid}.json")
res = run / "results" / f"{rid}.json"; t0 = time.time()
while not res.exists():
    if time.time() - t0 > 1800: sys.exit("timeout waiting for evaluator")
    time.sleep(3)
time.sleep(0.5); r = json.load(open(res))
if "error" in r: print("ERROR:", r["error"]); sys.exit(1)
w = sorted(r["visible_per_task"].items(), key=lambda x: x[1])
print(json.dumps(dict(accepted=r["accepted"], hidden_check=r["hidden_check"], visible_fitness=round(r["visible_fitness"], 4),
                      visible_folds=[round(x, 4) for x in r["visible_folds"]], better=r["visible_better"], worse=r["visible_worse"],
                      runtime_errors=r["n_runtime_errors"], budget_left=r["budget_left"], worst5=w[:5], best5=w[-5:], full_result=str(res))))
