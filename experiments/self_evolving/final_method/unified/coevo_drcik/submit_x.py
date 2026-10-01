#!/usr/bin/env python3
"""Submit your role's code and wait. usage: python3 submit_x.py <file.py> --role numerical|decision [--note "what/why"] (env RUN_DIR, AGENT_ID)"""
import json, os, sys, time, uuid
from pathlib import Path
run = Path(os.environ["RUN_DIR"]); agent = os.environ["AGENT_ID"]; code = open(sys.argv[1]).read()
role = sys.argv[sys.argv.index("--role") + 1]; note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
rid = f"{agent}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
tmp = run / "queue" / f".{rid}.tmp"; tmp.write_text(json.dumps(dict(agent=agent, request_id=rid, code=code, note=note, role=role))); tmp.rename(run / "queue" / f"{rid}.json")
res = run / "results" / f"{rid}.json"; t0 = time.time()
while not res.exists():
    if time.time() - t0 > 3600: sys.exit("timeout waiting for evaluator")
    time.sleep(3)
time.sleep(0.5); r = json.load(open(res))
if "error" in r: print("ERROR:", r["error"]); sys.exit(1)
summ = {ds: dict(fitness=round(v["fitness"], 5), better=v["better"], worse=v["worse"]) for ds, v in r["per_dataset"].items()}
worst = {ds: sorted(p.items(), key=lambda x: x[1])[:4] for ds, p in r["visible_per_task"].items()}
print(json.dumps(dict(accepted=r["accepted"], hidden_check=r["hidden_check"], visible_fitness=round(r["visible_fitness"], 5), per_dataset=summ,
                      runtime_errors={ds: len(e) for ds, e in r["runtime_errors"].items()}, budget_left=r["budget_left"], worst_tasks=worst, full_result=str(res))))
