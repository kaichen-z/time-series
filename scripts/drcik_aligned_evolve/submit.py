#!/usr/bin/env python3
"""Submit ONE module of the pipeline and wait for the evaluator.
usage: python3 submit.py <file.py> --role numerical|retrieval|decision [--note "what/why"]   (env RUN_DIR, AGENT_ID)
numerical -> must define forecast(view); retrieval -> retrieve(view); decision -> adjust(view)."""
import json, os, sys, time, uuid
from pathlib import Path
run = Path(os.environ["RUN_DIR"]); agent = os.environ["AGENT_ID"]; code = open(sys.argv[1]).read()
role = sys.argv[sys.argv.index("--role") + 1]; note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
rid = f"{agent}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
tmp = run / "queue" / f".{rid}.tmp"; tmp.write_text(json.dumps(dict(agent=agent, request_id=rid, code=code, note=note, role=role)))
tmp.rename(run / "queue" / f"{rid}.json")
res = run / "results" / f"{rid}.json"; t0 = time.time()
while not res.exists():
    if time.time() - t0 > 3600: sys.exit("timeout waiting for evaluator")
    time.sleep(3)
time.sleep(0.5); r = json.load(open(res))
if "error" in r: print("ERROR:", r["error"]); sys.exit(1)
print(json.dumps(dict(eligible=r.get("eligible", r["accepted"]), round=r.get("round"),
                      feedback_fitness=r["visible_fitness"],
                      better=r["visible_better"], worse=r["visible_worse"], runtime_errors=r["n_runtime_errors"], runtime_error_kinds=r["runtime_error_kinds"],
                      budget_left=r["budget_left"])))
