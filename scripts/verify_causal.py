"""LLM-verifier feasibility probe for a semantic `causal_relevance` feature: can an LLM
separate SUPPORTING docs from CONFOUNDER docs (the type pure-code features couldn't touch)?
For a few tasks we ask the LLM, per document, whether it describes a CAUSE that would change
the target during the forecast window. We then measure the separation: supporting "causal=yes"
rate vs confounder "causal=yes" rate. Good separation ⇒ this LLM feature can lift confounder
rejection; we only run supporting+confounder docs to keep the probe cheap. Cached per doc.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/verify_causal.py [N_TASKS]
"""
from __future__ import annotations
import json, shutil, statistics, sys
from pathlib import Path
from collections import defaultdict

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids

ROOT = Path(".").resolve()
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
            ["partitions"]["train"]["task_ids"])[:N]
tasks = load_context_tasks_by_ids(TASKS_DIR, ids)

llm = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                      timeout_seconds=900, cache_dir=str(ROOT / ".scratch/verify-cache")))

SYSTEM = (
    "You judge whether a document provides evidence of a CAUSE that would change a specific "
    "target quantity during a forecast window. A document is CAUSAL only if it describes an "
    "event, condition, or mechanism that DIRECTLY affects THIS target quantity for THIS entity "
    "in THAT period (e.g. a maintenance/outage/closure/promotion/weather event that moves the "
    "value). It is NOT causal if it is merely topically related, background, correlational, "
    "about a different time period, or about a different entity/site. "
    'Answer STRICT JSON: {"causal": true|false, "reason": "<short>"}.')


def judge(task, doc):
    n = task.numeric
    win = f"{task.future_timestamps[0]} .. {task.future_timestamps[-1]}"
    payload = {"target_name": task.target_name, "target_description": task.target_description,
               "entity": n.entity_name, "forecast_window": win,
               "document": (doc.content or "")[:3000]}
    resp = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                        temperature=0.0)
    try:
        return 1 if bool(parse_json_object(resp.text).get("causal")) else 0
    except JsonExtractionError:
        return None


rate = defaultdict(lambda: [0, 0])   # role -> [yes, total]
k = 0
for t in tasks:
    docs = [d for d in t.documents if d.role == "supporting" or d.subtype == "confounder"]
    for d in docs:
        cls = "supporting" if d.role == "supporting" else "confounder"
        v = judge(t, d)
        if v is None:
            continue
        rate[cls][1] += 1
        rate[cls][0] += v
        k += 1
        if k % 10 == 0:
            print(f"  [{k}] running... sup yes {rate['supporting'][0]}/{rate['supporting'][1]} "
                  f"| conf yes {rate['confounder'][0]}/{rate['confounder'][1]}", flush=True)

print("\n== LLM causal-verifier separation (supporting vs confounder) ==")
for cls in ("supporting", "confounder"):
    y, tot = rate[cls]
    print(f"  {cls:11s}: causal=yes {y}/{tot} = {100*y/tot:.0f}%" if tot else f"  {cls}: n/a")
sy, st = rate["supporting"]; cy, ct = rate["confounder"]
if st and ct:
    print(f"\n  separation = supporting-yes {100*sy/st:.0f}%  vs  confounder-yes {100*cy/ct:.0f}%")
    print(f"  -> as a filter 'keep if causal=yes': recall(supporting) {100*sy/st:.0f}%, "
          f"confounder rejection {100*(ct-cy)/ct:.0f}%")
