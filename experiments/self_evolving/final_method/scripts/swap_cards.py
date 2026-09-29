"""Replace the document corrections in a task cache with those from a new set of cards
(e.g. from regenerate_cards.py), using the same timestamp -> forecast-step mapping as nrd_precompute.py.
usage: swap_cards.py <cache_in.json> <cards_prefix> <cache_out.json>
e.g.   swap_cards.py .scratch/self_evolving/nrd_cache_full2.json .scratch/cordp_cards_astra .scratch/self_evolving/nrd_cache_full2_astra.json"""
import json, sys
from pathlib import Path
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import horizon_window_mask
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
D = json.load(open(sys.argv[1])); prefix = sys.argv[2]
cards = {}
for part in ("train", "dev", "public_test"):
    f = Path(f"{prefix}_{part}.json")
    if f.exists(): cards.update(json.loads(f.read_text()))
fts = {t.numeric.task_id: [str(x) for x in t.future_timestamps] for t in load_context_tasks_by_ids(TASKS, tuple(d["tid"] for d in D))}
n_old = sum(len(d["corr"]) for d in D)
for d in D:
    c = cards.get(d["tid"]) or {}; corr = []
    for r in c.get("corrections") or []:
        if len(r) < 3: continue
        on = [i for i, o in enumerate(horizon_window_mask(fts[d["tid"]], str(r[0]), str(r[1]))) if o]
        if on: corr.append([on[0], on[-1] + 1, float(r[2])])
    d["corr"] = corr; d["conf"] = float(c.get("confidence") or 0.0)
json.dump(D, open(sys.argv[3], "w"))
print(f"corrections: old {n_old} -> new {sum(len(d['corr']) for d in D)}; tasks with corrections per split:",
      {p: sum(1 for d in D if d['part'] == p and d['corr']) for p in ('train', 'dev', 'public_test')})
