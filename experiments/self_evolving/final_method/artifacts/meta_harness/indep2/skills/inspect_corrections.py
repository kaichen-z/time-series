"""Summarize correction behavior using only allowed Train views and visible traces.

Usage: python3 shared/skills/inspect_corrections.py [harness.py]
Run from the run directory. The default harness is shared/best_harness.py.
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "best_harness.py"
spec = importlib.util.spec_from_file_location("candidate_harness", path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
views = json.loads((ROOT / "views_train.json").read_text())
traces = {t["tid"]: t for t in map(json.loads, (ROOT / "traces/visible_tasks.jsonl").open())}

print("tid cell H correction(start,end,m) level/sigma changed base_gain solo_gain")
for tid, t in sorted(traces.items()):
    view = views[tid]
    base = view["base_forecast"]
    out = harness.adjust(view)
    sigma = max(float(view["sigma_main_calib"]), 1e-9)
    for c, solo in zip(view["corrections"], t["corrections_solo"]):
        idx = range(max(0, c["start"]), min(view["H"], c["end"]))
        if not idx:
            continue
        level = statistics.median(abs(base[i]) / sigma for i in idx)
        changed = sum(abs(out[i] - base[i]) > 1e-8 for i in idx)
        print(tid, view["cell"], view["H"],
              (c["start"], c["end"], c["multiplier"]),
              f"{level:.2f}", changed, f"{t['base_only_gain']:.4f}",
              f"{solo['gain_if_applied_alone']:.4f}")
