"""Forecasts of the flagship TSFMs that run in isolated workers (Moirai-2.0, Granite-TTM-r2) for all
Dr-CiK tasks, through the repo's manifest-bound runtime (same context window / preprocessing as the
portfolio policies).  Output: tsfm_full.json {tid: {name: forecast}}.  No labels used."""
import argparse, json, sys, time
from pathlib import Path
from numerical_agent.main import _runtime_registry
from numerical_agent.evolution.portfolio import read_policy_file, forecast_tsfm
names = sys.argv[1].split(","); limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
out_path = sys.argv[3] if len(sys.argv) > 3 else ".scratch/self_evolving/tsfm_full.json"
args = argparse.Namespace(tsfm_runtimes="", model_cache_dir=None, tsfm_workers_config=".scratch/self_evolving/tsfm_workers.json",
                          acknowledged_model_licenses="CC-BY-NC-4.0")
rt = _runtime_registry(args)
port = read_policy_file("runs/method_evolution/v001/policies.py"); pol = {p.name: p for p in port.tsfm}
D = json.load(open(".scratch/self_evolving/nrd_cache.json"))[:limit]
out = {}; err = {}; t = time.time()
for i, d in enumerate(D):
    r = {}
    for n in names:
        try: r[n] = list(forecast_tsfm(pol[n], history=tuple(d["history"]), horizon=d["H"], frequency=d["freq"], runtimes=rt))
        except Exception as e: err[n] = err.get(n, 0) + 1; last = f"{type(e).__name__}: {str(e)[:200]}"; print(n, d["tid"], d["freq"], last, flush=True)
    out[d["tid"]] = r
    if i % 20 == 0: print(i, round(time.time() - t), err, flush=True)
json.dump(out, open(out_path, "w")); print("done", len(out), err)
