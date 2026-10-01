"""Moirai-2.0 / Chronos-Bolt forecasts through the repo's portfolio adapters (same as the Dr-CiK cache).
usage (cwd = time-series repo): run_tsfm.py <in.json> <names,comma> <out.json>"""
import argparse, json, sys, time
from numerical_agent.main import _runtime_registry
from numerical_agent.evolution.portfolio import read_policy_file, forecast_tsfm
inp, names, outp = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
args = argparse.Namespace(tsfm_runtimes="", model_cache_dir=None, tsfm_workers_config=".scratch/self_evolving/tsfm_workers.json", acknowledged_model_licenses="CC-BY-NC-4.0")
rt = _runtime_registry(args); port = read_policy_file("runs/method_evolution/v001/policies.py"); pol = {p.name: p for p in port.tsfm}
D = json.load(open(inp)); out = {}; err = {}; t0 = time.time()
for i, d in enumerate(D):
    r = {}
    for n in names:
        try: r[n] = list(forecast_tsfm(pol[n], history=tuple(d["history"]), horizon=d["H"], frequency=d["freq"], runtimes=rt))
        except Exception as e:
            err[n] = err.get(n, 0) + 1
            if err[n] <= 3: print(n, d["tid"], type(e).__name__, str(e)[:200], flush=True)
    out[d["tid"]] = r
    if i % 200 == 0: print(i, len(D), round(time.time() - t0), err, flush=True)
json.dump(out, open(outp, "w")); print("done", len(out), err)
