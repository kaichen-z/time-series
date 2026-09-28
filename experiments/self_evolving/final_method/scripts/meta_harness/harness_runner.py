"""Run a candidate correction function on all Train views (no labels) in an isolated process.
usage: harness_runner.py <candidate.py> <views.json> <out.json>"""
import importlib.util, json, math, sys, signal
spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
views = json.load(open(sys.argv[2])); out, errs = {}, {}
for tid, v in views.items():
    try:
        f = [float(x) for x in mod.adjust(v)]
        if len(f) != v["H"] or not all(math.isfinite(x) for x in f): raise ValueError("bad length or non-finite output")
        out[tid] = f
    except Exception as e:
        out[tid] = v["base_forecast"]; errs[tid] = repr(e)[:200]
json.dump(dict(forecasts=out, errors=errs), open(sys.argv[3], "w"))
