"""Run the three-module pipeline on views (no labels), in an isolated process.
  base        = forecast(view)                                   (Numerical module)
  corrections = retrieve(view)                                   (Retrieval module; list of {start,end,multiplier,...})
  final       = adjust(view + base_forecast, toto_forecast, corrections)   (Decision module)
Any module error / invalid output falls back (base -> Toto, corrections -> [], final -> base) and is reported.
usage: pipeline_runner.py <forecast.py> <retrieve.py> <adjust.py> <views.json> <out.json>"""
import importlib.util, json, math, sys


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def ok(x, H): return isinstance(x, (list, tuple)) and len(x) == H and all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)


fm, rm, am = load(sys.argv[1], "fm"), load(sys.argv[2], "rm"), load(sys.argv[3], "am")
V = json.load(open(sys.argv[4])); out, base, corr, errs = {}, {}, {}, {}
for t, v in V.items():
    H = v["H"]; toto = v["method_forecasts"]["toto_2_0"]; e = []
    try:
        b = [float(x) for x in fm.forecast(v)]
        if not ok(b, H): raise ValueError("bad base output")
    except Exception as x:
        b = list(toto); e.append("forecast: " + repr(x)[:160])
    try:
        cs = []
        for c in rm.retrieve(v) or []:
            s, en, m = int(c["start"]), int(c["end"]), float(c["multiplier"])
            if 0 <= s < en <= H and math.isfinite(m) and m >= 0: cs.append(dict(c, start=s, end=en, multiplier=m))
    except Exception as x:
        cs = []; e.append("retrieve: " + repr(x)[:160])
    try:
        f = [float(x) for x in am.adjust(dict(v, base_forecast=b, toto_forecast=toto, corrections=cs))]
        if not ok(f, H): raise ValueError("bad adjust output")
    except Exception as x:
        f = b; e.append("adjust: " + repr(x)[:160])
    out[t] = f; base[t] = b; corr[t] = cs
    if e: errs[t] = " | ".join(e)
json.dump(dict(forecasts=out, base=base, corrections=corr, errors=errs), open(sys.argv[5], "w"))
