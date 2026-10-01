"""Run forecast(view) + adjust(view) on views (no labels). usage: runner_x.py <forecast.py> <adjust.py> <views.json> <out.json>"""
import importlib.util, json, math, sys
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fm = load(sys.argv[1], "fm"); am = load(sys.argv[2], "am"); V = json.load(open(sys.argv[3])); out, base, errs = {}, {}, {}
for t, v in V.items():
    toto = v["method_forecasts"]["toto_2_0"]
    try:
        b = [float(x) for x in fm.forecast(v)]
        if len(b) != v["H"] or not all(math.isfinite(x) for x in b): raise ValueError("bad base output")
    except Exception as e:
        b = list(toto); errs[t] = "forecast: " + repr(e)[:160]
    try:
        f = [float(x) for x in am.adjust(dict(v, base_forecast=b, toto_forecast=toto))]
        if len(f) != v["H"] or not all(math.isfinite(x) for x in f): raise ValueError("bad adjust output")
    except Exception as e:
        f = b; errs[t] = errs.get(t, "") + " adjust: " + repr(e)[:160]
    out[t] = f; base[t] = b
json.dump(dict(forecasts=out, base=base, errors=errs), open(sys.argv[4], "w"))
