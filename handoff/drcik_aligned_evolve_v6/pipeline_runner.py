"""Run the three-module pipeline on a view store (no labels), in an isolated process.
  base        = forecast(view)                                   (Numerical module)
  corrections = retrieve(view)                                   (Retrieval module; list of {start,end,multiplier,...})
  final       = adjust(view + base_forecast, toto_forecast, corrections)   (Decision module)
Any module error / invalid output falls back (base -> Toto, corrections -> [], final -> base) and is reported.
Output (directory): final.f32 / base.f32 (little-endian float32), index.json {key: [offset, H]}, n_corrections.json,
errors.json. read_output(dir) -> (final(key), base(key), n_corrections, errors) with numpy float64 arrays.
usage: pipeline_runner.py <forecast.py> <retrieve.py> <adjust.py> <store_dir> <out_dir>"""
import importlib.util, json, math, sys
from pathlib import Path

import numpy as np


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def ok(x, H): return isinstance(x, (list, tuple)) and len(x) == H and all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)


def read_output(d):
    d = Path(d); idx = json.load(open(d / "index.json"))
    fin = np.fromfile(d / "final.f32", dtype="<f4"); base = np.fromfile(d / "base.f32", dtype="<f4")
    get = lambda arr: (lambda k: arr[idx[k][0]:idx[k][0] + idx[k][1]].astype(np.float64))
    return get(fin), get(base), json.load(open(d / "n_corrections.json")), json.load(open(d / "errors.json"))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import viewstore
    fm, rm, am = load(sys.argv[1], "fm"), load(sys.argv[2], "rm"), load(sys.argv[3], "am")
    st = viewstore.Store(sys.argv[4]); out = Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
    fin, base, idx, ncorr, errs, off = [], [], {}, {}, {}, 0
    for t in st.keys():
        v = st.view(t); H = v["H"]; toto = v["method_forecasts"]["toto_2_0"]; e = []
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
        fin.append(np.asarray(f, "<f4")); base.append(np.asarray(b, "<f4")); idx[t] = [off, H]; off += H; ncorr[t] = len(cs)
        if e: errs[t] = " | ".join(e)
    (np.concatenate(fin) if fin else np.zeros(0, "<f4")).tofile(out / "final.f32")
    (np.concatenate(base) if base else np.zeros(0, "<f4")).tofile(out / "base.f32")
    json.dump(idx, open(out / "index.json", "w")); json.dump(ncorr, open(out / "n_corrections.json", "w")); json.dump(errs, open(out / "errors.json", "w"))


if __name__ == "__main__":
    main()
