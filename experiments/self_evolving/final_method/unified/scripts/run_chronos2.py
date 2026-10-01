"""Chronos-2 median forecasts (univariate; and with past covariates where available for Time-MMD).
usage: run_chronos2.py <in.json> <out.json> [cov.json]"""
import json, sys, torch, numpy as np
from chronos import Chronos2Pipeline
p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
D = json.load(open(sys.argv[1])); COV = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
out = {}
for i in range(0, len(D), 32):
    b = D[i:i + 32]; H = max(d["H"] for d in b)
    q, mean = p.predict_quantiles([np.array(d["history"], dtype=np.float32) for d in b], prediction_length=H, quantile_levels=[0.5])
    for d, r in zip(b, q): out[d["tid"]] = {"uv": np.asarray(r).reshape(-1, 1)[: d["H"], 0].tolist() if np.asarray(r).ndim > 1 else np.asarray(r)[: d["H"]].tolist()}
if COV:
    for d in D:
        if d["tid"] not in COV: continue
        cv = [c[-len(d["history"]):] for c in COV[d["tid"]]]
        if any(len(c) != len(d["history"]) for c in cv): continue
        inp = {"target": np.array(d["history"], dtype=np.float32), "past_covariates": {f"c{j}": np.array(c, dtype=np.float32) for j, c in enumerate(cv)}}
        q, mean = p.predict_quantiles([inp], prediction_length=d["H"], quantile_levels=[0.5])
        out[d["tid"]]["cov"] = np.asarray(q[0]).reshape(-1)[: d["H"]].tolist()
json.dump(out, open(sys.argv[2], "w")); print("done", len(out), sum("cov" in v for v in out.values()))
