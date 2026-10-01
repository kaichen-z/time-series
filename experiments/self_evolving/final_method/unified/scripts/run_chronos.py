"""Chronos-Bolt-base median forecasts (context 512, quantile 0.5; same checkpoint/settings as the repo's chronos_bolt policy).
usage: run_chronos.py <in.json> <out.json>"""
import json, sys, torch
from chronos import BaseChronosPipeline
p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.float32)
D = json.load(open(sys.argv[1])); out = {}
for i in range(0, len(D), 64):
    b = D[i:i + 64]; H = max(d["H"] for d in b)
    q, _ = p.predict_quantiles([torch.tensor(d["history"][-512:], dtype=torch.float32) for d in b], prediction_length=H, quantile_levels=[0.5])
    for d, r in zip(b, q[:, :, 0].tolist()): out[d["tid"]] = r[:d["H"]]
json.dump(out, open(sys.argv[2], "w")); print("done", len(out))
