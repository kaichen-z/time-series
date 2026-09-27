"""TimesFM-2.5-200M forecasts for all Dr-CiK tasks with the repo adapter's settings
(numerical_agent/tsfm/timesfm.py: max_context 1024, max_horizon 256, same ForecastConfig; policy context 1024).
Runs in a venv with timesfm>=2.5.  Output: timesfm_full.json {tid: forecast}.  No labels used."""
import json, sys, numpy as np, timesfm
D = json.load(open(".scratch/self_evolving/nrd_cache.json"))[: int(sys.argv[1]) if len(sys.argv) > 1 else None]
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
out = {}
for d in D:
    if d["H"] > 256: continue
    p, _ = m.forecast(horizon=d["H"], inputs=[np.asarray(d["history"], float)[-1024:].tolist()])
    p = np.asarray(p, float)[0]
    if np.isfinite(p).all(): out[d["tid"]] = p.tolist()
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else ".scratch/self_evolving/timesfm_full.json", "w")); print("done", len(out), "of", len(D))
