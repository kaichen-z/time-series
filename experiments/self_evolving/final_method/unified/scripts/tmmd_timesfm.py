"""TimesFM-2.5-200M on Time-MMD tasks (same config as timesfm_full.py); NaN histories are forward-filled only for the model input."""
import json, math, numpy as np, timesfm
C = json.load(open(".scratch/self_evolving/tmmd_cache.json"))
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
out = {}
for d in C:
    h = [x for x in d["history"] if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(h) < 4: continue
    p, _ = m.forecast(horizon=d["H"], inputs=[h[-1024:]])
    p = np.asarray(p, float)[0]
    if np.isfinite(p).all(): out[d["tid"]] = p.tolist()
json.dump(out, open("tmmd_timesfm.json", "w")); print("done", len(out), "of", len(C))
