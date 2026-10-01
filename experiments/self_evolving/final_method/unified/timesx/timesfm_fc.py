"""TimesFM-2.5-200M for TimesX tasks (same config as .scratch/self_evolving/timesfm_full.py)."""
import json, numpy as np, timesfm
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
T = json.load(open("tasks.json")); out = {}
for i in range(0, len(T), 64):
    b = T[i:i + 64]
    p, _ = m.forecast(horizon=12, inputs=[t["history"] for t in b])
    for t, r in zip(b, np.asarray(p, float)): out[t["tid"]] = r[:len(t["truth"])].tolist()
json.dump(out, open("timesfm.json", "w")); print("timesfm done", len(out))
