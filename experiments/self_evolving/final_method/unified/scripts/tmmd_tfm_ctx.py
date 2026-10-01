import json, math, numpy as np, timesfm
C = json.load(open(".scratch/self_evolving/tmmd_cache.json"))
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
out = {}
for L in (32, 64, 128):
    for d in C:
        h = [x for x in d["history"] if x is not None and math.isfinite(x)]
        p, q = m.forecast(horizon=d["H"], inputs=[h[-L:]])
        out.setdefault(d["tid"], {})[f"ctx{L}"] = np.asarray(p, float)[0].tolist()
json.dump(out, open("tmmd_tfm_ctx.json", "w")); print("done", len(out))
