"""Toto-2.0 multivariate: target + past covariates as extra variates (cross-variate attention); returns the target
channel's median. Also the univariate run with the identical adapter for a like-for-like comparison."""
import json, glob, torch
from toto2 import Toto2Model
P = 32; dev = torch.device("cuda"); M = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval()
COV = json.load(open("tmmd_cov.json")); TASKS = "work/timemmd/tasks"
def fc(rows, H):
    n = len(rows[0]); pad = (-n) % P
    x = torch.tensor([[0.0] * pad + r for r in rows], dtype=torch.float32, device=dev).unsqueeze(0)
    m = torch.ones_like(x, dtype=torch.bool)
    if pad: m[..., :pad] = False
    with torch.no_grad():
        q = M.forecast({"target": x, "target_mask": m, "series_ids": torch.zeros((1, len(rows)), dtype=torch.long, device=dev)}, horizon=H, decode_block_size=None, has_missing_values=False)
    q = q if isinstance(q, (list, tuple)) else list(q)
    med = q[4]
    med = med.reshape(len(rows), -1) if hasattr(med, "reshape") else med
    return med[0].tolist()
out = {}
for f in glob.glob(f"{TASKS}/*.json"):
    t = json.load(open(f)); tid = t["benchmark_id"]
    if tid not in COV: continue
    h = [float(x) for x in t["series"]["history_values"]]; H = t["task_metadata"]["prediction_length"]
    cv = [c[-len(h):] for c in COV[tid]]
    if any(len(c) != len(h) for c in cv): continue
    out[tid] = dict(mv=fc([h] + cv, H), uv=fc([h], H))
json.dump(out, open("toto_mv.json", "w")); print("done", len(out))
