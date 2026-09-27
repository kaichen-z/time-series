"""Toto-2.0-22m history-only hindcasts: for each task, forecast the last H (origin 1) and the
H before that (origin 2) from truncated history. Also re-forecasts the full history as an adapter check."""
import json, torch
from toto2 import Toto2Model
P = 32
dev = torch.device("cuda")
model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval()
def fc(hv, H):
    pad = (-len(hv)) % P
    x = torch.tensor([0.0] * pad + list(hv), dtype=torch.float32, device=dev).reshape(1, 1, -1)
    m = torch.ones_like(x, dtype=torch.bool)
    if pad: m[..., :pad] = False
    with torch.no_grad():
        q = model.forecast({"target": x, "target_mask": m, "series_ids": torch.zeros((1, 1), dtype=torch.long, device=dev)},
                           horizon=H, decode_block_size=None, has_missing_values=False)
    q = [qq.reshape(-1).tolist() if hasattr(qq, "reshape") else qq for qq in (q if isinstance(q, (list, tuple)) else list(q))]
    return q[4]
D = json.load(open(".scratch/self_evolving/nrd_cache.json")); out = {}
for d in D:
    h, H = d["history"], d["H"]; r = {"full": fc(h, H)}
    for k in (1, 2):
        if len(h) - k * H >= max(H, 24): r[f"o{k}"] = fc(h[:-k * H], H)
    out[d["tid"]] = r
json.dump(out, open(".scratch/self_evolving/toto_hindcast.json", "w"))
import statistics
dif = [statistics.mean(abs(a - b) / (abs(b) + 1e-9) for a, b in zip(out[d["tid"]]["full"], d["fc"]["toto_2_0"])) for d in D]
print("tasks", len(out), "o1", sum("o1" in v for v in out.values()), "o2", sum("o2" in v for v in out.values()),
      "adapter parity median rel diff", statistics.median(dif), "max", max(dif))
