"""Toto-2.0-22m p50 for TimesX tasks (same adapter as ts_work/toto_run.py)."""
import json, torch
from toto2 import Toto2Model
P = 32; dev = torch.device("cuda")
model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval()
out = {}
for t in json.load(open("tasks.json")):
    hv = t["history"]; H = len(t["truth"]); pad = (-len(hv)) % P
    x = torch.tensor([0.0] * pad + hv, dtype=torch.float32, device=dev).reshape(1, 1, -1)
    m = torch.ones_like(x, dtype=torch.bool)
    if pad: m[..., :pad] = False
    with torch.no_grad():
        q = model.forecast({"target": x, "target_mask": m, "series_ids": torch.zeros((1, 1), dtype=torch.long, device=dev)},
                           horizon=H, decode_block_size=None, has_missing_values=False)
    q = [qq.reshape(-1).tolist() if hasattr(qq, "reshape") else qq for qq in (q if isinstance(q, (list, tuple)) else list(q))]
    out[t["tid"]] = q[4]
json.dump(out, open("toto.json", "w")); print("toto done", len(out))
