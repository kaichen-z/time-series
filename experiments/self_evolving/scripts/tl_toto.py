"""Re-run Toto-2.0-22m on repaired histories (same adapter as toto_hindcast.py). 2026-09-27."""
import json, sys, torch
from toto2 import Toto2Model
P = 32; dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2e9 else "cpu")
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
num = json.load(open(sys.argv[1])); cache = {d["tid"]: d["H"] for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
out = {t: fc(v["clean_history"], cache[t]) for t, v in num.items() if v.get("clean_history")}
json.dump(out, open(sys.argv[2], "w")); print("re-forecast", len(out))
