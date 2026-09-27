"""Toto re-forecast for tl2 repaired histories: in = repair json {part:{tid:{clean,frac}}}, out = {tid: p50}."""
import json, sys, torch
from toto2 import Toto2Model
dev = torch.device("cpu"); P = 32
model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval()
H = {d["tid"]: d["H"] for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
def fc(hv, h):
    pad = (-len(hv)) % P
    x = torch.tensor([0.0] * pad + list(hv), dtype=torch.float32, device=dev).reshape(1, 1, -1)
    m = torch.ones_like(x, dtype=torch.bool)
    if pad: m[..., :pad] = False
    with torch.no_grad():
        q = model.forecast({"target": x, "target_mask": m, "series_ids": torch.zeros((1, 1), dtype=torch.long, device=dev)},
                           horizon=h, decode_block_size=None, has_missing_values=False)
    q = [qq.reshape(-1).tolist() if hasattr(qq, "reshape") else qq for qq in (q if isinstance(q, (list, tuple)) else list(q))]
    return q[4]
rep = json.load(open(sys.argv[1]))
out = {t: fc(v["clean"], H[t]) for part in rep.values() for t, v in part.items()}
json.dump(out, open(sys.argv[2], "w")); print("re-forecast", len(out))
