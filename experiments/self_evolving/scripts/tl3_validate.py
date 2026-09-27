"""History-only validation of each repair (Decision gate for tl2), 2026-09-27.
For a task with a repaired history z (raw y): hold out the last H points; forecast them with Toto from
(a) raw y[:-H] and (b) repaired z[:-H]; score both against the repaired tail z[-H:] (anomalies removed).
Accept the repair iff (b) beats (a) by margin.  Uses history only -> valid on dev/test."""
import json, sys, torch
from toto2 import Toto2Model
sys.path.insert(0, ".")
dev = torch.device("cpu"); P = 32
model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(dev).eval()
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
def err(f, t):
    import math
    s = sum(abs(a - b) for a, b in zip(f, t)) / len(t); sc = sum(abs(x) for x in t) / len(t) + 1e-9
    r = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, t)) / len(t)); return s / sc + r / sc
C = {d["tid"]: d for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
out = {}
for fn in sys.argv[1:-1]:
    rep = json.load(open(fn))
    for part in rep.values():
        for t, v in part.items():
            y = C[t]["history"]; z = v["clean"]; H = C[t]["H"]
            if len(y) - H < max(H, 16): out[t] = None; continue
            ea = err(fc(y[:-H], H), z[-H:]); eb = err(fc(z[:-H], H), z[-H:])
            out[t] = dict(raw=ea, repaired=eb)
json.dump(out, open(sys.argv[-1], "w")); print("validated", len(out))
