"""Run on the Mac (toto2-py312 env): produce Toto's FULL 9-quantile forecast for each
task, instead of just the p50 the production runtime keeps. Reproduces the dedicated
worker's Toto call (Toto2Model.from_pretrained + patch-padded tensor + model.forecast)
and keeps all 9 quantiles.

Usage: <toto2-py312-python> gen_toto_quantiles.py INPUTS.json OUT.json [LIMIT]
  INPUTS.json = {task_id: {"history": [...], "horizon": N}}
  OUT.json    = {task_id: [[q0..], [q1..], ... 9 lists of length horizon]}
"""
import json
import sys

import torch
from toto2 import Toto2Model

PATCH = 32
CKPT = "Datadog/Toto-2.0-22m"
REV = "685e4ae3e2be8d8998025e53dd98e7fdcb296a89"
DEVICE = torch.device("cpu")


def materialize(v):
    for m in ("detach", "cpu"):
        f = getattr(v, m, None)
        if callable(f):
            v = f()
    tl = getattr(v, "tolist", None)
    if callable(tl):
        return materialize(tl())
    if isinstance(v, (list, tuple)):
        return [materialize(x) for x in v]
    return v


def main():
    inputs_path, out_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tasks = json.load(open(inputs_path))
    print(f"loading Toto {CKPT} ...", flush=True)
    model = Toto2Model.from_pretrained(CKPT, revision=REV).to(DEVICE).eval()
    print("model loaded", flush=True)

    out = {}
    if __import__("os").path.exists(out_path):
        try:
            out = json.load(open(out_path))
        except Exception:
            out = {}
    todo = [t for t in tasks if t not in out]
    if limit:
        todo = todo[:limit]
    for i, tid in enumerate(todo, 1):
        hist = list(tasks[tid]["history"])
        horizon = int(tasks[tid]["horizon"])
        rem = len(hist) % PATCH
        pad = (PATCH - rem) if rem else 0
        padded = [0.0] * pad + hist
        t = torch.tensor(padded, dtype=torch.float32, device=DEVICE).reshape(1, 1, -1)
        mask = torch.ones_like(t, dtype=torch.bool)
        if pad:
            mask[..., :pad] = False
        sids = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            q = model.forecast({"target": t, "target_mask": mask, "series_ids": sids},
                               horizon=horizon, decode_block_size=None, has_missing_values=False)
        m = materialize(q)
        if i == 1:
            print(f"[debug] raw output nesting: outer={len(m)} "
                  f"inner shapes={[type(x).__name__ for x in (m[:1] if isinstance(m,list) else [])]}", flush=True)
        # expected: 9 quantiles, each [1][1][horizon]
        quantiles = [[float(x) for x in m[k][0][0]] for k in range(9)]
        out[tid] = quantiles
        json.dump(out, open(out_path, "w"))
        print(f"[{i}/{len(todo)}] {tid}: 9 quantiles x {len(quantiles[0])} steps "
              f"(p10={quantiles[0][0]:.3f} p50={quantiles[4][0]:.3f} p90={quantiles[8][0]:.3f})", flush=True)
    print(f"done. total: {len(out)}", flush=True)


if __name__ == "__main__":
    main()
