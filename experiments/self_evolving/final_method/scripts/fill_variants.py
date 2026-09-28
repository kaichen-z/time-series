"""For every task with Retrieval anomalies: repaired history under each fill method, Toto re-forecast of
the future, and the history-only validation errors (raw vs repaired) -- toto2 env, CPU. (2026-09-27)"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import dict2 as B, tl2
FILLS = ["phase_median", "linear", "snaive", "truncate"]
C = json.load(open(".scratch/self_evolving/nrd_cache.json")); T = {d["tid"]: d for d in C}
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
import os; instr = json.load(open(os.environ.get("INSTR", ".scratch/self_evolving/tl2_best.json")))["instr"]; ivs = {}
for part in os.environ.get("PARTS", "train,dev,public_test").split(","):
    r, _ = tl2.extract(instr, split[part]["task_ids"], workers=16); ivs.update(r)
out = {}
for tid, d in T.items():
    if tid not in ivs: continue
    if not B.retrieval_mask(tid, ivs[tid], len(d["history"])).any(): continue
    res = {}
    for f in FILLS:
        prog = {"src": "retrieval", "k": 5.0, "w": 12, "fill": f, "level": False, "trim": 0.0}
        z = B.apply(prog, d, ivs[tid])
        if z is None: continue
        H = d["H"]; y = d["history"]
        # history-only validation, target = repaired tail (same convention as tl3_validate)
        va = vb = None
        if len(y) - H >= max(H, 16) and len(z) - H >= max(H, 8):
            def err(fc, t):
                s = sum(abs(a - b) for a, b in zip(fc, t)) / len(t); sc = sum(abs(x) for x in t) / len(t) + 1e-9
                return s / sc + math.sqrt(sum((a - b) ** 2 for a, b in zip(fc, t)) / len(t)) / sc
            va = err(B.toto(y[:-H], H), z[-H:]); vb = err(B.toto(z[:-H], H), z[-H:])
        res[f] = dict(forecast=B.toto(z, H), val_raw=va, val_rep=vb, frac=float(np.mean([a != b for a, b in zip(z, y)])) if len(z) == len(y) else None)
    out[tid] = res
json.dump(out, open(os.environ.get("OUT", ".scratch/self_evolving/fill_variants.json"), "w")); print("tasks", len(out))
