"""Repair evaluation + Decision gate (max repaired fraction) fitted on Train, applied to Dev."""
import json, sys
from common.metrics import drcik_point_metrics
D = {d["tid"]: d for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
rep = json.load(open(sys.argv[1])); fc = json.load(open(sys.argv[2]))
def jt(f, t): x = drcik_point_metrics(D[t]["truth"], f, cap=5.0); return x["smae"], x["srmse"]
def summ(part, gate):
    sb = so = rb = ro = 0.0; w = r = 0
    tids = [t for t in D if D[t]["part"] == part]
    for t in tids:
        b = jt(D[t]["fc"]["toto_2_0"], t); o = b
        if t in rep[part] and rep[part][t]["frac"] <= gate and t in fc: o = jt(fc[t], t)
        sb += b[0]; so += o[0]; rb += b[1]; ro += o[1]; dj = sum(b) - sum(o); w += dj > 1e-9; r += dj < -1e-9
    return dict(joint=((sb + rb) - (so + ro)) / (sb + rb), smae=(sb - so) / sb, srmse=(rb - ro) / rb, w=w, r=r)
for g in (1.0, 0.5, 0.3, 0.15):
    a, b = summ("train", g), summ("dev", g)
    print(f"  gate frac<={g}: train joint {a['joint']:+.2%} (sMAE {a['smae']:+.2%}, {a['w']}/{a['r']}) | dev joint {b['joint']:+.2%} (sMAE {b['smae']:+.2%}, sRMSE {b['srmse']:+.2%}, {b['w']}/{b['r']})")
