"""Numerical step on TimesX (no text): Toto/TimesFM weight w and shrink-to-last s, chosen on TRAIN per domain
(grid, joint = sMAE+sRMSE), checked on dev, test reported once."""
import json, itertools
from metrics import summary, M
T = json.load(open("tasks.json")); TO = json.load(open("toto.json")); TF = json.load(open("timesfm.json"))
def fc(t, w, s):
    last = t["history"][-1]; return [(1 - s) * (w * a + (1 - w) * b) + s * last for a, b in zip(TO[t["tid"]], TF[t["tid"]])]
def joint(ts, w, s): return sum(sum(M(t["truth"], fc(t, w, s), cap=5.0)[k] for k in ("smae", "srmse")) for t in ts) / len(ts)
G = list(itertools.product([i / 10 for i in range(11)], [0, .1, .2, .3]))
tr = [t for t in T if t["part"] == "train"]
glob = min(G, key=lambda g: joint(tr, *g)); print("global", glob)
per = {d: min(G, key=lambda g: joint([t for t in tr if t["domain"] == d], *g)) for d in sorted({t["domain"] for t in T})}; print("per-domain", per)
cfgs = {"toto": lambda t: (1, 0), "timesfm": lambda t: (0, 0), "avg": lambda t: (.5, 0), "global": lambda t: glob, "per-domain": lambda t: per[t["domain"]]}
for part in ("dev", "test_id", "test_ood"):
    ts = [t for t in T if t["part"] == part]
    for n, c in cfgs.items():
        s = summary(ts, {t["tid"]: fc(t, *c(t)) for t in ts})
        print(f"{part:8s} {n:10s} sMAE {s['smae']:.4f} sRMSE {s['srmse']:.4f} nMAE {s['nmae']:.3f} nMSE {s['nmse']:.3f}")
