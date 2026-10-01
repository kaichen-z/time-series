import json, sys
from metrics import summary, snaive
T = json.load(open("tasks.json")); TO = json.load(open("toto.json")); TF = json.load(open("timesfm.json"))
F = {"toto": TO, "timesfm": TF, "snaive": {t["tid"]: snaive(t["history"], 12, t["freq"]) for t in T},
     "last": {t["tid"]: [t["history"][-1]] * 12 for t in T},
     "avg(toto,tfm)": {k: [(a + b) / 2 for a, b in zip(TO[k], TF[k])] for k in TO}}
for part in ("train", "dev", "test_id", "test_ood"):
    ts = [t for t in T if t["part"] == part]
    for n, f in F.items():
        s = summary(ts, f); print(f"{part:8s} {n:14s} sMAE {s['smae']:.4f} sRMSE {s['srmse']:.4f} nMAE {s['nmae']:.3f} nMSE {s['nmse']:.3f}")
