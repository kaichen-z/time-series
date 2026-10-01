"""Metrics for TimesX: Dr-CiK sMAE/sRMSE (common.metrics.drcik_point_metrics, cap 5) and the PostTime paper's
nMAE/nMSE (per-example MAE/MSE divided by the seasonal-naive MAE/MSE, then averaged; the naive forecast repeats the last H=12 history values, which reproduces the paper's TimesFM-2.5 ID numbers most closely)."""
import sys, math
sys.path.insert(0, ".")
from common.metrics import drcik_point_metrics as M

def snaive(h, H, freq):
    p = H
    return [h[-p + (i % p)] for i in range(H)]

def paper(truth, fc, base):
    mae = sum(abs(a - b) for a, b in zip(truth, fc)) / len(truth); mse = sum((a - b) ** 2 for a, b in zip(truth, fc)) / len(truth)
    bm = sum(abs(a - b) for a, b in zip(truth, base)) / len(truth); bs = sum((a - b) ** 2 for a, b in zip(truth, base)) / len(truth)
    return (mae / bm if bm > 1e-12 else None), (mse / bs if bs > 1e-12 else None)

def summary(tasks, F):
    r = dict(smae=[], srmse=[], nmae=[], nmse=[])
    for t in tasks:
        f = F[t["tid"]]; m = M(t["truth"], f, cap=5.0); r["smae"].append(m["smae"]); r["srmse"].append(m["srmse"])
        a, b = paper(t["truth"], f, snaive(t["history"], len(t["truth"]), t["freq"]))
        if a is not None: r["nmae"].append(a)
        if b is not None: r["nmse"].append(b)
    return {k: sum(v) / len(v) for k, v in r.items()}
