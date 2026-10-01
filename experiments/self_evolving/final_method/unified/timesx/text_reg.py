"""Learned text correction for TimesX (option 2): LLM card -> signed horizon-level signal; per-domain slope learned on
TRAIN by least squares on log(mean truth / mean base); applied as a multiplier exp(k * signal) to the whole horizon.
usage: text_reg.py <views.json> <truth.json>"""
import json, sys, math, statistics
import numpy as np
sys.path.insert(0, "."); from metrics import M
V = json.load(open(sys.argv[1])); T = json.load(open(sys.argv[2]))
def sig(v, mode):
    H = v["H"]; s = 0.0
    for c in v["corrections"]:
        m = max(0.5, min(2.0, c["multiplier"])); cov = (min(H, c["end"]) - max(0, c["start"])) / H
        s += math.log(m) * cov
    return s * (v["doc_confidence"] if mode == "conf" else 1.0)
def target(k): b = np.mean(V[k]["base_forecast"]); y = np.mean(T[k]["truth"]); return math.log(y / b) if b > 0 and y > 0 else 0.0
dom = lambda k: V[k]["group"].split("|")[0]
for mode in ("plain", "conf"):
    tr = [k for k in V if V[k]["part"] == "train"]; K = {}
    for d in sorted({dom(k) for k in V}):
        ks = [k for k in tr if dom(k) == d]; x = np.array([sig(V[k], mode) for k in ks]); y = np.array([target(k) for k in ks])
        k_ = float(x @ y / (x @ x)) if (x @ x) > 0 else 0.0; K[d] = max(0.0, min(1.0, k_))
        corr = np.corrcoef(x[x != 0], y[x != 0])[0, 1] if (x != 0).sum() > 3 else float("nan")
        print(f"{mode:5s} {d:14s} n={len(ks)} nonzero={(x != 0).sum()} corr={corr:+.3f} slope={k_:+.3f} -> k={K[d]:.2f}")
    for part in ("train", "dev", "public_test", "ood_test"):
        ks = [k for k in V if V[k]["part"] == part]
        if not ks: continue
        jb = jo = 0; w = l = 0
        for k in ks:
            f = [x * math.exp(K[dom(k)] * sig(V[k], mode)) for x in V[k]["base_forecast"]]
            a = sum(M(T[k]["truth"], V[k]["base_forecast"], cap=5)[z] for z in ("smae", "srmse")); b = sum(M(T[k]["truth"], f, cap=5)[z] for z in ("smae", "srmse"))
            jb += a / len(ks); jo += b / len(ks); w += b < a - 1e-9; l += b > a + 1e-9
        print(f"   {mode} {part:11s} n={len(ks)} joint base {jb:.4f} -> {jo:.4f} ({(jb - jo) / jb:+.2%}) better/worse {w}/{l}")
