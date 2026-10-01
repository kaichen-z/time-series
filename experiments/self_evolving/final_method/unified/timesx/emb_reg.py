"""Option 2b: text-embedding residual model for TimesX. Base = unified numeric blend (per-domain Toto/TimesFM, as in
build_views.py). Target = log(mean truth / mean base) over the horizon. Ridge on (a) numeric-only features and
(b) numeric + MiniLM embedding of the event text; alpha by 5-fold CV on TRAIN (all 2173 windows); dev = check; test once.
Prediction applied as multiplier exp(pred) on the whole horizon."""
import json, math, sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
sys.path.insert(0, "."); from metrics import M, paper, snaive
T = [t for t in json.load(open("tasks.json"))]; TO = json.load(open("toto.json")); TF = json.load(open("timesfm.json"))
BL = {"CommodityPrice": (0.8, 0.3), "Currency": (1.0, 0.3), "SearchTrend": (0.0, 0.0)}
for t in T:
    w, s = BL[t["domain"]]; last = t["history"][-1]
    t["base"] = [(1 - s) * (w * a + (1 - w) * b) + s * last for a, b in zip(TO[t["tid"]], TF[t["tid"]])]
    bm, ym = np.mean(t["base"]), np.mean(t["truth"])
    t["y"] = float(np.clip(math.log(ym / bm), -0.5, 0.5)) if bm > 0 and ym > 0 else 0.0
    h = np.array(t["history"], float); sc = np.mean(np.abs(h[-12:])) + 1e-9
    t["num"] = [np.mean(h[-4:]) / sc - 1, np.mean(h[-12:]) / (np.mean(h[-48:]) + 1e-9) - 1, np.std(h[-12:]) / sc,
                np.mean(t["base"]) / sc - 1, float(t["domain"] == "CommodityPrice"), float(t["domain"] == "Currency")]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
E = enc.encode([t["scenario"] + " " + t["holiday"] for t in T], batch_size=128, show_progress_bar=False, normalize_embeddings=True)
Xn = np.array([t["num"] for t in T]); Xt = np.hstack([Xn, E]); y = np.array([t["y"] for t in T])
tr = np.array([t["part"] == "train" for t in T])
for name, X in (("numeric-only", Xn), ("numeric+text", Xt)):
    r = RidgeCV(alphas=np.logspace(-1, 4, 12), cv=5).fit(X[tr], y[tr]); pred = r.predict(X)
    print(f"== {name}: alpha {r.alpha_:.2f}, train R2 {r.score(X[tr], y[tr]):.3f}")
    for part in ("train", "dev", "test_id", "test_ood"):
        ids = [i for i, t in enumerate(T) if t["part"] == part]; jb = jo = 0; w = l = 0; na = []; nb = []
        for i in ids:
            t = T[i]; f = [x * math.exp(pred[i]) for x in t["base"]]
            a = sum(M(t["truth"], t["base"], cap=5)[z] for z in ("smae", "srmse")); b = sum(M(t["truth"], f, cap=5)[z] for z in ("smae", "srmse"))
            jb += a / len(ids); jo += b / len(ids); w += b < a - 1e-9; l += b > a + 1e-9
            p1, p2 = paper(t["truth"], f, snaive(t["history"], 12, t["freq"]))
            if p1 is not None: na.append(p1)
            if p2 is not None: nb.append(p2)
        R2 = 1 - np.sum((y[ids] - pred[ids]) ** 2) / np.sum((y[ids] - y[ids].mean()) ** 2)
        print(f"   {part:9s} n={len(ids):4d} joint {jb:.4f} -> {jo:.4f} ({(jb - jo) / jb:+.2%}) better/worse {w}/{l}  R2 {R2:+.3f}  nMAE {np.mean(na):.3f} nMSE {np.mean(nb):.3f}")
