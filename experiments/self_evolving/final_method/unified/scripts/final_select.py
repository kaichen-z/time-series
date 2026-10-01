"""Final unified selection rule (2026-10-01): if the dataset's dev split has >= 100 tasks, choose on dev among the
framework variants (v0 base, v1 horizon split, v2 horizon split + Chronos-2) and single foundation models (Toto,
TimesFM, Chronos-2) by mean joint error; otherwise keep the dataset's own evolved version. Test reported once."""
import json, sys
import numpy as np
sys.path.insert(0, "work/timesx"); from metrics import M, paper, snaive
def sc(rows, key):
    a = np.mean([M(r["truth"], r[key], cap=5)["smae"] for r in rows]); b = np.mean([M(r["truth"], r[key], cap=5)["srmse"] for r in rows]); return a, b
res = {}
for ds in ("tmmd", "timesx"):
    V = {v: json.load(open(f"final/{ds}_{v}.json")) for v in ("v0", "v1", "v2")}
    base = V["v2"]; rows = {}
    for t, r in base.items():
        rows[t] = dict(part=r["part"], truth=r["truth"], history=r["history"], freq=r["freq"], toto=r["toto"], timesfm=r["timesfm"], chronos2=r["chronos2"],
                       **{f"framework_{v}": V[v][t]["framework"] for v in V})
    dev = [r for r in rows.values() if r["part"] == "dev"]
    cands = ["framework_v0", "framework_v1", "framework_v2", "toto", "timesfm", "chronos2"]
    devsc = {c: sc(dev, c) for c in cands}; pick = min(cands, key=lambda c: sum(devsc[c]))
    print(f"== {ds}: dev n={len(dev)}  " + "  ".join(f"{c} {a:.4f}/{b:.4f}" for c, (a, b) in devsc.items()) + f"  -> PICK {pick}")
    for part in ("test", "test_ood"):
        ts = [r for r in rows.values() if r["part"] == part]
        if not ts: continue
        for c in (pick, "toto", "timesfm"):
            a, b = sc(ts, c); ex = ""
            if ds == "timesx":
                pa = [paper(r["truth"], r[c], snaive(r["history"], len(r["truth"]), r["freq"])) for r in ts]
                ex = f" nMAE {np.mean([x for x, y in pa if x is not None]):.3f} nMSE {np.mean([y for x, y in pa if y is not None]):.3f}"
            print(f"   {part:9s} {c:14s} n={len(ts):4d} sMAE {a:.4f} sRMSE {b:.4f}{ex}")
