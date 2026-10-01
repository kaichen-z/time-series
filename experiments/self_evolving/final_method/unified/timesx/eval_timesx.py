"""usage: eval_timesx.py <views.json> <truth.json> name=harness.py ...  (scores vs the view's base forecast)"""
import json, sys, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from metrics import M, paper
V, T = sys.argv[1], json.load(open(sys.argv[2])); tmp = Path(tempfile.mkdtemp())
RUN = ".scratch/self_evolving/coevo/mh/harness_runner.py"
def score(ids, F):
    r = dict(smae=0, srmse=0, nmae=[], nmse=[])
    for t in ids:
        m = M(T[t]["truth"], F[t], cap=5.0); r["smae"] += m["smae"] / len(ids); r["srmse"] += m["srmse"] / len(ids)
        h = T[t]["truth"]; a, b = paper(h, F[t], T[t]["naive"]) if "naive" in T[t] else (None, None)
    return r
for spec in sys.argv[3:]:
    name, path = spec.split("=", 1); o = tmp / f"{name}.json"
    subprocess.run([sys.executable, RUN, path, V, str(o)], check=True, timeout=3600)
    R = json.load(open(o)); F = R["forecasts"]
    for part in ("train", "dev", "public_test", "ood_test"):
        ids = [t for t in F if T[t]["part"] == part]
        if not ids: continue
        s = score(ids, F); b = score(ids, {t: T[t]["base"] for t in ids})
        jo = [sum(M(T[t]["truth"], F[t], cap=5.0)[k] for k in ("smae", "srmse")) for t in ids]
        jb = [sum(M(T[t]["truth"], T[t]["base"], cap=5.0)[k] for k in ("smae", "srmse")) for t in ids]
        w = sum(x < y - 1e-9 for x, y in zip(jo, jb)); l = sum(x > y + 1e-9 for x, y in zip(jo, jb))
        print(f"{name:11s} {part:11s} n={len(ids):4d} sMAE {s['smae']:.4f} sRMSE {s['srmse']:.4f} (base {b['smae']:.4f}/{b['srmse']:.4f}) better/worse {w}/{l} err {len(R['errors'])}", flush=True)
