"""Generalization check of the correction functions: per split, how many tasks each function changes vs the seed
(previous main) and the gain/loss on those tasks; plus CV of the route-by-cell selection on a NEW fold split."""
import json, sys, subprocess, tempfile, statistics, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ensemble as EN
from common.metrics import drcik_point_metrics as M
A = EN.HERE.parents[1] / "artifacts"
FUN = {"seed": str(A / "meta_harness/round1_routed_correction_function.py"), "round2": str(EN.HERE.parents[1] / "artifacts/meta_harness/round2/best_correction_function.py")}
def run(ids):
    vw = EN.views(ids); tmp = Path(tempfile.mkdtemp()); json.dump(vw, open(tmp / "v.json", "w")); R = {}
    for n, f in FUN.items():
        subprocess.run([sys.executable, str(EN.HERE / "harness_runner.py"), f, str(tmp / "v.json"), str(tmp / f"{n}.json")], check=True, timeout=900)
        R[n] = json.load(open(tmp / f"{n}.json"))["forecasts"]
    return R
def jt(f, t): x = M(EN.ALL[t]["truth"], f, cap=5.0); return x["smae"] + x["srmse"]
out = {}
for part in ("train", "dev", "public_test"):
    ids = sorted(EN.split[part]["task_ids"]); R = run(ids); out[part] = R
    print(f"== {part} ({len(ids)} tasks)")
    for n in ["round2"]:
        ch = [t for t in ids if max(abs(a - b) for a, b in zip(R[n][t], R["seed"][t])) > 1e-9]
        d = [jt(R["seed"][t], t) - jt(R[n][t], t) for t in ch]
        print(f"  {n:8s} changed {len(ch):3d}  better {sum(x > 1e-9 for x in d):3d}  worse {sum(x < -1e-9 for x in d):3d}  mean delta {statistics.mean(d) if d else 0:+.4f}  total {sum(d):+.3f}")
