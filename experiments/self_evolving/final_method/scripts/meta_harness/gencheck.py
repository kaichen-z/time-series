"""Generalization check of the correction functions: per split, how many tasks each function changes vs the seed
(previous main) and the gain/loss on those tasks; plus CV of the route-by-cell selection on a NEW fold split."""
import json, sys, subprocess, tempfile, statistics, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ensemble as EN
from common.metrics import drcik_point_metrics as M
A = EN.HERE.parents[1] / "artifacts"
FUN = dict(EN.FUNCS); FUN["seed"] = str(EN.HERE / "seed_harness.py"); FUN["routed"] = str(A / "meta_harness/round1_routed_correction_function.py")
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
    for n in list(EN.NAMES) + ["routed"]:
        ch = [t for t in ids if max(abs(a - b) for a, b in zip(R[n][t], R["seed"][t])) > 1e-9]
        d = [jt(R["seed"][t], t) - jt(R[n][t], t) for t in ch]
        print(f"  {n:8s} changed {len(ch):3d}  better {sum(x > 1e-9 for x in d):3d}  worse {sum(x < -1e-9 for x in d):3d}  mean delta {statistics.mean(d) if d else 0:+.4f}  total {sum(d):+.3f}")
# CV of route-by-cell on a new random split of Train (3 folds), using the 5 functions' Train outputs
R = out["train"]; tr = sorted(EN.split["train"]["task_ids"]); rng = random.Random(123); sh = tr[:]; rng.shuffle(sh); F = [sh[i::3] for i in range(3)]
cells = {t: EN.ALL[t]["cell"] for t in tr}
for k in range(3):
    fit = [t for j in range(3) if j != k for t in F[j]]; route = {}
    for c in set(cells[t] for t in fit):
        ts = [t for t in fit if cells[t] == c]; route[c] = max(EN.NAMES, key=lambda n: statistics.mean(jt(R["seed"][t], t) - jt(R[n][t], t) for t in ts))
    held = F[k]
    g = lambda n_of: statistics.mean(jt(R["seed"][t], t) - jt(R[n_of(t)][t], t) for t in held)
    print(f"CV fold {k}: held-out mean gain vs seed  routed(learned on other folds) {g(lambda t: route.get(cells[t], 'shared3')):+.4f}  shared3 {g(lambda t: 'shared3'):+.4f}  single {g(lambda t: 'single'):+.4f}")
