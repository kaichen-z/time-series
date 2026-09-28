"""Unified end-to-end evaluator for the whole pipeline (2026-09-28).

A pipeline config has one entry per role:
  numerical : {"program": part-1 combination program, "calib": nrd4 magnitude calibrator}
  retrieval : {"instr": path of extraction instructions, "validator": nrd4 per-correction weights}
  decision  : {"fill": fill method, "margin": repair margin or None, "strength", "trust_gate"}
Every config is scored the same way: Train only, 3 stratified folds, per fold robust gain vs Toto
(mean + PEN x mean negative gain), fitness = mean over folds - STD_W x std over folds.
Dev / test are never touched here.
"""
import copy, hashlib, json, os, statistics, subprocess, sys
from pathlib import Path
ROOT = Path.cwd(); S = ROOT / ".scratch/self_evolving"          # run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(1, str(S))
os.environ.setdefault("CACHE", str(S / "nrd_cache_full2.json"))
import num_part1 as P1                # program run / mutate over the 43-method dictionary
import nrd4 as N4, nrd_coevolve as C
from common.metrics import drcik_point_metrics

PEN, STD_W = 0.5, 0.25
TH = json.load(open(S / "toto_hindcast.json"))
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
TRAIN_IDS = set(split["train"]["task_ids"])
D = [N4.R3.prep_task(d, TH) for d in json.load(open(os.environ["CACHE"])) if d["tid"] in TRAIN_IDS]
for d in D: d["_sig"] = {}; h = d["history"]; d["_last"], d["_lo"], d["_hi"] = h[-1], min(h), max(h)
C.FOLD_MODE = "strat"; FOLDS = C.gfolds(D, 3)
FV_DIR = S / "coevo/fv"; FV_DIR.mkdir(parents=True, exist_ok=True)
TOTO_PY = os.environ.get("TOTO_PY", sys.executable)   # interpreter of the toto2 environment
_FV = {}


def instr_key(path):
    return hashlib.sha1(json.load(open(path))["instr"].encode()).hexdigest()[:12]


def fill_variants(instr_path):
    """Train-only fill variants (Toto re-forecast on repaired history + history-only validation) for one
    set of extraction instructions; computed once per instruction text and cached."""
    k = instr_key(instr_path)
    if k in _FV: return _FV[k]
    f = FV_DIR / f"fv_{k}.json"
    if not f.exists():
        env = dict(os.environ, INSTR=str(instr_path), OUT=str(f), PARTS="train", CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=str(ROOT), TMPDIR=str(ROOT / ".scratch"))
        subprocess.run([TOTO_PY, str(S / "fill_variants.py")], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _FV[k] = json.load(open(f)); return _FV[k]


def team_of(cfg):
    return {"numerical": {"calib": cfg["numerical"]["calib"], "trust": N4.trust_profile(D)},
            "retrieval": cfg["retrieval"]["validator"],
            "decision": {"strength": cfg["decision"]["strength"], "trust_gate": cfg["decision"]["trust_gate"]}}


def forecast(cfg, d, fv=None, team=None):
    fv = fv if fv is not None else fill_variants(cfg["retrieval"]["instr"]); team = team or team_of(cfg)
    dd = copy.copy(d); dd["fc"] = dict(d["fc"]); dg = cfg["decision"]
    v = fv.get(d["tid"], {}).get(dg["fill"])
    if dg["margin"] is not None and v and v["val_raw"] is not None and v["val_rep"] < v["val_raw"] * (1 - dg["margin"]):
        dd["fc"]["toto_2_0"] = v["forecast"]
    dd["fc"] = {"toto_2_0": P1.run(cfg["numerical"]["program"], dd)}
    return N4.run_team(team, dd)


def jt(f, d): x = drcik_point_metrics(d["truth"], f, cap=5.0); return x["smae"] + x["srmse"]


def evaluate(cfg, folds=None):
    """-> dict(fitness, folds, per_task{tid: gain}, mean_gain, worse, better); fitness over `folds` (default all 3)"""
    folds = FOLDS if folds is None else folds
    fv = fill_variants(cfg["retrieval"]["instr"]); team = team_of(cfg)
    per = {d["tid"]: d["base_jt"] - jt(forecast(cfg, d, fv, team), d) for d in D}
    fl = []
    for fo in folds:
        g = [per[d["tid"]] for d in fo]; fl.append(statistics.mean(g) + PEN * statistics.mean(min(0.0, x) for x in g))
    return dict(fitness=statistics.mean(fl) - STD_W * statistics.pstdev(fl), folds=fl, per_task=per,
                mean_gain=statistics.mean(per.values()), better=sum(x > 1e-9 for x in per.values()),
                worse=sum(x < -1e-9 for x in per.values()))


def main_config():
    """the current main method (GitHub final_method) as a config"""
    A = ROOT / "experiments/self_evolving/final_method/artifacts"
    prog = json.load(open(A / "numerical_part1_dictionary.json"))["elites"][0]["program"]
    team = json.load(open(A / "nrd4_final_teams.json"))["final"]["1"]["team"]
    return {"numerical": {"program": prog, "calib": team["numerical"]["calib"]},
            "retrieval": {"instr": str(S / "tl2_best.json"), "validator": team["retrieval"]},
            "decision": {"fill": "phase_median", "margin": 0.3, "strength": team["decision"]["strength"],
                         "trust_gate": team["decision"]["trust_gate"]}}


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else main_config()
    r = evaluate(cfg)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "per_task"}))
