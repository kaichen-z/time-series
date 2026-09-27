"""Final method, full pipeline evaluation (2026-09-27).
Numerical part 1: the evolved combination program (artifacts/numerical_part1_dictionary.json, best elite)
builds the base forecast from the dictionary methods (Toto + 48 statistical methods, stat_full.py); its Toto term uses the forecast on the repaired
history when the history-only validation accepts the repair (part 2, history repair); the nrd4 team then
corrects future steps (part 2, three-agent correction).  Prints before -> after vs Toto per split / seed."""
import argparse, copy, json, sys
sys.path.insert(0, ".scratch/self_evolving")
import nrd4 as N4
import num_part1 as P1
from common.metrics import drcik_point_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--part1", required=True); ap.add_argument("--repair", required=True)
ap.add_argument("--fill-variants", required=True, help="fill_variants.py output: per task and fill method, the Toto forecast on the repaired history and the history-only validation errors")
ap.add_argument("--fill", default="phase_median")   # learned on Train (select_fill_gate.py)
ap.add_argument("--teams", required=True)
ap.add_argument("--margin", type=float, default=0.3)   # learned on Train (select_fill_gate.py)
ap.add_argument("--parts", default="train,dev,public_test")
a = ap.parse_args()
TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
D = [N4.R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache_full.json"))]
for d in D: d["_sig"] = {}; h = d["history"]; d["_last"], d["_lo"], d["_hi"] = h[-1], min(h), max(h)
prog = json.load(open(a.part1))["elites"][0]["program"]
rep = json.load(open(a.repair)); R = json.load(open(a.teams))
FV = {t: x[a.fill] for t, x in json.load(open(a.fill_variants)).items() if x.get(a.fill)}
fc = {t: v["forecast"] for t, v in FV.items()}
val = {t: dict(raw=v["val_raw"], repaired=v["val_rep"]) for t, v in FV.items() if v["val_raw"] is not None}


def m(f, d): x = drcik_point_metrics(d["truth"], f, cap=5.0); return x["smae"], x["srmse"]


for seed, v in R["final"].items():
    print(f"seed {seed}:")
    for part in a.parts.split(","):
        S = [d for d in D if d["part"] == part]; sb = so = rb = ro = 0.0; w = r = 0
        for d in S:
            b = m(d["fc"]["toto_2_0"], d); dd = copy.copy(d); dd["fc"] = dict(d["fc"]); vv = val.get(d["tid"])
            if d["tid"] in rep.get(part, {}) and vv and vv["repaired"] < vv["raw"] * (1 - a.margin): dd["fc"]["toto_2_0"] = fc[d["tid"]]
            dd["fc"] = {"toto_2_0": P1.run(prog, dd)}
            o = m(N4.run_team(v["team"], dd), d)
            sb += b[0]; so += o[0]; rb += b[1]; ro += o[1]; dj = sum(b) - sum(o); w += dj > 1e-9; r += dj < -1e-9
        n = len(S)
        print(f"  {part}: sMAE {sb/n:.4f} -> {so/n:.4f} ({(sb-so)/sb:+.2%}), sRMSE {rb/n:.4f} -> {ro/n:.4f} ({(rb-ro)/rb:+.2%}), improved/worsened {w}/{r}")
