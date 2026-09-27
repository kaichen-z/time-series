"""Final method, step 5: gated history repair feeds Toto's input history; the nrd4 team then corrects the
future.  A repair is used only if the history-only validation accepted it (repaired back-test error <
(1 - margin) x raw back-test error).  Reports sMAE / sRMSE / joint vs Toto per split and per seed."""
import argparse, copy, json, sys
sys.path.insert(0, ".scratch/self_evolving")
import nrd4 as N4
from common.metrics import drcik_point_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--repair", required=True, help="output of extract_and_repair.py")
ap.add_argument("--forecasts", required=True, help="output of tl2_toto.py (Toto on repaired histories)")
ap.add_argument("--validation", required=True, help="output of tl3_validate.py")
ap.add_argument("--teams", required=True, help="nrd4 final teams json (artifacts/nrd4_final_teams.json)")
ap.add_argument("--margin", type=float, default=0.1)
ap.add_argument("--parts", default="train,dev,public_test")
a = ap.parse_args()

TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
D = [N4.R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))]
for d in D: d["_sig"] = {}
rep = json.load(open(a.repair)); fc = json.load(open(a.forecasts)); val = json.load(open(a.validation))
R = json.load(open(a.teams))


def m(f, d):
    x = drcik_point_metrics(d["truth"], f, cap=5.0); return x["smae"], x["srmse"]


for seed, v in R["final"].items():
    team = v["team"]; line = []
    for part in a.parts.split(","):
        sb = so = rb = ro = 0.0; w = r = 0; nrep = 0
        for d in [d for d in D if d["part"] == part]:
            b = m(d["fc"]["toto_2_0"], d); vv = val.get(d["tid"])
            ok = d["tid"] in rep.get(part, {}) and vv is not None and vv["repaired"] < vv["raw"] * (1 - a.margin)
            dd = d
            if ok:
                dd = copy.copy(d); dd["fc"] = {"toto_2_0": fc[d["tid"]]}; nrep += 1
            o = m(N4.run_team(team, dd), d)
            sb += b[0]; so += o[0]; rb += b[1]; ro += o[1]; dj = sum(b) - sum(o); w += dj > 1e-9; r += dj < -1e-9
        n = len([x for x in D if x["part"] == part])
        line.append(f"{part}: repaired {nrep}, sMAE {sb / n:.4f} -> {so / n:.4f} ({(sb - so) / sb:+.2%}), "
                    f"sRMSE {rb / n:.4f} -> {ro / n:.4f} ({(rb - ro) / rb:+.2%}), "
                    f"joint {((sb + rb) - (so + ro)) / (sb + rb):+.2%}, improved/worsened {w}/{r}")
    print(f"seed {seed}:\n  " + "\n  ".join(line))
