"""One-time dev / (exploratory) test check of harness files: builds dev/test views the same way as prep_mh.py
(main part-1 program, phase_median repair with margin 0.3 using fill_variants.json), runs each harness."""
import json, sys, copy, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coevolution"))
import e2e as E
from common.metrics import drcik_point_metrics as M
N4 = E.N4
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
FV = json.load(open(".scratch/self_evolving/fill_variants.json"))
prog = json.load(open("experiments/self_evolving/final_method/artifacts/numerical_part1_main.json"))["elites"][0]["program"]
cfg = E.main_config(); team = E.team_of(cfg)          # trust profile / calib from Train
ALL = [N4.R3.prep_task(d, E.TH) for d in json.load(open(E.os.environ["CACHE"]))]
for d in ALL: d["_sig"] = {}; h = d["history"]; d["_last"], d["_lo"], d["_hi"] = h[-1], min(h), max(h)
for part in sys.argv[1].split(","):
    ids = set(split[part]["task_ids"]); T = [d for d in ALL if d["tid"] in ids]; views = {}
    for d in T:
        dd = copy.copy(d); dd["fc"] = dict(d["fc"]); v = FV.get(d["tid"], {}).get("phase_median")
        if v and v["val_raw"] is not None and v["val_rep"] < v["val_raw"] * 0.7: dd["fc"]["toto_2_0"] = v["forecast"]
        base = E.P1.run(prog, dd)
        views[d["tid"]] = dict(tid=d["tid"], H=d["H"], freq=d["freq"], history=d["history"], base_forecast=base, toto_forecast=d["fc"]["toto_2_0"],
            documents=[x[:3000] for x in d["docs"]], doc_confidence=d["conf"], docbase=d["docbase"], cell=d["cell"],
            cell_toto_backtest_error=team["numerical"]["trust"].get(d["cell"], 1.0), task_toto_backtest_error=d["toto_h"],
            sigma_main_calib=N4.sigma_of(team["numerical"]["calib"], d["history"], d["H"], d["freq"]),
            corrections=[dict(start=s, end=e, multiplier=m) for s, e, m in d["corr"]])
    vf = Path(tempfile.mkdtemp()) / "v.json"; json.dump(views, open(vf, "w"))
    for h in sys.argv[2:]:
        name, path = h.split("=", 1); of = vf.parent / f"{name}.json"
        subprocess.run([sys.executable, str(Path(__file__).parent / "harness_runner.py"), path, str(vf), str(of)], check=True, timeout=600)
        out = json.load(open(of))["forecasts"]; sb = so = rb = ro = 0; w = r = 0
        for d in T:
            b = M(d["truth"], d["fc"]["toto_2_0"], cap=5.0); o = M(d["truth"], out[d["tid"]], cap=5.0)
            sb += b["smae"]; so += o["smae"]; rb += b["srmse"]; ro += o["srmse"]; dj = b["smae"] + b["srmse"] - o["smae"] - o["srmse"]; w += dj > 1e-9; r += dj < -1e-9
        n = len(T); print(f"{part:11s} {name:12s} sMAE {sb/n:.4f}->{so/n:.4f} sRMSE {rb/n:.4f}->{ro/n:.4f} W/R {w}/{r}", flush=True)
