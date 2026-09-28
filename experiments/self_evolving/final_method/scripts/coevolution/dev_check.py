"""One-time dev check of final configs (dev never used during search)."""
import json, sys, statistics
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import e2e as E
from common.metrics import drcik_point_metrics as M
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]; dev = set(split[E.os.environ.get("PART", "dev")]["task_ids"])
DV = [E.N4.R3.prep_task(d, E.TH) for d in json.load(open(E.os.environ["CACHE"])) if d["tid"] in dev]
for d in DV: d["_sig"] = {}; h = d["history"]; d["_last"], d["_lo"], d["_hi"] = h[-1], min(h), max(h)
PART = E.os.environ.get("PART", "dev")
FV = json.load(open(".scratch/self_evolving/fill_variants.json"))
for name, path in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    cfg = E.main_config() if path == "main" else json.load(open(path)); team = E.team_of(cfg)
    sb = so = rb = ro = 0; w = r = 0
    for d in DV:
        b = M(d["truth"], d["fc"]["toto_2_0"], cap=5.0); o = M(d["truth"], E.forecast(cfg, d, FV, team), cap=5.0)
        sb += b["smae"]; so += o["smae"]; rb += b["srmse"]; ro += o["srmse"]; dj = b["smae"] + b["srmse"] - o["smae"] - o["srmse"]; w += dj > 1e-9; r += dj < -1e-9
    n = len(DV); print(f"{name:10s} {PART} sMAE {sb/n:.4f}->{so/n:.4f} sRMSE {rb/n:.4f}->{ro/n:.4f} W/R {w}/{r}")
