"""Build the agent-visible material for a step-2 run: visible-fold traces + task README.
usage: build_workspace.py <run_dir>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E
import tl2

RUN = Path(sys.argv[1]).resolve(); (RUN / "shared/traces").mkdir(parents=True, exist_ok=True)
for sub in ("shared/notes", "shared/skills", "shared/attempts", "queue", "results"): (RUN / sub).mkdir(parents=True, exist_ok=True)
cfg = E.main_config(); fv = E.fill_variants(cfg["retrieval"]["instr"]); team = E.team_of(cfg)
instr = json.load(open(cfg["retrieval"]["instr"]))["instr"]
ivs, _ = tl2.extract(instr, sorted(E.TRAIN_IDS))
VIS = [d for f in (E.FOLDS[0], E.FOLDS[1]) for d in f]
with open(RUN / "shared/traces/visible_tasks.jsonl", "w") as fo:
    for d in VIS:
        out = E.forecast(cfg, d, fv, team); v = fv.get(d["tid"], {})
        fo.write(json.dumps(dict(
            tid=d["tid"], fold=0 if d in E.FOLDS[0] else 1, group=d["group"], freq=d["freq"], H=d["H"], cell=d["cell"],
            history=d["history"], truth=d["truth"], method_forecasts=d["fc"],
            documents=[x[:2000] for x in d["docs"]], doc_confidence=d["conf"], docbase=d["docbase"],
            future_corrections=[dict(start=s, end=e, multiplier=m) for s, e, m in d["corr"]],
            toto_history_backtest_error=d["toto_h"],
            extracted_past_intervals=ivs.get(d["tid"], []),
            repair_variants={k: dict(val_raw=x["val_raw"], val_rep=x["val_rep"], frac=x["frac"]) for k, x in v.items()},
            main_method_forecast=out, main_method_gain=d["base_jt"] - E.jt(out, d), toto_joint_error=d["base_jt"])) + "\n")
print("visible tasks", len(VIS))
