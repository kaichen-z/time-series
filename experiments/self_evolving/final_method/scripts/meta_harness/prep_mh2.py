"""Meta-Harness setup for the future-correction stage: per Train task, the inputs a correction function may
use (no labels) + private truth; plus visible-fold traces. Seed function = the current main method's team."""
import json, sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coevolution"))
import e2e as E
N4 = E.N4
RUN = Path(sys.argv[1]).resolve()
for s in ("shared/traces", "shared/notes", "shared/skills", "shared/attempts", "shared/harness", "private", "queue", "results"): (RUN / s).mkdir(parents=True, exist_ok=True)
cfg = E.main_config()
cfg["numerical"]["program"] = json.load(open("experiments/self_evolving/final_method/artifacts/numerical_part1_main.json"))["elites"][0]["program"]
fv = E.fill_variants(cfg["retrieval"]["instr"]); team = E.team_of(cfg); dg = cfg["decision"]
NEWF = json.load(open(sys.argv[2]))["folds"]   # new fold split (from the base-forecast round)
E.FOLDS = [[d for d in E.D if d["tid"] in set(f)] for f in NEWF]
views, truth = {}, {}
vis_ids = {d["tid"] for f in (E.FOLDS[0], E.FOLDS[1]) for d in f}
for d in E.D:
    dd = copy.copy(d); dd["fc"] = dict(d["fc"]); v = fv.get(d["tid"], {}).get(dg["fill"])
    if v and v["val_raw"] is not None and v["val_rep"] < v["val_raw"] * (1 - dg["margin"]): dd["fc"]["toto_2_0"] = v["forecast"]
    base = E.P1.run(cfg["numerical"]["program"], dd)
    sig = N4.sigma_of(team["numerical"]["calib"], d["history"], d["H"], d["freq"])
    views[d["tid"]] = dict(tid=d["tid"], H=d["H"], freq=d["freq"], history=d["history"], base_forecast=base,
        toto_forecast=d["fc"]["toto_2_0"], documents=[x[:3000] for x in d["docs"]], doc_confidence=d["conf"], docbase=d["docbase"],
        cell=d["cell"], cell_toto_backtest_error=team["numerical"]["trust"].get(d["cell"], 1.0), task_toto_backtest_error=d["toto_h"],
        sigma_main_calib=sig, corrections=[dict(start=s, end=e, multiplier=m) for s, e, m in d["corr"]])
    truth[d["tid"]] = d["truth"]
json.dump(views, open(RUN / "shared/views_train.json", "w"))
json.dump(dict(truth=truth, base_jt={d["tid"]: d["base_jt"] for d in E.D}, folds=[[d["tid"] for d in f] for f in E.FOLDS]), open(RUN / "private/eval_data.json", "w"))
# visible traces: truth, main-method forecast, solo effect of every correction
with open(RUN / "shared/traces/visible_tasks.jsonl", "w") as fo:
    for d in E.D:
        if d["tid"] not in vis_ids: continue
        vw = views[d["tid"]]; mainf = E.forecast(cfg, d, fv, team); base = vw["base_forecast"]
        solo = []
        for s, e, m in d["corr"]:
            out = list(E.N4.apply_bounded_delta(base, E.N4.R3.apply_ev(base, [(s, e, m)], 1.0)))
            solo.append(dict(start=s, end=e, multiplier=m, gain_if_applied_alone=E.jt(base, d) - E.jt(out, d)))
        fo.write(json.dumps(dict(vw, truth=d["truth"], main_method_forecast=mainf, main_method_gain=d["base_jt"] - E.jt(mainf, d),
                                 base_only_gain=d["base_jt"] - E.jt(base, d), corrections_solo=solo)) + "\n")
print("views", len(views), "visible", len(vis_ids))
