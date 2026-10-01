"""Run dir for cross-dataset Numerical<->Decision co-evolution. usage: prep_x.py <run_dir>"""
import json, sys, random, shutil, subprocess
from pathlib import Path
sys.path.insert(0, "work/timesx"); from metrics import M
HERE = Path(__file__).resolve().parent; RUN = Path(sys.argv[1]).resolve(); DS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("drcik", "tmmd", "timesx")
for s in ("shared/traces", "shared/notes", "shared/skills", "shared/attempts", "shared/numerical", "shared/decision", "private", "queue", "results"): (RUN / s).mkdir(parents=True, exist_ok=True)
def jt(f, y): m = M(y, f, cap=5.0); return m["smae"] + m["srmse"]
ev = {}
for ds in DS:
    V = json.load(open(HERE / f"data/views_{ds}.json")); T = json.load(open(HERE / f"data/truth_{ds}.json"))
    tv = {t: v for t, v in V.items() if v["part"] == "train"}; json.dump(tv, open(RUN / f"shared/views_train_{ds}.json", "w"))
    rng = random.Random(2026); by = {}
    for t in sorted(tv): by.setdefault(f"{tv[t]['domain']}|{tv[t]['freq']}", []).append(t)
    folds = [[], [], []]; i = rng.randrange(3)
    for g in sorted(by):
        ts = by[g][:]; rng.shuffle(ts)
        for t in ts: folds[i % 3].append(t); i += 1
    bjt = {t: jt(tv[t]["method_forecasts"]["toto_2_0"], T[t]["truth"]) for t in tv}
    ev[ds] = dict(truth={t: T[t]["truth"] for t in tv}, base_jt=bjt, scale=sum(bjt.values()) / len(bjt), folds=folds)
json.dump(ev, open(RUN / "private/eval_data.json", "w"))
shutil.copy(HERE / "seed_forecast.py", RUN / "shared/numerical/0000_seed.py"); shutil.copy(HERE / "seed_adjust.py", RUN / "shared/decision/0000_seed.py")
shutil.copy(HERE / "seed_forecast.py", RUN / "shared/best_forecast.py"); shutil.copy(HERE / "seed_adjust.py", RUN / "shared/best_adjust.py")
# traces: visible tasks of every dataset (truth, per-method joint errors, seed base / final gains)
for ds in DS:
    subprocess.run([sys.executable, str(HERE / "runner_x.py"), str(HERE / "seed_forecast.py"), str(HERE / "seed_adjust.py"), str(RUN / f"shared/views_train_{ds}.json"), str(RUN / f"private/_seed_{ds}.json")], check=True)
    o = json.load(open(RUN / f"private/_seed_{ds}.json")); V = json.load(open(RUN / f"shared/views_train_{ds}.json")); E = ev[ds]; vis = set(E["folds"][0] + E["folds"][1])
    with open(RUN / f"shared/traces/visible_{ds}.jsonl", "w") as fo:
        for t in sorted(vis):
            y = E["truth"][t]
            fo.write(json.dumps(dict(tid=t, dataset=ds, fold=0 if t in E["folds"][0] else 1, truth=y, toto_joint_error=round(E["base_jt"][t], 4),
                method_joint_errors={m: round(jt(f, y), 4) for m, f in V[t]["method_forecasts"].items()},
                seed_base_forecast=o["base"][t], seed_final_forecast=o["forecasts"][t],
                seed_base_gain=round(E["base_jt"][t] - jt(o["base"][t], y), 4), seed_final_gain=round(E["base_jt"][t] - jt(o["forecasts"][t], y), 4))) + "\n")
json.dump(dict(phase="numerical"), open(RUN / "shared/phase.json", "w"))
print({ds: [len(f) for f in ev[ds]["folds"]] for ds in DS}, {ds: round(ev[ds]["scale"], 4) for ds in DS})
