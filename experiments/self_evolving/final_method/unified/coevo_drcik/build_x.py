"""Views for cross-dataset co-evolution: one view schema for Dr-CiK, Time-MMD and TimesX (no labels in views).
Writes coevo_x/data/views_<ds>.json (all parts) and coevo_x/data/truth_<ds>.json."""
import json, math
from pathlib import Path
U = Path(__file__).resolve().parents[1]; S = ".scratch/self_evolving/"; X = "work/timesx/"
OUT = U / "coevo_x/data"; OUT.mkdir(exist_ok=True)
BT_T = json.load(open(U / "bt/toto.json")); BT_F = json.load(open(U / "bt/tfm.json"))
KEEP = ("H", "freq", "history", "cell", "documents", "doc_confidence", "docbase", "cell_toto_backtest_error", "task_toto_backtest_error", "sigma_main_calib", "corrections")
def ok(v): return v is not None and len(v) > 0 and all(x is not None and math.isfinite(x) for x in v)
def finish(ds, rows):
    V, T = {}, {}
    train_vars = {r["var"] for r in rows if r["part"] == "train"}
    for r in rows:
        v = r["view"]; H = len(r["truth"]); h = [float(x) for x in r["history"]]
        if not (ok(r["truth"]) and ok(h) and ok(r["mf"].get("toto_2_0"))): continue
        mf = {k: [float(x) for x in f] for k, f in r["mf"].items() if ok(f) and len(f) == H}
        bts = []
        for k in (1, 2, 3):
            key = f"{ds}|{r['tid']}|{k}"
            if key in BT_T:
                cut = len(h) - k * H
                bts.append(dict(cut=cut, toto_2_0=BT_T[key], timesfm_2_5=BT_F.get(key, BT_T[key]), target=h[cut:cut + H]))
        out = {k: v[k] for k in KEEP if k in v}
        out.update(tid=r["tid"], dataset=ds, part=r["part"], domain=r["domain"], variable=r["var"], seen_variable=r["var"] in train_vars,
                   H=H, freq=r["freq"], history=h, method_forecasts=mf, toto_forecast_repaired_history=r.get("rep"), backtests=bts)
        V[r["tid"]] = out; T[r["tid"]] = dict(truth=[float(x) for x in r["truth"]], part=r["part"])
    json.dump(V, open(OUT / f"views_{ds}.json", "w")); json.dump(T, open(OUT / f"truth_{ds}.json", "w"))
    import collections; print(ds, len(V), collections.Counter(v["part"] for v in V.values()))
# Dr-CiK
MHR = S + "coevo/mhr/runs/shared3/"
V = {**json.load(open(MHR + "shared/views_train.json")), **json.load(open("/tmp/mhr_dev.json")), **json.load(open("/tmp/mhr_test.json"))}
TR = {**json.load(open(MHR + "private/truth.json")), **json.load(open("/tmp/mhr_dev_truth.json")), **json.load(open("/tmp/mhr_test_truth.json"))}
FV = json.load(open(S + "fill_variants.json")); TF = json.load(open(S + "timesfm_full.json")); C = {d["tid"]: d for d in json.load(open(S + "nrd_cache.json"))}
rows = []
for t, v in V.items():
    d = C[t]; fv = FV.get(t, {}).get("phase_median"); mf = dict(v["method_forecasts"]); mf.setdefault("timesfm_2_5", TF.get(t))
    rep = fv["forecast"] if (fv and fv["val_raw"] is not None and fv["val_rep"] < fv["val_raw"] * 0.7) else None
    rows.append(dict(tid=t, part={"public_test": "test"}.get(d["part"], d["part"]), domain=d["group"].split("|")[0], freq=d["group"].split("|")[1], var=d["group"].split("|")[0],
                     history=v["history"], truth=TR[t]["truth"], mf=mf, rep=rep, view=v))
finish("drcik", rows)
# Time-MMD
V = json.load(open(S + "coevo/tmmd/views_all.json")); TR = json.load(open(S + "coevo/tmmd/truth_all.json")); TF = json.load(open(U / "tmmd_timesfm.json"))
MO = json.load(open(U / "tsfm/moirai_tmmd.json")); CH = json.load(open(U / "tsfm/chronos_tmmd.json")); rows = []
for t, v in V.items():
    dom, fq = v["group"].split("|")
    rows.append(dict(tid=t, part={"public_test": "test"}.get(v["part"], v["part"]), domain=dom, freq=fq, var=dom, history=[x for x in v["history"] if x is not None and math.isfinite(x)],
                     truth=TR[t]["truth"], mf=dict(toto_2_0=v["toto_forecast"], timesfm_2_5=TF.get(t), moirai_2_0=MO.get(t, {}).get("moirai_2_0"), chronos_bolt=CH.get(t)), view=v))
finish("tmmd", rows)
# TimesX
V = json.load(open("/tmp/timesx/views_all.json")); TR = json.load(open("/tmp/timesx/truth_all.json")); TF = json.load(open(X + "timesfm.json"))
MO = json.load(open(U / "tsfm/moirai_timesx.json")); CH = json.load(open(U / "tsfm/chronos_timesx.json")); rows = []
for t, v in V.items():
    rows.append(dict(tid=t, part={"public_test": "test", "ood_test": "test_ood"}.get(v["part"], v["part"]), domain=v["group"].split("|")[0], freq=v["freq"], var=t.rsplit("_", 1)[0],
                     history=v["history"], truth=TR[t]["truth"], mf=dict(toto_2_0=v["toto_forecast"], timesfm_2_5=TF.get(t), moirai_2_0=MO.get(t, {}).get("moirai_2_0"), chronos_bolt=CH.get(t)), view=v))
finish("timesx", rows)
