"""TimesX views for the correction-function harness (same fields as Dr-CiK / Time-MMD views).
usage (cwd = time-series repo): python build_views.py <variant drcik|outlook> <base toto|avg> <out_views.json> <out_truth.json>
Only tasks with a card are included. Trust profile from TimesX TRAIN only; calibrator = Dr-CiK main team's."""
import json, sys, statistics
X = "work/timesx/"
sys.path.insert(0, "./.scratch/self_evolving")
import nrd4 as N4
VAR, BASE = sys.argv[1], sys.argv[2]
BLEND = {"CommodityPrice": (0.8, 0.3), "Currency": (1.0, 0.3), "SearchTrend": (0.0, 0.0)}
T = json.load(open(X + "tasks.json")); TO = json.load(open(X + "toto.json")); TF = json.load(open(X + "timesfm.json"))
TH = json.load(open(X + "toto_hindcast.json")); CARDS = json.load(open(X + f"cards_{VAR}.json"))
PART = {"train": "train", "dev": "dev", "test_id": "public_test", "test_ood": "ood_test"}
D = []
for t in T:
    c = CARDS.get(t["tid"])
    if c is None or "error" in c: c = {"corrections": [], "confidence": 0.0}
    fts = [x[:10] for x in t["future_ts"]]; corr = []
    for r in c.get("corrections") or []:
        on = [i for i, f in enumerate(fts) if str(r[0])[:10] <= f <= str(r[1])[:10]]
        if on: corr.append([on[0], on[-1] + 1, float(r[2])])
    docs = [t["scenario"], t["holiday"], t["covariates"]]
    d = dict(tid=t["tid"], part=PART[t["part"]], group=f"{t['domain']}|{t['freq']}", freq="daily" if t["freq"].endswith("D") else "weekly",
             H=len(t["truth"]), history=t["history"], truth=t["truth"], fc={"toto_2_0": TO[t["tid"]], "timesfm_2_5": TF[t["tid"]]},
             docs=docs, conf=float(c.get("confidence") or 0.0), corr=corr)
    D.append(N4.R3.prep_task(d, TH))
calib = json.load(open("/tmp/wt_self_evolving/experiments/self_evolving/final_method/artifacts/nrd4_final_teams.json"))["final"]["1"]["team"]["numerical"]["calib"]
trust = N4.trust_profile([d for d in D if d["part"] == "train"])
V = {}
for d in D:
    if BASE == "toto": b = d["fc"]["toto_2_0"]
    elif BASE == "avg": b = [(a + c) / 2 for a, c in zip(d["fc"]["toto_2_0"], d["fc"]["timesfm_2_5"])]
    else:   # blend: per-domain weight/shrink chosen on TimesX train by blend.py
        w, sh = BLEND[d["group"].split("|")[0]]; last = d["history"][-1]
        b = [(1 - sh) * (w * a + (1 - w) * c) + sh * last for a, c in zip(d["fc"]["toto_2_0"], d["fc"]["timesfm_2_5"])]
    V[d["tid"]] = dict(tid=d["tid"], H=d["H"], freq=d["freq"], history=d["history"], base_forecast=b, toto_forecast=d["fc"]["toto_2_0"],
        documents=[x[:3000] for x in d["docs"]], doc_confidence=d["conf"], docbase=d["docbase"], cell=d["cell"],
        cell_toto_backtest_error=trust.get(d["cell"], 1.0), task_toto_backtest_error=d["toto_h"],
        sigma_main_calib=N4.sigma_of(calib, d["history"], d["H"], d["freq"]),
        corrections=[dict(start=s, end=e, multiplier=m) for s, e, m in d["corr"]], group=d["group"], part=d["part"])
json.dump(V, open(sys.argv[3], "w"))
json.dump({d["tid"]: dict(truth=d["truth"], toto=d["fc"]["toto_2_0"], base=V[d["tid"]]["base_forecast"], group=d["group"], part=d["part"]) for d in D}, open(sys.argv[4], "w"))
import collections
print(len(V), "views", collections.Counter(v["part"] for v in V.values()), "with corrections", sum(1 for v in V.values() if v["corrections"]))
