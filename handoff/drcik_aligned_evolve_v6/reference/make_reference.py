#!/usr/bin/env python3
"""Freeze the Train-only rolling-CV selection into (a) the host reference used as the per-task baseline of the
evolution score and (b) the numerical seed module the agents start from.  Both are the same pure-numerical blend
(no retrieval / decision step), one weight vector + shrink per dataset x frequency.
usage: make_reference.py --cv seed_cv_timesx.json --cv seed_cv_time_mmd.json --pkg PACKAGE_DIR"""
import argparse, hashlib, json
from pathlib import Path

P = argparse.ArgumentParser(); P.add_argument("--cv", type=Path, action="append", required=True)
P.add_argument("--pkg", type=Path, required=True); A = P.parse_args()

table, evidence, cvinfo = {}, {}, {}
for f in A.cv:
    r = json.load(open(f)); evidence[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    table[r["dataset"]] = {fk: dict(weights=g["weights"], shrink=g["shrink"]) for fk, g in r["groups"].items()}
    cvinfo[r["dataset"]] = dict(members=r["members"], shrink_grid=r["shrink_grid"], weight_step=r["weight_step"], folds=r["folds"],
                                groups={fk: dict(n_tasks=g["n_tasks"], fold_test_sizes=g["cv_folds"]["n_test"],
                                                 cv_mean_joint_error=g["cv_mean_joint_error"], selection_rule=g["selection_rule"])
                                        for fk, g in r["groups"].items()})

BODY = '''

MEMBERS = ("toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt", "seasonal")
TABLE = __TABLE__


def freq_key(freq):
    f = str(freq).lower()
    return "weekly" if ("w" in f or "week" in f) else ("monthly" if "month" in f or f in ("m", "1m", "ms") else "daily")


def seasonal(h, H, p):
    return [h[-p + (i % p)] if p and len(h) >= p else h[-1] for i in range(H)]


def forecast(view):
    mf = view["method_forecasts"]; H = view["H"]; h = view["history"]; toto = mf["toto_2_0"]
    fk = freq_key(view["freq"]); cfg = TABLE[view["dataset"]][fk]
    sea = seasonal(h, H, {"daily": 7, "weekly": 52, "monthly": 12}[fk]); last = h[-1]
    members = [sea if m == "seasonal" else mf.get(m, toto) for m in MEMBERS]
    w = [cfg["weights"][m] for m in MEMBERS]; s = cfg["shrink"]
    return [(1 - s) * sum(wi * mem[i] for wi, mem in zip(w, members)) + s * last for i in range(H)]
'''.replace("__TABLE__", json.dumps(table, indent=1, sort_keys=True))

ref_doc = ('"""Protocol v6 frozen REFERENCE forecaster (host-side baseline of the evolution score).\n'
           "Pure numerical blend selected ONLY on official Train with forward-chaining (rolling-origin) CV, one weight vector\n"
           "and shrink-to-last per dataset x frequency (TABLE).  Missing members fall back to Toto.  Must not change during a\n"
           'run: its SHA-256 is recorded in stage.json and LOCK.json.  forecast(view) -> list of H floats."""')
seed_doc = ('"""Seed Numerical module (protocol v6): the same Train-only rolling-CV-selected blend as the frozen reference\n'
            "(one weight vector over Toto / TimesFM / Moirai / Chronos-Bolt / seasonal-naive plus shrink-to-last per dataset x\n"
            "frequency).  Agents start from this program; the evolution score is measured against the reference, so a\n"
            'submission only gains by beating it.  forecast(view) -> list of H floats."""')
(A.pkg / "reference").mkdir(exist_ok=True)
ref = A.pkg / "reference/reference_forecast.py"; ref.write_text(ref_doc + BODY)
(A.pkg / "seeds/seed_forecast.py").write_text(seed_doc + BODY)
man = dict(schema_version=1, description="Train-only rolling-origin CV reference (= v6 numerical seed)", table=table,
           seed_equals_reference=True, aggregation="fold-macro (equal weight per forward-chained fold), not task-micro",
           sampling="fixed-seed (0) subsample per (frequency, horizon) drawn BEFORE blocking; sampled task ids in the CV evidence files",
           cv=cvinfo,
           cv_evidence_sha256=evidence, reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
json.dump(man, open(A.pkg / "reference/manifest.json", "w"), indent=1)
print(json.dumps(dict(table=table, reference_sha256=man["reference_sha256"]), indent=1))
