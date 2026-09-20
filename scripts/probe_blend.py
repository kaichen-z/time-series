"""Probe: give the decision agent MORE numerical info -- expose seasonal_naive as a 2nd
candidate and let MethodBlend combine it with Toto, context-conditioned (blend only when a
document signal is present). Q1: is seasonal_naive competitive with Toto? Q2: does blending
help on test (overall, and on document-signal tasks)?
Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/probe_blend.py"""
from __future__ import annotations
import json, statistics
from pathlib import Path
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.controller import Controller, SelectBase, MethodBlend, CorDPAdjust, run_controller

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
cards = json.loads((ROOT / ".scratch/cordp_cards_public_test.json").read_text())
ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
            ["partitions"]["public_test"]["task_ids"])
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(
    "external/Dr-CiK/full-download/Dr-CiK_public/tasks", ids)}
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def j(truth, fc):
    m = drcik_point_metrics(truth, list(fc), cap=CAP); return (m["smae"] + m["srmse"]) / 2.0


DATA = []
for tid in ids:
    t = tasks.get(tid)
    if t is None or tid not in cards:
        continue
    n = t.numeric; truth = list(n.future_values)
    try:
        toto = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        snv = tuple(fs.forecast("seasonal_naive", tuple(n.history_values), n.prediction_length, n.frequency))
    except Exception:
        continue
    if len(toto) != len(truth) or len(snv) != len(truth):
        continue
    c = cards.get(tid) or {}
    DATA.append(dict(tid=tid, truth=truth, toto=toto, snv=snv, conf=float(c.get("confidence") or 0.0),
                     corrs=tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3),
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     fts=[str(x) for x in t.future_timestamps]))

# Q1: seasonal_naive vs Toto (whole horizon joint, lower=better)
tj = statistics.mean(j(d["truth"], d["toto"]) for d in DATA)
sj = statistics.mean(j(d["truth"], d["snv"]) for d in DATA)
print(f"Q1: mean joint  Toto {tj:.4f}  vs  seasonal_naive {sj:.4f}  -> "
      f"{'Toto stronger' if tj < sj else 'seasonal stronger'} ({len(DATA)} tasks)")

# Q2: context-conditioned MethodBlend (blend toward seasonal only when doc signal present)
print("\nQ2: MethodBlend(other=seasonal_naive) context-conditioned, whole-horizon joint delta vs Toto:")
for w in [0.2, 0.3, 0.5]:
    for cm in [0.7, 0.8]:
        ctrl = Controller(steps=(SelectBase(), MethodBlend(other="seasonal_naive", weight=w, conf_min=cm)))
        deltas = []; touched = 0
        for d in DATA:
            out, _ = run_controller(ctrl, {"toto_2_0": d["toto"], "seasonal_naive": d["snv"]},
                                    (), d["hv"], d["hts"], d["fts"], cordp_conf=d["conf"])
            if any(abs(a - b) > 1e-9 for a, b in zip(d["toto"], out)):
                touched += 1
            deltas.append(j(d["truth"], d["toto"]) - j(d["truth"], out))
        wins = sum(1 for x in deltas if x > 1e-9); reg = sum(1 for x in deltas if x < -1e-9)
        print(f"  w={w} conf>={cm}: touched {touched} | mean joint gain {statistics.mean(deltas):+.4f} "
              f"| wins {wins} reg {reg}")
fs.close()
