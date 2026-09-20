"""Data-analog magnitude: does the DATA supply a better correction size than the LLM's guess?

Insight: the 16 HARM tasks are LLM MAGNITUDE errors. The document gives the STRUCTURE (which window,
which direction); the exact size should come from a DATA ANALOG -- the historical level of comparable
periods (task_152: weekend/weekday = 0.626 matched the true -43%). So for each correction window we
compute a data-analog multiplier from history (level of calendar-matching periods vs the overall
level) and compare, on test99, the correction gain using: the LLM multiplier, the data-analog
multiplier, and a blend -- all trust-all through the kernel. If the data-analog beats the LLM guess,
that is the lever toward the +22% ceiling (better perception, not a better gate).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/data_analog.py
"""
from __future__ import annotations
import datetime as dt
import json, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def _parse(ts):
    try:
        return dt.datetime.fromisoformat(str(ts).replace("T", " "))
    except Exception:
        return None


def data_analog_mult(hv, hts, win_fts):
    """Level of history periods whose (weekday, hour) matches the correction window, relative to the
    overall history level -> a grounded multiplier. None if too few analogs (magnitude wall)."""
    sig = set()
    for f in win_fts:
        d = _parse(f)
        if d:
            sig.add((d.weekday(), d.hour))
    if not sig:
        return None
    match = [hv[i] for i, h in enumerate(hts) if (dd := _parse(h)) and (dd.weekday(), dd.hour) in sig]
    if len(match) < 3:
        return None
    overall = statistics.mean(hv) if hv else 0.0
    if abs(overall) < 1e-9:
        return None
    return statistics.mean(match) / overall


def load():
    cards = json.loads((ROOT / ".scratch/cordp_cards_public_test.json").read_text())
    ids = [t for t in SPLIT["public_test"]["task_ids"] if t in cards]
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, tuple(ids))}
    data = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric; truth = list(n.future_values); H = len(truth)
        toto = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(toto) != H:
            continue
        fts = [str(x) for x in t.future_timestamps]; hts = [str(x) for x in t.history_timestamps]
        hv = list(n.history_values)
        wins = []
        for r in (cards[tid].get("corrections") or []):
            if len(r) < 3:
                continue
            on = [i for i, m in enumerate(horizon_window_mask(fts, str(r[0]), str(r[1]))) if m]
            if not on:
                continue
            da = data_analog_mult(hv, hts, [fts[i] for i in on])
            wins.append(dict(idx=(on[0], on[-1] + 1), llm=float(r[2]), data=da))
        if wins:
            data.append(dict(tid=tid, toto=toto, truth=truth, wins=wins))
    return data


def apply(d, pick):
    """pick(win) -> multiplier or None (skip). Applies to base then the kernel."""
    out = list(d["toto"])
    for w in d["wins"]:
        m = pick(w)
        if m is None:
            continue
        s, e = w["idx"]
        for i in range(s, min(e, len(out))):
            out[i] = d["toto"][i] * m
    return list(apply_bounded_delta(d["toto"], out))


def agg(data, pick):
    bs = br = os_ = or_ = 0.0; n = 0; wins = reg = 0
    for d in data:
        out = apply(d, pick)
        mb = drcik_point_metrics(d["truth"], d["toto"], cap=CAP)
        mo = drcik_point_metrics(d["truth"], out, cap=CAP)
        bs += mb["smae"]; br += mb["srmse"]; os_ += mo["smae"]; or_ += mo["srmse"]; n += 1
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
    return (bs - os_) / bs, (br - or_) / br, wins, reg


D = load()
nda = sum(1 for d in D for w in d["wins"] if w["data"] is not None)
ntot = sum(len(d["wins"]) for d in D)
print(f"== data-analog magnitude vs LLM magnitude (test99, {len(D)} tasks, trust-all, kernel) ==")
print(f"windows with a data-analog: {nda}/{ntot} (rest have no calendar analog -> magnitude wall)\n")

policies = {
    "LLM mult (baseline)": lambda w: w["llm"],
    "data-analog (fallback LLM)": lambda w: w["data"] if w["data"] is not None else w["llm"],
    "blend 0.5 (fallback LLM)": lambda w: (0.5 * w["llm"] + 0.5 * w["data"]) if w["data"] is not None else w["llm"],
    "data-analog ONLY (skip if none)": lambda w: w["data"],
}
print(f"{'policy':32s} {'sMAE':>8s} {'sRMSE':>8s} {'wins/reg':>10s}")
print("-" * 62)
for name, pick in policies.items():
    sm, sr, wn, rg = agg(D, pick)
    print(f"{name:32s} {sm:>+7.2%} {sr:>+7.2%} {f'{wn}/{rg}':>10s}")
fs.close()
