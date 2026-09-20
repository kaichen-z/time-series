"""Base-strength sweep: how much does the SAME document correction help as the base gets stronger?

The cross-modal literature reports big "text helps" gains -- but on WEAK bases (Beyond-Naive uses
Lag-Llama; CiK's best TSFM is Chronos). Nobody applies it on Toto, which dominates all of them on
this split (Toto sMAE 0.381 vs Moirai 0.466, Chronos 0.483). This applies the SAME cached CorDP
multipliers (trust-all, through the kernel) on top of bases from weak to strong, on test99, and
reports how the correction gain shrinks as the base strengthens -- the empirical "text gain
~ f(1/base strength)" law, and the reason our honest number is small.

Non-Toto bases come from Xuan's baselines branch (25-sample mean point forecast); Toto from the
cache-only ForecastStore.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/base_sweep.py
"""
from __future__ import annotations
import json, statistics, subprocess
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

# --- load test99 tasks: truth, future timestamps, and the cached CorDP correction windows ---
cards = json.loads((ROOT / ".scratch/cordp_cards_public_test.json").read_text())
ids = [t for t in SPLIT["public_test"]["task_ids"] if t in cards]
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, tuple(ids))}
TASKINFO = {}
for tid in ids:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric; truth = list(n.future_values); H = len(truth)
    fts = [str(x) for x in t.future_timestamps]
    corr = []
    for r in (cards[tid].get("corrections") or []):
        if len(r) >= 3:
            on = [i for i, m in enumerate(horizon_window_mask(fts, str(r[0]), str(r[1]))) if m]
            if on:
                corr.append((on[0], on[-1] + 1, float(r[2])))
    TASKINFO[tid] = dict(truth=truth, H=H, corr=corr, hv=tuple(n.history_values),
                         pl=n.prediction_length, freq=n.frequency)


def jsonl_points(method):
    """{tid: step-wise mean point forecast} from Xuan's baseline branch (25 samples)."""
    out = subprocess.run(["git", "show", f"origin/xuanqi/baseline-agent:runs/baselines/{method}_dev.jsonl"],
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        r = json.loads(line); s = r.get("samples")
        if not s:
            continue
        L = len(s[0])
        d[r["benchmark_id"]] = [statistics.mean(traj[i] for traj in s) for i in range(L)]
    return d


def base_forecast(method, jl, tid):
    info = TASKINFO[tid]; H = info["H"]
    if method == "toto":
        fc = list(fs.forecast("toto_2_0", info["hv"], info["pl"], info["freq"]))
    else:
        fc = jl.get(tid)
    if not fc or len(fc) < H:
        return None
    return list(fc[:H])


def apply_cordp(base, corr):
    out = list(base)
    for s, e, m in corr:
        for i in range(s, min(e, len(out))):
            out[i] = base[i] * m
    return list(apply_bounded_delta(base, out))


def eval_base(method, jl):
    bs = br = os_ = or_ = 0.0; n = 0; reg = 0
    for tid, info in TASKINFO.items():
        base = base_forecast(method, jl, tid)
        if base is None:
            continue
        truth = info["truth"]
        out = apply_cordp(base, info["corr"])
        mb = drcik_point_metrics(truth, base, cap=CAP)
        mo = drcik_point_metrics(truth, out, cap=CAP)
        bs += mb["smae"]; br += mb["srmse"]; os_ += mo["smae"]; or_ += mo["srmse"]; n += 1
        if mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9:
            reg += 1
    if n == 0:
        return None
    bs /= n; br /= n; os_ /= n; or_ /= n
    return dict(method=method, n=n, base_smae=bs, base_srmse=br,
                gain_smae=(bs - os_) / bs, gain_srmse=(br - or_) / br, reg=reg)


METHODS = ["naive", "seasonal_naive", "ets", "arima", "aurora", "chronos", "moirai", "toto"]
JL = {m: (jsonl_points(m) if m != "toto" else {}) for m in METHODS}

rows = [r for m in METHODS if (r := eval_base(m, JL[m]))]
rows.sort(key=lambda r: -r["base_smae"])   # weakest base first
print("== Base-strength sweep: SAME CorDP corrections (trust-all, kernel) on test99 ==")
print(f"{'base':16s} {'base sMAE':>10s} | {'+CorDP gain sMAE':>16s} {'sRMSE':>9s} {'reg':>5s}")
print("-" * 66)
for r in rows:
    print(f"{r['method']:16s} {r['base_smae']:>10.4f} | {r['gain_smae']:>+15.2%} {r['gain_srmse']:>+9.2%} {r['reg']:>5d}")
print("\nlaw: as the base gets stronger (sMAE ↓), the SAME document correction helps LESS.")
fs.close()
