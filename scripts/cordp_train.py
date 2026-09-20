"""CorDP Route-A on the 80-TRAIN set (point metric = windowed MAE), using cached Toto p50
from the champion ForecastStore (no Mac needed for this pass). Same correction mechanism as
cordp_probe.py: show the LLM the base forecast numbers + documents, ask for per-window
multipliers anchored on the base scale. Reports per-task windowed relative MAE improvement +
gate analysis (confidence, window-fraction), so we can see on a BIGGER, representative set
whether CorDP + a learnable gate is net-positive. CRPS upgrade comes once Mac q9 lands.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/cordp_train.py
"""
from __future__ import annotations
import json, os, shutil, statistics
from pathlib import Path

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse, horizon_window_mask
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore

ROOT = Path(".").resolve()
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
MULT_LO = float(os.environ.get("MULT_LO", "0.6"))
MULT_HI = float(os.environ.get("MULT_HI", "1.6"))
SHRINK = float(os.environ.get("SHRINK", "1.0"))

PART = os.environ.get("PARTITION", "train")
train_ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
                  ["partitions"][PART]["task_ids"])
NT = len(train_ids)
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, train_ids)}

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)

llm = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                      timeout_seconds=900, cache_dir=str(ROOT / ".scratch/cordp-cache")))

SYSTEM = (
    "You are a forecasting-correction module. A strong statistical model has ALREADY produced "
    "a base forecast (median values with timestamps). Your ONLY job: decide whether the "
    "documented real-world events imply the TRUE values in specific sub-windows will differ "
    "from this base, and if so by what MULTIPLICATIVE factor.\n"
    "Rules:\n"
    "1. Anchor to the shown base scale. A correction multiplier is typically within [0.3, 3.0]; "
    "never produce values orders of magnitude off the base.\n"
    "2. Correct ONLY sub-windows where a document explicitly implies a change (holiday closure, "
    "outage, strike, promotion, weather event...). Give the timestamps and a one-line rationale "
    "quoting the document. Do NOT rescale the whole horizon on vague 'stabilized/steady' wording "
    "-- the base already has the level.\n"
    "3. If the documents do NOT clearly imply any deviation from the base, return "
    "\"relevant\": false with an empty corrections list. Do not guess.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, '
    '"end_timestamp": iso, "multiplier": number, "rationale": str}], "confidence": number}.')


def wd(ts):
    p = _parse(ts); return p.weekday() if p else -1


def build_user(task, base):
    n = task.numeric
    hv, hts = list(n.history_values), [str(x) for x in task.history_timestamps]
    fts = [str(x) for x in task.future_timestamps]
    we = [x for x, t in zip(hv, hts) if wd(t) >= 5]
    wk = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
    hist = {"typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
            "overall_mean": round(statistics.mean(hv), 4),
            "weekday_mean": round(statistics.mean(wk), 4) if wk else None,
            "weekend_mean": round(statistics.mean(we), 4) if we else None,
            "last_values": [round(x, 4) for x in hv[-12:]]}
    docs = [{"document_id": d.document_id, "content": d.content[:3500]} for d in task.documents]
    return json.dumps({"target_name": task.target_name,
                       "target_description": task.target_description, "frequency": n.frequency,
                       "history_summary": hist, "documents": docs,
                       "base_forecast_median": [[fts[i], round(base[i], 4)] for i in range(len(fts))]},
                      ensure_ascii=False)


def mae(a, b, idx):
    return statistics.mean(abs(a[i] - b[i]) for i in idx) if idx else 0.0


rows = []
cards = {}          # per-task CorDP perception for the evolvable CorDPAdjust primitive
for k, tid in enumerate(train_ids, 1):
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    truth = list(n.future_values)
    try:
        base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    except Exception as exc:
        print(f"[{k}/{NT}] {tid}: FS miss {exc}"); continue
    if len(base) != len(truth):
        print(f"[{k}/{NT}] {tid}: len mismatch"); continue
    fts = [str(x) for x in t.future_timestamps]
    resp = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": build_user(t, base)}],
                        temperature=0.0)
    try:
        out = parse_json_object(resp.text)
    except JsonExtractionError:
        print(f"[{k}/{NT}] {tid}: parse fail"); continue
    try:
        conf = float(out.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    raw_corrs = []
    for c in (out.get("corrections") or []):
        try:
            raw_corrs.append([c.get("start_timestamp"), c.get("end_timestamp"),
                              float(c.get("multiplier"))])
        except (TypeError, ValueError):
            continue
    cards[tid] = {"confidence": conf, "corrections": raw_corrs}
    Path(f".scratch/cordp_cards_{PART}.json").write_text(json.dumps(cards, default=str))
    corrected = list(base)
    mask = [False] * len(truth)
    applied = []
    for c in (out.get("corrections") or []):
        try:
            m = float(c.get("multiplier"))
        except (TypeError, ValueError):
            continue
        m = 1.0 + SHRINK * (m - 1.0)
        m = max(MULT_LO, min(MULT_HI, m))
        wm = horizon_window_mask(fts, str(c.get("start_timestamp")), str(c.get("end_timestamp")))
        if not any(wm):
            continue
        for i in range(len(truth)):
            if wm[i]:
                mask[i] = True; corrected[i] = base[i] * m
        applied.append((m, sum(wm)))
    idx = [i for i in range(len(truth)) if mask[i]]
    scale = statistics.median([abs(x) for x in truth]) or 1.0
    bw, cw = mae(base, truth, idx), mae(corrected, truth, idx)
    if idx:
        b_, t_ = statistics.mean([base[i] for i in idx]), statistics.mean([truth[i] for i in idx])
        oracle = (t_ / b_) if b_ else float("nan")
    else:
        oracle = float("nan")
    rows.append(dict(tid=tid, conf=conf, n_win=len(idx), H=len(truth),
                     base_mae=bw, corr_mae=cw, scale=scale, oracle=oracle,
                     mults=[a[0] for a in applied]))
    Path(f".scratch/cordp_{PART}.json").write_text(json.dumps(rows, default=str))
    tag = ("BETTER" if cw < bw - 1e-9 else "worse" if cw > bw + 1e-9 else "noop") if idx else "abstain"
    print(f"[{k}/{NT}] {tid}: conf={conf:.2f} win={len(idx)}/{len(truth)} "
          f"mults={[round(a[0],2) for a in applied]} oracle={oracle:.2f} "
          f"MAE {bw:.3f}->{cw:.3f} {tag}", flush=True)

fs.close()

# aggregate + gate sweep
app = [r for r in rows if r["n_win"] > 0]
print(f"\n== CorDP {PART} (windowed MAE) : {len(app)} applied / {len(rows)} tasks ==")


def agg(label, cond):
    keep = [r for r in app if cond(r)]
    if not keep:
        print(f"  {label}: 0 applied"); return
    rels = [(r["base_mae"] - r["corr_mae"]) / r["base_mae"] for r in keep if r["base_mae"] > 1e-9]
    # scale-normalized absolute improvement (fair cross-task sum)
    dnorm = sum((r["base_mae"] - r["corr_mae"]) / r["scale"] for r in keep)
    wins = sum(1 for r in keep if r["corr_mae"] < r["base_mae"] - 1e-9)
    reg = sum(1 for r in keep if r["corr_mae"] > r["base_mae"] + 1e-9)
    print(f"  {label}: {len(keep):2d} applied | mean rel {statistics.mean(rels):+.1%} | "
          f"norm abs gain {dnorm:+.3f} | wins {wins} reg {reg}")


agg("no gate            ", lambda r: True)
agg("conf>=0.70         ", lambda r: r["conf"] >= 0.70)
agg("wfrac<=0.30        ", lambda r: r["n_win"] / r["H"] <= 0.30)
agg("wfrac<=0.30&conf>=.7", lambda r: r["n_win"] / r["H"] <= 0.30 and r["conf"] >= 0.70)
agg("wfrac<=0.20&conf>=.8", lambda r: r["n_win"] / r["H"] <= 0.20 and r["conf"] >= 0.80)

# FROZEN GATE (picked on 80-train): the headline generalization number on a held-out split
print("\n-- FROZEN GATE from train: wfrac<=0.20 & conf>=0.80 (applied, not tuned here) --")
agg("FROZEN             ", lambda r: r["n_win"] / r["H"] <= 0.20 and r["conf"] >= 0.80)
