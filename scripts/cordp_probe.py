"""Route A minimal validation: CorDP-style correction instead of effect-extraction.

The old pipeline asked retrieval for a structured {direction, magnitude}; magnitude came
back `unknown` because the LLM was asked in the abstract. CorDP (Beyond-Naive-Prompting
2508.09904) instead SHOWS the LLM the base forecast numbers + documents and asks it to
CORRECT them -- the magnitude is now anchored on concrete Toto values, so "unknown"
becomes an actual multiplier. We score CRPS (from the real Toto 9-quantile fan) on the
corrected window vs the Toto base, per task, and also print the LLM multiplier next to the
ORACLE multiplier (truth/base) so we can see if anchoring dissolved the magnitude wall.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/cordp_probe.py
"""
from __future__ import annotations
import json, os, shutil, statistics
from pathlib import Path

MULT_LO = float(os.environ.get("MULT_LO", "0.6"))   # bounded-adjustment kernel: clamp the
MULT_HI = float(os.environ.get("MULT_HI", "1.6"))   # LLM multiplier so a wrong call can't blow up
SHRINK = float(os.environ.get("SHRINK", "1.0"))     # partial trust: eff = 1 + SHRINK*(m-1)
CONF_MIN = float(os.environ.get("CONF_MIN", "0.0")) # abstain if self-reported confidence below this

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse, horizon_window_mask

ROOT = Path(".").resolve()
Q9 = json.loads((ROOT / ".scratch/dev_toto_q9.json").read_text())
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
if os.environ.get("TASK_SET", "").lower() == "all":
    KEY_TASKS = list(Q9.keys())
else:
    KEY_TASKS = os.environ.get("TASK_SET", "task_152,task_46,task_206").split(",")

llm = ClaudeCLIClient(ClaudeCLIConfig(
    binary=shutil.which("claude") or "claude",
    model="haiku",                       # match the haiku-big run's proposer
    timeout_seconds=900, cache_dir=str(ROOT / ".scratch/cordp-cache")))

tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, tuple(KEY_TASKS))}


def pinball(tau, q, y):
    return (y - q) * tau if y >= q else (q - y) * (1.0 - tau)


def crps9(qmat, truth, idx=None):
    idx = range(len(truth)) if idx is None else idx
    tot, n = 0.0, 0
    for i in idx:
        tot += 2.0 * sum(pinball(TAUS[k], qmat[k][i], truth[i]) for k in range(9)) / 9.0
        n += 1
    return tot / n if n else 0.0


def wd(ts):
    p = _parse(ts); return p.weekday() if p else -1


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
    "quoting the document.\n"
    "3. If the documents do NOT clearly imply any deviation from the base, return "
    "\"relevant\": false with an empty corrections list. Do not guess.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, '
    '"end_timestamp": iso, "multiplier": number, "rationale": str}], "confidence": number}.')


def build_user(task, base_p50):
    n = task.numeric
    hv, hts = list(n.history_values), [str(x) for x in task.history_timestamps]
    fts = [str(x) for x in task.future_timestamps]
    we = [x for x, t in zip(hv, hts) if wd(t) >= 5]
    wk = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
    hist = {
        "typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
        "overall_mean": round(statistics.mean(hv), 4),
        "weekday_mean": round(statistics.mean(wk), 4) if wk else None,
        "weekend_mean": round(statistics.mean(we), 4) if we else None,
        "last_values": [round(x, 4) for x in hv[-12:]],
    }
    docs = [{"document_id": d.document_id, "content": d.content[:3500]} for d in task.documents]
    base = [[fts[i], round(base_p50[i], 4)] for i in range(len(fts))]
    return json.dumps({
        "target_name": task.target_name,
        "target_description": task.target_description,
        "frequency": n.frequency,
        "history_summary": hist,
        "documents": docs,
        "base_forecast_median": base,
    }, ensure_ascii=False)


rows = []
for tid in KEY_TASKS:
    t = tasks.get(tid)
    if t is None or tid not in Q9:
        print(f"{tid}: SKIP (missing task or q9)"); continue
    n = t.numeric
    truth = list(n.future_values)
    qmat = Q9[tid]
    if len(qmat) != 9 or len(qmat[0]) != len(truth):
        print(f"{tid}: SKIP (q9 shape {len(qmat)}x{len(qmat[0])} vs truth {len(truth)})"); continue
    p50 = qmat[4]
    fts = [str(x) for x in t.future_timestamps]
    resp = llm.complete(system=SYSTEM,
                        messages=[{"role": "user", "content": build_user(t, p50)}],
                        temperature=0.0)
    try:
        out = parse_json_object(resp.text)
    except JsonExtractionError as exc:
        print(f"{tid}: LLM parse fail: {exc}"); continue
    try:
        conf = float(out.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    corrs = out.get("corrections") or []
    relevant = bool(out.get("relevant"))
    if conf < CONF_MIN:                              # confidence gate: abstain when unsure
        corrs, relevant = [], False
    # build corrected fan + affected mask
    mask = [False] * len(truth)
    corrected = [[qmat[k][i] for i in range(len(truth))] for k in range(9)]
    applied = []
    for c in corrs:
        try:
            m = float(c.get("multiplier"))
        except (TypeError, ValueError):
            continue
        m = 1.0 + SHRINK * (m - 1.0)                    # partial trust toward no-change
        m = max(MULT_LO, min(MULT_HI, m))               # kernel bound
        wm = horizon_window_mask(fts, str(c.get("start_timestamp")), str(c.get("end_timestamp")))
        if not any(wm):
            continue
        for i in range(len(truth)):
            if wm[i]:
                mask[i] = True
                for k in range(9):
                    corrected[k][i] = qmat[k][i] * m
        applied.append((m, sum(wm), (c.get("rationale") or "")[:80]))
    idx = [i for i in range(len(truth)) if mask[i]]
    base_full, corr_full = crps9(qmat, truth), crps9(corrected, truth)
    base_win = crps9(qmat, truth, idx) if idx else 0.0
    corr_win = crps9(corrected, truth, idx) if idx else 0.0
    # oracle multiplier on the affected window
    if idx:
        bw = statistics.mean([p50[i] for i in idx]); tw = statistics.mean([truth[i] for i in idx])
        oracle_m = (tw / bw) if bw else float("nan")
    else:
        oracle_m = float("nan")
    rows.append(dict(tid=tid, relevant=relevant, conf=conf, n_win=len(idx),
                     base_full=base_full, corr_full=corr_full,
                     base_win=base_win, corr_win=corr_win, oracle_m=oracle_m, applied=applied))
    print(f"\n{tid}: relevant={relevant} conf={conf:.2f} affected_steps={len(idx)}", flush=True)
    for m, ns, why in applied:
        print(f"    mult={m:.3f} on {ns} steps | oracle_mult={oracle_m:.3f} | {why}", flush=True)
    print(f"    CRPS window: base {base_win:.4f} -> corrected {corr_win:.4f} "
          f"({'BETTER' if corr_win < base_win else 'worse'}) | "
          f"full: {base_full:.4f} -> {corr_full:.4f}", flush=True)
    Path(".scratch/cordp_probe.json").write_text(json.dumps(rows, default=str))

print("\n== CorDP Route-A summary (windowed CRPS) ==")
win = [r for r in rows if r["n_win"] > 0]
for r in win:
    rel = (r["base_win"] - r["corr_win"]) / r["base_win"] if r["base_win"] else 0.0
    print(f"  {r['tid']}: {rel:+.1%}  (LLM mult vs oracle {r['oracle_m']:.2f})")
if win:
    mrel = statistics.mean((r["base_win"] - r["corr_win"]) / r["base_win"] for r in win if r["base_win"])
    reg = sum(1 for r in win if r["corr_win"] > r["base_win"] + 1e-9)
    print(f"  MEAN windowed rel improvement {mrel:+.1%} | regressions {reg}/{len(win)}")
else:
    print("  no task received a correction (all relevant=false) -> CorDP found no groundable edit")
