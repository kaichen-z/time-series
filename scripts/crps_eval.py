"""CRPS evaluation with REAL Toto 9-quantile forecasts (from the Mac). For each dev task
the base predictive distribution is Toto's 9 quantiles; we then test document-driven
edits inside the event window and score CRPS (approximated from the 9 quantiles via
pinball loss). Reported as PER-TASK relative improvement (auto-normalized, so large-scale
tasks don't dominate) + regression count.

Edits:
  - shift:  move all in-window quantiles toward the LLM-chosen regime level (as in point mode)
  - widen:  inflate the in-window spread around the median (admit event uncertainty)
  - both.
Widening is near-zero-regression by construction; if it improves CRPS, the probabilistic
framing is the way document evidence should be used.

Run (after quantiles are on the server): PYTHONPATH=$PWD .venv/bin/python scripts/crps_eval.py
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, _parse, horizon_window_mask)

ROOT = Path(".").resolve()
Q9 = json.loads((ROOT / ".scratch/dev_toto_q9.json").read_text())
raw = json.loads((ROOT / ".scratch/effect_cards_f8d9d5862942.json").read_text())
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text())
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
import os
WIDEN = float(os.environ.get("WIDEN","2.0"))
dev_ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
                ["partitions"]["dev"]["task_ids"])
tasks = {t.numeric.task_id: t for t in
         load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", dev_ids)}
FIELDS = list(EvidenceEffect.__dataclass_fields__.keys())


def pinball(tau, q, y):
    return (y - q) * tau if y >= q else (q - y) * (1.0 - tau)


def crps9(qmat, truth):
    H = len(truth)
    tot = 0.0
    for i in range(H):
        tot += 2.0 * sum(pinball(TAUS[k], qmat[k][i], truth[i]) for k in range(9)) / 9.0
    return tot / H


def window(tid, fts):
    m = [False] * len(fts)
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        e = EvidenceEffect(**{**kw, "direction": _canon_direction(kw.get("direction"))})
        if e.grounded and e.start_timestamp and e.end_timestamp:
            m = [a or b for a, b in zip(m, horizon_window_mask(fts, e.start_timestamp, e.end_timestamp))]
    return m


def reflevel(hv, hts, ref):
    def wd(t):
        p = _parse(t); return p.weekday() if p else -1
    if ref == "weekend":
        v = [x for x, t in zip(hv, hts) if wd(t) >= 5]; return statistics.mean(v) if v else None
    if ref == "weekday":
        v = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]; return statistics.mean(v) if v else None
    if ref in ("low_day", "high_day"):
        bd = {}
        for x, t in zip(hv, hts):
            p = _parse(t)
            if p: bd.setdefault(p.date(), []).append(x)
        dm = sorted(statistics.mean(z) for z in bd.values()) if bd else []
        return (dm[0] if ref == "low_day" else dm[-1]) if dm else None
    return None


def edit(qmat, mask, factor, widen):
    H = len(qmat[0])
    out = [[qmat[k][i] for i in range(H)] for k in range(9)]
    for i in range(H):
        if not mask[i]:
            continue
        med = qmat[4][i]
        for k in range(9):
            v = qmat[k][i] * factor                          # shift
            v = med * factor + (v - med * factor) * widen    # widen around shifted median
            out[k][i] = v
    return out


rows = []
for tid in Q9:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    truth = list(n.future_values)
    qmat = Q9[tid]
    if len(qmat) != 9 or len(qmat[0]) != len(truth):
        continue
    fts = [str(x) for x in t.future_timestamps]
    hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
    mask = window(tid, fts)
    ref = REFS.get(tid, "none")
    lvl = reflevel(hv, hts, ref) if ref not in ("none", "overall", "recent") else None
    bw = statistics.mean([qmat[4][i] for i in range(len(truth)) if mask[i]]) if any(mask) else 0.0
    factor = (lvl / bw) if (lvl is not None and bw > 0) else 1.0
    base = crps9(qmat, truth)
    widen = crps9(edit(qmat, mask, 1.0, WIDEN), truth)
    shift = crps9(edit(qmat, mask, factor, 1.0), truth)
    both = crps9(edit(qmat, mask, factor, WIDEN), truth)
    rows.append(dict(tid=tid, win=any(mask), base=base, widen=widen, shift=shift, both=both, ref=ref))

win = [r for r in rows if r["win"] and r["base"] > 1e-9]
print(f"\n== CRPS with REAL Toto quantiles: {len(rows)} dev tasks ({len(win)} windowed) ==")
print(f"  mean base CRPS (windowed) = {statistics.mean([r['base'] for r in win]):.4f}")
for label in ["widen", "shift", "both"]:
    rel = [(r['base'] - r[label]) / r['base'] for r in win]        # per-task relative improvement
    reg = sum(1 for r in win if r[label] > r['base'] + 1e-9)
    print(f"  {label:6s}: mean rel improvement {statistics.mean(rel):+.4f} "
          f"({'IMPROVES' if statistics.mean(rel) > 0 else 'worse'}), regressions {reg}/{len(win)}")
