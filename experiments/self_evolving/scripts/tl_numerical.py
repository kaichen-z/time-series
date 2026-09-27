"""Numerical side of the state-timeline method (2026-09-27).
* history repair: impute non-recurring history anomalies (seasonal same-phase median, else linear), so
  Toto can be re-run on the cleaned history (tl_toto.py, GPU);
* state effects: for every state label that is 'on' somewhere in the forecast, measure its effect from the
  HISTORY (value / same-phase baseline when on vs off) and turn it into per-step correction factors;
* everything here uses history values and the extracted timeline only (no future values).
"""
from __future__ import annotations
import json, statistics
from datetime import datetime
from pathlib import Path
import numpy as np

TASKS = Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks")


def ts(x):
    s = str(x).replace("Z", "").replace("T", " ")[:19]
    try: return datetime.fromisoformat(s)
    except ValueError:
        try: return datetime.fromisoformat(s[:10])
        except ValueError: return None


def period_of(freq):
    f = (freq or "").lower()
    if "5 min" in f: return 288
    if "1 min" in f or f == "minute": return 60
    if "hour" in f: return 24
    if "day" in f: return 7
    if "second" in f: return 1
    return 1


def masks(raw, iv):
    s, e = ts(iv["start"]), ts(iv["end"])
    if s is None or e is None: return None, None
    if e < s: s, e = e, s
    ht = [ts(x) for x in raw["series"]["history_timestamps"]]; ft = [ts(x) for x in raw["series"]["future_timestamps"]]
    hm = np.array([t is not None and s <= t <= e for t in ht]); fm = np.array([t is not None and s <= t <= e for t in ft])
    return hm, fm


def repair_history(raw, ivs):
    """returns cleaned history (or None if nothing to repair) -- non-recurring anomalies only."""
    y = np.array(raw["series"]["history_values"], float); bad = np.zeros(len(y), bool)
    for iv in ivs:
        if iv["kind"] == "history_anomaly" and not iv["recurs_in_future"]:
            hm, _ = masks(raw, iv)
            if hm is not None: bad |= hm
    if not bad.any() or bad.all(): return None
    p = period_of(raw["task_metadata"].get("frequency")); z = y.copy(); n = len(y)
    for i in np.where(bad)[0]:
        same = [y[j] for j in range(i % p if p > 1 else 0, n, p if p > 1 else 1) if not bad[j]] if p > 1 else []
        if len(same) >= 2: z[i] = float(np.median(same))
        else:
            good = np.where(~bad)[0]; z[i] = float(np.interp(i, good, y[good]))
    return z.tolist()


def state_factors(raw, ivs, min_on=2):
    """per-step multiplicative factors for the forecast from states measured on the history."""
    y = np.array(raw["series"]["history_values"], float); H = len(raw["series"]["future_timestamps"])
    p = period_of(raw["task_metadata"].get("frequency")); n = len(y)
    by = {}
    for iv in ivs:
        if iv["kind"] != "state": continue
        hm, fm = masks(raw, iv)
        if hm is None: continue
        a = by.setdefault(iv["label"].strip().lower(), [np.zeros(n, bool), np.zeros(H, bool), iv["direction"]])
        a[0] |= hm; a[1] |= fm
    fac = np.ones(H); info = {}
    for lab, (hm, fm, dr) in by.items():
        if not fm.any() or hm.sum() < min_on or (~hm).sum() < min_on: continue
        # same-phase baseline from 'off' steps
        base = np.empty(n)
        for i in range(n):
            ph = [y[j] for j in range(i % p, n, p) if not hm[j]] if p > 1 else [y[j] for j in range(n) if not hm[j]]
            base[i] = np.median(ph) if ph else np.median(y[~hm])
        ok = np.abs(base) > 1e-9
        r_on = np.median((y[hm & ok] / base[hm & ok])) if (hm & ok).sum() >= min_on else None
        if r_on is None or not np.isfinite(r_on): continue
        r_on = float(np.clip(r_on, 0.0, 5.0))
        mix = (hm.mean() * r_on + (1 - hm.mean()) * 1.0)          # what an unconditioned forecaster averages
        f_on, f_off = r_on / mix, 1.0 / mix
        # only move steps whose state is known: on-steps -> f_on, and off-steps inside the forecast -> f_off
        fac = fac * np.where(fm, f_on, f_off)
        info[lab] = dict(r_on=round(r_on, 3), n_on=int(hm.sum()), n_off=int((~hm).sum()), fut_on=int(fm.sum()))
    return fac.tolist(), info


def build(timeline_path, out_path):
    tl = json.load(open(timeline_path)); out = {}
    for tid, ivs in tl.items():
        raw = json.loads((TASKS / f"{tid}.json").read_text())
        fac, info = state_factors(raw, ivs)
        out[tid] = dict(clean_history=repair_history(raw, ivs), factors=fac, state_info=info,
                        n_intervals=len(ivs), kinds={k: sum(1 for i in ivs if i["kind"] == k) for k in ("history_anomaly", "state", "future_event")})
    json.dump(out, open(out_path, "w")); return out


if __name__ == "__main__":
    import sys
    o = build(sys.argv[1], sys.argv[2])
    print("tasks", len(o), "with repair", sum(1 for v in o.values() if v["clean_history"]), "with state factors", sum(1 for v in o.values() if v["state_info"]))
