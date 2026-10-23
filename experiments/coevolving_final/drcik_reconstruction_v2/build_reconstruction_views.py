#!/usr/bin/env python3
"""Rebuild the old Dr-CiK mixed views with a newly generated Luna timeline.

This intentionally preserves the historical coevo_x view, Haiku cards, raw
forecasts, and frozen forecast/adjust programs.  Only the old
``fill_variants.phase_median`` branch is regenerated.  Repair semantics and
the 30% history-only gate match the archived 2026-09-27 implementation:
measurement_error, one_off_disturbance, and recurring_pattern intervals that
do not recur in the future are phase-median-filled; the repaired Toto forecast
is exposed iff val_rep < 0.7 * val_raw.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np

REPAIR_KINDS = {"measurement_error", "one_off_disturbance", "recurring_pattern"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ts(x):
    s = str(x).replace("Z", "").replace("T", " ")[:19]
    for cand in (s, s[:16], s[:10]):
        try:
            return datetime.fromisoformat(cand)
        except ValueError:
            pass
    return None


def hm_time(x):
    try:
        h, m = str(x).split(":")[:2]
        return dtime(int(h), int(m))
    except Exception:
        return None


def interval_mask(stamps, iv):
    s, e = ts(iv.get("start")), ts(iv.get("end"))
    if s is None or e is None:
        return np.zeros(len(stamps), bool)
    if e < s:
        s, e = e, s
    if e.hour == 0 and e.minute == 0 and len(str(iv.get("end"))) <= 10:
        e = e + timedelta(days=1) - timedelta(seconds=1)
    ds, de = hm_time(iv.get("daily_start")), hm_time(iv.get("daily_end"))
    out = []
    for t in stamps:
        ok = t is not None and s <= t <= e
        if ok and iv.get("recurrence") == "daily" and ds and de:
            ok = ds <= t.time() <= de if ds <= de else (t.time() >= ds or t.time() <= de)
        if ok and iv.get("recurrence") == "every_n_days" and iv.get("every_n_days"):
            n = int(iv["every_n_days"]) or 1
            ok = ((t - s).days % n) == 0 or iv.get("kind") != "recurring_pattern"
        out.append(ok)
    return np.asarray(out, bool)


def period_of(freq):
    f = (freq or "").lower()
    if "5 min" in f:
        return 288
    if "min" in f:
        return 60
    if "hour" in f:
        return 24
    if "day" in f:
        return 7
    return 1


def phase_median(history, timestamps, freq, intervals):
    y = np.asarray(history, float)
    stamps = [ts(x) for x in timestamps]
    bad = np.zeros(len(y), bool)
    for iv in intervals:
        if iv.get("kind") in REPAIR_KINDS and not iv.get("recurs_in_future"):
            bad |= interval_mask(stamps, iv)
    if not bad.any() or bad.mean() >= 0.9:
        return None, float(bad.mean())
    p, z, n, good = period_of(freq), y.copy(), len(y), np.where(~bad)[0]
    for i in np.where(bad)[0]:
        same = [y[j] for j in range(i % p, n, p) if not bad[j]] if p > 1 else []
        z[i] = float(np.median(same)) if len(same) >= 2 else float(np.interp(i, good, y[good]))
    return z.tolist(), float(bad.mean())


def err(fc, target):
    fc, target = np.asarray(fc, float), np.asarray(target, float)
    scale = float(np.abs(target).mean()) + 1e-9
    d = fc - target
    return float(np.abs(d).mean()) / scale + math.sqrt(float((d * d).mean())) / scale


def prepare(a):
    old = json.load(open(a.old_views))
    timeline = json.load(open(a.timeline))
    inputs, meta = [], {}
    for tid, view in old.items():
        raw = json.load(open(a.tasks / f"{tid}.json"))
        series = raw["series"]
        history = [float(x) for x in series["history_values"]]
        assert history == view["history"], f"history mismatch: {tid}"
        z, frac = phase_median(history, series["history_timestamps"], view["freq"], timeline[tid])
        if z is None:
            continue
        H = view["H"]
        rec = {"frac": frac, "history": z, "H": H}
        inputs.append({"tid": f"{tid}|full", "history": z, "H": H, "freq": view["freq"]})
        if len(history) - H >= max(H, 16) and len(z) - H >= max(H, 8):
            raw_bt = next((b["toto_2_0"] for b in view.get("backtests", []) if b.get("cut") == len(history) - H), None)
            if raw_bt is None:
                inputs.append({"tid": f"{tid}|rawbt", "history": history[:-H], "H": H, "freq": view["freq"]})
            inputs.append({"tid": f"{tid}|repbt", "history": z[:-H], "H": H, "freq": view["freq"]})
            rec["raw_bt_cached"] = raw_bt
            rec["target"] = z[-H:]
        meta[tid] = rec
    json.dump(inputs, open(a.inputs, "w"))
    json.dump(meta, open(a.meta, "w"))
    print(json.dumps({"candidate_tasks": len(meta), "toto_inputs": len(inputs)}))


def finish(a):
    old = json.load(open(a.old_views))
    q = json.load(open(a.toto_q))["q"]
    meta = json.load(open(a.meta))
    views = copy.deepcopy(old)
    accepted, audit = [], {}
    for tid, view in views.items():
        view["toto_forecast_repaired_history"] = None
        if tid not in meta:
            audit[tid] = {"candidate": False, "accepted": False}
            continue
        m = meta[tid]
        val_raw = val_rep = None
        if "target" in m:
            rawfc = m.get("raw_bt_cached") or q[f"{tid}|rawbt"][4]
            val_raw = err(rawfc, m["target"])
            val_rep = err(q[f"{tid}|repbt"][4], m["target"])
        accept = val_raw is not None and val_rep < 0.7 * val_raw
        if accept:
            view["toto_forecast_repaired_history"] = [float(x) for x in q[f"{tid}|full"][4][:view["H"]]]
            accepted.append(tid)
        audit[tid] = {"candidate": True, "accepted": accept, "frac": m["frac"], "val_raw": val_raw, "val_rep": val_rep}
    json.dump(views, open(a.out_views, "w"))
    json.dump(audit, open(a.audit, "w"), indent=1)
    receipt = {
        "schema": "drcik-mixed-reconstruction-v2",
        "note": "new nondeterministic Luna extraction; historical Haiku cards/raw views/frozen programs preserved",
        "old_views_sha256": sha(a.old_views),
        "timeline_sha256": sha(a.timeline),
        "toto_q_sha256": sha(a.toto_q),
        "out_views_sha256": sha(a.out_views),
        "candidate_tasks": len(meta),
        "accepted_tasks": accepted,
        "accepted_count": len(accepted),
    }
    json.dump(receipt, open(a.receipt, "w"), indent=1)
    print(json.dumps({"accepted_count": len(accepted), "accepted_tasks": accepted}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "finish"])
    p.add_argument("--old-views", type=Path, required=True)
    p.add_argument("--timeline", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--toto-q", type=Path, required=True)
    p.add_argument("--out-views", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    prepare(a) if a.mode == "prepare" else finish(a)


if __name__ == "__main__":
    main()
