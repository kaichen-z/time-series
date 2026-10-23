#!/usr/bin/env python3
"""Descriptive, strictly walk-forward drift audit on dated Time-MMD Train rows.

The full Train set has already been inspected, so this develops a protocol rather
than producing fresh confirmation. Undated rows are excluded: admitting them to
an event-time audit would silently assign information to an unknown past.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_controller import pack_rows
from native_uncertainty_risk_gate import build_rows
from time_mmd_chronological_event_holdout import latest_event_date, point_gain
from time_mmd_temporal_frequency_release import fit_gate


def decisions(rows, pred, indices, threshold):
    out = []
    for k, i in enumerate(indices):
        j = int(np.argmax(pred[k]))
        use = bool(pred[k, j] > threshold)
        out.append((i, j, use))
    return out


def covariate_shift(train_x, valid_x):
    """Prediction-time-visible diagonal shift scores; no labels used."""
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    z = np.abs((valid_x.mean(axis=0) - mu) / np.maximum(sd, 1e-8))
    active = sd > 1e-8
    z_active = z[active]
    psi = []
    for j in np.where(active)[0]:
        edges = np.unique(np.quantile(train_x[:, j], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(train_x[:, j], bins=edges)[0].astype(float)
        q = np.histogram(valid_x[:, j], bins=edges)[0].astype(float)
        p = np.maximum(p / p.sum(), 1e-6)
        q = np.maximum(q / q.sum(), 1e-6)
        psi.append(float(np.sum((q - p) * np.log(q / p))))
    return {
        "standardized_mean_l2": float(np.sqrt(np.mean(z_active ** 2))),
        "standardized_mean_max": float(np.max(z_active)),
        "psi_mean": float(np.mean(psi)) if psi else 0.0,
        "psi_max": float(np.max(psi)) if psi else 0.0,
    }


def evaluate(initial_fraction=0.4, n_eval_blocks=6, recent_fraction=0.25):
    rows = build_rows("time_mmd")
    dates = [latest_event_date(view) for view, _ in pack_rows("time_mmd")]
    dated = np.asarray(sorted((i for i, d in enumerate(dates) if d is not None),
                              key=lambda i: (dates[i], i)))
    undated_n = len(rows) - len(dated)
    X = np.stack([r["x"] for r in rows]); Y = np.stack([r["target"] for r in rows])

    initial_n = int(initial_fraction * len(dated))
    edges = np.linspace(initial_n, len(dated), n_eval_blocks + 1, dtype=int)
    recent_n = max(1000, int(recent_fraction * len(dated)))
    history = defaultdict(list)
    blocks = []
    pooled = {"base": [], "expanding": [], "recent": [], "stable_recent": []}

    for b in range(n_eval_blocks):
        start, stop = int(edges[b]), int(edges[b + 1])
        valid = dated[start:stop]
        expanding_train = dated[:start]
        recent_train = dated[max(0, start - recent_n):start]

        _, et, em = fit_gate(rows, X, Y, expanding_train, 21000 + 100 * b)
        _, rt, rm = fit_gate(rows, X, Y, recent_train, 31000 + 100 * b)
        ed = decisions(rows, em.predict(X[valid]), valid, et)
        rd = decisions(rows, rm.predict(X[valid]), valid, rt)

        base, expanding, recent, stable = [], [], [], []
        by_freq = defaultdict(lambda: {"base": [], "expanding": [], "recent": [], "stable": []})
        counts = defaultdict(int)
        released = sorted(f for f, gains in history.items()
                          if len(gains) >= 2 and gains[-1] > 0 and gains[-2] > 0)
        for (i, ej, eu), (_, rj, ru) in zip(ed, rd):
            r = rows[i]; freq = r["stratum"].split("|")[0]; bp = r["base_pair"]
            ep = r["arm_pairs"][ej] if eu else bp
            rp = r["arm_pairs"][rj] if ru else bp
            su = bool(ru and freq in released)
            sp = r["arm_pairs"][rj] if su else bp
            base.append(bp); expanding.append(ep); recent.append(rp); stable.append(sp)
            by_freq[freq]["base"].append(bp)
            by_freq[freq]["expanding"].append(ep)
            by_freq[freq]["recent"].append(rp)
            by_freq[freq]["stable"].append(sp)
            counts[f"expanding_{freq}"] += int(eu)
            counts[f"recent_{freq}"] += int(ru)
            counts[f"stable_{freq}"] += int(su)

        freq_results = {}
        for freq, vals in sorted(by_freq.items()):
            fr = {
                "expanding": point_gain(vals["base"], vals["expanding"]),
                "recent": point_gain(vals["base"], vals["recent"]),
                "stable_recent": point_gain(vals["base"], vals["stable"]),
            }
            freq_results[freq] = fr
            history[freq].append(fr["recent"]["gains"]["joint"])

        result = {
            "block": b, "start_date": dates[valid[0]].isoformat(),
            "end_date": dates[valid[-1]].isoformat(), "n": len(valid),
            "expanding_train_n": len(expanding_train), "recent_train_n": len(recent_train),
            "released_from_two_prior_positive_blocks": released,
            "covariate_shift_vs_recent_train": covariate_shift(X[recent_train], X[valid]),
            "expanding": point_gain(base, expanding), "recent": point_gain(base, recent),
            "stable_recent": point_gain(base, stable), "by_frequency": freq_results,
            "counts": dict(counts),
        }
        blocks.append(result)
        for key, vals in (("base", base), ("expanding", expanding), ("recent", recent),
                          ("stable_recent", stable)):
            pooled[key].extend(vals)

    return {
        "dataset": "time_mmd",
        "status": "descriptive protocol development; all Train periods were already inspected",
        "protocol": "dated rows only; 40% initial history then six strict next-block forecasts; recent window=25% of dated rows; stable_recent releases a frequency only after its two immediately prior recent-window block gains are both positive",
        "n_dated": len(dated), "n_undated_excluded": undated_n,
        "initial_n": initial_n, "recent_window_n": recent_n,
        "blocks": blocks,
        "pooled": {
            "expanding": point_gain(pooled["base"], pooled["expanding"]),
            "recent": point_gain(pooled["base"], pooled["recent"]),
            "stable_recent": point_gain(pooled["base"], pooled["stable_recent"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("time_mmd_prequential_drift_audit.json"))
    args = ap.parse_args()
    out = evaluate()
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
