#!/usr/bin/env python3
"""Post-hoc temporal frequency-release diagnostic for Time-MMD Train."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_controller import pack_rows
from native_uncertainty_ablation import crps_fast, horizon_key, sample_array
from native_uncertainty_risk_gate import build_rows
from risk_aware_statistical_gate import make_model, select_threshold, summarize
from time_mmd_chronological_event_holdout import latest_event_date, point_gain


def fit_gate(rows, X, Y, train, seed):
    inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
    inner_pred = np.zeros((len(train), Y.shape[1]))
    for inner in range(3):
        apos = np.where(inner_id != inner)[0]
        bpos = np.where(inner_id == inner)[0]
        if len(apos) and len(bpos):
            inner_pred[bpos] = make_model(seed + inner).fit(
                X[train[apos]], Y[train[apos]]).predict(X[train[bpos]])
    threshold = select_threshold(inner_pred, rows, train)["threshold"]
    model = make_model(seed + 10).fit(X[train], Y[train])
    return inner_pred, threshold, model


def evaluate(samples=100):
    rows = build_rows("time_mmd")
    dates = [latest_event_date(view) for view, _ in pack_rows("time_mmd")]
    dated = sorted(d for d in dates if d is not None)
    cut1 = dated[int(0.6 * len(dated))]
    cut2 = dated[int(0.8 * len(dated))]
    undated = np.asarray([i for i, d in enumerate(dates) if d is None])
    early = np.asarray([i for i, d in enumerate(dates) if d is not None and d < cut1])
    middle = np.asarray([i for i, d in enumerate(dates) if d is not None and cut1 <= d < cut2])
    late = np.asarray([i for i, d in enumerate(dates) if d is not None and d >= cut2])
    train1 = np.concatenate([undated, early])
    train2 = np.concatenate([undated, early, middle])
    X = np.stack([r["x"] for r in rows]); Y = np.stack([r["target"] for r in rows])

    _, threshold1, model1 = fit_gate(rows, X, Y, train1, 10100)
    pm = model1.predict(X[middle])
    middle_base, middle_gate = [], []
    middle_by_freq = defaultdict(lambda: [[], []])
    for k, i in enumerate(middle):
        r = rows[i]; j = int(np.argmax(pm[k])); use = bool(pm[k, j] > threshold1)
        bp = r["base_pair"]; gp = r["arm_pairs"][j] if use else bp
        middle_base.append(bp); middle_gate.append(gp)
        freq = r["stratum"].split("|")[0]
        middle_by_freq[freq][0].append(bp); middle_by_freq[freq][1].append(gp)
    middle_results = {f: point_gain(a, b) for f, (a, b) in middle_by_freq.items()}
    released = sorted(f for f, result in middle_results.items()
                      if result["gains"]["sMAE"] >= 0 and result["gains"]["sRMSE"] >= 0)

    inner2, threshold2, model2 = fit_gate(rows, X, Y, train2, 10200)
    base_bank, release_bank = defaultdict(list), defaultdict(list)
    for k, i in enumerate(train2):
        r = rows[i]; j = int(np.argmax(inner2[k])); freq = r["stratum"].split("|")[0]
        use = bool(inner2[k, j] > threshold2 and freq in released)
        pred = r["arms"][j] if use else r["base"]
        h = horizon_key(r["stratum"])
        base_bank[h].append(r["truth"] - r["base"])
        release_bank[h].append(r["truth"] - pred)
    base_bank = {k: np.stack(v) for k, v in base_bank.items()}
    release_bank = {k: np.stack(v) for k, v in release_bank.items()}

    pl = model2.predict(X[late])
    late_base, full_gate, release_gate = [], [], []
    base_crps, release_crps = [], []
    counts = defaultdict(int)
    for k, i in enumerate(late):
        r = rows[i]; j = int(np.argmax(pl[k])); freq = r["stratum"].split("|")[0]
        use = bool(pl[k, j] > threshold2)
        released_use = bool(use and freq in released)
        bp = r["base_pair"]
        late_base.append(bp)
        full_gate.append(r["arm_pairs"][j] if use else bp)
        release_gate.append(r["arm_pairs"][j] if released_use else bp)
        counts[f"proposed_{freq}"] += int(use)
        counts[f"released_{freq}"] += int(released_use)
        pred = r["arms"][j] if released_use else r["base"]
        h = horizon_key(r["stratum"])
        bres = sample_array(base_bank[h], samples, 1030000 + i)
        gres = sample_array(release_bank[h], samples, 1040000 + i)
        base_crps.append(min(5.0, crps_fast(r["base"][None, :] + bres, r["truth"]) / r["scale"]))
        release_crps.append(min(5.0, crps_fast(pred[None, :] + gres, r["truth"]) / r["scale"]))
    bc, gc = float(np.mean(base_crps)), float(np.mean(release_crps))
    return {
        "dataset": "time_mmd",
        "status": "post-hoc diagnostic; final 20% was already opened before this rule was proposed",
        "protocol": "0-60% event-time train; 60-80% frequency release; 80-100% diagnostic",
        "cut1": cut1.isoformat(), "cut2": cut2.isoformat(),
        "n_early_plus_undated": len(train1), "n_middle": len(middle), "n_late": len(late),
        "middle_full_gate": point_gain(middle_base, middle_gate),
        "middle_by_frequency": middle_results, "released_frequencies": released,
        "late_full_gate": point_gain(late_base, full_gate),
        "late_frequency_release_gate": point_gain(late_base, release_gate),
        "late_frequency_release_crps": {"base": bc, "gate": gc, "gain": (bc - gc) / bc},
        "counts": dict(counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("time_mmd_temporal_frequency_release.json"))
    args = ap.parse_args()
    out = evaluate(args.samples)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
