#!/usr/bin/env python3
"""Train-only chronological event-date holdout for the frozen Time-MMD gate."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_controller import pack_rows
from native_uncertainty_ablation import crps_fast, horizon_key, sample_array
from native_uncertainty_risk_gate import build_rows
from risk_aware_statistical_gate import make_model, select_threshold, summarize


def latest_event_date(view):
    dates = []
    for doc in view.get("documents", []) or []:
        for event in doc.get("events", []) if isinstance(doc, dict) else []:
            for key in ("time_end", "time_start"):
                value = event.get(key)
                if value:
                    try:
                        dates.append(dt.date.fromisoformat(str(value)[:10]))
                    except ValueError:
                        pass
    return max(dates) if dates else None


def point_gain(base, candidate):
    b, c = summarize(base), summarize(candidate)
    return {
        "base": b, "candidate": c,
        "gains": {m: (b[m] - c[m]) / b[m] for m in ("sMAE", "sRMSE", "joint")},
    }


def evaluate(samples=100, random_repeats=20):
    rows = build_rows("time_mmd")
    dates = [latest_event_date(view) for view, _ in pack_rows("time_mmd")]
    assert len(rows) == len(dates)
    dated = sorted(d for d in dates if d is not None)
    cutoff = dated[int(0.8 * len(dated))]
    train = np.asarray([i for i, d in enumerate(dates) if d is None or d < cutoff])
    valid = np.asarray([i for i, d in enumerate(dates) if d is not None and d >= cutoff])

    X = np.stack([r["x"] for r in rows])
    Y = np.stack([r["target"] for r in rows])
    inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
    inner_pred = np.zeros((len(train), Y.shape[1]))
    for inner in range(3):
        apos = np.where(inner_id != inner)[0]
        bpos = np.where(inner_id == inner)[0]
        if len(apos) and len(bpos):
            inner_pred[bpos] = make_model(9100 + inner).fit(
                X[train[apos]], Y[train[apos]]).predict(X[train[bpos]])
    threshold = select_threshold(inner_pred, rows, train)["threshold"]

    base_bank, gate_bank = defaultdict(list), defaultdict(list)
    for k, i in enumerate(train):
        r = rows[i]
        j = int(np.argmax(inner_pred[k]))
        use = bool(inner_pred[k, j] > threshold)
        pred = r["arms"][j] if use else r["base"]
        h = horizon_key(r["stratum"])
        base_bank[h].append(r["truth"] - r["base"])
        gate_bank[h].append(r["truth"] - pred)
    base_bank = {k: np.stack(v) for k, v in base_bank.items()}
    gate_bank = {k: np.stack(v) for k, v in gate_bank.items()}

    pg = make_model(9200).fit(X[train], Y[train]).predict(X[valid])
    use = np.max(pg, axis=1) > threshold
    chosen = np.argmax(pg, axis=1)
    base_pairs, gate_pairs, base_crps, gate_crps = [], [], [], []
    random_pairs = [[] for _ in range(random_repeats)]
    for k, i in enumerate(valid):
        r = rows[i]
        j = int(chosen[k])
        pred = r["arms"][j] if use[k] else r["base"]
        bp = r["base_pair"]
        gp = r["arm_pairs"][j] if use[k] else bp
        base_pairs.append(bp); gate_pairs.append(gp)
        h = horizon_key(r["stratum"])
        bres = sample_array(base_bank[h], samples, 930000 + i)
        gres = sample_array(gate_bank[h], samples, 940000 + i)
        base_crps.append(min(5.0, crps_fast(r["base"][None, :] + bres, r["truth"]) / r["scale"]))
        gate_crps.append(min(5.0, crps_fast(pred[None, :] + gres, r["truth"]) / r["scale"]))
        for rep in range(random_repeats):
            if use[k]:
                rng = np.random.default_rng(950000 + rep * 100000 + i)
                arm = int(rng.integers(0, len(r["arms"])))
                random_pairs[rep].append(r["arm_pairs"][arm])
            else:
                random_pairs[rep].append(bp)

    point = point_gain(base_pairs, gate_pairs)
    random_gains = [point_gain(base_pairs, rp)["gains"] for rp in random_pairs]
    bcrps, gcrps = float(np.mean(base_crps)), float(np.mean(gate_crps))
    by_frequency = {}
    for freq in sorted({rows[i]["stratum"].split("|")[0] for i in valid}):
        pos = [k for k, i in enumerate(valid) if rows[i]["stratum"].startswith(freq + "|")]
        by_frequency[freq] = point_gain([base_pairs[k] for k in pos], [gate_pairs[k] for k in pos])
    return {
        "dataset": "time_mmd", "protocol": "Train-only event-date chronological 80/20 holdout",
        "cutoff": cutoff.isoformat(), "dated_rows": len(dated),
        "undated_train_rows": sum(d is None for d in dates),
        "n_train": len(train), "n_valid": len(valid), "threshold": threshold,
        "coverage": float(np.mean(use)), "point": point,
        "horizon_residual_crps": {
            "base": bcrps, "gate": gcrps, "gain": (bcrps - gcrps) / bcrps,
        },
        "random_arm_same_positions": {
            m: {"mean": float(np.mean([x[m] for x in random_gains])),
                "min": float(np.min([x[m] for x in random_gains])),
                "max": float(np.max([x[m] for x in random_gains])),
                "fraction_ge_learned": float(np.mean([x[m] >= point["gains"][m] for x in random_gains]))}
            for m in ("sMAE", "sRMSE", "joint")
        },
        "by_frequency": by_frequency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--random-repeats", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("time_mmd_chronological_event_holdout.json"))
    args = ap.parse_args()
    out = evaluate(args.samples, args.random_repeats)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
