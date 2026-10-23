#!/usr/bin/env python3
"""Train-only random controls for the Time-MMD conservative point gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from native_uncertainty_risk_gate import build_rows
from risk_aware_statistical_gate import make_model, select_threshold, summarize


def gain(base, candidate):
    b, c = summarize(base), summarize(candidate)
    return {
        "sMAE": (b["sMAE"] - c["sMAE"]) / b["sMAE"],
        "sRMSE": (b["sRMSE"] - c["sRMSE"]) / b["sRMSE"],
        "joint": (b["joint"] - c["joint"]) / b["joint"],
    }


def evaluate(dataset: str, repeats: int = 20):
    rows = build_rows(dataset)
    X = np.stack([r["x"] for r in rows])
    Y = np.stack([r["target"] for r in rows])
    all_base, all_learned = [], []
    all_random_arm = [[] for _ in range(repeats)]
    all_random_both = [[] for _ in range(repeats)]
    folds = []

    for outer in range(3):
        train = np.asarray([i for i, r in enumerate(rows) if r["fold"] != outer])
        valid = np.asarray([i for i, r in enumerate(rows) if r["fold"] == outer])
        inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
        inner_pred = np.zeros((len(train), Y.shape[1]))
        for inner in range(3):
            apos = np.where(inner_id != inner)[0]
            bpos = np.where(inner_id == inner)[0]
            if len(apos) and len(bpos):
                inner_pred[bpos] = make_model(5100 + outer * 10 + inner).fit(
                    X[train[apos]], Y[train[apos]]).predict(X[train[bpos]])
        threshold = select_threshold(inner_pred, rows, train)["threshold"]
        pg = make_model(5200 + outer).fit(X[train], Y[train]).predict(X[valid])
        use = np.max(pg, axis=1) > threshold
        learned_arm = np.argmax(pg, axis=1)
        n_fire = int(use.sum())

        base_pairs = [rows[i]["base_pair"] for i in valid]
        learned_pairs = [
            rows[i]["arm_pairs"][learned_arm[k]] if use[k] else rows[i]["base_pair"]
            for k, i in enumerate(valid)
        ]
        all_base.extend(base_pairs)
        all_learned.extend(learned_pairs)

        fold_ra, fold_rb = [], []
        for rep in range(repeats):
            rng = np.random.default_rng(730000 + outer * 1000 + rep)
            random_arms = np.asarray([
                rng.integers(0, len(rows[i]["arms"])) for i in valid
            ])
            ra = [
                rows[i]["arm_pairs"][random_arms[k]] if use[k] else rows[i]["base_pair"]
                for k, i in enumerate(valid)
            ]
            random_use = np.zeros(len(valid), dtype=bool)
            if n_fire:
                random_use[rng.choice(len(valid), n_fire, replace=False)] = True
            random_arms_both = np.asarray([
                rng.integers(0, len(rows[i]["arms"])) for i in valid
            ])
            rb = [
                rows[i]["arm_pairs"][random_arms_both[k]] if random_use[k]
                else rows[i]["base_pair"]
                for k, i in enumerate(valid)
            ]
            all_random_arm[rep].extend(ra)
            all_random_both[rep].extend(rb)
            fold_ra.append(gain(base_pairs, ra))
            fold_rb.append(gain(base_pairs, rb))

        folds.append({
            "outer_fold": outer,
            "n_valid": len(valid),
            "threshold": threshold,
            "fires": n_fire,
            "coverage": n_fire / len(valid),
            "learned": gain(base_pairs, learned_pairs),
            "random_arm_mean": {m: float(np.mean([x[m] for x in fold_ra]))
                                for m in ("sMAE", "sRMSE", "joint")},
            "random_both_mean": {m: float(np.mean([x[m] for x in fold_rb]))
                                 for m in ("sMAE", "sRMSE", "joint")},
        })

    learned = gain(all_base, all_learned)
    ra = [gain(all_base, candidate) for candidate in all_random_arm]
    rb = [gain(all_base, candidate) for candidate in all_random_both]

    def dist(values):
        return {
            m: {
                "mean": float(np.mean([x[m] for x in values])),
                "std": float(np.std([x[m] for x in values], ddof=1)),
                "min": float(np.min([x[m] for x in values])),
                "max": float(np.max([x[m] for x in values])),
                "fraction_ge_learned": float(np.mean([x[m] >= learned[m] for x in values])),
            }
            for m in ("sMAE", "sRMSE", "joint")
        }

    return {
        "dataset": dataset,
        "protocol": "Train-only 3-fold outer CV; same coverage random controls",
        "n": len(rows),
        "repeats": repeats,
        "learned": learned,
        "random_arm_at_learned_positions": dist(ra),
        "random_positions_and_arm": dist(rb),
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("time_mmd", "timesx"), default="time_mmd")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("point_gate_random_controls.json"))
    args = ap.parse_args()
    out = evaluate(args.dataset, args.repeats)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
