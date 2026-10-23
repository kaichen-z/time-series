#!/usr/bin/env python3
"""Train-only attribution ablations for Time-MMD empirical uncertainty.

The point gate and every outer/inner split are identical to
native_uncertainty_risk_gate.py.  Only the residual-bank conditioning changes,
so CRPS differences can be attributed to calibration granularity rather than
to a different point policy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from native_uncertainty_risk_gate import build_rows
from risk_aware_statistical_gate import make_model, select_threshold, summarize


MODES = ("expert", "stratum", "horizon")


def horizon_key(stratum: str) -> str:
    return stratum.split("|", 1)[1]


def sample_array(bank: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return bank[rng.choice(len(bank), n, replace=len(bank) < n)]


def crps_fast(samples: np.ndarray, truth: np.ndarray) -> float:
    """Exact empirical CRPS in O(n log n), equivalent to pairwise form."""
    a = np.mean(np.abs(samples - truth[None, :]), axis=0)
    xs = np.sort(samples, axis=0)
    n = len(xs)
    weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    half_pair = np.sum(weights * xs, axis=0) / (n * n)
    return float(np.mean(a - half_pair))


def evaluate(dataset: str, samples: int = 100):
    rows = build_rows(dataset)
    X = np.stack([r["x"] for r in rows])
    Y = np.stack([r["target"] for r in rows])
    all_base_pairs, all_gate_pairs = [], []
    crps_all = {m: {"base": [], "gate": []} for m in MODES}
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

        selected = select_threshold(inner_pred, rows, train)
        threshold = selected["threshold"]
        base = {m: defaultdict(list) for m in MODES}
        gate = {m: defaultdict(list) for m in MODES}

        for k, i in enumerate(train):
            r = rows[i]
            j = int(np.argmax(inner_pred[k]))
            use = bool(inner_pred[k, j] > threshold)
            pred = r["arms"][j] if use else r["base"]
            name = r["names"][j] if use else "ABSTAIN_V6"
            bres = r["truth"] - r["base"]
            gres = r["truth"] - pred
            h = horizon_key(r["stratum"])
            keys = {
                "expert": (r["stratum"], name),
                "stratum": r["stratum"],
                "horizon": h,
            }
            for mode, key in keys.items():
                base[mode][key].append(bres)
                gate[mode][key].append(gres)

        # Stack once per bank rather than once per held-out row.
        for mode in MODES:
            base[mode] = {key: np.stack(value) for key, value in base[mode].items()}
            gate[mode] = {key: np.stack(value) for key, value in gate[mode].items()}

        model = make_model(5200 + outer).fit(X[train], Y[train])
        pg = model.predict(X[valid])
        fold_base_pairs, fold_gate_pairs = [], []
        fold_crps = {m: {"base": [], "gate": []} for m in MODES}
        expert_backoffs = 0
        for k, i in enumerate(valid):
            r = rows[i]
            j = int(np.argmax(pg[k]))
            use = bool(pg[k, j] > threshold)
            pred = r["arms"][j] if use else r["base"]
            name = r["names"][j] if use else "ABSTAIN_V6"
            h = horizon_key(r["stratum"])
            keys = {
                "expert": (r["stratum"], name),
                "stratum": r["stratum"],
                "horizon": h,
            }
            # Match the original protocol: expert cells with <30 examples
            # back off to the stratum bank for both base and gate.
            expert_cell = gate["expert"].get(keys["expert"])
            if expert_cell is None or len(expert_cell) < 30:
                keys["expert"] = r["stratum"]
                base["expert"][r["stratum"]] = base["stratum"][r["stratum"]]
                gate["expert"][r["stratum"]] = gate["stratum"][r["stratum"]]
                expert_backoffs += 1
            for mi, mode in enumerate(MODES):
                key = keys[mode]
                bb = base[mode][key]
                gb = gate[mode][key]
                assert len(bb) and len(gb)
                bres = sample_array(bb, samples, 610000 + outer * 100000 + i * 8 + mi)
                gres = sample_array(gb, samples, 620000 + outer * 100000 + i * 8 + mi)
                bc = min(5.0, crps_fast(r["base"][None, :] + bres, r["truth"]) / r["scale"])
                gc = min(5.0, crps_fast(pred[None, :] + gres, r["truth"]) / r["scale"])
                fold_crps[mode]["base"].append(bc)
                fold_crps[mode]["gate"].append(gc)
                crps_all[mode]["base"].append(bc)
                crps_all[mode]["gate"].append(gc)
            bp = r["base_pair"]
            gp = r["arm_pairs"][j] if use else bp
            fold_base_pairs.append(bp)
            fold_gate_pairs.append(gp)

        all_base_pairs.extend(fold_base_pairs)
        all_gate_pairs.extend(fold_gate_pairs)
        fb, fg = summarize(fold_base_pairs), summarize(fold_gate_pairs)
        folds.append({
            "outer_fold": outer,
            "n_train": len(train),
            "n_valid": len(valid),
            "threshold": threshold,
            "coverage": float(np.mean([np.max(x) > threshold for x in pg])),
            "point_gain": {
                "sMAE": (fb["sMAE"] - fg["sMAE"]) / fb["sMAE"],
                "sRMSE": (fb["sRMSE"] - fg["sRMSE"]) / fb["sRMSE"],
            },
            "expert_backoffs": expert_backoffs,
            "crps": {
                mode: {
                    "base": float(np.mean(fold_crps[mode]["base"])),
                    "gate": float(np.mean(fold_crps[mode]["gate"])),
                    "gain": float((np.mean(fold_crps[mode]["base"]) -
                                   np.mean(fold_crps[mode]["gate"])) /
                                  np.mean(fold_crps[mode]["base"])),
                }
                for mode in MODES
            },
        })

    bp, gp = summarize(all_base_pairs), summarize(all_gate_pairs)
    return {
        "dataset": dataset,
        "protocol": "Train-only identical 3-fold outer CV point gate; residual-conditioning ablation",
        "n": len(rows),
        "point": {
            "base": bp,
            "gate": gp,
            "gains": {m: (bp[m] - gp[m]) / bp[m] for m in ("sMAE", "sRMSE")},
        },
        "crps": {
            mode: {
                "base": float(np.mean(crps_all[mode]["base"])),
                "gate": float(np.mean(crps_all[mode]["gate"])),
                "gain": float((np.mean(crps_all[mode]["base"]) -
                               np.mean(crps_all[mode]["gate"])) /
                              np.mean(crps_all[mode]["base"])),
                "fold_gains": [f["crps"][mode]["gain"] for f in folds],
            }
            for mode in MODES
        },
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("time_mmd", "timesx"), default="time_mmd")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("native_uncertainty_ablation_time_mmd.json"))
    args = ap.parse_args()
    out = evaluate(args.dataset, args.samples)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
