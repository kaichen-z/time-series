#!/usr/bin/env python3
"""Train-only outer-CV native empirical uncertainty for the Time-MMD risk gate.

For every outer fold, point routing and threshold selection use only outer-train
labels.  Uncertainty is an empirical residual distribution learned from
inner-OOF predictions on the outer-train rows.  Residual banks are conditioned
on frequency, horizon and selected expert, with a frequency/horizon backoff.
The held-out outer fold is used only once for scoring sMAE/sRMSE/sCRPS.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_statistical_ensemble import entity_key, fold_of, safe_forecasts
from evaluate_controller import pack_rows
from evaluate_v6_incumbent import incumbent_modules, run_pipeline
from risk_aware_statistical_gate import (features, make_model, pair,
                                         select_threshold, stat_experts,
                                         summarize)


def crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean univariate empirical CRPS across horizon."""
    a = np.mean(np.abs(samples - truth[None, :]), axis=0)
    # E|X-X'| computed horizon-wise; at most 100 samples.
    b = 0.5 * np.mean(np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1))
    return float(np.mean(a - b))


def build_rows(dataset: str):
    inc = incumbent_modules(dataset)
    rows = []
    for view, truth in pack_rows(dataset):
        truth = np.asarray(truth, float)
        raw = safe_forecasts(view)
        base = np.asarray(run_pipeline(inc, view), float)
        names, arms = stat_experts(view, raw)
        bp = pair(base, truth)
        aps = np.stack([pair(a, truth) for a in arms])
        rows.append({
            "x": features(view, base, raw), "truth": truth, "base": base,
            "arms": [np.asarray(a, float) for a in arms], "base_pair": bp,
            "arm_pairs": aps, "target": bp.sum() - aps.sum(axis=1),
            "fold": fold_of(view), "entity": entity_key(view),
            "stratum": f'{view.get("freq")}|H={view["H"]}', "names": names,
            "scale": float(np.mean(np.abs(truth))) + 1e-12,
        })
    return rows


def sample_bank(bank: list[np.ndarray], n: int, seed: int) -> np.ndarray:
    a = np.stack(bank)
    rng = np.random.default_rng(seed)
    if len(a) >= n:
        return a[rng.choice(len(a), n, replace=False)]
    return a[rng.choice(len(a), n, replace=True)]


def evaluate(dataset: str, samples: int = 100):
    rows = build_rows(dataset)
    X = np.stack([r["x"] for r in rows])
    Y = np.stack([r["target"] for r in rows])
    all_base_pairs, all_gate_pairs = [], []
    base_crps, gate_crps, folds = [], [], []
    usage = defaultdict(int)

    for outer in range(3):
        train = np.asarray([i for i, r in enumerate(rows) if r["fold"] != outer])
        valid = np.asarray([i for i, r in enumerate(rows) if r["fold"] == outer])
        inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
        inner_pred = np.zeros((len(train), Y.shape[1]))
        for inner in range(3):
            apos = np.where(inner_id != inner)[0]
            bpos = np.where(inner_id == inner)[0]
            if not len(apos) or not len(bpos):
                continue
            inner_pred[bpos] = make_model(5100 + outer * 10 + inner).fit(
                X[train[apos]], Y[train[apos]]).predict(X[train[bpos]])

        selected = select_threshold(inner_pred, rows, train)
        threshold = selected["threshold"]

        # Inner-OOF point choices yield honest outer-train residual banks.
        base_bank = defaultdict(list)
        gate_bank = defaultdict(list)
        gate_backoff = defaultdict(list)
        for k, i in enumerate(train):
            r = rows[i]
            j = int(np.argmax(inner_pred[k]))
            use = bool(inner_pred[k, j] > threshold)
            pred = r["arms"][j] if use else r["base"]
            name = r["names"][j] if use else "ABSTAIN_V6"
            base_bank[r["stratum"]].append(r["truth"] - r["base"])
            gate_bank[(r["stratum"], name)].append(r["truth"] - pred)
            gate_backoff[r["stratum"]].append(r["truth"] - pred)

        model = make_model(5200 + outer).fit(X[train], Y[train])
        pg = model.predict(X[valid])
        fold_base_pairs, fold_gate_pairs, fold_bc, fold_gc = [], [], [], []
        min_cond = 30
        for k, i in enumerate(valid):
            r = rows[i]
            j = int(np.argmax(pg[k]))
            use = bool(pg[k, j] > threshold)
            pred = r["arms"][j] if use else r["base"]
            name = r["names"][j] if use else "ABSTAIN_V6"
            usage[name] += 1
            gb = gate_bank[(r["stratum"], name)]
            if len(gb) < min_cond:
                gb = gate_backoff[r["stratum"]]
            bb = base_bank[r["stratum"]]
            assert bb and gb
            bres = sample_bank(bb, samples, 530000 + outer * 100000 + i)
            gres = sample_bank(gb, samples, 540000 + outer * 100000 + i)
            bc = min(5.0, crps(r["base"][None, :] + bres, r["truth"]) / r["scale"])
            gc = min(5.0, crps(pred[None, :] + gres, r["truth"]) / r["scale"])
            bp = r["base_pair"]
            gp = r["arm_pairs"][j] if use else bp
            fold_base_pairs.append(bp); fold_gate_pairs.append(gp)
            fold_bc.append(bc); fold_gc.append(gc)
        all_base_pairs.extend(fold_base_pairs); all_gate_pairs.extend(fold_gate_pairs)
        base_crps.extend(fold_bc); gate_crps.extend(fold_gc)
        fb, fg = summarize(fold_base_pairs), summarize(fold_gate_pairs)
        folds.append({
            "outer_fold": outer, "n_train": len(train), "n_valid": len(valid),
            "threshold": threshold, "coverage": float(np.mean([
                np.max(x) > threshold for x in pg])),
            "base": {**fb, "sCRPS": float(np.mean(fold_bc))},
            "gate": {**fg, "sCRPS": float(np.mean(fold_gc))},
            "gains": {
                "sMAE": (fb["sMAE"] - fg["sMAE"]) / fb["sMAE"],
                "sRMSE": (fb["sRMSE"] - fg["sRMSE"]) / fb["sRMSE"],
                "sCRPS": (np.mean(fold_bc) - np.mean(fold_gc)) / np.mean(fold_bc),
            },
        })

    b, g = summarize(all_base_pairs), summarize(all_gate_pairs)
    b["sCRPS"] = float(np.mean(base_crps)); g["sCRPS"] = float(np.mean(gate_crps))
    gains = {m: (b[m] - g[m]) / b[m] for m in ("sMAE", "sRMSE", "sCRPS")}
    return {
        "dataset": dataset,
        "protocol": "Train-only 3-fold outer CV; inner-OOF empirical residual uncertainty",
        "uncertainty": "frequency×horizon×selected-expert residual bank; frequency×horizon backoff when n<30",
        "samples": samples, "n": len(rows), "base": b, "gate": g,
        "gains": gains, "all_three_nonregressing": all(v >= 0 for v in gains.values()),
        "usage": dict(sorted(usage.items(), key=lambda kv: -kv[1])), "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("time_mmd", "timesx"), default="time_mmd")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("native_uncertainty_risk_gate.json"))
    args = ap.parse_args()
    out = evaluate(args.dataset, args.samples)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
