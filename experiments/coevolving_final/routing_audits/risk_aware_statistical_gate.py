#!/usr/bin/env python3
"""Train-only nested grouped-CV risk gate over v6 and statistical experts.

The gate observes history statistics and forecast disagreement only.  Labels are
used exclusively inside grouped Train folds to predict each expert's advantage
over the locked v6 incumbent.  A confidence threshold permits abstention back
to v6.  Dev/Test are never opened by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from analyze_statistical_ensemble import MODELS, entity_key, fold_of, safe_forecasts
from evaluate_controller import pack_rows
from evaluate_v6_incumbent import incumbent_modules, run_pipeline


def corr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag or np.std(x[:-lag]) < 1e-12 or np.std(x[lag:]) < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def stat_experts(view, raw: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    h = np.asarray(view["history"], float); H = int(view["H"])
    last = np.full(H, h[-1])
    recent = np.full(H, np.mean(h[-min(12, len(h)):]))
    n = min(32, len(h)); slope = float(np.polyfit(np.arange(n), h[-n:], 1)[0]) if n > 1 else 0.0
    drift = h[-1] + slope * np.arange(1, H + 1)
    damped = h[-1] + slope * np.cumsum(0.85 ** np.arange(1, H + 1))
    f = str(view.get("freq", "")).upper()
    period = 7 if f in ("D", "1D", "DAILY") else 52 if f in ("W", "1W", "WEEKLY") else 12
    seasonal = last if len(h) < period else np.asarray([h[-period + t % period] for t in range(H)])
    names = ["last", "recent_mean", "drift", "damped_trend", "seasonal", *MODELS]
    values = [last, recent, drift, damped, seasonal, *[raw[:, j] for j in range(raw.shape[1])]]
    return names, values


def features(view, inc: np.ndarray, raw: np.ndarray) -> np.ndarray:
    h = np.asarray(view["history"], float); H = int(view["H"])
    scale = float(np.mean(np.abs(h))) + 1e-8
    d = np.diff(h); n = min(32, len(h)); slope = float(np.polyfit(np.arange(n), h[-n:], 1)[0]) if n > 1 else 0.0
    recent = float(np.mean(h[-min(12, len(h)):]))
    freq = str(view.get("freq", "")).upper()
    freq_bits = [float(freq in group) for group in (("D", "1D", "DAILY"), ("W", "1W", "WEEKLY"), ("H", "1H", "HOURLY"))]
    base = [
        np.log1p(len(h)), np.log1p(H), *freq_bits,
        float(np.mean(h) / scale), float(np.std(h) / scale), float(h[-1] / scale),
        float((h[-1] - recent) / scale), float(slope / scale),
        float(np.std(d) / scale) if len(d) else 0.0,
        corr(h, 1), corr(h, 7), corr(h, 12), corr(h, 52),
        float(np.mean(np.isclose(h, 0))),
        float(np.mean(np.sign(d[1:]) != np.sign(d[:-1]))) if len(d) > 2 else 0.0,
    ]
    means = np.mean(raw, axis=0) / scale
    slopes = (raw[-1] - raw[0]) / max(H - 1, 1) / scale
    spread = np.std(raw, axis=1) / scale
    base.extend(means.tolist()); base.extend(slopes.tolist())
    base.extend([float(np.mean(spread)), float(np.max(spread)),
                 float(np.mean(np.abs(inc - h[-1])) / scale),
                 float(np.mean(np.abs(inc - np.mean(h))) / scale)])
    return np.nan_to_num(np.asarray(base, float), nan=0.0, posinf=10.0, neginf=-10.0)


def pair(pred, truth):
    scale = float(np.mean(np.abs(truth))) + 1e-12
    z = np.asarray(pred) - truth
    return np.asarray([min(5., np.mean(np.abs(z)) / scale),
                       min(5., np.sqrt(np.mean(z*z)) / scale)])


def summarize(pairs):
    a = np.asarray(pairs, float)
    return {"n": len(a), "sMAE": float(a[:, 0].mean()), "sRMSE": float(a[:, 1].mean()),
            "joint": float(a.sum(1).mean())}


def make_model(seed):
    return ExtraTreesRegressor(n_estimators=96, min_samples_leaf=24, max_features=0.75,
                               random_state=seed, n_jobs=-1)


def select_threshold(pred_gain, rows, indices):
    base_pairs = [rows[i]["base_pair"] for i in indices]
    base = summarize(base_pairs)
    positive = np.maximum(pred_gain.max(axis=1), 0)
    candidates = sorted(set([0.0, *np.quantile(positive, [0.25, .4, .5, .6, .7, .8, .9, .95]).tolist()]))
    best = (base["joint"], float("inf"), base, 0.0)
    for threshold in candidates:
        pairs, group = [], defaultdict(lambda: [[], []])
        for k, i in enumerate(indices):
            j = int(np.argmax(pred_gain[k])); use = pred_gain[k, j] > threshold
            p = rows[i]["arm_pairs"][j] if use else rows[i]["base_pair"]
            pairs.append(p); group[rows[i]["stratum"]][0 if use else 1].append(p)
        score = summarize(pairs)
        # Require both global point metrics non-regressing.  Avoid a router that
        # buys its mean gain through >1% relative damage in any frequency/H group.
        safe = score["sMAE"] <= base["sMAE"] and score["sRMSE"] <= base["sRMSE"]
        for s in {rows[i]["stratum"] for i in indices}:
            ids = [i for i in indices if rows[i]["stratum"] == s]
            sb = summarize([rows[i]["base_pair"] for i in ids])
            sp = summarize([pairs[k] for k, i in enumerate(indices) if rows[i]["stratum"] == s])
            safe &= sp["joint"] <= sb["joint"] * 1.01 + 1e-12
        coverage = float(np.mean(pred_gain.max(axis=1) > threshold))
        if safe and score["joint"] < best[0]:
            best = (score["joint"], threshold, score, coverage)
    return {"threshold": best[1] if np.isfinite(best[1]) else 1e9,
            "inner_score": best[2], "inner_coverage": best[3], "inner_base": base}


def evaluate(dataset):
    inc = incumbent_modules(dataset); rows = []
    for view, truth in pack_rows(dataset):
        truth = np.asarray(truth, float); raw = safe_forecasts(view); base = np.asarray(run_pipeline(inc, view), float)
        names, arms = stat_experts(view, raw)
        bp = pair(base, truth); aps = np.stack([pair(a, truth) for a in arms])
        rows.append({"x": features(view, base, raw), "base_pair": bp, "arm_pairs": aps,
                     "target": bp.sum() - aps.sum(axis=1), "fold": fold_of(view),
                     "entity": entity_key(view), "stratum": f'{view.get("freq")}|H={view["H"]}',
                     "names": names})
    X = np.stack([r["x"] for r in rows]); Y = np.stack([r["target"] for r in rows])
    base_all, gate_all, receipts, usage = [], [], [], defaultdict(int)
    output_rows = []
    for outer in range(3):
        train = np.asarray([i for i,r in enumerate(rows) if r["fold"] != outer]); valid = np.asarray([i for i,r in enumerate(rows) if r["fold"] == outer])
        inner_pred = np.zeros((len(train), Y.shape[1]))
        inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
        for inner in range(3):
            a = train[inner_id != inner]; bpos = np.where(inner_id == inner)[0]; b = train[bpos]
            if not len(a) or not len(b): continue
            inner_pred[bpos] = make_model(1000 + outer*10 + inner).fit(X[a], Y[a]).predict(X[b])
        selected = select_threshold(inner_pred, rows, train)
        model = make_model(2000 + outer).fit(X[train], Y[train]); pg = model.predict(X[valid])
        chosen = []
        for k, i in enumerate(valid):
            j = int(np.argmax(pg[k])); use = pg[k,j] > selected["threshold"]
            base_all.append(rows[i]["base_pair"])
            gate_all.append(rows[i]["arm_pairs"][j] if use else rows[i]["base_pair"])
            name = rows[i]["names"][j] if use else "ABSTAIN_V6"
            usage[name] += 1; chosen.append(name)
            output_rows.append((outer, rows[i]["stratum"], rows[i]["base_pair"],
                                rows[i]["arm_pairs"][j] if use else rows[i]["base_pair"]))
        fold_base = summarize([x[2] for x in output_rows if x[0] == outer])
        fold_gate = summarize([x[3] for x in output_rows if x[0] == outer])
        receipts.append({"outer_fold": outer, "n_train": len(train), "n_valid": len(valid),
                         **selected, "outer_coverage": float(np.mean(np.asarray(chosen) != "ABSTAIN_V6")),
                         "outer_base": fold_base, "outer_gate": fold_gate,
                         "outer_standardized_gain": (fold_base["joint"]-fold_gate["joint"])/fold_base["joint"]})
    base, gate = summarize(base_all), summarize(gate_all)
    by_stratum = {}
    for s in sorted({x[1] for x in output_rows}):
        sb = summarize([x[2] for x in output_rows if x[1] == s]); sg = summarize([x[3] for x in output_rows if x[1] == s])
        by_stratum[s] = {"base": sb, "gate": sg,
                         "standardized_gain": (sb["joint"]-sg["joint"])/sb["joint"]}
    return {"dataset": dataset, "protocol": "Train-only 3x3 nested document/entity grouped CV",
            "features": "history statistics + model disagreement only", "n": len(rows),
            "base": base, "risk_gate": gate,
            "standardized_joint_gain": (base["joint"]-gate["joint"])/base["joint"],
            "usage": dict(sorted(usage.items(), key=lambda kv: -kv[1])),
            "by_stratum": by_stratum, "fold_receipts": receipts}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=("timesx","time_mmd","all"), default="all")
    ap.add_argument("--out", type=Path, default=Path("risk_aware_statistical_gate.json")); args=ap.parse_args()
    ds=("timesx","time_mmd") if args.dataset=="all" else (args.dataset,)
    out={d:evaluate(d) for d in ds}; args.out.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))


if __name__ == "__main__": main()
