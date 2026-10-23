#!/usr/bin/env python3
"""Train-only nested OOF document veto for TimesX numerical expert actions."""
from __future__ import annotations

import argparse
import datetime as dt
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


def parse_time(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def document_features(view) -> np.ndarray:
    events = []
    for doc in view.get("documents", []) or []:
        events.extend(doc.get("events", []) if isinstance(doc, dict) else [])
    cutoff = parse_time((view.get("history_timestamps") or [None])[-1])
    conf, signed, ages, lengths = [], [], [], []
    direction_map = {"up": 1.0, "increase": 1.0, "positive": 1.0,
                     "down": -1.0, "decrease": -1.0, "negative": -1.0,
                     "mixed": 0.0, "neutral": 0.0}
    for event in events:
        c = float(event.get("confidence", 0.0) or 0.0)
        d = direction_map.get(str(event.get("direction", "")).lower(), 0.0)
        conf.append(c); signed.append(c * d)
        lengths.append(len(str(event.get("evidence", "")).split()))
        when = parse_time(event.get("time_end") or event.get("time_start"))
        if cutoff is not None and when is not None:
            ages.append(max(0.0, (cutoff - when).total_seconds() / 86400.0))
    if not events:
        return np.zeros(12, float)
    return np.asarray([
        np.log1p(len(events)), np.mean(conf) if conf else 0.0,
        np.max(conf) if conf else 0.0, np.mean(signed) if signed else 0.0,
        np.sum(signed) / max(np.sum(conf), 1e-8),
        np.mean(np.asarray(signed) > 0), np.mean(np.asarray(signed) < 0),
        np.mean(np.asarray(signed) == 0),
        np.log1p(min(ages)) if ages else 0.0,
        np.log1p(np.mean(ages)) if ages else 0.0,
        np.log1p(max(ages)) if ages else 0.0,
        np.log1p(np.mean(lengths)) if lengths else 0.0,
    ], float)


def build_rows():
    inc = incumbent_modules("timesx")
    rows = []
    for view, truth in pack_rows("timesx"):
        truth = np.asarray(truth, float)
        raw = safe_forecasts(view)
        base = np.asarray(run_pipeline(inc, view), float)
        names, arms = stat_experts(view, raw)
        bp = pair(base, truth)
        aps = np.stack([pair(a, truth) for a in arms])
        rows.append({
            "x": features(view, base, raw), "doc": document_features(view),
            "base": base, "arms": [np.asarray(a, float) for a in arms],
            "base_pair": bp, "arm_pairs": aps,
            "target": bp.sum() - aps.sum(axis=1), "names": names,
            "fold": fold_of(view), "entity": entity_key(view),
            "stratum": f'{view.get("freq")}|H={view["H"]}',
        })
    return rows


def action_features(row, predicted_gain, arm, doc=None):
    delta = (row["arms"][arm] - row["base"])
    scale = np.mean(np.abs(row["base"])) + 1e-8
    onehot = np.zeros(len(row["arms"]), float); onehot[arm] = 1.0
    action = np.asarray([
        predicted_gain[arm], np.max(predicted_gain),
        np.mean(delta) / scale, np.mean(np.abs(delta)) / scale,
        np.max(np.abs(delta)) / scale,
    ])
    parts = [row["x"], action, onehot]
    if doc is not None:
        direction = np.sign(np.mean(delta))
        alignment = np.asarray([direction * doc[3], direction * doc[4]])
        parts.extend([doc, alignment])
    return np.concatenate(parts)


def gain(base_pairs, candidate_pairs):
    b, c = summarize(base_pairs), summarize(candidate_pairs)
    return {
        "score": c,
        "gains": {m: (b[m] - c[m]) / b[m] for m in ("sMAE", "sRMSE", "joint")},
    }


def evaluate():
    rows = build_rows()
    X = np.stack([r["x"] for r in rows])
    Y = np.stack([r["target"] for r in rows])
    conditions = ("numerical_gate", "numeric_veto", "real_doc_veto", "shuffled_doc_veto")
    all_pairs = {c: [] for c in conditions}
    all_base, folds = [], []

    for outer in range(3):
        train = np.asarray([i for i, r in enumerate(rows) if r["fold"] != outer])
        valid = np.asarray([i for i, r in enumerate(rows) if r["fold"] == outer])
        inner_id = np.asarray([int(rows[i]["entity"][8:16], 16) % 3 for i in train])
        inner_pred = np.zeros((len(train), Y.shape[1]))
        for inner in range(3):
            apos = np.where(inner_id != inner)[0]
            bpos = np.where(inner_id == inner)[0]
            if len(apos) and len(bpos):
                inner_pred[bpos] = make_model(8100 + outer * 10 + inner).fit(
                    X[train[apos]], Y[train[apos]]).predict(X[train[bpos]])
        selected = select_threshold(inner_pred, rows, train)
        threshold = selected["threshold"]

        # Honest inner-OOF numerical actions become veto training examples.
        veto_ids, veto_y, veto_num, veto_doc = [], [], [], []
        for k, i in enumerate(train):
            j = int(np.argmax(inner_pred[k]))
            if inner_pred[k, j] > threshold:
                veto_ids.append(i)
                veto_y.append(rows[i]["target"][j])
                veto_num.append(action_features(rows[i], inner_pred[k], j))
                veto_doc.append(action_features(rows[i], inner_pred[k], j, rows[i]["doc"]))
        veto_y = np.asarray(veto_y)
        numeric_model = make_model(8200 + outer).fit(np.stack(veto_num), veto_y)
        doc_model = make_model(8300 + outer).fit(np.stack(veto_doc), veto_y)

        pg = make_model(8400 + outer).fit(X[train], Y[train]).predict(X[valid])
        rng = np.random.default_rng(8500 + outer)
        shuffled_doc = np.stack([rows[i]["doc"] for i in valid]).copy()
        # Wrong-document control preserves stratum-level document marginals.
        for stratum in sorted({rows[i]["stratum"] for i in valid}):
            pos = np.asarray([k for k, i in enumerate(valid) if rows[i]["stratum"] == stratum])
            shuffled_doc[pos] = shuffled_doc[rng.permutation(pos)]

        fold_base = [rows[i]["base_pair"] for i in valid]
        fold_pairs = {c: [] for c in conditions}
        counts = defaultdict(int)
        for k, i in enumerate(valid):
            r = rows[i]
            j = int(np.argmax(pg[k]))
            proposed = bool(pg[k, j] > threshold)
            bp, ap = r["base_pair"], r["arm_pairs"][j]
            fold_pairs["numerical_gate"].append(ap if proposed else bp)
            if not proposed:
                for c in ("numeric_veto", "real_doc_veto", "shuffled_doc_veto"):
                    fold_pairs[c].append(bp)
                continue
            counts["proposed"] += 1
            pn = float(numeric_model.predict(action_features(r, pg[k], j)[None, :])[0])
            pr = float(doc_model.predict(action_features(r, pg[k], j, r["doc"])[None, :])[0])
            ps = float(doc_model.predict(action_features(r, pg[k], j, shuffled_doc[k])[None, :])[0])
            decisions = {"numeric_veto": pn > 0, "real_doc_veto": pr > 0,
                         "shuffled_doc_veto": ps > 0}
            for c, keep in decisions.items():
                fold_pairs[c].append(ap if keep else bp)
                counts[f"{c}_kept"] += int(keep)
        all_base.extend(fold_base)
        for c in conditions:
            all_pairs[c].extend(fold_pairs[c])
        folds.append({
            "outer_fold": outer, "n_train": len(train), "n_valid": len(valid),
            "threshold": threshold, "counts": dict(counts),
            "results": {c: gain(fold_base, fold_pairs[c]) for c in conditions},
        })

    base = summarize(all_base)
    results = {c: gain(all_base, all_pairs[c]) for c in conditions}
    return {
        "dataset": "timesx", "n": len(rows),
        "protocol": "Train-only 3-fold outer CV; numerical proposal + OOF-trained veto",
        "document_control": "outer-valid documents permuted within stratum; same real-doc veto model",
        "base": base, "results": results,
        "document_increment_over_numeric_veto": {
            m: results["real_doc_veto"]["gains"][m] - results["numeric_veto"]["gains"][m]
            for m in ("sMAE", "sRMSE", "joint")
        },
        "document_increment_over_shuffled": {
            m: results["real_doc_veto"]["gains"][m] - results["shuffled_doc_veto"]["gains"][m]
            for m in ("sMAE", "sRMSE", "joint")
        },
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("timesx_document_veto_oof.json"))
    args = ap.parse_args()
    out = evaluate()
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
