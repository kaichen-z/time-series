#!/usr/bin/env python3
"""Train-only evaluator for three-module candidates against locked v6 incumbents."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

from evaluate_controller import pack_rows

HERE = Path(__file__).resolve().parent
# Frozen v6 programs (time_mmd_recovery_20261011T0544Z / timesx_recovery_20261011T0421Z), vendored here.
INCUMBENT = {
    "time_mmd": HERE / "v6_incumbent/time_mmd",
    "timesx": HERE / "v6_incumbent/timesx",
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def modules(root: Path, tag: str):
    return (
        load(root / "forecast.py", tag + "_forecast"),
        load(root / "retrieve.py", tag + "_retrieve"),
        load(root / "adjust.py", tag + "_adjust"),
    )


def incumbent_modules(dataset: str):
    root = INCUMBENT[dataset]
    return (
        load(root / "final_forecast.py", dataset + "_inc_forecast"),
        load(root / "final_retrieve.py", dataset + "_inc_retrieve"),
        load(root / "final_adjust.py", dataset + "_inc_adjust"),
    )


def run_pipeline(mods, view):
    forecast, retrieve, adjust = mods
    horizon = int(view["H"])
    toto = [float(x) for x in view["method_forecasts"]["toto_2_0"]]
    try:
        base = [float(x) for x in forecast.forecast(view)]
        if len(base) != horizon or not all(math.isfinite(x) for x in base):
            raise ValueError("invalid base")
    except Exception:
        base = toto
    try:
        corrections = []
        for item in retrieve.retrieve(view) or []:
            start, end, multiplier = int(item["start"]), int(item["end"]), float(item["multiplier"])
            if 0 <= start < end <= horizon and multiplier >= 0 and math.isfinite(multiplier):
                corrections.append(dict(item, start=start, end=end, multiplier=multiplier))
    except Exception:
        corrections = []
    try:
        final = [float(x) for x in adjust.adjust(dict(
            view, base_forecast=base, toto_forecast=toto, corrections=corrections,
        ))]
        if len(final) != horizon or not all(math.isfinite(x) for x in final):
            raise ValueError("invalid final")
    except Exception:
        final = base
    return final


def metrics(prediction, truth):
    scale = statistics.mean(abs(float(x)) for x in truth) + 1e-12
    errors = [float(a) - float(b) for a, b in zip(prediction, truth)]
    smae = min(5.0, statistics.mean(abs(x) for x in errors) / scale)
    srmse = min(5.0, math.sqrt(statistics.mean(x * x for x in errors)) / scale)
    return smae, srmse


def fold_of(view) -> int:
    payload = json.dumps({
        "freq": view["freq"], "H": view["H"], "target": view.get("target_description", ""),
        "history_n": len(view["history"]), "history_head": view["history"][:3], "history_tail": view["history"][-3:],
    }, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % 3


def evaluate(root: Path, dataset: str):
    candidate = modules(root, dataset + "_candidate")
    incumbent = incumbent_modules(dataset)
    groups = {fold: [] for fold in range(3)}
    for view, truth in pack_rows(dataset):
        old = metrics(run_pipeline(incumbent, view), truth)
        new = metrics(run_pipeline(candidate, view), truth)
        groups[fold_of(view)].append((old, new))

    def summarize(rows):
        old_mae = statistics.mean(row[0][0] for row in rows)
        old_rmse = statistics.mean(row[0][1] for row in rows)
        new_mae = statistics.mean(row[1][0] for row in rows)
        new_rmse = statistics.mean(row[1][1] for row in rows)
        old_joint, new_joint = old_mae + old_rmse, new_mae + new_rmse
        return {
            "n": len(rows), "incumbent_sMAE": old_mae, "candidate_sMAE": new_mae,
            "sMAE_gain": old_mae - new_mae, "incumbent_sRMSE": old_rmse,
            "candidate_sRMSE": new_rmse, "sRMSE_gain": old_rmse - new_rmse,
            "incumbent_joint": old_joint, "candidate_joint": new_joint,
            "standardized_gain": (old_joint - new_joint) / (old_joint + 1e-12),
        }

    all_rows = [row for fold in groups.values() for row in fold]
    overall = summarize(all_rows)
    folds = {str(fold): summarize(rows) for fold, rows in groups.items()}
    feasible = (overall["sMAE_gain"] >= -1e-12 and overall["sRMSE_gain"] >= -1e-12
                and min(row["standardized_gain"] for row in folds.values()) >= -0.0005)
    return {"dataset": dataset, "overall": overall, "folds": folds, "feasible": feasible}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dataset", choices=INCUMBENT, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.candidate, args.dataset), indent=2))
