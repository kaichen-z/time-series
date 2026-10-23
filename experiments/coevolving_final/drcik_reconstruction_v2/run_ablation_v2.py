#!/usr/bin/env python3
"""Deterministic component ablations for the Dr-CiK reconstruction-v2 pipeline.

All variants reuse the exact v2 views, cached TSFM forecasts, frozen forecast
and adjust programs, and official task files.  There are no model calls.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = HERE / "views_drcik_mixed_v2.json"
POINT = HERE / "predictions_drcik_mixed_v2.json"
TASKS = Path(os.environ.get("TS_DRCIK_TASKS", "data/Dr-CiK_public/tasks"))
FORECAST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_forecast_drcik.py"
ADJUST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_adjust_drcik.py"
OUT = HERE / "ablation_v2.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def raw_corrections(base, view):
    """Historical no-validator control: apply every card, clip once to +/-50%."""
    base = np.asarray(base, dtype=float)
    mult = np.ones(view["H"], dtype=float)
    for card in view.get("corrections") or []:
        start, end = int(card["start"]), int(card["end"])
        if 0 <= start < end <= view["H"] and math.isfinite(float(card["multiplier"])):
            mult[start:end] *= float(card["multiplier"])
    proposed = base * mult
    return (base + np.clip(proposed - base, -0.5 * np.abs(base), 0.5 * np.abs(base))).tolist()


def predict(view, fm, am, variant):
    raw_toto = view["method_forecasts"]["toto_2_0"]
    if variant == "without_model_combination":
        base = list(view.get("toto_forecast_repaired_history") or raw_toto)
    else:
        base = [float(x) for x in fm.forecast(view)]

    if variant == "without_document_corrections":
        final = base
    elif variant == "without_validator":
        final = raw_corrections(base, view)
    else:
        final = [float(x) for x in am.adjust(dict(
            view, base_forecast=base, toto_forecast=raw_toto))]
    assert len(final) == view["H"] and np.isfinite(final).all()
    return final, base


def metrics(predictions, views):
    by_part = {}
    for tid, pred in predictions.items():
        truth = np.asarray(json.load(open(TASKS / f"{tid}.json"))["series"]["future_values"], float)
        pred = np.asarray(pred, float)
        scale = float(np.abs(truth).mean()) + 1e-12
        smae = min(5.0, float(np.abs(pred - truth).mean()) / scale)
        srmse = min(5.0, math.sqrt(float(((pred - truth) ** 2).mean())) / scale)
        by_part.setdefault(views[tid]["part"], []).append((smae, srmse))
    return {part: {"n": len(rows),
                   "sMAE": float(np.mean([x[0] for x in rows])),
                   "sRMSE": float(np.mean([x[1] for x in rows]))}
            for part, rows in by_part.items()}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    views = json.load(open(VIEWS))
    cached = json.load(open(POINT))
    fm, am = load(FORECAST, "ablation_fm"), load(ADJUST, "ablation_am")
    variants = ["full", "without_history_repair", "without_model_combination",
                "without_document_corrections", "without_validator"]
    results, checks = {}, {}
    full_predictions = None
    for variant in variants:
        predictions = {}
        for tid, original in views.items():
            view = dict(original)
            if variant == "without_history_repair":
                view["toto_forecast_repaired_history"] = None
            predictions[tid], _ = predict(view, fm, am, variant)
        results[variant] = metrics(predictions, views)
        if variant == "full":
            full_predictions = predictions
            diffs = [float(np.max(np.abs(np.asarray(predictions[t]) -
                                               np.asarray(cached["forecasts"][t])))) for t in views]
            checks["full_vs_cached_max_abs_diff"] = max(diffs)
            checks["full_vs_cached_changed_tasks_gt_1e-9"] = sum(x > 1e-9 for x in diffs)
        else:
            changed = [float(np.max(np.abs(np.asarray(predictions[t]) -
                                                np.asarray(full_predictions[t])))) for t in views]
            checks[f"{variant}_changed_tasks_gt_1e-9"] = sum(x > 1e-9 for x in changed)
    receipt = {
        "schema": "drcik-reconstruction-v2-component-ablation",
        "model_calls": 0,
        "disclosure": "Exploratory public-Test-informed reconstruction v2; not sealed confirmation.",
        "definitions": {
            "full": "frozen v2 forecast and adjust programs",
            "without_history_repair": "set repaired-history Toto forecast to null",
            "without_model_combination": "use repaired Toto if accepted, otherwise raw Toto, as base; retain frozen document adjust",
            "without_document_corrections": "retain frozen numerical forecast, bypass the document adjust program",
            "without_validator": "apply all cached correction multipliers and clip once to +/-50% of base",
            "without_coevolution": "not run: no unique current-v2 counterfactual program pair is defined",
        },
        "provenance": {"views_sha256": sha(VIEWS), "cached_predictions_sha256": sha(POINT),
                       "forecast_program_sha256": sha(FORECAST), "adjust_program_sha256": sha(ADJUST),
                       "script_sha256": sha(Path(__file__))},
        "checks": checks,
        "results": results,
    }
    json.dump(receipt, open(OUT, "x"), indent=1)
    print(json.dumps({"checks": checks, "results": results}, indent=1))


if __name__ == "__main__":
    main()
