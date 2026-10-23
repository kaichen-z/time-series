#!/usr/bin/env python3
"""Export task-level Dr-CiK v2 Decision effects for the paper scatter plot.

The pre-adjust numerical base and post-adjust final forecast use the same
frozen programs as the reconstruction-v2 ablation.  Decision state is kept at
the observable task level: ``applied`` if the frozen adjust program changes at
least one prediction, otherwise ``no_intervention``.  The program does not
invent card-level accept/weaken/reject labels that were not logged.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = HERE / "views_drcik_mixed_v2.json"
TASKS = Path(os.environ.get("TS_DRCIK_TASKS", "data/Dr-CiK_public/tasks"))
FORECAST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_forecast_drcik.py"
ADJUST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_adjust_drcik.py"
OUT = HERE / "decision_task_scatter_dev_test_v2.csv"
RECEIPT = HERE / "decision_task_scatter_dev_test_v2_receipt.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def errors(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    scale = float(np.abs(truth).mean()) + 1e-12
    smae = min(5.0, float(np.abs(pred - truth).mean()) / scale)
    srmse = min(5.0, math.sqrt(float(((pred - truth) ** 2).mean())) / scale)
    return smae, srmse


def main():
    if OUT.exists() or RECEIPT.exists():
        raise SystemExit("refusing to overwrite existing export")
    views = json.load(open(VIEWS))
    fm, am = load(FORECAST, "scatter_fm"), load(ADJUST, "scatter_am")
    rows = []
    for task_id, view in views.items():
        if view["part"] not in {"dev", "test"}:
            continue
        toto = np.asarray(view["method_forecasts"]["toto_2_0"], float)
        base = np.asarray(fm.forecast(view), float)
        final = np.asarray(am.adjust(dict(
            view, base_forecast=base.tolist(), toto_forecast=toto.tolist()
        )), float)
        truth = np.asarray(json.load(open(TASKS / f"{task_id}.json"))["series"]["future_values"], float)
        assert len(base) == len(final) == len(truth) == view["H"]
        assert np.isfinite(base).all() and np.isfinite(final).all()
        toto_smae, toto_srmse = errors(toto, truth)
        base_smae, base_srmse = errors(base, truth)
        final_smae, final_srmse = errors(final, truth)
        max_change = float(np.max(np.abs(final - base)))
        state = "applied" if max_change > 1e-9 else "no_intervention"
        rows.append({
            "split": view["part"],
            "task_id": task_id,
            "horizon": view["H"],
            "card_count": len(view.get("corrections") or []),
            "decision_state": state,
            "toto_smae": toto_smae,
            "toto_srmse": toto_srmse,
            "toto_joint_error": toto_smae + toto_srmse,
            "numerical_base_smae": base_smae,
            "numerical_base_srmse": base_srmse,
            "numerical_base_joint_error": base_smae + base_srmse,
            "final_smae": final_smae,
            "final_srmse": final_srmse,
            "final_joint_error": final_smae + final_srmse,
            "delta_smae": final_smae - base_smae,
            "delta_srmse": final_srmse - base_srmse,
            "delta_joint_error": (final_smae + final_srmse) - (base_smae + base_srmse),
            "improved_smae": int(final_smae < base_smae - 1e-12),
            "max_abs_prediction_change": max_change,
        })
    rows.sort(key=lambda row: (row["split"], row["task_id"]))
    with open(OUT, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_split = {}
    for split in ("dev", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        applied = [r for r in split_rows if r["decision_state"] == "applied"]
        unchanged = [r for r in split_rows if r["decision_state"] == "no_intervention"]
        by_split[split] = {
            "n": len(split_rows),
            "applied": len(applied),
            "no_intervention": len(unchanged),
            "applied_improved_smae": sum(r["improved_smae"] for r in applied),
            "applied_worsened_smae": sum(r["delta_smae"] > 1e-12 for r in applied),
            "applied_mean_base_smae": float(np.mean([r["numerical_base_smae"] for r in applied])),
            "applied_mean_final_smae": float(np.mean([r["final_smae"] for r in applied])),
            "unchanged_max_abs_prediction_change": max(r["max_abs_prediction_change"] for r in unchanged),
        }
    receipt = {
        "schema": "drcik-v2-decision-task-scatter",
        "scope": "Dev and public Test tasks; frozen v2 programs; zero model calls",
        "n": len(rows),
        "by_split": by_split,
        "checks": {
            "all_rows_finite": all(math.isfinite(float(v)) for r in rows for k, v in r.items()
                                     if k not in {"split", "task_id", "decision_state"}),
        },
        "provenance": {
            "views_sha256": sha(VIEWS),
            "forecast_sha256": sha(FORECAST),
            "adjust_sha256": sha(ADJUST),
            "csv_sha256": sha(OUT),
            "script_sha256": sha(Path(__file__)),
        },
        "note": "Use numerical_base versus final to isolate Decision; Toto columns are reference only.",
    }
    with open(RECEIPT, "x") as handle:
        json.dump(receipt, handle, indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
