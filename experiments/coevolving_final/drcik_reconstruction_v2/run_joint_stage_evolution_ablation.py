#!/usr/bin/env python3
"""Replay the archived Dr-CiK joint-stage seed and final programs on v2 views.

This is a deterministic, zero-model-call comparison.  It isolates the
additional numerical/decision program evolution inside the archived joint
coevolution stage; the seed already contains programs evolved in earlier
stages, so this is not a full "without all coevolution" counterfactual.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import copy
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = HERE / "views_drcik_mixed_v2.json"
CACHED = HERE / "predictions_drcik_mixed_v2.json"
TASKS = Path(os.environ.get("TS_DRCIK_TASKS", "data/Dr-CiK_public/tasks"))
PROGRAMS = {
    "joint_stage_seed": (
        Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/seed_forecast.py",
        Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/seed_adjust.py",
    ),
    "joint_stage_evolved": (
        Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_forecast_drcik.py",
        Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_adjust_drcik.py",
    ),
}
OUT = HERE / "joint_stage_evolution_ablation_v2_point_scrps.json"
QDIR = Path(os.environ.get("TS_DRCIK_QUANTILES", "data/native_scrps/drcik"))
REPAIRED_Q = HERE / "toto_repair_q_v2.json"
LEVELS = np.arange(1, 10, dtype=float) / 10
S = 100


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score(predictions, views):
    grouped = {}
    for tid, pred in predictions.items():
        truth = np.asarray(json.load(open(TASKS / f"{tid}.json"))["series"]["future_values"], float)
        pred = np.asarray(pred, float)
        scale = float(np.abs(truth).mean()) + 1e-12
        smae = min(5.0, float(np.abs(pred - truth).mean()) / scale)
        srmse = min(5.0, math.sqrt(float(((pred - truth) ** 2).mean())) / scale)
        grouped.setdefault(views[tid]["part"], []).append((smae, srmse))
    return {
        part: {
            "n": len(rows),
            "sMAE": float(np.mean([r[0] for r in rows])),
            "sRMSE": float(np.mean([r[1] for r in rows])),
        }
        for part, rows in grouped.items()
    }


def repaired_quantiles(raw, median, horizon):
    q = np.asarray(raw, dtype=float)[:, :horizon].copy()
    assert q.shape == (9, horizon) and np.isfinite(q).all()
    q[4] = np.asarray(median, dtype=float)
    for j in range(5, 9):
        q[j] = np.maximum(q[j], q[j - 1])
    for j in range(3, -1, -1):
        q[j] = np.minimum(q[j], q[j + 1])
    return q


def inverse_cdf(q, u):
    if u <= LEVELS[0]:
        return q[0]
    if u >= LEVELS[-1]:
        return q[-1]
    j = int(np.searchsorted(LEVELS, u)) - 1
    w = (u - LEVELS[j]) / (LEVELS[j + 1] - LEVELS[j])
    return (1 - w) * q[j] + w * q[j + 1]


def scrps_task(paths, truth):
    x = np.sort(np.asarray(paths, dtype=float), axis=0)
    y = np.asarray(truth, dtype=float)
    n = x.shape[0]
    first = np.abs(x - y[None, :]).mean(axis=0)
    i = np.arange(1, n + 1)[:, None]
    second = (2 * ((2 * i - n - 1) * x).sum(axis=0)) / (2 * n * n)
    return min(5.0, float((first - second).mean() / (np.abs(y).mean() + 1e-12)))


def score_scrps(views, forecast, adjust, fq, tq, rq):
    grouped = {}
    for tid, original in views.items():
        if original["part"] not in ("dev", "test"):
            continue
        horizon = original["H"]
        q_toto = repaired_quantiles(fq[tid], original["method_forecasts"]["toto_2_0"], horizon)
        q_tfm = repaired_quantiles(tq[tid], original["method_forecasts"]["timesfm_2_5"], horizon)
        q_rep = None
        if original.get("toto_forecast_repaired_history") is not None:
            q_rep = repaired_quantiles(rq[f"{tid}|full"], original["toto_forecast_repaired_history"], horizon)
        paths = []
        for sample in range(S):
            u = (sample + 0.5) / S
            view = copy.copy(original)
            methods = copy.copy(original["method_forecasts"])
            methods["toto_2_0"] = inverse_cdf(q_toto, u).tolist()
            methods["timesfm_2_5"] = inverse_cdf(q_tfm, u).tolist()
            view["method_forecasts"] = methods
            if q_rep is not None:
                view["toto_forecast_repaired_history"] = inverse_cdf(q_rep, u).tolist()
            base = [float(x) for x in forecast.forecast(view)]
            paths.append([float(x) for x in adjust.adjust(dict(
                view, base_forecast=base, toto_forecast=methods["toto_2_0"]
            ))])
        truth = json.load(open(TASKS / f"{tid}.json"))["series"]["future_values"]
        grouped.setdefault(original["part"], []).append(scrps_task(paths, truth))
    return {part: {"n": len(vals), "sCRPS": float(np.mean(vals))} for part, vals in grouped.items()}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    views = json.load(open(VIEWS))
    cached = json.load(open(CACHED))["forecasts"]
    fq = json.load(open(QDIR / "toto_q.json"))["q"]
    tq = json.load(open(QDIR / "timesfm_q.json"))["q"]
    rq = json.load(open(REPAIRED_Q))["q"]
    predictions, results = {}, {}
    for arm, (forecast_path, adjust_path) in PROGRAMS.items():
        forecast = load(forecast_path, f"{arm}_forecast")
        adjust = load(adjust_path, f"{arm}_adjust")
        arm_predictions = {}
        for tid, view in views.items():
            base = [float(x) for x in forecast.forecast(view)]
            final = [float(x) for x in adjust.adjust(dict(
                view, base_forecast=base,
                toto_forecast=view["method_forecasts"]["toto_2_0"],
            ))]
            assert len(final) == view["H"] and np.isfinite(final).all()
            arm_predictions[tid] = final
        predictions[arm] = arm_predictions
        results[arm] = score(arm_predictions, views)
        for part, probabilistic in score_scrps(views, forecast, adjust, fq, tq, rq).items():
            results[arm][part].update(probabilistic)

    final_diffs = [
        float(np.max(np.abs(np.asarray(predictions["joint_stage_evolved"][tid]) - np.asarray(cached[tid]))))
        for tid in views
    ]
    seed_diffs = [
        float(np.max(np.abs(np.asarray(predictions["joint_stage_seed"][tid]) - np.asarray(predictions["joint_stage_evolved"][tid]))))
        for tid in views
    ]
    receipt = {
        "schema": "drcik-reconstruction-v2-joint-stage-evolution-ablation",
        "model_calls": 0,
        "scope": (
            "Same v2 views and metrics; compares the archived joint coevolution stage's unique "
            "0000 seed program pair with its final program pair. The seed already incorporates "
            "earlier-stage evolution, so this isolates additional joint-stage evolution only."
        ),
        "results": results,
        "checks": {
            "evolved_vs_cached_max_abs_diff": max(final_diffs),
            "evolved_vs_cached_changed_tasks_gt_1e-9": sum(d > 1e-9 for d in final_diffs),
            "seed_vs_evolved_changed_tasks_gt_1e-9": sum(d > 1e-9 for d in seed_diffs),
        },
        "provenance": {
            "views_sha256": sha(VIEWS),
            "cached_predictions_sha256": sha(CACHED),
            "toto_quantiles_sha256": sha(QDIR / "toto_q.json"),
            "timesfm_quantiles_sha256": sha(QDIR / "timesfm_q.json"),
            "repaired_toto_quantiles_sha256": sha(REPAIRED_Q),
            "seed_forecast_sha256": sha(PROGRAMS["joint_stage_seed"][0]),
            "seed_adjust_sha256": sha(PROGRAMS["joint_stage_seed"][1]),
            "evolved_forecast_sha256": sha(PROGRAMS["joint_stage_evolved"][0]),
            "evolved_adjust_sha256": sha(PROGRAMS["joint_stage_evolved"][1]),
            "script_sha256": sha(Path(__file__)),
        },
    }
    json.dump(receipt, open(OUT, "x"), indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
