#!/usr/bin/env python3
"""Score mixed reconstruction v2 with the paper's native-quantile sCRPS rule.

The rule matches Table 1: 100 deterministic stratified trajectories, native
0.1--0.9 TSFM quantiles repaired around the exact cached median, endpoint
clamping, a shared quantile level across models and steps, empirical CRPS,
per-task scaling by mean absolute truth, cap 5, then an unweighted mean over
the 99 public-Test tasks.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QDIR = Path(os.environ.get("TS_DRCIK_QUANTILES", "data/native_scrps/drcik"))
TASKS = Path(os.environ.get("TS_DRCIK_TASKS", "data/Dr-CiK_public/tasks"))
FORECAST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_forecast_drcik.py"
ADJUST = Path(__file__).resolve().parents[3] / "experiments/self_evolving/final_method/unified/coevo_drcik/evolved_adjust_drcik.py"
VIEWS = HERE / "views_drcik_mixed_v2.json"
POINT = HERE / "predictions_drcik_mixed_v2.json"
REPAIRED_Q = HERE / "toto_repair_q_v2.json"
OUT = HERE / "scrps_mixed_v2.json"
LEVELS = np.arange(1, 10, dtype=float) / 10
S = 100


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def repaired_quantiles(raw, median, horizon):
    q = np.asarray(raw, dtype=float)[:, :horizon].copy()
    if q.shape != (9, horizon) or not np.isfinite(q).all():
        raise ValueError(f"bad quantile shape {q.shape}, expected {(9, horizon)}")
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
    scale = 1.0 / (np.abs(y).mean() + 1e-12)
    return min(5.0, float(scale * (first - second).mean()))


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    views = json.load(open(VIEWS))
    point = json.load(open(POINT))
    fq = json.load(open(QDIR / "toto_q.json"))["q"]
    tq = json.load(open(QDIR / "timesfm_q.json"))["q"]
    rq = json.load(open(REPAIRED_Q))["q"]
    fm, am = load_module(FORECAST, "mixed_fm"), load_module(ADJUST, "mixed_am")
    test_ids = [t for t, v in views.items() if v["part"] == "test"]
    assert len(test_ids) == 99 and set(test_ids).issubset(point["forecasts"])
    per_task = {}
    median_check = {}
    for tid in test_ids:
        original = views[tid]
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
            base = [float(x) for x in fm.forecast(view)]
            final = [float(x) for x in am.adjust(dict(view, base_forecast=base, toto_forecast=methods["toto_2_0"]))]
            assert len(final) == horizon and np.isfinite(final).all()
            paths.append(final)
        truth = json.load(open(TASKS / f"{tid}.json"))["series"]["future_values"]
        assert len(truth) == horizon
        per_task[tid] = scrps_task(paths, truth)
        # Diagnostic only: mean of the two central stratified paths versus
        # the frozen point prediction. Nonlinearity can make this nonzero.
        central = np.mean(np.asarray(paths)[49:51], axis=0)
        median_check[tid] = float(np.max(np.abs(central - np.asarray(point["forecasts"][tid]))))
    result = {
        "schema": "drcik-mixed-reconstruction-v2-scrps",
        "n": len(test_ids),
        "S": S,
        "sCRPS": float(np.mean(list(per_task.values()))),
        "per_task": per_task,
        "central_path_max_abs_diff": median_check,
        "provenance": {
            "views_sha256": sha(VIEWS),
            "point_predictions_sha256": sha(POINT),
            "forecast_program_sha256": sha(FORECAST),
            "adjust_program_sha256": sha(ADJUST),
            "toto_quantiles_sha256": sha(QDIR / "toto_q.json"),
            "timesfm_quantiles_sha256": sha(QDIR / "timesfm_q.json"),
            "repaired_toto_quantiles_sha256": sha(REPAIRED_Q),
            "scorer_sha256": sha(Path(__file__)),
        },
        "disclosure": "Exploratory public-Test-informed reconstruction v2; not sealed confirmation.",
    }
    json.dump(result, open(OUT, "x"), indent=1)
    print(json.dumps({"n": result["n"], "S": S, "sCRPS": result["sCRPS"],
                      "max_central_path_diff": max(median_check.values())}, indent=1))


if __name__ == "__main__":
    main()
