#!/usr/bin/env python3
"""Host-only equal-dataset evaluator for a single joint ForecastController."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import statistics
import sys
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
V5 = HERE  # viewstore.py is vendored next to this file
DR = Path(os.environ.get("TS_DRCIK_AUX_ROOT", "data/drcik_aux"))
sys.path.insert(0, str(V5))
import viewstore  # noqa: E402


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint(pred, truth):
    if len(pred) != len(truth) or not all(math.isfinite(float(x)) for x in pred):
        raise ValueError("forecast must have the exact horizon and finite values")
    scale = statistics.mean(abs(float(x)) for x in truth) + 1e-12
    diff = [float(a) - float(b) for a, b in zip(pred, truth)]
    smae = min(5.0, statistics.mean(abs(x) for x in diff) / scale)
    srmse = min(5.0, math.sqrt(statistics.mean(x * x for x in diff)) / scale)
    return smae + srmse


def pack_rows(dataset: str):
    with open(HERE / "private" / f"{dataset}_train.pkl", "rb") as handle:
        yield from pickle.load(handle)


def drcik_rows():
    root = DR / "drcik_full/dry_mixed"
    views = json.loads((root / "s2/views_train.json").read_text())
    events = json.loads((root / "events/anomalies_train.json").read_text())
    truth = json.loads((root / "truth_train.json").read_text())
    raw = {name: json.loads((DR / f"native_scrps/drcik/{short}_q.json").read_text())["q"]
           for name, short in (("toto_2_0", "toto"), ("timesfm_2_5", "timesfm"),
                               ("chronos_bolt", "chronos"), ("moirai_2_0", "moirai"))}
    masked = json.loads((HERE / "artifacts/drcik_train_reliability/forecast/timesfm_q.json").read_text())["q"]
    masks = json.loads((HERE / "artifacts/drcik_train_reliability/masks.json").read_text())
    for tid in sorted(views):
        view = dict(views[tid])
        view["dataset"] = "drcik"
        view["method_forecasts"] = {name: values[tid][4] for name, values in raw.items()}
        view["method_forecasts"]["timesfm_2_5_reliability"] = masked[tid][4]
        view["reliability_fraction"] = masks[tid]["mask_fraction"]
        view["reliability_available"] = bool(masks[tid]["n_masked"])
        view["documents"] = [{"events": events[tid]}]
        yield view, truth[tid]


def evaluate(path: Path):
    candidate = load_module(path)
    seed = load_module(HERE / "seed_controller.py")
    rows = {"drcik": drcik_rows(), "time_mmd": pack_rows("time_mmd"), "timesx": pack_rows("timesx")}
    report = {}
    for dataset, iterator in rows.items():
        old, new, gains = [], [], []
        for view, truth in iterator:
            a = joint(seed.forecast(view), truth)
            b = joint(candidate.forecast(view), truth)
            old.append(a); new.append(b); gains.append(a - b)
        base = statistics.mean(old)
        report[dataset] = {
            "n": len(old), "seed_joint": base, "candidate_joint": statistics.mean(new),
            "gain": statistics.mean(gains), "standardized_gain": statistics.mean(gains) / (base + 1e-12),
            "wins": sum(x > 1e-12 for x in gains), "losses": sum(x < -1e-12 for x in gains),
            "worst": min(gains),
        }
    report["objective"] = statistics.mean(v["standardized_gain"] for v in report.values())
    report["feasible"] = all(v["standardized_gain"] >= -0.001 for v in report.values() if isinstance(v, dict))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.candidate)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
