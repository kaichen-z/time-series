#!/usr/bin/env python3
"""Approximate a numerical-program change using only visible task traces.

This is a screening tool, not the pipeline evaluator. It excludes accepted
history repairs, infers current correction scaling from the main forecast,
and uses Toto's reported joint error to approximate each task's scale.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TRACE = Path(__file__).resolve().parents[1] / "traces" / "visible_tasks.jsonl"
BEST = Path(__file__).resolve().parents[1] / "best_config.json"


def loss(pred, truth):
    error = pred - truth
    return np.mean(np.abs(error)) + np.sqrt(np.mean(error * error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", help="additional method in the program")
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--shrink", type=float, help="new shrink toward last value")
    args = parser.parse_args()
    config = json.loads(BEST.read_text())
    program = config["numerical"]["program"]
    old_shrink = program["shrink"]
    new_shrink = old_shrink if args.shrink is None else args.shrink
    terms = program["terms"]
    rows = []
    skipped = []
    for line in TRACE.read_text().splitlines():
        task = json.loads(line)
        if task["tid"] in {"task_43", "task_193"}:
            skipped.append(task["tid"])
            continue
        methods = task["method_forecasts"]
        toto = np.asarray(methods["toto_2_0"], dtype=float)
        truth = np.asarray(task["truth"], dtype=float)
        current = np.asarray(task["main_method_forecast"], dtype=float)
        weighted = np.zeros_like(toto)
        total_weight = 0.0
        for method, weight in terms:
            weighted += weight * np.asarray(methods.get(method, toto), dtype=float)
            total_weight += weight
        old_base = (1 - old_shrink) * weighted / total_weight + old_shrink * task["history"][-1]
        if args.method:
            alternate = np.asarray(methods.get(args.method, toto), dtype=float)
            weighted += args.weight * alternate
            total_weight += args.weight
        new_base = (1 - new_shrink) * weighted / total_weight + new_shrink * task["history"][-1]
        # Reapply the inferred existing document correction to the changed base.
        correction = np.ones_like(current)
        nonzero = np.abs(old_base) > 1e-6
        correction[nonzero] = current[nonzero] / old_base[nonzero]
        candidate = new_base * correction
        toto_joint = task["toto_joint_error"]
        scale = loss(toto, truth) / toto_joint if toto_joint else np.nan
        delta = (loss(current, truth) - loss(candidate, truth)) / scale
        rows.append((task["tid"], task["fold"], float(delta)))
    for fold in sorted({row[1] for row in rows}):
        subset = [row[2] for row in rows if row[1] == fold]
        print(f"fold {fold}: sum proxy gain {sum(subset):+.4f}; tasks {len(subset)}")
    print("worst 5:", sorted(rows, key=lambda row: row[2])[:5])
    print("best 5:", sorted(rows, key=lambda row: row[2])[-5:])
    print("excluded accepted repair tasks:", ", ".join(skipped))


if __name__ == "__main__":
    main()
