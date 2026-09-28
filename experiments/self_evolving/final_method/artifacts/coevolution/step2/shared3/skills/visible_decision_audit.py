#!/usr/bin/env python3
"""Audit decision outcomes using only the allowed visible task trace."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "traces" / "visible_tasks.jsonl"


def joint_error(forecast, truth):
    truth = np.asarray(truth, dtype=float)
    residual = np.asarray(forecast, dtype=float) - truth
    scale = np.mean(np.abs(truth))
    return min(5.0, (np.mean(np.abs(residual)) + np.sqrt(np.mean(residual**2))) / scale)


def seed_base(task):
    """Rebuild the numerical part of the trace's original main method."""
    methods = task["method_forecasts"]
    toto = np.asarray(methods["toto_2_0"], dtype=float)
    arima = np.asarray(methods["arima_auto"], dtype=float)
    blended = (1.1005169383119369 * toto + 0.059 * arima) / (1.1005169383119369 + 0.059)
    shrink = 0.20153008320833654
    return (1 - shrink) * blended + shrink * task["history"][-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort", choices=("effect", "trust", "task"), default="effect")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in TRACE.open()]
    rows = []
    for task in tasks:
        if not task["future_corrections"] or task["repair_variants"]:
            continue  # A repaired Toto forecast cannot be reconstructed from this trace.
        base = seed_base(task)
        main_forecast = np.asarray(task["main_method_forecast"], dtype=float)
        effect = joint_error(base, task["truth"]) - joint_error(main_forecast, task["truth"])
        rows.append((task["tid"], task["toto_history_backtest_error"], effect,
                     max(abs(main_forecast - base)),
                     ",".join(str(c["multiplier"]) for c in task["future_corrections"])))
    keys = {"effect": lambda r: r[2], "trust": lambda r: float("inf") if r[1] is None else r[1],
            "task": lambda r: int(r[0].split("_")[1])}
    for tid, trust, effect, max_change, multipliers in sorted(rows, key=keys[args.sort]):
        print(f"{tid:9} trust={str(round(trust, 3)) if trust is not None else 'NA':>5} "
              f"correction_effect={effect:+.4f} max_forecast_change={max_change:.2f} "
              f"multipliers={multipliers}")


if __name__ == "__main__":
    main()
