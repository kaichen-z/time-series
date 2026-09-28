#!/usr/bin/env python3
"""Audit which document corrections changed the current forecast on visible tasks."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    shared = args.run_dir / "shared"
    cfg = json.loads((shared / "best_config.json").read_text())
    program = cfg["numerical"]["program"]
    terms = program["terms"]
    total_weight = sum(weight for _, weight in terms)

    with (shared / "traces" / "visible_tasks.jsonl").open() as handle:
        for line in handle:
            task = json.loads(line)
            if not task["future_corrections"]:
                continue
            methods = task["method_forecasts"]
            base = sum(np.asarray(methods[name]) * weight for name, weight in terms) / total_weight
            base = (1 - program["shrink"]) * base + program["shrink"] * task["history"][-1]
            final = np.asarray(task["main_method_forecast"])
            for correction in task["future_corrections"]:
                start, end = correction["start"], correction["end"]
                if start >= end or start >= len(base):
                    continue
                segment = slice(start, min(end, len(base)))
                # A task with accepted history repair changes the Toto term too.
                # Report its observed ratio but flag it as confounded below.
                valid = np.abs(base[segment]) > 1e-8
                ratios = final[segment][valid] / base[segment][valid]
                applied = float(np.median(ratios)) if len(ratios) else float("nan")
                confounded = task["tid"] in {"task_43", "task_193"}
                print(
                    task["tid"], task["fold"], task["group"],
                    f"m={correction['multiplier']:.3g}",
                    f"applied={applied:.3f}",
                    f"conf={task['doc_confidence']:.2f}",
                    f"docbase={task['docbase']:.2f}",
                    f"gain={task['main_method_gain']:.3f}",
                    "repair_confounded" if confounded else "",
                )


if __name__ == "__main__":
    main()
