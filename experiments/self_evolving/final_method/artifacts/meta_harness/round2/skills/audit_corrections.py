"""Audit a correction harness against permitted round-two visible data only."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


SHARED = Path(__file__).resolve().parents[1]


def load_function(path):
    spec = importlib.util.spec_from_file_location("audit_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adjust


def changed(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if abs(x - y) > 1e-8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, default=SHARED / "best_harness.py")
    parser.add_argument("--attempt", type=Path, default=None,
                        help="accepted attempt JSON for per-task gains")
    args = parser.parse_args()
    attempts = sorted((SHARED / "attempts").glob("*.json"))
    accepted = [p for p in attempts if json.loads(p.read_text()).get("accepted")]
    attempt_path = args.attempt or accepted[-1]
    gains = json.loads(attempt_path.read_text())["visible_per_task"]
    views = json.loads((SHARED / "views_train.json").read_text())
    traces = {x["tid"]: x for x in map(json.loads,
              (SHARED / "traces" / "visible_tasks.jsonl").open())}
    adjust = load_function(args.harness)
    changed_tasks = []
    unused_positive = []
    invalid = []
    for tid, view in views.items():
        out = adjust(view)
        if len(out) != view["H"] or not all(isinstance(x, (int, float))
                                              and math.isfinite(x) for x in out):
            invalid.append(tid)
            continue
        bins = changed(out, view["base_forecast"])
        if bins:
            changed_tasks.append(tid)
        if tid not in traces:
            continue
        trace = traces[tid]
        good_solo = [(c["start"], c["end"], c["multiplier"],
                      round(c["gain_if_applied_alone"], 4))
                     for c in trace["corrections_solo"]
                     if c["gain_if_applied_alone"] > 0.01]
        if good_solo and not bins:
            unused_positive.append((tid, good_solo))
        if bins:
            base = trace["base_only_gain"]
            gain = gains.get(tid)
            print(tid, view["cell"], "bins", len(bins), "gain", gain,
                  "delta_from_base", round(gain - base, 4) if gain is not None else "?")
    print("\nChanged tasks:", len(changed_tasks), "of", len(views))
    print("Invalid outputs:", invalid)
    print("Unused positive solo corrections (>0.01):", unused_positive)
    print("Attempt:", attempt_path.name)


if __name__ == "__main__":
    main()
