"""Audit correction windows using only Train inputs and approved visible traces.

Run: python3 shared/skills/correction_audit.py [--all]
No hidden, dev, or test labels are loaded.
"""
import argparse
import json
from pathlib import Path


SHARED = Path(__file__).resolve().parents[1]


def ratio(actual, predicted):
    return actual / predicted if abs(predicted) > 1e-9 else None


def fmt(value):
    return "na" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Include input-only tasks")
    args = parser.parse_args()
    views = json.loads((SHARED / "views_train.json").read_text())
    traces = {row["tid"]: row for row in
              map(json.loads, (SHARED / "traces/visible_tasks.jsonl").open())}
    print("task      fold  cell          H  correction       seed_delta  truth/base near window")
    for tid, view in sorted(views.items(), key=lambda item: int(item[0].split("_")[1])):
        trace = traces.get(tid)
        if trace is None and not args.all:
            continue
        gain_delta = (trace["main_method_gain"] - trace["base_only_gain"]
                      if trace else None)
        for correction in view["corrections"]:
            start, end, mult = (correction[k] for k in ("start", "end", "multiplier"))
            ratios = ""
            if trace:
                lo = max(0, start - 1)
                hi = min(view["H"], end + 1)
                values = [fmt(ratio(trace["truth"][i], view["base_forecast"][i]))
                          for i in range(lo, hi)]
                if len(values) > 8:
                    values = values[:4] + ["..."] + values[-3:]
                ratios = " ".join(values)
            print(f"{tid:9} {'vis' if trace else 'inp':4} {view['cell']:12} "
                  f"{view['H']:3}  [{start},{end}) x{mult:g}"
                  f"  {fmt(gain_delta):>10}  {ratios}")


if __name__ == "__main__":
    main()
