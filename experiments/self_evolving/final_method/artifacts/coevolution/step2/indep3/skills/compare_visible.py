#!/usr/bin/env python3
"""Compare visible per-task gains from two recorded submissions.

Uses only shared/attempts/*.json; it never reads hidden task labels.
"""

import argparse
import json
from pathlib import Path


def load(path: Path):
    data = json.loads(path.read_text())
    if "visible_per_task" not in data:
        raise ValueError(f"no visible_per_task in {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    left, right = baseline["visible_per_task"], candidate["visible_per_task"]
    if left.keys() != right.keys():
        raise ValueError("attempts have different visible task sets")
    changes = sorted((round(right[tid] - left[tid], 4), tid) for tid in left)
    print(f"fitness: {baseline['visible_fitness']:.6f} -> {candidate['visible_fitness']:.6f}")
    print(f"hidden check: {candidate.get('hidden_check', 'unknown')}")
    print(f"changed tasks: {sum(abs(delta) > 0.0001 for delta, _ in changes)}")
    for label, rows in (("largest losses", changes[: args.limit]),
                        ("largest gains", changes[-args.limit:][::-1])):
        print(label)
        for delta, tid in rows:
            if abs(delta) > 0.0001:
                print(f"  {tid:10} {delta:+.4f}")


if __name__ == "__main__":
    main()
