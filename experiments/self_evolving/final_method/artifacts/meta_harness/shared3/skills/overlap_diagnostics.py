"""Inspect overlapping extracted corrections using permitted visible traces only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    path = args.shared / "traces" / "visible_tasks.jsonl"
    for line in path.open():
        task = json.loads(line)
        corrections = task["corrections"]
        overlaps = []
        for i, left in enumerate(corrections):
            for j, right in enumerate(corrections[i + 1:], i + 1):
                start = max(0, left["start"], right["start"])
                end = min(task["H"], left["end"], right["end"])
                if start < end:
                    overlaps.append((i, j, start, end))
        if not overlaps:
            continue
        print(task["tid"], task["cell"], "H=", task["H"])
        for i, correction in enumerate(corrections):
            solo = task["corrections_solo"][i]["gain_if_applied_alone"]
            delta = solo - task["base_only_gain"]
            print("  ", i, correction, f"solo_delta={delta:+.4f}")
        print("   overlaps:", overlaps)


if __name__ == "__main__":
    main()
