"""Audit visible attempt changes and extracted event endpoints.

Reads only the explicitly permitted visible trace and submitted attempt files.
"""
import argparse
import hashlib
import json
from pathlib import Path


SHARED = Path(__file__).resolve().parents[1]


def signature(task):
    """Group repeated series while retaining the full correction geometry."""
    values = [task["base_forecast"], task["truth"], task["corrections"]]
    payload = json.dumps(values, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", help="Attempt number, e.g. 0005")
    parser.add_argument("after", help="Attempt number, e.g. 0007")
    args = parser.parse_args()

    def attempt(number):
        matches = list((SHARED / "attempts").glob(f"{number}*.json"))
        if len(matches) != 1:
            raise SystemExit(f"Expected one attempt for {number}; found {len(matches)}")
        return json.loads(matches[0].read_text())

    before, after = attempt(args.before), attempt(args.after)
    traces = {
        task["tid"]: task
        for task in (json.loads(line) for line in (SHARED / "traces" / "visible_tasks.jsonl").open())
    }
    print(f"Fitness: {before['visible_fitness']:.6f} -> {after['visible_fitness']:.6f}")
    changes = []
    for tid, gain in after["visible_per_task"].items():
        delta = gain - before["visible_per_task"][tid]
        if abs(delta) > 1e-6:
            changes.append((tid, delta, traces[tid]))
    print(f"Changed tasks: {len(changes)}; distinct series and correction patterns: "
          f"{len({signature(task) for _, _, task in changes})}")
    for tid, delta, task in sorted(changes, key=lambda row: -abs(row[1])):
        print(f"{tid:>9} delta={delta:+.4f} group={signature(task)} "
              f"cell={task['cell']} corrections={task['corrections']}")
        for correction in task["corrections"]:
            s, e = correction["start"], min(correction["end"], task["H"])
            if e <= s or e < 2:
                continue
            ratios = []
            for step in (max(s, e - 2), e - 1):
                base = task["base_forecast"][step]
                ratios.append(round(task["truth"][step] / base, 3) if abs(base) > 1e-9 else None)
            print(f"           penultimate / final truth:base = {ratios}")


if __name__ == "__main__":
    main()
