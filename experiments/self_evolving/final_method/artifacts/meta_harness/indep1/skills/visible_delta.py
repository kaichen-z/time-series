"""Summarize which visible tasks changed between two submitted harnesses.

Usage: python3 visible_delta.py 0006_i1 0014_i1
Reads only allowed visible attempt results and Train views.
"""
import json
import pathlib
import sys


SHARED = pathlib.Path(__file__).resolve().parents[1]


def load_attempt(name):
    path = SHARED / "attempts" / (name.removesuffix(".json") + ".json")
    return json.loads(path.read_text())


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: visible_delta.py OLD_ATTEMPT NEW_ATTEMPT")
    old, new = (load_attempt(arg) for arg in sys.argv[1:])
    views = json.loads((SHARED / "views_train.json").read_text())
    print(f"fitness: {old['visible_fitness']:.6f} -> {new['visible_fitness']:.6f}")
    print(f"hidden check: {new['hidden_check']}; runtime errors: {new['n_runtime_errors']}")
    changes = []
    for tid, gain in new["visible_per_task"].items():
        delta = gain - old["visible_per_task"][tid]
        if abs(delta) > 0.0001:
            changes.append((tid, delta))
    print(f"changed visible tasks: {len(changes)}")
    for tid, delta in sorted(changes, key=lambda item: abs(item[1]), reverse=True):
        view = views[tid]
        title = view["documents"][0].strip().splitlines()[0][:65] if view["documents"] else ""
        print(f"{tid:9} {delta:+.4f}  {view['cell']:12}  {title}")


if __name__ == "__main__":
    main()
