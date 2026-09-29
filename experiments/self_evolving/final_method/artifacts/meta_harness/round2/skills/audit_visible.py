#!/usr/bin/env python3
"""Compare two harnesses on permitted Train views and visible truth only.

Usage: python3 shared/skills/audit_visible.py OLD.py NEW.py
Prints changed task IDs, whether visible L1/L2 error fell, and validates output shape.
This is a diagnostic, not a replacement for submit_h.py's official score.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_adjust(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adjust


def main(old_path, new_path):
    views = json.loads((ROOT / 'views_train.json').read_text())
    traces = {row['tid']: row for row in
              map(json.loads, (ROOT / 'traces/visible_tasks.jsonl').read_text().splitlines())}
    old, new = load_adjust(old_path, 'old_harness'), load_adjust(new_path, 'new_harness')
    changed = 0
    for tid, view in views.items():
        before, after = old(view), new(view)
        h = view['H']
        for label, forecast in [('old', before), ('new', after)]:
            if len(forecast) != h or not all(math.isfinite(x) for x in forecast):
                raise ValueError(f'{tid}: invalid {label} forecast')
        if before == after:
            continue
        changed += 1
        if tid in traces:
            truth = traces[tid]['truth']
            l1 = sum(abs(a-y)-abs(b-y) for a,b,y in zip(before,after,truth))
            l2 = sum((a-y)**2-(b-y)**2 for a,b,y in zip(before,after,truth))
            print(f'{tid}: visible L1 delta {l1:+.3f}, L2 delta {l2:+.3f}')
        else:
            print(f'{tid}: unlabeled change')
    print(f'{changed} / {len(views)} views changed; all outputs finite and length H')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
