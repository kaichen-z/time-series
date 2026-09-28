"""Compare two harnesses on permitted visible traces and all unlabeled views."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def load_adjust(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adjust


def raw_error(prediction, truth):
    residual = [a - b for a, b in zip(prediction, truth)]
    return (sum(abs(x) for x in residual) / len(residual)
            + math.sqrt(sum(x * x for x in residual) / len(residual)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--shared', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    old, new = load_adjust(args.baseline), load_adjust(args.candidate)
    visible = {x['tid']: x for x in map(json.loads,
               (args.shared / 'traces' / 'visible_tasks.jsonl').open())}
    views = json.loads((args.shared / 'views_train.json').read_text())
    for tid, view in views.items():
        a, b = old(view), new(view)
        assert len(a) == len(b) == view['H']
        assert all(math.isfinite(x) for x in a + b)
        if not any(abs(x - y) > 1e-8 for x, y in zip(a, b)):
            continue
        if tid in visible:
            truth = visible[tid]['truth']
            delta = raw_error(a, truth) - raw_error(b, truth)
            print(f'{tid}: raw MAE+RMSE improvement {delta:+.6g}')
        else:
            print(f'{tid}: unlabeled view changed')


if __name__ == '__main__':
    main()
