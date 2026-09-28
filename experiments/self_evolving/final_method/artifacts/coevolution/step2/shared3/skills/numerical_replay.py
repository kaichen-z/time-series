"""Replay base-forecast program changes on visible tasks without repair variants.

Use only shared/traces/visible_tasks.jsonl. The main forecast supplies the
unchanged document-correction factor. Repaired Toto is unavailable in traces,
so those tasks are excluded. This is a screening tool, not the evaluator.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = [json.loads(line) for line in (ROOT / 'traces/visible_tasks.jsonl').open()]
BEST = json.loads((ROOT / 'best_config.json').read_text())
SEED_PROGRAM = {
    'terms': [['toto_2_0', 1.1005169383119369], ['arima_auto', .059]],
    'diff': None, 'shrink': .20153008320833654, 'clip': None,
}


def forecast(task, program):
    fs = task['method_forecasts']
    terms = program['terms']
    total = sum(weight for _, weight in terms)
    out = sum(weight / total * np.asarray(fs.get(method, fs['toto_2_0']), float)
              for method, weight in terms)
    if program.get('diff'):
        a, b, c = program['diff']
        out += c * (np.asarray(fs.get(a, fs['toto_2_0']), float)
                    - np.asarray(fs.get(b, fs['toto_2_0']), float))
    last = task['history'][-1]
    out = (1 - program['shrink']) * out + program['shrink'] * last
    if program.get('clip') is not None:
        hist = np.asarray(task['history'], float)
        pad = program['clip'] * np.ptp(hist)
        out = np.clip(out, hist.min() - pad, hist.max() + pad)
    return out


def joint(pred, truth):
    truth = np.asarray(truth, float)
    residual = pred - truth
    return min(5, (np.mean(np.abs(residual)) + np.sqrt(np.mean(residual ** 2)))
               / np.mean(np.abs(truth)))


def replay(program, reference=None):
    # The trace's main_method_forecast was produced by the seed program.
    reference = reference or SEED_PROGRAM
    rows = []
    for task in TASKS:
        if task['repair_variants']:
            continue
        old = forecast(task, reference)
        new = forecast(task, program)
        main = np.asarray(task['main_method_forecast'], float)
        factor = np.divide(main, old, out=np.ones_like(main), where=np.abs(old) > 1e-9)
        pred = main + factor * (new - old)
        gain = task['toto_joint_error'] - joint(pred, task['truth'])
        rows.append((task['fold'], task['tid'], gain, gain - task['main_method_gain']))
    folds = []
    for fold in (0, 1):
        gains = np.asarray([row[2] for row in rows if row[0] == fold])
        folds.append(gains.mean() + 0.5 * np.minimum(gains, 0).mean())
    fitness = np.mean(folds) - 0.25 * np.std(folds)
    return fitness, rows


if __name__ == '__main__':
    base = BEST['numerical']['program']
    score0, _ = replay(base)
    print('reference', round(score0, 6))
    methods = sorted(set.intersection(*(set(t['method_forecasts']) for t in TASKS)))
    for method in methods:
        if method == 'toto_2_0':
            continue
        for weight in (.02, .05):
            program = dict(base)
            total = sum(w for _, w in base['terms'])
            program['terms'] = [[m, w * (1 - weight)] for m, w in base['terms']] + [[method, total * weight]]
            score, rows = replay(program)
            changes = np.asarray([r[3] for r in rows])
            print(method, weight, round(score - score0, 5),
                  sum(changes > 0), sum(changes < 0),
                  round(np.median(changes), 5))
