"""Summarize permitted visible correction traces; no hidden labels are read."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    trace = args.shared / 'traces' / 'visible_tasks.jsonl'
    for line in trace.open():
        task = json.loads(line)
        base = task['base_forecast']
        sigma = max(abs(task['sigma_main_calib']), 1e-9)
        for correction, solo in zip(task['corrections'], task['corrections_solo']):
            start = max(0, correction['start'])
            end = min(task['H'], correction['end'])
            if end <= start:
                continue
            level = sum(abs(x) for x in base[start:end]) / (end - start)
            shift_sigma = abs(correction['multiplier'] - 1) * level / sigma
            solo_delta = solo['gain_if_applied_alone'] - task['base_only_gain']
            print(f"{task['tid']:9} {task['cell']:12} "
                  f"m={correction['multiplier']:6.2f} "
                  f"width={(end-start)/task['H']:.2f} "
                  f"shift/sigma={shift_sigma:10.2f} "
                  f"solo_delta={solo_delta:+.3f}")


if __name__ == '__main__':
    main()
