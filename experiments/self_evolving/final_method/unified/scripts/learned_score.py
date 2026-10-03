"""Score one run or collect all learned and zero-shot metric files."""
import json, sys
from pathlib import Path
from learned_common import OUT, score

ROOT = Path(__file__).resolve().parents[5]
if len(sys.argv) == 3:
    score(sys.argv[1], sys.argv[2]); raise SystemExit

rows = []
zero = ROOT / "runs/zero_shot_baselines/metrics.json"
if zero.exists(): rows.extend(json.load(open(zero)))
for path in sorted(OUT.glob("*/*/metrics.json")):
    rows.extend(json.load(open(path)))
json.dump(rows, open(OUT / "metrics.json", "w"), indent=2)
for row in rows:
    print(f"{row['dataset']:7s} {row['split']:8s} {row['model']:12s} "
          f"{row['tasks']:4d}/{row['expected']:4d} {str(row['smae']):>6s} {str(row['srmse']):>6s}")
