# Visible result comparison

`compare_visible.py` compares the visible per-task gains in two recorded submissions. It reports changed tasks and the largest gains and losses, which helps identify whether a global parameter change affected many tasks or only a few threshold decisions. It reads `shared/attempts/` files only and does not inspect hidden labels.

From an agent workspace under the run directory:

```bash
python3 ../shared/skills/compare_visible.py \
  ../shared/attempts/0005_i3.json ../shared/attempts/0012_i3.json
```

Use the comparison as a diagnostic, not as a reason to optimize isolated visible tasks. A candidate still needs the evaluator's hidden check.
