# Visible correction audit

Run `python3 shared/skills/audit_visible.py 0005 0007` from any directory to compare two submitted attempts. The script reports visible fitness, changed task gains, duplicate series groups, and truth-to-base ratios at extracted event endpoints. Use it to distinguish a repeated benchmark instance from an independent pattern and to inspect interval boundary assumptions.

It reads only `shared/attempts/*.json` and `shared/traces/visible_tasks.jsonl`. It does not access hidden labels or run the evaluator.
