# Visible correction audit

Run `python3 shared/skills/audit_corrections.py` from any directory. It loads
`shared/best_harness.py`, the latest accepted attempt, and only the permitted
round-two Train views and visible traces. It reports which tasks changed from
the base forecast, their visible gain change, invalid outputs, and unused
corrections with a positive solo gain. Pass `--harness` and `--attempt` to
audit a particular candidate and its corresponding submitted result.

The script does not estimate fitness or inspect dev, test, or hidden labels.
