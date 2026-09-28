# Visible correction diagnostics

Run `python3 shared/skills/correction_diagnostics.py` from any directory. The script reads only `shared/traces/visible_tasks.jsonl` and prints each correction's multiplier, horizon fraction, implied shift divided by `sigma_main_calib`, and its solo gain relative to the uncorrected base. It is a screening aid: solo gains do not account for overlapping corrections and a visible pattern may reflect only one task family.

To audit a proposed harness against an earlier one, run `python3 shared/skills/compare_visible_harnesses.py OLD.py NEW.py`. It checks exact output length and finiteness on all 80 unlabeled views, lists every changed task, and reports the change in raw MAE+RMSE only for the 54 permitted visible traces. Positive values mean the candidate has lower raw error. This local comparison does not reproduce the capped, normalized fitness or the hidden check.

Run `python3 shared/skills/overlap_diagnostics.py` to list the visible tasks with overlapping correction windows. It reports overlap steps and each correction's solo gain relative to the uncorrected base. These solo gains are not additive in an overlap; inspect the raw forecast and truth before proposing an aggregation rule.
