# Correction audits

`correction_audit.py` lists every extracted correction, the seed method's visible task gain relative to the base, and the ratio of visible truth to base forecast around its window. The gain is a task-level value repeated for each correction of that task. It reads only `shared/views_train.json` and `shared/traces/visible_tasks.jsonl`. Input-only tasks appear with `--all`, without outcome ratios.

Run from anywhere with `python3 /path/to/shared/skills/correction_audit.py`; use `--all` to include all 80 Train inputs. The ratio sequence begins one step before each window when possible and ends one step after it, making timestamp boundary mistakes easy to spot. Long sequences show the first four and last three ratios. `na` means the base is zero or the task is input-only.
