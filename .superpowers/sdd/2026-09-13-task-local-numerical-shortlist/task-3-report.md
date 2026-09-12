# Task 3 report: fit shortlist priors

## RED

Added tests covering canonical candidate ordering, successful and failed/missing forecasts, finite worst-case penalties, morphology scores, and OOF complement-only fitting. Before implementation, pytest failed during collection because `fit_candidate_priors` was not defined.

## GREEN

Implemented `fit_candidate_priors` in `task_shortlist.py` and `fit_oof_shortlist_priors` in `task_local_evolution.py`. Both consume only exact Train `TaskLocalTaskRow` truth/forecast data, score successful forecasts with `drcik_point_metrics` and `joint_scaled_error`, use `linear_quantile(..., 0.9)`, and assign missing/failed forecasts finite 5.0 penalties. OOF folds fit complement task IDs; final priors fit all manifest Train IDs. Diagnostics are ignored and no diagnostic function is called.

Evidence:

```text
python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_evolution.py
25 passed
git diff --check
```

## Concerns

Legacy `fit_group_candidate_supply` and release construction were intentionally left unchanged per the Task 3 ledger ruling; Task 4 owns schema-v3 release wiring. Existing protected P3 worktree changes were not staged.

## Commit

`feat(numerical): fit shortlist priors`
