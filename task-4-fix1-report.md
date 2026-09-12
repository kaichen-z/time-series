# Task 4 fix round 1

## Evidence

- GREEN: `python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_ensemble.py tests/test_task_local_evolution.py tests/test_task_local_ensemble_cli.py tests/test_numerical_champion_loop.py` — `73 passed`.
- Hygiene: `git diff --check` passed.

## Change

- Formal V3 OOF now constructs each held-out task's persisted shortlist from a fold-complement history-only prior tuple before calling forecast or diagnostics, and evaluates the V3 tournament over only those rows.
- OOF-rejected formal runs now persist a sorted Train shortlist index and a run-manifest index commitment before writing their terminal marker.

## Concern

- The formal process has no supplied persisted Task 3 forecast-outcome prior artifact. Its V3 coordinator therefore uses deterministic history-only baseline priors (distinct per complement cohort) rather than invoking legacy group supplies or all-candidate materialization. A future input artifact can replace this baseline directly without changing the shortlist-before-materialization boundary.
