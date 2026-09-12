# Task 4 report

## RED/GREEN evidence

- RED: `python -m pytest -q tests/test_task_local_ensemble.py -k v3` failed at collection with `ImportError: cannot import name 'TaskLocalEnsembleReleaseV3'` before V3 implementation.
- GREEN: `python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_ensemble.py tests/test_task_local_evolution.py tests/test_task_local_ensemble_cli.py tests/test_numerical_champion_loop.py` passed: `73 passed`.
- Hygiene: `git diff --check` passed.

## Change

- Added strict schema-v3 task-local release parsing and canonical encoding while preserving exact legacy v1/v2 parsing.
- Added shortlist-only runtime materialization, persisted canonical shortlist files/index bindings, V3 Public shortlist reconstruction, and an explicit legacy-only `numerical_loop` boundary.

## Commit

`feat(numerical): run per-task local shortlist`.

## Concerns

- The formal runner still uses the existing Train-wide historical row collection to fit Task 3 priors and derive its OOF acceptance report; V3 runtime/Dev/Public materialization is shortlist-only. A subsequent tightening should replace that legacy OOF report construction with fold-complement shortlist materialization end-to-end.
