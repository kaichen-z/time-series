# Task 4 fix round 2

## RED/GREEN evidence

- RED: the strict prior-bundle regression initially failed collection because `_load_candidate_priors_bundle` did not exist.
- GREEN: `python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_ensemble.py tests/test_task_local_evolution.py tests/test_task_local_ensemble_cli.py tests/test_numerical_champion_loop.py` — `74 passed`.
- Hygiene: `git diff --check` passed.

## Change

- Formal execution requires a strict Task 3 prior bundle binding schema, grouping, Dictionary, fold priors, final priors, and canonical payload fingerprint.
- OOF materialization consumes its exact fold tuple; final release and Dev use the all-Train tuple.
- OOF shortlist/index association is now task-ID keyed, preventing fold-order/partition-order misbinding.
- The shortlist-only spy covers an eight-name selection from a twenty-candidate universe.

## Commit

`fix(numerical): bind fitted oof priors`.
