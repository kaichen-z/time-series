# Task 5 report — partial closure

## RED

`python -m pytest -q tests/test_package_numerical_supply.py -k verified_task_shortlist`
failed as intended with `TypeError: bound_numerical_package() got an unexpected keyword argument 'shortlist'`.

`python -m pytest -q tests/test_evolution_v2_numerical_artifacts.py -k shortlist_artifacts`
failed as intended because `ArtifactKindV2.TASK_SHORTLIST` was absent.

## GREEN

After the scoped changes:

```text
python -m pytest -q tests/test_package_numerical_supply.py tests/test_evolution_v2_numerical_artifacts.py
81 passed in 3.44s
git diff --check
clean
```

## Implemented

- Registered canonical typed `TASK_SHORTLIST`, `SHORTLIST_POLICY`, and
  `SHORTLIST_INDEX` artifacts.
- Added an optional, explicitly verified schema-v2 shortlist binding to
  `bound_numerical_package`. It retains the full release, projects only the
  ordered shortlist into active/ranked/diagnostic package state, requires the
  protected Anchor, and binds shortlist/policy/dictionary/diagnostic
  fingerprints.
- Legacy callers retain their previous no-shortlist behavior.

## Concerns / remaining closure

The current QD freeze input supplies only an archive/children and has no
Task-4 task-shortlist, policy, index, or local diagnostic artifact objects.
Consequently it cannot truthfully emit the required strict envelope-v2
references, and the Hyperband cache path computes keys before freeze. Completing
the P2/P3 restoration and cache requirement needs a coordinated producer
interface that carries task-local evidence into evaluation, then a versioned
envelope/index plus shared persistence/kernel verifier. This report deliberately
does not fabricate those artifacts or perform local forecasts during restore.

Protected P3 files were not edited or staged.
