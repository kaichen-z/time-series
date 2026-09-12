# Task 5 fix round 3 report

## Implemented

- `HINDCAST_DIAGNOSTICS` now has the producer's exact task-input field set.
- Evidence diagnostics parsing binds the expected task ID, requires canonical
  on-disk bytes, validates ordered shortlist-subset rows, and recursively
  freezes loaded JSON values.
- Schema-v2 freeze retains the complete parent catalog plus selected child
  specs; the four selected QD members remain only the ranking/provenance set.
- Evidence-aware child materialization and freeze builders pass the immutable
  shortlist and its exact diagnostics SHA into package binding. Missing
  shortlist identities fail instead of silently falling back.

## Verification

```text
python -m pytest -q tests/test_evolution_v2_numerical_adapters.py -k 'freeze or terminal_index or artifact'
66 passed, 549 deselected
python -m pytest -q tests/test_task_local_ensemble_cli.py tests/test_evolution_v2_numerical_artifacts.py tests/test_package_numerical_supply.py
93 passed
git diff --check
clean
```

## Scope

Cache and versioned frozen-envelope/P3 restoration work remains intentionally
out of scope for this round. Protected bridge files were not changed.
