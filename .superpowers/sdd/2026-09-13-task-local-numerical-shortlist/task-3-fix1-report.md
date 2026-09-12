# Task 3 fix 1 report

## Change

Canonicalized and sorted validated `task_ids` before all prior aggregation. Added a regression asserting reversed caller order yields identical prior payloads and fingerprints.

## Verification

```text
python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_evolution.py
26 passed
git diff --check
```

## Commit

`fix(numerical): canonicalize prior task order`
