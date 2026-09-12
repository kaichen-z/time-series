# Task 4 fix round 5

## Evidence

- The two-fold coordinator spy now asserts both full-forecast and diagnostic calls are exactly each fold's eight shortlisted names.
- Both accepted and rejected formal terminal paths call the shared `_bind_shortlist_index` helper; parametrized coverage verifies its canonical index/manifest fingerprint closure.
- Focused suite: `78 passed`; `git diff --check` passed.

## Commit

`fix(numerical): close shortlist artifacts`.
