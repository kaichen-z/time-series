# Task 4 fix round 4

## Evidence

- Added a two-fold, 21-candidate V3 coordinator spy. Distinct fold priors select and materialize their own exact eight candidates only.
- Added compact strict loader mutations for fingerprint, grouping, Dictionary, fold-key, and namespace failure paths.
- Focused tests: `21 passed`; `git diff --check` passed.

## Commit

`test(numerical): cover v3 shortlist routing`.
