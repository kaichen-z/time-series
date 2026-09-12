# Task 4 fix round 3

## RED/GREEN evidence

- RED: the Task 3 bundle/loader round-trip had no producer API before this change.
- GREEN: focused Task 3 bundle and shell-contract tests passed; full focused verification follows this report update.

## Change

- Added Task 3 canonical candidate-prior bundle build/write APIs. The payload contains exact fold-complement priors, final priors, grouping and Dictionary identities, and a canonical fingerprint.
- The formal shell runner now requires and forwards `TASK_LOCAL_CANDIDATE_PRIORS_FILE`.
- Added producer-to-loader canonical round-trip coverage and dry-run argument coverage.

## Commit

`fix(numerical): produce shortlist priors`.
