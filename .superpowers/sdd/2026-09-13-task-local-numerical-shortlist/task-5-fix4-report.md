# Task 5 fix round 4 report

## Runtime fixes

- Fixed the direct evidence-bundle construction `NameError` by binding the
  diagnostics parser to the index entry task ID.
- Canonical-byte verification now covers policy, index, shortlist, and
  diagnostics files before parsing.
- Complete-catalog freeze treats rank as projection-local when identical
  ranked forecast materializations merge, preventing first-freeze rejection.

## Concern

The cache-row contract is serialized throughout Hyperband, runner, and
persistence. A safe local-evidence identity change requires updating all
constructors and exact payload fixtures in one atomic pass; it was not
completed in this speed-limited round, so no default/fabricated evidence SHA
was introduced.
