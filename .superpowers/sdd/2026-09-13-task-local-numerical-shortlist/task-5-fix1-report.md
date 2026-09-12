# Task 5 fix round 1 report

## Root cause

The prior package-only hook was not connected to a sealed Task 4 evidence
producer. The review correctly identified that it therefore could not reach
freeze, restore, or pre-evaluation cache identity.

## RED/GREEN

The existing package shortlist test was updated to require an explicit
canonical diagnostics SHA. Before the production change it had no such API;
the focused package regression is green:

```text
python -m pytest -q tests/test_package_numerical_supply.py -k verified_task_shortlist
1 passed
```

## Implemented this round

- Added the immutable `TaskLocalEvidenceBundleV1` loader/validator skeleton
  and pure adapter evidence accessors.
- Upgraded Task 4 shortlist-index bytes to bind policy/Public status and write
  its canonical policy artifact.
- Registered `HINDCAST_DIAGNOSTICS` and changed package binding to require an
  explicit diagnostic SHA; it rejects missing shortlist candidates rather than
  silently projecting them away.

## Remaining

The v2 envelope dispatch, object persistence, shared P3 verifier, and
cache-key propagation still need one coordinated pass. They were not claimed
as complete here; the adapter now supplies the deterministic pre-evaluation
seam those changes require.
