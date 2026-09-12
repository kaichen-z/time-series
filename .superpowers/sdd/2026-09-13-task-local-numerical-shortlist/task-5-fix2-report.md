# Task 5 fix round 2 report

## RED/GREEN

The terminal index regression initially failed because it constructed the old
index shape without policy/Public commitments. It was updated to the required
canonical shape and verifies the diagnostics payload identity.

```text
python -m pytest -q tests/test_task_local_ensemble_cli.py tests/test_evolution_v2_numerical_artifacts.py tests/test_package_numerical_supply.py
93 passed in 4.21s
git diff --check
clean
```

## Producer closure

- Task 4 now creates a canonical per-task diagnostics payload with task/input
  identity, sorted rows, and closed Public flag; the index fingerprints these
  exact bytes and the terminal writer persists every diagnostics object before
  index/manifest.
- The evidence loader validates diagnostic payload schema, ordering, shortlist
  subset, canonical hashes, and freezes its copied mappings.
- The artifact catalog has a typed diagnostics kind and invokes Task 1
  shortlist/policy parsers for their respective artifacts.

## Remaining

The requested complete-catalog `freeze_qd_supply` projection is still not
implemented in this round. The existing QD child registry exposes only its
four ranked members, so wiring 6--10 evidence names there without a complete
catalog materializer would produce a false closure. Cache/envelope work also
remains intentionally out of scope for this producer/projection-only round.
