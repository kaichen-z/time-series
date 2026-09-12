# Task 5A cache identity

Implemented the shortlist-evidence cache boundary only.

- Cache keys have explicit branches: version 1 for explicit evidence-less legacy
  work and version 2 for a required `local_evidence_sha256`.
- `TaskCacheRowV2` and `HyperbandTaskResultV2` persist both the evidence digest
  and cache identity version. Missing or mismatched values are rejected.
- The numerical runner obtains `adapter.local_evidence_sha256_for(candidate, task)`
  before cache lookup or evaluation. Hyperband supports the same callback.
- Persistence recomputes evidence-aware task and rung keys. Changed evidence
  produces a deterministic cache miss; mutated persisted evidence is rejected.

Verification:

```text
python -m pytest -q tests/test_evolution_v2_numerical_hyperband.py -k 'cache_identity_binds or evidence_bound or changed_evidence'
11 passed
python -m pytest -q tests/test_evolution_v2_numerical_persistence.py -k 'mutated_local_evidence'
1 passed
python -m pytest -q tests/test_evolution_v2_numerical_runner.py -k 'rung_binds_adapter_local_evidence'
1 passed
git diff --check
```

The existing unrelated `evolving_loop/v2/real/bridges.py` and
`tests/test_evolution_v2_real_cooperative.py` worktree changes were not staged.
