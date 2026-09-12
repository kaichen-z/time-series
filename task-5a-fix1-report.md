# Task 5A fix 1: legacy cache payload parser

`TaskCacheRowV2` and `HyperbandTaskResultV2` now accept only their complete
pre-Task-5A payload schemas: three and seven fields respectively. Each parser
normalizes the old shape to `local_evidence_sha256=None` and
`cache_identity_version=1`. The exact current schema remains strict; partial
or mixed field sets are rejected.

This permits explicit no-evidence cache/resume replay while preserving the v2
evidence boundary: a legacy row still cannot match a v2 evidence lookup.

Verification:

```text
python -m pytest -q tests/test_evolution_v2_numerical_hyperband.py -k 'exact_legacy_cache_payloads or legacy_cache_row_resumes or evidence_bound or changed_evidence'
4 passed
python -m pytest -q tests/test_evolution_v2_numerical_persistence.py -k 'mutated_local_evidence'
1 passed
python -m pytest -q tests/test_evolution_v2_numerical_runner.py -k 'rung_binds_adapter_local_evidence'
1 passed
git diff --check
```
