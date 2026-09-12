# Final cache fix report

## Scope

Fixed `evolving_loop/v2/numerical_qd/contracts.py` and focused Hyperband cache
regressions only.

## Change

No-evidence cache identities now use the exact eight-dependency payload from
baseline `54cb948`; `cache_identity_version` is omitted from that hashed legacy
payload. Evidence-bound identities retain version 2 and the evidence digest.
Exact three-field and seven-field legacy payloads continue to parse and resume,
while partial or mixed payloads are rejected. Legacy rows are misses when a v2
evidence digest is required.

## Verification

- `pytest -q tests/test_evolution_v2_numerical_hyperband.py` — 108 passed.
- `pytest -q tests/test_evolution_v2_numerical_safety.py` — 22 passed, 1
  unrelated pre-existing fixture failure in
  `test_real_run_artifacts_exclude_environment_path_public_and_dev_sentinels`
  (the expected injected Dev sentinel was absent).
