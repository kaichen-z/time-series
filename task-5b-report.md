# Task 5B: frozen shortlist closure

Implemented the minimal versioned frozen handoff. Schema 1 keeps its exact six
fields and two-field entries. Schema 2 seals policy, Dictionary and index
identities plus task-input, shortlist, diagnostics and package references, with
package-component cross-checks. Seed and final projection envelopes consume the
adapter's immutable evidence bundle.

The runner copies typed evidence before each frozen pair and checks returned
content identities. Task 4 policy, shortlist and diagnostics objects retain
their original `canonical_json_bytes` encoding and SHA; the index uses the
declared compact V2 identity. Store reads and catalog checks recognize only
those exact Task 4 schemas. The frozen pair remains exactly `{supply, registry}`.

One pure verifier checks raw bytes, strict typed/canonical parsing, index and
entry joins, task identities, candidate scopes, Anchor weight/count constraints,
and exact fallback. Completed-store loading adds trusted task/history checks;
Kernel handoff calls the same verifier without trusted tasks. Neither path
performs forecasts or hindcasts. Index and diagnostics syntax is now validated
before typed evidence writes, including nested diagnostic dataclass fields.

Task 5A cache implementation and protected `real/bridges.py` / cooperative test
were not changed or staged.

## Verification

- RED: missing `import_numerical_seed(..., evidence=...)` and seven malformed
  index/diagnostics payloads that previously reached the object store.
- `tests/test_evolution_v2_numerical_persistence.py`,
  `tests/test_evolution_v2_numerical_artifacts.py`, and
  `tests/test_evolution_v2_real_numerical.py`: **287 passed**.
- `tests/test_task_local_ensemble_cli.py` and
  `tests/test_package_numerical_supply.py`: **30 passed**.
- `tests/test_evolution_v2_frozen_shortlist.py`: **22 passed** (128.65 seconds).
  The suite verifies v1 replay, v2 exact schemas, reference/weight/fallback
  mutation rejection, byte-only Kernel success/rejection, completed seed reload,
  and final catalog-preserving projection persistence.
- `git diff --check`: clean.

## Existing baseline failures, outside Task 5B

`test_qd_projection_prefers_new_cell_coverage_and_binds_all_sources` raises
`complete catalog has conflicting task materialization`. Reproduced by loading
`git show HEAD:evolving_loop/v2/numerical_qd/adapters.py` into the adapter module
and running that exact test; the same failure occurs with the working adapter.

`test_kernel_typed_promotion_binds_full_train_evaluation_and_exact_winner[False]`
expects one accepted step but gets zero. Reproduced with both adapter and runner
modules loaded from their exact HEAD sources (27.96 seconds), as well as the
working implementation. The existing freeze/projection behavior was kept out
of scope. The new final-projection test uses an empty archive to exercise
evidence persistence without that pre-existing duplicate materialization path.
