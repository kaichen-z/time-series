# Task 3 report — P2 persistence handoff and real numerical Host

## Outcome

Implemented the public completed-P2 frozen-pair loader, the in-memory Numerical
QD payload entry point, the cache-only real Host, and the root-context P2 bridge.
The Host binds the historical cache-compatible ForecastStore identity exactly:

`90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2`

The Host loads only the frozen 80 Train plus 20 Dev IDs, uses the configured
Toto worker endpoint with `chronos,timesfm`, shares one
`CodexCLIClient(gpt-5.6-luna, medium)`, keeps Retrieval Skills read-only, uses
real Retrieval/Decision constructors, exposes an identified zero-external-work
resource reporter for cache hits, and closes both forecast and runtime owners.

## TDD evidence

### RED 1 — completed frozen-pair loader

Command:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py::test_completed_store_loads_exact_active_frozen_pair
```

Observed expected failure:

```text
AttributeError: 'NumericalQDRunStore' object has no attribute 'load_active_frozen_pair'
1 failed in 37.70s
```

The production change that makes this test pass is the authenticated public
loader. The fixture is a real deterministic P2 run; the final version limits
task budget to produce a valid seed-only completion and proves that no-promotion
completion restores the exact active pair.

### GREEN 1 — loader identity/corruption behaviors

Command:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py::test_completed_store_loads_exact_active_frozen_pair tests/test_evolution_v2_real_numerical.py::test_active_pair_loader_rejects_changed_or_duplicate_pair
```

Observed:

```text
2 passed in 40.99s
```

The loader authenticates the completed runner checkpoint, immutable catalog,
Kernel state, completion envelope, and active accepted Bundle. It validates all
pair-shaped artifacts, requires exactly one release/registry identity match,
and restores the envelope against the caller's complete 100-task authority.

### RED 2 — payload API and real Host

Command:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py::test_payload_entrypoint_runs_from_root_derived_payload_without_path_overlap tests/test_evolution_v2_real_numerical.py::test_real_host_owns_exact_tasks_cache_agents_and_resource_cleanup
```

Observed expected failures:

```text
AttributeError: module 'evolving_loop.v2.cli' has no attribute 'numerical_evolve_payload'
ImportError: cannot import name 'host' from 'evolving_loop.v2.real'
2 failed in 0.61s
```

An intermediate rerun produced one pass and one fixture-only failure because an
empty `PolicyPortfolio` violates the repository's fixed flagship contract. The
fixture was corrected to use `PolicyPortfolio.flagship5()`; production behavior
was not weakened.

### GREEN 2 — complete Task 3 behavior

Command:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py
```

Observed:

```text
4 passed in 14.00s
```

### RED/GREEN 3 — manifest Codex executable binding

Final self-review added an assertion that the Host binds the executable named
by the `codex_cli` runtime location, in addition to the contracted model and
reasoning effort. Before the production correction, the exact Host test failed:

```text
AssertionError: assert 'codex' == '<repo-root>/codex'
1 failed in 0.69s
```

After passing `runtime_locations["codex_cli"]` as `CodexCLIConfig.binary`:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py::test_real_host_owns_exact_tasks_cache_agents_and_resource_cleanup
1 passed in 0.71s
```

## Compatibility evidence

Legacy path CLI:

```text
python -m pytest -q tests/test_evolution_v2_numerical_cli.py
34 passed in 118.14s
```

The full numerical runner command reached one nondeterministic acceptance
failure after sustained suite load and was interrupted after the traceback:

```text
test_kernel_typed_promotion_binds_full_train_evaluation_and_exact_winner[True]
AssertionError: assert 0 == 1
1 failed, 44 passed in 794.88s
```

The immediately preceding `[False]` parameter had passed with the identical
`run_fixture(task_budget=920)` setup. Root-cause isolation then passed both the
exact failed case and the paired parameter set:

```text
python -m pytest -q 'tests/test_evolution_v2_numerical_runner.py::test_kernel_typed_promotion_binds_full_train_evaluation_and_exact_winner[True]'
1 passed in 30.91s

python -m pytest -q tests/test_evolution_v2_numerical_runner.py::test_kernel_typed_promotion_binds_full_train_evaluation_and_exact_winner
2 passed in 56.54s
```

This is timing/environmental behavior in the legacy budget-sensitive suite, not
a deterministic Task 3 regression. No runner or budget code was changed.

## Self-review

- The payload API retains the old descriptor-based path preflight for the CLI,
  while verified in-memory payloads use output-only admission.
- Caller-supplied input SHA records remain the exact adapter/operator authority;
  the derived config is detached and reparsed before execution.
- The public loader returns typed artifacts plus the object SHA only; it does
  not expose or return private catalog paths.
- A cache miss remains `ForecastStore`'s real `CacheMissError`; no Host layer
  substitutes a synthetic forecast.
- Mutation check: changed catalog bytes, ambiguous pair catalog, wrong active
  identities, wrong task envelopes, path-overlap regression, non-cache Host,
  wrong model binding, public-task loading, and missing resource closure are all
  covered by the new tests or existing exact contracts.
- `evolving_loop/v2/real/runner.py` and `evolving_loop/v2/real/__init__.py` were
  not touched.

## Final verification

The earlier combined behavior/regression verification was green before the
one-line, Host-only executable binding:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py tests/test_evolution_v2_numerical_cli.py
38 passed in 129.82s
```

The post-binding combined rerun completed 37 tests and hit a different legacy
smoke timing failure: its fixed wall budget recorded 30.357971 seconds and
finalized with zero archive cells, failing `occupied_cells >= 1`:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py tests/test_evolution_v2_numerical_cli.py
1 failed, 37 passed in 136.16s
```

The exact legacy case immediately passed in isolation, confirming it was not a
deterministic result of the Host binding:

```text
python -m pytest -q tests/test_evolution_v2_numerical_cli.py::test_smoke_executes_real_runner_and_writes_truthful_completion
1 passed in 35.37s
```

Fresh Task 3 verification after the final Host binding:

```text
python -m pytest -q tests/test_evolution_v2_real_numerical.py
4 passed in 13.51s
```

Fresh syntax, import, identity, and whitespace verification:

```text
python -m py_compile evolving_loop/v2/cli.py evolving_loop/v2/numerical_qd/persistence.py evolving_loop/v2/real/host.py evolving_loop/v2/real/bridges.py tests/test_evolution_v2_real_numerical.py
python -c 'from evolving_loop.v2.real.host import EXPECTED_REAL_FORECAST_STORE_IDENTITY, RealHostRuntimeV2, build_real_host; from evolving_loop.v2.real.bridges import run_real_numerical; from evolving_loop.v2.cli import numerical_evolve_payload; print(EXPECTED_REAL_FORECAST_STORE_IDENTITY)'
git diff --check
exit 0; identity 90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2
```
