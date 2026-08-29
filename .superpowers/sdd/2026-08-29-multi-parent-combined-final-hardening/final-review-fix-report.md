# Final Review Fix Report

Base commit: `5863fc6eaa189f7e313ec772915b28e03138d83f`

## Scope

Fixed exactly the three Important findings from the final review:

- even-cardinality Combined median arithmetic at finite float extremes;
- at-most-once in-process materialization of shared leaf outcomes for every
  status, without persisting retryable failures;
- runtime-derived frozen evaluator ranking validation for the exact
  Statistical plus current TSFM/Combined namespace.

No LLM or TSFM weights, model identities, checkpoints, runtime options,
history-only boundaries, parent graph rules, or formal splits changed.

## RED evidence

### Even median

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_portfolio.py \
  -k 'even_median_is_finite_for_extreme_values'
2 failed, 2 passed, 61 deselected in 0.15s
```

The two- and four-parent same-sign `sys.float_info.max` cases produced a
non-finite median and silently used `fallback=toto_2_0`. The cancellation cases
already returned zero.

### Shared failed leaf

```text
../../.venv/bin/python -m pytest -q tests/test_numerical_selector_script.py \
  -k 'shared_failed_tsfm_leaf'
2 failed, 31 deselected in 0.39s
```

Both `INVALID` and `CRASHED` TSFM leaves executed twice when two Combined
policies shared the same forecast key.

### Frozen ranking namespace

```text
../../.venv/bin/python -m pytest -q tests/test_frozen_two_stage_evaluation.py \
  -k 'runtime_ranking_namespace or ranking_namespace_before_forecasting'
4 failed, 1 passed, 9 deselected in 0.21s
```

The valid 104-candidate portfolio failed the fixed 103 gate. Missing,
duplicate, and extra rankings also reached only the fixed-count error rather
than an exact runtime namespace check.

## Implementation

- Median sorts each point once, returns the center for odd cardinality, and
  sends only the two centers of an even cardinality through the existing
  overflow/cancellation-stable mean.
- `ForecastStore` memoizes canonical materialized leaf `Outcome`s by its
  existing content-addressed forecast key for the lifetime of one store.
  `INVALID` and `CRASHED` remain absent from the persistent cache, so a fresh
  store retries a failed transport.
- The frozen evaluator validates the portfolio namespace, derives candidate
  names as `tuple(module.names()) + portfolio.names`, and rejects duplicate,
  missing, or extra ranking names before `_training_outcomes` or forecasting.

## GREEN evidence

Focused TDD cycles:

```text
median: 5 passed, 60 deselected in 0.07s
shared leaf/status/retry: 4 passed, 29 deselected in 3.27s
ranking namespace: 5 passed, 9 deselected in 0.14s
```

The shared-leaf regression verifies one failed TSFM runtime call across two
Combined consumers in one store, then a second call from a fresh store for both
`INVALID` and `CRASHED`.

Required focused suite:

```text
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py \
  tests/test_numerical_selector_script.py \
  tests/test_frozen_two_stage_evaluation.py
112 passed in 1.53s
```

Branch-wide suite:

```text
../../.venv/bin/python -m pytest -q
1939 passed, 1 skipped, 12 warnings in 58.25s
```

The warnings are third-party statsmodels deprecation/non-stationary parameter
warnings from existing evolution-loop tests.

## Final static verification

The final freshness run covers:

```text
../../.venv/bin/python -m compileall -q numerical_agent tests
bash -n scripts/run_task_conditioned_screening.sh
git diff --check
```

All commands exit zero without output.

## Delivery

One scoped Conventional Commit; no push or merge.
