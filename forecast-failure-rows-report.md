# Forecast failure rows report

Restored exception bindings in both forecast materialization paths.

- A full forecast exception now produces the intended failed row:
  `RuntimeError: forecast failed` for full materialization and
  `shortlisted_runtime_failure: RuntimeError: forecast failed` for a closed
  shortlist.
- A diagnostic-only failure for the Anchor still retains the valid full
  forecast, leaves `diagnostic` as `None`, and has no failure reason.

Verification:

- Focused RED: `pytest -q tests/test_task_local_ensemble_cli.py -k 'forecast_exception_materializes_failure_rows or anchor_diagnostic_failure_preserves_its_full_forecast'` — 1 failed with the expected unbound `error`, 1 passed.
- Focused GREEN: same command — 2 passed.
