# Task 6 Report: Migration audit and one-task smoke

## Status

Implementation and local verification are complete on `feature/morphology-single-smoke`.
The scoped commit is created after this report is finalized. Nothing was pushed or merged.

## Changes

- Added a semantic AST audit over the active Numerical evolution modules. Legacy
  MASE/MAE/sMAPE/RMSSE operations are permitted only in an explicit function-level
  allowlist for diagnostics, serialization, or opt-in legacy readers.
- Migrated Morphology assumption ranking and package ranking from legacy MASE summaries
  to capped Dr-CiK sMAE/sRMSE joint summaries and per-metric stability fields.
- Removed the final active MASE seam from residual correction and long-horizon handling:
  residual clipping scales now come from aligned historical fold truth, never from
  `mase_scale`, MAE, or MASE diagnostics.
- Made both explicit legacy Decision readers reject `median_smape`, which has no valid
  lossless migration into the scaled pair policy.
- Made active Markdown renderers consume the already-bound canonical metric-policy
  fingerprint instead of synthesizing a missing value.
- Extended the deterministic morphology smoke artifact with a canonical `selection`
  contract, read-only evolution boundary, and separate Statistical, TSFM, and Combined
  execution/failure summaries.
- Updated the English/Chinese Numerical documentation to state the schema-v2 scaled-pair
  authority while retaining the prior 99-task numbers unchanged and explicitly historical.
- Migrated one stale mixed-runtime integration fixture to the exact schema-v2 dictionary and
  config envelope required by the active loader.

## TDD evidence

- Initial Task-6 contract RED: the audit found legacy ranking fields in active Morphology
  paths, the renderer test found fallback fingerprints, and the fake smoke lacked its
  selection/evolution/family contract.
- Explicit legacy Decision RED: `median_smape` was advertised by migration maps but could
  not be consumed by the active policy.
- Deep active-path RED after the first pass:
  `2 failed, 1 passed`; the semantic audit and behavior test proved that changing only the
  legacy `mase_scale` changed residual-correction behavior.
- After the fix, the exact audit/behavior slice passed: `3 passed, 122 deselected`.
- Selector and smoke compatibility passed: `125 passed`.
- Independent review found that two scaled-only consumers still used a different
  tie-break sequence than `DecisionPolicy`. Two regression tests failed before the fix;
  assumption leaders and ranked alternatives now share the canonical
  median-joint/recent-joint/worst-joint/median-sMAE/median-sRMSE/name order.

## Deterministic task_42 fake smoke

Command:

```bash
python -m numerical_agent.run_morphology_smoke \
  --task-file /Users/yyoraa/time-series/external/Dr-CiK/sample/tasks/task_42.json \
  --results-path /tmp/numerical-task42-fake-smoke-task6-final.json \
  --llm-backend fake \
  --hindcast-folds 2
```

Observed artifact:

- Metric-policy fingerprint:
  `fe5cf0fd10839f93a3dea81cb90a63641ad520a589bdce875bd11095a2a8bf8d`.
- Decision metrics: `smae`, `srmse`; Dev read-only; Public/hidden mutation disabled.
- Statistical: 6 attempted, 6 successful.
- TSFM: 5 attempted, 0 successful, 5 explicitly unavailable.
- Combined: 5 attempted, 0 successful, 5 explicitly unavailable because their required
  leaves did not materialize.
- Selected candidate: `naive_last` with weight 1.0.
- Post-freeze trusted execution diagnostics: sMAE `0.2248109293847605`, sRMSE
  `0.2750891381127843`, MAE `104.47841999999999`, MASE `1.3769179524948298`.

This is an execution smoke only, not a benchmark result. In particular, unavailable TSFM
and Combined families remain visible and are not relabelled as successful.

## Real task_42 smoke decision

No real-checkpoint smoke was run. Local Python environments and a worker-config file exist,
but no checked-in attestation proves that the exact immutable checkpoints were loaded and
successfully smoked. `docs/tsfm-runtime-matrix.md` explicitly records that no such attestation
exists, and the repository search found no attestation artifact. Per the Task-6 safety boundary,
an environment directory is not sufficient evidence and no model was downloaded, substituted,
or fabricated.

The exact prerequisite for the real smoke is an operator-reviewed artifact bundle
(`methods.py`, `skills.py`, `policies.py`, Screening and Decision sources), a local deployment
binding, and a persisted attestation for the exact checkpoint/adapter/runtime identity (including
the immutable checkpoint or model digest) showing a successful real-checkpoint load.

## Verification

- Required focused suite before review: `505 passed in 6.07s`.
- Final expanded focused suite after the review fix: `514 passed in 6.11s`.
- The first full-suite attempt exposed one stale schema-v1 integration fixture:
  `1 failed, 2275 passed, 1 skipped in 55.83s`. The fixture was migrated through the explicit
  legacy dictionary converter into the strict schema-v2 envelope.
- Pre-review full suite: `2277 passed, 1 skipped in 54.77s`.
- Final post-review full suite: `2279 passed, 1 skipped in 56.61s`.
- `python -m compileall -q common numerical_agent`: passed.
- `git diff --check`: passed.
- Both HTML documents passed duplicate-ID and missing-fragment-link checks.
- `runs/numerical_morphology/` remained 10 files with the same aggregate SHA-256 before and
  after Task 6:
  `703fbe89573c7e86a6e96a50f79629c08d8a7bddc90113663ddd340a52501b25`.

## Self-review

- Confirmed active ordering reads capped joint sMAE/sRMSE summaries and active acceptance
  retains independent per-metric guards; diagnostic legacy fields are not decision inputs.
- Confirmed Statistical, TSFM, and Combined candidates use the same scaled scoring kernel and
  smoke failures remain family-specific.
- Confirmed active schema/fingerprint consumers fail closed; report renderers cannot invent a
  missing fingerprint.
- Confirmed Dev remains read-only and no 80/20, Public-99, official hidden, or probabilistic
  evaluation was run.
- Confirmed the old 99-task tables and their numeric values were not rewritten.
- Independent review reported no other Critical or Important findings after identifying
  the two ordering inconsistencies above; both were fixed and reverified.

## Concerns

The repository is not currently eligible for the requested real-checkpoint smoke because it lacks
the required immutable load attestation. This is an honest prerequisite gap, not a failed
forecasting result.
