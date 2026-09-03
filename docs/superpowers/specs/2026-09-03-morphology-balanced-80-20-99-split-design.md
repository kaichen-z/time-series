# Morphology-Balanced Dr-CiK 80/20/99 Split v2

Date: 2026-09-03  
Status: approved in chat; implementation pending  
Scope: local labeled Dr-CiK `public_dev` tasks only

## 1. Problem

The existing manifest, `splits/drcik_public_80_20_99_v1.json`, is deterministic and
entity-disjoint, but its assignment objective balances only frequency, forecast-horizon bucket,
reasoning hops, and origin. On the current baseline ledger, most fixed forecasters have materially
higher capped sMAE and sRMSE on Public-99 than on Train-80. The effect is visible even for methods
that never participate in self-evolution, so it is not solely an evolved-policy overfit.

The original split did not use future values or model errors, which was correct. The missing piece
is label-free forecast-difficulty stratification: historical periodicity, trend, intermittency,
noise, outliers, stationarity, recent regime change, available-history length, and the ratio between
forecast horizon and history length.

## 2. Goals

Create a new deterministic manifest, `splits/drcik_public_80_20_99_v2.json`, that:

- contains exactly 80 Train, 20 Dev, and 99 Public Regression tasks;
- keeps every entity wholly inside one partition;
- balances coarse metadata and history-only morphology across all three partitions;
- never reads or derives from future values, ground-truth evidence, document labels, forecast
  errors, model names, or prior evaluation outcomes;
- records enough schema and distribution metadata to reproduce and audit the assignment; and
- leaves v1 byte-for-byte unchanged for historical result reproduction.

## 3. Non-goals and Scientific Boundary

The v2 split is not a newly unseen test set. The project has already observed results on all 199
labeled tasks and used the old Public-99 repeatedly. V2 is therefore a better-balanced development
and engineering-regression protocol, not fresh generalization evidence. The official Hidden-80
submission remains the only locally unscored final test.

V2 selection must not optimize equal baseline accuracy across partitions. Baseline error is used
only after the manifest is frozen to audit whether history-profile balance reduced the observed
difficulty gap. The assignment algorithm itself is blind to all future labels and model outputs.

## 4. Approaches Considered

1. Repartition only the existing Train-80 and Dev-20. This preserves Public-99 identity but cannot
   address the measured Train/Public difficulty imbalance.
2. Repartition all 199 labeled tasks using history-only morphology while preserving v1. This is the
   selected approach because it addresses the imbalance without label-based task selection.
3. Replace 80/20/99 with repeated group cross-validation. This would improve uncertainty estimates
   but would remove the fixed protocol required by the current evolution and frozen-evaluation
   interfaces.

## 5. History-Only Profile

Each public task is converted to the same deterministic TaskProfile semantics used by Numerical
screening. Only these inputs are allowed:

- `series.history_values`;
- `task_metadata.frequency`;
- `task_metadata.prediction_length`; and
- the entity name, used only for grouping.

The split profile contains closed, coarse buckets rather than raw floating-point measurements:

- frequency;
- horizon bucket and history-length bucket;
- horizon/history ratio bucket;
- trend direction and trend-strength bucket;
- periodicity-strength and periodicity-confidence buckets;
- intermittent/dense class and zero-fraction bucket;
- signed/nonnegative and integer/continuous flags;
- noise-scale and outlier-fraction buckets;
- stationary/nonstationary class; and
- recent-regime confidence bucket.

Bucket boundaries are fixed constants in code and versioned by a profile-schema identifier. They
are selected before any v2 baseline evaluation. Continuous values are clipped to finite bounded
ranges before bucketing; empty or non-finite histories fail closed.

The manifest stores only aggregate bucket counts and task/entity membership. It does not copy raw
history, future values, task-level profiles, documents, or labels.

## 6. Assignment Objective

Assignment remains deterministic and entity-disjoint. Candidate assignments are generated from a
fixed seed and stable hashes exactly as in v1. For each assignment, the host computes:

1. exact requested-size feasibility;
2. normalized absolute distribution error for the original metadata strata;
3. normalized absolute distribution error for every morphology bucket;
4. the maximum single-bin deviation across partitions; and
5. a deterministic membership signature for tie-breaking.

Candidates are ordered lexicographically by:

1. lowest maximum morphology-bin deviation;
2. lowest total morphology imbalance;
3. lowest original-metadata imbalance; and
4. stable membership signature.

This prevents a large set of easy bins from hiding one badly imbalanced rare regime. No term in the
objective may depend on target values after the forecast origin or on any model prediction.

## 7. Manifest Contract

V2 increments the split schema and adds:

- `profile_schema` and its SHA-256 fingerprint;
- `selection_uses_history_values: true`;
- explicit false declarations for future values, GT evidence, document labels, and model metrics;
- morphology stratification feature names and bucket boundaries;
- per-partition metadata and morphology distributions;
- objective components for the accepted assignment; and
- a canonical manifest SHA-256 covering every field except the digest itself.

Consumers continue reading `partitions.<name>.task_ids`, so existing Train/Dev/Public loaders need no
semantic change. Formal runners opt into v2 explicitly at first; repository-wide defaults do not
switch until baseline regrouping and integrity checks pass.

## 8. Validation and Tests

Tests must be written before production changes and must prove:

- exact 80/20/99 sizes, full task coverage, unique IDs, and entity disjointness;
- deterministic output under input reordering;
- no future/GT/document/model-result content reaches the profile or manifest;
- changing only forbidden fields cannot change the split;
- changing a permitted history profile can change the balance score;
- malformed, empty, or non-finite history fails closed;
- v1 generation and the committed v1 artifact remain unchanged;
- v2 has lower maximum and total history-profile imbalance than v1 on the 199-task source; and
- every existing consumer can load a v2 fixture through the unchanged task-ID contract.

After v2 is generated and frozen, fixed baseline forecasts are regrouped without tuning. The audit
reports per split:

- mean and standard error of capped sMAE and sRMSE;
- medians, P90/P95, raw-tail maxima, and clipped-task counts;
- results by frequency, horizon ratio, and morphology group; and
- Train/Public gaps with paired uncertainty where task identity permits it.

Failure to reduce the label-free profile imbalance rejects v2. Failure to equalize realized
forecast errors does not trigger another task reshuffle; it is reported as residual distribution
shift.

## 9. Migration

1. Add the v2 profile/assignment implementation and tests without changing defaults.
2. Generate and commit `drcik_public_80_20_99_v2.json` from the same 199 public labeled records.
3. Verify v1 immutability and emit a v1-versus-v2 balance report.
4. Regroup or rerun fixed baselines once on v2 and publish the difficulty audit.
5. If integrity and balance gates pass, run future Numerical evolution with explicit v2 paths.
6. Keep every existing v1 result labeled with its original manifest digest; never compare v1 Train
   directly with v2 Public as if they were one protocol.

## 10. Expected Outcome

V2 should make Train, Dev, and Public more similar in observable historical morphology, so Train
cross-fit gains are a less optimistic guide to Public behavior. It cannot guarantee identical
forecast errors because entity-specific future regimes remain unknown at split time. The primary
success criterion is a demonstrably better label-free balance with intact leakage boundaries, not
post-hoc equalized accuracy.
