# Dr-CiK Scaled-Metric Unification Design

Date: 2026-08-30

Status: approved in chat for specification; implementation pending

## 1. Objective

Unify every performance-based Numerical Agent decision around the two Dr-CiK
point-forecast metrics:

- `sMAE`: MAE divided by the mean absolute value of the scored truth horizon;
- `sRMSE`: RMSE divided by the same scale.

Each per-task metric is capped at `5.0` before cross-task aggregation. The two
metrics have equal standing. MASE, MAE, sMAPE, and RMSSE may remain in artifacts
for diagnostics and backward-readable reports, but they must no longer determine
filtering, ranking, mutation targets, Child acceptance, or release selection.

This is a workflow-contract migration, not a display-only rename.

## 2. Scope

The migration covers all performance-based Numerical stages:

1. Statistical-method execution outcomes and cached evaluation records;
2. Dictionary filtering and method status evolution;
3. Task-conditioned Screening evolution;
4. Historical rolling hindcast diagnostics;
5. Runtime Numerical candidate selection and Safe-Anchor protection;
6. Combined candidate proposal, comparison, and evolution;
7. Morphology hypothesis credit, consistency gates, and assumption-guided ranking;
8. Train screening, full Train evaluation, read-only Dev acceptance, release freezing,
   and reports;
9. Frozen Public evaluation and future hidden-submission output summaries.

Deterministic applicability checks remain feature-based. For example, Croston may
still require an intermittent, nonnegative series. The metric migration applies
whenever observed performance is used to choose, filter, rank, repair, quarantine,
accept, or reject something.

Retrieval evidence verification and non-numerical safety gates are outside this
migration unless they consume a Numerical performance score.

## 3. Canonical metric kernel

There will be one canonical implementation for both historical hindcasts and
labelled Train/Dev/Public evaluation.

For one scored horizon:

1. `scale = mean(abs(truth))`;
2. `sMAE_raw = MAE(truth, prediction) / scale`;
3. `sRMSE_raw = RMSE(truth, prediction) / scale`;
4. `sMAE = min(5.0, sMAE_raw)`;
5. `sRMSE = min(5.0, sRMSE_raw)`.

If `scale` is zero, an exact forecast receives zero and any nonzero error receives
the cap after its raw score becomes infinite. Non-finite inputs remain invalid.

The canonical record stores raw and capped values plus clipping flags. Cross-task
means always use capped values. Raw values remain available for tail-risk audits.

### Runtime label boundary

Runtime selection never sees the task's real future. It computes sMAE and sRMSE
only on validation windows cut from the observed history. Train-only evolution may
use Train labels. Dev labels are read once by the trusted evaluator only. Public
and hidden outcomes never feed proposals or mutable policies.

## 4. Joint objective and safety contract

The common scalar used only for deterministic ordering is:

`joint_scaled_error = (sMAE + sRMSE) / 2`

The scalar does not replace the two metric fields. Every acceptance gate checks
sMAE and sRMSE separately.

### Candidate eligibility

A candidate is eligible only when it:

- meets the configured successful-fold coverage;
- has finite forecasts of the exact requested horizon;
- passes explosion, clipping, provenance, and runtime checks;
- has finite sMAE and sRMSE records for the required folds.

### Candidate ranking

Eligible candidates are ordered by:

1. safety violations and catastrophic-tail flags;
2. median joint scaled error;
3. recent-window joint scaled error;
4. worst-fold joint scaled error;
5. median sMAE;
6. median sRMSE;
7. deterministic candidate name.

Recent-regime routing may change which validated window is considered "recent",
but it may not substitute MASE or sMAPE into the ranking.

### Parent/Child acceptance

A Child passes a Train or Dev metric gate only when, within the configured numerical
tolerance:

- its aggregate sMAE is no worse than the Parent;
- its aggregate sRMSE is no worse than the Parent;
- at least one of the two is strictly better;
- coverage, invalid/crash exposure, clipped-task counts, and required tail quantiles
  do not regress.

This is a Pareto-style acceptance rule. An improvement in one metric cannot buy an
unbounded regression in the other.

## 5. Stage behavior

### 5.1 Method execution and dictionary filtering

Every successful method outcome records sMAE and sRMSE. Filtering diagnostics group
methods by task morphology and compare each method with the configured baseline using
both deltas:

- `delta_sMAE = method_sMAE - baseline_sMAE`;
- `delta_sRMSE = method_sRMSE - baseline_sRMSE`.

`keep` and `specialized` require repeatable evidence under the joint objective and
must not be justified by MASE alone. Crash, invalid, and genuine NotApplicable states
remain separate from poor forecasts. Repair/quarantine decisions use both scaled
metrics plus execution health.

### 5.2 Task-conditioned Screening

Applicability clauses still use deterministic TaskProfile fields such as periodicity,
trend, intermittency, regime, frequency, history length, and horizon. Performance
evidence for learning those clauses changes to group-wise sMAE and sRMSE deltas.

Oracle retention is defined by the best eligible joint scaled error, with ties retaining
all Pareto-equivalent candidates. Screening acceptance continues to constrain candidate
pool size, crash/invalid exposure, dictionary diversity, and oracle retention.

### 5.3 Historical hindcasting and Numerical Selector

`HindcastFold` and `CandidateDiagnostics` gain first-class sMAE/sRMSE fields for median,
recent, worst, dispersion, and clipping. MASE-based catastrophe and ranking fields are
replaced by scaled-metric equivalents.

The Safe-Anchor can be overridden only when the challenger passes separate sMAE and
sRMSE regret gates on all required folds and any long-horizon audit. Ensemble and
Combined candidates are replayed from materialized leaf forecasts and scored by the
same contract.

### 5.4 Combined evolution

The proposer receives group-level winsorized delta-sMAE and delta-sRMSE, coverage,
tail-risk, disagreement, and parent provenance. It may create Statistical+Statistical,
TSFM+TSFM, or TSFM+Statistical compositions, subject to existing execution and identity
constraints.

Successive-halving screens, full Train comparison, and Dev acceptance all use the
Pareto rule. The LLM never receives Dev labels or raw future values.

### 5.5 Morphology assumptions

The Morphology Reasoner still produces evidence-bound hypotheses, not forecast values.
Candidate quality, regret, and counterfactual credit fields presented to it use sMAE
and sRMSE. The consistency gate rejects an assumption when referenced candidates lack
enough scaled-metric folds or would bypass either Safe-Anchor metric guard.

### 5.6 Reporting

All new result summaries lead with:

- mean and median capped sMAE;
- mean and median capped sRMSE;
- standard errors and configured tail quantiles;
- coverage, invalid/crash counts, and clipped-task counts;
- paired wins/ties/losses under the joint objective.

MASE, MAE, sMAPE, and RMSSE are labelled `diagnostic_only` when retained. They cannot
appear in an active policy's ranking or acceptance configuration.

## 6. Schema and compatibility

The migration must bump affected policy, cache, checkpoint, and frozen-release schema
versions. New active artifacts fail closed if required sMAE/sRMSE fields or metric-policy
fingerprints are missing.

Legacy MASE/sMAPE artifacts remain readable only through an explicit legacy/reporting
path. They are not silently converted into active policies because their historical
selection decisions were made under a different objective.

Existing published experimental results remain unchanged and continue to describe the
metric contract under which they were produced. New experiments receive new run IDs,
policy fingerprints, and result manifests.

## 7. Error handling and invariants

- Missing either scaled metric makes a performance record incomplete.
- Non-finite predictions remain invalid rather than receiving a misleading score.
- Cache keys bind metric schema, cap, aggregation, and ranking policy.
- Train, Dev, Public, and hidden split boundaries remain unchanged.
- Dev evaluation stays read-only; rejection returns the exact Parent policy.
- Public/hidden evaluation cannot write Skills, mutate prompts, or create Children.
- Statistical, TSFM, and Combined candidates use exactly the same scoring kernel.

## 8. Verification strategy

Implementation follows test-driven development with these required test classes:

1. Golden metric tests for zero-scale, near-zero, ordinary, clipped, and multi-trajectory
   inputs;
2. Outcome/cache/schema round trips containing both raw and capped metrics;
3. Adversarial tests proving MASE/sMAPE changes cannot alter active ranking or acceptance;
4. Screening tests where sMAE and sRMSE disagree;
5. Selector tests for Pareto ordering, separate tail gates, Safe-Anchor protection, and
   deterministic ties;
6. Combined evolution tests for one-metric regression, cross-fold regression, and
   accepted dual-metric improvement;
7. Morphology tests proving assumption guidance cannot bypass either metric gate;
8. Legacy-artifact tests proving explicit read-only compatibility and fail-closed active
   loading;
9. End-to-end fake single-task and 8/2 smoke runs;
10. A small real Numerical smoke run before any new 80/20 evolution.

The migration is complete only when searches of active ranking, filtering, mutation,
and acceptance paths find no MASE or sMAPE dependency and all affected tests pass.

## 9. Expected user-visible result

After implementation, every Numerical decision can be explained using the same pair:

> This candidate was retained, ranked, selected, or accepted because of its historical
> or Train-only sMAE and sRMSE performance, with separate protection against regression
> in either metric.

No component may claim Dr-CiK alignment while still making the underlying decision from
MASE or sMAPE.
