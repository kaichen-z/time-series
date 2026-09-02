# Task-Local Adaptive Ensemble Design

Date: 2026-09-02
Status: approved in chat; pending written-spec review
Scope: Numerical Agent Champion evolution and runtime selection only

## 1. Goal

Replace brittle globally fitted routing thresholds with a deterministic,
history-only local tournament for each task. The accepted Champion remains the
safe fallback. Dictionary candidates may override or blend with it only when
the current series' own rolling hindcasts provide stable paired sMAE and sRMSE
evidence.

The first implementation targets the current Toto Champion, but the interfaces
must accept any future reviewed Champion as the anchor.

## 2. Non-goals

- Do not train, fine-tune, merge, or alter TSFM parameters.
- Do not let an LLM choose numeric weights, thresholds, scores, or promotion.
- Do not use Dev, Public-99, hidden labels, Retrieval, or documents to construct
  a Numerical policy.
- Do not add horizon-segment-specific weights in v1; use one full-horizon
  weight vector per task.
- Do not search arbitrary candidate subsets at runtime. Screening and the
  frozen release provide a bounded reviewed candidate supply.
- Do not reopen the already-consumed Public-99 regression set.

## 3. Existing components to reuse

- `ForecastStore` for materialized leaf forecasts and cached hindcasts.
- `TaskProfile` and task-conditioned screening for bounded candidate supply.
- `CandidateDiagnostics` for history-only fold forecasts and failures.
- canonical capped/raw Dr-CiK sMAE and sRMSE metrics.
- Champion release, fallback, fingerprint, artifact, and read-only Dev gates.
- structural Champion proposal evolution for selecting candidate families;
  the proposer never owns local numeric weights.

## 4. Group-aware cross-fitting

The external 80-task Train partition is the only fitting/evolution authority.
It is divided into five deterministic outer folds by indivisible task groups.

Group membership is the transitive union of:

1. normalized nonempty `entity_name` equality;
2. exact canonical history fingerprint equality; and
3. an exact source-series identifier when the task metadata supplies one.

Future values and truth-derived fingerprints are forbidden in group creation.
Groups are assigned to folds with deterministic size balancing and morphology
stratification. No group may occur in two folds. The fold manifest and grouping
implementation fingerprint are bound into the run manifest and checkpoints.

For each outer fold, proposal structure and any global eligibility constants
are derived from the other four folds. The held-out fold produces only OOF
scores. Every reported Train task metric must therefore be out-of-fold.

After structure selection, the exact frozen mechanism is reconstructed from
all 80 Train tasks. The untouched 20-task Dev partition is opened exactly once
for acceptance. Public-99 remains unavailable to this workflow.

## 5. Conditional-uplift evidence

Every candidate is compared with the anchor only on tasks where the local gate
would activate it. Reports must separately contain:

- activation support and activation rate;
- activated-task sMAE and sRMSE win/tie/loss counts;
- mean and median paired deltas for both metrics;
- capped and raw P90/P95 tails;
- maximum per-task and per-fold regret;
- successful-fold coverage, crash, invalid, and fallback counts;
- exact-anchor equality on non-activated tasks.

A recipe with insufficient activated support is unproven, not good. Global
mean improvement cannot hide low activation precision behind fallback ties.
The proposer receives only bounded anonymous group aggregates, never task IDs,
truth arrays, forecasts, or per-task activation records.

## 6. Task-local tournament

For one incoming task, trusted host code performs the following using history
only:

1. Run the frozen screening policy and retain at most eight reviewed diverse
   candidates, always including the anchor.
2. Reuse or compute rolling-origin hindcasts for those candidates.
3. Require finite forecasts, minimum fold coverage, and exact paired folds with
   the anchor.
4. Remove candidates Pareto-dominated by the anchor on median sMAE and sRMSE.
5. Reject candidates whose worst fold, raw tail, failure rate, or instability
   exceeds the frozen risk budget.
6. Retain at most two specialists, preferring different forecast behavior and
   Dictionary families when robust scores tie.
7. Evaluate a small frozen weight grid with anchor weight in `[0.5, 1.0]`,
   nonnegative specialist weights, and weights summing exactly to one.
8. Rank weights by median joint scaled error, then worst-fold joint error,
   then median sMAE, median sRMSE, greater anchor weight, and canonical name.
9. Activate the ensemble only when it Pareto-improves the anchor and clears a
   frozen minimum robust margin. Otherwise return the anchor forecast exactly.

The host returns the selected names, weights, fold support, rejection/fallback
reason, and fingerprints. It never exposes fold truths or per-fold forecasts to
an LLM.

## 7. Evolution ownership

The self-evolve proposer may change only:

- which reviewed candidate families enter the bounded tournament;
- structural diversity requirements;
- which reviewed history-only diagnostic fields a recipe requests.

The host owns grouping, fold construction, metric calculation, Pareto checks,
risk budgets, numeric weight search, tie-breaking, and final acceptance.
Rejected OOF evidence may guide later Train proposals only through the existing
sanitized aggregate channel. The exact active Parent remains unchanged until
the one-shot Dev gate accepts the reconstructed v2 policy.

## 8. Acceptance

OOF Train promotion requires all of the following:

- 100% anchor fallback coverage;
- no regression in mean capped sMAE or mean capped sRMSE;
- a strict improvement in at least one primary metric;
- no raw/capped P90 or P95 regression beyond the frozen tolerance;
- no clipped-count, crash, invalid, or coverage regression;
- bounded maximum task and fold regret;
- minimum activated support and minimum activated-task win precision;
- improvements represented in multiple entity/history groups.

Dev applies the same frozen mechanism and thresholds without refitting. A Dev
failure preserves the Parent byte-for-byte. Public-99 is not evaluated again.

## 9. Failure behavior

- Missing or invalid specialist: remove that specialist and recompute only if
  the remaining frozen recipe is legal; otherwise exact anchor fallback.
- Missing anchor: typed task failure; never silently substitute another model.
- Insufficient hindcast folds or ambiguous evidence: exact anchor fallback.
- Group collision or fold leakage: reject the run before any proposer or Dev
  access.
- Nonfinite arithmetic or invalid weights: reject the local ensemble and use
  the anchor.
- LLM, cache, or runtime failure during Train proposal: preserve Parent and a
  typed rejection trace.

## 10. Implementation boundaries

Add one focused task-local selection module rather than expanding the existing
controller further. The expected surfaces are:

- immutable group-fold manifest and builder;
- immutable local tournament policy, evidence, and result dataclasses;
- deterministic tournament executor over existing diagnostics;
- conditional-uplift aggregation for Champion feedback;
- adapter from the accepted tournament result into `NumericalForecastPackage`;
- CLI/config bindings and immutable policy fingerprints.

The existing single/combined Champion runtime remains supported. The v2 path
is explicit and opt-in until its fake and OOF/Dev checks pass.

## 11. Tests and experiment

Implementation uses strict TDD and stops at Critical correctness boundaries.
Important/Minor hardening outside the experiment path is recorded for later.

Required tests:

1. entity/history groups never cross folds and are order-independent;
2. grouping never reads future values;
3. task-local weights are deterministic, normalized, and anchor `>= 0.5`;
4. sMAE-only or sRMSE-only improvements cannot bypass Pareto safety;
5. missing/invalid specialists return the exact anchor;
6. conditional-uplift support and W/T/L exclude fallback ties;
7. no labels, task IDs, raw forecasts, Dev, or Public enter proposal feedback;
8. accepted package replay is bit-exact and LLM-free;
9. deterministic fake integration plus the current 80-task OOF run;
10. exactly one frozen 20-task Dev evaluation if OOF gates pass.

Success means the v2 mechanism produces an OOF-safe policy and either passes
Dev or honestly retains Toto. A claimed improvement requires both capped sMAE
and sRMSE reporting; no result is selected from Public-99.
