# Specialist Atlas Routing for Numerical Evolution

Date: 2026-09-02
Status: approved in chat; implementation pending
Scope: Numerical Dictionary discovery and task-local selection only

## 1. Goal

Turn the existing reviewed Dictionary into a task-conditioned specialist supply instead of asking
an LLM to guess arbitrary parent combinations. The system must learn, from the 80 Train tasks only,
which existing Statistical, TSFM, and Combined candidates reliably complement the active Champion.
At inference it combines this cross-fitted prior with the current task's history-only hindcasts and
returns either a conservative local blend or the exact Champion forecast.

This change does not train, fine-tune, merge, or alter any TSFM parameters. It fits only a small,
deterministic, interpretable routing table over already materialized forecasts.

## 2. Evidence and current limitation

A read-only analysis of the current ForecastStore found 81 active reviewed candidates: 72
Statistical, 5 TSFM, and 4 Combined. On the 80 Train tasks:

- Toto mean joint scaled error is 0.4091;
- the future-aware per-task oracle over the current global top eight is 0.3143, a 23.2% lower
  diagnostic upper bound, with 48 tasks improved;
- the future-aware oracle over all 81 active candidates is 0.2627, a 35.8% lower diagnostic upper
  bound, with 65 tasks improved.

These oracle numbers use Train future values and are not attainable runtime results. They establish
only that the Dictionary contains complementary signal. In contrast, the existing task-local OOF
run improved joint error by about 2.25% but failed activation precision and regret gates. The later
confidence-routed version activated only five Train tasks and zero Dev tasks. The bottleneck is
therefore candidate discovery and routing calibration, not absence of candidate forecasts.

The existing `_bounded_names` ranks candidates mainly by global absolute performance and family
coverage. It does not optimize conditional improvement over the Champion or complementary task
coverage. Discrete exact/coarse morphology buckets are also too sparse for only 80 Train tasks.

## 3. Approaches considered

1. Continue LLM structural search. This preserves the current architecture but parent discovery is
   weakly informed and has produced narrow one-task activations.
2. Build a deterministic Specialist Atlas and cross-fitted local calibrator. This is the selected
   approach because it uses cached evidence, remains interpretable, and does not train a forecast
   model.
3. Train an end-to-end meta-selector. This is out of scope because the dataset is small and the
   project does not want another learned neural component.

## 4. Specialist Atlas

For every reviewed candidate and every Train task, trusted host code records a paired comparison to
the active Champion using capped and raw sMAE and sRMSE. The Atlas includes only derived,
task-identity-free routing features:

- deterministic TaskProfile fields and coarse history/horizon buckets;
- history-only hindcast deltas, fold coverage, dispersion, and worst-fold regret;
- candidate availability and family;
- normalized forecast disagreement with the Champion;
- paired future deltas used only as the Train fitting target.

All fitting is five-fold group-aware cross-fitting. Entity-, source-, or identical-history-connected
tasks remain in one fold. A held-out task and every connected task are excluded from the Atlas used
to route that task.

Candidate discovery is based on conditional uplift and complementarity, not global mean rank. A
greedy host-owned selection retains the Champion plus at most eleven specialists that add distinct
cross-fitted task coverage. A candidate must have finite execution, support in multiple independent
groups, and bounded raw regret. Statistical, TSFM, and Combined candidates use the same evidence
contract; no family receives unconditional retention.

## 5. Interpretable local calibrator

For each candidate, the router estimates paired improvement using a deterministic nearest-neighbor
lookup over the Atlas. Distance uses a small frozen feature set:

- periodicity, trend, intermittency, recent regime, and sign;
- frequency, history-length bucket, and horizon ratio;
- paired hindcast sMAE/sRMSE margins and their dispersion;
- forecast disagreement with the Champion.

Numeric features are robustly scaled from the four fitting folds. Categorical mismatches use fixed
host-owned penalties. The neighbor count is selected from a small closed grid inside the Train
cross-fit. Each prediction is shrunk toward the global candidate prior when local support is sparse.
The output is an empirical probability of joint improvement, robust sMAE/sRMSE effect bounds,
independent-group support, and a regret bound. No free-form LLM output participates in routing.

A candidate qualifies only when:

- both predicted metric effects are nonnegative and the joint effect is positive;
- the lower confidence bound remains positive after shrinkage;
- at least three current-task paired hindcast origins support the same direction;
- estimated and observed worst-fold/raw regret remain within the frozen budget; and
- evidence comes from at least two independent Train groups.

## 6. Forecast construction

The existing task-local executor remains the numerical authority. For each task it receives the
Champion and at most two qualified, behaviorally diverse specialists. It searches a fixed 0.1 weight
grid with Champion weight at least 0.5. Optional early/late horizon routing is allowed only when both
regions have at least three genuine paired origins; otherwise the full-horizon route is used.

If evidence is missing, contradictory, nonfinite, or below the confidence margin, the output is the
exact Champion forecast byte-for-byte. Candidate failure never changes the Champion result.

## 7. Self-evolution role

The first Atlas version evaluates the existing Dictionary without an LLM. After cross-fitted routing,
the host groups unresolved Champion losses by sanitized morphology and error signatures. Only those
residual aggregates may be sent to the Numerical proposer.

The proposer may suggest:

- an existing specialist family to add to the Atlas supply;
- a closed Statistical/TSFM/Combined recipe using reviewed parents; or
- a new Statistical method implementation through the existing Git evolution path.

The proposer never chooses weights, metric thresholds, neighbors, promotion, or task identities.
Every new Child re-enters the same Atlas cross-fit. Thus the agent evolves the Dictionary against
specific uncovered residual regimes rather than guessing combinations globally.

## 8. Lifecycle and data boundaries

1. Use 64 Build tasks to compare candidate supplies and the closed calibrator grid by group-aware
   OOF evidence.
2. Use the remaining 16 Train tasks as Calibration without changing candidate structure afterward.
3. If both Train stages pass strict dual-metric, coverage, precision, clipping, and tail gates,
   reconstruct the frozen Atlas from all 80 Train tasks.
4. Open 20 Dev tasks exactly once for acceptance without refitting.
5. If Dev accepts, evaluate Public-99 exactly once as report-only regression evidence.

Dev or Public results never return to Atlas construction or the proposer. A rejected run preserves
the exact Parent release.

## 9. Fast experiment and success criteria

The first experiment is cache-only and does not call an LLM or a TSFM runtime. It compares, on 80
Train group-aware OOF:

- fixed Toto;
- existing task-local routing;
- global top-eight supply with the new calibrator; and
- Atlas-selected supply with the new calibrator.

Proceed to the sealed 20 Dev only if the Atlas variant:

- Pareto-improves mean capped sMAE and sRMSE, with one strict gain;
- improves mean joint scaled error by at least 1% over Toto;
- has at least 60% activated-task win precision;
- activates in at least two independent groups;
- does not regress capped or raw P90/P95 tails, clipping, failures, or coverage; and
- keeps maximum task regret within the existing 0.25 limits for both metrics.

The 1% floor is an experiment screen, not a promise. Oracle figures must never be reported as achieved
performance. Public-99 is opened only after Dev acceptance.

## 10. Implementation boundary

Add a focused Atlas/calibration module and a cache-only experiment runner. Reuse the existing
ForecastStore, grouping manifest, TaskProfile, CandidateDiagnostics, task-local executor, metric
implementation, release fingerprints, and Dev/Public gates. Do not modify Retrieval or Decision.

Only Critical correctness, leakage, or metric-authority failures block this experiment. Important and
Minor hardening outside the direct experimental path is recorded and deferred, matching the agreed
research-project reliability standard.
