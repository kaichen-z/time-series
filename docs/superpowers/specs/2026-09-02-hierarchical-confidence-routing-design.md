# Hierarchical Confidence Routing for Numerical v2

Date: 2026-09-02
Status: approved in chat; awaiting written-spec review
Scope: Train-only refinement of the task-local Numerical ensemble

## 1. Motivation

The first 80-task out-of-fold run improved mean sMAE from 0.31230 to
0.30669 and mean sRMSE from 0.50593 to 0.49309, but it activated on 28 tasks
with only 16 wins and 12 losses. Activation precision was 57.1% instead of
the required 60%, and maximum task regrets of 0.2534 sMAE and 0.2505 sRMSE
slightly exceeded the 0.25 limits.

The direction therefore has measurable value, but current routing relies too
heavily on a task's small three-fold hindcast. The next version must improve
the reliability of the activation decision rather than weaken the acceptance
gate.

## 2. Goal and non-goals

Build a deterministic hierarchical confidence gate that combines:

1. cross-fitted evidence from similar Train tasks;
2. the current task's history-only rolling hindcasts; and
3. optional early/late horizon evidence when each region has enough support.

Toto remains the exact safe anchor. A specialist or ensemble may override it
only when both the group prior and local evidence support a Pareto-safe gain in
sMAE and sRMSE.

This change does not train, fine-tune, merge, or modify TSFM parameters. It
does not use Dev or Public labels for fitting. It does not lower the existing
OOF, tail, clipping, coverage, or Dev acceptance standards.

## 3. Cross-fitted hierarchical evidence

For each held-out Train group and each candidate or legal weight recipe, fit
evidence using only the other four outer folds. Evidence is organized into
three levels:

- exact morphology: frequency, history/horizon buckets, periodicity, trend,
  intermittency, recent regime, and sign;
- coarse morphology: periodicity, trend, intermittency, and recent regime;
- global candidate evidence.

Use the most specific level with enough independent task groups. Minimum
support is eight tasks for exact morphology, twelve for coarse morphology,
and twenty for global evidence. Smaller samples fall back to the next level;
they never borrow the held-out task or its connected entity/history group.

Each evidence record stores only aggregate paired values:

- independent task-group support;
- wins, ties, and losses versus Toto for sMAE, sRMSE, and their joint mean;
- median paired improvement for both primary metrics;
- median absolute deviation of paired improvements;
- capped and raw P90/P95 regret;
- failure, clipping, and coverage counts.

No task ID, truth array, forecast array, Dev value, or Public value enters the
frozen release or proposer feedback.

## 4. Current-task local evidence

Use five history-only rolling origins where history length permits it and
deterministically reduce to four or three origins for longer forecast horizons.
Tasks that cannot supply three genuine origins remain lower-confidence; no
origin is padded or fabricated.

For every candidate/weight recipe, compute paired improvement over Toto on
the exact same origins. Local evidence must satisfy all of the following:

- at least three successful paired origins;
- nonnegative median improvement in both sMAE and sRMSE;
- positive median joint improvement;
- no raw catastrophic fold;
- worst-fold joint regret within the frozen budget;
- no inconsistent fold truths or forecast horizons.

The gate combines group and local evidence conservatively. It uses a
Beta-Binomial posterior with a neutral Beta(1,1) prior for win probability and
a robust effect margin based on median minus scaled MAD. Activation requires:

- posterior probability that win rate exceeds 0.5 of at least 0.80;
- positive robust joint-effect margin at the group-prior layer, with neither
  individual group margin worse than -0.05;
- positive robust effect margins for both sMAE and sRMSE in the current task;
- local and group evidence pointing in the same direction; and
- the existing deterministic Pareto and tail gates.

The thresholds and formula are host-owned and fingerprinted. An LLM cannot
emit probabilities, metric values, weights, or acceptance decisions.

## 5. Horizon-aware routing

When the forecast horizon and five-origin diagnostics provide at least three
valid paired observations for both regions, evaluate two fixed regions:

- early: first half of the horizon;
- late: remaining horizon.

Each region independently chooses from the same closed anchor-heavy weight
grid. Toto weight remains at least 0.5 in each region. A region that lacks
support or fails confidence checks uses Toto exactly. If neither region is
independently supported, use the full-horizon gate; if that also fails, return
the original Toto forecast byte-for-byte.

No arbitrary change point or LLM-authored boundary is allowed in this version.

## 6. Candidate and weight selection

The release still supplies at most eight candidates including Toto and at
most two active specialists. Candidate ordering is:

1. confidence-qualified at the most specific evidence level;
2. lower raw tail regret;
3. higher robust paired effect in both metrics;
4. greater independent group support;
5. family/forecast diversity; and
6. canonical candidate name.

Weights remain the fixed nonnegative 0.1 grid, sum exactly to one, and keep
Toto at or above 0.5. A more aggressive mixture may not win merely because it
has a better mean: confidence, Pareto, worst-fold, and raw-tail gates precede
ranking.

## 7. Evolution and data boundaries

The lifecycle is:

1. fit hierarchical evidence in five-fold group-aware Train cross-fitting;
2. generate one OOF report over all 80 Train tasks;
3. if rejected, expose only sanitized Train aggregates for the next version;
4. if accepted, reconstruct the exact policy from all 80 Train tasks;
5. open 20 Dev once and accept or reject without refitting; and
6. after Dev acceptance, run Public-99 exactly once as report-only evidence.

Public-99 is mandatory for a frozen accepted version but can never authorize
another mutation of that version. A Dev rejection also cannot be tuned away
using Dev task results.

## 8. Acceptance and expected outcome

Keep the existing strict dual-metric acceptance gates. In particular, do not
raise the 0.25 maximum task-regret limit or lower the 60% activated precision
requirement merely to make the first result pass.

The immediate target is to retain most of the first run's mean gains while
removing enough low-confidence activations to achieve:

- at least 60% activated precision on Train OOF;
- maximum task regret no greater than 0.25 for both metrics;
- non-regressing capped/raw P90 and P95 tails;
- improvement or equality in both mean capped metrics with one strict gain;
- activation across at least two independent groups; and
- exact Toto equality for every non-activated task.

This routing change may improve reliability by a few percentage points. A
large improvement over Toto will additionally require genuinely stronger
specialist/Combined children; the confidence gate must not overstate the
ceiling of the current Dictionary.

## 9. Implementation and verification

Extend the focused task-local modules rather than the large Champion
controller:

- add immutable hierarchical evidence and confidence-policy schemas;
- add cross-fitted evidence construction and strict canonical parsing;
- add five-origin and early/late scoring to the local tournament;
- bind all evidence, thresholds, source hashes, and metric policies into the
  release fingerprint;
- preserve the existing v1 release/runtime path for compatibility; and
- extend the formal evolution and Public CLIs without changing their data
  authority order.

Tests must cover fold leakage, sparse-group fallback, posterior/robust-margin
boundaries, disagreement between group and local evidence, early/late exact
fallback, dual-metric and raw-tail safety, deterministic fingerprints,
label-free runtime behavior, Dev unopened after OOF rejection, and Public
report-only isolation.
