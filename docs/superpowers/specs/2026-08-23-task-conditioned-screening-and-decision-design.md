# Task-Conditioned Dictionary Screening and Decision Design

**Status:** Proposed for review  
**Date:** 2026-08-23  
**Scope:** Numerical Agent only; history-only forecasting over the existing 103 candidates

## 1. Outcome

Complete the current dictionary-filtering work as two separately measurable and separately
frozen stages:

```text
103-candidate Master Dictionary
  (93 statistical + 5 TSFM + 5 Combined)
                 |
                 v
Phase A: Task-conditioned screening
  Historical series -> Task Profile -> Active Dictionary
                 |
                 | freeze the screening policy
                 v
Phase B: Task-conditioned decision
  Active candidates + history-only hindcasts -> one forecast or a guarded ensemble
                 |
                 v
Trusted scoring with future labels
```

Phase A answers **which methods may compete on this task**. Phase B answers **which active
method, or which small active ensemble, should produce the final forecast**. The same component
must never answer both questions during evaluation.

This separation is required because the completed 80/20/99 filtering experiment improved the
selectable-candidate success rate from 45.98% to 91.51%, but the existing global ranker selected
`toto_2_0` for all 99 Test tasks. That run demonstrated dictionary cleaning, not a useful
task-conditioned decision rule.

## 2. Goals

1. Produce an explicit, inspectable `TaskProfile` from historical numbers only.
2. Learn a reusable conditional screening policy over all 103 candidates.
3. Materialize a different `ActiveDictionary` for every task, with inclusion and exclusion
   reasons.
4. Evaluate screening independently of final forecast selection.
5. Freeze the accepted screening policy before evolving the Decision Agent.
6. Select or safely combine active forecasts using history-only validation evidence.
7. Evaluate the two contributions separately on the fixed 80 Train / 20 Dev / 99 Public Test
   split.

## 3. Non-goals

- No documents, retrieved evidence, GT evidence, or future values are inputs to the Numerical
  Agent.
- No LLM weight training.
- No joint evolution with Retrieval or contextual Decision Agents in this phase.
- No TSFM identity, checkpoint, license, or reviewed runtime substitution.
- No method is discarded merely because it is globally weak; specialists remain available when
  their applicability condition holds.
- Public Test is never used for mutation, threshold selection, acceptance, or retry decisions.

## 4. Terminology

- **Master Dictionary:** all 103 candidates and their reusable status/applicability policy.
- **Task Profile:** deterministic history-only measurements for one series.
- **Screening Policy:** the evolved mapping from Task Profile to candidate eligibility.
- **Active Dictionary:** the candidates eligible for one task after screening.
- **Decision Policy:** the evolved mapping from active-candidate diagnostics to a final method or
  ensemble.
- **Global oracle:** the lowest-MASE successful candidate among all 103 candidates, used only by
  the trusted Train/Dev evaluator.
- **Active oracle:** the lowest-MASE successful candidate retained in the Active Dictionary, also
  used only by the trusted evaluator.

## 5. Shared information boundary

At inference time both stages may read:

- historical timestamps and values;
- target frequency and forecast horizon;
- reviewed candidate metadata;
- deterministic history-only analysis;
- candidate forecasts and hindcasts produced without future labels.

They must not read:

- `future`, `future_values`, labels, or trusted metrics for the current forecast origin;
- Dr-CiK `gt_evidence`, document roles, document subtypes, or supporting-answer annotations;
- Train/Dev/Test names or task identifiers that encode the split;
- another task's raw future trajectory.

The trusted evaluator may use future labels after a forecast or Active Dictionary is frozen. It
returns only aggregate or typed feedback allowed by the relevant evolution stage.

## 6. Phase A: complete task-conditioned screening

### 6.1 Task Profile

Replace the current small tuple of tags with one deterministic profile. It reuses the existing
analysis-skill functions and adds the target horizon and basic data-shape fields.

```python
@dataclass(frozen=True)
class TaskProfile:
    task_id: str                 # excluded from Agent prompts
    frequency: str
    history_length: int
    horizon: int
    zero_fraction: float
    signed: bool
    integer_valued: bool
    trend_direction: str
    trend_strength: float
    periodicity_periods: tuple[int, ...]
    periodicity_strength: float
    periodicity_confidence: float
    outlier_fraction: float
    noise_relative_scale: float
    likely_stationary: bool
    stationarity_score: float
    recent_regime_start: int | None
    recent_regime_confidence: float
    intermittency_adi: float
    intermittency_cv2: float
```

All fields must be finite or explicitly optional. Profile creation must be deterministic,
side-effect free, and tested on empty-invalid, short, constant, intermittent, trending,
seasonal, outlier-heavy, signed, and regime-shift histories.

### 6.2 Applicability rule

The current `applicability: tuple[str, ...]` is a single conjunction and cannot express a method
that is useful for either of two regimes. Replace it with a typed disjunction of conjunctions:

```python
@dataclass(frozen=True)
class ApplicabilityClause:
    all_tags: tuple[str, ...] = ()
    feature_tests: tuple[FeatureTest, ...] = ()

@dataclass(frozen=True)
class ApplicabilityPolicy:
    any_of: tuple[ApplicabilityClause, ...]  # OR across clauses; AND inside a clause
```

A `FeatureTest` contains only a reviewed profile field, one operator from
`<, <=, ==, >=, >, in`, and a finite literal. Arbitrary Python or LLM-generated expressions are
forbidden. An empty `any_of` means broadly applicable. Rules are parsed from an AST-safe Python
literal, as the current dictionary already is.

Candidate states remain:

- `keep`: selectable, usually broad;
- `specialized`: selectable only when its policy matches;
- `repair`: valuable identity retained but implementation is not selectable;
- `quarantine`: unsafe or repeatedly invalid implementation;
- `discard`: allowed only with the existing strict dominance evidence.

### 6.3 Active Dictionary contract

For every task the screen materializes:

```json
{
  "task_profile_hash": "...",
  "screening_policy_hash": "...",
  "active": [
    {
      "name": "croston_sba",
      "family": "statistical",
      "matched_clause": 0,
      "screen_confidence": 0.88,
      "reason_codes": ["intermittent", "nonnegative"]
    }
  ],
  "excluded": [
    {
      "name": "holt_winters_multiplicative",
      "reason_code": "requires_strictly_positive_history"
    }
  ],
  "fallback_applied": false
}
```

`screen_confidence` is computed from Train-calibrated, versioned statistics; it is not an LLM
confidence sampled separately for every task.

The Active Dictionary must contain at least three candidates. When available, it must contain at
least one statistical candidate and one TSFM candidate. A reviewed stable fallback set is added
only if a learned policy would violate this invariant. The fallback is recorded explicitly and is
part of the screening score.

### 6.4 Filter Agent input and output

The single self-evolving Filter Agent receives Train-only summaries:

- current Master Dictionary entries;
- profile-schema documentation;
- per-method success, NotApplicable, Crash, Invalid, MASE rank, and top-quartile rate;
- the same measurements conditioned on profile buckets;
- false-exclusion and false-inclusion diagnoses from the trusted evaluator;
- only the current bounded target batch.

It returns typed changes to status or `ApplicabilityPolicy`. It cannot edit method source,
TSFM bindings, Combined parent identities, the scorer, or the split.

### 6.5 Screening metrics

Phase A must not use the forecast chosen by the existing global selector. It is scored only as a
screen:

- **Active coverage:** fraction of tasks with a nonempty valid Active Dictionary.
- **Active success rate:** successful executions divided by active execution attempts.
- **Failure exposure:** Crash + Invalid among active attempts.
- **NotApplicable exposure:** NotApplicable among active attempts.
- **Compression:** median and mean active candidates per task divided by 103.
- **Global-oracle retention:** fraction of tasks whose global oracle remains active.
- **Active-oracle regret:** `(active_oracle_MASE - global_oracle_MASE) /
  (1 + global_oracle_MASE)`, with a fixed penalty if no active candidate succeeds.
- **Family diversity:** active family counts and fallback frequency.

Future labels are used only inside the trusted evaluator to compute oracle metrics. The Filter
Agent sees aggregate error categories and conditioned counts, never raw future values or per-task
future trajectories.

### 6.6 Screening evolution and acceptance

Use the existing frozen priority batching and Git-tracked Parent/Child lifecycle:

```text
Parent Screening Policy
  -> Filter Agent proposes one bounded Child
  -> schema and no-leak validation
  -> Train screening evaluation
  -> read-only Dev screening evaluation
  -> accept/reject
  -> accepted Child becomes next Parent
```

For the first complete run, retain four batches of at most 24 targets so the 77 previously
identified candidates are covered without prompt overflow. A later generation may revisit a
method only when new accepted rules change its measured false-inclusion or false-exclusion
behavior.

A Child is accepted only when all of the following hold:

1. Train and Dev Active coverage remain 100%.
2. Dev global-oracle retention is at least 95% and does not fall by more than one task relative
   to the Parent.
3. Dev mean active-oracle regret is no worse than the Parent by more than 0.01.
4. Dev failure exposure does not increase.
5. At least one of active success rate, failure exposure, NotApplicable exposure, or compression
   improves strictly on Train.
6. The same improved dimension does not regress on Dev beyond 0.5 percentage points.
7. No family invariant or fallback invariant is violated.

The thresholds are frozen before the formal 80/20/99 run. Public Test is evaluated exactly once
after the screening policy is frozen.

### 6.7 Phase-A artifacts

```text
screening/
  task_profile.py
  policy.py
  evaluator.py
  evolution.py
  artifacts/
    parent_screening_policy.py
    generation_N_child_screening_policy.py
    frozen_screening_policy.py
    train_active_dictionaries.jsonl
    dev_active_dictionaries.jsonl
    screening_trace.json
    screening_report.md
```

The policy is executable Python data, not a JSON-only method library. JSONL is used only for
immutable per-task results and traces.

## 7. Freeze boundary

Phase B may start only after Phase A has produced:

- `frozen_screening_policy.py` and its SHA-256;
- a passing Train/Dev screening report;
- materialized Train and Dev Active Dictionaries;
- a controller test proving unseen tasks can be screened without labels.

The Decision experiment records the screening-policy hash. Any Phase-A change invalidates all
Phase-B caches and results.

## 8. Phase B: task-conditioned decision

### 8.1 Candidate diagnostics

Only active candidates are executed. For each successful active candidate, compute history-only
rolling-origin diagnostics:

- median and worst hindcast MASE, MAE, and sMAPE;
- completed folds and fold success rate;
- recent-fold versus older-fold error;
- forecast level, slope, amplitude, and bias relative to held-out historical folds;
- stability across cutoffs;
- pairwise forecast diversity for possible ensembles;
- candidate family and the Phase-A screening reason codes.

The default validation is three expanding-window folds. Each fold horizon is
`min(final_horizon, max(1, available_history // 4))`. A candidate needs at least two successful
folds. Candidate execution and hindcasting share the existing hard timeout and runtime-isolation
boundaries.

### 8.2 Decision Agent contract

Input:

```json
{
  "task_profile": {"history-only fields": "..."},
  "active_dictionary": ["candidate identities and screening reasons"],
  "candidate_diagnostics": ["history-only hindcast and forecast-shape summaries"],
  "current_decision_policy": {"versioned rubric and weights": "..."}
}
```

Output:

```json
{
  "mode": "single|ensemble",
  "selected": ["candidate_name"],
  "weights": [1.0],
  "confidence": 0.0,
  "reason_codes": ["best_recent_hindcast", "stable_across_cutoffs"]
}
```

The controller rejects unknown, inactive, failed, or insufficiently validated candidates.
Ensembles contain at most three candidates, use nonnegative weights summing to one, and are
allowed only when their historical-fold blend beats the best member and the members are not
near-duplicates. Otherwise the best single candidate is used.

### 8.3 Decision self-evolution

The Parent is a versioned `decision_policy.py` containing:

- ranking-feature weights;
- recent-fold weighting;
- minimum fold coverage;
- ensemble diversity and improvement thresholds;
- conservative fallback rules;
- the Decision Agent prompt/rubric, when LLM arbitration is enabled.

One Meta-Harness Agent receives Train-only selection failures and aggregate diagnostics and may
modify only these fields. The Python controller validates the Child, replays Train, and then uses
read-only Dev for acceptance. The frozen Phase-A policy and Active Dictionaries cannot change.

### 8.4 Decision metrics and acceptance

Report:

- final mean and median MASE, MAE, and sMAPE;
- forecast coverage;
- selection regret relative to the active oracle;
- catastrophic-tail rate (`MASE > 10`);
- method and family selection diversity;
- ensemble frequency;
- fallback frequency.

A Decision Child is accepted only if:

1. Dev coverage does not decrease.
2. Dev mean MASE improves, or median MASE improves without worsening mean MASE by more than 1%.
3. Dev catastrophic-tail rate does not increase.
4. Dev mean active-oracle regret does not increase.
5. No Test result has been read.

MASE is the primary acceptance metric. MAE and sMAPE are reported as secondary metrics because
Dr-CiK tasks have different scales and may contain near-zero values.

## 9. Required comparisons

Use the same candidate outcomes and 80/20/99 split for all rows:

| Experiment | Screening | Decision |
|---|---|---|
| A. Current baseline | all current selectable candidates | existing global cross-task ranker |
| B. Screening only | frozen task-conditioned Active Dictionary | existing ranker |
| C. Decision only | all current selectable candidates | new task-conditioned Decision |
| D. Full system | frozen task-conditioned Active Dictionary | new task-conditioned Decision |
| E. Toto reference | singleton `toto_2_0` | fixed |

This factorial comparison reveals whether gains come from screening, decision, or their
interaction. The Public Test table is generated once after both policies and all baselines are
frozen.

## 10. Implementation sequence

1. Add typed `TaskProfile`, applicability clauses, and Active Dictionary materialization.
2. Add Phase-A-only metrics and remove final selection MASE from the filter acceptance gate.
3. Migrate the current 103-entry frozen dictionary to the new policy schema.
4. Run unit tests and an 8 Train / 2 Dev smoke test drawn only from the formal Train partition.
5. Run and freeze the formal 80 Train / 20 Dev screening evolution.
6. Add candidate hindcast diagnostics and the Decision policy/controller.
7. Run unit tests and an 8 Train / 2 Dev Decision smoke test, again drawn only from the formal
   Train partition, against the frozen screen.
8. Evolve and freeze Decision on 80 Train / 20 Dev.
9. Evaluate A-E once on the 99-task Public Test and write one final report.

## 11. Verification requirements

Unit and adversarial tests must cover:

- profile determinism and label independence;
- OR-of-AND applicability semantics and rejected arbitrary expressions;
- explicit inclusion/exclusion reasons;
- fallback and family invariants;
- screening metrics independent of the old final selector;
- an aggressive filter rejected for losing the oracle;
- a useful specialist retained only on its matching regime;
- Test access prohibited before freeze;
- Decision cannot select an inactive or failed candidate;
- hindcast folds use historical prefixes only;
- ensemble weights and improvement gate;
- cache invalidation when either frozen policy hash changes;
- no mutation of method source, TSFM identities, or trusted scorers.

## 12. Completion criteria

The work is complete when:

1. Every one of the 199 public tasks can materialize a valid Task Profile and Active Dictionary.
2. Phase A reduces active failure/NotApplicable exposure and candidate count while preserving at
   least 95% Dev oracle retention and 100% coverage.
3. Phase B no longer collapses by construction to one globally best method; any observed
   singleton collapse must be justified by per-task hindcasts and reported as a result, not hidden.
4. The full-system Dev MASE is no worse than both the current global-selector baseline and the
   frozen-screen/old-selector baseline.
5. The 99-task Public Test is scored only after both policies are frozen, with all five comparison
   rows and per-task artifacts retained.
