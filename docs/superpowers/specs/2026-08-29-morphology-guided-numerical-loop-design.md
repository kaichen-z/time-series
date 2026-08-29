# Morphology-Guided Numerical Evolution Design

**Date:** 2026-08-29

**Status:** Approved in chat; written design pending final user review

**Scope:** Numerical multi-parent Combined evolution, history-only Morphology reasoning, and the
sanitized hand-off to two-stage Retrieval

## Objective

Build one no-weight-training Numerical loop in which historical morphology actively controls
candidate eligibility, Combined construction, history-only routing, and assumption validation.
The loop must emit both an executable point forecast and falsifiable, tool-grounded assumptions
that a later Retrieval round can verify. An unverified natural-language assumption must never be
allowed to change a numerical trajectory by itself.

## Existing State

The repository already contains the necessary partial components:

- `screening.profile_task()` deterministically converts historical values into a `TaskProfile`
  using reviewed trend, periodicity, intermittency, outlier, stationarity, noise, and recent-regime
  skills.
- `CombinedPolicy` executes two-to-five materialized Statistical/TSFM leaf parents with
  `weighted_mean`, `median`, `trimmed_mean`, or a two-parent history-only `route`.
- `propose_combined_child()` can request strict `add`, `repair`, `fork`, and `remove` operations,
  but its diagnostics do not yet contain morphology-group evidence and it is not wired into the
  formal 80/20 runner.
- The current uncommitted `MorphologyReasoner` uses only historical values, reviewed tools, active
  candidate names, and fixed budgets to produce a grounded `MorphologyCard`.
- `select_assumption_guided_forecast()` can restrict the safe Numerical Selector to candidates
  supported by ranked assumptions while retaining baseline-protection gates.
- two-stage Retrieval already accepts only the four sanitized assumption fields
  `assumption_id`, `kind`, `claim`, and `failure_condition` in Round 2.

The missing link is a formal controller that makes morphology useful inside Numerical evolution,
then freezes a complete Numerical/Morphology release before Retrieval sees it.

## Architecture

Morphology is split into two authorities with different responsibilities.

### A. Deterministic Numerical Morphology

`TaskProfile` remains the runtime authority for numerical routing. It is computed by reviewed
Python functions and contains no LLM-authored values. It controls:

- task-conditioned Statistical/TSFM/Combined eligibility;
- the candidate namespace visible to the Morphology Reasoner;
- Combined route signals and thresholds;
- history-only grouping used to summarize Train evidence for Combined evolution;
- assumption-consistency checks.

The supported initial group predicates are fixed and typed:

- periodic versus non-periodic;
- trending versus flat/stationary;
- intermittent versus dense;
- recent-regime versus stable-history;
- noisy/outlier-heavy versus regular;
- short versus long history;
- short versus long forecast horizon.

No LLM may create a new executable feature or arbitrary Python predicate during a formal run.

### B. Grounded Forecast Assumptions

After active candidates have been executed and hindcasted, the bounded `MorphologyReasoner`
chooses reviewed analysis tools and historical windows. It emits a `MorphologyCard` containing:

- short- and long-term descriptions;
- exact tool calls and Python-produced observations;
- one-to-seven falsifiable assumptions;
- the executed call IDs grounding each assumption;
- active candidate names affected by each assumption;
- a failure condition and prior confidence.

This card helps Numerical selection by narrowing and ranking already-executed candidates. It does
not write code, call TSFMs, read documents, or create a new forecast array.

## End-to-End Numerical Flow

```text
historical values + frequency + horizon
  -> reviewed Python morphology skills
  -> deterministic TaskProfile
  -> task-conditioned screening
  -> materialize each active Statistical/TSFM leaf exactly once
  -> materialize current Combined policies from those leaf outcomes
  -> history-only hindcasting and CandidateDiagnostics
  -> bounded MorphologyReasoner tool loop
  -> grounded MorphologyCard
  -> deterministic assumption-consistency gate
  -> assumption-guided safe Numerical Selector
  -> NumericalForecastPackage
```

The package contains:

```text
NumericalForecastPackage = {
  task_profile,
  active_candidate_names,
  candidate_diagnostics,
  morphology_card,
  accepted_assumptions,
  rejected_assumptions,
  selection_decision,
  final_forecast,
  component_fingerprints
}
```

Only the final forecast, selected method/weights, safe diagnostics, and sanitized assumptions are
public inference outputs. Internal tool results and rejected assumptions remain audit artifacts.

## Morphology-Guided Combined Evolution

The formal Train controller derives sanitized `MorphologyGroupEvidence` from Train tasks. Each
record contains only:

- a fixed group identifier and typed predicate;
- task and entity support counts;
- candidate coverage and failure rates;
- winsorized aggregate sMAE and sRMSE deltas relative to the protected baseline;
- parent forecast disagreement;
- the names of reviewed eligible Statistical/TSFM leaves.

It never contains raw future values, per-timestamp labels, documents, Retrieval artifacts, Dev
metrics, Public metrics, or hidden data. The Combined proposal prompt receives this evidence and
may propose a route or mixture justified by a supported morphology group. Python remains the
authority for names, manifests, weights, arithmetic, fallbacks, signal thresholds, and final
execution.

Evolution follows:

```text
Parent portfolio
  -> Train-only morphology-group diagnostics
  -> Luna target/operation diagnosis
  -> Terra strict Combined proposal
  -> atomic Python validation
  -> small Train screen
  -> complete Train evaluation
  -> read-only Dev evaluation
  -> accept exact Child or return exact Parent
```

The initial `93 Statistical + 5 TSFM + 5 Combined` portfolio is only a seed. Accepted `add` or
`fork` operations may increase the Combined count up to the existing hard ceiling. Statistical
and TSFM leaf identities remain immutable for this coordinate.

## Assumption-Consistency Gate

An assumption may influence Numerical selection only when all of the following hold:

1. every supporting call ID refers to an actually executed reviewed tool;
2. the broad/recent window coverage contract is satisfied;
3. every referenced candidate is active and has a valid executed forecast;
4. the assumption kind is compatible with deterministic `TaskProfile` evidence;
5. its candidate has enough successful hindcast folds;
6. it does not bypass the protected baseline, worst-fold, catastrophe, or coverage gates;
7. the final ranked set satisfies kind and leading-candidate diversity limits.

Failure removes the assumption from routing and records a typed rejection reason. It does not
remove the underlying numerical candidate and does not change the protected fallback.

Train labels may be used only by the trusted post-forecast credit evaluator to improve the next
generation's policy. A task's future label cannot alter that task's already-produced forecast or
Morphology Card. Dev, Public Regression, hidden inference, and frozen inference never update
Morphology policy or Skills.

## Retrieval Hand-Off

Round 1 remains assumption-blind. After provisional Decision identifies named gaps, the host
projects only accepted assumptions into the existing Round-2 schema:

```json
{
  "assumption_id": "six_step_cycle",
  "kind": "seasonality",
  "claim": "The supported six-step cycle will persist.",
  "failure_condition": "The cycle was temporary or its phase changed."
}
```

The hand-off excludes candidate names, forecasts, weights, hindcast metrics, tool observations,
source code, future labels, and evaluator-only metadata. Retrieval evidence may later help the
Decision Agent choose among executed candidates; it cannot mutate the frozen Numerical release.

## Formal Data Protocol

- **8 Train / 2 Dev smoke:** validates execution, schemas, cache keys, and read-only Dev behavior.
- **80 Train:** proposes and screens Combined/Morphology children. Successive halving may prune
  clearly inferior children before complete evaluation.
- **20 Dev:** exactly-once, read-only acceptance. A Child must have non-worse winsorized mean sMAE
  and sRMSE, at least one strict improvement, non-worse catastrophic rate and coverage, and no
  unacceptable entity concentration.
- **99 Public Regression:** accessed only after the Numerical, Morphology, Retrieval, and Decision
  releases are frozen. It is a regression set, not a sealed test.
- **Official hidden 80:** inference/submission only. No labels, local scoring, learning, or release
  mutation.

## Models and Budgets

- diagnosis and target selection: `gpt-5.6-luna`, low reasoning;
- Combined and Morphology mutation: `gpt-5.6-terra`, medium reasoning;
- a single `gpt-5.6-sol`, high retry is allowed only after a typed transient/backend or repeated
  schema failure and must be recorded;
- Morphology inference uses at most four turns, eight total reviewed tool calls, and three calls
  per turn;
- all prompts, budgets, model identities, policy sources, and split hashes are part of the release
  fingerprint.

## Compatibility

The feature is opt-in for new experiments. Existing frozen artifacts continue to use their stored
policy and deterministic legacy assumption path. Loading an old release must not silently enable
LLM Morphology. A formal Morphology-guided release must carry explicit configuration and component
fingerprints.

## Files and Ownership

Expected production boundaries:

- `numerical_agent/evolution/morphology.py`: bounded history-only tool loop and card schema;
- `numerical_agent/evolution/morphology_credit.py`: Train-only post-forecast credit;
- `numerical_agent/evolution/screening.py`: deterministic `TaskProfile` authority;
- `numerical_agent/evolution/combined_evolution.py`: sanitized morphology-group proposal input;
- `numerical_agent/evolution/numerical_selector.py`: assumption-consistency-aware safe selection;
- `numerical_agent/run_selector_evolution.py`: formal 8/2 and 80/20 controller wiring;
- `numerical_agent/evaluate_frozen_two_stage.py`: frozen release loading and write-free evaluation;
- `evolving_loop/harness.py` and Retrieval schemas: four-field Round-2 projection only;
- focused tests beside the existing morphology, portfolio, selector, frozen-evaluation, Retrieval,
  and CLI suites.

## Verification and Deliverables

Implementation is complete only when:

1. every new behavior was developed with a failing test first;
2. the 8/2 smoke proves end-to-end Numerical/Morphology execution;
3. the 80/20 trace proves Dev is read-only and produces an accepted release or an explicit Parent
   retention;
4. release bytes and fingerprints reload deterministically;
5. Round 1 is demonstrably assumption-blind and Round 2 sees only the four safe fields;
6. existing legacy/frozen results remain byte-compatible;
7. the focused and full repository test suites pass in an isolated workspace;
8. compact manifests, reports, and commands are documented under `runs/`; caches and transcripts
   remain ignored.

## Non-Goals

- no LLM, TSFM, or Statistical-model weight training;
- no checkpoint/model parameter merging;
- no document retrieval inside the Numerical loop;
- no natural-language generation of forecast arrays;
- no Public/hidden optimization;
- no claim of improvement until complete Dev acceptance and frozen regression evaluation exist.
