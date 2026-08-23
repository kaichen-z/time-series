# Task-Conditioned Screening and Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve and freeze a task-conditioned screen over 103 numerical candidates, evolve a history-only numerical selector over the screened candidates, and evaluate the frozen two-stage system once on the 99-task Public Test split.

**Architecture:** Phase A converts each historical series into a typed profile and applies an evolved, AST-safe screening policy to produce an explicit Active Dictionary. Phase B runs history-only rolling-origin validation for active candidates and applies an evolved typed decision policy to select one forecast or a guarded ensemble. Train proposes changes, Dev accepts them, and Public Test remains inaccessible until both policy hashes are frozen.

**Tech Stack:** Python 3.12+, dataclasses, `ast.literal_eval`, existing `OutcomeCache` and `PolicyOutcomeCache`, Codex CLI LLM client, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-task-conditioned-screening-and-decision-design.md`

## Global Constraints

- Candidate inventory remains exactly 93 statistical + 5 TSFM + 5 Combined identities.
- Screening and selection inputs are history-only; future labels remain inside trusted evaluators.
- TSFM checkpoints, revisions, licenses, method IDs, and Combined parent identities are immutable.
- Phase A is frozen before Phase B evolution begins.
- Public Test is read and scored exactly once after both policies are frozen.
- MASE is primary; MAE and sMAPE are secondary; probabilistic sCRPS is outside this point-forecast implementation.
- New production behavior follows test-first RED/GREEN cycles.

---

### Task 1: Typed Task Profile

**Files:**
- Create: `numerical_agent/evolution/screening.py`
- Create: `tests/test_evolution_screening.py`

**Interfaces:**
- Consumes: `Task`, `analyze_series(history, frequency)`.
- Produces: `TaskProfile`, `profile_task(task: Task) -> TaskProfile`, `TaskProfile.to_public_payload()`.

- [ ] **Step 1: Write failing profile tests**

Add tests proving deterministic profiles for constant, intermittent, periodic, trending, signed,
outlier-heavy, and recent-regime histories. Assert that `to_public_payload()` contains no
`task_id`, `future`, or split field.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_screening.py -k profile`

Expected: import failure because `screening.py` does not exist.

- [ ] **Step 3: Implement the profile**

Create a frozen dataclass with exact fields from the design. Build it from existing history-only
analysis skills, derive outlier fraction from returned indices, and normalize ADI from
`average_nonzero_gap`. Reject empty or non-finite histories through the existing `Task`/skill
validation path.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_screening.py -k profile`

Expected: all selected tests pass.

### Task 2: Typed Applicability and Active Dictionary

**Files:**
- Modify: `numerical_agent/evolution/screening.py`
- Modify: `tests/test_evolution_screening.py`

**Interfaces:**
- Produces: `FeatureTest`, `ApplicabilityClause`, `ApplicabilityPolicy`, `ScreeningEntry`,
  `ScreeningPolicy`, `ActiveCandidate`, `ExcludedCandidate`, `ActiveDictionary`, and
  `materialize_active_dictionary(policy, profile)`.

- [ ] **Step 1: Write failing policy tests**

Test OR across clauses, AND inside a clause, reviewed operators only, finite literals, unknown
profile fields, contradictory categorical tags, broad empty policy, deterministic reasons,
minimum-three fallback, and statistical/TSFM family preservation.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_screening.py -k 'policy or active or fallback'`

Expected: missing interface failures.

- [ ] **Step 3: Implement typed evaluation**

Implement field lookup without `eval`, `exec`, or arbitrary attributes. `in` accepts only a tuple
of finite/string literals. Materialization evaluates only `keep` and `specialized` entries,
records the matched clause, records one exclusion reason for every other entry, and adds reviewed
fallback identities only when invariants fail.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_screening.py -k 'policy or active or fallback'`

Expected: all selected tests pass.

### Task 3: Independent Screening Evaluator

**Files:**
- Modify: `numerical_agent/evolution/screening.py`
- Modify: `tests/test_evolution_screening.py`

**Interfaces:**
- Produces: `ScreeningScore`, `evaluate_screening(policy, tasks, outcomes) -> ScreeningScore`,
  `compare_screening(parent, child) -> ScreeningGateResult`.

- [ ] **Step 1: Write failing evaluator tests**

Construct outcomes in which the old final selector would choose the same method before and after
screening. Assert that screening success rate, failure exposure, NotApplicable exposure,
compression, global-oracle retention, normalized active-oracle regret, family diversity, and
fallback rate still change independently. Add an aggressive-filter case that improves
compression but loses the oracle and must be rejected on Dev.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_screening.py -k 'score or oracle or gate'`

Expected: missing evaluator failures.

- [ ] **Step 3: Implement the trusted metrics**

Use labels only through already-scored `Outcome.mase`. Compute a global oracle from successful
candidates, an active oracle from successful active candidates, and the normalized regret
specified in the design. Implement the frozen acceptance thresholds: 100% coverage, at least 95%
Dev oracle retention, no more than one-task retention loss, at most 0.01 mean-regret increase, no
Dev failure-exposure increase, and a strict Train reliability/compression improvement that does
not regress by more than 0.5 percentage points on Dev.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_screening.py -k 'score or oracle or gate'`

Expected: all selected tests pass.

### Task 4: Screening Policy Evolution and Migration

**Files:**
- Create: `numerical_agent/evolution/screening_evolution.py`
- Modify: `numerical_agent/evolution/filtering.py`
- Modify: `numerical_agent/run_filter_evolution.py`
- Create: `tests/test_evolution_screening_evolution.py`
- Modify: `tests/test_run_filter_evolution.py`

**Interfaces:**
- Consumes: legacy `FilterDictionary`, cached Train/Dev outcomes, `LLMClient`.
- Produces: `migrate_filter_dictionary(...)`, `evolve_screening_once(...)`, an AST-safe
  `screening_policy.py`, per-generation result JSON, and Train/Dev Active Dictionary JSONL.

- [ ] **Step 1: Write failing migration and Agent-boundary tests**

Test exact migration of current simple AND tags into one typed clause; typed JSON Agent responses;
24-target limit; unknown names/fields/operators; immutable identity/family; no source changes;
Train-only prompt content; Dev-label exclusion; and rejection leaving the Parent byte-identical.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_screening_evolution.py tests/test_run_filter_evolution.py`

Expected: missing migration/evolution interfaces.

- [ ] **Step 3: Implement bounded evolution**

Reuse frozen target batches. Give the Filter Agent current entries, conditioned Train summaries,
false inclusions, and false exclusions. Permit only status and typed applicability changes. Use
`evaluate_screening` and `compare_screening`; remove old selected-forecast MASE from the Phase-A
acceptance decision while retaining it only in legacy reports.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_screening_evolution.py tests/test_run_filter_evolution.py tests/test_evolution_filtering.py`

Expected: all selected tests pass.

### Task 5: Phase-A Cached 8/2 and 80/20 Runs

**Files:**
- Create: `scripts/run_task_conditioned_screening.sh`
- Create: `tests/test_task_conditioned_screening_script.py`
- Generate: `runs/task_conditioned_screening/<run-id>/...`

**Interfaces:**
- Produces: frozen Phase-A policy hash, screen trace, Active Dictionaries, and screening report.

- [ ] **Step 1: Write a failing dry-run script test**

Assert propagation of repository, split, task catalog, statistical cache, TSFM cache, model,
8/2 or 80/20 sizes, generation count, and output directory. Assert no Public-Test argument is
accepted by the evolution command.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_task_conditioned_screening_script.py`

Expected: missing script failure.

- [ ] **Step 3: Implement the runner and report**

The script first verifies exact cache hits for all requested Train/Dev method/task pairs, then
runs four frozen target batches. It writes hashes before and after every Agent call and freezes
only an accepted policy.

- [ ] **Step 4: Verify GREEN and run 8/2**

Run: `pytest -q tests/test_task_conditioned_screening_script.py`

Then run the script with `SCREEN_TRAIN=8 SCREEN_DEV=2`; inspect the report and no-leak manifest.

- [ ] **Step 5: Run and freeze 80/20**

Run with `SCREEN_TRAIN=80 SCREEN_DEV=20`. Do not run Public Test. Record the frozen policy path,
SHA-256, oracle retention, active success/failure rates, compression, and fallback rate.

### Task 6: History-Only Candidate Hindcasts

**Files:**
- Create: `numerical_agent/evolution/numerical_selector.py`
- Create: `tests/test_evolution_numerical_selector.py`

**Interfaces:**
- Consumes: frozen `ActiveDictionary`, `Task`, candidate runtime callbacks.
- Produces: `HindcastFold`, `CandidateDiagnostics`,
  `diagnose_active_candidates(task, active, runner, config)`.

- [ ] **Step 1: Write failing fold and diagnostic tests**

Assert three expanding-window folds use only earlier historical prefixes; final future is never
passed to the runner; at least two folds must succeed; median/recent/worst MASE, RMSSE, normalized
bias, MASE MAD, slope error, phase error, amplitude ratio, explosion flag, and pairwise diversity
are finite or typed optional values. Add crash, timeout, constant-history scaling, and short-history
cases.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_numerical_selector.py -k 'fold or diagnostic'`

Expected: missing selector module failure.

- [ ] **Step 3: Implement diagnostics and cache identity**

Use three folds with `min(final_horizon, max(1, available_history // 4))`; require two successful
folds. Use MASE as primary and RMSSE/MAE/sMAPE as secondary, with explicit constant-series
handling. Include task-history hash, candidate identity, screening-policy hash, fold definition,
and runtime settings in the cache key.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_numerical_selector.py -k 'fold or diagnostic'`

Expected: all selected tests pass.

### Task 7: Numerical Selection and Guarded Ensemble

**Files:**
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Modify: `tests/test_evolution_numerical_selector.py`

**Interfaces:**
- Produces: `DecisionPolicy`, `SelectionDecision`,
  `select_numerical_forecast(policy, profile, active, diagnostics, forecasts)`.

- [ ] **Step 1: Write failing selection tests**

Test reliability gate, Pareto front over median/recent/worst MASE and MASE MAD, deterministic
tie-breaking, no inactive selection, no failed selection, recent-regime preference, catastrophic
tail rejection, and top-three ensemble constraints. Assert an ensemble is rejected unless its
historical-fold blend beats the best member and members pass the diversity threshold.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_numerical_selector.py -k 'select or ensemble or pareto'`

Expected: missing selection behavior failures.

- [ ] **Step 3: Implement the initial policy**

Use hard reliability gates, then a Pareto front, then lexicographic median MASE, recent MASE,
worst MASE, MASE MAD, normalized bias, and candidate name. Allow a validated ensemble only when
historical folds prove strict improvement; otherwise select one method. Return typed reason codes
and confidence derived from the gap to the next valid candidate.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_evolution_numerical_selector.py -k 'select or ensemble or pareto'`

Expected: all selected tests pass.

### Task 8: Numerical Selector Evolution and Freeze

**Files:**
- Create: `numerical_agent/evolution/selector_evolution.py`
- Create: `numerical_agent/run_selector_evolution.py`
- Create: `scripts/run_numerical_selector_evolution.sh`
- Create: `tests/test_evolution_selector_evolution.py`
- Create: `tests/test_numerical_selector_script.py`

**Interfaces:**
- Produces: evolved/frozen `decision_policy.py`, generation traces, decisions JSONL, and Train/Dev
  reports bound to the frozen screening-policy hash.

- [ ] **Step 1: Write failing mutation and acceptance tests**

Test that the Meta-Harness Agent may change only ranking order/weights, recent-fold weighting,
fold coverage, ensemble diversity/improvement thresholds, fallback rules, and a bounded rubric.
Reject scorer, split, task profile, candidate identity, active dictionary, cache, and future-label
changes. Assert Dev coverage, mean/median MASE, catastrophic rate, and active-oracle regret gates.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_evolution_selector_evolution.py tests/test_numerical_selector_script.py`

Expected: missing evolution/runner failures.

- [ ] **Step 3: Implement evolution and successive halving**

Use 8/2 for the smoke run. For formal 80/20, screen each Child on a fixed four-task Train subset,
promote at most three Children, complete 80 Train for promoted Children, and use 20 read-only Dev
tasks for final acceptance. Persist transient runtime failures separately from forecasting
failures and support checkpoint/resume.

- [ ] **Step 4: Verify GREEN and run 8/2**

Run both test files, then execute the 8/2 script against the frozen Phase-A hash. Confirm at least
two task-conditioned decisions are materialized; do not require artificial selection diversity.

- [ ] **Step 5: Run and freeze 80/20**

Complete formal Train/Dev evolution, write `frozen_decision_policy.py`, and record its SHA-256 and
the bound screening-policy SHA-256. Do not inspect Public Test.

### Task 9: One-Time 99-Task Evaluation

**Files:**
- Create: `numerical_agent/evaluate_frozen_two_stage.py`
- Create: `tests/test_frozen_two_stage_evaluation.py`
- Generate: `runs/task_conditioned_screening/<run-id>/FINAL_TWO_STAGE_REPORT.md`

**Interfaces:**
- Consumes: frozen screening and decision hashes, exact 99-task partition, caches/runtimes.
- Produces: the five frozen comparison rows and per-task immutable artifacts.

- [ ] **Step 1: Write failing freeze/no-leak tests**

Assert evaluation refuses missing/mismatched hashes, dirty policy files, non-99 task partitions,
and any mutation/LLM client. Assert rows A-E share tasks, candidate outcomes, and metrics.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_frozen_two_stage_evaluation.py`

Expected: missing evaluator failure.

- [ ] **Step 3: Implement frozen evaluation**

Generate Current baseline, Screening-only, Decision-only, Full system, and Toto singleton rows.
Report mean/median MASE, RMSSE, MAE, sMAPE, coverage, active-oracle regret, catastrophic rate,
method/family diversity, ensemble rate, and paired win/loss/tie counts. Never write back policy,
skills, prompts, or caches used for acceptance.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_frozen_two_stage_evaluation.py`

Expected: all selected tests pass.

- [ ] **Step 5: Run Public Test once and verify artifacts**

Execute the evaluator once. Verify exact task count, policy hashes, zero mutation calls, complete
per-task outputs, and report consistency. Run focused tests, `compileall`, shell syntax checks, and
`git diff --check` before reporting results.

