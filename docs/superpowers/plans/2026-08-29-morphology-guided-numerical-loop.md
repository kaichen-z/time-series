# Morphology-Guided Numerical Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make reviewed history-only morphology guide Statistical/TSFM eligibility,
multi-parent Combined evolution, safe Numerical selection, and the four-field assumption hand-off
to two-stage Retrieval, then run the frozen 80/20 and Public-99 experiment chain.

**Architecture:** Deterministic Python `TaskProfile` remains the numerical routing authority. A
bounded LLM tool loop may produce grounded assumptions only after candidates are executed; a
deterministic gate decides whether those assumptions may narrow the safe selector. Train-only
morphology-group evidence guides strict Combined proposals, while Dev/Public/hidden paths remain
read-only and releases bind every component fingerprint.

**Tech Stack:** Python 3.12+, frozen dataclasses, strict JSON, Codex CLI backends, pytest, Dr-CiK
80/20/99 split, existing Statistical and TSFM worker runtimes.

**Spec:** `docs/superpowers/specs/2026-08-29-morphology-guided-numerical-loop-design.md`

## Global Constraints

- Do not train or modify LLM, TSFM, or Statistical-model weights.
- Numerical/Morphology inference may read only historical values, frequency, horizon, reviewed
  method identities, and history-derived diagnostics.
- Round 1 Retrieval remains assumption-blind; Round 2 receives only `assumption_id`, `kind`,
  `claim`, and `failure_condition`.
- Dev is exactly-once and read-only. Public-99 is frozen regression only. Hidden-80 is
  inference/submission only.
- Keep Toto/Safe-Anchor baseline protection, worst-fold, coverage, and catastrophe gates.
- Use `gpt-5.6-luna` low for diagnosis and `gpt-5.6-terra` medium for Combined/Morphology
  proposals. Record any one-call `gpt-5.6-sol` high fallback.
- Preserve legacy behavior unless the new mode is explicitly enabled.
- Work in an isolated worktree. Do not absorb dirty main-worktree WIP without file-by-file review.

---

### Task 1: Establish the isolated baseline and WIP inventory

**Files:**
- Read: `docs/superpowers/specs/2026-08-29-morphology-guided-numerical-loop-design.md`
- Read: `numerical_agent/evolution/{screening,numerical_selector,combined_evolution}.py`
- Create: `.superpowers/sdd/2026-08-29-morphology-guided-loop/wip-inventory.md`

**Interfaces:**
- Consumes: clean `main` plus a read-only inventory of uncommitted WIP.
- Produces: isolated branch `feature/morphology-guided-numerical-loop` with a green baseline.

- [ ] **Step 1: Create the isolated worktree**

Use `using-git-worktrees`. Verify `.worktrees/` is ignored before creation.

- [ ] **Step 2: Record but do not copy WIP**

List each dirty main-worktree path, apparent responsibility, and the plan task that supersedes it.
Exclude caches, hidden inputs, and secrets.

- [ ] **Step 3: Run the clean focused baseline**

```bash
pytest -q \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_numerical_selector.py \
  tests/test_numerical_selector_script.py \
  tests/test_retrieval_e2e.py
```

Expected: zero failures/errors. Stop and diagnose any baseline failure.

- [ ] **Step 4: Commit**

```bash
git add .superpowers/sdd/2026-08-29-morphology-guided-loop/wip-inventory.md
git commit -m "docs(numerical): inventory morphology WIP"
```

### Task 2: Implement bounded history-only Morphology Cards

**Files:**
- Create: `numerical_agent/evolution/morphology.py`
- Create: `tests/test_evolution_morphology.py`
- Modify: `numerical_agent/evolution/__init__.py`

**Interfaces:**
- Consumes: `LLMClient`, reviewed `analysis_skills_template.py` functions, history, frequency,
  horizon, and active candidate names/families.
- Produces: `MorphologyReasoner.reason(...) -> MorphologyCard` with assumptions grounded in exact
  executed tool-call IDs.

- [ ] **Step 1: Write RED tests**

The wished-for API is:

```python
card = MorphologyReasoner(fake_client, max_turns=3, max_tool_calls=4).reason(
    history=history,
    frequency="D",
    horizon=3,
    active_names=("seasonal_naive", "toto_2_0"),
    families={"seasonal_naive": "statistical", "toto_2_0": "tsfm"},
)
assert card.assumption_call_ids("weekly_cycle") == ("broad_period", "recent_period")
```

Also reject unknown tools, invented/duplicate call IDs, invalid windows, inactive candidates,
non-finite outputs, schema drift, and finalization without distinct broad/recent inspections.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_evolution_morphology.py -x`.

Expected: import/attribute failure for the absent Morphology API.

- [ ] **Step 3: Implement the minimal tool loop**

Add immutable `MorphologyToolCall`, `MorphologyObservation`, `AssumptionGrounding`, and
`MorphologyCard`. Accept exact `tool` and `final` JSON actions, execute only reviewed functions,
enforce budgets, and fingerprint canonical JSON with `allow_nan=False`.

- [ ] **Step 4: Verify GREEN**

```bash
pytest -q tests/test_evolution_morphology.py tests/test_analysis_skills.py
```

- [ ] **Step 5: Commit**

```bash
git add numerical_agent/evolution/morphology.py numerical_agent/evolution/__init__.py \
  tests/test_evolution_morphology.py
git commit -m "feat(numerical): add morphology reasoner"
```

### Task 3: Gate assumptions and assign Train-only credit

**Files:**
- Create: `numerical_agent/evolution/morphology_consistency.py`
- Create: `numerical_agent/evolution/morphology_credit.py`
- Create: `tests/test_evolution_morphology_consistency.py`
- Modify: `tests/test_evolution_morphology.py`

**Interfaces:**
- Consumes: `MorphologyCard`, `TaskProfile`, active names, diagnostics, forecasts, and policy.
- Produces `check_morphology_assumptions(...) -> AssumptionConsistencyResult` and
  `assign_tool_call_credit(...) -> MorphologyCreditTrace`.

- [ ] **Step 1: Write RED consistency tests**

```python
result = check_morphology_assumptions(
    card,
    profile=profile,
    active_names=active,
    diagnostics=diagnostics,
    forecasts=forecasts,
    min_successful_folds=3,
)
assert tuple(x.assumption_id for x in result.accepted) == ("weekly_cycle",)
assert result.rejected["trend_persistence"] == "profile_incompatible"
```

Cover periodic/trend/intermittent/regime compatibility, inactive/failed candidates, insufficient
folds, catastrophe/worst-fold protection, and diversity. Rejection must leave the candidate and
protected fallback available.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_evolution_morphology_consistency.py -x`.

- [ ] **Step 3: Implement the deterministic gate**

Use typed kind-to-profile predicates and existing `rank_diverse_assumptions`; never execute LLM
text or synthesize a forecast.

- [ ] **Step 4: Add RED/GREEN credit tests**

Prove `future_truth` exists only in the trusted credit evaluator, cannot mutate the frozen card,
and each reward is the marginal sMAE/sRMSE improvement after grounded calls become available.

Run:

```bash
pytest -q tests/test_evolution_morphology.py tests/test_evolution_morphology_consistency.py
```

- [ ] **Step 5: Commit**

```bash
git add numerical_agent/evolution/morphology_consistency.py \
  numerical_agent/evolution/morphology_credit.py \
  tests/test_evolution_morphology.py tests/test_evolution_morphology_consistency.py
git commit -m "feat(numerical): gate morphology assumptions"
```

### Task 4: Feed morphology-group evidence into Combined proposals

**Files:**
- Modify: `numerical_agent/evolution/combined_evolution.py`
- Modify: `numerical_agent/evolution/portfolio.py`
- Create: `tests/test_evolution_combined_morphology.py`
- Modify: `tests/test_evolution_portfolio.py`

**Interfaces:**
- Consumes: fixed `TaskProfile` buckets and trusted Train-only aggregate outcome metrics.
- Produces: immutable `MorphologyGroupEvidence` in `CombinedProposalDiagnostics` and executable
  Combined routes over reviewed signals.

- [ ] **Step 1: Write RED aggregate tests**

```python
evidence = MorphologyGroupEvidence(
    group_id="periodic_high_confidence",
    feature="periodicity_strength",
    operator="at_least",
    threshold=0.6,
    task_count=8,
    entity_count=3,
    eligible_leaves=("timesfm_2_5", "seasonal_naive"),
    baseline="toto_2_0",
    winsorized_smae_delta=-0.03,
    winsorized_srmse_delta=-0.01,
    coverage=1.0,
    failure_rate=0.0,
)
```

Reject raw task IDs/timestamps/futures/documents, unsupported features, insufficient entity
support, non-finite metrics, unknown leaves, and hostile containers.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_evolution_combined_morphology.py -x`.

- [ ] **Step 3: Implement strict aggregation and prompt projection**

Group Train outcomes by fixed profile predicates. Extend the proposal prompt with canonical
aggregates only. Keep parsing/application atomic and fail-closed.

- [ ] **Step 4: Extend reviewed route signals**

Add `noise_relative_scale`, `intermittency_adi`, `history_length`, `horizon`, and `horizon_ratio`
to prompt, parser, policy validation, and runtime. Compute them in reviewed Python only.

- [ ] **Step 5: Verify GREEN and compatibility**

```bash
pytest -q \
  tests/test_evolution_combined_morphology.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_policy_targetwise.py
```

- [ ] **Step 6: Commit**

```bash
git add numerical_agent/evolution/combined_evolution.py \
  numerical_agent/evolution/portfolio.py \
  tests/test_evolution_combined_morphology.py tests/test_evolution_portfolio.py
git commit -m "feat(numerical): guide combined proposals"
```

### Task 5: Integrate the Morphology-guided Numerical loop

**Files:**
- Create: `numerical_agent/evolution/numerical_loop.py`
- Create: `tests/test_numerical_morphology_loop.py`
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Modify: `numerical_agent/evolution/__init__.py`

**Interfaces:**
- Consumes: history-only `TaskProfile`, screened Statistical/TSFM leaves, materialized Combined
  forecasts, hindcast diagnostics, and an optional valid `MorphologyCard`.
- Produces: immutable `NumericalForecastPackage` containing the selected forecast, protected
  baseline, ranked alternatives, accepted assumptions, safe Retrieval hand-off, diagnostics,
  component fingerprints, and fallback reason.

- [ ] **Step 1: Write RED end-to-end unit tests**

Construct deterministic fake Statistical, TSFM, and Combined candidates. Prove morphology changes
eligibility before materialization; each leaf runtime executes at most once; Combined policies use
cached leaves; selected output is finite and horizon-sized; malformed morphology falls back to the
protected Safe-Anchor; assumption guidance cannot bypass coverage, worst-fold, catastrophe, or
fallback gates; and accepted top-k assumptions serialize to exactly four Retrieval fields.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_numerical_morphology_loop.py -x
```

Expected: import failure for `NumericalForecastPackage`/`run_numerical_loop`.

- [ ] **Step 3: Implement the orchestration boundary**

Add `run_numerical_loop(...) -> NumericalForecastPackage`. The host computes `TaskProfile`,
resolves the active dictionary, executes leaves once, evaluates Combined policies from materialized
forecasts, runs history-only hindcasts, optionally requests a Morphology card, applies deterministic
consistency checks, and then calls the protected selector. Preserve the legacy selector when no
Morphology reasoner is supplied.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

```bash
pytest -q tests/test_numerical_morphology_loop.py tests/test_evolution_numerical_selector.py tests/test_evolution_portfolio.py tests/test_evolution_screening.py
```

- [ ] **Step 5: Commit**

```bash
git add numerical_agent/evolution/numerical_loop.py numerical_agent/evolution/numerical_selector.py numerical_agent/evolution/__init__.py tests/test_numerical_morphology_loop.py
git commit -m "feat(numerical): integrate morphology loop"
```

### Task 6: Add the formal 8/2 and 80/20 Numerical evolution runner

**Files:**
- Create: `numerical_agent/run_numerical_morphology_evolution.py`
- Create: `scripts/run_numerical_morphology_evolution.sh`
- Create: `tests/test_numerical_morphology_evolution_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: pinned Dr-CiK split manifest, seed Numerical release, Train-only outcome cache, Codex
  backend configuration, evolution budgets, and output directory.
- Produces: append-only generation trace, accepted/rejected child artifacts, frozen Numerical
  release, read-only Dev report, and component/model/input fingerprints.

- [ ] **Step 1: Write RED CLI and lifecycle tests**

Use eight fake Train tasks plus two fake Dev tasks. Assert the runner loads Train first; never
decodes Dev labels during proposal/screening/full-Train; diagnoses at most eight targets; evaluates
every Child on identical screen tasks; promotes at most two Children; consults Dev exactly once;
never writes Skills from Dev; resumes without repeating completed calls; records every model and
reasoning effort; and leaves exact Parent bytes unchanged after rejection.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_numerical_morphology_evolution_cli.py -x
```

- [ ] **Step 3: Implement the runner and shell entry point**

Required CLI shape:

```bash
python -m numerical_agent.run_numerical_morphology_evolution --split-manifest configs/drcik_split_80_20_99.json --seed-release runs/numerical_selector/releases/safe_anchor_v1 --output runs/numerical_morphology/formal_80_20 --diagnosis-model gpt-5.6-luna --diagnosis-reasoning low --proposal-model gpt-5.6-terra --proposal-reasoning medium --screen-train 8 --promote 2 --train-limit 80 --dev-limit 20
```

The wrapper forwards every option, supports `--dry-run`, uses strict bash mode, and embeds no
credentials.

- [ ] **Step 4: Verify GREEN**

```bash
pytest -q tests/test_numerical_morphology_evolution_cli.py
bash -n scripts/run_numerical_morphology_evolution.sh
scripts/run_numerical_morphology_evolution.sh --dry-run
```

- [ ] **Step 5: Commit**

```bash
git add numerical_agent/run_numerical_morphology_evolution.py scripts/run_numerical_morphology_evolution.sh tests/test_numerical_morphology_evolution_cli.py pyproject.toml README.md
git commit -m "feat(numerical): add morphology evolution runner"
```

### Task 7: Connect accepted Numerical assumptions to Retrieval Round 2

**Files:**
- Modify: `evolving_loop/harness.py`
- Modify: `evolving_loop/cli.py`
- Modify: `evolving_loop/retrieval_agent/two_stage_agent.py`
- Modify: `tests/test_retrieval_e2e.py`
- Create: `tests/test_numerical_retrieval_handoff.py`

**Interfaces:**
- Consumes: accepted `NumericalForecastPackage` and explicit `retrieval_mode=two_stage`.
- Produces: assumption-blind Round 1 evidence, provisional Decision gaps, four-field targeted Round
  2 evidence, and the final Decision payload.

- [ ] **Step 1: Write RED hand-off tests**

Assert recursively that Round 1 contains no assumptions, candidate scores, or Numerical selection.
Assert Round 2 receives only `assumption_id`, `kind`, `claim`, and `failure_condition`. Cover
empty/malformed assumptions, injection strings, rejected assumptions, no-evidence Round 2, and
legacy `single_pass` compatibility.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_numerical_retrieval_handoff.py -x
```

- [ ] **Step 3: Implement the typed adapter**

Do not expose candidate forecasts or diagnostics to Retrieval. Validate and copy only the four
strings, enforce count/length budgets, and fall back to Round 1 if Round 2 is invalid/unavailable.

- [ ] **Step 4: Verify GREEN**

```bash
pytest -q tests/test_numerical_retrieval_handoff.py tests/test_retrieval_e2e.py
```

- [ ] **Step 5: Commit**

```bash
git add evolving_loop/harness.py evolving_loop/cli.py evolving_loop/retrieval_agent/two_stage_agent.py tests/test_numerical_retrieval_handoff.py tests/test_retrieval_e2e.py
git commit -m "feat(retrieval): consume morphology assumptions"
```

### Task 8: Run deterministic verification before any paid experiment

**Files:**
- Create: `runs/numerical_morphology/smoke_8_2/RUN_MANIFEST.json`
- Create: `runs/numerical_morphology/smoke_8_2/README.md`
- Modify: `README.md`

- [ ] **Step 1: Run focused and full tests**

```bash
pytest -q tests/test_evolution_morphology.py tests/test_evolution_morphology_consistency.py tests/test_evolution_combined_morphology.py tests/test_numerical_morphology_loop.py tests/test_numerical_morphology_evolution_cli.py tests/test_numerical_retrieval_handoff.py tests/test_retrieval_e2e.py
pytest -q
```

- [ ] **Step 2: Run static verification**

```bash
python -m compileall -q numerical_agent evolving_loop tests
bash -n scripts/run_numerical_morphology_evolution.sh
git diff --check
```

- [ ] **Step 3: Run a deterministic fake 8/2 smoke**

Run two generations with cached fake forecasts and fake LLM clients. Verify one rejection, one
accepted candidate, exactly-once Dev, reproducible hashes, and no Public task IDs.

- [ ] **Step 4: Document the smoke honestly**

Record exact command, commit, split fingerprint, fixtures, metrics, accepted/rejected policy, and
limitations. Do not describe it as a real forecasting result.

- [ ] **Step 5: Commit**

```bash
git add README.md runs/numerical_morphology/smoke_8_2/RUN_MANIFEST.json runs/numerical_morphology/smoke_8_2/README.md
git commit -m "test(numerical): verify morphology loop"
```

### Task 9: Run and freeze the real Numerical 8/2 then 80/20 experiment

**Files:**
- Create: `runs/numerical_morphology/pilot_8_2/`
- Create: `runs/numerical_morphology/formal_80_20/`
- Create: `docs/results/NUMERICAL_MORPHOLOGY_80_20_REPORT.md`

- [ ] **Step 1: Freeze the run manifest before inference**

Pin code commit, Dr-CiK split hash, 103-candidate dictionary hash, five TSFM
manifests/checkpoints, baseline releases, prompts, models/reasoning, budgets, metrics, seeds, and
acceptance rules.

- [ ] **Step 2: Execute the real 8/2 pilot**

Use Luna-low diagnosis and Terra-medium proposals. Stop if runtime/model/checkpoint failure makes
methods incomparable; never score a transport failure as forecasting failure.

- [ ] **Step 3: Inspect only pilot artifacts**

Require finite forecasts, real task-conditioned candidate variation, zero label leakage,
meaningful Combined proposals, and at least one accepted/rejected assumption trace. Any code fix
requires a tested commit and a new run ID.

- [ ] **Step 4: Execute the formal 80 Train / 20 Dev run once**

Do not inspect Public-99. Produce Parent/Child Train metrics, exactly-once Dev metrics, selection
trace, assumption/card/Combined summaries, failure counts, and accepted release hashes.

- [ ] **Step 5: Freeze or retain Parent**

Publish a new Numerical release only if every predeclared gate passes. Otherwise retain the exact
Parent and report failed hypotheses without further Dev tuning.

- [ ] **Step 6: Commit reproducible, nonsecret artifacts**

```bash
git add runs/numerical_morphology/pilot_8_2 runs/numerical_morphology/formal_80_20 docs/results/NUMERICAL_MORPHOLOGY_80_20_REPORT.md
git commit -m "results(numerical): freeze morphology experiment"
```

### Task 10: Evolve Retrieval and Decision against the frozen Numerical release

**Files:**
- Modify: `configs/retrieval_evolution_80_20.json`
- Create: `runs/retrieval_evolution/morphology_numerical_80_20/`
- Create: `runs/decision_evolution/morphology_retrieval_80_20/`
- Create: `docs/results/RETRIEVAL_DECISION_80_20_REPORT.md`

- [ ] **Step 1: Bind the accepted Numerical release into the Harness seed**

Fingerprint the Numerical release, morphology policy, Combined policy, selector, TSFM registry,
and assumption projection. Refuse non-v000 Retrieval/Decision evolution if any binding is missing
or mismatched.

- [ ] **Step 2: Run Retrieval-first evolution**

Use the existing two-stage controller and authenticated checkpoint/anchor protocol. Train screen
uses eight complete-entity cases, promotes at most two proposals, then runs remaining Train folds
and exactly-once read-only Dev. Public IDs/bodies remain unopened.

- [ ] **Step 3: Freeze or retain Retrieval**

Accept only a provenance-bound release that improves predeclared evidence and final forecast
gates. Otherwise preserve exact v000/Parent bytes.

- [ ] **Step 4: Run Decision coordinate evolution second**

Decision may modify only its prompt/genome/approved policy fields. It must not change Numerical or
Retrieval fingerprints and must retain baseline/tail-risk protection.

- [ ] **Step 5: Freeze or retain Decision and document the trace**

Report separate Numerical, Retrieval, and Decision rewards plus final sMAE/sRMSE. Do not claim
co-evolution when only one coordinate changed.

- [ ] **Step 6: Commit**

```bash
git add configs/retrieval_evolution_80_20.json runs/retrieval_evolution/morphology_numerical_80_20 runs/decision_evolution/morphology_retrieval_80_20 docs/results/RETRIEVAL_DECISION_80_20_REPORT.md
git commit -m "results(harness): freeze coordinate evolution"
```

### Task 11: Run one frozen Public-99 regression and publish the comparison

**Files:**
- Create: `runs/frozen_public99/morphology_full_system/`
- Create: `docs/results/MORPHOLOGY_FULL_SYSTEM_PUBLIC99_REPORT.md`
- Modify: `docs/forecasting_pipeline_full_2026-08-26.html`
- Modify: `docs/forecasting_pipeline_full_2026-08-26_en.html`

- [ ] **Step 1: Verify every release is frozen**

Require exact fingerprints for dictionary, TSFM manifests/checkpoints, screening, selector,
morphology, Combined, Retrieval, Decision, code commit, and Public split. Require `llm_calls=0`,
`mutation_calls=0`, and write-free Skill libraries.

- [ ] **Step 2: Evaluate the frozen comparison once**

Compare on identical 99 tasks: fixed Toto; previous Safe-Anchor; morphology-guided Numerical-only;
Numerical plus frozen two-stage Retrieval; and the full Numerical + Retrieval + Decision system.
Report Dr-CiK-aligned winsorized sMAE/sRMSE, failures, medians, MAE/sMAPE diagnostics, pairwise
wins/ties/losses, bootstrap intervals, and tail-risk sensitivity.

- [ ] **Step 3: Mark Public-99 consumed for this release family**

Write `evaluation_complete.json` with result/input/release hashes. Never use these 99 outcomes to
modify policy or rerun a tuned comparison in the same release family.

- [ ] **Step 4: Update English and Chinese reports**

Lead with the actual frozen outcome; separate implementation from forecasting results; identify
accepted/rejected coordinates; disclose failures and incomparable tasks.

- [ ] **Step 5: Verify artifacts and commit**

```bash
python -m json.tool runs/frozen_public99/morphology_full_system/evaluation_complete.json >/dev/null
python -m compileall -q numerical_agent evolving_loop
git diff --check
git add runs/frozen_public99/morphology_full_system docs/results/MORPHOLOGY_FULL_SYSTEM_PUBLIC99_REPORT.md docs/forecasting_pipeline_full_2026-08-26.html docs/forecasting_pipeline_full_2026-08-26_en.html
git commit -m "results(harness): publish morphology regression"
```

## Completion Criteria

- Morphology changes parent eligibility/routing before candidate selection.
- Grounded assumptions are advisory, top-k, consistency-gated, and projected to Retrieval through
  exactly four fields.
- Combined proposals can use 2–5 Statistical/TSFM parents and reviewed morphology route signals.
- Numerical, Retrieval, and Decision releases have independent immutable fingerprints and
  coordinate-specific acceptance traces.
- Real 80/20 evolution completes before any new Public-99 access.
- Public-99 is evaluated once, frozen, and never used for subsequent tuning.
- All focused/full tests, compilation, shell syntax, artifact validation, and diff checks pass.
