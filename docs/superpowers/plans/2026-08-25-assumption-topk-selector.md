# Assumption-Guided Top-k Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a history-only assumption generator, diverse Top-k routing, and a baseline-protected Verifier to the Numerical Selector.

**Architecture:** A focused assumption module converts TaskProfile fields into typed, falsifiable hypotheses and ranks each hypothesis using existing hindcast diagnostics. The existing Decision selector then operates only on the diverse Top-k-supported pool plus reviewed TSFM anchors, so current safety gates remain intact.

**Tech Stack:** Python 3.10+, frozen dataclasses, pytest, existing Numerical Selector and TaskProfile APIs.

**Spec:** `docs/superpowers/specs/2026-08-25-assumption-topk-selector.md`

## Global Constraints

- History-only: no documents, future values, GT evidence, or public-test labels enter generation or ranking.
- Public 99 must not be accessed.
- Existing Toto baseline protection and Combined gates stay authoritative.
- Old frozen DecisionPolicy sources must remain parseable with safe defaults.
- Changes are made in the existing main checkout because the user previously required direct main-branch work.

---

### Task 1: Typed Assumption Generation

**Files:**
- Create: `numerical_agent/evolution/assumptions.py`
- Test: `tests/test_evolution_assumptions.py`

**Interfaces:**
- Consumes: `TaskProfile`, active names, family map, and `CandidateDiagnostics`.
- Produces: `ForecastAssumption`, `RankedAssumption`, `generate_forecast_assumptions()`, and `rank_diverse_assumptions()`.

- [ ] **Step 1: Write failing tests** for periodic, intermittent, fallback, label-free, unique-kind, unique-leading-candidate, and anchor-retention behavior using literal TaskProfile fixtures.
- [ ] **Step 2: Run `python -m pytest -q tests/test_evolution_assumptions.py`** and verify failure because the module/API does not exist.
- [ ] **Step 3: Implement the minimal typed generator and deterministic diverse ranker** with finite confidence validation and explicit method-name routing tables.
- [ ] **Step 4: Re-run the focused tests** and require all assumption tests to pass.

### Task 2: Assumption-Guided Verifier

**Files:**
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Modify: `tests/test_evolution_numerical_selector.py`

**Interfaces:**
- Consumes: `DecisionPolicy`, `TaskProfile`, active candidates, diagnostics, forecasts, and families.
- Produces: `select_assumption_guided_forecast()` and assumption identifiers in `SelectionDecision`.

- [ ] **Step 1: Write failing tests** proving irrelevant candidates cannot reach the Verifier, reviewed anchors remain available, Top-k does not increase final ensemble size, and assumption tracing is deterministic.
- [ ] **Step 2: Run the new focused tests** and verify they fail because assumption-guided selection is absent.
- [ ] **Step 3: Add typed DecisionPolicy controls** (`assumption_guidance_enabled`, `assumption_top_k`, `assumption_candidates_per_hypothesis`, `assumption_min_confidence`) and implement the wrapper around the existing selector.
- [ ] **Step 4: Re-run assumption and selector tests** and require green results.

### Task 3: Meta-Harness and Evaluation Integration

**Files:**
- Modify: `numerical_agent/evolution/selector_evolution.py`
- Modify: `numerical_agent/evaluate_frozen_two_stage.py`
- Modify: `tests/test_evolution_selector_evolution.py`
- Modify: `tests/test_frozen_two_stage_evaluation.py`

**Interfaces:**
- Consumes: policy source JSON/Python and `DecisionCase`.
- Produces: backward-compatible policy parsing, bounded assumption-policy variants, and assumption-guided Train/Dev/frozen evaluation.

- [ ] **Step 1: Write failing tests** for source round-trip, legacy defaults, strict mutation bounds, Train/Dev evaluation routing, and frozen evaluation routing.
- [ ] **Step 2: Run the focused tests** and confirm failures identify the missing schema/routing behavior.
- [ ] **Step 3: Extend the exact policy schema and bounded search**, use `profile_task(case.task)` inside trusted history-only evaluation, and call the assumption-guided Verifier in both development and frozen paths.
- [ ] **Step 4: Run all focused selector and frozen-evaluation tests** and require green results.

### Task 4: 80/20 Experiment and Verification

**Files:**
- Output only: `runs/numerical_selector/assumption_topk_80_20_20260825/`

**Interfaces:**
- Consumes: the existing conditional-v11 screening artifacts and cached 80/20 DecisionCases.
- Produces: a frozen-candidate report and an accept/reject decision without opening Public 99.

- [ ] **Step 1: Run the existing selector evolution command** with three generations and four entity-grouped Train validation folds.
- [ ] **Step 2: Compare Parent and Child on Train and read-only Dev** for sMAE, sRMSE, P90/P95, clipped counts, and oracle regret.
- [ ] **Step 3: Run `python -m pytest -q`**, `python -m compileall -q numerical_agent common`, shell syntax checks, and `git diff --check`.
- [ ] **Step 4: Report whether the assumption-guided policy is accepted**, while leaving Public 99 untouched.

