# Conservative TSFM Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the old Toto-first Safe-Anchor fallback while allowing TimesFM, Statistical, and Combined challengers to override it only with stable history-only dominance evidence.

**Architecture:** Add one typed `conservative_tsfm` baseline strategy. It starts from Toto, permits TimesFM to become the task anchor only after three ordinary hindcasts plus one long-horizon audit satisfy the same conservative gate used by later challengers, and then evaluates Statistical/Combined candidates against the routed anchor. Search is bounded to two fixed children with 2% and 5% minimum median-improvement margins.

**Tech Stack:** Python 3, dataclasses, pytest, existing Numerical Selector and frozen-evaluation artifacts.

**Spec:** User-approved design in the 2026-08-26 conversation: Safe-Anchor fallback, conservative Toto/TimesFM routing, same dominance guard for downstream challengers, 80/20 development only, no reuse of the consumed 99-task Public Test.

## Global Constraints

- Do not read or score the consumed 99-task Public Test during development.
- Use only history-only hindcasts and long-horizon audit folds for routing.
- Preserve legacy `toto_first` and `minimax_tsfm` behavior and policy parsing.
- Generate exactly two conservative children: 2% and 5% minimum improvement.
- Require at least three strict wins across three ordinary folds plus one long audit, no positive worst-fold regret, and at least 75% audit coverage.
- Keep full forecast coverage and reject policies that worsen Dev tail/clipping gates.

---

### Task 1: Conservative Anchor Routing

**Files:**
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Test: `tests/test_evolution_numerical_selector.py`

**Interfaces:**
- Consumes: `DecisionPolicy`, `CandidateDiagnostics`, existing ordinary folds and `long_horizon_fold`.
- Produces: `baseline_strategy="conservative_tsfm"` and a routed TSFM anchor used by the existing single/Combined verifier.

- [ ] **Step 1: Write failing behavioral tests**

Add literal synthetic-fold cases proving that the router keeps Toto on audit regret or insufficient improvement, routes to TimesFM only after at least three of four strict wins with no positive regret, and evaluates a Statistical challenger against the routed TimesFM anchor.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_evolution_numerical_selector.py -k conservative_tsfm`

Expected: FAIL because `conservative_tsfm` is not an accepted strategy and no routed-anchor behavior exists.

- [ ] **Step 3: Implement the minimum routing behavior**

Extend `DecisionPolicy` validation, route from Toto to TimesFM only through a path-aligned dominance helper, and feed the resulting anchor into the existing challenger and combination gates. Preserve fail-closed behavior when audit evidence is missing.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_evolution_numerical_selector.py -k conservative_tsfm`

Expected: all selected tests pass.

### Task 2: Bounded Two-Child Search and Serialization

**Files:**
- Modify: `numerical_agent/evolution/selector_evolution.py`
- Test: `tests/test_evolution_selector_evolution.py`

**Interfaces:**
- Consumes: a parent `DecisionPolicy`.
- Produces: `bounded_conservative_router_candidates(parent)` returning parent plus fixed 2% and 5% conservative children; policy render/parse/hash remains deterministic.

- [ ] **Step 1: Write failing round-trip and bounded-search tests**

Assert the new strategy round-trips and the candidate function returns exactly the parent and two children with coverage `0.75`, zero ordinary/audit regret, and margins `0.02`/`0.05`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_evolution_selector_evolution.py -k conservative_tsfm`

Expected: FAIL because the bounded candidate function does not exist.

- [ ] **Step 3: Implement the typed bounded neighborhood**

Add the two-child constructor without adding a larger grid or changing legacy proposal search.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_evolution_selector_evolution.py -k conservative_tsfm`

Expected: all selected tests pass.

### Task 3: 80/20 Development Experiment

**Files:**
- Modify only if required: `numerical_agent/run_task_conditioned_audit_experiment.py`
- Create runtime artifacts under: `runs/numerical_selector/conservative_tsfm_router_80_20_20260826/`
- Test if runner behavior changes: `tests/test_task_conditioned_audit_experiment.py`

**Interfaces:**
- Consumes: existing cached 80 Train / 20 Dev `DecisionCase` artifacts and the two fixed children.
- Produces: one frozen development policy, per-child Train/Dev scores, changed-task counts, and an explicit `test_accessed: false` marker.

- [ ] **Step 1: Add a failing runner test only if a new runner mode is necessary**

The test must execute the real argument parsing/evaluation boundary on small cached cases and assert that no Test split is loaded.

- [ ] **Step 2: Run the two-child Train search**

Evaluate parent and both children with entity-disjoint Train folds. Select by mean sMAE subject to sRMSE, P90/P95, clipping, coverage, and oracle-regret gates.

- [ ] **Step 3: Perform one read-only Dev comparison**

Accept a child only if it improves Dev mean sMAE while preserving coverage, sRMSE, tail, and clipping constraints. Do not access the 99-task Public Test.

- [ ] **Step 4: Run focused and full verification**

Run focused selector/evaluator tests, `python -m pytest -q`, `python -m compileall -q numerical_agent tests`, and `git diff --check`.

