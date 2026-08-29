# Multi-Parent Combined Final Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six remaining review findings so the evolvable multi-parent Combined implementation is safe, numerically total, compatible with variable portfolios, and ready for final merge review.

**Architecture:** Keep the existing canonical policy and execution graph. Harden three independent boundaries: the typed LLM proposal adapter, overflow-stable pointwise combination, and the task-conditioned screening command's derived candidate-count contract. Each task has its own RED/GREEN cycle and independent reviewer gate.

**Tech Stack:** Python 3, frozen dataclasses, strict JSON parsing, pytest, Git worktrees.

**Spec:** `docs/superpowers/specs/2026-08-28-evolvable-multi-parent-combined-design.md`

## Global Constraints

- Do not train, fine-tune, merge, or modify LLM or TSFM weights.
- Preserve the five manifest-bound TSFM identities, checkpoints, adapters, runtime options, and order.
- Combined inference and mutation may consume only historical task fields, typed label-free aggregates, and already materialized leaf forecasts.
- Do not expose future values, retrieved documents, GT evidence, role/subtype labels, Public Regression labels, hidden labels, runtime secrets, or checkpoint substitutions.
- Every Combined policy has two to five unique leaf parents, at least one TSFM parent, and no Combined-to-Combined dependency.
- Policy artifacts remain literal-only, and every invalid mutation returns the exact Parent object.
- Formal 80/20 proposal acceptance remains a separate follow-up; this plan adds no performance claim.

---

### Task 1: Seal the Agent Proposal Contract

**Files:**
- Modify: `numerical_agent/evolution/combined_evolution.py`
- Test: `tests/test_evolution_combined_evolution.py`

**Interfaces:**
- Consumes: exact `PolicyPortfolio`, exact `CombinedPolicy` members, exact `CombinedProposalDiagnostics`, reviewed Statistical names, and one LLM response.
- Produces: one bounded canonical proposal without polymorphic serialization, ambiguous wrappers, or unbounded diagnostic values.

- [ ] **Step 1: Write failing exact-diagnostics tests**

Assert that `CombinedProposalDiagnostics` accepts only exact built-in types: positive exact `int` for `history_length`, non-negative exact finite `float` for disagreement fields, and non-negative exact `int` counts. Reject booleans, integer disagreement, numeric subclasses, `NaN`, infinities, negative values, values above documented caps, and non-exact dataclass subclasses before an LLM call. Use caps of `1_000_000` for lengths/counts and `1_000_000.0` for disagreement aggregates.

- [ ] **Step 2: Write failing response-wrapper tests**

Accept exactly one of: a bare JSON object, one closed `<think>...</think>` prefix plus a bare object, or one JSON fence containing one object. Reject stacked think-plus-fence wrappers, multiple think wrappers, multiple fences, concatenated objects, trailing text, and unmatched wrappers.

- [ ] **Step 3: Write failing serialization and DSL tests**

Construct valid subclasses of `PolicyPortfolio`, `CombinedPolicy`, and TSFM policy records whose overridden serialization would emit a sentinel. Assert rejection before serialization/LLM. Pin the prompt's complete enforced DSL: exact canonical keys and JSON types; two-to-five unique leaf parents; at least one fixed TSFM; no Combined parent; all four operators; weight, route, empty non-route branch, fallback, signal, identifier, reason, portfolio-size, unique-target, and add/repair/fork/remove naming constraints; all five TSFM names but no manifest internals.

- [ ] **Step 4: Run RED tests**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py \
  -k 'diagnostic_contract or wrapper_contract or nonpolymorphic or complete_dsl'
```

Expected: the new adversarial cases fail against `a9732a2`.

- [ ] **Step 5: Implement the minimal hardening**

Validate exact object/member types before serialization. Serialize canonical fields by direct attribute access, never `to_payload()` dispatch. Make wrappers mutually exclusive. Enforce exact diagnostic types and caps in `CombinedProposalDiagnostics.__post_init__`. Expand the fixed system/user schema text so every Python-enforced rule is stated explicitly.

- [ ] **Step 6: Verify and commit**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_policy_targetwise.py \
  tests/test_evolving_agent_llm.py
../../.venv/bin/python -m compileall -q numerical_agent tests
git diff --check
git add numerical_agent/evolution/combined_evolution.py tests/test_evolution_combined_evolution.py
git commit -m 'fix(numerical): seal combined proposal contract'
```

---

### Task 2: Make Combined Arithmetic Total and Overflow-Stable

**Files:**
- Modify: `numerical_agent/evolution/portfolio.py`
- Test: `tests/test_evolution_portfolio.py`

**Interfaces:**
- Consumes: finite horizon-aligned materialized parent forecasts.
- Produces: either a finite Combined forecast, an explicit successful fallback, or a canonical `INVALID` outcome; arithmetic exceptions never escape.

- [ ] **Step 1: Write failing extreme-value tests**

Cover `weighted_mean`, `median`, `trimmed_mean`, and route using finite values near `sys.float_info.max`. Reproduce the reviewed five-parent trimmed-mean overflow. Assert overflow/cancellation never raises from `combine_materialized_outcome()` or `_run_combined()`.

- [ ] **Step 2: Write failing fallback/status tests**

When composition is non-finite or arithmetic raises, assert a valid explicit fallback succeeds and records `fallback=<name>`. Without a valid fallback, assert `OutcomeStatus.INVALID`, empty forecast, and a sanitized detail. Preserve `CRASHED > INVALID > NOT_APPLICABLE` when a parent itself fails.

- [ ] **Step 3: Run RED tests**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_portfolio.py \
  -k 'overflow or extreme or arithmetic_fallback'
```

Expected: the current trimmed mean raises `OverflowError`.

- [ ] **Step 4: Implement overflow-stable pointwise operators**

Use scale-normalized weighted averaging and an overflow-stable mean rather than raw `math.fsum(values) / n`. Validate every point immediately. Translate operator/signal arithmetic failures into the same invalid/fallback path as structurally invalid parent output. Do not change normal-range golden forecasts.

- [ ] **Step 5: Verify and commit**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py \
  tests/test_numerical_selector_script.py \
  tests/test_evolution_filtering.py
../../.venv/bin/python -m compileall -q numerical_agent tests
git diff --check
git add numerical_agent/evolution/portfolio.py tests/test_evolution_portfolio.py
git commit -m 'fix(numerical): stabilize combined arithmetic'
```

---

### Task 3: Derive Screening Candidate Counts and Reverify the Branch

**Files:**
- Modify: `numerical_agent/run_task_conditioned_screening.py`
- Modify: `numerical_agent/README.md`
- Modify: `README.md`
- Modify: `scripts/run_task_conditioned_screening.sh`
- Test: `tests/test_task_conditioned_screening_script.py`

**Interfaces:**
- Consumes: Statistical module names plus the current one-to-32 Combined portfolio.
- Produces: a screening run whose expected candidate count and default ceiling are derived at runtime rather than fixed at 103.

- [ ] **Step 1: Write failing variable-count CLI tests**

Create 103- and 104-entry parent screening policies from a 93-method module plus ten or eleven portfolio policies. Assert both pass when their entries exactly match the runtime namespace. Assert missing, duplicate, or extra entries fail before cache/model work.

- [ ] **Step 2: Write failing ceiling/default tests**

When `--screen-max-candidates` is omitted, assert the effective ceiling equals the derived candidate count. When explicitly provided, require `1 <= ceiling <= derived_count`. Update shell dry-run tests so no literal `103` default remains while the historical initial 103 count remains documented as an example.

- [ ] **Step 3: Run RED tests**

```bash
../../.venv/bin/python -m pytest -q tests/test_task_conditioned_screening_script.py
```

Expected: the 104-candidate path fails the fixed `len(parent.entries) != 103` gate.

- [ ] **Step 4: Implement derived count validation and truthful docs**

Compute the runtime namespace from the parsed Statistical module and current `PolicyPortfolio`; require the parent entry-name set to equal it exactly. Derive the default ceiling only after loading that namespace. Keep the documented 93+5+5=103 as the initial portfolio, and state that accepted Combined additions increase the runtime count.

- [ ] **Step 5: Run focused and full verification**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_task_conditioned_screening_script.py \
  tests/test_evolution_screening.py \
  tests/test_evolution_screening_evolution.py
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m compileall -q numerical_agent tests
bash -n scripts/run_task_conditioned_screening.sh
git diff --check
```

Expected: `1873+` tests pass with the existing single intentional skip, and all static checks exit zero.

- [ ] **Step 6: Commit**

```bash
git add README.md numerical_agent/README.md \
  numerical_agent/run_task_conditioned_screening.py \
  scripts/run_task_conditioned_screening.sh \
  tests/test_task_conditioned_screening_script.py
git commit -m 'fix(numerical): derive screening portfolio size'
```

---

### Final Review Gate

- [ ] Dispatch one fresh high-tier reviewer over `a9732a2..HEAD` and the original design plus this plan.
- [ ] Fix every Critical or Important finding under a new task-specific review cycle.
- [ ] Rerun the full suite, compileall, shell syntax, and `git diff --check` after the final fix.
- [ ] Do not merge or push until the reviewer returns `Ready to merge: Yes`.
