# Evolvable Multi-Parent Combined Forecasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Numerical Combined candidates variable-length, multi-parent, executable, and Agent-mutable across TSFM--TSFM and TSFM--Statistical families without training model weights.

**Architecture:** Replace the fixed pair schema with a canonical typed `CombinedPolicy` over two to five leaf parents. Keep all ordinary mutations inside a literal-only DSL, execute parents once before applying reviewed operators, and expose an atomic structured-operation adapter for an LLM Meta-Harness. Preserve the five current policies through a legacy migration path and keep formal 80/20 acceptance outside this change.

**Tech Stack:** Python 3.12, frozen dataclasses, AST literal parsing, existing Numerical outcome/cache/runtime APIs, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-evolvable-multi-parent-combined-design.md`

## Global Constraints

- Do not train, fine-tune, merge, or modify LLM or TSFM weights.
- Preserve the five manifest-bound TSFM identities, checkpoints, adapters, runtime options, and order.
- Combined execution may read only historical task fields and already materialized leaf forecasts.
- No future values, retrieved documents, GT evidence, role/subtype labels, Public Regression labels, or hidden labels may enter mutation or inference.
- Combined policies have two to five unique leaf parents, at least one TSFM parent, and no Combined-to-Combined dependency.
- Policy artifacts remain literal-only Python and are never imported or executed.
- Invalid mutation batches are atomic and return the exact Parent portfolio.
- Formal Train/Dev acceptance remains a separate follow-up; this plan provides the executable and Agent-proposal boundary only.

---

### Task 1: Canonical Multi-Parent Policy Contract

**Files:**
- Modify: `numerical_agent/evolution/portfolio.py`
- Test: `tests/test_evolution_portfolio.py`

**Interfaces:**
- Consumes: existing `TSFMPolicy`, `PolicyPortfolio`, `render_policy_source()`, and `parse_policy_source()`.
- Produces: canonical `CombinedPolicy(name, parents, operator, weights, signal, threshold, above_parent, below_parent, fallback_parent)` and legacy-payload migration.

- [ ] **Step 1: Write failing canonical-policy tests**

Add tests that construct and round-trip:

```python
CombinedPolicy(
    name="combined_tsfm_median",
    parents=("toto_2_0", "timesfm_2_5", "chronos_bolt"),
    operator="median",
    weights=(),
    signal="periodicity_strength",
    threshold=0.0,
    above_parent="",
    below_parent="",
    fallback_parent="toto_2_0",
)
```

Assert rejection of duplicate parents, fewer than two or more than five parents,
unknown operators, nonempty median weights, route policies with other than two
parents, route branches outside `parents`, invalid fallback, negative/non-finite
weights, and weights whose sum differs from one.

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py -k 'multi_parent or canonical_combined or invalid_combined'
```

Expected: failures because the new fields/operators do not exist.

- [ ] **Step 3: Implement the canonical dataclass and validation**

In `portfolio.py`:

```python
CombinedOperator = Literal["weighted_mean", "median", "trimmed_mean", "route"]

@dataclass(frozen=True)
class CombinedPolicy:
    name: str
    parents: tuple[str, ...]
    operator: CombinedOperator
    weights: tuple[float, ...] = ()
    signal: str = "periodicity_strength"
    threshold: float = 0.0
    above_parent: str = ""
    below_parent: str = ""
    fallback_parent: str = ""
```

Normalize no values in `__post_init__`; reject noncanonical input so equality,
rendering, and hashing remain deterministic.

- [ ] **Step 4: Add legacy source migration tests**

Create a literal source containing the old fields:

```python
{
    "name": "combined_timesfm_seasonal",
    "tsfm_parent": "timesfm_2_5",
    "statistical_parent": "seasonal_naive",
    "mode": "blend",
    "weight": 0.65,
    "signal": "periodicity_strength",
    "threshold": 0.45,
    "tsfm_when": "above",
}
```

Assert it parses to `parents=("timesfm_2_5", "seasonal_naive")`, operator
`weighted_mean`, weights `(0.65, 0.35)`, and fallback `timesfm_2_5`. Add the
equivalent `route` migration test and assert re-rendered source contains only the
new schema.

- [ ] **Step 5: Verify legacy tests fail, then implement migration**

Add `_combined_from_payload()` that accepts either the exact legacy keys or the
exact canonical keys, never a mixture. Keep `_exact_payload()` for TSFM records.
Reject unknown fields before constructing a policy.

- [ ] **Step 6: Run focused tests and commit**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_portfolio.py
git add numerical_agent/evolution/portfolio.py tests/test_evolution_portfolio.py
git commit -m 'refactor(numerical): generalize combined policy'
```

Expected: all portfolio tests pass.

---

### Task 2: Variable Portfolio and Atomic Mutations

**Files:**
- Modify: `numerical_agent/evolution/portfolio.py`
- Test: `tests/test_evolution_portfolio.py`

**Interfaces:**
- Consumes: canonical `CombinedPolicy` from Task 1 and parsed `MethodModule.names()`.
- Produces: `validate_parents()`, `add_combined()`, `remove_combined()`, `fork_combined()`, and parent-changing Combined repair through `replace()`.

- [ ] **Step 1: Write failing portfolio-shape tests**

Assert:

- one to 32 uniquely named Combined policies are accepted;
- zero and 33 policies are rejected;
- fixed TSFM identity/order is still rejected when changed;
- a TSFM--TSFM policy validates;
- a TSFM--Statistical policy validates;
- a three-parent cross-family policy validates;
- an all-Statistical policy, unknown parent, and Combined parent are rejected by
  `validate_parents(method_names)`.

- [ ] **Step 2: Write failing atomic mutation tests**

Use an immutable Parent and assert:

```python
child = parent.add_combined(new_policy)
repaired = child.replace(new_policy.name, changed_parent_policy)
forked = repaired.fork_combined(new_policy.name, fork_policy)
removed = forked.remove_combined(fork_policy.name)
```

Invalid add/replace/fork/remove calls must raise `PolicyError` and leave
`render_policy_source(parent)` byte-identical. Removing the final Combined policy
must fail.

- [ ] **Step 3: Run mutation tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py -k 'portfolio_shape or atomic or add_combined or fork_combined'
```

- [ ] **Step 4: Implement variable validation and functional mutations**

Remove `FLAGSHIP_COMBINED_NAMES` as an identity gate. Preserve the five initial
policies in `flagship5()`, but validate only tuple bounds, uniqueness, and parent
structure. Return fresh `PolicyPortfolio` values from every mutation method; never
mutate a tuple or policy in place.

Rename `validate_statistical_parents()` to:

```python
def validate_parents(self, method_names: Sequence[str]) -> None:
    ...
```

Derive TSFM names from `self.tsfm`; treat `method_names` as Statistical leaves;
reject parent references to names in `self.combined`.

- [ ] **Step 5: Update call sites and run focused tests**

Update `evaluate_portfolio()` to call `validate_parents(module.names())`.

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py tests/test_evolution_filtering.py
```

- [ ] **Step 6: Commit**

```bash
git add numerical_agent/evolution/portfolio.py \
  tests/test_evolution_portfolio.py tests/test_evolution_filtering.py
git commit -m 'feat(numerical): add combined mutations'
```

---

### Task 3: Multi-Parent Operators and Explicit Fallback

**Files:**
- Modify: `numerical_agent/evolution/portfolio.py`
- Test: `tests/test_evolution_portfolio.py`

**Interfaces:**
- Consumes: materialized `Outcome` rows keyed by `(method, task_id)`.
- Produces: `_combine_forecasts(policy, parent_outcomes, task)` and generalized `_run_combined()`.

- [ ] **Step 1: Write failing execution tests**

Use fixed forecasts so expected values are exact:

- TSFM--TSFM weighted mean of `(10, 20)` with `(0.25, 0.75)` equals `17.5`;
- three-parent median of `(10, 20, 100)` equals `20`;
- five-parent trimmed mean of `(0, 10, 20, 30, 100)` equals `20`;
- TSFM--Statistical weighted mean executes;
- a three-parent TSFM--TSFM--Statistical mean executes;
- route selects the explicit above/below parent from a reviewed history signal;
- failed non-fallback parent returns the successful fallback and records
  `fallback=<name>` in `Outcome.detail`;
- failed fallback does not fabricate a forecast;
- each TSFM runtime is called once per task even when multiple Combined policies
  consume its outcome.

- [ ] **Step 2: Run execution tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py -k 'weighted or median or trimmed or route or fallback or runtime_once'
```

- [ ] **Step 3: Implement reviewed operator execution**

Resolve all parent `Outcome` objects from the materialized map. Apply operators
pointwise with `zip(..., strict=True)`. Use `statistics.median()` for median and
sort each horizon's parent values for trimmed mean. Reuse `_scored()` as the final
finite/horizon validator.

Failure precedence is `CRASHED`, then `INVALID`, then `NOT_APPLICABLE`. Fallback
produces a successful scored Combined outcome only when the fallback parent is
successful and structurally valid; its degraded use is recorded in `detail`.

- [ ] **Step 4: Reapply the existing TSFM invalid-output refactor**

Bring the main worktree's pre-existing, task-related refactor into this isolated
branch without touching unrelated dirty files:

- add `PolicyNotApplicable` and `InvalidTSFMForecastError`;
- extract label-free `forecast_tsfm()`;
- preserve `INVALID` rather than converting a wrong-length TSFM output to crash.

Add the existing regression asserting a wrong-length TSFM result is `INVALID`.

- [ ] **Step 5: Run focused tests and commit**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_portfolio.py
git add numerical_agent/evolution/portfolio.py tests/test_evolution_portfolio.py
git commit -m 'feat(numerical): execute combined graphs'
```

---

### Task 4: Agent-Ready Structured Combined Operations

**Files:**
- Create: `numerical_agent/evolution/combined_evolution.py`
- Test: `tests/test_evolution_combined_evolution.py`

**Interfaces:**
- Consumes: `PolicyPortfolio`, `CombinedPolicy`, Statistical method names, a strict JSON response, and an optional LLM client.
- Produces: `parse_combined_operations()`, `apply_combined_operations()`, `propose_combined_child()`, and `COMBINED_EVOLUTION_SYSTEM`.

- [ ] **Step 1: Write failing strict-schema tests**

Define accepted JSON:

```json
{
  "operations": [
    {
      "op": "add",
      "reason": "TSFM disagreement is reduced by a robust center",
      "policy": {
        "name": "combined_three_tsfm_median",
        "parents": ["toto_2_0", "timesfm_2_5", "chronos_bolt"],
        "operator": "median",
        "weights": [],
        "signal": "periodicity_strength",
        "threshold": 0.0,
        "above_parent": "",
        "below_parent": "",
        "fallback_parent": "toto_2_0"
      }
    }
  ]
}
```

Test `add`, `repair`, `fork`, and `remove`. Reject malformed JSON, duplicate
targets, unknown operations, unknown/extra fields, more than eight operations,
TSFM checkpoint fields, scorer/split fields, and policies that violate Task 1.

- [ ] **Step 2: Run schema tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement parser and atomic apply**

Use `common.llm.parse_json_object()` for one-object parsing, exact key sets for
every operation, and immutable `PolicyPortfolio` methods from Task 2. Apply to a
local candidate and call `validate_parents()` only after the full batch. On any
error, raise `CombinedEvolutionError`; the caller retains the exact Parent object.

- [ ] **Step 4: Write failing LLM boundary tests**

Use `FakeLLMClient` to assert `propose_combined_child()`:

- sends only canonical policy payloads, reviewed Statistical names, bounded
  diagnostics, and allowed operations;
- does not send task future values, documents, GT evidence, roles, runtime
  secrets, checkpoint substitutions, or Dev/Public identifiers;
- returns a changed Child for a valid operation batch;
- returns the exact Parent plus a rejection reason for schema or policy failure.

- [ ] **Step 5: Implement the bounded proposal adapter**

Signature:

```python
def propose_combined_child(
    parent: PolicyPortfolio,
    *,
    statistical_names: Sequence[str],
    diagnostics: Mapping[str, object],
    agent: LLMClient,
) -> CombinedProposalResult:
    ...
```

`CombinedProposalResult` contains `parent`, `child`, canonical operations,
`changed`, and a sanitized `rejection_reason`. It does not score or accept the
Child; the formal Train/Dev controller owns that later decision.

- [ ] **Step 6: Run focused tests and commit**

```bash
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py
git add numerical_agent/evolution/combined_evolution.py \
  tests/test_evolution_combined_evolution.py
git commit -m 'feat(numerical): add combined proposal adapter'
```

---

### Task 5: Compatibility, Documentation, and Full Verification

**Files:**
- Modify: `numerical_agent/README.md`
- Modify: `README.md`
- Modify: `tests/test_evolution_portfolio.py`
- Modify: `tests/test_evolution_combined_evolution.py`

**Interfaces:**
- Consumes: all Tasks 1--4 interfaces.
- Produces: documented no-training workflow and verified backward compatibility.

- [ ] **Step 1: Add the migration and usage documentation**

Document:

```text
history-only leaves
-> Agent proposes typed Combined operations
-> Python validates and materializes Child
-> Train/Dev controller evaluates Child
-> accepted policy source becomes next Git generation
```

State explicitly that this implementation does not train weights and does not yet
wire the proposal adapter into the formal 80/20 command.

- [ ] **Step 2: Add final compatibility assertions**

Assert the legacy five-policy factory retains all old names and forecasts after
canonical migration. Replace hard-coded `103` candidate assertions with a value
derived from `93 + len(portfolio.names)` so future accepted Combined additions do
not require weakening tests.

- [ ] **Step 3: Run focused Numerical verification**

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_filtering.py \
  tests/test_evolution_selector_evolution.py \
  tests/test_evolution_numerical_selector.py
```

- [ ] **Step 4: Run the full repository suite and static checks**

```bash
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m compileall -q numerical_agent tests
git diff --check
```

Expected: all tests pass; no syntax or whitespace errors.

- [ ] **Step 5: Review the final diff against the spec**

Confirm:

- no weight training or model mutation was added;
- manifest bindings remain immutable;
- no inference path reads future labels or Retrieval artifacts;
- every new production behavior was observed RED before GREEN;
- the original main-worktree dirty files remain unchanged;
- the isolated branch contains only scoped commits.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md numerical_agent/README.md \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_combined_evolution.py
git commit -m 'docs(numerical): explain combined evolution'
```
