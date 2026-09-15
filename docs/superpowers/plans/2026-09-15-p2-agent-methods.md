# P2 AlphaEvolve/Voyager/PromptBreeder Integration Plan

**Goal:** Add bounded program context, deterministic curriculum/reuse, and
archive-backed prompt evolution to P2 while preserving all Host boundaries.

**Spec:** `docs/superpowers/specs/2026-09-15-p2-agent-methods-design.md`

## Global Constraints

- P2 produces the complete executable Dictionary; P3 alone selects per task.
- Metrics, verifier, splits, artifact validation, runtime identity, budget, and
  promotion authority remain fixed.
- Only Host-verified source bytes and sanitized Train feedback may influence
  curriculum, reusable-method retrieval, or prompt credit.
- Provider inputs are exact-schema, bounded, canonical, deterministic, and
  contain no Dev/Public values, raw forecasts, raw objectives, paths, or live
  objects.
- Existing sealed runs remain readable.

### Task 1: Strategy contracts and pure selection

**Files:**
- Create: `evolving_loop/v2/numerical_qd/agent_methods.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`
- Test: `tests/test_evolution_v2_numerical_agent_methods.py`

Implement immutable exact contracts for curriculum targets, verified reusable
program records, mutation-prompt lineage, and bounded prompt population state.
Implement deterministic pure functions that derive unoccupied/least-visited/
failure-matched targets, select feasible reusable methods, select prompt
lineages, insert prompt children, and apply Host-only Train credit. Use SHA
tie-breaking and bounded capacities. Write and observe focused RED tests first.

### Task 2: AlphaEvolve/Voyager proposer boundary

**Files:**
- Modify: `evolving_loop/v2/numerical_qd/proposers.py`
- Modify: `evolving_loop/v2/numerical_qd/map_elites.py`
- Test: `tests/test_evolution_v2_numerical_proposers.py`
- Test: `tests/test_evolution_v2_numerical_map_elites.py`

Extend the exact primitive request with bounded curriculum targets and
SHA-bound verified program records. Validate source identities and exclude
unverified/failed/quarantined methods. Include the context in the LLM wire
payload and keep the existing Host-owned normalization/execution gate.

### Task 3: PromptBreeder and runner integration

**Files:**
- Modify: `evolving_loop/v2/numerical_qd/runner.py`
- Modify only as required: `evolving_loop/v2/numerical_qd/contracts.py`
- Modify only as required: `evolving_loop/v2/numerical_qd/persistence.py`
- Modify only as required: `evolving_loop/v2/numerical_qd/artifacts.py`
- Test: `tests/test_evolution_v2_numerical_runner.py`
- Test: `tests/test_evolution_v2_numerical_persistence.py`

Restore the sampled genome's own task prompt, derive curriculum/reusable
program context, bind the selected mutation prompt, and persist all selected
identities before provider dispatch. Update only the sampled prompt lineage
from Host Train outcomes. Resume must reproduce the same selection and cannot
resample an unfinished generation. Preserve legacy read compatibility.

### Task 4: Regression, smoke, and documentation

Run all numerical-QD, cooperative, real-runner, and P3 Dictionary tests. Add a
small deterministic end-to-end smoke proving P2 program/prompt evolution seals
a complete Dictionary that P3 consumes. Update Numerical QD/cooperative docs to
describe the three mechanisms and their non-paper-reproduction scope.
