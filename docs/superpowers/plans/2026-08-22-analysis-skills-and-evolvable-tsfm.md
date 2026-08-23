# Analysis Skills and Evolvable TSFM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add history-only reusable skills and co-evolve five manifest-bound TSFM invocation policies plus five executable Combined policies beside the 93 Python methods.

**Architecture:** A validated skill module is injected into evolved Python forecasters and hashed into their cache identity. Five typed TSFM policies adapt the existing runtime registry; the LLM may change only policy fields while Python validators preserve the reviewed model binding.

**Tech Stack:** Python 3.12+, stdlib statistics/AST, existing numerical-agent runtime registry, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-analysis-skills-and-evolvable-tsfm-design.md`

## Global Constraints

- Skills are strictly history-only and deterministic.
- TSFM checkpoint, adapter, license, revision, and model identity are immutable.
- Exactly five TSFM invocation policies and five Combined policies are evolvable.
- Enabling 103-candidate mode fails closed unless all five runtimes resolve.
- No model is loaded by unit tests or CLI construction.
- Existing uncommitted user changes are preserved.

---

### Task 1: History-only analysis skills

**Files:**
- Create: `numerical_agent/evolution/analysis_skills.py`
- Create: `numerical_agent/evolution/analysis_skills_template.py`
- Modify: `numerical_agent/evolution/execution.py`
- Modify: `numerical_agent/evolution/cache.py`
- Modify: `numerical_agent/evolution/prompts.py`
- Modify: `numerical_agent/evolution/repository_bootstrap.py`
- Test: `tests/test_evolution_analysis_skills.py`

**Interfaces:** Nine deterministic history-only analysis functions injected into method modules.

- [x] Add failing synthetic-signal, injection, safety, and cache-identity tests.
- [x] Implement the minimal skill API, injection, cache hashing, prompts, and bootstrap seed.
- [x] Run the focused tests to green.

### Task 2: Typed TSFM and Combined policies

**Files:**
- Create: `numerical_agent/evolution/portfolio.py`
- Create: `numerical_agent/evolution/policy_targetwise.py`
- Modify: `numerical_agent/evolution/targetwise.py`
- Modify: `numerical_agent/run_evolution.py`
- Modify: `numerical_agent/evolution/prompts.py`
- Test: `tests/test_evolution_tsfm_policy.py`
- Test: `tests/test_targetwise_evolution.py`

**Interfaces:** Five manifest-bound TSFM policies plus five fixed-lineage Combined policies with
history-only applicability, context, preprocessing, calibration, blending, and routing fields.

- [x] Add failing tests for exact five-model/five-combination construction and 93+5+5 count reporting.
- [x] Add adversarial tests rejecting model identity and Combined-parent substitution.
- [x] Add failing tests proving policy targets can be screened, validated, accepted, and rejected.
- [x] Implement runtime execution, typed policy mutation, and content-addressed TSFM caching.
- [x] Run the focused tests to green.

### Task 3: Runner and documentation

**Files:**
- Modify: `scripts/run_method_evolution.sh`
- Modify: `numerical_agent/README.md`
- Modify: `README.md`
- Add: `runs/method_evolution/v001/skills.py` in the nested evolution repository.
- Test: `tests/test_method_evolution_script.py`

**Interfaces:** `ME_FOUNDATION_PORTFOLIO=flagship5` and the existing TSFM deployment/runtime
settings produce a run that reports `93 Python + 5 TSFM + 5 Combined = 103 candidates`.

- [x] Add failing runner assertions.
- [x] Add CLI propagation plus nested skill and policy seeds.
- [ ] Run focused and full tests, compilation, shell syntax, and diff checks.
