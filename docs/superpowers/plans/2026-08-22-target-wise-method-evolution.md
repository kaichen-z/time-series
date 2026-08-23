# Target-Wise Forecasting Method Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cached, target-wise forecasting-method evolution loop in which each selected method creates an independent child and one invalid child cannot reject the others.

**Architecture:** Content-addressed outcome caching avoids rerunning unchanged Parent methods. A selector produces unique target proposals with bounded actions, and the mutator handles one target at a time. Each candidate is screened independently; every survivor is then rebased, revalidated, and committed in sequence. Multi-method merge remains available only in the legacy batch strategy in v1.

**Tech Stack:** Python 3.12+, dataclasses, hashlib, JSON, pathlib, multiprocessing-based method executor, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-08-22-target-wise-method-evolution-design.md`

## Global Constraints

- Keep `methods.py` compatible with existing bootstrap and execution commands.
- Do not expose future values, GT evidence, document roles, or hidden labels to an LLM.
- Cache keys must bind method source, complete task input/label payload, isolation mode, and schema version.
- Invalid or stale cache files are misses and must not abort evolution.
- Same-name repair, fork, deletion, duplicate-target, and action gates remain enforced. Merge is batch-only.
- Train/mini-dev partition is exactly 16/4 for the real smoke run.

---

### Task 1: Content-Addressed Method Outcome Cache

**Files:**
- Create: `numerical_agent/evolution/cache.py`
- Modify: `numerical_agent/evolution/execution.py`
- Test: `tests/test_evolution_cache.py`

**Interfaces:**
- Consumes: `MethodDefinition`, `Task`, `Outcome`, and the existing isolated method executor.
- Produces: `OutcomeCache.evaluate_method(method, tasks, isolated=False) -> tuple[Outcome, ...]` and cache statistics.

- [ ] **Step 1: Write failing tests** for a miss followed by a hit, source changes, task changes, isolation changes, corrupt JSON, and atomic reconstruction of `Outcome`.
- [ ] **Step 2: Run** `runs/method_evolution/.venv/bin/python -m pytest -q tests/test_evolution_cache.py` and confirm failures are caused by the missing cache.
- [ ] **Step 3: Implement** deterministic task serialization, SHA-256 keys, atomic JSON writes, and safe cache reads in `cache.py`.
- [ ] **Step 4: Add** a focused single-method execution helper in `execution.py`; reuse existing timeout and subprocess containment.
- [ ] **Step 5: Run** cache and execution tests and confirm they pass.

### Task 2: Unique Target Proposals and Allowed Action Sets

**Files:**
- Modify: `numerical_agent/evolution/__init__.py`
- Modify: `numerical_agent/evolution/prompts.py`
- Test: `tests/test_evolution_loop.py`

**Interfaces:**
- Consumes: selector JSON and identity-contract payloads.
- Produces: `TargetProposal(name, allowed_actions, reason)` and a one-target mutator request.

- [ ] **Step 1: Write failing tests** showing that verified repair targets allow `repair` or `fork`, fork-only methods allow only `fork`, and duplicate targets fail before mutation.
- [ ] **Step 2: Run** the focused tests and confirm the new proposal API is missing.
- [ ] **Step 3: Implement** `TargetProposal`, parse at most three unique targets, and derive allowed actions without permitting deletion escalation.
- [ ] **Step 4: Update** selector and mutator prompts so one request contains exactly one target and the response contains zero or one operation.
- [ ] **Step 5: Run** focused prompt/parser/identity tests.

### Task 3: Independent Candidate Evaluation

**Files:**
- Create: `numerical_agent/evolution/targetwise.py`
- Modify: `numerical_agent/evolution/__init__.py`
- Test: `tests/test_targetwise_evolution.py`

**Interfaces:**
- Consumes: `OutcomeCache`, `TargetProposal`, Parent `MethodModule`, selector/mutator clients, train and validation tasks.
- Produces: `CandidateResult` and `evolve_targets_once(...) -> TargetWiseGeneration`.

- [ ] **Step 1: Write failing tests** where two targets create two candidates, the first is invalid, and the second remains independently eligible.
- [ ] **Step 2: Run** the target-wise tests and confirm they fail because the controller is absent.
- [ ] **Step 3: Implement** one-target mutation, operation validation, in-memory child application, and changed-method discovery.
- [ ] **Step 4: Implement** deterministic four-task screening and full-train promotion using cached unchanged outcomes.
- [ ] **Step 5: Implement** mini-dev mean/median MASE gates and per-operation rejection reasons.
- [ ] **Step 6: Run** target-wise, identity, module, execution, and cache tests.

### Task 4: CLI, Tracing, and Compatibility

**Files:**
- Modify: `numerical_agent/run_evolution.py`
- Modify: `scripts/run_method_evolution.sh`
- Test: `tests/test_evolution_loop.py`
- Test: `tests/test_method_evolution_script.py`

**Interfaces:**
- Consumes: existing evolution CLI arguments.
- Produces: `--evolution-strategy batch|targetwise`, `--outcome-cache-dir`, `--max-targets`, and `--screen-tasks`.

- [ ] **Step 1: Write failing CLI and shell dry-run tests** for the four new options and their defaults.
- [ ] **Step 2: Run** the focused tests and confirm the arguments are missing.
- [ ] **Step 3: Add** CLI dispatch while retaining `batch` as a compatibility option and making `targetwise` opt-in for the first real run.
- [ ] **Step 4: Trace** cache hits/misses, each target proposal, candidate rejection, validation metrics, promotion, and elapsed seconds.
- [ ] **Step 5: Run** CLI, script, and tracing tests.

### Task 5: Verification and Real 20-Task Experiment

**Files:**
- Modify only if verification reveals a scoped defect.
- Artifacts: `runs/method_evolution/<targetwise-run>/`

**Interfaces:**
- Consumes: the 16/4 public Dr-CiK split and Parent commit `d2ffec3`.
- Produces: a complete trace, cache statistics, candidate results, elapsed time, and accepted Parent/Child commit evidence.

- [ ] **Step 1: Run** focused tests for cache, identity, execution, module operations, loop, CLI, and shell script.
- [ ] **Step 2: Run** compileall, `bash -n scripts/run_method_evolution.sh`, and `git diff --check`.
- [ ] **Step 3: Run** the real target-wise generation with Luna selector, Terra mutator, 16 train tasks, 4 mini-dev tasks, three targets, and two children per target only if the implemented interface supports independent sampling safely.
- [ ] **Step 4: Inspect** every generated operation, verify Parent preservation for forks/rejections, and report cache-hit rate and elapsed time.
- [ ] **Step 5: Run** the full repository test suite and separate pre-existing failures from regressions introduced by this plan.
