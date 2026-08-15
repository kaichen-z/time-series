# LLM-Only Three-Agent Co-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing LLM-only Coding, Retrieval, and Decision harness evolve through attributable mutations and accept only held-out development improvements.

**Architecture:** Add an explicit evolution target to the existing policy engine, preserving the current label firewall and train/dev elitism. Targeted modes mutate one role at a time; the existing `auto` mode remains the whole-system controller. Harden Source Engineer instructions so a candidate either edits its isolated worktree or fails explicitly.

**Tech Stack:** Python 3, dataclasses, pytest, Codex CLI, existing sandbox and Dr-CiK evaluator.

**Spec:** `docs/superpowers/specs/2026-08-14-llm-only-three-agent-coevolution-design.md`

## Global Constraints

- Do not expose future values, GT evidence, or document role/subtype labels to inference agents.
- Keep `setting=llm_only`; do not inject the statistical method dictionary or a TSFM candidate.
- Do not change the scorer, split, label firewall, sandbox, or resource ceilings.
- Do not push or modify the remote `khoutaibi` branch.
- Accept a child only after strict development reward improvement.

---

### Task 1: Targeted Prompt/Genome Evolution

**Files:**
- Modify: `evolving_agent/co_evolution.py`
- Modify: `evolving_agent/cli.py`
- Test: `tests/test_co_evolution.py`
- Test: `tests/test_unified_cli.py`

**Interfaces:**
- Consumes: `HarnessPolicy`, `PolicyEvaluation`, `CoEvolutionConfig`.
- Produces: `EvolutionTarget = Literal["auto", "coding", "retrieval", "decision"]` and CLI `--evolve-target`.

- [ ] Write a failing test proving `coding` prompt evolution can only replace a Coding prompt even when Retrieval has the lowest module reward.
- [ ] Run the targeted test and verify it fails because the engine currently always selects the weakest module.
- [ ] Add `target: EvolutionTarget = "auto"` to `CoEvolutionConfig` and resolve `auto` through `weakest_agent`.
- [ ] Add `--evolve-target` to the evolution CLI and pass it into `CoEvolutionConfig`.
- [ ] Run targeted tests and verify they pass.
- [ ] Write a failing test proving coding-target Genome proposals preserve Retrieval, Decision, workflow, aggregation, and evidence-adjustment fields.
- [ ] Implement a bounded role-specific Genome proposal filter.
- [ ] Run targeted and full tests.

### Task 2: Source Engineer Must Implement

**Files:**
- Modify: `evolving_agent/source_evolution.py`
- Test: `tests/test_source_evolution.py`

**Interfaces:**
- Consumes: `SOURCE_ENGINEER_PROMPT` and `_run_engineer`.
- Produces: an instruction contract requiring an edit or an explicit failure result without a clarification question.

- [ ] Write a failing assertion that the Source prompt contains direct-execution and no-confirmation requirements.
- [ ] Run the assertion and verify it fails on the current prompt.
- [ ] Add the minimal direct-execution wording and require the final message to report changed files.
- [ ] Run targeted and full tests.

### Task 3: Comparable LLM-Only Experiment Runner

**Files:**
- Modify: `evolving_agent/scripts/run_llm_only_evolutions.sh`
- Test: `tests/test_evolution_scripts.py`
- Modify: `docs/EVOLVING_AGENT.md`

**Interfaces:**
- Consumes: the unified CLI, frozen subset manifest, and `EA_*` environment overrides.
- Produces: repeatable `coding`, `retrieval`, `decision`, and `auto` runs with isolated output directories.

- [ ] Write a dry-run test for `EA_EVOLVE_TARGET=coding` and explicit `PYTHON=python` propagation.
- [ ] Verify the test fails before the wrapper forwards the target.
- [ ] Forward `--evolve-target` and document the four target modes.
- [ ] Run script tests and the full suite.

### Task 4: Smoke and Frozen Pilot

**Files:**
- Runtime artifacts only: `runs/llm_only_targeted_20260814/`

**Interfaces:**
- Consumes: public Dr-CiK tasks and the frozen 30-task stratified subset.
- Produces: progress logs, checkpoints, policies, traces, and a result summary.

- [ ] Run a three-task `prompt + llm_only + coding` smoke with one generation and one child.
- [ ] Inspect train/dev rewards, mutation legality, and label firewall artifacts.
- [ ] If execution fails, reproduce with a focused test and repair through TDD.
- [ ] Run the frozen 30-task `prompt + llm_only + coding` pilot.
- [ ] If no child is accepted, classify the failure as candidate-generation, Coding coverage, Retrieval, or Decision regret; change only the diagnosed component and rerun with the same manifest.
- [ ] Stop only after an accepted held-out improvement or an external blocker that cannot be resolved locally.
- [ ] Run frozen holdout inference once after the policy is selected and report it separately from development.
