# Reproducible Three-Agent Co-Evolution Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every evaluated co-evolution policy preserve its validated skills, expose candidate-generation and selection diagnostics, and generate genuinely distinct child genomes.

**Architecture:** Extend the existing `HarnessPolicy` artifact with serialized skill snapshots and hydrate those snapshots in the existing harness factory. Snapshot evaluation-local libraries after train learning and before development evaluation, while keeping the trusted label firewall unchanged. Add aggregate candidate diagnostics to existing resolved outcomes and evolution traces, then expose a repeatable 30-task auto-Genome launcher.

**Tech Stack:** Python 3, frozen dataclasses, JSON artifacts, pytest, Bash, existing Codex CLI and sandbox.

**Spec:** `docs/superpowers/specs/2026-08-15-reproducible-coevolution-bundle-design.md`

## Global Constraints

- Keep Coding `setting=llm_only`; do not inject a statistical dictionary or TSFM candidate.
- Do not expose documents or resolved labels to Coding.
- Do not expose future values, GT evidence, document roles, or subtypes during inference.
- Do not change the trusted scorer, label firewall, entity-disjoint split, sandbox, or holdout gate.
- Do not push or modify the remote `khoutaibi` branch.
- Official sCRPS is outside this implementation because the current harness emits deterministic trajectories.

---

### Task 1: Inheritable Skill Snapshots

**Files:**
- Modify: `evolving_agent/co_evolution.py`
- Modify: `evolving_agent/cli.py`
- Test: `tests/test_co_evolution.py`
- Test: `tests/test_evolving_cli.py`

**Interfaces:**
- Produces: `HarnessPolicy.coding_skills`, `retrieval_skills`, and `decision_skills` as JSON-safe tuples of records.
- Produces: `snapshot_policy_skills(policy: HarnessPolicy, harness: EvolvingForecastHarness) -> HarnessPolicy`.
- Consumes: existing `Skill`, `RetrievalSkill`, `DecisionSkill`, and library `all()` methods.

- [ ] Write a failing round-trip test for a policy containing all three skill record types.
- [ ] Run the test and verify `HarnessPolicy` rejects the new fields.
- [ ] Add JSON-safe skill fields and list-to-tuple loading.
- [ ] Write a failing factory test proving a policy-embedded Coding skill is available when disk libraries are empty.
- [ ] Run the test and verify the skill is absent.
- [ ] Hydrate merged, evaluation-local libraries from base library plus policy records.
- [ ] Write a failing test proving a learned child skill is present in the accepted policy artifact.
- [ ] Snapshot parent and child libraries after train evaluation and use the snapshots for acceptance/checkpoint output.
- [ ] Run the targeted tests.

### Task 2: Candidate and Selection Diagnostics

**Files:**
- Modify: `evolving_agent/metrics.py`
- Modify: `evolving_agent/evaluation.py`
- Modify: `evolving_agent/co_evolution.py`
- Modify: `evolving_agent/cli.py`
- Test: `tests/test_evolving_agent_metrics.py`
- Test: `tests/test_co_evolution.py`

**Interfaces:**
- Produces: `spearman_rank_correlation(left: list[float], right: list[float]) -> float` with deterministic average ranks for ties.
- Extends: `ResolvedOutcome` with `candidate_count` and `hindcast_future_rank_correlation`.
- Extends: `PolicyEvaluation.diagnostics` and `EvolutionStep` diagnostic dictionaries.

- [ ] Write failing tests for perfect, inverse, and tied Spearman rankings.
- [ ] Run them and verify the helper is missing.
- [ ] Implement deterministic average-rank Spearman correlation.
- [ ] Write a failing resolved-outcome test for candidate count and hindcast/future rank correlation.
- [ ] Extend scoring without changing primary reward semantics.
- [ ] Write a failing policy-evaluation test for mean final, Best-of-K, selection-regret, candidate-count, and rank-correlation diagnostics.
- [ ] Aggregate diagnostics and persist them in progress and evolution traces.
- [ ] Add the same summary fields to the non-evolution `run` result.
- [ ] Run targeted tests.

### Task 3: Diverse Complete Children

**Files:**
- Modify: `evolving_agent/co_evolution.py`
- Test: `tests/test_co_evolution.py`

**Interfaces:**
- Changes: `CoEvolutionEngine.mutate(parent, evaluation, *, child_index=0)`.
- Adds: `child_index` and a structural-diversity instruction to the Meta-Harness payload.

- [ ] Write a failing test proving two children receive different `child_index` payloads.
- [ ] Pass the loop index into mutation and require a distinct structural proposal.
- [ ] Exclude full skill source from `current_policy`; include compact skill counts/names instead.
- [ ] Run targeted tests.

### Task 4: Frozen 30-Task Co-Evolution Protocol

**Files:**
- Create: `evolving_agent/scripts/run_coevolution_pilot30.sh`
- Modify: `docs/EVOLVING_AGENT.md`
- Test: `tests/test_evolution_scripts.py`

**Interfaces:**
- Consumes: existing unified `evolve` CLI and `run_llm_only_evolutions.sh` conventions.
- Produces: one-command 30-task `genome + auto` run with configurable paths, seed, generations, children, model, cache, and dry-run mode.

- [ ] Write a failing dry-run test for limit 30, `target=auto`, Genome mode, and two children.
- [ ] Add the executable Bash launcher with environment-variable overrides.
- [ ] Document the bundle contents, diagnostics, and pilot command.
- [ ] Run shell syntax and targeted tests.

### Task 5: Verification and Smoke Run

**Files:**
- Runtime artifacts only under `runs/`.

**Interfaces:**
- Produces: test evidence and a no-label smoke artifact; does not publish or push.

- [ ] Run `python -m pytest -q`.
- [ ] Run `bash -n evolving_agent/scripts/run_coevolution_pilot30.sh`.
- [ ] Run the new launcher in dry-run mode and inspect its rendered command.
- [ ] Run a minimal three-task, one-generation smoke with a shared Codex cache.
- [ ] Verify the saved policy contains reproducible skill snapshots and trace diagnostics.
- [ ] Report remaining limits before any expensive 30-task run.
