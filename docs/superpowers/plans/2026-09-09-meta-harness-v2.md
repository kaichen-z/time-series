# Meta-Harness V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve a complete three-agent Harness Genome with role-isolated controls, coordinated Children, and bounded Train-only cross-generation memory, then run one real small evolve experiment.

**Architecture:** Extend the existing `CoEvolutionEngine` rather than adding a second evolution stack. A new closed contract module parses Meta-Harness V2 proposals and memory; the engine owns scheduling/evaluation, while the CLI enables the mode explicitly. The existing Host label firewall, sMAE/sRMSE Pareto comparison, successive halving, Dev gate, and exact Parent rollback remain authoritative.

**Tech Stack:** Python 3.11 dataclasses, existing `LLMClient`/Codex CLI adapter, pytest, JSON checkpoints, current `EvolvingForecastHarness`.

**Spec:** `docs/superpowers/specs/2026-09-09-meta-harness-v2-design.md`

## Global Constraints

- Meta-Harness V2 is opt-in; legacy modes remain byte-compatible when the flag is absent.
- Exactly four initial shapes are Coding-only, Retrieval-only, Decision-only, and joint multi-role.
- Joint mutation changes at least two role scopes; workflow cannot satisfy that minimum by itself.
- Only Train aggregates enter memory; Dev and Public/Holdout never enter a proposal prompt.
- Host-owned verification, metrics, sandbox, split, resource ceilings, and rollback are immutable.
- No independent memory Agent or vector store is introduced.

---

### Task 1: Closed Meta-Harness proposal and memory contracts

**Files:**
- Create: `evolving_loop/meta_harness_v2.py`
- Create: `tests/test_meta_harness_v2.py`

**Interfaces:**
- Produces: `MetaChildKind`, `MetaHarnessProposal.from_payload(...)`, `MetaTrainMemoryRecord`, `child_kind_for_slot(...)`, `policy_change_scope(...)`, and `META_HARNESS_V2_PROMPT`.
- Consumes: canonical primitive Parent/Child payloads; it does not import or instantiate `HarnessPolicy`.

- [ ] **Step 1: Write the failing exact-schema and shape tests**

```python
def test_meta_proposal_requires_exact_closed_schema():
    payload = legal_meta_payload(scope=("coding",))
    payload["future_values"] = [999.0]
    with pytest.raises(ValueError, match="exact schema"):
        MetaHarnessProposal.from_payload(payload)


def test_four_initial_slots_have_ablation_and_joint_shapes():
    assert tuple(child_kind_for_slot(i) for i in range(4)) == (
        "coding", "retrieval", "decision", "joint"
    )
```

- [ ] **Step 2: Run the focused tests and observe import/contract failures**

Run: `pytest -q tests/test_meta_harness_v2.py`

Expected: FAIL because `evolving_loop.meta_harness_v2` does not exist.

- [ ] **Step 3: Implement strict dataclasses and canonical serialization**

Implement an exact top-level key set, exact primitive types, non-empty prompts/hypothesis/changelog, existing numeric ceilings, closed aggregation/workflow values, normalized unique scopes, and finite non-negative Train metrics. `MetaTrainMemoryRecord.to_payload()` must contain no task/document/forecast/Dev/Public fields.

- [ ] **Step 4: Add scope-diff and hostile-payload tests**

Test missing fields, duplicate/unknown scopes, booleans in integer fields, unknown workflow stages, non-finite metrics, task identities, and the rule that a joint slot needs two changed roles.

- [ ] **Step 5: Run Task 1 tests**

Run: `pytest -q tests/test_meta_harness_v2.py`

Expected: all Task 1 tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add evolving_loop/meta_harness_v2.py tests/test_meta_harness_v2.py
git commit -m "feat(evolution): add meta-harness contract"
```

### Task 2: Integrate ablations and Train-only memory into CoEvolutionEngine

**Files:**
- Modify: `evolving_loop/co_evolution.py`
- Modify: `tests/test_co_evolution.py`
- Modify: `tests/test_meta_harness_v2.py`

**Interfaces:**
- Consumes: Task 1 proposal/memory types.
- Produces: `CoEvolutionConfig.meta_harness_v2`, `meta_memory_window`, four-slot proposal scheduling, checkpointed canonical memory, and unchanged `(HarnessPolicy, tuple[EvolutionStep, ...])` output.

- [ ] **Step 1: Write failing mutation-shape tests**

```python
def test_meta_v2_requests_three_role_ablations_and_one_joint_child():
    engine = CoEvolutionEngine(fake_llm, factory, CoEvolutionConfig(
        generations=1, children_per_generation=4, meta_harness_v2=True
    ))
    children = [engine.mutate(parent, evaluation, child_index=i) for i in range(4)]
    assert [meta_scope(child) for child in children] == [
        ("coding",), ("retrieval",), ("decision",),
        ("coding", "retrieval"),
    ]
```

Add separate tests proving a coding slot cannot alter Decision, a joint slot
cannot change only workflow, declared scope must equal actual scope, and malformed
V2 output returns an unchanged Child without weakening legacy parsing.

- [ ] **Step 2: Run focused RED tests**

Run: `pytest -q tests/test_meta_harness_v2.py tests/test_co_evolution.py -k 'meta_v2 or train_memory'`

Expected: FAIL because the engine has no V2 configuration or scheduling.

- [ ] **Step 3: Add opt-in engine configuration and strict proposal path**

When `meta_harness_v2` is true, validate `mode == "genome"`, `target == "auto"`,
`children_per_generation >= 4`, and positive `meta_memory_window`. Use
`META_HARNESS_V2_PROMPT`; include `requested_child_kind` and `train_evolution_memory`
in the user payload; parse with `MetaHarnessProposal`; build a normal
`HarnessPolicy`; recompute actual scopes; reject scope/slot mismatches.

- [ ] **Step 4: Write failing Train-memory and Dev-isolation tests**

Capture every LLM prompt across two generations. Assert generation two contains
only canonical records derived from generation-one Train evaluations and that
distinctive Dev values, task IDs, future values, document text, and failure traces
never occur in the prompt.

- [ ] **Step 5: Implement memory recording and checkpoint round trip**

Record every valid/invalid Child after its Train result is known. Persist
`meta_harness_v2`, `meta_memory_window`, and canonical `meta_memory` in checkpoint
schema 2. Accept schema-1 checkpoints only when V2 is disabled. On resume,
strictly parse every memory record before the next LLM call.

- [ ] **Step 6: Test ordinary and successive-halving paths**

Run:

```bash
pytest -q tests/test_meta_harness_v2.py tests/test_co_evolution.py
```

Expected: all tests PASS, including legacy checkpoint and mutation tests.

- [ ] **Step 7: Commit Task 2**

```bash
git add evolving_loop/co_evolution.py tests/test_co_evolution.py tests/test_meta_harness_v2.py
git commit -m "feat(evolution): evolve complete harness genome"
```

### Task 3: CLI, reports, and deterministic end-to-end smoke

**Files:**
- Modify: `evolving_loop/cli.py`
- Modify: `tests/test_evolving_cli.py`
- Create: `scripts/run_meta_harness_v2.sh`
- Modify: `docs/EVOLVING_AGENT.md`
- Modify: `docs/unified-coevolution-design-2026-09-07.html`
- Modify: `tests/test_unified_coevolution_design_html.py`

**Interfaces:**
- Consumes: `CoEvolutionConfig(meta_harness_v2=True)`.
- Produces: `--meta-harness-v2`, a reproducible wrapper, trace metadata naming the V2 contract and memory policy, and an updated architecture page.

- [ ] **Step 1: Write failing CLI validation tests**

```python
def test_meta_harness_v2_cli_requires_four_genome_children():
    args = build_parser().parse_args([
        "evolve", "--meta-harness-v2", "--evolution-mode", "prompt",
        "--children", "3"
    ])
    with pytest.raises(ValueError, match="Meta-Harness V2"):
        evolve_command(args)
```

Also assert the flag is false by default and legacy commands construct the same
configuration as before.

- [ ] **Step 2: Run CLI RED tests**

Run: `pytest -q tests/test_evolving_cli.py -k meta_harness_v2`

Expected: FAIL because the parser has no flag.

- [ ] **Step 3: Wire the flag and result surface**

Add `--meta-harness-v2`; validate it before task/model creation; pass it to
`CoEvolutionConfig`; include `meta_harness_v2`, `child_shapes`, and
`memory_policy: train_only_last_3` in the command result and trace header.

- [ ] **Step 4: Add wrapper and deterministic fake smoke**

The wrapper invokes one generation, four Children, successive halving, and
explicit output/checkpoint/progress paths. A test replaces the LLM and harness
factory with deterministic fakes, proves all four Children execute, and verifies
the accepted policy or exact Parent rollback.

- [ ] **Step 5: Update documentation to make Meta-Harness primary**

Replace the “one coordinate per global step” statement with: the full Harness
Genome is the Parent, coordinate-isolated Children are controls, the joint Child
tests interaction, and component parsers remain safety compilers. Mark the current
V2 experiment as three-agent Harness evolution and the Dictionary/package adapter
as not yet connected.

- [ ] **Step 6: Run affected verification**

```bash
pytest -q tests/test_meta_harness_v2.py tests/test_co_evolution.py \
  tests/test_evolving_cli.py tests/test_unified_coevolution_design_html.py
bash -n scripts/run_meta_harness_v2.sh
python -m compileall -q evolving_loop
git diff --check
```

- [ ] **Step 7: Commit Task 3**

```bash
git add evolving_loop/cli.py tests/test_evolving_cli.py \
  scripts/run_meta_harness_v2.sh docs/EVOLVING_AGENT.md \
  docs/unified-coevolution-design-2026-09-07.html \
  tests/test_unified_coevolution_design_html.py
git commit -m "feat(evolution): expose meta-harness v2"
```

### Task 4: Run and report one real small evolution

**Files:**
- Create: `runs/meta_harness_v2/<timestamp>/run_command.txt`
- Create: `runs/meta_harness_v2/<timestamp>/result.json`
- Create: `runs/meta_harness_v2/<timestamp>/REPORT.md`

**Interfaces:**
- Consumes: local Dr-CiK public-development tasks, Codex CLI `gpt-5.6-sol`, medium reasoning, one generation, four Children, and the V2 wrapper.
- Produces: an explicit Parent/Child Train and Dev comparison, accepted version, change scopes, interaction hypothesis, call/runtime status, and no Public/Holdout score.

- [ ] **Step 1: Run a no-model preflight**

Verify task source readability, exact requested count, model command availability,
output-directory freshness, and that the run does not select hidden/Public tasks.

- [ ] **Step 2: Run one 8/2 real evolve experiment**

```bash
scripts/run_meta_harness_v2.sh \
  /Users/yyoraa/time-series/external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  runs/meta_harness_v2/real_8_2_gpt56sol
```

The wrapper uses `--limit 10 --dev-fraction 0.2 --holdout-fraction 0`, one
generation, four Children, `--successive-halving --screen-train-tasks 6
--screen-promote 2`, and distinct policy/trace/checkpoint/progress files.

- [ ] **Step 3: Recompute the report from artifacts**

Read the canonical trace/checkpoint, report Parent and every Child Train sMAE and
sRMSE, state why each Child was pruned or accepted, and verify no Dev value appears
in `train_evolution_memory`. Do not claim improvement when the accepted version is
the Parent.

- [ ] **Step 4: Run final regression verification**

```bash
pytest -q tests/test_meta_harness_v2.py tests/test_co_evolution.py \
  tests/test_evolving_cli.py tests/test_unified_coevolution_design_html.py
python -m compileall -q evolving_loop
git diff --check
```

- [ ] **Step 5: Commit source documentation and keep run artifacts explicit**

Commit tracked source/tests/docs only. If `runs/` is intentionally ignored, leave
the report local and provide its absolute path and hashes rather than forcing it
into Git.
