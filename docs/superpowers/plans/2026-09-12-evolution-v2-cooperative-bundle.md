# Evolution V2 Cooperative Bundle Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast, reproducible research prototype that co-evolves Numerical, Retrieval, Decision, and joint Bundle Children through the complete three-Agent pipeline using resumable UCB or Thompson scheduling.

**Architecture:** Add a small `evolving_loop.v2.cooperative` package that wraps existing Project 2 Numerical artifacts and legacy Retrieval/Decision runtime types in canonical V2 artifacts. A complete-pipeline adapter evaluates one scheduled Child, the existing Kernel accepts or rejects it, and a compact Project 3 checkpoint owns scheduler resume state. The unified `evolve` CLI dispatches cooperative configs without changing existing fake or Numerical commands.

**Tech Stack:** Python 3.11+, frozen dataclasses, canonical JSON and SHA-256, existing `EvolutionBundleV2`, `EvolutionKernel`, `BudgetLedger`, Project 2 `FrozenNumericalArtifactsV2`, legacy `RetrievalGenome`/`HarnessPolicy`/`PackagePipelineEvaluator`, and pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-evolution-v2-cooperative-bundle-design.md`

## Global Constraints

- This is a research prototype, not a new production-hardening pass.
- Blocking smoke data is exactly 4 Train / 1 Dev; the optional pilot is exactly 8 Train / 2 Dev.
- One run performs at most four scheduled steps and proposes exactly one Child per step.
- Both `ucb` and `thompson` must be implemented under the same scheduler-state contract.
- Numerical-, Retrieval-, Decision-, and joint-Child paths must execute; joint changes at least two principal scopes atomically.
- Every scored Child is evaluated through the complete Numerical -> Retrieval -> Decision pipeline.
- Proposal and scheduler inputs are Train-only aggregates; Dev metrics remain inside Kernel acceptance evidence; Public tasks are rejected.
- Mean capped sMAE and sRMSE may not regress by more than `1e-12`; Dev joint capped error must improve by more than `1e-12`.
- Existing Project 1 canonical serialization, Kernel permits, Bundle ownership, promotion, and budget APIs remain authoritative.
- No per-forecast artifact write or `fsync` is added. Persist one aggregate per candidate/split and one checkpoint per closed step.
- Do not add new TOCTOU, symlink, inode, hostile-filesystem, forged-receipt, or crash-window machinery.
- Do not run 80/20, Public-99, real network clients, the full Numerical runner suite, or the full `test_evolution_v2_*` wildcard as a blocking gate.
- Existing `evolve` fake mode and `numerical-evolve` behavior must remain unchanged.
- Target implementation and blocking verification time is 2.5–3.5 hours.

## File Map

- `evolving_loop/v2/cooperative/contracts.py`: canonical module, candidate, scheduler, checkpoint, and result artifacts.
- `evolving_loop/v2/cooperative/schedulers.py`: pure discounted-UCB and deterministic-Thompson selection/update.
- `evolving_loop/v2/cooperative/adapters.py`: typed Numerical pool, Retrieval, Decision, artifact catalog, and complete-pipeline adapters.
- `evolving_loop/v2/cooperative/proposals.py`: one deterministic typed proposal for each single arm and composable joint proposals.
- `evolving_loop/v2/cooperative/runner.py`: bounded four-step loop, aggregate cache, Kernel transition, checkpoint, and resume.
- `evolving_loop/v2/cooperative/__init__.py`: stable public Project 3 imports.
- `evolving_loop/v2/kernel.py`: backward-compatible Host scheduler-state sealing and joint-Numerical release binding.
- `evolving_loop/v2/cli.py`: cooperative dispatch behind the existing `evolve` command.
- `configs/evolution_v2/cooperative/smoke-{ucb,thompson}.json`: runnable 4/1 profiles.
- `tests/build_evolution_v2_cooperative_fixture.py`: deterministic canonical 4/1 seed inputs.
- `tests/test_evolution_v2_cooperative_{contracts,schedulers,adapters,proposals,pipeline,runner,cli,safety}.py`: focused prototype tests.
- `docs/evolution-v2-cooperative-bundle.md`: command, artifacts, resume, and optional pilot/80-20 experiment notes.
- `README.md`: one short Project 3 CLI link.

---

### Task 1: Canonical Artifacts and Both Schedulers

**Estimated time:** 25 minutes

**Files:**
- Create: `evolving_loop/v2/cooperative/__init__.py`
- Create: `evolving_loop/v2/cooperative/contracts.py`
- Create: `evolving_loop/v2/cooperative/schedulers.py`
- Create: `tests/test_evolution_v2_cooperative_contracts.py`
- Create: `tests/test_evolution_v2_cooperative_schedulers.py`

**Interfaces:**
- Consumes: `canonical_v2_bytes(payload)`, `fingerprint_payload(payload)`, `require_sha256(value, field)`, and `MutationTarget`.
- Produces: `RetrievalModuleV2`, `DecisionModuleV2`, `BundleCandidateV2`, `SchedulerArmStateV2`, `CooperativeSchedulerStateV2`, `CooperativeCheckpointV2`, and `CooperativeRunResultV2`.
- Produces: `select_arm(state) -> str` and `record_outcome(state, arm, *, train_reward, normalized_cost, accepted) -> CooperativeSchedulerStateV2`.

- [ ] **Step 1: Write failing canonical artifact tests**

```python
def test_joint_candidate_requires_two_changed_scopes(parent_bundle):
    candidate = BundleCandidateV2(
        schema_version=1,
        target="joint",
        parent_bundle_sha256=parent_bundle.fingerprint(),
        operator="paired_typed_mutation",
        numerical_release_sha256=None,
        numerical_registry_sha256=None,
        retrieval_release_sha256="a" * 64,
        decision_policy_sha256=None,
    )
    with pytest.raises(ValueError, match="at least two"):
        candidate.to_child(parent_bundle)


def test_scheduler_state_rejects_dev_or_task_feedback():
    payload = scheduler_state().to_payload()
    payload["dev_metrics"] = {"mean_smae": 0.1}
    with pytest.raises(ValueError, match="exact schema"):
        CooperativeSchedulerStateV2.from_payload(payload)
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_contracts.py`

Expected: collection fails because `evolving_loop.v2.cooperative` does not exist.

- [ ] **Step 3: Implement the exact canonical contracts**

Use frozen slotted dataclasses. Every class exposes `from_payload`,
`to_payload`, `canonical_bytes`, and `fingerprint`. Implement
`BundleCandidateV2.to_child` exactly as:

```python
def to_child(self, parent: EvolutionBundleV2) -> EvolutionBundleV2:
    if self.parent_bundle_sha256 != parent.fingerprint():
        raise ValueError("candidate Parent mismatch")
    changes: dict[str, object] = {}
    if self.numerical_release_sha256 is not None:
        if self.numerical_registry_sha256 is None:
            raise ValueError("Numerical candidate requires release and registry")
        changes["numerical"] = (
            self.numerical_release_sha256,
            self.numerical_registry_sha256,
        )
    elif self.numerical_registry_sha256 is not None:
        raise ValueError("Numerical candidate requires release and registry")
    if self.retrieval_release_sha256 is not None:
        changes["retrieval"] = self.retrieval_release_sha256
    if self.decision_policy_sha256 is not None:
        changes["decision"] = self.decision_policy_sha256
    expected = 2 if self.target == "joint" else 1
    if (self.target == "joint" and len(changes) < expected) or (
        self.target != "joint" and tuple(changes) != (self.target,)
    ):
        raise ValueError("candidate scope does not match target; joint needs at least two")
    return parent.provisional_child(self.target, changes)
```

`CooperativeSchedulerStateV2` fields are exactly `schema_version`, `mode`,
`seed`, `draw_counter`, `completed_step`, `discount`, `arms`. Arm keys must be
the enabled subset of `numerical`, `retrieval`, `decision`, `joint` in canonical
order and must not contain evaluator-only names.

- [ ] **Step 4: Write failing UCB and Thompson behavior tests**

```python
def test_ucb_visits_untried_arms_in_canonical_order():
    state = state_for("ucb", arms=("numerical", "retrieval", "decision", "joint"))
    selected = []
    for _ in range(4):
        arm = select_arm(state)
        selected.append(arm)
        state = record_outcome(
            state, arm, train_reward=0.0, normalized_cost=0.25, accepted=False
        )
    assert selected == ["numerical", "retrieval", "decision", "joint"]


def test_thompson_resume_repeats_the_same_next_draw():
    state = exercised_state("thompson", seed=17)
    restored = CooperativeSchedulerStateV2.from_payload(state.to_payload())
    assert select_arm(restored) == select_arm(state)
```

- [ ] **Step 5: Run scheduler tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_schedulers.py`

Expected: tests fail because `select_arm` and `record_outcome` are absent.

- [ ] **Step 6: Implement pure scheduler selection and update**

For UCB use:

```python
score = (
    arm.discounted_reward_sum / arm.attempts
    - arm.discounted_cost_sum / arm.attempts
    + math.sqrt(2.0 * math.log(total_attempts + 1.0) / arm.attempts)
)
```

For Thompson seed each arm draw with:

```python
draw_seed = int.from_bytes(
    hashlib.sha256(f"{state.seed}:{state.draw_counter}:{name}".encode()).digest()[:8],
    "big",
)
sample = random.Random(draw_seed).betavariate(
    1 + arm.acceptances,
    1 + arm.attempts - arm.acceptances,
)
score = sample - arm.discounted_cost_sum / arm.attempts
```

Break ties by canonical arm order. `record_outcome` multiplies all existing
reward/cost sums by `discount`, updates the selected arm, increments
`completed_step`, and increments `draw_counter` only for Thompson.

- [ ] **Step 7: Run Task 1 tests**

Run: `pytest -q tests/test_evolution_v2_cooperative_contracts.py tests/test_evolution_v2_cooperative_schedulers.py`

Expected: all pass in under 10 seconds.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/v2/cooperative tests/test_evolution_v2_cooperative_contracts.py tests/test_evolution_v2_cooperative_schedulers.py
git commit -m "feat(cooperative): add artifacts and schedulers"
```

---

### Task 2: Numerical, Retrieval, and Decision Adapters

**Estimated time:** 35 minutes

**Files:**
- Create: `evolving_loop/v2/cooperative/adapters.py`
- Create: `tests/test_evolution_v2_cooperative_adapters.py`
- Modify: `evolving_loop/v2/cooperative/__init__.py`

**Interfaces:**
- Consumes: Task 1 module artifacts; `FrozenNumericalArtifactsV2`; `RetrievalGenome`; `RetrievalSkillLibrary`; `HarnessPolicy`; `TwoStageRetrievalAgent`; `DecisionAgent`.
- Produces: `CooperativeArtifactCatalog.add_*`, `resolve_numerical`, `resolve_retrieval`, and `resolve_decision`.
- Produces: `NumericalCoordinateAdapter.propose(parent)`, `RetrievalCoordinateAdapter.propose(parent, step)`, and `DecisionCoordinateAdapter.propose(parent, step)`.

- [ ] **Step 1: Write failing real-type adapter tests**

```python
def test_numerical_adapter_returns_a_different_frozen_pair(seed_pair, alternate_pair):
    adapter = NumericalCoordinateAdapter((alternate_pair,))
    assert adapter.propose(seed_pair).release.fingerprint == alternate_pair.release.fingerprint


def test_retrieval_adapter_changes_one_typed_field(retrieval_module):
    child = RetrievalCoordinateAdapter().propose(retrieval_module, step=0)
    parent_genome = retrieval_module.genome
    child_genome = child.genome
    changed = [
        name
        for name in child_genome.to_payload()
        if child_genome.to_payload()[name] != parent_genome.to_payload()[name]
    ]
    assert changed == ["round1_strategy"]
    assert child_genome.require_counterevidence_search is True


def test_decision_adapter_changes_only_the_prompt(decision_module):
    child = DecisionCoordinateAdapter(("prompt variant",)).propose(decision_module, step=0)
    assert child.prompt == "prompt variant"
    assert child.skills == decision_module.skills
    assert child.enable_evidence_adjustments == decision_module.enable_evidence_adjustments
    assert child.max_evidence_adjustments == decision_module.max_evidence_adjustments
    assert child.aggregation == decision_module.aggregation
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_adapters.py`

Expected: tests fail because `cooperative.adapters` is absent.

- [ ] **Step 3: Implement the artifact catalog and Numerical adapter**

`CooperativeArtifactCatalog` keeps typed in-memory objects during a run and
writes only their canonical payloads through a supplied `write_object` callback.
`add_numerical` verifies `release.fingerprint == registry.release_sha256` and
`registry.fingerprint == envelope.registry_sha256`. Lookup is by the exact pair
of Bundle SHA fields; missing or mismatched objects raise `ValueError`.

`NumericalCoordinateAdapter` sorts its supplied frozen pairs by
`(release.fingerprint, registry.fingerprint)` and returns the first pair whose
identities differ from the Parent. It returns `None` when the pool is exhausted.

- [ ] **Step 4: Implement bounded Retrieval and Decision adapters**

Retrieval mutation cycles are exact:

```python
ROUND1_NEXT = {
    "timeline_first": "entity_first",
    "entity_first": "contrastive",
    "contrastive": "timeline_first",
}
ROUND2_NEXT = {
    "counterevidence_first": "gap_first",
    "gap_first": "causal_chain_first",
    "causal_chain_first": "counterevidence_first",
}
```

Use `dataclasses.replace` and advance `version` to `v{int(version[1:]) + 1:03d}`
with `parent` set to the old version. Step modulo four changes respectively
round-one strategy, round-two strategy, second-round trigger, or
`max_selected_documents` within the existing `[1, 20]` bound. Construct a new
`RetrievalModuleV2` with the changed Genome, the same source-release identity,
and byte-identical Skill payload; do not mutate Skills or Host verification
booleans.

Decision mutation selects `prompts[step % len(prompts)]`; if that equals the
Parent prompt it advances once. The optional settings cycle changes only
`max_evidence_adjustments` within `[0, 3]` or `aggregation` between `last` and
`mean`. The Decision adapter never accepts a Retrieval artifact argument, so it
cannot alter that scope.

- [ ] **Step 5: Run adapter tests plus representative legacy contracts**

Run: `pytest -q tests/test_evolution_v2_cooperative_adapters.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py`

Expected: all pass; no legacy file is modified.

- [ ] **Step 6: Commit**

```bash
git add evolving_loop/v2/cooperative/adapters.py evolving_loop/v2/cooperative/__init__.py tests/test_evolution_v2_cooperative_adapters.py
git commit -m "feat(cooperative): adapt three agent modules"
```

---

### Task 3: Single/Joint Proposals and Complete Pipeline Evaluation

**Estimated time:** 30 minutes

**Files:**
- Create: `evolving_loop/v2/cooperative/proposals.py`
- Create: `tests/test_evolution_v2_cooperative_proposals.py`
- Create: `tests/test_evolution_v2_cooperative_pipeline.py`
- Modify: `evolving_loop/v2/cooperative/adapters.py`
- Modify: `evolving_loop/v2/cooperative/__init__.py`

**Interfaces:**
- Consumes: Task 2 artifact catalog and coordinate adapters; `PackagePipelineEvaluator._evaluate_components`.
- Produces: `propose_bundle_candidate(parent, arm, catalog, adapters, feedback, step) -> BundleCandidateV2 | None`.
- Produces: `CooperativePipelineAdapter.evaluate(bundle, tasks, stage) -> PackageEvaluation` and `sanitize_train_feedback(parent_eval, child_eval, normalized_cost) -> SanitizedEvolutionFeedback`.

- [ ] **Step 1: Write failing ownership and joint proposal tests**

```python
@pytest.mark.parametrize("arm", ["numerical", "retrieval", "decision"])
def test_single_proposal_changes_only_its_arm(arm, proposal_context):
    candidate = propose_bundle_candidate(arm=arm, **proposal_context)
    child = candidate.to_child(proposal_context["parent"])
    assert changed_scopes(proposal_context["parent"], child) == (arm,)


def test_joint_proposal_combines_two_real_module_children(proposal_context):
    candidate = propose_bundle_candidate(arm="joint", **proposal_context)
    child = candidate.to_child(proposal_context["parent"])
    assert changed_scopes(proposal_context["parent"], child) == (
        "numerical",
        "retrieval",
    )
```

- [ ] **Step 2: Run proposal tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_proposals.py`

Expected: tests fail because `propose_bundle_candidate` is absent.

- [ ] **Step 3: Implement single and joint proposal composition**

For a single arm, call only that coordinate adapter, add the resulting artifact
to the catalog, and construct one `BundleCandidateV2`. For `joint`, request
single proposals in `numerical`, `retrieval`, `decision` order and take the
first two non-null changed scopes. Return `None` if fewer than two exist.
Proposal input is only `SanitizedEvolutionFeedback`; reject a plain mapping or
any feedback whose payload contains a reserved Dev/Public/future key.

- [ ] **Step 4: Write a failing complete-pipeline behavior test**

```python
def test_pipeline_adapter_runs_all_three_agents_and_binds_module_identities(
    pipeline, accepted_bundle, tasks, trace
):
    result = pipeline.evaluate(accepted_bundle, tasks, stage="train")
    assert result.task_count == len(tasks)
    assert trace == [
        (task.numeric.task_id, "numerical", "retrieval_round1", "decision")
        for task in tasks
    ]
    assert result.coverage == 1.0


def test_train_feedback_contains_no_dev_task_or_forecast_values(parent_eval, child_eval):
    feedback = sanitize_train_feedback(parent_eval, child_eval, normalized_cost=0.25)
    encoded = feedback.canonical_bytes()
    assert b"dev" not in encoded.lower()
    assert b"future" not in encoded.lower()
    assert b"task_id" not in encoded.lower()
    assert b"forecast" not in encoded.lower()
```

- [ ] **Step 5: Run pipeline tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_pipeline.py`

Expected: tests fail because `CooperativePipelineAdapter` is absent.

- [ ] **Step 6: Implement the complete pipeline adapter**

Resolve the exact numerical pair, retrieval artifact, and decision artifact
from Bundle SHA fields. Build one read-only empty or release-backed
`RetrievalSkillLibrary`, then call:

```python
PackagePipelineEvaluator._evaluate_components(
    candidate_sha256=bundle.fingerprint(),
    registry=numerical.registry,
    tasks=tuple(tasks),
    stage=stage,
    retrieval_factory=lambda: retrieval_factory(retrieval.genome, skills),
    decision_factory=lambda: decision_factory(decision),
    metric_cap=self.metric_cap,
    expected_retrieval_sha256=retrieval.genome.fingerprint(),
    expected_decision_prompt_sha256=hashlib.sha256(
        decision.prompt.encode("utf-8")
    ).hexdigest(),
)
```

The deterministic smoke factories return real `TwoStageRetrievalAgent` and
`DecisionAgent` instances backed by deterministic in-process LLM clients.
Do not replace `PackagePipelineEvaluator` with a fake score function.

`sanitize_train_feedback` exposes only aggregate relative joint improvement,
invalid/catastrophic/fallback categories, and normalized cost.

- [ ] **Step 7: Run Task 3 tests**

Run: `pytest -q tests/test_evolution_v2_cooperative_proposals.py tests/test_evolution_v2_cooperative_pipeline.py tests/test_package_coordinate_e2e.py`

Expected: all pass in under 20 seconds.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/v2/cooperative tests/test_evolution_v2_cooperative_proposals.py tests/test_evolution_v2_cooperative_pipeline.py
git commit -m "feat(cooperative): evaluate bundle children"
```

---

### Task 4: Kernel-Sealed Runner and Exact Resume

**Estimated time:** 55 minutes

**Files:**
- Create: `evolving_loop/v2/cooperative/runner.py`
- Create: `tests/test_evolution_v2_cooperative_runner.py`
- Modify: `evolving_loop/v2/kernel.py`
- Modify: `tests/test_evolution_v2_kernel.py`
- Modify: `evolving_loop/v2/cooperative/__init__.py`

**Interfaces:**
- Consumes: Tasks 1–3; `EvolutionKernel`; `BudgetPlan`; `ResourceUse`; `V2RunStore`.
- Produces: `run_cooperative_evolution(output_dir, config, seed_artifacts, tasks, adapters, *, resume=False, stop_after=None) -> CooperativeRunResultV2`.
- Extends: `EvolutionKernel.evaluate_transition(parent, child, *, target, evaluation, permit=None, host_scheduler_state_sha256=None)` without changing default behavior.

- [ ] **Step 1: Write failing Kernel scheduler-seal and joint-Numerical tests**

```python
def test_kernel_seals_host_scheduler_state_without_granting_candidate_ownership(kernel_case):
    scheduler_sha = "9" * 64
    accepted = kernel_case.kernel.evaluate_transition(
        kernel_case.parent,
        kernel_case.child,
        target="retrieval",
        evaluation=kernel_case.passed_evaluation,
        permit=kernel_case.permit,
        host_scheduler_state_sha256=scheduler_sha,
    )
    assert accepted.scheduler_state_sha256 == scheduler_sha


def test_joint_numerical_transition_validates_the_release_pair(joint_kernel_case):
    accepted = joint_kernel_case.accept_with_scheduler("8" * 64)
    assert accepted.numerical_release_sha256 == joint_kernel_case.child.numerical_release_sha256
    assert accepted.retrieval_release_sha256 == joint_kernel_case.child.retrieval_release_sha256
```

- [ ] **Step 2: Run Kernel tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_kernel.py -k 'host_scheduler or joint_numerical'`

Expected: the first test raises an unexpected-keyword `TypeError`; the second
fails to retain Numerical accepted-release references for a joint Child.

- [ ] **Step 3: Add the minimal backward-compatible Kernel extension**

Add the optional keyword only to `evaluate_transition`. Validate a non-null
value with `require_sha256`. Pass it to `_seal_acceptance` and
`AcceptanceEvidence.scheduler_state_sha256`; when null, use the Parent SHA as
before. Determine Numerical release validation from
`"numerical" in changed_scopes(parent, child)`, not only `target == "numerical"`.
Do not alter candidate scope rules or any budget/material API.

- [ ] **Step 4: Run the focused existing Kernel and Bundle suite**

Run: `pytest -q tests/test_evolution_v2_bundle.py tests/test_evolution_v2_kernel.py`

Expected: all pass and unchanged calls retain their previous accepted Bundle
fingerprints in existing fixtures.

- [ ] **Step 5: Write failing runner accept/reject/resume tests**

```python
def test_runner_executes_four_arms_and_preserves_rejected_parent(run_case):
    result = run_case.run(scheduler="ucb")
    assert result.attempted_arms == (
        "numerical", "retrieval", "decision", "joint"
    )
    assert result.accepted_steps == 1
    assert result.rejected_steps == 3
    rejected = [row for row in run_case.progress() if row["decision"] == "reject"]
    assert all(row["active_before"] == row["active_after"] for row in rejected)


@pytest.mark.parametrize("scheduler", ["ucb", "thompson"])
def test_closed_step_resume_matches_uninterrupted_bytes(run_case, scheduler):
    full = run_case.run(scheduler=scheduler, directory="full")
    run_case.run(scheduler=scheduler, directory="resumed", stop_after=2)
    resumed = run_case.run(scheduler=scheduler, directory="resumed", resume=True)
    assert resumed.to_payload() == full.to_payload()
    assert run_case.read("resumed/evaluation_complete.json") == run_case.read(
        "full/evaluation_complete.json"
    )
```

- [ ] **Step 6: Run runner tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_runner.py`

Expected: tests fail because `run_cooperative_evolution` is absent.

- [ ] **Step 7: Implement the four-step runner**

For each step:

1. select one arm;
2. build one candidate from Train-only prior feedback;
3. reserve one Kernel evaluation with task execution estimate
   `2 * (len(train) + len(dev))`;
4. evaluate Parent and Child on Train, using the aggregate cache for Parent;
5. update and persist scheduler state from Train metrics only;
6. evaluate Dev only when Train is eligible;
7. close the Kernel evaluation with actual task count, elapsed time, and bytes;
8. call `evaluate_transition` with the new scheduler fingerprint;
9. append one progress row and atomically write `cooperative_checkpoint.json`.

Use these pure gates:

```python
def train_eligible(parent, child, tolerance):
    return (
        child.coverage == 1.0
        and child.invalid_count <= parent.invalid_count
        and child.catastrophic_count <= parent.catastrophic_count
        and child.mean_smae <= parent.mean_smae + tolerance
        and child.mean_srmse <= parent.mean_srmse + tolerance
    )


def dev_passed(parent, child, tolerance):
    parent_joint = (parent.mean_smae + parent.mean_srmse) / 2.0
    child_joint = (child.mean_smae + child.mean_srmse) / 2.0
    return train_eligible(parent, child, tolerance) and (
        child_joint < parent_joint - tolerance
    )
```

A no-candidate step calls `record_outcome` with `train_reward=-1.0`,
`normalized_cost=0.0`, `accepted=False`, appends a closed progress row, and
checkpoints without opening Dev. `stop_after` is test-only and stops after a
closed step. A complete resume reads and returns the existing terminal object
without writes.

- [ ] **Step 8: Run runner, Kernel, and Project 2 frozen-pair tests**

Run: `pytest -q tests/test_evolution_v2_cooperative_runner.py tests/test_evolution_v2_kernel.py tests/test_evolution_v2_bundle.py tests/test_evolution_v2_numerical_adapters.py -k 'FrozenNumericalArtifactsV2 or cooperative or host_scheduler or joint_numerical or not slow'`

Expected: all selected tests pass in under 60 seconds.

- [ ] **Step 9: Commit**

```bash
git add evolving_loop/v2/cooperative evolving_loop/v2/kernel.py tests/test_evolution_v2_cooperative_runner.py tests/test_evolution_v2_kernel.py
git commit -m "feat(cooperative): run resumable bundle search"
```

---

### Task 5: Unified CLI, Two Smoke Profiles, and 4/1 Fixture

**Estimated time:** 35 minutes

**Files:**
- Create: `evolving_loop/v2/cooperative/config.py`
- Create: `configs/evolution_v2/cooperative/smoke-ucb.json`
- Create: `configs/evolution_v2/cooperative/smoke-thompson.json`
- Create: `tests/build_evolution_v2_cooperative_fixture.py`
- Create: `tests/test_evolution_v2_cooperative_cli.py`
- Modify: `evolving_loop/v2/cli.py`
- Modify: `evolving_loop/v2/cooperative/__init__.py`

**Interfaces:**
- Consumes: Task 4 runner and the existing task/release parsing helpers in `evolving_loop.v2.cli`.
- Produces: `CooperativeConfigV2.from_payload`, `load_cooperative_config`, and the public `cooperative_evolve` function shown in Step 5.
- Extends: existing `evolve` parser with optional seed input flags; fake two-argument mode remains valid.

- [ ] **Step 1: Write failing strict-config and parser tests**

```python
def test_cooperative_config_is_a_strict_prototype_profile(smoke_config_payload):
    config = CooperativeConfigV2.from_payload(smoke_config_payload)
    assert config.max_steps == 4
    assert config.children_per_step == 1
    assert config.control.runner == "production"


def test_evolve_parser_accepts_cooperative_seed_inputs():
    args = build_parser().parse_args([
        "evolve", "--config", "c.json", "--seed-supply", "s.json",
        "--task-manifest", "t.json", "--retrieval-release", "r.json",
        "--decision-policy", "d.json", "--output-dir", "out",
    ])
    assert args.command == "evolve"
    assert args.retrieval_release.name == "r.json"
```

- [ ] **Step 2: Run config/CLI tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_cli.py -k 'config or parser'`

Expected: collection or parsing fails because the cooperative config and flags
do not exist.

- [ ] **Step 3: Implement `CooperativeConfigV2` and ship two configs**

Require the exact fields from the design. Enforce `max_steps == 4`,
`children_per_step == 1`, `0.0 < discount <= 1.0`, finite non-negative cost
weight, `metric_cap > 0`, `acceptance_tolerance == 1e-12`, non-Public embedded
control, and exact `ResourceUse` ceilings. Use 600 seconds for both smoke
profiles and change only `control.scheduler` between their canonical files.

- [ ] **Step 4: Build the deterministic canonical 4/1 fixture generator**

The script writes exactly:

```text
tests/fixtures/evolution_v2_cooperative/seed_supply.json
tests/fixtures/evolution_v2_cooperative/tasks_4_1.json
tests/fixtures/evolution_v2_cooperative/retrieval_release.json
tests/fixtures/evolution_v2_cooperative/decision_policy.json
```

It uses four labeled Train `ContextTask` payloads and one labeled Dev payload,
no Public member, a seed Supply with safe anchor plus one alternative, a
candidate Retrieval release with empty Skills, and a deterministic Decision
prompt. Write files with `canonical_v2_bytes`; invoking the generator twice
must produce byte-identical outputs.

- [ ] **Step 5: Implement cooperative dispatch in the unified `evolve` command**

Read the config once as canonical JSON. If its keys match `EvolutionV2Config`,
run the existing `_evolve` path and reject cooperative-only flags. If its keys
match `CooperativeConfigV2`, require all four seed paths, parse the 4/1 task
manifest, build deterministic smoke factories, and call
`run_cooperative_evolution`. Keep `numerical-evolve` unchanged.

Expose this programmatic seam for pilots:

```python
def cooperative_evolve(
    config_path: Path,
    seed_supply_path: Path,
    task_manifest_path: Path,
    retrieval_release_path: Path,
    decision_policy_path: Path,
    output: Path,
    *,
    host_runtime=None,
) -> dict[str, object]:
    return _cooperative_evolve(
        config_path,
        seed_supply_path,
        task_manifest_path,
        retrieval_release_path,
        decision_policy_path,
        output,
        host_runtime=host_runtime,
    )
```

When `host_runtime is None`, only `smoke` is legal and deterministic in-process
clients are used. A pilot supplies factories and frozen Numerical alternatives
through `host_runtime`.

- [ ] **Step 6: Write and run end-to-end CLI tests**

```python
@pytest.mark.parametrize("scheduler", ["ucb", "thompson"])
def test_unified_evolve_completes_cooperative_smoke(tmp_path, scheduler):
    completed = run_cli(tmp_path, scheduler)
    assert completed["status"] == "cooperative_complete"
    assert completed["attempted_arms"] == [
        "numerical", "retrieval", "decision", "joint"
    ]
    assert completed["public_test_accessed"] is False


def test_complete_cli_resume_is_byte_identical_and_read_only(tmp_path):
    first = run_cli(tmp_path, "ucb")
    before = snapshot(tmp_path / "run")
    second = run_cli(tmp_path, "ucb")
    assert second == first
    assert snapshot(tmp_path / "run") == before
```

Run: `pytest -q tests/test_evolution_v2_cooperative_cli.py`

Expected: all pass in under 90 seconds.

- [ ] **Step 7: Run fake and Numerical CLI compatibility tests**

Run: `pytest -q tests/test_evolution_v2_cli.py tests/test_evolution_v2_numerical_cli.py -k 'parser or fake or smoke or complete or resume'`

Expected: all selected tests pass with unchanged fake and Numerical summaries.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/v2 configs/evolution_v2/cooperative tests/build_evolution_v2_cooperative_fixture.py tests/fixtures/evolution_v2_cooperative tests/test_evolution_v2_cooperative_cli.py
git commit -m "feat(cli): expose cooperative evolution"
```

---

### Task 6: Isolation Tests, User Documentation, and Fast Exit Gate

**Estimated time:** 25 minutes

**Files:**
- Create: `tests/test_evolution_v2_cooperative_safety.py`
- Create: `docs/evolution-v2-cooperative-bundle.md`
- Modify: `README.md`
- Modify: Project 3 files only if an exit-gate test first demonstrates a defect.

**Interfaces:**
- Consumes: all Project 3 public APIs and CLI artifacts.
- Produces: documented smoke/pilot commands and the Project 3 exit evidence.

- [ ] **Step 1: Write failing information-flow and aggregate-write tests**

```python
def test_public_membership_is_rejected_before_proposal(run_case, public_task):
    before = run_case.proposal_count
    with pytest.raises(ValueError, match="Public"):
        run_case.run(tasks=(*run_case.tasks, public_task))
    assert run_case.proposal_count == before


def test_dev_values_never_enter_scheduler_or_proposal_feedback(run_case):
    run_case.run()
    payloads = run_case.scheduler_and_feedback_payloads()
    forbidden = (b"future_values", b"dev_metrics", b"task_id", b"forecast")
    assert all(token not in payload for payload in payloads for token in forbidden)


def test_runner_persists_no_per_forecast_artifacts(run_case):
    result = run_case.run()
    assert result.evaluated_candidates > 0
    files = run_case.relative_files()
    assert not any("forecast" in path or "task_result" in path for path in files)
    assert len([path for path in files if path.startswith("evaluations/")]) <= 8
```

- [ ] **Step 2: Run safety tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_cooperative_safety.py`

Expected: at least one assertion fails until the runner rejects Public
membership before proposal and limits persistence to aggregate evaluations.

- [ ] **Step 3: Fix only demonstrated Project 3 defects with TDD**

Keep the test that reproduces each defect, implement the smallest correction,
and rerun the named test before moving to the next. Do not expand the fix into
new filesystem hardening or a full Project 2 validation pass.

- [ ] **Step 4: Write concise research-prototype documentation**

Document:

- the Project 2 Supply -> Project 3 Bundle data flow;
- the exact UCB and Thompson smoke commands;
- the four mutation scopes and complete-pipeline acceptance rule;
- checkpoint/resume artifacts;
- the statement that 4/1 smoke is CI evidence, 8/2 is an optional pilot, and
  80/20/four-hour are optional experiments rather than completion gates; and
- the explicit exclusions: DGM, L1 evolution, Public scoring, per-forecast
  persistence, and production filesystem attack testing.

Add one README link and no architectural duplication.

- [ ] **Step 5: Run the Project 3 fast blocking gate**

Run:

```bash
pytest -q \
  tests/test_evolution_v2_cooperative_contracts.py \
  tests/test_evolution_v2_cooperative_schedulers.py \
  tests/test_evolution_v2_cooperative_adapters.py \
  tests/test_evolution_v2_cooperative_proposals.py \
  tests/test_evolution_v2_cooperative_pipeline.py \
  tests/test_evolution_v2_cooperative_runner.py \
  tests/test_evolution_v2_cooperative_cli.py \
  tests/test_evolution_v2_cooperative_safety.py \
  tests/test_evolution_v2_bundle.py \
  tests/test_evolution_v2_kernel.py \
  tests/test_package_coordinate_e2e.py
```

Expected: all pass in under three minutes.

- [ ] **Step 6: Run two fresh real CLI smoke commands**

```bash
python tests/build_evolution_v2_cooperative_fixture.py
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/cooperative/smoke-ucb.json \
  --seed-supply tests/fixtures/evolution_v2_cooperative/seed_supply.json \
  --task-manifest tests/fixtures/evolution_v2_cooperative/tasks_4_1.json \
  --retrieval-release tests/fixtures/evolution_v2_cooperative/retrieval_release.json \
  --decision-policy tests/fixtures/evolution_v2_cooperative/decision_policy.json \
  --output-dir /tmp/evolution-v2-cooperative-ucb
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/cooperative/smoke-thompson.json \
  --seed-supply tests/fixtures/evolution_v2_cooperative/seed_supply.json \
  --task-manifest tests/fixtures/evolution_v2_cooperative/tasks_4_1.json \
  --retrieval-release tests/fixtures/evolution_v2_cooperative/retrieval_release.json \
  --decision-policy tests/fixtures/evolution_v2_cooperative/decision_policy.json \
  --output-dir /tmp/evolution-v2-cooperative-thompson
```

Expected: each prints canonical JSON with `status="cooperative_complete"`, all
four attempted arms, and `public_test_accessed=false`. Use new empty directories
if either exact `/tmp` target already exists; do not delete an unknown run.

- [ ] **Step 7: Run the narrow compatibility gate**

Run: `pytest -q tests/test_evolution_v2_cli.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py tests/test_evolution_v2_numerical_compatibility.py -k 'not full_cli_execution and not real_source'`

Expected: all selected tests pass. Do not run the full Numerical runner or
80/20 suite.

- [ ] **Step 8: Commit**

```bash
git add tests/test_evolution_v2_cooperative_safety.py docs/evolution-v2-cooperative-bundle.md README.md evolving_loop/v2/cooperative evolving_loop/v2/kernel.py evolving_loop/v2/cli.py
git commit -m "docs(cooperative): finish prototype gate"
```

## Implementation Time Budget

| Task | Budget |
|---|---:|
| 1. Contracts and schedulers | 25 min |
| 2. Three adapters | 35 min |
| 3. Proposals and pipeline | 30 min |
| 4. Runner and resume | 55 min |
| 5. CLI and fixture | 35 min |
| 6. Safety, docs, exit gate | 25 min |
| **Total** | **205 min (3 h 25 min)** |

The budget includes focused tests and two deterministic CLI smokes. It excludes
optional 8/2 real-client pilots, 80/20 experiments, Public evaluation, and
Project 4 DGM work.
