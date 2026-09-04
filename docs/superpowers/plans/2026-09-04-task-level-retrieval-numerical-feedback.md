# Task-Level Retrieval-to-Numerical Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make verified per-task Retrieval and Decision evidence from cycle 1 available to the cycle-2 Numerical proposer, and expose a two-cycle 8/2/2 interaction smoke that proves the full N-R-D loop ran without accessing Public-99.

**Architecture:** Add a closed, immutable task-feedback contract under `common/evolution_core`, then let the package evaluator capture host-only successful inference traces and project them through a validating snapshot builder. The package proposal boundary carries only the sanitized cases, while the interaction-smoke runner persists their fingerprints and injects them only into cycle-2 Numerical prompts.

**Tech Stack:** Python 3 dataclasses, existing package co-evolution runner, pytest, SHA-256 canonical JSON identities.

**Spec:** `docs/superpowers/specs/2026-09-04-task-level-retrieval-numerical-feedback-design.md`

**Implementation status (2026-09-04):** Tasks 1-5 are implemented. The combined package/Retrieval/Numerical regression set passes 821 tests. The model-facing projection intentionally omits Train/Dev membership so the Numerical proposer cannot learn partition-specific rules; membership remains host-only in `PackageTaskFeedbackLedger`.

## Global Constraints

- Train and Dev may provide evolution evidence; Public-99 may not be loaded or projected.
- Retrieval may influence only the next Numerical proposal and may not edit or veto a forecast.
- Model-facing evidence contains no benchmark/document identity, quote, forecast, label, residual, score, or repository path.
- Case identities are request-local and forbidden in accepted Numerical output.
- Formal 80/20 acceptance gates remain unchanged.
- An interaction-smoke completion is successful only when both N-R-D cycles and non-empty cycle-2 feedback were exercised.

---

### Task 1: Closed task-feedback contract

**Files:**
- Create: `common/evolution_core/task_feedback.py`
- Create: `tests/test_task_evidence_feedback.py`

**Interfaces:**
- Produces: `TaskMorphologyProjection`, `TaskEvidenceCase`, and `TaskEvidenceProjection` immutable dataclasses.
- Produces: `TaskEvidenceProjection.to_payload() -> dict[str, object]`, `snapshot_sha256`, and `projection_sha256`.

- [ ] **Step 1: Write failing contract tests**

```python
def _valid_case():
    return TaskEvidenceCase(
        case_id="case_000_deadbeef",
        morphology=TaskMorphologyProjection(
            frequency="daily", history="medium", horizon="short",
            trend="strong_up", periodicity="weak", intermittency="dense",
            recent_regime="stable",
        ),
        assumption_id="trend_ready",
        claim="The recent trend persists.",
        failure_condition="A verified event reverses the trend.",
        stance="falsified", target_match="matched", window_relation="overlaps",
        magnitude_status="present", mechanism="event_shock",
        decision_action="rejected", evidence_chain_sha256="c" * 64,
    )

def test_projection_is_closed_canonical_and_contains_no_host_identity():
    projection = TaskEvidenceProjection(
        source_bundle_sha256="a" * 64,
        request_namespace_sha256="b" * 64,
        cases=(_valid_case(),),
    )
    assert projection.to_payload()["cases"][0]["case_id"].startswith("case_")
    assert "task_17" not in json.dumps(projection.to_payload())

def test_projection_rejects_sensitive_fields():
    with pytest.raises(TaskFeedbackError):
        replace(_valid_case(), claim="Read task_17 future_values")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_task_evidence_feedback.py`

Expected: collection fails because `common.evolution_core.task_feedback` does not exist.

- [ ] **Step 3: Implement exact enums, canonical ordering, recursive leakage checks, and fingerprints**

```python
@dataclass(frozen=True)
class TaskEvidenceCase:
    case_id: str
    morphology: TaskMorphologyProjection
    assumption_id: str
    claim: str
    failure_condition: str
    stance: Literal["supported", "falsified", "uncertain"]
    target_match: Literal["matched", "unmatched"]
    window_relation: Literal["overlaps", "precedes", "after", "unknown"]
    magnitude_status: Literal["present", "missing", "conflicting", "not_applicable"]
    mechanism: Literal["event_shock", "promotion", "policy_change", "capacity_change", "measurement_error", "regime_change", "calendar_effect", "macroeconomic", "unknown"]
    decision_action: Literal["selected", "rejected", "unresolved"]
    evidence_chain_sha256: str
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_task_evidence_feedback.py`

- [ ] **Step 5: Commit**

```bash
git add common/evolution_core/task_feedback.py tests/test_task_evidence_feedback.py
git commit -m "feat(evolution): add task feedback contract"
```

### Task 2: Trusted trace capture and snapshot construction

**Files:**
- Create: `evolving_loop/package_task_feedback.py`
- Modify: `evolving_loop/package_pipeline_evaluator.py`
- Modify: `tests/test_task_evidence_feedback.py`
- Create: `tests/test_package_task_feedback.py`

**Interfaces:**
- Produces: `PackageTaskFeedbackLedger.record(bundle, task, result)`; the ledger owns authoritative Train/Dev membership.
- Produces: `PackageTaskFeedbackLedger.build_projection(bundle, task_ids, generation) -> TaskEvidenceProjection`.
- Consumes: verified `NumericalTwoStageResult` values only after package inference contract checks pass.

- [ ] **Step 1: Write failing tests for Train/Dev projection, Public rejection, mixed-bundle rejection, and stance/action derivation**

```python
ledger.record(bundle, train_task, verified_result)
projection = ledger.build_projection(bundle, (train_task.numeric.task_id,), generation=3)
assert projection.cases[0].stance == "falsified"
assert projection.cases[0].decision_action == "rejected"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_task_evidence_feedback.py tests/test_package_task_feedback.py`

- [ ] **Step 3: Implement deterministic morphology buckets and verified trace ledger**

The ledger keys traces by principal Numerical/Retrieval/Decision fingerprints plus task identity. It rejects missing tasks, Public membership, duplicate task/assumption pairs, unknown assumption IDs, fallback inference, and bundle-fingerprint mismatch as one whole snapshot.

- [ ] **Step 4: Add evaluator capture after successful `run_numerical_two_stage` validation**

```python
if self.task_feedback_ledger is not None:
    self.task_feedback_ledger.record(bundle, task, result)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_task_evidence_feedback.py tests/test_package_task_feedback.py`

- [ ] **Step 6: Commit**

```bash
git add evolving_loop/package_task_feedback.py evolving_loop/package_pipeline_evaluator.py tests/test_task_evidence_feedback.py tests/test_package_task_feedback.py
git commit -m "feat(evolution): build verified task feedback"
```

### Task 3: Numerical proposal injection and output firewall

**Files:**
- Modify: `evolving_loop/package_candidate_proposal.py`
- Modify: `evolving_loop/package_numerical_evolution.py`
- Modify: `numerical_agent/evolution/champion_controller.py`
- Modify: `numerical_agent/evolution/champion_proposal.py`
- Modify: `tests/test_package_candidate_proposal.py`
- Modify: `tests/test_evolution_champion_proposal.py`

**Interfaces:**
- `PackageProposalFeedback.task_evidence: TaskEvidenceProjection | None`.
- `ChampionProposerAdapter.propose(parent, evidence, *, generation: int, task_evidence: TaskEvidenceProjection | None = None)`.
- The model prompt receives `task_evidence`, while response parsing rejects every supplied case ID and evidence digest.

- [ ] **Step 1: Write failing tests proving cycle-1 empty input, cycle-2 exact projection, and case-ID/digest output rejection**

```python
payload = _proposal_payload(parent, inventory, evidence, task_evidence=projection, minimum=1, maximum=1)
assert payload["task_evidence"] == projection.to_payload()
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_package_candidate_proposal.py tests/test_evolution_champion_proposal.py`

- [ ] **Step 3: Thread the optional exact projection through package and Champion adapters**

No-feedback callers retain the existing behavior. Numerical output containing a case handle or chain digest consumes the one schema retry and then fails closed.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_package_candidate_proposal.py tests/test_evolution_champion_proposal.py`

- [ ] **Step 5: Commit**

```bash
git add evolving_loop/package_candidate_proposal.py evolving_loop/package_numerical_evolution.py numerical_agent/evolution/champion_controller.py numerical_agent/evolution/champion_proposal.py tests/test_package_candidate_proposal.py tests/test_evolution_champion_proposal.py
git commit -m "feat(evolution): feed retrieval evidence to numerical"
```

### Task 4: Two-cycle interaction smoke and durable evidence

**Files:**
- Modify: `evolving_loop/package_artifacts.py`
- Modify: `evolving_loop/run_package_coevolution.py`
- Modify: `tests/test_run_package_coevolution.py`
- Modify: `tests/test_package_artifacts.py`

**Interfaces:**
- CLI: `--interaction-smoke` requires `--cycles 2 --children-per-coordinate 1`.
- CLI: `--feedback-mode {none,task}` selects control or treatment and is included in run identity.
- Artifact: `task_feedback/task-feedback-<generation>.json` binds audit and projection hashes.
- Completion: `full_chain_exercised` and status `complete` or `incomplete_chain`.

- [ ] **Step 1: Write failing parser, orchestration, artifact, resume, and completion tests**

```python
args = parser.parse_args([*required, "--interaction-smoke", "--cycles", "2",
                          "--children-per-coordinate", "1", "--feedback-mode", "task"])
_validate_mode(args)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_run_package_coevolution.py tests/test_package_artifacts.py`

- [ ] **Step 3: Implement the isolated smoke-only Retrieval publication rule**

For interaction smoke only, a Retrieval Child may publish after schema, ownership, provenance, full task coverage, no Public access, and successful verified inference; forecast-gain failures remain recorded but do not block the isolated smoke lineage.

- [ ] **Step 4: Inject feedback only at the second Numerical phase and persist its hashes**

Control mode persists an explicit empty projection. Treatment mode requires at least one verified case; absence is `incomplete_chain`, not success.

- [ ] **Step 5: Require exactly two ordered N-R-D cycles for successful completion**

```python
full_chain = tuple(step["target"] for step in recorder.steps) == (
    "numerical", "retrieval", "decision", "numerical", "retrieval", "decision"
)
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_run_package_coevolution.py tests/test_package_artifacts.py`

- [ ] **Step 7: Commit**

```bash
git add evolving_loop/package_artifacts.py evolving_loop/run_package_coevolution.py tests/test_run_package_coevolution.py tests/test_package_artifacts.py
git commit -m "feat(evolution): add interaction smoke"
```

### Task 5: Regression verification and runnable handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-09-04-task-level-retrieval-numerical-feedback.md`

**Interfaces:**
- Confirms the old one-cycle smoke and formal runner keep their existing authority.
- Confirms a local deterministic interaction fixture creates a non-empty task-feedback artifact without Public access.

- [ ] **Step 1: Run all package-evolution and proposal tests**

Run: `pytest -q tests/test_task_evidence_feedback.py tests/test_package_task_feedback.py tests/test_package_candidate_proposal.py tests/test_evolution_champion_proposal.py tests/test_run_package_coevolution.py tests/test_package_artifacts.py tests/test_package_coordinate_e2e.py`

- [ ] **Step 2: Run the broader package/retrieval/numerical regression set**

Run: `pytest -q tests/test_package_*.py tests/test_numerical_retrieval_handoff.py tests/test_two_stage_retrieval.py`

- [ ] **Step 3: Run the deterministic local interaction smoke fixture**

Run the new test fixture through both feedback modes and verify treatment produces non-empty `task_evidence`, both outputs have `public_test_accessed=false`, and both completion markers have `full_chain_exercised=true`.

- [ ] **Step 4: Check the final diff and commit plan progress**

```bash
git diff --check
git status --short
git add docs/superpowers/plans/2026-09-04-task-level-retrieval-numerical-feedback.md
git commit -m "docs(evolution): record feedback implementation"
```
