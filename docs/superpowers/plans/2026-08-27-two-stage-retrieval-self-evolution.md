# Two-Stage Retrieval Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage, evidence-chain Retrieval Agent whose prompts, strategies, budgets, and declarative skills can evolve on the frozen 80 Train tasks and be accepted once on the read-only 20 Dev tasks without exposing labels to inference.

**Architecture:** Round 1 retrieves independently from target metadata, timestamps, and documents. A fixed Decision gap interface then emits sanitized named gaps, and Round 2 receives only verified Round 1 evidence plus anonymized Morphology assumptions; deterministic host code verifies and merges both rounds before final Decision. A Retrieval-specific typed Genome evolves three bounded child scopes with Train-only successive halving, while immutable quote, provenance, split, scorer, and frozen-inference gates remain outside the mutable policy.

**Tech Stack:** Python 3.10+, frozen dataclasses, strict JSON parsing, existing `common.llm.LLMClient`, existing Dr-CiK `ContextTask`, existing sMAE/sRMSE metrics, pytest, Git-tracked JSON/Markdown releases.

**Spec:** `docs/superpowers/specs/2026-08-27-two-stage-retrieval-self-evolution-design.md`

## Global Constraints

- Use `splits/drcik_public_80_20_99_v1.json`: 80 Train for mutation, 20 Dev for one final read-only acceptance, and no access to the consumed 99-task Public Regression set during evolution.
- Hidden Dr-CiK tasks are inference-only; no scoring, skill writes, Genome writes, or mutation calls are permitted.
- Round 1 receives no Coding candidates, Morphology assumptions, hindcast scores, resolved future values, GT evidence, or document role/subtype labels.
- Round 2 receives no candidate names, forecast arrays, Python source, hindcast scores, resolved future values, GT evidence, or document role/subtype labels.
- Exact quote, provenance, temporal-window, magnitude, split, metric, resource, and evaluator checks are immutable host behavior.
- The current single-pass `RetrievalAgent` remains available as an explicit baseline; the default behavior does not silently change for old policies.
- Retrieval evolution may mutate only typed Retrieval Genome fields and declarative Retrieval Skills; it may not modify the scorer, verifier, data loader, split, sandbox, tests, or another principal agent.
- sMAE and sRMSE are the official point-forecast metrics. Acceptance is Pareto-safe on both and requires at least one strict improvement.
- A Retrieval failure must preserve the pure Numerical fallback; a Round 2 failure must preserve verified Round 1 evidence.
- Use the existing repository as the only Git repository. Do not create nested `.git` directories under release or run directories.
- Preserve unrelated dirty worktree changes. At execution time, create an isolated worktree from the intended integration commit before modifying source files.

## File Structure

New focused modules:

```text
evolving_loop/
├── morphology_adapter.py                 # Optional sanitized bridge to Numerical Morphology
├── coordinate_evolution.py               # One-module-per-generation orchestration
└── retrieval_agent/
    ├── schemas.py                        # Strict inference and evidence-chain contracts
    ├── policy.py                         # RetrievalGenome and immutable release artifacts
    ├── verifier.py                       # Quote/provenance/chain verification and merge
    ├── two_stage_agent.py                # Round 1 and Round 2 LLM calls
    ├── credit.py                         # Train-only chain/skill marginal attribution
    ├── evolution.py                      # Retrieval-specific children and Train/Dev search
    └── releases/v000/                    # Hand-written seed release
```

Existing modules extended in place:

```text
evolving_loop/retrieval_agent/agent.py          # Backward-compatible RetrievalResult adapter
evolving_loop/retrieval_agent/skill_library.py  # Typed, versioned, non-destructive skills
evolving_loop/decision_agent/agent.py           # Sanitized named gap output
evolving_loop/harness.py                        # Explicit single-pass/two-stage topology
evolving_loop/evaluation.py                     # Retrieval and chain diagnostics
evolving_loop/co_evolution.py                   # Embed/freeze accepted Retrieval releases
evolving_loop/frozen_inference.py               # Rich Retrieval audit output, zero writes
evolving_loop/cli.py                            # Retrieval evolution/inference commands
scripts/run_retrieval_evolution.sh              # Reproducible 80/20 runner
```

---

### Task 1: Strict Retrieval Schemas and Sanitized Boundaries

**Files:**
- Create: `evolving_loop/retrieval_agent/schemas.py`
- Modify: `evolving_loop/retrieval_agent/__init__.py`
- Test: `tests/test_retrieval_schemas.py`

**Interfaces:**
- Consumes: `ContextTask.retrieval_view()`, sanitized Morphology assumption records, and LLM JSON objects.
- Produces: `RetrievalAssumption`, `EvidenceCitation`, `EvidenceChain`, `RetrievalGap`, `RetrievalRoundResult`, `FinalRetrievalCard`, `build_round1_payload()`, `build_round2_payload()`, and strict `from_payload()` constructors.

- [ ] **Step 1: Write failing schema tests.** Add literal fixtures that prove unknown fields, duplicate IDs, invalid enum values, non-finite magnitudes, malformed timestamps, and candidate-name leakage are rejected.

```python
def test_round1_payload_is_assumption_blind(context_task):
    payload = build_round1_payload(context_task, skills=())
    encoded = json.dumps(payload, sort_keys=True)
    assert "documents" in payload
    for forbidden in (
        "coding_hypotheses", "assumptions", "future_values", "gt_evidence",
        "role", "subtype", "hindcast_smae", "hindcast_srmse",
    ):
        assert forbidden not in encoded


def test_round2_rejects_candidate_identity_and_scores():
    with pytest.raises(RetrievalContractError, match="forbidden round-two field"):
        RetrievalAssumption.from_payload({
            "assumption_id": "a_trend",
            "kind": "trend_persistence",
            "claim": "The recent trend persists.",
            "failure_condition": "A regime reversal occurs.",
            "candidate_id": "linear_trend",
            "hindcast_smae": 0.8,
        })
```

- [ ] **Step 2: Run the focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_schemas.py`

Expected: collection fails because `evolving_loop.retrieval_agent.schemas` does not exist.

- [ ] **Step 3: Implement frozen schema types with exact-key parsing.** Use `Mapping[str, object]`, explicit key sets, finite-number validation, identifier validation, duplicate detection, and enum literals. Do not use permissive `**payload` construction.

```python
class RetrievalContractError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalAssumption:
    assumption_id: str
    kind: str
    claim: str
    failure_condition: str

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalAssumption":
        required = {"assumption_id", "kind", "claim", "failure_condition"}
        if set(raw) != required:
            raise RetrievalContractError("forbidden round-two field")
        values = {key: str(raw[key]).strip() for key in required}
        if not values["assumption_id"].isidentifier() or any(
            not values[key] for key in ("kind", "claim", "failure_condition")
        ):
            raise RetrievalContractError("invalid retrieval assumption")
        return cls(**values)
```

`EvidenceChain` must contain stable `chain_id`, claim, entity/target matches, temporal relation, mechanism, direction, magnitude fields, inclusive timestamps, citations, missing links, skill IDs, addressed assumption IDs, stance, and `numeric_eligible`. `FinalRetrievalCard` must contain both stage results plus deterministic merged chains, selected IDs, rejections, unresolved contradictions, and completeness.

- [ ] **Step 4: Run schema tests and the existing Retrieval tests.**

Run: `python -m pytest -q tests/test_retrieval_schemas.py tests/test_evolving_agent_agent.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the schema boundary.**

```bash
git add evolving_loop/retrieval_agent/schemas.py \
  evolving_loop/retrieval_agent/__init__.py tests/test_retrieval_schemas.py
git commit -m "feat(retrieval): add strict stage schemas"
```

---

### Task 2: Typed Retrieval Genome and Release Contract

**Files:**
- Create: `evolving_loop/retrieval_agent/policy.py`
- Create: `evolving_loop/retrieval_agent/releases/v000/genome.json`
- Create: `evolving_loop/retrieval_agent/releases/v000/round1_prompt.md`
- Create: `evolving_loop/retrieval_agent/releases/v000/round2_prompt.md`
- Create: `evolving_loop/retrieval_agent/releases/v000/skills.json`
- Create: `evolving_loop/retrieval_agent/releases/v000/manifest.json`
- Test: `tests/test_retrieval_policy.py`

**Interfaces:**
- Consumes: a complete strict JSON Genome and release directory.
- Produces: `RetrievalGenome.from_payload()`, `RetrievalGenome.to_payload()`, `RetrievalGenome.fingerprint()`, `RetrievalRelease.load()`, and `write_retrieval_release()`.

- [ ] **Step 1: Write failing round-trip, range, enum, hash, and path tests.** Require exact fields, immutable tuples, no unknown strategy, budgets within fixed ranges, and a release whose hashes bind every artifact.

```python
def test_seed_release_is_self_consistent():
    release = RetrievalRelease.load(
        Path("evolving_loop/retrieval_agent/releases/v000")
    )
    assert release.genome.version == "v000"
    assert release.manifest["genome_sha256"] == release.genome.fingerprint()
    assert release.skills == ()


def test_genome_cannot_disable_host_verification():
    raw = RetrievalGenome.seed().to_payload()
    raw["require_temporal_overlap"] = False
    with pytest.raises(RetrievalPolicyError, match="cannot weaken"):
        RetrievalGenome.from_payload(raw)
```

- [ ] **Step 2: Run the focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_policy.py`

Expected: import failure for the missing policy module.

- [ ] **Step 3: Implement the complete Genome contract.** Fields and bounds are fixed as follows:

```python
ROUND1_STRATEGIES = {"timeline_first", "entity_first", "contrastive"}
ROUND2_STRATEGIES = {"counterevidence_first", "gap_first", "causal_chain_first"}
SECOND_ROUND_TRIGGERS = {"on_named_gap", "on_incomplete_chain", "always", "never"}
BOUNDS = {
    "max_selected_documents": (1, 20),
    "max_evidence_chains": (1, 12),
    "max_citations_per_chain": (1, 8),
}


@dataclass(frozen=True)
class RetrievalGenome:
    schema_version: int
    version: str
    parent: str | None
    round1_prompt: str
    round2_prompt: str
    round1_strategy: str
    round2_strategy: str
    second_round_trigger: str
    max_selected_documents: int
    max_evidence_chains: int
    max_citations_per_chain: int
    require_counterevidence_search: bool
    require_target_match: bool
    require_temporal_overlap: bool
    active_skill_ids: tuple[str, ...]
```

The seed release uses `timeline_first`, `counterevidence_first`, `on_named_gap`, document/chain/citation budgets `8/4/4`, and all three immutable requirements set to `true`.

- [ ] **Step 4: Implement atomic release writing.** Write into a sibling temporary directory, hash the canonical Genome, prompts, and Skills, create `manifest.json`, then rename into `releases/vNNN`. Reject an existing destination and any path containing `.git`.

- [ ] **Step 5: Run focused tests and validate release JSON.**

Run: `python -m pytest -q tests/test_retrieval_policy.py && python -m json.tool evolving_loop/retrieval_agent/releases/v000/genome.json >/dev/null && git diff --check`

Expected: all commands exit zero.

- [ ] **Step 6: Commit the seed policy and release contract.**

```bash
git add evolving_loop/retrieval_agent/policy.py \
  evolving_loop/retrieval_agent/releases/v000 tests/test_retrieval_policy.py
git commit -m "feat(retrieval): add versioned genome releases"
```

---

### Task 3: Versioned Declarative Retrieval Skills

**Files:**
- Modify: `evolving_loop/retrieval_agent/skill_library.py`
- Modify: `evolving_loop/skill_learning.py`
- Modify: `evolving_loop/co_evolution.py`
- Test: `tests/test_retrieval_skill_lifecycle.py`
- Modify: `tests/test_evolving_agent_skill_library.py`
- Modify: `tests/test_co_evolution.py`

**Interfaces:**
- Consumes: legacy Retrieval skill rows, typed skill rows, and transactional operations `add`, `repair`, `specialize`, `merge`, and `quarantine`.
- Produces: versioned `RetrievalSkill`, `RetrievalSkillOperation`, `RetrievalSkillLibrary.apply_operations()`, stage/applicability filtering, and backward-compatible policy snapshots.

- [ ] **Step 1: Write failing migration and lifecycle tests.** Cover legacy load, lineage-preserving repair, narrowed specialization, merge ancestry, quarantine/reactivation, atomic rejection, read-only clone, and the absence of destructive delete.

```python
def test_skill_operations_are_atomic_and_non_destructive(tmp_path):
    library = RetrievalSkillLibrary(tmp_path / "skills.json", [seed_skill()])
    operations = (
        RetrievalSkillOperation.repair("explicit_window", repaired_skill()),
        RetrievalSkillOperation.quarantine("unknown_skill", "not registered"),
    )
    with pytest.raises(RetrievalSkillError, match="unknown_skill"):
        library.apply_operations(operations)
    assert library.get_by_id("explicit_window").version == 1
    assert not library.path.exists()
```

- [ ] **Step 2: Run the focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_skill_lifecycle.py tests/test_evolving_agent_skill_library.py tests/test_co_evolution.py`

Expected: failures identify missing typed lifecycle APIs and legacy serialization support.

- [ ] **Step 3: Replace the flat skill row with a strict versioned schema.** Preserve `name` as a compatibility property while making `skill_id` the stable identity.

```python
@dataclass(frozen=True)
class RetrievalSkill:
    skill_id: str
    version: int
    parent_version: int | None
    stage: Literal["round1", "round2", "both"]
    status: Literal["candidate", "accepted", "specialized", "quarantined"]
    name: str
    description: str
    applicability: RetrievalApplicability
    query_steps: tuple[str, ...]
    required_chain_fields: tuple[str, ...]
    counterevidence_rule: str
    failure_conditions: tuple[str, ...]
    validated_task_ids: tuple[str, ...] = ()
    validated_entities: tuple[str, ...] = ()
    validation_smae_gain: float | None = None
    validation_srmse_gain: float | None = None
```

Legacy rows map to version `1`, stage `both`, status `accepted` only when both historical validation metrics exist, and string applicability/query/verification fields map to conservative tuple fields. Saving always emits the new schema version.

- [ ] **Step 4: Implement transactional operations and prompt projection.** Validate the complete proposed library before replacing `_skills`; use a temporary file and `os.replace`; expose only active, stage-matching, applicable skills to each prompt. A `persist=False` clone must never create or replace a file.

- [ ] **Step 5: Update skill learning and policy snapshots.** `OutcomeSkillLearner` may create only `candidate` skills. Promotion to `accepted` or `specialized` belongs to the Retrieval evolution evaluator after the cross-task gates pass. `snapshot_policy_skills()` serializes the complete typed records without executable content.

- [ ] **Step 6: Run focused and compatibility tests.**

Run: `python -m pytest -q tests/test_retrieval_skill_lifecycle.py tests/test_evolving_agent_skill_library.py tests/test_co_evolution.py tests/test_evolving_harness.py`

Expected: all selected tests pass, including old policy fixtures.

- [ ] **Step 7: Commit the skill lifecycle.**

```bash
git add evolving_loop/retrieval_agent/skill_library.py \
  evolving_loop/skill_learning.py evolving_loop/co_evolution.py \
  tests/test_retrieval_skill_lifecycle.py \
  tests/test_evolving_agent_skill_library.py tests/test_co_evolution.py
git commit -m "feat(retrieval): version retrieval skills"
```

---

### Task 4: Deterministic Evidence-Chain Verifier and Merge

**Files:**
- Create: `evolving_loop/retrieval_agent/verifier.py`
- Modify: `evolving_loop/retrieval_agent/agent.py`
- Test: `tests/test_retrieval_verifier.py`
- Modify: `tests/test_evolving_agent_agent.py`

**Interfaces:**
- Consumes: `ContextTask`, one untrusted stage payload, allowed skill IDs, allowed assumption IDs, and an optional verified Round 1 result.
- Produces: `verify_round_result()` and `merge_verified_rounds()`; `FinalRetrievalCard.to_legacy_result()` keeps existing Decision/candidate code working.

- [ ] **Step 1: Write adversarial verifier tests.** Cover fabricated quotes, split quotes, duplicate citations, cross-entity claims, target mismatch, future-window mismatch, magnitude without digits, invalid multipliers, incomplete chain eligibility, unknown skills, unknown assumptions, and Round 2 attempts to replace Round 1.

```python
def test_round2_cannot_erase_or_replace_round1_chain(context_task, round1):
    raw = valid_round2_payload(
        chain_id=round1.chains[0].chain_id,
        claim="Contradictory replacement text",
    )
    verified = verify_round_result(
        context_task, raw, stage="round2",
        allowed_skill_ids=(), allowed_assumption_ids=("a_trend",),
    )
    merged = merge_verified_rounds(round1, verified)
    assert merged.chains[0] == round1.chains[0]
    assert "round2_chain_identity_conflict" in merged.rejected


def test_incomplete_chain_never_becomes_numeric_eligible(context_task):
    verified = verify_round_result(
        context_task,
        payload_missing_magnitude_link(),
        stage="round1",
        allowed_skill_ids=(),
        allowed_assumption_ids=(),
    )
    assert verified.chains[0].numeric_eligible is False
```

- [ ] **Step 2: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_verifier.py tests/test_evolving_agent_agent.py`

Expected: import failure for `verifier.py` and missing chain fields.

- [ ] **Step 3: Extract and strengthen quote verification.** Move `_verified_quote_spans()` into `verifier.py`, retain its exact normalized-substring behavior, and make `agent.py` import it for backward compatibility. Verify each citation independently and retain only the exact accepted spans.

- [ ] **Step 4: Implement deterministic chain validation.** Compute chain identity from canonical entity, target, claim, citation identities, mechanism, and time window. Set `numeric_eligible=True` only when all required fields are verified and the magnitude/window rules pass. Qualitative verified chains remain in the card with explicit `missing_links`.

```python
def stable_chain_id(chain: EvidenceChain) -> str:
    payload = {
        "claim": chain.claim,
        "citations": [(item.document_id, item.exact_quote) for item in chain.citations],
        "mechanism": chain.mechanism,
        "start": chain.start_timestamp,
        "end": chain.end_timestamp,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "chain_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

- [ ] **Step 5: Implement non-destructive merge.** Round 1 wins identity conflicts. Round 2 may append new chains, citations, addressed assumptions, counterevidence, and unresolved contradictions, but cannot delete or overwrite verified Round 1 content.

- [ ] **Step 6: Add a legacy adapter.** Preserve the public `Evidence`, `EvidenceImpact`, and `RetrievalResult` names. Convert eligible chain impacts into existing typed `EvidenceImpact` rows so `_decision_candidates()` continues to apply only host-verified numeric edits.

- [ ] **Step 7: Run focused tests.**

Run: `python -m pytest -q tests/test_retrieval_verifier.py tests/test_evolving_agent_agent.py tests/test_evolving_harness.py`

Expected: all selected tests pass.

- [ ] **Step 8: Commit the immutable evidence boundary.**

```bash
git add evolving_loop/retrieval_agent/verifier.py \
  evolving_loop/retrieval_agent/agent.py \
  tests/test_retrieval_verifier.py tests/test_evolving_agent_agent.py
git commit -m "feat(retrieval): verify evidence chains"
```

---

### Task 5: Two-Stage Agent, Named Gaps, and Harness Topology

**Files:**
- Create: `evolving_loop/retrieval_agent/two_stage_agent.py`
- Create: `evolving_loop/morphology_adapter.py`
- Modify: `evolving_loop/decision_agent/agent.py`
- Modify: `evolving_loop/harness.py`
- Modify: `evolving_loop/cli.py`
- Test: `tests/test_two_stage_retrieval.py`
- Modify: `tests/test_evolving_harness.py`
- Modify: `tests/test_evolving_cli.py`

**Interfaces:**
- Consumes: a `RetrievalGenome`, task-safe target/documents, a sanitized assumption card, executed Decision candidates, and typed Skills.
- Produces: `TwoStageRetrievalAgent.run_round1()`, `TwoStageRetrievalAgent.run_round2()`, `RetrievalGap`, `DecisionResult.gaps`, `MorphologyAdapter.assumptions()`, and explicit `HarnessRuntimeConfig.retrieval_mode`.

- [ ] **Step 1: Write failing information-boundary tests.** Inspect every `FakeLLMClient` call and prove Round 1 has no assumptions/scores and Round 2 has only anonymized assumptions plus named gaps. Also prove Decision receives forecasts but its gap projection strips values and scores before Round 2.

```python
def test_two_stage_prompt_boundaries(two_stage_harness, task):
    result = two_stage_harness.run(task)
    first = json.loads(two_stage_harness.retrieval.llm.calls[0]["messages"][0]["content"])
    second = json.loads(two_stage_harness.retrieval.llm.calls[1]["messages"][0]["content"])
    assert "assumptions" not in first
    assert set(second["assumptions"][0]) == {
        "assumption_id", "kind", "claim", "failure_condition"
    }
    encoded = json.dumps(second, sort_keys=True)
    assert result.retrieval.evidence
    for forbidden in ("candidate_id", "forecast", "hindcast_smae", "hindcast_srmse", "code"):
        assert forbidden not in encoded
```

- [ ] **Step 2: Write failing topology/fallback tests.** Cover single-pass compatibility, Round 1 parse failure to Numerical fallback, no-gap skip, Round 2 failure preserving Round 1, named-gap trigger, incomplete-chain trigger, and fixed budgets.

- [ ] **Step 3: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_two_stage_retrieval.py tests/test_evolving_harness.py tests/test_evolving_cli.py`

Expected: failures identify missing two-stage APIs and CLI controls.

- [ ] **Step 4: Implement the two LLM stages.** Both methods use strict payload builders, `temperature=0.0`, schema parsing, verifier calls, and budget truncation before any result reaches the harness.

```python
class TwoStageRetrievalAgent:
    def run_round1(self, task: ContextTask) -> RetrievalRoundResult:
        payload = build_round1_payload(task, self.skills.for_stage("round1"))
        raw = self._complete(self.genome.round1_prompt, payload)
        return verify_round_result(
            task, raw, stage="round1",
            allowed_skill_ids=self._skill_ids("round1"),
            allowed_assumption_ids=(),
        )

    def run_round2(
        self,
        task: ContextTask,
        round1: RetrievalRoundResult,
        gaps: tuple[RetrievalGap, ...],
        assumptions: tuple[RetrievalAssumption, ...],
    ) -> RetrievalRoundResult:
        payload = build_round2_payload(task, round1, gaps, assumptions,
                                       self.skills.for_stage("round2"))
        raw = self._complete(self.genome.round2_prompt, payload)
        return verify_round_result(
            task, raw, stage="round2",
            allowed_skill_ids=self._skill_ids("round2"),
            allowed_assumption_ids=tuple(item.assumption_id for item in assumptions),
        )
```

- [ ] **Step 5: Implement the Morphology bridge without a Retrieval dependency cycle.** Define a small `MorphologyProvider` protocol in `morphology_adapter.py`. Its Numerical adapter maps only `assumption_id`, `kind`, `claim`, and `failure_condition` from the existing Morphology Card. Two-stage production construction requires this provider; a Morphology runtime failure is recorded and skips Round 2 while preserving verified Round 1 plus the Numerical fallback. Never synthesize Round 2 assumptions from candidate names, candidate text, scores, or forecasts.

- [ ] **Step 6: Extend Decision output with sanitized named gaps.** Add `gaps: tuple[RetrievalGap, ...] = ()` after existing default fields. Parse only known assumption IDs and enumerated gap types/priorities. Unknown IDs or free-form candidate fields are rejected; invalid gaps force `requested_more_retrieval=False` without invalidating an otherwise valid provisional selection.

- [ ] **Step 7: Add the explicit two-stage harness path.** Keep `retrieval_mode="single_pass"` as the backward-compatible default. In `two_stage`, run Coding and Morphology, Round 1, provisional Decision, optional Round 2, deterministic merge, candidate rebuild, and final Decision. Do not implement the new topology through a free-form repeated `workflow` list.

```python
@dataclass(frozen=True)
class HarnessRuntimeConfig:
    workflow: tuple[str, ...] = ("retrieve", "decide")
    retrieval_mode: str = "single_pass"
    enable_evidence_adjustments: bool = True
    max_evidence_adjustments: int = 3
    decision_aggregation: str = "last"
```

- [ ] **Step 8: Add CLI construction controls.** Add `--retrieval-mode single-pass|two-stage` and `--retrieval-release-path`. `_factory()` loads a verified release only for two-stage mode and keeps legacy policy behavior otherwise.

- [ ] **Step 9: Run focused and baseline compatibility tests.**

Run: `python -m pytest -q tests/test_two_stage_retrieval.py tests/test_evolving_harness.py tests/test_evolving_cli.py tests/test_co_evolution.py`

Expected: all selected tests pass; legacy fixtures still make one Retrieval call.

- [ ] **Step 10: Commit the executable topology.**

```bash
git add evolving_loop/retrieval_agent/two_stage_agent.py \
  evolving_loop/morphology_adapter.py evolving_loop/decision_agent/agent.py \
  evolving_loop/harness.py evolving_loop/cli.py \
  tests/test_two_stage_retrieval.py tests/test_evolving_harness.py \
  tests/test_evolving_cli.py
git commit -m "feat(retrieval): add two-stage topology"
```

---

### Task 6: Retrieval Diagnostics and Marginal Evidence Credit

**Files:**
- Create: `evolving_loop/retrieval_agent/credit.py`
- Modify: `evolving_loop/evaluation.py`
- Modify: `evolving_loop/harness.py`
- Modify: `evolving_loop/skill_learning.py`
- Modify: `evolving_loop/co_evolution.py`
- Test: `tests/test_retrieval_credit.py`
- Modify: `tests/test_evolving_agent_metrics.py`
- Modify: `tests/test_co_evolution.py`

**Interfaces:**
- Consumes: frozen inference artifacts, resolved Train labels, document labels/GT evidence held by the trusted evaluator, candidate snapshots before/after each verified complete chain, and used Skill IDs.
- Produces: `EvidenceChainCredit`, `RetrievalTaskDiagnostics`, `assign_chain_credit()`, expanded `ResolvedOutcome`, and Retrieval-specific aggregate diagnostics.

- [ ] **Step 1: Write failing label-firewall and attribution tests.** Prove labels are absent during inference, incomplete chains receive zero forecast utility, Retrieval receives contextual-oracle gain, Decision regret stays separate, joint Skill reward is not duplicated, and leave-one-Skill-out replay identifies necessary Skills.

```python
def test_chain_credit_uses_candidate_pool_gain_not_final_decision(task, harness_result):
    report = assign_chain_credit(task, harness_result)
    assert report.coding_oracle_smae - report.contextual_oracle_smae == pytest.approx(
        sum(item.marginal_smae_gain for item in report.chains)
    )
    assert report.decision_smae_regret == pytest.approx(
        report.final_smae - report.contextual_oracle_smae
    )


def test_joint_skill_credit_requires_leave_one_out(task, joint_skill_result):
    report = assign_chain_credit(task, joint_skill_result)
    chain = report.chains[0]
    assert chain.skill_credit == ()
    validated = validate_skill_necessity(task, joint_skill_result, chain.chain_id)
    assert {item.skill_id for item in validated if item.necessary} == {"window_search"}
```

- [ ] **Step 2: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_credit.py tests/test_evolving_agent_metrics.py tests/test_co_evolution.py`

Expected: missing credit types and outcome fields.

- [ ] **Step 3: Store immutable candidate-pool snapshots in `HarnessResult`.** Record the numeric-only pool, then the pool after each numeric-eligible chain in stable chain order. Snapshots contain only executed candidate IDs and forecasts; they are created before labels become available.

- [ ] **Step 4: Implement trusted post-resolution credit.** Score each snapshot with `drcik_point_metrics()`, cap task-level metrics with the same repository cap used by formal reports, and calculate marginal improvements from one snapshot to the next. Supporting recall, GT-evidence recall, distractor avoidance, quote validity, complete-chain rate, invalid count, and catastrophic count are separate fields rather than one opaque scalar.

```python
@dataclass(frozen=True)
class RetrievalTaskDiagnostics:
    supporting_recall: float
    gt_evidence_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    contextual_oracle_smae_gain: float
    contextual_oracle_srmse_gain: float
    invalid_count: int
    catastrophic_count: int
    chain_credit: tuple[EvidenceChainCredit, ...]
```

- [ ] **Step 5: Update `ResolvedOutcome` and `evaluation_diagnostics()`.** Keep legacy fields readable, add the Retrieval vector, and ensure `weakest_agent()` uses contextual-oracle gains rather than final Decision error when diagnosing Retrieval.

- [ ] **Step 6: Enforce Skill promotion gates after cross-task aggregation.** Candidate Skills remain candidates until they have at least three Train tasks from two entities, exact-quote validity `1.0`, non-worse sMAE and sRMSE, one strict gain, no added catastrophe, and necessary leave-one-out credit. Dev remains write-free and only determines whether the release containing them is accepted.

- [ ] **Step 7: Run focused tests.**

Run: `python -m pytest -q tests/test_retrieval_credit.py tests/test_evolving_agent_metrics.py tests/test_co_evolution.py tests/test_evolving_harness.py`

Expected: all selected tests pass.

- [ ] **Step 8: Commit trusted Retrieval attribution.**

```bash
git add evolving_loop/retrieval_agent/credit.py evolving_loop/evaluation.py \
  evolving_loop/harness.py evolving_loop/skill_learning.py \
  evolving_loop/co_evolution.py tests/test_retrieval_credit.py \
  tests/test_evolving_agent_metrics.py tests/test_co_evolution.py
git commit -m "feat(retrieval): attribute evidence utility"
```

---

### Task 7: Retrieval-Specific Train-Only Evolution Engine

**Files:**
- Create: `evolving_loop/retrieval_agent/evolution.py`
- Test: `tests/test_retrieval_evolution.py`

**Interfaces:**
- Consumes: Parent `RetrievalGenome`, typed Skill Library, 80 Train tasks, 20 Dev tasks, a Retrieval harness factory, mutation LLM, and frozen evaluator/split hashes.
- Produces: three scoped Children per generation, Train-only successive halving, checkpoint/resume, one final Dev comparison, accepted/rejected release result, and complete evolution trace.

- [ ] **Step 1: Write failing scope, scheduling, and acceptance tests with a fake evaluator.** Verify exact child scopes A/B/C, eight screen Train tasks, at most two promotions, entity-disjoint full-Train folds, no Dev between generations, one final Parent/Child Dev comparison, Pareto gates, P90/P95 gates, catastrophe veto, checkpoint hashes, and transient-failure retry.

```python
def test_dev_is_read_once_after_all_train_generations(fake_evaluator):
    engine = RetrievalEvolutionEngine(
        FakeLLMClient(valid_child_responses(three_generations=3)),
        fake_evaluator,
        RetrievalEvolutionConfig(generations=3, screen_tasks=8, promote=2),
    )
    engine.evolve(RetrievalGenome.seed(), train_tasks(80), dev_tasks(20))
    stages = [call.stage for call in fake_evaluator.calls]
    assert stages.count("parent_dev") == 1
    assert stages.count("child_dev") == 1
    assert stages.index("parent_dev") > max(
        index for index, value in enumerate(stages) if "train" in value
    )


def test_child_cannot_mutate_another_scope(parent, parent_evaluation):
    proposal = child_a_payload(round2_strategy="gap_first")
    child = parse_scoped_child(parent, proposal, scope="round1")
    assert child is None
```

- [ ] **Step 2: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_retrieval_evolution.py`

Expected: import failure for the missing evolution engine.

- [ ] **Step 3: Implement exact evaluation records and acceptance vector.**

```python
@dataclass(frozen=True)
class RetrievalEvaluation:
    version: str
    task_count: int
    mean_final_smae: float
    mean_final_srmse: float
    mean_contextual_oracle_smae: float
    mean_contextual_oracle_srmse: float
    p90_smae: float
    p95_smae: float
    supporting_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    invalid_count: int
    catastrophic_count: int
    task_traces: tuple[dict[str, object], ...]

    def dev_accepts(self, parent: "RetrievalEvaluation", tolerance: float) -> bool:
        return (
            self.task_count == parent.task_count
            and self.mean_contextual_oracle_smae
                <= parent.mean_contextual_oracle_smae + tolerance
            and self.mean_contextual_oracle_srmse
                <= parent.mean_contextual_oracle_srmse + tolerance
            and (
                self.mean_contextual_oracle_smae
                    < parent.mean_contextual_oracle_smae - tolerance
                or self.mean_contextual_oracle_srmse
                    < parent.mean_contextual_oracle_srmse - tolerance
            )
            and self.mean_final_smae <= parent.mean_final_smae + tolerance
            and self.mean_final_srmse <= parent.mean_final_srmse + tolerance
            and self.p90_smae <= parent.p90_smae + tolerance
            and self.p95_smae <= parent.p95_smae + tolerance
            and self.supporting_recall >= parent.supporting_recall - 0.02
            and self.distractor_avoidance >= parent.distractor_avoidance - 0.02
            and self.exact_quote_validity == 1.0
            and self.catastrophic_count <= parent.catastrophic_count
            and self.invalid_count <= parent.invalid_count
        )
```

- [ ] **Step 4: Implement three fixed mutation prompts.** Child A owns Round 1 prompt/strategy/document budget; B owns chain/citation budgets, counterevidence, target/time requirements, and both chain instructions; C owns Round 2 prompt/strategy/trigger and Round 2 Skills. Reject any proposal that changes an unowned field after parsing the complete candidate.

- [ ] **Step 5: Implement Train-only successive halving.** Evaluate Parent and all three Children on the same eight Train cases. Prune invalid/dominated Children, promote at most two, complete the remaining Train cases, and compare entity-disjoint fold vectors. Cache inference by task ID plus Genome/Skill/verifier hashes.

- [ ] **Step 6: Implement generation and final Dev logic.** The provisional Parent may change after each Train generation. After the last generation, evaluate the original Parent and final Train winner exactly once on Dev with all libraries `persist=False` and every writer/evolver disabled. Reject on any acceptance gate failure. Assert that no Dev metric, task trace, or failure example is ever included in a later mutation request.

- [ ] **Step 7: Implement checkpoint/resume.** Bind schema version, dataset split hash, verifier/evaluator hashes, metric cap, task completion, Parent fingerprint, Child fingerprints, and random seed. A mismatched checkpoint aborts instead of resuming against different science.

- [ ] **Step 8: Run evolution-engine tests.**

Run: `python -m pytest -q tests/test_retrieval_evolution.py tests/test_retrieval_policy.py tests/test_retrieval_skill_lifecycle.py`

Expected: all selected tests pass.

- [ ] **Step 9: Commit the Retrieval Meta-Harness.**

```bash
git add evolving_loop/retrieval_agent/evolution.py tests/test_retrieval_evolution.py
git commit -m "feat(retrieval): evolve typed retrieval genomes"
```

---

### Task 8: CLI, Frozen Inference, and Reproducible Runner

**Files:**
- Modify: `evolving_loop/cli.py`
- Modify: `evolving_loop/frozen_inference.py`
- Create: `scripts/run_retrieval_evolution.sh`
- Modify: `tests/test_evolving_cli.py`
- Test: `tests/test_retrieval_frozen_inference.py`
- Test: `tests/test_retrieval_runner.py`

**Interfaces:**
- Consumes: v000 or accepted Retrieval release, frozen `80/20/99` manifest, explicit LLM/backend settings, and output paths.
- Produces: `--evolution retrieval`, `--inference retrieval`, a resumable shell command, final release/report artifacts, and zero-write frozen output.

- [ ] **Step 1: Write failing parser, dispatch, shell dry-run, and frozen-write tests.** Verify root and legacy CLI forms, exact 80/20 counts, no Public Regression IDs in the evolution trace, no hidden scoring, no Skill/Genome mutation calls in frozen mode, and report inclusion of both Retrieval stages.

```python
def test_retrieval_evolution_cli_has_frozen_defaults():
    args = build_parser().parse_args([
        "--evolution", "retrieval",
        "--tasks-file", "external/Dr-CiK/sample/tasks.jsonl",
        "--split-manifest", "splits/drcik_public_80_20_99_v1.json",
    ])
    assert args.screen_train_tasks == 8
    assert args.screen_promote == 2
    assert args.retrieval_mode == "two-stage"


def test_hidden_retrieval_inference_is_write_free(tmp_path, hidden_tasks, factory):
    before = directory_fingerprint(tmp_path)
    run_frozen_inference(
        accepted_policy(), hidden_tasks, factory,
        output_dir=tmp_path / "output", score_public=False,
        artifact_kind="retrieval",
    )
    assert library_write_paths(tmp_path) == ()
    assert mutation_call_count(factory) == 0
    assert before == directory_fingerprint(tmp_path, exclude=("output",))
```

- [ ] **Step 2: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_evolving_cli.py tests/test_retrieval_frozen_inference.py tests/test_retrieval_runner.py`

Expected: parser and dispatch failures for the new mode.

- [ ] **Step 3: Add explicit CLI modes.** Extend `EVOLUTION_CHOICES` and `INFERENCE_CHOICES` with `retrieval`; add `--retrieval-mode`, `--retrieval-release-path`, `--screen-train-tasks 8`, `--screen-promote 2`, and immutable split/hash inputs. The Retrieval evolution dispatcher selects exactly the manifest Train and Dev IDs and never loads Public Regression records into its task list.

- [ ] **Step 4: Save accepted artifacts atomically.** On acceptance, write `releases/vNNN`; embed its payload in the saved `HarnessPolicy`; write Train/Dev summaries, scope-specific Child changelogs, rejection reasons, and hashes. On rejection, keep the Parent release and write traces only under `runs/`.

- [ ] **Step 5: Extend frozen reports.** Add Round 1/2 chains, rejected citations, gaps, assumption stances, Skill IDs, release hash, and `labels_accessed` to each run report. Do not add GT evidence or document roles to inference reports.

- [ ] **Step 6: Add the executable shell runner.** It uses environment variables only for runtime choices and emits the exact command before execution.

```bash
#!/usr/bin/env bash
set -euo pipefail

TASKS_FILE=${TASKS_FILE:-external/Dr-CiK/sample/tasks.jsonl}
SPLIT_FILE=${SPLIT_FILE:-splits/drcik_public_80_20_99_v1.json}
MODEL=${MODEL:-gpt-5.4}
EFFORT=${EFFORT:-high}
RUN_DIR=${RUN_DIR:-runs/retrieval_evolution/formal_80_20}

python -m evolving_loop.cli \
  --evolution retrieval \
  --tasks-file "$TASKS_FILE" \
  --split-manifest "$SPLIT_FILE" \
  --retrieval-mode two-stage \
  --llm-backend codex \
  --codex-model "$MODEL" \
  --codex-reasoning-effort "$EFFORT" \
  --generations 3 \
  --screen-train-tasks 8 \
  --screen-promote 2 \
  --checkpoint-path "$RUN_DIR/checkpoint.json" \
  --progress-path "$RUN_DIR/progress.jsonl" \
  --policy-path "$RUN_DIR/best_policy.json" \
  --trace-path "$RUN_DIR/evolution_trace.json"
```

- [ ] **Step 7: Run focused checks.**

Run: `python -m pytest -q tests/test_evolving_cli.py tests/test_retrieval_frozen_inference.py tests/test_retrieval_runner.py && bash -n scripts/run_retrieval_evolution.sh`

Expected: all tests and shell syntax checks pass.

- [ ] **Step 8: Commit deployment surfaces.**

```bash
git add evolving_loop/cli.py evolving_loop/frozen_inference.py \
  scripts/run_retrieval_evolution.sh tests/test_evolving_cli.py \
  tests/test_retrieval_frozen_inference.py tests/test_retrieval_runner.py
git commit -m "feat(retrieval): expose frozen evolution CLI"
```

---

### Task 9: Coordinate-Isolated Multi-Agent Evolution

**Files:**
- Create: `evolving_loop/coordinate_evolution.py`
- Modify: `evolving_loop/co_evolution.py`
- Modify: `evolving_loop/cli.py`
- Test: `tests/test_coordinate_evolution.py`
- Modify: `tests/test_co_evolution.py`

**Interfaces:**
- Consumes: accepted Numerical/Morphology policy, accepted Retrieval release, Decision policy, module diagnostics, and an ordered phase configuration.
- Produces: `CoordinateEvolutionController`, Retrieval-first/Decision-second phases, and alternating one-module-per-generation co-evolution.

- [ ] **Step 1: Write failing ownership and sequencing tests.** Prove Retrieval evolution freezes Coding/Morphology and Decision, Decision evolution freezes Retrieval, alternating mode changes exactly one principal module, and an unchanged/non-improving phase cannot alter the accepted bundle.

```python
def test_coordinate_cycle_mutates_one_principal_module_at_a_time(controller):
    accepted, trace = controller.run(seed_bundle(), train_tasks(), dev_tasks())
    assert [step.target for step in trace] == ["retrieval", "decision", "retrieval"]
    for step in trace:
        changed = set(step.child_fingerprints) - set(step.parent_fingerprints)
        assert changed <= {step.target}
        assert len(changed) <= 1
    assert accepted.public_test_accessed is False
```

- [ ] **Step 2: Run focused tests and verify RED.**

Run: `python -m pytest -q tests/test_coordinate_evolution.py tests/test_co_evolution.py`

Expected: missing coordinate controller and policy embedding behavior.

- [ ] **Step 3: Add the accepted Retrieval release to `HarnessPolicy`.** Store a canonical typed payload and fingerprint, not a mutable filesystem reference. Keep old policy JSON parseable by supplying the v000 seed only when `retrieval_mode="two_stage"`; old single-pass policies remain unchanged.

- [ ] **Step 4: Implement coordinate phases.** The controller first invokes `RetrievalEvolutionEngine` with other module fingerprints frozen, then invokes the existing Decision-targeted engine with the accepted Retrieval payload frozen. Alternating mode chooses the weakest module from separate Retrieval gain and Decision regret diagnostics and dispatches exactly one scoped engine.

- [ ] **Step 5: Enforce acceptance ownership.** Each phase returns a new immutable bundle only after its own Train/Dev gates. Rejected phases return the byte-identical Parent bundle. The controller may diagnose another module but never mutate it in the same generation.

- [ ] **Step 6: Add CLI phase selection.** Add `--coordinate-phase retrieval|decision|alternate` under the existing Genome evolution command; require an accepted Retrieval release before `decision` or `alternate` phases.

- [ ] **Step 7: Run focused tests.**

Run: `python -m pytest -q tests/test_coordinate_evolution.py tests/test_co_evolution.py tests/test_evolving_cli.py`

Expected: all selected tests pass.

- [ ] **Step 8: Commit coordinate evolution.**

```bash
git add evolving_loop/coordinate_evolution.py evolving_loop/co_evolution.py \
  evolving_loop/cli.py tests/test_coordinate_evolution.py \
  tests/test_co_evolution.py tests/test_evolving_cli.py
git commit -m "feat(evolution): coordinate retrieval and decision"
```

---

### Task 10: End-to-End Safety, Compatibility, and Documentation

**Files:**
- Create: `tests/test_retrieval_e2e.py`
- Modify: `README.md`
- Modify: `docs/SELF_EVOLUTION_FRAMEWORK.md`
- Modify: `docs/forecasting_pipeline_full_2026-08-27_en.html`
- Modify: `docs/forecasting_pipeline_full_2026-08-26.html`

**Interfaces:**
- Consumes: the complete implementation and fake deterministic tasks/LLMs.
- Produces: an end-to-end proof of the two-stage boundary, backward-compatible baseline, reproducible commands, and an accurate architecture/result-status explanation.

- [ ] **Step 1: Write one complete fake-LLM end-to-end test.** The fixture must include an assumption-blind Round 1, a provisional named gap, a Round 2 counterevidence chain, final Decision, delayed public scoring, and a candidate Skill that remains unpromoted until cross-task evidence exists.

- [ ] **Step 2: Write frozen and adversarial end-to-end tests.** Cover hidden unlabeled inference, prompt injection, malformed JSON, transient LLM failure, budget exhaustion, restart/resume, read-only Dev, no Public Regression access, and a legacy single-pass policy.

- [ ] **Step 3: Run the full Retrieval-focused suite.**

Run:

```bash
python -m pytest -q \
  tests/test_retrieval_schemas.py \
  tests/test_retrieval_policy.py \
  tests/test_retrieval_skill_lifecycle.py \
  tests/test_retrieval_verifier.py \
  tests/test_two_stage_retrieval.py \
  tests/test_retrieval_credit.py \
  tests/test_retrieval_evolution.py \
  tests/test_retrieval_frozen_inference.py \
  tests/test_retrieval_runner.py \
  tests/test_coordinate_evolution.py \
  tests/test_retrieval_e2e.py
```

Expected: all Retrieval-focused tests pass.

- [ ] **Step 4: Update English and Chinese documentation.** State exactly what is implemented, distinguish single-pass baseline from two-stage Retrieval, show input/output schemas, explain three Child scopes and 80/20 acceptance, and label all empirical results as not yet run until a real experiment exists.

- [ ] **Step 5: Run repository-wide verification.**

Run:

```bash
python -m pytest -q
python -m compileall -q common evolving_loop numerical_agent
bash -n scripts/run_retrieval_evolution.sh
git diff --check
```

Expected: pytest reports zero failures; compile, shell, and whitespace checks exit zero.

- [ ] **Step 6: Run a one-task offline smoke with fake LLM responses.** This validates assembly without spending model tokens or opening Dev/Public Regression labels.

Run: `python -m pytest -q tests/test_retrieval_e2e.py::test_fake_two_stage_smoke`

Expected: one passing test and zero filesystem writes outside its temporary directory.

- [ ] **Step 7: Commit documentation and end-to-end verification.**

```bash
git add tests/test_retrieval_e2e.py README.md docs/SELF_EVOLUTION_FRAMEWORK.md \
  docs/forecasting_pipeline_full_2026-08-27_en.html \
  docs/forecasting_pipeline_full_2026-08-26.html
git commit -m "docs: explain retrieval self-evolution"
```

## Execution Checkpoints

After Tasks 1–4, review the immutable contracts before any LLM topology changes are accepted.

After Tasks 5–6, run the two-stage fake task and inspect both prompt payloads plus the trusted credit trace.

After Tasks 7–9, review mutation-scope enforcement, Train-only scheduling, one-time Dev access, checkpoint hashes, and frozen writes before enabling a real Codex run.

After Task 10, do not start the 80/20 experiment automatically. First freeze the implementation commit, seed release hash, model name/reasoning effort, token/task budgets, split hash, verifier hash, metric cap, and output directory in a run manifest.
