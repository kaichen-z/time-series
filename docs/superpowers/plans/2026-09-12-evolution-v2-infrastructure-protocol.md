# Evolution V2 Infrastructure Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute five typed/versioned L1 proposal kinds with compatibility evaluation, automatic accept/reject, old-version replay, resume, and unified CLI smoke.

**Architecture:** Add a data-only protocol manifest and fixed Host runtime registry beside the immutable L0 Kernel. Compare old/proposed runtimes through the existing complete cooperative pipeline and publish a separate L1 release, preserving Bundle/L0 identities.

**Tech Stack:** Python 3.11+, frozen dataclasses, existing canonical JSON/SHA-256/store/budget helpers, Project 3 `CooperativePipelineAdapter`, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-12-evolution-v2-infrastructure-protocol-design.md`

## Global Constraints

- Implementation plus focused verification targets 120 minutes across five tasks.
- Candidate kinds are exactly `backbone`, `loader`, `verifier_strategy`, `diagnostic_metric`, and `schema_migration`.
- Candidates contain typed/versioned data only; Host code resolves allowlisted implementation IDs and candidates cannot supply imports, source, callbacks, paths, or L0 policy overrides.
- `KernelProtocolCommitment`, `EvolutionBundleV2`, canonical task materialization, primary scoring, split commitments, label firewall, promotion rules, and existing mutation ownership remain unchanged.
- The committed compatibility corpus is four Train tasks and one disjoint Dev task, plus two frozen archive Bundles; all Public membership is rejected before runtime or proposals.
- The proposal sequence is committed before evaluation; proposal inputs contain no Dev values, future labels, task IDs, or Public results.
- Acceptance requires compatibility and finite complete-pipeline Train/Dev nonregression within `1e-12`; strict improvement is not required for infrastructure-only changes.
- Persist at closed proposal boundaries; complete resume is read-only and returns byte-identical semantic completion bytes.
- Smoke is offline, deterministic, and limited to 120 seconds; no new dependencies or model downloads.
- Exclude real large-model runs, 80/20 or four-hour runs, OS sandboxing, extreme migration matrices, per-forecast persistence, and whole-repository regression gates.

## Execution and ownership

Read the linked spec before execution. Project 4 must finish its focused gate
before Task 1 begins; documentation planning may overlap. Existing P3/P4 source
is consumed read-only; Task 5 alone edits the shared `evolving_loop/v2/cli.py`
after their implementation finishes. Keep all L1 behavior in the new package.
This is not authorization to modify Kernel/Bundle ownership.

Five task budgets are 15, 30, 30, 25, and 20 minutes. Use one implementer and one
task review at a time, with bounded review of spec and quality. The controller
may prepare the next brief while the implementer works. Implementers must not
spawn agents. Capture BASE before each dispatch; report commands and actual
RED/GREEN evidence to the matching `task-N-report.md`. Reviewers consume brief,
report, and generated diff package; they need not repeat passing tests.

Files by responsibility: `contracts.py` defines canonical artifacts;
`runtime.py` resolves executable adapters; `compatibility.py` owns corpus checks
and real scoring; `runner.py` owns fixed authority/persistence/resume;
`cli.py` owns fixture/config/entry dispatch; `__init__.py` exports stable APIs.
The only other edits are six focused test modules, one smoke config, one user
guide, and one README link.

---

### Task 1: Typed protocol artifacts and exact component ownership

**Budget:** 15 minutes.

**Files:** Create `evolving_loop/v2/protocol/__init__.py`, `evolving_loop/v2/protocol/contracts.py`, `tests/test_evolution_v2_protocol_contracts.py`.

**Interfaces:** Consume `canonical_v2_bytes`, `fingerprint_payload`, `require_sha256` from `evolving_loop.v2.contracts`. Produce the classes below and `ProtocolProposalV2.to_child(parent: InfrastructureProtocolV2) -> InfrastructureProtocolV2`. Every contract implements strict `from_payload`, `to_payload`, `canonical_bytes`, and `fingerprint`.

Exact data fields:

```python
KINDS = ("backbone", "loader", "verifier_strategy", "diagnostic_metric", "schema_migration")
# ProtocolComponentV2:
# kind, implementation_id, implementation_version, artifact_schema_version
# InfrastructureProtocolV2:
# schema_version, protocol_version, parent_protocol_sha256,
# l0_commitment_sha256, components
# ProtocolProposalV2:
# schema_version, parent_protocol_sha256, kind, replacement,
# compatibility_corpus_sha256
# ProtocolReleaseV2:
# schema_version, protocol_sha256, l0_commitment_sha256,
# evidence_sha256, frozen_bundle_sha256, runtime_fingerprint
```

- [ ] Write failing tests, using local constructors for the five components and a version-1 seed protocol:

```python
def test_proposal_versions_only_one_l1_component():
    seed = seed_protocol()
    component = ProtocolComponentV2("backbone", "history_mean", 1, 1)
    proposal = ProtocolProposalV2(1, seed.fingerprint(), "backbone", component, "c" * 64)
    child = proposal.to_child(seed)
    assert child.protocol_version == 2
    assert child.parent_protocol_sha256 == seed.fingerprint()
    assert child.l0_commitment_sha256 == seed.l0_commitment_sha256
    assert child.components[1:] == seed.components[1:]
    assert InfrastructureProtocolV2.from_payload(child.to_payload()) == child

def test_mismatched_kind_and_l0_override_are_rejected():
    seed = seed_protocol()
    proposal = proposal_payload(seed, kind="loader")
    proposal["replacement"]["kind"] = "backbone"
    with pytest.raises(ValueError):
        ProtocolProposalV2.from_payload(proposal)
    payload = seed.to_payload()
    payload["metric_policy"] = "0" * 64
    with pytest.raises(ValueError):
        InfrastructureProtocolV2.from_payload(payload)
```

- [ ] Run `pytest -q tests/test_evolution_v2_protocol_contracts.py`; confirm missing API failure.
- [ ] Implement frozen dataclasses with exact-key validation, positive integer versions excluding bool, exact component order/uniqueness, and known kind tags. `to_child` checks the parent SHA, matching kind, changed replacement, and unchanged L0. Use the following construction rule:

```python
components = tuple(replacement if item.kind == kind else item for item in parent.components)
return InfrastructureProtocolV2(
    1, parent.protocol_version + 1, parent.fingerprint(),
    parent.l0_commitment_sha256, components,
)
```

- [ ] Run the same file; require GREEN including unsupported schema/version and duplicate-kind cases.
- [ ] Commit only Task 1 files: `git commit -m "feat(protocol): add versioned L1 contracts"`.

---

### Task 2: Five executable adapters and deterministic migration

**Budget:** 30 minutes.

**Files:** Create `evolving_loop/v2/protocol/runtime.py`, `tests/test_evolution_v2_protocol_runtime.py`; modify package `__init__.py` exports.

**Interfaces:** Consume Task 1 contracts and P3 `CooperativePipelineAdapter`. Produce `ProtocolRuntimeRegistry`, `ProtocolRuntime`, `migrate_envelope(payload: dict, target_version: int) -> dict`. Registry method `resolve(protocol: InfrastructureProtocolV2, host_inputs: ProtocolHostInputs) -> ProtocolRuntime`; runtime methods `load_tasks() -> tuple[ContextTask, ...]`, `evaluate(bundle, tasks, stage) -> PackageEvaluation`, `verify(evidence: dict) -> bool`, `diagnostics(history: tuple[float, ...], forecast: tuple[float, ...]) -> dict[str, float]`. Define `ProtocolHostInputs` here as a frozen Host-only dataclass holding raw fixture records, the P3 catalog, frozen numerical artifacts, verified retrieval/decision factories, and committed task metadata; it is never candidate input.

Implementation IDs and behavior:

```python
# backbone: last_value/1, history_mean/1
# loader: canonical_json/1, alternate_history_json/1, changed_history_json/1
# verifier_strategy: exact_support/1, deduplicate_support/1
# diagnostic_metric: forecast_spread/1, absolute_movement/1
# schema_migration: identity_envelope/1 (artifact schema 1), envelope_v2/1 (schema 2)
def forecast_last(history, horizon):
    return (history[-1],) * horizon
def forecast_mean(history, horizon):
    return (sum(history) / len(history),) * horizon
```

`changed_history_json` is an allowlisted negative fixture which deliberately
changes history and must fail compatibility. Keep it test/smoke scoped.

- [ ] Write failing observable-behavior tests:

```python
def test_backbone_versions_change_real_pipeline_forecasts(runtime_case):
    old, new = runtime_case.backbone_pair(history=(0.0, 2.0), future=(1.0,))
    assert old.forecast == (2.0,)
    assert new.forecast == (1.0,)
    assert new.evaluation.mean_smae < old.evaluation.mean_smae

def test_alternate_loader_preserves_canonical_task_identity(runtime_case):
    a, b = runtime_case.loader_pair()
    assert runtime_case.task_hashes(a) == runtime_case.task_hashes(b)

def test_migration_preserves_embedded_identity_and_is_idempotent():
    content = {"bundle": "f" * 64}
    old = {"schema_version": 1, "artifact": content, "artifact_sha256": fingerprint_payload(content)}
    new = migrate_envelope(old, 2)
    assert new == {"schema_version": 2, "content": content, "content_sha256": old["artifact_sha256"]}
    assert migrate_envelope(new, 2) == new
```

- [ ] Run `pytest -q tests/test_evolution_v2_protocol_runtime.py`; confirm RED.
- [ ] Implement a fixed mapping `(kind, implementation_id, implementation_version)` to Host adapters; reject unresolved entries before evaluation. Loader maps alternate history key to the canonical record, then uses existing Host materialization. Build a correctly hashed derived Numerical registry for each protocol and reuse P3 artifact/factory resolution. Call the unchanged `PackagePipelineEvaluator._evaluate_components` with `candidate_sha256=fingerprint_payload({"protocol_sha256": protocol.fingerprint(), "bundle_sha256": bundle.fingerprint()})`, the derived registry, and the same verified Retrieval/Decision factories and primary metric cap. Do not substitute altered registry bytes under an old SHA or bypass P3 catalog identity checks. Archived Bundles remain frozen recipes. Baseline Host verification always executes before strategy filtering. Diagnostic implementations return only finite extra fields.
- [ ] Implement migration using the exact v1/v2 key sets above: validate SHA before conversion; v1-to-v2 and v2-to-v2 succeed, v2-to-v1 and unknown versions reject. Existing referenced artifact payloads and their SHA stay unchanged. Return new dictionaries instead of mutating the input.
- [ ] Add verifier tests for valid, fabricated document, nonexistent support, and evaluator-only evidence; add diagnostic tests proving primary evaluation fields are unchanged. Run this test file and `tests/test_evolution_v2_cooperative_pipeline.py` once; require GREEN.
- [ ] Commit Task 2 files: `git commit -m "feat(protocol): execute infrastructure adapters"`.

---

### Task 3: Compatibility corpus, archive re-evaluation, and sealed gates

**Budget:** 30 minutes.

**Files:** Create `evolving_loop/v2/protocol/compatibility.py`, `tests/test_evolution_v2_protocol_compatibility.py`; modify package exports.

**Interfaces:** Consume Task 1/2 APIs. Define `CompatibilityCorpusV2` with exact fields `schema_version`, `l0_commitment_sha256`, `train_task_sha256s`, `dev_task_sha256s`, `archive_bundle_sha256s`, `artifact_sha256s`, `verifier_fixture_sha256`; require 4 Train, 1 Dev, 2 Bundles, disjoint task identities, and an explicit verified non-Public split manifest. Define `CompatibilityEvidenceV2` with exact fields `schema_version`, `old_protocol_sha256`, `proposed_protocol_sha256`, `corpus_sha256`, `runtime_fingerprint`, `checks`, `train_rows`, `migration_mapping`, `sealed_dev_sha256`. `checks` has exactly `l0`, `tasks`, `firewall`, `artifacts`, `verifier`, `train`, `dev` boolean entries. `train_rows` contain protocol/Bundle SHA and canonical aggregate metrics only. Produce `check_compatibility(old, proposed, corpus, registry, *, host_inputs, sealed_store) -> CompatibilityEvidenceV2` and `decide_protocol(evidence) -> ProtocolDecisionV2`, whose fields are `schema_version`, `decision`, and `reason_codes`.

- [ ] Write failing gate tests with a spy around real pipeline evaluations:

```python
def test_changed_task_hash_rejects_before_dev(compatibility_case):
    evidence = compatibility_case.check("changed_history_json")
    decision = decide_protocol(evidence)
    assert decision.decision == "reject"
    assert "task_hash_mismatch" in decision.reason_codes
    assert compatibility_case.dev_calls == 0

def test_each_archive_bundle_is_scored_under_both_protocols(compatibility_case):
    evidence = compatibility_case.check("forecast_spread")
    assert decide_protocol(evidence).decision == "accept"
    assert len(evidence.train_rows) == 4
    assert len({(r["protocol_sha256"], r["bundle_sha256"]) for r in evidence.train_rows}) == 4
    assert compatibility_case.old_score_cache_misses_for_new_protocol == 2

def test_public_membership_rejected_before_resolving_runtime(compatibility_case):
    with pytest.raises(ValueError, match="Public"):
        compatibility_case.with_public_membership().check("history_mean")
    assert compatibility_case.runtime_resolutions == 0
```

- [ ] Run `pytest -q tests/test_evolution_v2_protocol_compatibility.py`; confirm RED.
- [ ] Implement the seven ordered checks from the spec, stopping before inference on invalid commitments and before Dev on compatibility/Train failure. Compare each archive Bundle with itself across protocols; never carry old scores into a proposed protocol's cache. Use cache keys `(protocol_sha, runtime_sha, corpus_sha, bundle_sha, stage)` and fixed seed from the input commitment. Use this unchanged primary gate:

```python
def nonregressing(old, new):
    return (
        new.coverage == 1.0
        and new.invalid_count <= old.invalid_count
        and new.catastrophic_count <= old.catastrophic_count
        and new.mean_smae <= old.mean_smae + 1e-12
        and new.mean_srmse <= old.mean_srmse + 1e-12
    )
```

Explicitly reject nonfinite aggregates before the comparison. Existing
`PackageEvaluation.mean_smae` and `.mean_srmse` are capped primary means;
`.mean_smae_raw` and `.mean_srmse_raw` are diagnostics and never gate acceptance.

- [ ] Persist Dev aggregates only in `sealed_store`; evidence contains their SHA and boolean gate. Stable reasons are `l0_mismatch`, `task_hash_mismatch`, `label_boundary`, `artifact_migration`, `verifier_failure`, `train_regression`, `dev_regression`, or `compatible`. The decision is accept only when every check passes. Preserve primary scoring for diagnostic changes.
- [ ] Add focused tests for artifact SHA mismatch, verifier rejection, Dev regression, and migration identity preservation; run this file plus runtime tests once, require GREEN.
- [ ] Commit Task 3 files: `git commit -m "feat(protocol): gate compatibility migrations"`.

---

### Task 4: Automatic release publication, exact resume, frozen handoff

**Budget:** 25 minutes.

**Files:** Create `evolving_loop/v2/protocol/runner.py`, `tests/test_evolution_v2_protocol_runner.py`; modify package exports.

**Interfaces:** Consume Tasks 1–3 and P1 `BudgetPlan`/`BudgetLedger`/`ResourceUse` plus canonical store helpers. Produce `run_protocol_evolution(output_dir: Path, config: dict, input_manifest: dict, registry: ProtocolRuntimeRegistry, *, host_inputs: ProtocolHostInputs, stop_after: int | None = None) -> dict`; `freeze_protocol_handoff(release: ProtocolReleaseV2, bundle: EvolutionBundleV2, *, resolve) -> dict`. Repeating the runner verifies/resumes automatically. `resolve(sha: str) -> dict` reads exact canonical referenced objects.

- [ ] Write failing lifecycle tests:

```python
def test_rejection_keeps_active_pointer_bytes(protocol_run_case):
    before = protocol_run_case.active_bytes()
    result = protocol_run_case.run_one("changed_history_json")
    assert result["rejected"] == 1
    assert protocol_run_case.active_bytes() == before

def test_closed_proposal_resume_matches_completion(protocol_run_case):
    full = protocol_run_case.run("full")
    protocol_run_case.run("resumed", stop_after=2)
    resumed = protocol_run_case.run("resumed")
    assert resumed == full
    assert protocol_run_case.completion_bytes("resumed") == protocol_run_case.completion_bytes("full")
    before = protocol_run_case.snapshot("resumed")
    assert protocol_run_case.run("resumed") == full
    assert protocol_run_case.snapshot("resumed") == before

def test_old_protocol_remains_runnable_after_activation(protocol_run_case):
    before = protocol_run_case.replay_seed_protocol()
    protocol_run_case.run("full")
    assert protocol_run_case.replay_seed_protocol() == before
```

- [ ] Run `pytest -q tests/test_evolution_v2_protocol_runner.py`; confirm RED.
- [ ] Implement precommitted sequential replacement templates and Host-derived parent linkage. Each step persists proposal, evaluates compatibility, reads evidence back, decides, publishes only accepted releases, records progress, and atomically checkpoints. Host publication verifies all stored evidence references and L0/corpus/runtime consistency; a candidate never returns an active pointer. Preserve previous protocol/component objects and envelope originals.

```python
proposal = ProtocolProposalV2(1, active.fingerprint(), replacement.kind, replacement, corpus.fingerprint())
child = proposal.to_child(active)
evidence = check_compatibility(active, child, corpus, registry, host_inputs=host_inputs, sealed_store=sealed_store)
decision = decide_protocol(evidence)
# Persist and reread evidence before creating ProtocolReleaseV2.
# Publish active_protocol.json only if decision.decision == "accept".
# Append progress then seal the closed-step checkpoint.
```

- [ ] Bind checkpoint to exact input/config/runtime/L0 hashes, completed object SHAs, next index, active release SHA, progress prefix hash, and budget counters. Resume mismatches raise `ValueError` before adapters run. Closed-boundary interruption requires no partially completed evaluation reconstruction; report incomplete writes explicitly. Final semantic result excludes elapsed timing.
- [ ] Implement frozen handoff with exact fields `schema_version`, `bundle_sha256`, `l0_commitment_sha256`, `protocol_sha256`, `release_sha256`, `acceptance_evidence_sha256`, `runtime_fingerprint`, `public_test_accessed`. Resolve references; require non-null Bundle acceptance evidence and matching Bundle/release L0. `acceptance_evidence_sha256` refers to protocol release evidence; Bundle evidence remains pinned through the Bundle. Reject an unaccepted proposal SHA and do not execute Public tasks.
- [ ] Run runner tests including changed manifest, corrupted evidence SHA, frozen handoff mismatch, and old replay; require GREEN. Commit: `git commit -m "feat(protocol): publish resumable L1 releases"`.

---

### Task 5: Unified CLI smoke, compact fixture, documentation

**Budget:** 20 minutes.

**Files:** Create `evolving_loop/v2/protocol/cli.py`, `configs/evolution_v2/protocol/smoke.json`, `tests/test_evolution_v2_protocol_cli.py`, `docs/evolution-v2-infrastructure-protocol.md`; modify `evolving_loop/v2/cli.py`, package exports, and one README link. Do not edit source/cooperative package files.

**Interfaces:** Produce `add_protocol_parsers(subparsers) -> None`, `dispatch_protocol(args) -> dict`, `make_smoke_inputs(output_dir: Path) -> Path`. Add unified `protocol-evolve` and `protocol-make-smoke-inputs` commands with the flags specified in the spec. Strict smoke config fields are `schema_version=1`, `profile="smoke"`, `seed=17`, `max_proposals=6`, `hard_limit_seconds=120`.

Input manifest fields are exactly `schema_version`, `l0_commitment`,
`runtime_fingerprint`, `corpus`, `seed_protocol`, `host_input_files`,
`frozen_bundle_sha256`, and `replacement_templates`. `host_input_files` is a
Host-only tuple of `(role, relative_path, sha256)` records whose bytes are
verified; roles cover tasks, catalog/referenced artifacts, and runtime inputs.
Require four Train/one Dev membership, two accepted archived Bundles, and
closure of all referenced artifacts. The builder may use P3 fixture construction
but owns its generated files; no P3 fixture overwrite.

- [ ] Write failing unified parser and complete smoke tests:

```python
def test_protocol_command_uses_unified_entrypoint():
    args = build_parser().parse_args(["protocol-evolve", "--config", "c.json", "--input-manifest", "i.json", "--output-dir", "run"])
    assert args.command == "protocol-evolve"

def test_offline_smoke_runs_all_five_kinds_and_one_rejection(cli_case):
    result = cli_case.run_fresh()
    assert result["status"] == "protocol_evolution_complete"
    assert result["accepted"] == 5
    assert result["rejected"] == 1
    assert result["public_test_accessed"] is False
    assert cli_case.attempted_kinds() == set(KINDS)
    assert cli_case.verified_frozen_handoff()
```

- [ ] Run `pytest -q tests/test_evolution_v2_protocol_cli.py`; confirm RED.
- [ ] Implement parser delegation without changing existing dispatch branches. Fixture order is backbone `history_mean`, loader `alternate_history_json`, verifier `deduplicate_support`, diagnostic `absolute_movement`, migration `envelope_v2`, then rejected loader `changed_history_json`. Seed components use the first implementation in Task 2 for each kind. Build histories whose mean equals the future and whose last value differs, ensuring real pipeline improvement for the backbone while other changes preserve primary scores. Use two accepted frozen Bundles and deterministic in-process Retrieval/Decision factories; no synthetic evaluation scores.
- [ ] Document commands, version meaning, fixed L0 boundary, migration envelope scope, accepted/rejected artifacts, frozen handoff, resume, and exclusions. State that existing Public command remains validation-only.
- [ ] Run this single focused gate:

```bash
pytest -q tests/test_evolution_v2_protocol_contracts.py tests/test_evolution_v2_protocol_runtime.py tests/test_evolution_v2_protocol_compatibility.py tests/test_evolution_v2_protocol_runner.py tests/test_evolution_v2_protocol_cli.py tests/test_evolution_v2_cooperative_pipeline.py tests/test_evolution_v2_bundle.py
pytest -q tests/test_evolution_v2_cli.py -k 'parser or public or fake'
```

- [ ] Run one real unified offline smoke in fresh directories:

```bash
protocol_inputs=$(mktemp -d /tmp/evolution-v2-protocol-inputs.XXXXXX)
protocol_run=$(mktemp -d /tmp/evolution-v2-protocol-run.XXXXXX)
python -m evolving_loop.v2 protocol-make-smoke-inputs --output-dir "$protocol_inputs"
python -m evolving_loop.v2 protocol-evolve --config configs/evolution_v2/protocol/smoke.json --input-manifest "$protocol_inputs/input_manifest.json" --output-dir "$protocol_run"
```

Expect five acceptances, one compatibility rejection, terminal status, and
`public_test_accessed=false`. Resume equality is already covered by focused
tests; do not repeat full-suite or long experimental gates.

- [ ] Commit exact Task 5 files: `git commit -m "feat(cli): expose protocol evolution smoke"`.

## Final handoff

Completion requires all five task reports and focused task review verdicts,
the single smoke result, and a final scoped review of P5 changes. Summarize
actual commands/results and any deferred concern; do not present this plan as
implemented work. Preserve the SDD ledger until execution is complete and its
review record has been captured. Budget total: 120 minutes; optional experiments
and Public scoring are outside this estimate.
