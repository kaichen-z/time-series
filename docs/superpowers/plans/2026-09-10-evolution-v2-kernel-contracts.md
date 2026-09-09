# Evolution V2 Kernel and Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the immutable Evolution V2 kernel foundation: strict content-addressed contracts, complete Bundle identity and mutation-scope enforcement, append-only archives, resumable four-hour budget accounting, automatic promotion/rollback, deterministic fake fixtures, and a separate runnable V2 CLI without changing legacy evolution behavior.

**Architecture:** Add an isolated `evolving_loop.v2` package. Candidate-side code may construct data-only artifacts, but only the V2 kernel may validate scope, seal acceptance evidence, mutate the active pointer, open stages, or restore a prior release. Canonical payloads and immutable archive objects are content addressed; mutable run state is restricted to atomic checkpoint, budget, and active-pointer files. This project initializes and exercises the kernel with deterministic fake candidates; numerical quality-diversity, Hyperband, cooperative scheduling, DGM source mutation, and L1 protocol migration remain later projects.

**Tech Stack:** Python 3.10+ standard library, frozen dataclasses, `Protocol`, SHA-256, strict JSON/JSONL, atomic filesystem replacement, `argparse`, pytest, existing `common.payload` strict reader utilities.

**Spec:** `docs/superpowers/specs/2026-09-10-unified-evolution-v2-design.md`

## Global Constraints

- Add only the parallel `evolving_loop.v2` namespace and `configs/evolution_v2`; do not modify legacy controllers, CLI grammars, run directories, or frozen artifacts.
- L0 owns canonical serialization, task/split/protocol commitments, budget gates, acceptance sealing, active publication, and rollback. Candidate code receives no writable L0 object.
- V2 authoritative JSON must reject both NaN and positive/negative Infinity. Do not use the legacy infinity-sentinel behavior for V2 identities.
- Content identities exclude timestamps, local paths, clients, callbacks, file handles, and secrets. Runtime metadata that is not identity-bearing belongs in progress records.
- A single-coordinate Child changes exactly one of `numerical`, `retrieval`, or `decision`; a `joint` Child changes at least two. Harness, protocol, runtime, archive, scheduler, parent, and evidence fields cannot be smuggled through a module mutation.
- Dev values exist only in sealed acceptance evidence. Archive descriptors, objective vectors, scheduler inputs, and proposer-facing records are Train-only by schema.
- A rejected Child returns the exact Parent object and canonical bytes. A promoted Child must carry acceptance evidence bound to Parent, Child, protocol, and runtime identities.
- The formal profile has a 14,400-second hard limit and reserves its final 20% (2,880 seconds). No new search stage opens after the 11,520-second search deadline or when declared resource ceilings cannot cover it.
- `public-evaluate` is a non-learning validation boundary in this project. It validates a frozen accepted Bundle and writes an isolated Public run manifest; it does not mutate an archive, checkpoint, budget ledger, policy, or active Bundle.
- The Project 1 fake runner is explicitly identified as `deterministic_fake`; it proves kernel behavior but must never be reported as a real forecasting result.
- Every task ends with focused tests and a small Conventional Commit. Do not stage the user's unrelated dirty files.

## File Map

| File | Responsibility | Consumes | Produces |
|---|---|---|---|
| `evolving_loop/v2/contracts.py` | strict V2 JSON, SHA validation, artifact/config/protocol contracts | primitive payloads, config JSON | canonical bytes, fingerprints, `KernelProtocolCommitment`, `EvolutionV2Config` |
| `evolving_loop/v2/bundle.py` | complete Bundle identity and scope ownership | component/runtime fingerprints | `EvolutionBundleV2`, scope diff/validation |
| `evolving_loop/v2/archive.py` | immutable object store and append-only index | artifacts, Train-only records | content objects, lineage records |
| `evolving_loop/v2/budget.py` | resource ceilings, hard deadline, reserve, resume | profile config, monotonic clock | `BudgetPlan`, `BudgetLedger`, stage decisions |
| `evolving_loop/v2/store.py` | V2 run layout and crash-safe mutable files | canonical V2 payloads | manifests, checkpoints, progress, evidence, active Bundle |
| `evolving_loop/v2/kernel.py` | L0 acceptance/promotion/rollback authority | Bundle, evidence, archive, budget, store | accepted/rejected transition, active history |
| `evolving_loop/v2/fakes.py` | deterministic Project 1 exercise path | seed/config | legal accepted/rejected Children and evidence |
| `evolving_loop/v2/cli.py`, `__main__.py` | separate CLI and Public firewall | config/Bundle paths | initialized/fake-smoke/Public-validation runs |
| `configs/evolution_v2/*.json` | versioned profile inputs | none | smoke/pilot/formal defaults |
| `tests/test_evolution_v2_*.py` | contract, safety, resume, compatibility gates | all Project 1 interfaces | deterministic acceptance evidence |

## Existing Implementation Anchors

- `common/payload.py:10` supplies the duplicate-key-rejecting strict reader. V2 reuses that parser but not `common/payload.py:31` serialization, because the latter deliberately maps Infinity to legacy sentinels.
- `evolving_loop/package_artifacts.py:61` is the reviewed atomic-write pattern; V2 copies the pattern into its isolated namespace and routes all bytes through stricter V2 canonicalization.
- `evolving_loop/package_artifacts.py:277` is the legacy Package run store. It remains unchanged and is exercised only by compatibility tests.
- `evolving_loop/package_coordinate_evolution.py:54` is the legacy `PackageCoordinateBundle`. V2 does not extend or reinterpret it; an adapter in a later project may import its frozen bytes.
- `evolving_loop/cli.py:404` and `evolving_loop/cli.py:4452` remain the legacy parser and entry-point authority. V2 is reachable only through the new `evolving_loop/v2/__main__.py` entry point.

---

### Task 1: Strict canonical artifacts and V2 configuration

**Files:**
- Create: `evolving_loop/v2/__init__.py`
- Create: `evolving_loop/v2/contracts.py`
- Create: `tests/test_evolution_v2_contracts.py`

**Interfaces:**
- Produces: `EvolutionArtifact`, `KernelProtocolCommitment`, `SanitizedEvolutionFeedback`, `EvolutionV2Config`, `canonical_v2_bytes`, `fingerprint_payload`, `require_sha256`, `load_v2_config`.
- Consumes: JSON-compatible mappings and config files parsed with `common.payload.strict_json_loads`; no legacy artifact is rewritten.

- [ ] **Step 1: Write the failing canonicalization and schema tests**

```python
def test_v2_canonical_bytes_are_order_independent_and_strict():
    left = canonical_v2_bytes({"b": [2, 1], "a": "x"})
    right = canonical_v2_bytes({"a": "x", "b": [2, 1]})
    assert left == right
    assert fingerprint_payload({"b": [2, 1], "a": "x"}) == hashlib.sha256(left).hexdigest()
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            canonical_v2_bytes({"value": bad})

def test_v2_config_rejects_unknown_keys_and_wrong_boolean_integer_aliases(tmp_path):
    payload = valid_config_payload()
    payload["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        EvolutionV2Config.from_payload(payload)
    payload = valid_config_payload() | {"unowned_control": True}
    with pytest.raises(ValueError, match="exact schema"):
        EvolutionV2Config.from_payload(payload)
```

Also test non-string mapping keys, `Path`, callbacks, duplicate JSON keys, missing runtime fingerprints, invalid SHA-256 strings, unknown scheduler/profile/scope values, and formal hard-limit/reserve drift. Add a protocol test proving that changing any one L0 binding changes `KernelProtocolCommitment.fingerprint()`.

- [ ] **Step 2: Run the contract test and observe RED**

Run: `pytest -q tests/test_evolution_v2_contracts.py`

Expected: FAIL because `evolving_loop.v2` does not exist.

- [ ] **Step 3: Implement strict V2 canonical bytes and artifact protocol**

Use a recursive validator before serialization; do not call `standards_json_value`, because it intentionally converts Infinity for legacy raw-tail reports.

```python
JsonScalar = str | int | float | bool | None

def _strict_json_value(value: object, *, field: str = "payload") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{field} keys must be strings")
        return {key: _strict_json_value(item, field=f"{field}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item, field=f"{field}[]") for item in value]
    raise TypeError(f"{field} contains non-JSON value {type(value).__name__}")

def canonical_v2_bytes(payload: Mapping[str, object]) -> bytes:
    plain = _strict_json_value(payload)
    return (json.dumps(plain, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
```

Define the exact semantic protocol from the master spec and make `fingerprint_payload` hash these exact bytes.

Define `KernelProtocolCommitment` with exact SHA-256 fields for `task_materializer`, `split_manifest`, `metric_policy`, `label_firewall`, `artifact_validator`, `sandbox_policy`, and `promotion_policy`. Its canonical fingerprint is the only legal source for `EvolutionBundleV2.protocol_fingerprint`; raw caller-selected protocol strings are invalid.

Define `SanitizedEvolutionFeedback` as a closed Train-only primitive payload containing Parent SHA, Train evaluation SHA, Train objectives, Train behavior descriptors, failure categories, and remaining proposal budget. Recursively reject `dev_comparison`, `dev_metrics`, `public_ids`, `future_values`, `evaluator_labels`, and `holdout` keys. This establishes the Project 1 proposer seam; real task materialization and forecasting adapters arrive in Projects 2 and 3.

- [ ] **Step 4: Implement the closed Project 1 config contract**

`EvolutionV2Config` fields are exactly:

```python
schema_version: int                 # exactly 1 for the config schema
profile: Literal["smoke", "pilot", "formal", "public"]
seed: int
scheduler: Literal["ucb", "thompson"]
enabled_mutation_scopes: tuple[Literal["numerical", "retrieval", "decision", "joint"], ...]
archive_capacities: Mapping[str, int]
hyperband: Mapping[str, object]
runtime_fingerprints: Mapping[str, str]
kernel_protocol: KernelProtocolCommitment
hard_limit_seconds: int
finalization_reserve_fraction: float
runner: Literal["deterministic_fake", "production"]
```

Require exact keys, exact primitive types, non-empty unique scopes, positive capacities, canonical runtime SHA-256 values, a strictly parsed `KernelProtocolCommitment`, and `(hard_limit_seconds, reserve) == (14400, 0.2)` for `formal`. The config may name later algorithms but Project 1 must not pretend to execute them.

- [ ] **Step 5: Export only the stable Project 1 public surface**

In `evolving_loop/v2/__init__.py`, export contracts that later projects may consume. Do not import the CLI or fake runner at package import time.

- [ ] **Step 6: Run Task 1 tests**

Run: `pytest -q tests/test_evolution_v2_contracts.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add evolving_loop/v2/__init__.py evolving_loop/v2/contracts.py tests/test_evolution_v2_contracts.py
git commit -m "feat(evolution-v2): add strict contracts"
```

### Task 2: Complete Bundle identity and mutation ownership

**Files:**
- Create: `evolving_loop/v2/bundle.py`
- Create: `tests/test_evolution_v2_bundle.py`
- Modify: `evolving_loop/v2/__init__.py`

**Interfaces:**
- Consumes: canonical SHA-256 component, archive, scheduler, protocol, runtime, Parent, and evidence identities.
- Produces: immutable `EvolutionBundleV2`, `principal_fingerprints`, `changed_scopes`, `validate_child_scope`, `provisional_child`, and `seal_acceptance`.

- [ ] **Step 1: Write failing seed/Child identity tests**

```python
def test_seed_bundle_binds_every_replay_dependency():
    bundle = seed_bundle()
    assert set(bundle.to_payload()) == {
        "schema_version", "generation", "parent_bundle_sha256",
        "numerical_release_sha256", "numerical_registry_sha256",
        "retrieval_release_sha256", "decision_policy_sha256",
        "harness_policy_sha256", "archive_snapshot_sha256",
        "scheduler_state_sha256", "protocol_fingerprint",
        "runtime_fingerprints", "acceptance_evidence_sha256",
    }
    assert bundle.fingerprint() == hashlib.sha256(bundle.canonical_bytes()).hexdigest()

def test_single_coordinate_and_joint_children_enforce_actual_diff():
    parent = seed_bundle()
    numerical = parent.provisional_child("numerical", {"numerical": "1" * 64})
    assert changed_scopes(parent, numerical) == ("numerical",)
    with pytest.raises(BundleContractError, match="exactly one"):
        parent.provisional_child("retrieval", {"retrieval": "2" * 64, "decision": "3" * 64})
    joint = parent.provisional_child("joint", {"retrieval": "2" * 64, "decision": "3" * 64})
    assert changed_scopes(parent, joint) == ("retrieval", "decision")
```

Test seed Parent/evidence prohibition, positive-generation Parent requirement, evidence sealing only once, schema mismatch, empty runtime map, and mutation attempts against harness/protocol/runtime/archive/scheduler metadata.

- [ ] **Step 2: Run Bundle tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_bundle.py`

Expected: FAIL because the Bundle module is absent.

- [ ] **Step 3: Implement the frozen Bundle schema**

```python
@dataclass(frozen=True)
class EvolutionBundleV2:
    schema_version: int
    generation: int
    parent_bundle_sha256: str | None
    numerical_release_sha256: str
    numerical_registry_sha256: str
    retrieval_release_sha256: str
    decision_policy_sha256: str
    harness_policy_sha256: str
    archive_snapshot_sha256: str
    scheduler_state_sha256: str
    protocol_fingerprint: str
    runtime_fingerprints: Mapping[str, str]
    acceptance_evidence_sha256: str | None

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
```

Freeze `runtime_fingerprints` with `MappingProxyType`, sort it before serialization, and reject booleans as integers. `from_payload` must require the exact top-level schema and must never fill missing fields.

- [ ] **Step 4: Implement Host-computed scope diffs**

Map only the three principal identities:

```python
_PRINCIPAL_FIELDS = {
    "numerical": ("numerical_release_sha256", "numerical_registry_sha256"),
    "retrieval": ("retrieval_release_sha256",),
    "decision": ("decision_policy_sha256",),
}
```

`validate_child_scope` first verifies every non-principal authority field equals the Parent, except that every provisional Child must clear `acceptance_evidence_sha256` to `None`. It then recomputes changed scopes. Single targets must match exactly one scope; `joint` must change at least two. A Numerical change must replace both release and registry identities atomically. Sealing is a later Host-only operation and is never part of scope diffing.

`seal_acceptance(evidence_sha256, archive_snapshot_sha256, scheduler_state_sha256)` is the only constructor allowed to change Host-owned archive/scheduler identities. It starts from a scope-validated provisional Child, adds the evidence SHA and post-evaluation Host snapshots, and returns a new sealed Bundle. Candidate-side `provisional_child` cannot set any of those three values.

- [ ] **Step 5: Add exact rejection and round-trip tests**

Round-trip `to_payload -> from_payload` without byte changes. Test that a helper returning Parent on rejection uses `return parent`, then assert both object identity and byte equality:

```python
returned = reject_transition(parent, child)
assert returned is parent
assert returned.canonical_bytes() == parent.canonical_bytes()
```

- [ ] **Step 6: Run Task 2 tests**

Run: `pytest -q tests/test_evolution_v2_bundle.py tests/test_evolution_v2_contracts.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add evolving_loop/v2/__init__.py evolving_loop/v2/bundle.py tests/test_evolution_v2_bundle.py
git commit -m "feat(evolution-v2): define bundle authority"
```

### Task 3: Crash-safe V2 run store and append-only archive

**Files:**
- Create: `evolving_loop/v2/store.py`
- Create: `evolving_loop/v2/archive.py`
- Create: `tests/test_evolution_v2_store.py`
- Create: `tests/test_evolution_v2_archive.py`

**Interfaces:**
- Consumes: strict canonical payloads and `EvolutionArtifact` instances.
- Produces: `V2RunStore`, `ArchiveRecord`, `EvolutionArchive`, immutable objects/index, atomic checkpoint/active files, lineage reconstruction.

- [ ] **Step 1: Write failing run-layout and write-policy tests**

```python
def test_v2_store_creates_only_the_versioned_layout(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    assert {p.relative_to(store.root).as_posix() for p in store.root.rglob("*") if p.is_dir()} == {
        "acceptance", "archive", "archive/objects", "candidates", "evaluations", "canary"
    }

def test_write_once_is_idempotent_but_never_overwrites(tmp_path):
    store = V2RunStore.create(tmp_path / "run")
    store.write_run_manifest({"schema_version": 1, "system": "evolution_v2"})
    store.write_run_manifest({"schema_version": 1, "system": "evolution_v2"})
    with pytest.raises(StoreContractError, match="immutable"):
        store.write_run_manifest({"schema_version": 1, "system": "legacy"})
```

Also test atomic mutable writes, JSONL fsync append, nested candidate/evaluation paths, non-finite rejection, simple SHA-based path validation, and refusal to adopt an existing non-empty legacy directory.

- [ ] **Step 2: Write failing archive integrity and leakage tests**

```python
def test_archive_is_content_addressed_append_only_and_reconstructs_lineage(tmp_path):
    archive = EvolutionArchive(tmp_path / "archive")
    parent_artifact = fake_artifact("parent")
    parent = archive.append(parent_artifact, seed_record(parent_artifact))
    child_artifact = fake_artifact("child")
    child = archive.append(child_artifact, child_record(child_artifact, parents=(parent,)))
    assert (tmp_path / "archive" / "objects" / f"{child}.json").is_file()
    assert archive.lineage(child) == (parent, child)
    with pytest.raises(ArchiveContractError, match="already indexed"):
        archive.append(child_artifact, child_record(child_artifact, parents=(parent,)))

def test_archive_schema_has_no_dev_or_public_channel():
    payload = train_record_payload() | {"dev_metrics": {"smae": 0.1}}
    with pytest.raises(ArchiveContractError, match="exact schema"):
        ArchiveRecord.from_payload(payload)
```

Test truncated index line, object digest mismatch, missing parent, duplicate index entry, runtime/protocol mismatch, terminal invalid evaluation status, and stable archive snapshot SHA.

- [ ] **Step 3: Implement atomic and write-once primitives in `store.py`**

Reuse the proven temporary-file + `os.replace` pattern, but route bytes through `canonical_v2_bytes`:

```python
def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False,
                                         prefix=f".{path.name}.", suffix=".tmp") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
```

`write_once_json` permits identical idempotent bytes and rejects different bytes. `write_atomic_json` is allowed only for `checkpoint.json` and `accepted_bundle.json`; `budget_plan.json`, manifests, evidence, and objects are write-once. Progress and archive indexes are append-only JSONL.

- [ ] **Step 4: Implement the exact V2 directory API**

Required store methods:

```text
write_run_manifest(payload)
write_budget_plan(payload)
write_checkpoint(payload)
append_progress(payload)
write_candidate(candidate_sha256, payload)
write_evaluation(candidate_sha256, stage, payload)
write_acceptance(evidence_sha256, payload)
publish_active_bundle(bundle_payload)
write_canary(source_sha256, payload)
write_completion(payload)
```

Creation accepts an empty directory or resumes a directory whose manifest declares `system == "evolution_v2"`; it rejects any other non-empty directory before writing.

- [ ] **Step 5: Implement a closed Train-only `ArchiveRecord`**

Fields are exactly `schema_version`, `artifact_sha256`, `artifact_kind`, `parent_sha256s`, `mutation_operator`, `protocol_fingerprint`, `runtime_fingerprints`, `train_behavior_descriptors`, `train_objectives`, `evaluation_status`, `resource_use`, `accepted_release_sha256s`, and `source_lineage_sha256s`. Recursively reject reserved evaluator-only key segments such as `dev_metrics`, `dev_comparison`, `public_ids`, `future_values`, `evaluator_labels`, and `holdout`; do not use a broad substring ban that would reject harmless bindings such as `label_firewall_runtime`.

`EvolutionArchive.append(artifact, record)` requires `record.artifact_sha256 == artifact.fingerprint()`. It writes `artifact.to_payload()` to `objects/<artifact-sha256>.json`, fsyncs it, then appends an index envelope containing the exact record and its own `record_sha256`. The returned identity is the artifact SHA. On load, verify every index line, record digest, object filename, object digest, artifact/record binding, and referenced parent. `snapshot_sha256()` hashes the ordered canonical index bytes.

- [ ] **Step 6: Run Task 3 tests**

Run: `pytest -q tests/test_evolution_v2_store.py tests/test_evolution_v2_archive.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add evolving_loop/v2/store.py evolving_loop/v2/archive.py tests/test_evolution_v2_store.py tests/test_evolution_v2_archive.py
git commit -m "feat(evolution-v2): persist append-only state"
```

### Task 4: Resumable multi-resource budget and deadline reserve

**Files:**
- Create: `evolving_loop/v2/budget.py`
- Create: `tests/test_evolution_v2_budget.py`
- Modify: `evolving_loop/v2/__init__.py`

**Interfaces:**
- Consumes: `EvolutionV2Config`, injected monotonic clock, estimated/actual `ResourceUse`.
- Produces: immutable `BudgetPlan`, checkpointable `BudgetLedger`, `StagePermit`, closed denial reasons.

- [ ] **Step 1: Write failing profile, resource, and deadline tests**

```python
def test_formal_budget_reserves_last_twenty_percent():
    clock = FakeClock(0.0)
    ledger = BudgetLedger(formal_plan(), monotonic=clock)
    assert ledger.plan.hard_limit_seconds == 14_400
    assert ledger.plan.search_deadline_seconds == 11_520
    clock.advance(11_519)
    assert ledger.can_open_stage(ResourceUse(wall_seconds=1)).allowed is True
    clock.advance(1)
    denial = ledger.can_open_stage(ResourceUse(wall_seconds=1))
    assert denial.allowed is False
    assert denial.reason == "finalization_reserve"

def test_stage_must_fit_time_and_every_resource_ceiling():
    ledger = BudgetLedger(plan(ceilings={"task_executions": 8}))
    ledger.charge(ResourceUse(task_executions=7))
    assert ledger.can_open_stage(ResourceUse(task_executions=2)).reason == "task_executions_exhausted"
```

Cover wall seconds, task executions, LLM calls, input/output tokens, GPU-seconds, subprocesses, and artifact bytes. Test non-finite/negative values, actual use exceeding a reservation, double-close, checkpoint tampering, and resume with elapsed prior wall time.

- [ ] **Step 2: Run budget tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_budget.py`

Expected: FAIL because the budget module is absent.

- [ ] **Step 3: Implement typed resource arithmetic and profile plan**

```python
@dataclass(frozen=True)
class ResourceUse:
    wall_seconds: float = 0.0
    task_executions: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0
    subprocesses: int = 0
    artifact_bytes: int = 0

@dataclass(frozen=True)
class StagePermit:
    allowed: bool
    reason: str | None
    reservation_sha256: str | None
```

Validate exact types (`bool` is invalid for integer fields) and finite non-negative floats. `BudgetPlan.from_config` computes `search_deadline_seconds = hard_limit * (1 - reserve)` and stores all ceilings in its fingerprint.

- [ ] **Step 4: Implement reserve/open/close semantics**

`reserve_stage(stage_id, estimate)` atomically checks search time and every ceiling, rejects duplicate open IDs, and returns a digest-bound reservation. `close_stage(reservation, actual)` charges actual use and removes the open reservation; actual may be smaller than estimate but cannot exceed it without a closed `budget_overrun` record. `begin_finalization()` closes search permanently. No resume can reopen a completed or denied stage ID.

- [ ] **Step 5: Implement exact checkpoint round trip**

Checkpoint stores plan SHA, prior elapsed wall seconds, charged use, open reservations, closed stage IDs, and finalization state. Resume receives a fresh monotonic origin and adds new elapsed time to prior elapsed time. Reject a plan SHA mismatch before permitting any stage.

- [ ] **Step 6: Run Task 4 tests**

Run: `pytest -q tests/test_evolution_v2_budget.py tests/test_evolution_v2_contracts.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add evolving_loop/v2/__init__.py evolving_loop/v2/budget.py tests/test_evolution_v2_budget.py
git commit -m "feat(evolution-v2): enforce epoch budgets"
```

### Task 5: Immutable kernel acceptance and automatic promotion/rollback Host

**Files:**
- Create: `evolving_loop/v2/kernel.py`
- Create: `tests/test_evolution_v2_kernel.py`
- Modify: `evolving_loop/v2/__init__.py`

**Interfaces:**
- Consumes: Parent/Child Bundles, declared target, closed evaluation summary, budget permit, archive/store.
- Produces: `AcceptanceEvidence`, `EvolutionKernel.evaluate_transition`, `PromotionHost.activate`, `PromotionHost.rollback`, sealed active Bundle and immutable promotion history.

- [ ] **Step 1: Write failing authority and evidence-binding tests**

```python
def test_kernel_recomputes_scope_and_rejects_authority_smuggling(kernel):
    parent = seed_bundle()
    child = replace(valid_numerical_child(parent), protocol_fingerprint="9" * 64)
    with pytest.raises(KernelAuthorityError, match="protocol"):
        kernel.evaluate_transition(parent, child, target="numerical", evaluation=passing_eval())

def test_acceptance_evidence_binds_parent_child_protocol_runtime_and_dev(kernel):
    parent = seed_bundle()
    child = valid_numerical_child(parent)
    accepted = kernel.evaluate_transition(parent, child, target="numerical", evaluation=passing_eval())
    evidence = kernel.load_acceptance(accepted.acceptance_evidence_sha256)
    assert evidence.parent_bundle_sha256 == parent.fingerprint()
    assert evidence.candidate_bundle_sha256 == child.fingerprint()
    assert evidence.protocol_fingerprint == parent.protocol_fingerprint
    assert evidence.runtime_fingerprints == parent.runtime_fingerprints
```

Test fabricated evidence SHA, candidate evaluating/promoting itself, missing stage permit, parent drift, protocol/runtime drift, evidence resealing, archive entry omission, and Dev leakage into progress/archive.

Construct the kernel with a `KernelProtocolCommitment` object and require both Parent and Child `protocol_fingerprint` values to equal its recomputed fingerprint. On resume, re-read the full commitment from `run_manifest.json`; never trust only the Bundle's digest string.

- [ ] **Step 2: Write failing rejection, activation, and rollback tests**

```python
def test_rejection_preserves_exact_parent_and_never_moves_active_pointer(kernel):
    parent = kernel.active_bundle()
    returned = kernel.evaluate_transition(parent, valid_child(parent),
                                          target="retrieval", evaluation=failing_eval())
    assert returned is parent
    assert returned.canonical_bytes() == parent.canonical_bytes()
    assert kernel.active_bundle().fingerprint() == parent.fingerprint()

def test_host_promotes_without_human_approval_and_rolls_back_atomically(kernel):
    parent = kernel.active_bundle()
    accepted = kernel.evaluate_transition(parent, valid_child(parent),
                                          target="decision", evaluation=passing_eval())
    assert kernel.active_bundle().fingerprint() == accepted.fingerprint()
    restored = kernel.promotion_host.rollback(reason="canary_failed")
    assert restored.fingerprint() == parent.fingerprint()
```

Assert immutable history contains both activation and rollback evidence and that restart reconstructs the same active Bundle.

- [ ] **Step 3: Implement exact acceptance/evaluation contracts**

`ClosedEvaluation` contains only kernel-controlled aggregate status plus separate `train_evaluation_sha256` and sealed `dev_comparison` payload. `AcceptanceEvidence` has an exact schema and binds Parent SHA, provisional Child SHA, target, protocol, runtime, decision, and Dev comparison. It does **not** contain the sealed Child SHA: the sealed Child contains the evidence SHA, so including both directions would create an impossible circular hash. The immutable promotion record binds the resulting sealed Child SHA to the evidence SHA. Acceptance evidence exposes no method that produces proposer feedback.

- [ ] **Step 4: Implement the L0 transition order**

The method order is fixed:

```text
1. require active Parent fingerprint
2. validate budget stage permit
3. recompute and validate mutation scope
4. verify closed evaluation and protocol/runtime bindings
5. write immutable evaluation and archive record
6. write and re-read acceptance evidence bound to the provisional Child
7a. reject -> return the original Parent object
7b. accept -> seal Child with evidence SHA and post-evaluation archive/scheduler snapshots
8. persist sealed Bundle by content hash
9. append promotion record binding sealed Bundle SHA to evidence SHA, then atomically activate through PromotionHost
10. checkpoint budget, archive snapshot, and active identity
```

Any failure before step 9 leaves the active pointer unchanged. A failure during step 9 is recovered from the prior pointer stored in immutable promotion history.

- [ ] **Step 5: Implement automatic `PromotionHost` and rollback**

The Host alone owns `accepted_bundle.json`. Activation requires a sealed acceptance artifact already present in the store and appends a write-once promotion record before atomically replacing the active Bundle. Rollback requires a known prior active digest and a closed canary/integrity reason; no human-approval flag exists in the API.

Keep DGM source-canary scoring out of this task: later DGM code will call the same Host after producing its canary evidence.

- [ ] **Step 6: Run Task 5 tests**

Run: `pytest -q tests/test_evolution_v2_kernel.py tests/test_evolution_v2_bundle.py tests/test_evolution_v2_archive.py tests/test_evolution_v2_budget.py tests/test_evolution_v2_store.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add evolving_loop/v2/__init__.py evolving_loop/v2/kernel.py tests/test_evolution_v2_kernel.py
git commit -m "feat(evolution-v2): add promotion kernel"
```

### Task 6: Deterministic fake kernel loop and resumable checkpoint

**Files:**
- Create: `evolving_loop/v2/fakes.py`
- Create: `tests/test_evolution_v2_fakes.py`
- Modify: `evolving_loop/v2/kernel.py`

**Interfaces:**
- Consumes: smoke config, seed Bundle, fake clock.
- Produces: one accepted Numerical Child, one rejected Retrieval Child, deterministic artifacts, resumable checkpoint, explicit `deterministic_fake` completion.

- [ ] **Step 1: Write failing deterministic end-to-end tests**

```python
def test_fake_loop_accepts_then_rejects_and_is_byte_reproducible(tmp_path):
    first = run_fake_kernel(tmp_path / "first", smoke_config(seed=7))
    second = run_fake_kernel(tmp_path / "second", smoke_config(seed=7))
    assert first.accepted_bundle.canonical_bytes() == second.accepted_bundle.canonical_bytes()
    assert first.accepted_steps == 1
    assert first.rejected_steps == 1
    assert first.runner == "deterministic_fake"
    assert read_json(first.completion_path)["public_test_accessed"] is False

def test_fake_resume_skips_completed_immutable_stage(tmp_path):
    run_fake_kernel(tmp_path / "run", smoke_config(), stop_after_stage="fake-numerical")
    resumed = run_fake_kernel(tmp_path / "run", smoke_config(), resume=True)
    assert resumed.completed_stage_ids == ("fake-numerical", "fake-retrieval")
    assert count_evaluation_files(tmp_path / "run") == 2
```

Test seed determinism, config/checkpoint mismatch, tampered archive/object/evidence, exhausted budget, and exact Parent preservation after the rejected Child.

- [ ] **Step 2: Run fake-loop tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_fakes.py`

Expected: FAIL because the deterministic runner is absent.

- [ ] **Step 3: Implement closed deterministic fixtures**

Derive all fake component SHA values from `sha256(f"v2-fake:{seed}:{name}".encode())`. Do not read datasets, call an LLM, invoke a forecast backbone, inspect environment secrets, or use wall-clock timestamps. The fake Numerical evaluation is a fixed Pareto improvement; fake Retrieval is a fixed regression. Both pass through the real archive, budget, evidence, and promotion code.

- [ ] **Step 4: Add checkpoint sealing and verified resume**

Checkpoint fields are exactly config SHA, active Bundle SHA, archive snapshot SHA, budget ledger payload, ordered completed stage IDs, accepted/rejected counts, and `public_test_accessed: false`. Resume revalidates all referenced immutable bytes before computing the next stage. Resuming an already complete run is a verified no-op that returns its recorded result without appending files. Never heuristically repair a partial stage; ignore only an unreferenced temporary file.

- [ ] **Step 5: Write completion only after clean finalization**

Completion status is `deterministic_fake_complete`, includes runner identity, final Bundle SHA, counts, budget usage, and `public_test_accessed: false`. It must not contain metric claims labeled as real forecasting quality.

- [ ] **Step 6: Run Task 6 tests**

Run: `pytest -q tests/test_evolution_v2_fakes.py tests/test_evolution_v2_kernel.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add evolving_loop/v2/fakes.py evolving_loop/v2/kernel.py tests/test_evolution_v2_fakes.py
git commit -m "test(evolution-v2): add deterministic kernel loop"
```

### Task 7: Separate V2 CLI, configs, and non-learning Public boundary

**Files:**
- Create: `evolving_loop/v2/cli.py`
- Create: `evolving_loop/v2/__main__.py`
- Create: `configs/evolution_v2/smoke.json`
- Create: `configs/evolution_v2/pilot.json`
- Create: `configs/evolution_v2/formal.json`
- Create: `tests/test_evolution_v2_cli.py`

**Interfaces:**
- Consumes: config path, output directory, accepted Bundle path.
- Produces: `python -m evolving_loop.v2 evolve ...` and `public-evaluate ...`; leaves `evolving-agent` and `python -m evolving_loop` unchanged.

- [ ] **Step 1: Write failing parser and isolation tests**

```python
def test_v2_parser_exposes_only_parallel_commands():
    parser = build_parser()
    evolve = parser.parse_args(["evolve", "--config", "c.json", "--output-dir", "out"])
    public = parser.parse_args(["public-evaluate", "--bundle", "b.json", "--output-dir", "pub"])
    assert evolve.command == "evolve"
    assert public.command == "public-evaluate"

def test_v2_cli_never_dispatches_to_legacy_cli(monkeypatch, tmp_path):
    monkeypatch.setattr("evolving_loop.cli.main", lambda: pytest.fail("legacy CLI called"))
    assert main(["evolve", "--config", smoke_config_path(),
                 "--output-dir", str(tmp_path / "v2")]) == 0
```

Test missing flags, runner/profile mismatch, overlapping source-run/Public output directories, existing legacy output rejection, same-run resume (including completed-run no-op verification), production runner rejection in Project 1, and subprocess invocation through `python -m evolving_loop.v2`.

- [ ] **Step 2: Write failing Public-firewall tests**

Snapshot every file under the source V2 run, call `public-evaluate`, then assert source hashes/mtimes and archive line count are unchanged. The Public output must contain only `run_manifest.json`, a copied content-addressed Bundle object, and `evaluation_complete.json` with status `validated_only_no_public_evaluator`. It must not contain archive, checkpoint, candidates, acceptance, skills, policies, or budget state.

- [ ] **Step 3: Run CLI tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_cli.py`

Expected: FAIL because the V2 CLI and configs do not exist.

- [ ] **Step 4: Implement `evolve` without overstating Project 1 capability**

Behavior:

```text
runner=deterministic_fake -> execute/resume Task 6 and print canonical summary JSON
runner=production         -> fail before creating output with
                             "production evolution requires Project 2+ adapters"
profile=public            -> reject; use public-evaluate
```

No legacy imports are needed. `evolve` accepts only an empty directory or the exact resumable V2 run directory. `public-evaluate` resolves the source run from the supplied Bundle path and rejects an output path that equals, contains, or is contained by that source run.

- [ ] **Step 5: Implement `public-evaluate` as validation-only**

Parse `accepted_bundle.json` strictly as a sealed `EvolutionBundleV2`, verify its fingerprint and acceptance evidence reference format, require an empty new output directory, and copy the canonical Bundle bytes into `objects/<sha>.json`. Write an explicit validation-only completion. Do not load task data because the Public evaluator belongs to a later integration project.

- [ ] **Step 6: Add truthful profile configs**

- `smoke.json`: 600 seconds, deterministic fake, one fake Child per enabled test arm.
- `pilot.json`: 7,200 seconds, production runner, 8/2 intent.
- `formal.json`: 14,400 seconds, production runner, one scheduler-selected arm, reserve `0.2`.

All include scheduler, scopes, capacities, Hyperband policy, and runtime bindings computed as real deterministic config-component digests (never all-zero or undocumented strings). Comments are avoided because JSON is strict.

- [ ] **Step 7: Run Task 7 tests and manual smoke**

Run:

```bash
pytest -q tests/test_evolution_v2_cli.py
python -m evolving_loop.v2 evolve --config configs/evolution_v2/smoke.json --output-dir /tmp/evolution-v2-kernel-smoke
python -m evolving_loop.v2 public-evaluate --bundle /tmp/evolution-v2-kernel-smoke/accepted_bundle.json --output-dir /tmp/evolution-v2-public-validation
```

Expected: tests PASS; both commands exit 0; the first reports `deterministic_fake_complete`; the second reports `validated_only_no_public_evaluator`.

- [ ] **Step 8: Commit Task 7**

```bash
git add evolving_loop/v2/cli.py evolving_loop/v2/__main__.py configs/evolution_v2 tests/test_evolution_v2_cli.py
git commit -m "feat(evolution-v2): expose isolated CLI"
```

### Task 8: L0 hostile fixtures and legacy compatibility gate

**Files:**
- Create: `tests/test_evolution_v2_safety.py`
- Create: `tests/test_evolution_v2_compatibility.py`

**Interfaces:**
- Consumes: public V2 APIs, hostile payloads/source fixtures, existing legacy parsers and representative frozen release files.
- Produces: evidence that Project 1 authority is fail-closed and legacy surfaces are unchanged.

- [ ] **Step 1: Add parameterized hostile artifact tests**

Attempt each forbidden mutation independently: protocol fingerprint, runtime fingerprint, harness policy, archive snapshot, scheduler state, Parent SHA, fabricated acceptance SHA, Numerical release without registry, single-target multi-scope change, joint one-scope change, non-finite nested metric, path traversal SHA/stage, duplicate JSON key, Dev/Public/future/label archive key, and mutable active pointer without Host evidence.

```python
@pytest.mark.parametrize("field", (
    "protocol_fingerprint", "runtime_fingerprints", "harness_policy_sha256",
    "archive_snapshot_sha256", "scheduler_state_sha256",
))
def test_candidate_cannot_mutate_l0_authority(field, kernel):
    parent = kernel.active_bundle()
    child = authority_mutation(parent, field)
    with pytest.raises(KernelAuthorityError):
        kernel.evaluate_transition(parent, child, target="numerical", evaluation=passing_eval())
    assert kernel.active_bundle().canonical_bytes() == parent.canonical_bytes()
```

- [ ] **Step 2: Add candidate import/write isolation seam tests**

Project 1 establishes the boundary API rather than the final OS sandbox. Test that the proposer-facing Bundle request is a canonical primitive payload with no store, kernel, archive, promotion Host, callback, or filesystem path. Test the Project 1 Bundle proposal parser rejects source/module/path fields; do not impose that restriction on future typed DGM source artifacts. Mark the separate-process read-only sandbox as a required Project 4 integration, not as passing here.

- [ ] **Step 3: Add Public and Dev non-interference tests**

Use unique sentinel values in Dev comparison and Public Bundle validation. Assert sentinels occur only in acceptance evidence or Public output, never in archive index/objects, fake proposer request, progress records, budget checkpoint, or source run files.

- [ ] **Step 4: Add legacy compatibility tests**

Verify importing/running V2 does not change legacy parser defaults or representative frozen payloads:

```python
def test_v2_import_does_not_change_legacy_cli_defaults():
    before = evolving_loop.cli.build_parser().parse_args(["evolve"])
    import evolving_loop.v2
    after = evolving_loop.cli.build_parser().parse_args(["evolve"])
    assert vars(after) == vars(before)

def test_v2_smoke_does_not_rewrite_legacy_release(tmp_path):
    release = Path("evolving_loop/retrieval_agent/releases/v000/manifest.json")
    before = (release.read_bytes(), release.stat().st_mtime_ns)
    run_fake_kernel(tmp_path / "v2", smoke_config())
    assert (release.read_bytes(), release.stat().st_mtime_ns) == before
```

Also parse a representative `PackageCoordinateBundle` before and after V2 import and assert identical canonical bytes.

- [ ] **Step 5: Run safety and compatibility gates**

Run:

```bash
pytest -q tests/test_evolution_v2_safety.py tests/test_evolution_v2_compatibility.py
pytest -q tests/test_evolving_cli.py tests/test_package_coordinate_evolution.py tests/test_package_artifacts.py
```

Expected: all tests PASS; no legacy artifact changes appear in `git status`.

- [ ] **Step 6: Commit Task 8**

```bash
git add tests/test_evolution_v2_safety.py tests/test_evolution_v2_compatibility.py
git commit -m "test(evolution-v2): lock kernel authority"
```

### Task 9: Project 1 documentation and full verification checkpoint

**Files:**
- Create: `docs/evolution-v2-kernel.md`
- Modify: `README.md`
- Verify: all Project 1 source/config/test files

**Interfaces:**
- Consumes: implemented CLI and verified artifacts.
- Produces: truthful operator guide, Project 1 completion evidence, explicit handoff boundary for Project 2.

- [ ] **Step 1: Document only implemented behavior**

Document the L0/L1/L2/L3 boundary, strict V2 serialization difference from legacy Infinity sentinels, run layout, automatic promotion/rollback, checkpoint/resume rules, smoke command, validation-only Public command, and how to inspect fingerprints. State explicitly that production Numerical evolution, MAP-Elites, NSGA-II, Hyperband, cooperative schedulers, DGM source mutation, and protocol migration are not implemented by Project 1.

- [ ] **Step 2: Add a small README entry**

Link the master spec, this implementation plan, and `docs/evolution-v2-kernel.md`. Keep existing legacy usage intact and label V2 as parallel/experimental.

- [ ] **Step 3: Run the complete Project 1 suite**

```bash
pytest -q \
  tests/test_evolution_v2_contracts.py \
  tests/test_evolution_v2_bundle.py \
  tests/test_evolution_v2_store.py \
  tests/test_evolution_v2_archive.py \
  tests/test_evolution_v2_budget.py \
  tests/test_evolution_v2_kernel.py \
  tests/test_evolution_v2_fakes.py \
  tests/test_evolution_v2_cli.py \
  tests/test_evolution_v2_safety.py \
  tests/test_evolution_v2_compatibility.py
```

Expected: all Project 1 tests PASS.

- [ ] **Step 4: Run legacy regression and repository checks**

```bash
pytest -q \
  tests/test_evolution_core_contracts.py \
  tests/test_evolution_core_persistence.py \
  tests/test_package_artifacts.py \
  tests/test_package_coordinate_evolution.py \
  tests/test_run_package_coevolution.py \
  tests/test_evolving_cli.py
git diff --check
rg -n "TODO|FIXME|pass$|NotImplemented|similar to|and so on|placeholder" \
  evolving_loop/v2 configs/evolution_v2 tests/test_evolution_v2_*.py docs/evolution-v2-kernel.md
```

Expected: regression tests PASS; whitespace check is clean; placeholder scan has no unexplained matches. Pre-existing failures outside this set must be reported with exact commands and evidence, not silently absorbed.

- [ ] **Step 5: Re-run both CLI boundaries in fresh temporary directories**

Use `mktemp -d`, run deterministic smoke, resume it once, then run Public validation. Verify:

```text
formal config refuses Project 1 production execution before output creation
smoke creates exactly two closed fake evaluations
resume creates no duplicate immutable objects/index lines
accepted Bundle is the accepted Numerical Child
rejected Retrieval Child leaves that Bundle byte-identical
Public validation changes no source-run byte
all budget use and reserve fields are present
```

- [ ] **Step 6: Inspect the final diff for scope**

Run `git status --short`, `git diff --stat`, and `git diff -- evolving_loop/v2 configs/evolution_v2 tests/test_evolution_v2_*.py docs/evolution-v2-kernel.md README.md`. Confirm no existing legacy production module or user-owned dirty file is included.

- [ ] **Step 7: Commit Task 9**

```bash
git add README.md docs/evolution-v2-kernel.md
git commit -m "docs(evolution-v2): document kernel boundary"
```

## Project 1 Exit Gate

Project 2, Numerical quality-diversity, may start only when all of the following are evidenced in a fresh verification run:

- strict canonical fingerprints reproduce across processes;
- Bundle scope/authority hostile fixtures all fail closed;
- archive objects/index and lineage verify from disk;
- formal budget computes the exact 4-hour/20% reserve boundary and resumes without resetting use;
- accepted and rejected fake transitions reproduce byte-for-byte;
- automatic promotion and rollback reconstruct correctly after restart;
- Public validation is non-learning and cannot overlap or mutate the source run;
- Project 1 tests and the named legacy regression suite pass; and
- the final diff contains no changes to existing legacy controllers, backbones, loaders, metrics, verifiers, validators, promotion code, or frozen artifacts.
