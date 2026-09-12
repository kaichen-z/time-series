# Evolution V2 Real Bounded Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable real P2→P3→P4→P5 launcher with approved 30-minute and one-hour profiles, then execute a real authenticated 30-minute run using `gpt-5.6-luna` at medium reasoning effort.

**Architecture:** A new Host-owned root runner validates canonical real inputs, owns one aggregate deadline, derives child configs, and invokes the existing four project runners through typed bridges. Each child retains its own store; root handoffs commit verified frozen identities and root resume adopts only sealed child boundaries.

**Tech Stack:** Python 3.11+, frozen dataclasses, existing Evolution V2 canonical JSON/SHA-256/store/budget contracts, existing Codex CLI client, Toto-balanced v3 forecast cache, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-12-evolution-v2-real-one-hour-design.md`

## Global Constraints

- Approved profiles are exactly `real-30m` (1800 seconds) and `real-1h` (3600 seconds), with exact stage allocations from the spec.
- The first live run uses `real-30m`, `gpt-5.6-luna`, and reasoning effort `medium`.
- The real path must not import `tests`, instantiate `FakeLLMClient`, or call source/protocol smoke builders.
- P2 uses the complete frozen 80-Train/20-Dev authority; P3/P4/P5 evaluate deterministic Train4/Dev1 projections whose tasks come from that authority.
- Public task contents, labels, forecasts, and metrics are never loaded. Every completion records `public_test_accessed=false`.
- Forecast backbones, loaders, metrics, verifiers, artifact validation, L0, and promotion Host stay fixed.
- Root finalization reserve is never available to search; unused search time rolls only forward.
- Resume occurs only at verified sealed boundaries. Unknown interrupted work is charged its full grant and is not replayed in the same epoch.
- A seed-only or incomplete scientific result is valid only when reported honestly; it is never relabelled as an accepted evolved result.
- Tests use injected clocks and deterministic adapters. The real canary and requested 30-minute run are operational validation, not CI.

## File Structure

- `evolving_loop/v2/real/contracts.py`: exact profile, manifest, checkpoint, stage-record, and result contracts.
- `evolving_loop/v2/real/host.py`: real input loading, ForecastStore/Codex construction, task projection, and resource cleanup.
- `evolving_loop/v2/real/bridges.py`: derived configs and verified P2/P3 handoffs.
- `evolving_loop/v2/real/runner.py`: aggregate budget, state machine, child execution, resume, and root finalization.
- `evolving_loop/v2/source/bridge.py`: production P3-bundle source case.
- `evolving_loop/v2/protocol/bridge.py`: production P3-archive protocol case.
- `evolving_loop/v2/numerical_qd/persistence.py`: verified active frozen-pair loader.
- `evolving_loop/v2/cli.py`: `real-evolve` parsing and dispatch only.
- `configs/evolution_v2/real/`: exact 30-minute/one-hour controls and Toto-balanced v3 manifests.
- `tests/test_evolution_v2_real_*.py`: focused contracts, budgets, bridges, runner/resume, and CLI integration.

---

### Task 1: Exact profiles and root contracts

**Files:** Create `evolving_loop/v2/real/__init__.py`, `evolving_loop/v2/real/contracts.py`, `tests/test_evolution_v2_real_contracts.py`; create `configs/evolution_v2/real/real-30m.json`, `configs/evolution_v2/real/real-1h.json`.

**Interfaces:** Produce `PROFILE_SCHEDULES`, `RealScheduleV2`, `RealModelBindingV2`, `RealInputFileV2`, `RealRuntimeLocationV2`, `RealEvolutionManifestV2`, `RealStageRecordV2`, and `RealRunResultV2`. Every contract has strict `from_payload`, `to_payload`, `canonical_bytes`, and `fingerprint` behavior using existing V2 helpers.

- [ ] **Step 1: Write failing contract tests.** Cover exact schemas, canonical role order, relative path confinement, 64-hex identities, Luna-medium binding, schedule sums, and rejection of arbitrary limits.

```python
def test_approved_profiles_have_exact_total_and_reserve():
    assert PROFILE_SCHEDULES["real-30m"].allocations == {
        "p2": 840, "p3": 360, "p4": 120, "p5": 120, "finalization": 360,
    }
    assert PROFILE_SCHEDULES["real-1h"].total_seconds == 3600
    assert sum(PROFILE_SCHEDULES["real-1h"].allocations.values()) == 3600

def test_manifest_rejects_model_or_path_drift(valid_manifest_payload):
    valid_manifest_payload["model"]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="gpt-5.6-luna/medium"):
        RealEvolutionManifestV2.from_payload(valid_manifest_payload)
    valid_manifest_payload["model"]["reasoning_effort"] = "medium"
    valid_manifest_payload["files"][0]["relative_path"] = "../secret"
    with pytest.raises(ValueError, match="relative"):
        RealEvolutionManifestV2.from_payload(valid_manifest_payload)
```

- [ ] **Step 2: Run RED.** Run `python -m pytest -q tests/test_evolution_v2_real_contracts.py`; require failure because the package does not exist.
- [ ] **Step 3: Implement the exact contracts.** Use `MappingProxyType` for schedules. File roles are unique ordered records with `relative_path` and `sha256`; runtime locations are unique ordered records with `relative_path` and `identity_sha256`. Checkpoints contain `phase`, `stage_records`, `active_stage`, `carry_seconds`, `budget_checkpoint`, `handoff_sha256s`, and optional completion SHA. Result status is exactly `complete|incomplete|failed`.
- [ ] **Step 4: Add canonical profile JSON.** The 30-minute file contains `840/360/120/120/360`; the one-hour file contains `1680/720/240/240/720`. Re-read them with `strict_json_loads` and require byte equality with `canonical_v2_bytes`.
- [ ] **Step 5: Run GREEN.** Run the contract test file and `git diff --check`.
- [ ] **Step 6: Commit.** `git commit -m "feat(real): add bounded run contracts"`.

### Task 2: Root budget state machine and sealed resume

**Files:** Create `evolving_loop/v2/real/runner.py`, `tests/test_evolution_v2_real_runner.py`; modify `evolving_loop/v2/real/__init__.py`.

**Interfaces:** Define `RealStagePorts` with callables `run_p2(context)`, `seal_p2(context, result)`, through `run_p5/seal_p5`; each seal returns `SealedStageV2(completion_sha256, handoff_payload, summary, public_test_accessed)`. Produce:

```python
def run_real_evolution(
    output_dir: str | Path,
    manifest: RealEvolutionManifestV2,
    ports: RealStagePorts,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> RealRunResultV2: ...
```

- [ ] **Step 1: Write failing budget/state tests.** Use a fake clock and spy ports. Test base grants, P2-to-P3 carry, cascading carry, no final-reserve borrowing, overrun, stopped child, Public evidence failure, and call order.

```python
def test_unused_time_rolls_forward_without_touching_reserve(case):
    case.stage_durations.update(p2=600, p3=400, p4=100, p5=100)
    result = case.run("real-30m")
    assert case.grants == {"p2": 840, "p3": 600, "p4": 320, "p5": 240}
    assert result.status == "complete"
    assert case.finalization_started_at <= 1440

def test_unknown_inflight_work_is_fully_charged_and_not_replayed(case):
    case.crash_during("p3")
    with pytest.raises(SimulatedCrash):
        case.run("real-30m")
    resumed = case.resume()
    assert resumed.status == "incomplete"
    assert case.calls["p3"] == 1
    assert case.closed_charge("p3") == case.grant("p3")
```

- [ ] **Step 2: Run RED.** Run `python -m pytest -q tests/test_evolution_v2_real_runner.py`; require missing runner failure.
- [ ] **Step 3: Implement root persistence.** Use `V2RunStore`, `BudgetPlan`, and `BudgetLedger`; write the root checkpoint immediately after reservation and after every seal. Store child outputs under `p2/`…`p5/`, immutable handoffs under `handoffs/`, and one canonical progress row per sealed transition.
- [ ] **Step 4: Implement allocation and recovery.** Compute the exact formula from the spec, charge parent monotonic elapsed, reject overrun, adopt a completed child only after its seal validator passes, and otherwise close the unknown reservation at full grant with `incomplete`.
- [ ] **Step 5: Implement finalization.** Call `begin_finalization`, re-read all child completions and handoffs, require four false Public flags, write `result_summary.json`, then `evaluation_complete.json`. Completed resume performs no writes and returns identical canonical bytes.
- [ ] **Step 6: Run GREEN.** Run the runner and contract test files.
- [ ] **Step 7: Commit.** `git commit -m "feat(real): orchestrate shared run budget"`.

### Task 3: P2 persistence handoff and real numerical Host

**Files:** Modify `evolving_loop/v2/numerical_qd/persistence.py`, `evolving_loop/v2/numerical_qd/__init__.py`, `evolving_loop/v2/cli.py`; create `evolving_loop/v2/real/host.py`, `evolving_loop/v2/real/bridges.py`, `tests/test_evolution_v2_real_numerical.py`.

**Interfaces:** Add:

```python
def load_active_frozen_pair(
    self,
    *,
    tasks: Sequence[ContextTask],
) -> tuple[FrozenNumericalArtifactsV2, str]: ...

def numerical_evolve_payload(
    config_payload: Mapping[str, object],
    seed_payload: Mapping[str, object],
    task_manifest_payload: Mapping[str, object],
    output: Path,
    *,
    input_sha256s: Mapping[str, str],
    host_runtime: object,
    llm_client: LLMClient,
) -> dict[str, object]: ...

def build_real_host(
    manifest: RealEvolutionManifestV2,
    *, repo_root: Path, output_dir: Path,
) -> RealHostRuntimeV2: ...
```

`RealHostRuntimeV2` owns the complete 80/20 task set, a cache-backed `ForecastStore`, the `CodexCLIClient(gpt-5.6-luna, medium)`, read-only retrieval skills, real retrieval/decision factories, resource reporting, and `close()`.

- [ ] **Step 1: Write RED tests for the frozen pair loader.** Run a deterministic P2 fixture, load its active pair, and assert exact bundle/release/registry/task-envelope identities. Mutate or duplicate the matching frozen-pair object and require rejection. Verify seed-only completion loads correctly.
- [ ] **Step 2: Implement the loader inside `NumericalQDRunStore`.** Validate completed numerical state, resolve the active Bundle, find exactly one catalog `FROZEN_PAIR` whose supply/registry identities match it, parse and restore its envelope against all 100 committed tasks, and return the typed pair plus object SHA. Do not expose private object paths.
- [ ] **Step 3: Write RED tests for payload invocation and Host construction.** Prove a derived config may live semantically under the root without triggering input/output overlap, verify ForecastStore identity and Codex model/effort, assert only frozen Train/Dev IDs are loaded, and assert `close()` closes runtime resources.
- [ ] **Step 4: Implement `numerical_evolve_payload`.** Share parsing and adapter construction with the existing path API, use caller-supplied verified SHA records, and retain existing path preflight for the old CLI. Reparse derived config with `NumericalQDConfigV2.from_payload` before execution.
- [ ] **Step 5: Implement the real Host.** Reuse the repository's `read_policy_file`, screening loader, TSFM runtime registry, `ForecastStore`, `RetrievalSkillLibrary`, `TwoStageRetrievalAgent`, and `DecisionAgent`. Use cache-only forecast behavior for the admitted historical cache; an actual cache miss is a recorded real-stage failure, not a synthetic forecast.
- [ ] **Step 6: Run GREEN.** Run the new numerical tests plus `tests/test_evolution_v2_numerical_cli.py` and `tests/test_evolution_v2_numerical_runner.py`.
- [ ] **Step 7: Commit.** `git commit -m "feat(real): bridge numerical host output"`.

### Task 4: P3 real cooperative bridge and sealed closure

**Files:** Extend `evolving_loop/v2/real/bridges.py`, `evolving_loop/v2/real/host.py`; create `tests/test_evolution_v2_real_cooperative.py`.

**Interfaces:** Produce:

```python
def run_real_cooperative(
    *, p2: FrozenNumericalArtifactsV2, host: RealHostRuntimeV2,
    config_payload: Mapping[str, object], output_dir: Path,
) -> CooperativeRunResultV2: ...

def load_sealed_bundle_closure(
    root: Path,
    *, tasks: tuple[ContextTask, ...], host: RealHostRuntimeV2,
) -> P3BundleClosureV2: ...
```

`P3BundleClosureV2` contains the exact active Bundle, catalog-resolved numerical/retrieval/decision objects, closed candidate Bundles available for P5 archive replay, acceptance evidence, proposal-space manifest, and runtime identity.

- [ ] **Step 1: Write RED bridge tests.** Assert P3 receives the exact P2 pair, evaluates only the real Train4/Dev1 projection while its numerical registry remains bound to all 100 tasks, and uses the real shared Codex client in retrieval/decision factories.
- [ ] **Step 2: Persist proposal-space closure at the root bridge.** Write one canonical immutable P3 handoff manifest containing numerical alternatives, decision prompt choices/settings cycle, catalog identities, and Host runtime fingerprint. Verify its fields against P3's existing `input_sha256s` commitments so resume rejects drift without changing the P3 runner schema.
- [ ] **Step 3: Implement verified P3 closure loading.** Revalidate completion/checkpoint/progress, resolve every active coordinate and closed concrete candidate from the catalog, verify acceptance evidence where present, and never synthesize a second archive Bundle. If fewer than two concrete Bundles are available, mark the later P5 handoff unavailable with a stable reason.
- [ ] **Step 4: Implement real factories.** Adapt the existing package runner factories to P3 signatures `retrieval_factory(genome, skills)` and `decision_factory(module)`. Both use the shared Luna-medium client and read-only skills.
- [ ] **Step 5: Run GREEN.** Run the new test plus all `tests/test_evolution_v2_cooperative_*.py` files.
- [ ] **Step 6: Commit.** `git commit -m "feat(real): bridge cooperative bundle"`.

### Task 5: Production P4 and P5 Host cases

**Files:** Create `evolving_loop/v2/source/bridge.py`, `evolving_loop/v2/protocol/bridge.py`, `tests/test_evolution_v2_real_source_bridge.py`, `tests/test_evolution_v2_real_protocol_bridge.py`; modify package exports and `evolving_loop/v2/source/runner.py` only if needed to remove the implicit `vars(case)` requirement.

**Interfaces:** Produce:

```python
def build_source_case_from_p3(
    closure: P3BundleClosureV2,
    host: RealHostRuntimeV2,
    *, source_seed: SourceVariantV2, input_digest: str,
    empty_skill_path: Path,
) -> SourceBundleCaseV2: ...

def build_protocol_case_from_p3(
    closure: P3BundleClosureV2,
    host: RealHostRuntimeV2,
    *, hard_limit_seconds: int,
) -> ProtocolRunCaseV2: ...
```

The source case has exact Train4/Dev1 tasks, a byte-identical seed Bundle from P3, reconstructing catalog/adapters/pipeline factories, and `SourceMetaEvaluatorV2`. The protocol case contains `config`, `input_manifest`, `ProtocolRuntimeRegistry`, and `CompatibilityHostInputsV2` with `fixture_scope="ordinary"`.

- [ ] **Step 1: Write P4 RED tests.** Assert every `catalog_factory()` reconstructs the same P3 Bundle, factories use real Host callables, source protocol/runtime commitments match, tasks are entity-disjoint real projections, and production modules contain no `tests` or `FakeLLMClient` import.
- [ ] **Step 2: Implement the P4 bridge.** Extract the structural pattern from the test builder without importing it. Use `NumericalCoordinateAdapter`, `RetrievalCoordinateAdapter`, `DecisionCoordinateAdapter`, `CooperativePipelineAdapter`, and `SourceMetaEvaluatorV2`. Source seed policy is the admitted versioned policy source, not a test-controlled improving variant.
- [ ] **Step 3: Write P5 RED tests.** Assert exact two-Bundle corpus closure, four verifier fixtures, safe inference projections, empty Public SHA list, ordinary scope, and rejection of `changed_history_json`. Assert P4 provenance cannot alter the P3 bundle under test.
- [ ] **Step 4: Implement the P5 bridge.** Select the active P3 Bundle plus one distinct closed concrete P3 Bundle; require both catalog closures. If a distinct second Bundle does not exist, return stable `p5_handoff_unavailable` instead of fabricating one. Build real `ProtocolHostInputs`, `CompatibilityCorpusV2`, `CompatibilityHostInputsV2`, seed protocol, artifact envelopes, and replacement templates directly; call the generic runner, never `dispatch_protocol`.
- [ ] **Step 5: Run GREEN.** Run both new bridge tests plus the existing source and protocol test suites.
- [ ] **Step 6: Commit.** `git commit -m "feat(real): add source and protocol bridges"`.

### Task 6: CLI integration, root end-to-end tests, and manifests

**Files:** Modify `evolving_loop/v2/cli.py`, `evolving_loop/v2/real/runner.py`; create `configs/evolution_v2/real/real-30m-toto-balanced-v3.json`, `configs/evolution_v2/real/real-1h-toto-balanced-v3.json`, `configs/evolution_v2/real/source-seed.json`, `tests/test_evolution_v2_real_cli.py`; modify `README.md` with one command and artifact location.

**Interfaces:** Register:

```text
python -m evolving_loop.v2 real-evolve --manifest PATH --output-dir PATH
```

Dispatch loads canonical manifest relative to repository root, verifies every file SHA and runtime identity before output creation, builds `RealHostRuntimeV2`, runs `run_real_evolution`, always closes Host resources, and emits canonical result JSON to stdout.

- [ ] **Step 1: Write RED CLI tests.** Cover fresh complete run, byte-identical completed resume, changed manifest, overlapping output/input path, invalid runtime identity, Public contamination, child failure, and no production import from test/smoke modules.
- [ ] **Step 2: Implement CLI dispatch and cleanup.** Catch existing validation/I/O exceptions, return code 2 with concise error, and keep the existing numerical/cooperative/source/protocol commands unchanged.
- [ ] **Step 3: Write real manifests and source seed.** Bind the verified Toto-balanced v3 split, champion, forecast sources/cache identity, retrieval seed/skills, runtime fingerprints, and exact profile. Add a canonical `SourceVariantV2` seed whose `policy.py` returns the safe default arm and whose provenance commits the admitted versioned method sources. Include the explicit label-informed split opt-in and no secrets or absolute paths.
- [ ] **Step 4: Run focused integration.** Run:

```bash
python -m pytest -q \
  tests/test_evolution_v2_real_contracts.py \
  tests/test_evolution_v2_real_runner.py \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cooperative.py \
  tests/test_evolution_v2_real_source_bridge.py \
  tests/test_evolution_v2_real_protocol_bridge.py \
  tests/test_evolution_v2_real_cli.py
```

- [ ] **Step 5: Run the existing six-project regression gate.** Run the numerical, cooperative, source, protocol, V2 kernel, and V2 CLI test files already used by the branch gate. Require zero failures.
- [ ] **Step 6: Commit.** `git commit -m "feat(real): expose bounded evolution CLI"`.

### Task 7: Real canary and requested 30-minute evolution

**Files:** No source edits unless the canary exposes a reproducible defect; create fresh run directories only under `runs/evolution_v2/`.

**Interfaces:** Operational commands use the feature worktree's `.venv/bin/python` and the canonical module CLI. They never reuse historical output or authority directories.

- [ ] **Step 1: Run final preflight.** Verify clean worktree, exact manifest SHA, Codex CLI availability, authenticated one-request probe through `CodexCLIClient`, forecast-store identity, task counts 80/20, and zero Public materialization. Do not print credentials or cache contents.
- [ ] **Step 2: Run a short real canary.** Use a fresh canary manifest that reduces proposal/generation counts, retains real data/runtime/model paths, and supplies one known nonregressing concrete P3 alternate solely to exercise the full bridge. Require all four sealed stages, at least one real model call or recorded fallback, `public_test_accessed=false`, and successful read-only resume. A handoff-unavailable result means the canary did not pass.
- [ ] **Step 3: Diagnose canary failures systematically.** Reproduce with the smallest focused command, add a failing regression test, implement the minimal fix, rerun the focused and six-project gates, and commit the fix before retrying with a new canary directory.
- [ ] **Step 4: Start the requested real 30-minute run.** Run:

```bash
.venv/bin/python -m evolving_loop.v2 real-evolve \
  --manifest configs/evolution_v2/real/real-30m-toto-balanced-v3.json \
  --output-dir runs/evolution_v2/real-30m-luna-medium-20260912
```

- [ ] **Step 5: Validate the result.** Re-run the identical command to exercise completed/closed-boundary resume, verify all linked SHA identities, report per-stage time and proposal mechanism, list accepted/rejected/seed-only outcomes, and explicitly report whether the root is complete or scientifically incomplete.
- [ ] **Step 6: Commit only any documentation needed for the actual command/result.** Use `docs(real): record 30m evolution command`; do not commit generated run caches or credentials.
