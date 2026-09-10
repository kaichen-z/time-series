# Evolution V2 Numerical Quality-Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real, resumable Numerical Dictionary/Supply self-evolution loop with hybrid proposals, history-only MAP-Elites, constrained NSGA-II, checkpointed Hyperband, legacy Numerical execution adapters, frozen Supply export, and V2 Kernel promotion.

**Architecture:** New modules under `evolving_loop/v2/numerical_qd` own Project 2 contracts and algorithms while treating the existing Numerical runtime as a frozen adapter. A counter-based deterministic stream drives sampling, immutable Project 2 artifacts checkpoint every completed rung, and only the V2 Kernel may accept the atomic Numerical Supply/registry pair. The full three-Agent production runner remains closed until Project 3.

**Tech Stack:** Python 3 standard library, frozen dataclasses, strict canonical JSON/SHA-256, existing `BudgetLedger`/`EvolutionKernel`, existing Numerical package materializer/registry/runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-evolution-v2-numerical-qd-design.md`

## Global Constraints

- Do not modify the behavior or artifact bytes of legacy Numerical, Retrieval, Decision, package co-evolution, Public evaluation, or frozen release commands.
- All Project 2 JSON uses strict finite V2 canonical serialization; unknown fields, duplicate keys, NaN/Infinity, paths, callbacks, clients, and live Python objects fail closed.
- Proposers receive only primitive Parent data, archive summaries, sanitized Train feedback, remaining proposal budget, and the allowed mutation schema.
- Dev values appear only in V2 Kernel decision evidence; they never enter QD entries, scheduler state, prompts, mutation-policy credit, progress, or proposal memory.
- Public data is never opened by Project 2. Every completion records `public_test_accessed: false`.
- History-only descriptor thresholds and frequency mappings are immutable, versioned, and fingerprinted into the run protocol.
- MAP-Elites cells retain at most four constrained Pareto entries unless a strict config explicitly lowers the capacity.
- Parent category weights are exactly 40 underexplored, 30 elite, 20 failure-matched specialist, and 10 non-elite lineage stepping stone, normalized after removing empty categories.
- Hyperband brackets are exactly `explore: 8 -> 32 -> 80`, `confirm: 32 -> 80`, and `replay: 80`; Dev20 is not a Hyperband resource.
- Formal profiles use exactly 14,400 seconds and a 0.2 finalization reserve. Already consumed work is always charged.
- Constraint feasibility precedes NSGA-II dominance. Exact ties end with ascending artifact SHA.
- An accepted Numerical transition changes `(numerical_release_sha256, numerical_registry_sha256)` atomically; rejection returns the exact Parent object and bytes.
- LLM failure is a closed, charged provider attempt followed by deterministic fallback when budget remains. Deterministic CI never requires network access.
- DGM mutation of Evolver/Harness Python source remains Project 4 and L1 protocol migration remains Project 5.
- Each task follows RED/GREEN TDD, commits only its declared scope, and runs its dependency regression set before review.

---

### Task 1: Strict Numerical QD artifact contracts

**Files:**
- Create: `evolving_loop/v2/numerical_qd/__init__.py`
- Create: `evolving_loop/v2/numerical_qd/contracts.py`
- Create: `tests/test_evolution_v2_numerical_contracts.py`

**Interfaces:**
- Produces: `NumericalMemberV2`, `NumericalInventoryV2`, `MorphologyCellV2`, `NumericalObjectiveVectorV2`, `ConstraintReportV2`, `NumericalQDEntryV2`, `NumericalGenomeV2`, `MutationOperatorStatsV2`, `NumericalMutationPolicyV2`, `NumericalProposerPromptV2`, and `NumericalEvaluationV2`.
- Produces: every artifact's `from_payload`, `to_payload`, `canonical_bytes`, and `fingerprint` methods.
- Consumes: `canonical_v2_bytes`, `fingerprint_payload`, `require_sha256`, and recursive frozen-value helpers from `evolving_loop.v2.contracts`.

- [ ] **Step 1: Write failing exact-schema and immutability tests**

```python
def test_numerical_genome_is_canonical_and_deeply_immutable():
    genome = valid_genome()
    assert NumericalGenomeV2.from_payload(genome.to_payload()) == genome
    assert genome.fingerprint() == fingerprint_payload(genome.to_payload())
    with pytest.raises(TypeError):
        genome.runtime_fingerprints["python"] = "f" * 64

@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_objective_contract_rejects_nonfinite_or_boolean_numbers(value):
    with pytest.raises((TypeError, ValueError)):
        NumericalObjectiveVectorV2(value, 1.0, 1.0, 1.0, 1.0)
```

Also test every unknown/missing field, invalid SHA, duplicate member ID, unsupported family/status, unsorted or duplicate parents/cells, invalid constraint names, negative counters, and post-construction mutation of nested payloads.

- [ ] **Step 2: Run the contract tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_contracts.py`

Expected: collection fails because `evolving_loop.v2.numerical_qd.contracts` does not exist.

- [ ] **Step 3: Implement the closed dataclasses**

Use these exact public shapes:

```python
@dataclass(frozen=True, slots=True)
class MorphologyCellV2:
    trend: Literal["low", "medium", "high"]
    seasonality: Literal["none", "short", "long"]
    intermittency: Literal["low", "high"]
    regime: Literal["stable", "shift"]
    horizon: Literal["short", "medium", "long"]
    family: Literal["statistical", "tsfm", "combined", "program"]

@dataclass(frozen=True, slots=True)
class NumericalObjectiveVectorV2:
    mean_capped_smae: float
    mean_capped_srmse: float
    p95_capped_srmse: float
    mean_raw_joint_error: float
    normalized_execution_cost: float

@dataclass(frozen=True, slots=True)
class ConstraintReportV2:
    feasible: bool
    violations: tuple[str, ...]
```

`NumericalMemberV2` contains member ID, family, source/policy SHA, sorted Parent IDs, sorted applicability-cell fingerprints, and status `active|specialized|quarantined`. `NumericalInventoryV2` rejects duplicate IDs and requires at least one active member. `NumericalGenomeV2` uses the exact fields listed in the spec. Mutation statistics store non-negative integer attempts/feasible/promotions/insertions/credit. Evaluation binds candidate/task/split/runtime/protocol identities, rung, objectives, constraints, cells, Train diagnostic categories, cache SHAs, and finite `ResourceUse` payload.

- [ ] **Step 4: Verify canonical round trips across fresh interpreters**

Run the test helper twice with `subprocess.run([sys.executable, "-c", script])` and require identical bytes and SHA output.

- [ ] **Step 5: Run Task 1 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_contracts.py tests/test_evolution_v2_contracts.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add evolving_loop/v2/numerical_qd/__init__.py evolving_loop/v2/numerical_qd/contracts.py tests/test_evolution_v2_numerical_contracts.py
git commit -m "feat(numerical-qd): add strict artifacts"
```

### Task 2: Strict Project 2 config and history-only descriptors

**Files:**
- Create: `evolving_loop/v2/numerical_qd/descriptors.py`
- Create: `evolving_loop/v2/numerical_qd/config.py`
- Create: `tests/test_evolution_v2_numerical_descriptors.py`
- Create: `tests/test_evolution_v2_numerical_config.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `DescriptorPolicyV2.from_payload(payload)`, `NumericalQDConfigV2.from_payload(payload)`, `load_numerical_qd_config(path)`, and `describe_history(history, horizon, frequency, family, policy) -> MorphologyCellV2`.
- Consumes: Task 1 `MorphologyCellV2`; Project 1 `KernelProtocolCommitment` and `BudgetPlan` conventions.

- [ ] **Step 1: Write failing descriptor-invariance tests**

```python
def test_descriptor_uses_history_not_future_or_labels():
    first = describe_history(HISTORY, 12, "D", "statistical", policy())
    second = describe_history(tuple(HISTORY), 12, "D", "statistical", policy())
    assert first == second

def test_descriptor_threshold_edges_are_closed_and_deterministic():
    assert describe_history(constant_history(), 4, "D", "program", policy()).trend == "low"
    assert describe_history(intermittent_history(0.30), 4, "D", "program", policy()).intermittency == "high"
```

Test empty/short/nonfinite histories, invalid horizon/frequency/family, all bins, constant variance, configured seasonal lags, and that future/label sentinels cannot be supplied to the function.

- [ ] **Step 2: Write failing exact-config tests**

The config has exact top-level fields `schema_version`, `profile`, `seed`, `kernel_protocol`, `runtime_fingerprints`, `budget`, `fixed_bundle_components`, `descriptor_policy`, `mutation`, `map_elites`, `hyperband`, `proposer`, and `adapter`. `fixed_bundle_components` contains canonical Retrieval release, Decision policy, Harness policy, initial archive snapshot, and scheduler-state SHA values used to construct the generation-zero V2 Bundle after seed registry materialization. Reject additional fields and require formal `(hard_limit_seconds, finalization_reserve_fraction) == (14400, 0.2)`.

- [ ] **Step 3: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_descriptors.py tests/test_evolution_v2_numerical_config.py`

Expected: imports fail because config and descriptor modules are absent.

- [ ] **Step 4: Implement exact descriptor formulas**

Use policy-bound finite thresholds with these shipped defaults:

```text
trend score = abs(mean(last quarter) - mean(first quarter)) / max(pstdev(history), 1e-12)
trend: low < 0.15; medium < 0.75; high otherwise
intermittency: high when zero_fraction >= 0.30
regime: shift when abs(mean(second half) - mean(first half)) / max(pstdev, 1e-12) >= 0.50
horizon ratio = horizon / len(history): short <= 0.10; medium <= 0.30; long otherwise
seasonality: maximum finite autocorrelation over configured lags; none < 0.30,
             short when winning lag <= 24, long otherwise
```

Default seasonal lags are `H:[24,168]`, `D:[7,30]`, `W:[52]`, `M:[12]`, `Q:[4]`; unknown frequencies have no candidate lag. A lag requires at least two complete repeats. Descriptor output contains no score or history value.

- [ ] **Step 5: Implement strict config parsing**

Freeze every nested mapping. Validate positive capacities and proposal limits, exact sampling weights `40/30/20/10`, exact bracket rungs, unique mutation operators, provider `hybrid|deterministic`, and finite resource ceilings. Config paths are read strictly with duplicate-key rejection and canonical bytes.

- [ ] **Step 6: Run Task 2 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_config.py tests/test_evolution_v2_numerical_descriptors.py tests/test_evolution_v2_numerical_contracts.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add evolving_loop/v2/numerical_qd/config.py evolving_loop/v2/numerical_qd/descriptors.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_config.py tests/test_evolution_v2_numerical_descriptors.py
git commit -m "feat(numerical-qd): classify history niches"
```

### Task 3: Typed mutation grammar and hybrid proposers

**Files:**
- Create: `evolving_loop/v2/numerical_qd/mutation.py`
- Create: `evolving_loop/v2/numerical_qd/proposers.py`
- Create: `tests/test_evolution_v2_numerical_mutation.py`
- Create: `tests/test_evolution_v2_numerical_proposers.py`
- Modify: `evolving_loop/v2/numerical_qd/contracts.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `MutationProposalV2.from_payload`, `NormalizedProposalBatchV2`, `apply_mutation(parent, proposal) -> MutationResultV2`, `primitive_proposer_request(...) -> dict`, `DeterministicProposalProvider.propose(request) -> NormalizedProposalBatchV2`, `LLMProposalProvider.propose(request) -> NormalizedProposalBatchV2`, and `HybridProposalProvider.propose(request) -> NormalizedProposalBatchV2`.
- Consumes: Task 1 inventory/genome/policy/prompt artifacts and `common.llm.LLMClient` through a structural client protocol.

- [ ] **Step 1: Write one failing test for every tagged operation**

```python
@pytest.mark.parametrize("operation", ALL_VALID_OPERATION_PAYLOADS)
def test_each_mutation_round_trips_and_changes_only_its_owned_fields(operation):
    proposal = MutationProposalV2.from_payload(operation)
    result = apply_mutation(parent_state(), proposal)
    assert result.changed_fields == EXPECTED_FIELDS[proposal.operator]
```

Test `add`, `repair`, `fork`, `combine`, `route`, `specialize`, `crossover`, `remove`, `quarantine`, and `policy_tune`; reject wrong tag keys, excess parents, duplicate IDs, last-active-member removal, inventory overflow, undeclared cell, invalid source SHA, arbitrary code in non-source fields, and multi-target ownership.

- [ ] **Step 2: Write failing proposer-boundary tests**

Intercept the argument at each provider's actual `propose` method. Recursively assert JSON primitives and exact keys. Insert unique Dev/Public/future/path/callback/Kernel/Store sentinels one at a time and require request construction to reject them before provider execution.

- [ ] **Step 3: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_mutation.py tests/test_evolution_v2_numerical_proposers.py`

Expected: imports fail because mutation and proposer modules are absent.

- [ ] **Step 4: Implement the strict tagged union**

Use these exact operation key sets:

```python
OPERATION_KEYS = {
    "add": {"operator", "reason", "member"},
    "repair": {"operator", "reason", "member_id", "replacement"},
    "fork": {"operator", "reason", "member_id", "child"},
    "combine": {"operator", "reason", "parent_ids", "child"},
    "route": {"operator", "reason", "parent_ids", "child"},
    "specialize": {"operator", "reason", "member_id", "applicability_cells"},
    "crossover": {"operator", "reason", "parent_ids", "child"},
    "remove": {"operator", "reason", "member_id"},
    "quarantine": {"operator", "reason", "member_id"},
    "policy_tune": {"operator", "reason", "prompt", "credit_delta"},
}
```

Member source is a SHA reference; raw source is persisted by the Host before parsing the operation. `apply_mutation` is atomic and returns a new frozen inventory/policy/prompt state or raises without mutating Parent.

- [ ] **Step 5: Implement provider behavior**

The deterministic provider chooses an allowed operation from sorted feasible actions using the request's supplied counter draw. The LLM provider sends only canonical request JSON and the active prompt, caps response bytes, performs one strict JSON parse, and never retries internally. A raw LLM response uses exact keys `source_candidates` and `proposals`: source candidates contain a request-local ID plus code, while proposals may refer only to those local IDs. The Host validates each source with the existing parser/sandbox gate, computes canonical source SHAs, replaces local IDs with those SHAs, and only then constructs `MutationProposalV2`. `NormalizedProposalBatchV2` returns the immutable `(source_sha256, source_bytes)` artifacts to the caller; Task 8 persistence stores and rereads them before the proposal becomes eligible for evaluation. Hybrid records the LLM attempt result and uses deterministic fallback after unavailable/timeout/malformed/empty output. Provider result includes normalized proposals, source artifacts, resource use, provider identity, and closed failure reason.

- [ ] **Step 6: Prove prompt/policy evolution is Train-only**

Add a two-generation test: one feasible archive insertion increments integer credit and produces new policy/prompt SHA; a Dev comparison with a unique sentinel cannot change either SHA and never appears in the next proposer request.

- [ ] **Step 7: Run Task 3 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_mutation.py tests/test_evolution_v2_numerical_proposers.py tests/test_evolution_v2_numerical_contracts.py`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add evolving_loop/v2/numerical_qd/contracts.py evolving_loop/v2/numerical_qd/mutation.py evolving_loop/v2/numerical_qd/proposers.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_mutation.py tests/test_evolution_v2_numerical_proposers.py
git commit -m "feat(numerical-qd): add hybrid mutations"
```

### Task 4: Deterministic constrained NSGA-II

**Files:**
- Create: `evolving_loop/v2/numerical_qd/nsga2.py`
- Create: `tests/test_evolution_v2_numerical_nsga2.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `constraint_compare(left, right) -> int`, `non_dominated_fronts(entries)`, `crowding_distances(front)`, and `select_survivors(entries, capacity)`.
- Consumes: Task 1 `NumericalObjectiveVectorV2`, `ConstraintReportV2`, and QD entry identities.

- [ ] **Step 1: Write failing dominance/front tests**

```python
def test_feasible_always_dominates_infeasible():
    assert constraint_compare(feasible(), infeasible("coverage")) == -1

def test_pareto_front_keeps_tradeoffs_without_scalarization():
    assert tuple(item.sha for item in non_dominated_fronts(TRADEOFFS)[0]) == EXPECTED
```

Cover feasible/infeasible ordering, fewer violations, identical objectives, strict one-objective improvement, tradeoffs, duplicate SHA rejection, finite objective enforcement, and input-order invariance.

- [ ] **Step 2: Write failing crowding/capacity tests**

Require boundary entries to receive positive infinity internally, zero-range objectives to contribute zero, stable normalized distances for every objective, and SHA ordering for exact distance ties. The serialized survivor record must not contain Infinity; it stores deterministic rank/order only.

- [ ] **Step 3: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_nsga2.py`

Expected: import fails because `nsga2.py` is absent.

- [ ] **Step 4: Implement pairwise dominance and fronts**

Compare constraint feasibility before objective values. Feasible Pareto dominance means no objective is worse and at least one is better. Infeasible comparison uses `(len(violations), sorted(violations), artifact_sha)` only; objective values cannot rescue an infeasible candidate.

- [ ] **Step 5: Implement crowding and deterministic truncation**

Sort each objective ascending, mark endpoints as internal infinity, add `(next - previous) / (max - min)` for nonzero ranges, then choose descending crowding with ascending SHA as the final tie-break. Never serialize internal infinity.

- [ ] **Step 6: Run Task 4 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_nsga2.py tests/test_evolution_v2_numerical_contracts.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add evolving_loop/v2/numerical_qd/nsga2.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_nsga2.py
git commit -m "feat(numerical-qd): rank constrained fronts"
```

### Task 5: Bounded MAP-Elites archive and counter stream

**Files:**
- Create: `evolving_loop/v2/numerical_qd/map_elites.py`
- Create: `tests/test_evolution_v2_numerical_map_elites.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `CounterRandom(seed, stream, counter=0)`, `NumericalQDArchive.from_payload`, `archive.insert(entries)`, `archive.sample_parent(feedback, random_stream)`, and immutable `ArchiveInsertionV2`.
- Consumes: Task 1 cells/evaluations and Task 4 `select_survivors`.

- [ ] **Step 1: Write failing deterministic-stream tests**

```python
def test_counter_stream_replays_from_exact_counter():
    first = CounterRandom(7, "parents")
    prefix = tuple(first.randbelow(10_000) for _ in range(5))
    resumed = CounterRandom(7, "parents", counter=5)
    assert resumed.randbelow(10_000) == first.randbelow(10_000)
```

Hash canonical `{"seed":seed,"stream":stream,"counter":counter}` and use rejection sampling rather than modulo bias. Reject empty stream names, boolean seeds/counters, negative counters, and nonpositive bounds.

- [ ] **Step 2: Write failing archive insertion tests**

Test one candidate in multiple cells, immutable append history, idempotent same-entry insertion, conflicting same SHA rejection, per-cell capacity, constrained survivor selection, non-elite stepping-stone retention, and canonical snapshot identity.

- [ ] **Step 3: Write failing parent-category tests**

Construct archives where each category has one unique candidate and feed exact deterministic draws covering 0-39, 40-69, 70-89, and 90-99. Test each empty-category combination and require integer re-normalization with no category accidentally retaining probability.

- [ ] **Step 4: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_map_elites.py`

Expected: import fails because `map_elites.py` is absent.

- [ ] **Step 5: Implement archive and sampling**

Archive payload contains schema, capacity, sorted cell records, global immutable entry table, insertion log, and snapshot parent SHA. Underexplored means the lowest visit count; elites are rank-zero cell survivors; failure matched intersects sanitized failure categories; stepping stones are retained lineage members not presently elite. Sort every candidate set by artifact SHA before drawing.

- [ ] **Step 6: Run Task 5 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_map_elites.py tests/test_evolution_v2_numerical_nsga2.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add evolving_loop/v2/numerical_qd/map_elites.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_map_elites.py
git commit -m "feat(numerical-qd): retain niche elites"
```

### Task 6: Checkpointed Hyperband brackets and cache identities

**Files:**
- Create: `evolving_loop/v2/numerical_qd/hyperband.py`
- Create: `tests/test_evolution_v2_numerical_hyperband.py`
- Modify: `evolving_loop/v2/numerical_qd/contracts.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `HyperbandBracketV2`, `HyperbandRungV2`, `HyperbandStateV2`, `fixed_rung_manifest(task_groups, resource, split_sha256, protocol_sha256) -> RungManifestV2`, `evaluation_cache_key(candidate, task, split, metric, descriptor, runtime, protocol, adapter) -> str`, `choose_bracket(candidate_count, cache_coverage, remaining_budget, config) -> HyperbandBracketV2`, and `advance_hyperband(state, manifest, evaluations, budget_outcome) -> HyperbandAdvanceV2`.
- Consumes: Project 1 `BudgetLedger`/`ResourceUse`; Task 1 evaluations; Task 4 ranking.

- [ ] **Step 1: Write failing exact-bracket tests**

```python
@pytest.mark.parametrize("name,rungs", [
    ("explore", (8, 32, 80)),
    ("confirm", (32, 80)),
    ("replay", (80,)),
])
def test_registered_brackets_have_exact_rungs(name, rungs):
    assert HyperbandBracketV2.registered(name).resources == rungs
```

Reject renamed/extra/descending rungs, partial entity groups, duplicate task IDs, task manifests with the wrong split/protocol, and a requested resource larger than the committed Train universe.

- [ ] **Step 2: Write failing promotion and cache tests**

For three Children require explore survivors `3 -> 2 -> 1`. Larger populations use the configured integer reduction factor with at least one survivor. Cache identity must change independently for candidate, task bytes, split, metric, descriptor, runtime, protocol, or adapter SHA. Missing/malformed/mismatched cache rows consume normal work and cannot be promoted as hits.

- [ ] **Step 3: Write failing budget/deadline/resume tests**

Use `FakeClock` and real `BudgetLedger`. Verify each rung reserves before evaluation, actual work closes even on callback failure, finalization reserve blocks a rung before opening, already completed immutable tasks are not charged twice, and resume rejects an open/partial rung.

- [ ] **Step 4: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_hyperband.py`

Expected: import fails because Hyperband contracts are absent.

- [ ] **Step 5: Implement bracket planning and immutable advancement**

`advance_hyperband` accepts a frozen rung manifest and one Host evaluation callback. It returns a new state plus immutable task/evaluation results; it never writes files. Selection calls Task 4 ranking. The caller owns persistence ordering in Task 8.

- [ ] **Step 6: Run Task 6 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_hyperband.py tests/test_evolution_v2_budget.py tests/test_evolution_v2_numerical_nsga2.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add evolving_loop/v2/numerical_qd/hyperband.py evolving_loop/v2/numerical_qd/contracts.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_hyperband.py
git commit -m "feat(numerical-qd): allocate Hyperband rungs"
```

### Task 7: Existing Numerical runtime adapter and frozen Supply export

**Files:**
- Create: `evolving_loop/v2/numerical_qd/adapters.py`
- Create: `tests/test_evolution_v2_numerical_adapters.py`
- Modify: `evolving_loop/v2/numerical_qd/contracts.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `LegacyNumericalAdapter`, `ImportedNumericalSeedV2`, `MaterializedNumericalChildV2`, `FrozenNumericalArtifactsV2`, `FrozenNumericalRegistryEnvelopeV2`, `import_numerical_seed(...)`, `evaluate_numerical_child(...)`, and `freeze_qd_supply(...)`.
- Consumes: exact legacy `NumericalSupplyRelease`, `NumericalCoordinateCandidate`, `NumericalPackageProposer`, `NumericalPackageMaterializer`, `FrozenNumericalPackageRegistry`, `ContextTask`, and package metric/evaluation adapters.

- [ ] **Step 1: Write failing seed-import and no-rewrite tests**

Snapshot the supplied legacy Supply, registry manifest, source methods, runtime manifests, and representative frozen release bytes/mtimes. Import them, round-trip the V2 envelope, and assert every source snapshot remains unchanged.

- [ ] **Step 2: Write failing source/parser boundary tests**

Use one legal finite forecasting function and hostile candidates containing kernel imports, filesystem writes, dynamic import, `eval`, escape-hatch dunders, wrong signature, missing docstring, broad-exception fallback, wrong horizon, NaN, and a native-crash fixture. Require rejection or typed invalid outcome without changing Kernel/legacy files. Reuse `MethodModule`, `check_code`, and `IsolatedForecastRuntime`; do not introduce a second permissive parser.

- [ ] **Step 3: Write failing real materialization tests**

Build exactly 80 grouped Train plus 20 Dev deterministic `ContextTask` fixtures with cheap finite forecasts and a real existing Supply anchor. Exercise adapter construction of `NumericalRecipeFit`, `NumericalPackageMaterializer.materialize`, and `build_package_registry`. Assert release fingerprint equals registry `release_sha256`, every task is present exactly once, and each forecast is finite with exact horizon.

- [ ] **Step 4: Write failing QD projection tests**

Given more than four occupied cells, require a deterministic greedy projection that retains the anchor plus at most one alternative per existing family and at most four alternatives total. Order by newly covered cells, constrained rank, descending crowding, ascending normalized cost, then SHA. Bind QD snapshot, genome, policy, prompt, descriptor, and selected source identities in `source_fingerprints`.

- [ ] **Step 5: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_adapters.py`

Expected: import fails because `adapters.py` is absent.

- [ ] **Step 6: Implement the adapter without changing legacy modules**

Production construction wraps existing objects; test construction may inject a materializer/evaluator with the same typed boundary. All task inputs passed to candidate materialization are label-free projections. Host evaluation alone receives trusted futures and emits aggregate records without futures. Serialize every `NumericalForecastPackage` into a strict content-addressed V2 package envelope and bind the sorted package SHAs in `FrozenNumericalRegistryEnvelopeV2`; resume reconstructs a fresh `FrozenNumericalPackageRegistry` from verified tasks and packages rather than pickling a live registry.

- [ ] **Step 7: Run Task 7 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_adapters.py tests/test_package_numerical_supply.py tests/test_package_numerical_evolution.py`

Expected: all tests pass and no legacy artifact is modified.

- [ ] **Step 8: Commit Task 7**

```bash
git add evolving_loop/v2/numerical_qd/adapters.py evolving_loop/v2/numerical_qd/contracts.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_adapters.py
git commit -m "feat(numerical-qd): adapt frozen supply"
```

### Task 8: Immutable Project 2 store and runner checkpoint

**Files:**
- Create: `evolving_loop/v2/numerical_qd/persistence.py`
- Create: `tests/test_evolution_v2_numerical_persistence.py`
- Modify: `evolving_loop/v2/numerical_qd/contracts.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`

**Interfaces:**
- Produces: `NumericalQDRunStore.create(root)`, immutable `write_object`, `write_source`, `write_proposal_attempt`, `write_task_result`, `append_qd_entry`, `write_rung`, `write_state`, root-authority `write_completion`, and `resume(config_sha256, input_sha256s, kernel_checkpoint_sha256, budget_checkpoint_sha256) -> NumericalQDCheckpointV2`.
- Produces: `NumericalQDCheckpointV2` cross-bound to Project 1 Kernel/budget checkpoint SHA values.
- Consumes: Task 1/5/6 canonical artifacts and Project 1 atomic/write-once helpers.

- [ ] **Step 1: Write failing exact-layout tests**

Require this Project 2 subtree beneath the V2 run:

```text
numerical_qd/manifest.json
numerical_qd/checkpoint.json
numerical_qd/objects/{sha256}.json
numerical_qd/sources/{sha256}.py
numerical_qd/proposals/{attempt_sha256}.json
numerical_qd/results/{candidate_sha256}/{task_sha256}.json
numerical_qd/rungs/{rung_sha256}.json
numerical_qd/archive/entries.jsonl
evaluation_complete.json
```

Test path traversal, wrong content SHA, noncanonical JSON, source digest mismatch, conflicting immutable retry, JSONL prefix corruption, and output creation inside legacy/source paths.

- [ ] **Step 2: Write failing crash-boundary tests**

Inject failure after every persistence step. Resume may ignore only an unreferenced hidden atomic temporary file. It must reject partial referenced task/rung/archive/policy state, changed RNG counter, mismatched Kernel/budget checkpoint, and missing immutable bytes. Completed rung resume appends no duplicate work.

- [ ] **Step 3: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_persistence.py`

Expected: import fails because `persistence.py` is absent.

- [ ] **Step 4: Implement durable writes and verified resume**

Use write-once files, fsync file and containing directory before authority publication, canonical readback, and append-only hash-chained QD entries. Checkpoint fields are exactly config/input identities, active Bundle/genome, QD snapshot, policy/prompt/Hyperband state, counter, completed operation SHAs, budget checkpoint SHA, Kernel checkpoint SHA, and its own body SHA. Project 2 completion is written once at the V2 run root after Kernel finalization; the Numerical subtree never writes a second completion file.

- [ ] **Step 5: Run Task 8 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_persistence.py tests/test_evolution_v2_store.py tests/test_evolution_v2_archive.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 8**

```bash
git add evolving_loop/v2/numerical_qd/persistence.py evolving_loop/v2/numerical_qd/contracts.py evolving_loop/v2/numerical_qd/__init__.py tests/test_evolution_v2_numerical_persistence.py
git commit -m "feat(numerical-qd): persist resumable search"
```

### Task 9: Complete self-evolution runner and V2 Kernel promotion

**Files:**
- Create: `evolving_loop/v2/numerical_qd/runner.py`
- Create: `tests/test_evolution_v2_numerical_runner.py`
- Modify: `evolving_loop/v2/numerical_qd/__init__.py`
- Modify: `evolving_loop/v2/kernel.py`

**Interfaces:**
- Produces: `run_numerical_qd(output_dir, config, seed_supply, task_manifest, adapter, llm_client=None, resume=False, stop_after=None) -> NumericalQDRunResultV2`.
- Consumes: Tasks 1-8 plus real `EvolutionKernel.reserve_evaluation`, `close_evaluation`, and `evaluate_transition`.

- [ ] **Step 1: Write a failing deterministic multi-generation integration test**

```python
def test_qd_runner_evolves_supply_policy_and_prompt(tmp_path):
    result = run_fixture_qd(tmp_path / "run", seed=7)
    assert result.status == "numerical_qd_complete"
    assert result.occupied_cells >= 2
    assert result.accepted_steps >= 1
    assert result.mutation_policy_sha256 != result.seed_mutation_policy_sha256
    assert result.proposer_prompt_sha256 != result.seed_proposer_prompt_sha256
    assert result.public_test_accessed is False
```

Use real algorithm modules and a deterministic typed adapter fixture, not mocked MAP/NSGA/Hyperband decisions.

- [ ] **Step 2: Write failing accept/reject and resume tests**

Run one accepted Numerical Child then one Dev-rejected Child. Require atomic release/registry change on acceptance, exact Parent object/bytes on rejection, identical uninterrupted versus stop/resume artifacts, unchanged completed-run mtimes, exact resource charges, and deterministic QD/archive/policy/prompt identities.

- [ ] **Step 3: Write failing provider and deadline tests**

Script one legal LLM response, malformed response, timeout, and missing client. Verify provider attempt accounting and deterministic fallback. Advance `FakeClock` to each bracket boundary and the finalization reserve; no stage may open after the deadline.

- [ ] **Step 4: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_runner.py`

Expected: import fails because `runner.py` is absent.

- [ ] **Step 5: Implement the orchestration loop**

The runner performs Parent sampling, primitive request construction, hybrid proposal, atomic mutation, fixed-rung evaluation, QD insertion, Train credit update, Supply projection, Dev comparison, and Kernel Numerical transition in that order. Persist and reread each authority boundary through Task 8. A generation with no feasible Child records a closed no-improvement step and preserves Parent.

- [ ] **Step 6: Bind accepted release references**

For Project 2 Numerical archive records, populate accepted release references with the accepted Supply SHA and registry SHA in sorted order. Do not change the Project 1 fake-run ruling for records that have no typed release artifact.

- [ ] **Step 7: Run Task 9 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_runner.py tests/test_evolution_v2_kernel.py tests/test_evolution_v2_numerical_*.py`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 9**

```bash
git add evolving_loop/v2/numerical_qd/runner.py evolving_loop/v2/numerical_qd/__init__.py evolving_loop/v2/kernel.py tests/test_evolution_v2_numerical_runner.py
git commit -m "feat(numerical-qd): run self-evolution"
```

### Task 10: Numerical-only CLI and strict profiles

**Files:**
- Modify: `evolving_loop/v2/cli.py`
- Create: `configs/evolution_v2/numerical_qd/smoke.json`
- Create: `configs/evolution_v2/numerical_qd/pilot.json`
- Create: `configs/evolution_v2/numerical_qd/formal.json`
- Create: `tests/build_evolution_v2_numerical_fixture.py`
- Create: `tests/test_evolution_v2_numerical_cli.py`

**Interfaces:**
- Produces: `python -m evolving_loop.v2 numerical-evolve --config ... --seed-supply ... --task-manifest ... --output-dir ...`.
- Consumes: Task 2 config loader, Task 7 production adapter construction, and Task 9 runner.

- [ ] **Step 1: Write failing parser/preflight tests**

Test required flags, canonical input loading, output new-or-exact-resume rules, bidirectional filesystem-identity overlap rejection, input SHA binding before output creation, deterministic smoke, hybrid pilot/formal selection, completed no-op resume, and that the existing fake `evolve` and `public-evaluate` behaviors remain unchanged.

- [ ] **Step 2: Write failing truthful-completion tests**

Require `numerical_qd_complete`, Supply/registry/Bundle/QD/policy/prompt identities, occupied cells, accepted/rejected counts, provider attempts, exact budget payload, `dev_accessed` boolean, and `public_test_accessed:false`. Reject any field claiming Retrieval, Decision, joint co-evolution, Public score, or real-model use when unavailable.

- [ ] **Step 3: Run the tests and observe RED**

Run: `pytest -q tests/test_evolution_v2_numerical_cli.py`

Expected: parser rejects the missing `numerical-evolve` command and profile files are absent.

- [ ] **Step 4: Implement CLI dispatch and configs**

Keep Project 1 `evolve` behavior unchanged. Resolve and hash inputs before creating output. `smoke` uses deterministic fixtures; `pilot` and `formal` construct existing real adapters and hybrid provider, while allowing deterministic fallback when LLM is absent. `formal` carries exact 14,400/0.2 bounds. Full `evolve --runner production` continues to fail with a Project 3 requirement.

- [ ] **Step 5: Run manual commands in fresh temporary directories**

```bash
qd_smoke_root="$(mktemp -d /tmp/evolution-v2-numerical-qd.XXXXXX)"
python tests/build_evolution_v2_numerical_fixture.py --output-dir "$qd_smoke_root/fixtures"
python -m evolving_loop.v2 numerical-evolve --config configs/evolution_v2/numerical_qd/smoke.json --seed-supply "$qd_smoke_root/fixtures/seed_supply.json" --task-manifest "$qd_smoke_root/fixtures/task_manifest.json" --output-dir "$qd_smoke_root/run"
python -m evolving_loop.v2 numerical-evolve --config configs/evolution_v2/numerical_qd/smoke.json --seed-supply "$qd_smoke_root/fixtures/seed_supply.json" --task-manifest "$qd_smoke_root/fixtures/task_manifest.json" --output-dir "$qd_smoke_root/run"
```

The committed fixture builder writes exactly 80 grouped Train and 20 Dev tasks plus a matching canonical seed Supply, contains no Public records, and refuses a nonempty output directory. Assert the second command is a verified no-op and source inputs retain bytes/mtimes.

- [ ] **Step 6: Run Task 10 tests**

Run: `pytest -q tests/test_evolution_v2_numerical_cli.py tests/test_evolution_v2_cli.py tests/test_evolution_v2_numerical_runner.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 10**

```bash
git add evolving_loop/v2/cli.py configs/evolution_v2/numerical_qd tests/build_evolution_v2_numerical_fixture.py tests/test_evolution_v2_numerical_cli.py
git commit -m "feat(numerical-qd): expose supply CLI"
```

### Task 11: Project 2 safety, compatibility, documentation, and full gate

**Files:**
- Create: `tests/test_evolution_v2_numerical_safety.py`
- Create: `tests/test_evolution_v2_numerical_compatibility.py`
- Create: `docs/evolution-v2-numerical-qd.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Project 2 public APIs and the Project 1/legacy gates.
- Produces: a truthful operator guide and fresh Project 2 exit evidence.

- [ ] **Step 1: Add hostile information-flow tests**

Inject unique future, Dev, Public, evaluator-label, filesystem-path, Kernel, Store, callback, and environment-secret sentinels independently. Snapshot every proposer request, QD object, checkpoint, prompt, mutation-policy record, progress line, cache row, and source artifact. Only trusted task evaluation may see futures; only Kernel decision evidence may persist raw Dev; Public and secret sentinels appear nowhere.

- [ ] **Step 2: Add algorithm mutation-sensitivity tests**

In isolated interpreter probes, disable constraint-first ordering, category normalization, SHA tie-break, cache-key binding, rung checkpoint binding, or Train-only policy credit one at a time. Require at least one named new test to fail for each mutation, proving the gates are non-vacuous.

- [ ] **Step 3: Add legacy compatibility tests**

Snapshot representative legacy Supply, registry, method source, runtime manifest, frozen Retrieval release, and legacy CLI defaults in fresh interpreters before/after Project 2 import and execution. Compare canonical bytes and nanosecond mtimes. Run a pre-Project-2 Project 1 deterministic run and resume it under the new code.

- [ ] **Step 4: Document exact implemented behavior**

Document artifacts, algorithms, formulas, CLI, input preparation, deterministic fallback, LLM configuration boundary, run layout, resume/failure rules, inspection commands, and the exact Project 3 handoff. State that Retrieval/Decision/joint evolution, full production `evolve`, Public scoring, DGM source evolution, and L1 protocol migration remain unavailable.

- [ ] **Step 5: Run the complete Project 2 suite**

```bash
pytest -q \
  tests/test_evolution_v2_numerical_contracts.py \
  tests/test_evolution_v2_numerical_config.py \
  tests/test_evolution_v2_numerical_descriptors.py \
  tests/test_evolution_v2_numerical_mutation.py \
  tests/test_evolution_v2_numerical_proposers.py \
  tests/test_evolution_v2_numerical_nsga2.py \
  tests/test_evolution_v2_numerical_map_elites.py \
  tests/test_evolution_v2_numerical_hyperband.py \
  tests/test_evolution_v2_numerical_adapters.py \
  tests/test_evolution_v2_numerical_persistence.py \
  tests/test_evolution_v2_numerical_runner.py \
  tests/test_evolution_v2_numerical_cli.py \
  tests/test_evolution_v2_numerical_safety.py \
  tests/test_evolution_v2_numerical_compatibility.py
```

Expected: all tests pass with no skip except an explicitly documented external real-model availability probe; deterministic and scripted-LLM paths must never skip.

- [ ] **Step 6: Run Project 1 and legacy regression**

Run: `pytest -q tests/test_evolution_v2_*.py`

Run: `pytest -q tests/test_evolution_core_contracts.py tests/test_evolution_core_persistence.py tests/test_package_artifacts.py tests/test_package_coordinate_evolution.py tests/test_run_package_coevolution.py tests/test_evolving_cli.py`

Expected: both commands pass; no existing legacy artifact changes appear in `git status`.

- [ ] **Step 7: Run a fresh end-to-end smoke and inspect artifacts**

Create a fresh directory with `mktemp -d`. Build committed fixture inputs, run `numerical-evolve`, resume it once, and independently verify exact file set, immutable hashes/mtimes, multiple occupied cells, accepted and rejected transitions, changed policy/prompt identities, executable frozen Supply/registry, complete budget accounting, and zero Public access.

- [ ] **Step 8: Check scope and placeholders**

Run `git diff --check`, inspect `git diff --stat` from the plan base, and scan Project 2 sources/docs for `TODO|FIXME|pass$|NotImplemented|placeholder`. Explain legitimate protocol uses; no unfinished implementation may remain.

- [ ] **Step 9: Commit Task 11**

```bash
git add tests/test_evolution_v2_numerical_safety.py tests/test_evolution_v2_numerical_compatibility.py docs/evolution-v2-numerical-qd.md README.md
git commit -m "docs(numerical-qd): document self-evolution"
```

## Project 2 Exit Gate

Project 3 starts only after the Task 11 fresh gate demonstrates real executable Numerical Supply self-evolution, deterministic recovery, hybrid fallback, multi-niche retention, exact Hyperband/NSGA-II behavior, atomic V2 Kernel promotion, no Dev/Public feedback leakage, and unchanged Project 1/legacy behavior.
