# Champion–Challenger Numerical Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one production Numerical Agent workflow that evolves Dictionary-backed single, Combined, routing, segmented, and bounded-overlay forecasting Champions without making Toto mandatory.

**Architecture:** Reuse the existing executable Dictionary, `ForecastStore`, screening, Combined, morphology, and scaled-metric components. Add strict Champion contracts, a history-only proposer boundary, host-owned numeric expansion, Build-64 successive halving, one-shot Calibration-16 and Dev-20 gates, immutable releases, and a separate write-free Public-99 regression command.

**Tech Stack:** Python 3.12+, frozen dataclasses, strict JSON, SHA-256 fingerprints, existing Codex/Claude/Qwen `LLMClient` interfaces, Dr-CiK capped sMAE/sRMSE, pytest, Git-tracked Python Dictionary artifacts.

**Spec:** `docs/superpowers/specs/2026-09-01-champion-challenger-numerical-evolution-design.md`

## Global Constraints

- Do not train or merge TSFM weights.
- Toto is an optional initial Champion/fallback, not a mandatory Child parent.
- Statistical, TSFM, and Combined candidates are all eligible.
- The LLM owns structural assumptions only; trusted Python owns thresholds, weights, correction caps, scoring, and acceptance.
- Build feedback is labeled `adaptive_train_build_diagnostic` and makes no independent generalization claim.
- Calibration-16 and Dev-20 are read-only, one-shot gates and never feed later proposals.
- Public-99 is a frozen regression set and no evolution API may load or consume its labels.
- Failed Children preserve the exact Parent object and bytes.
- Capped mean sMAE/sRMSE are paired authorities; raw tails, clipping, coverage, and regret are mandatory safety gates.
- All new production behavior follows strict RED–GREEN TDD.
- Preserve the existing untracked `runs/numerical_morphology/` artifacts.

## File Structure

### New production modules

- `numerical_agent/evolution/forecast_store.py`: shared cached Statistical/TSFM/Combined execution extracted from the selector runner.
- `numerical_agent/evolution/champion.py`: immutable assumption, recipe, fitted-policy, release, parsing, and fingerprint contracts.
- `numerical_agent/evolution/champion_runtime.py`: history-only execution of frozen Champion operators.
- `numerical_agent/evolution/champion_evidence.py`: trusted task rows, grouped Build evidence, canonical scoring, and sanitized proposal projection.
- `numerical_agent/evolution/champion_proposal.py`: strict LLM structural proposal and host-owned numeric expansion.
- `numerical_agent/evolution/champion_controller.py`: Build successive halving, Parent preservation, Calibration/Dev lifecycle, checkpointing, and release publication.
- `numerical_agent/run_champion_evolution.py`: formal evolution CLI.
- `numerical_agent/evaluate_frozen_champion.py`: isolated frozen Public regression CLI.
- `scripts/run_champion_evolution.sh`: environment-variable wrapper and dry-run surface.

### Existing modules modified

- `numerical_agent/run_selector_evolution.py`: import the extracted `ForecastStore` without behavior change.
- `numerical_agent/evolution/numerical_loop.py`: execute an optional frozen Champion after existing materialization and diagnostics.
- `numerical_agent/evolution/numerical_package.py`: bind Champion release/assumption fingerprints into the package.
- `numerical_agent/README.md`: document the formal lifecycle and commands.
- `README.md`: add one concise entry point and result-status warning.

### New tests

- `tests/test_evolution_forecast_store.py`
- `tests/test_evolution_champion.py`
- `tests/test_evolution_champion_runtime.py`
- `tests/test_evolution_champion_evidence.py`
- `tests/test_evolution_champion_proposal.py`
- `tests/test_evolution_champion_controller.py`
- `tests/test_champion_evolution_cli.py`
- `tests/test_frozen_champion_evaluation.py`
- `tests/test_champion_evolution_e2e.py`

---

### Task 1: Extract the shared ForecastStore

**Files:**
- Create: `numerical_agent/evolution/forecast_store.py`
- Modify: `numerical_agent/run_selector_evolution.py`
- Create: `tests/test_evolution_forecast_store.py`
- Verify: `tests/test_numerical_selector_script.py`

**Interfaces:**
- Produces: `ForecastStore(root, module_path, skills_path, portfolio, runtimes, *, screening_hash, runtime_identity, statistical_time_budget_s, statistical_failure_limit)`.
- Produces: `ForecastStore.forecast(name, history, horizon, frequency) -> tuple[float, ...]`.
- Produces: `ForecastStore.identity_hash: str`, `.hits`, `.misses`, and `.close()`.
- Preserves: cache schema 3, active scaled-metric policy binding, leaf materialization once, Combined leaves only, and TSFM runtime identity.

- [ ] **Step 1: Write extraction characterization tests**

```python
def test_shared_store_materializes_one_leaf_once(tmp_path, monkeypatch):
    calls = []
    store = make_fixture_store(tmp_path, monkeypatch, calls)
    first = store.forecast("naive_last", (1.0, 2.0), 2, "D")
    second = store.forecast("naive_last", (1.0, 2.0), 2, "D")
    assert first == second == (2.0, 2.0)
    assert calls == ["naive_last"]
    assert (store.misses, store.hits) == (1, 1)


def test_shared_store_combined_reuses_materialized_leaves(tmp_path, monkeypatch):
    calls = []
    store = make_fixture_store(tmp_path, monkeypatch, calls)
    store.forecast("combined_timesfm_seasonal", HISTORY, 4, "D")
    assert calls.count("timesfm_2_5") == 1
    assert calls.count("seasonal_naive") == 1
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `pytest -q tests/test_evolution_forecast_store.py`

Expected: FAIL with `ModuleNotFoundError: numerical_agent.evolution.forecast_store`.

- [ ] **Step 3: Move the existing store without semantic changes**

Move `ForecastStore`, `_IsolatedStatisticalRuntime`, `_StructuralForecastInvalid`, and their store-only helpers from `run_selector_evolution.py` into `evolution/forecast_store.py`. Preserve the public constructor and cache key bytes. Replace the runner definition with:

```python
from .evolution.forecast_store import ForecastStore
```

- [ ] **Step 4: Verify new and legacy behavior**

Run: `pytest -q tests/test_evolution_forecast_store.py tests/test_numerical_selector_script.py`

Expected: PASS with no cache-fixture changes.

- [ ] **Step 5: Commit**

```bash
git add numerical_agent/evolution/forecast_store.py numerical_agent/run_selector_evolution.py tests/test_evolution_forecast_store.py
git commit -m "refactor(numerical): share forecast store"
```

---

### Task 2: Define immutable Champion contracts

**Files:**
- Create: `numerical_agent/evolution/champion.py`
- Create: `tests/test_evolution_champion.py`

**Interfaces:**
- Produces: `EvolutionAssumption`, `ChampionRecipe`, `FittedChampionPolicy`, and `ChampionRelease` frozen dataclasses.
- Produces: `parse_champion_recipe(payload) -> ChampionRecipe`.
- Produces: `parse_champion_release(payload) -> ChampionRelease`.
- Produces: `champion_fingerprint(value) -> str` over canonical JSON.
- Consumes: candidate names and family identities supplied by the executable Dictionary.

- [ ] **Step 1: Write strict schema and identity tests**

```python
def test_recipe_supports_non_toto_combined_and_horizon_route():
    recipe = parse_champion_recipe({
        "name": "timesfm_seasonal_challenger",
        "kind": "horizon_route",
        "parents": ["timesfm_2_5", "seasonal_naive"],
        "fallback_parent": "timesfm_2_5",
        "assumptions": [assumption_payload(operator="horizon_route")],
    })
    assert recipe.parents == ("timesfm_2_5", "seasonal_naive")
    assert "toto_2_0" not in recipe.parents


@pytest.mark.parametrize("mutation", [
    lambda p: {**p, "unknown": 1},
    lambda p: {**p, "parents": ["same", "same"]},
    lambda p: {**p, "fallback_parent": "not_a_parent"},
])
def test_recipe_rejects_schema_or_namespace_drift(mutation):
    with pytest.raises(ChampionContractError):
        parse_champion_recipe(mutation(valid_recipe_payload()))


def test_failed_child_can_return_exact_parent_object():
    parent = fixture_release("parent")
    child = fixture_release("child")
    assert preserve_parent(parent, child, accepted=False) is parent
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `pytest -q tests/test_evolution_champion.py`

Expected: FAIL because `champion.py` does not exist.

- [ ] **Step 3: Implement the exact dataclasses and parsers**

```python
AssumptionOperator = Literal[
    "select", "route", "horizon_route", "weighted", "median", "bounded_overlay"
]


@dataclass(frozen=True)
class EvolutionAssumption:
    assumption_id: str
    candidate_name: str
    feature: str
    direction: Literal["above", "below"]
    horizon_region: Literal["early", "late", "full"]
    operator: AssumptionOperator
    rationale: str
    failure_condition: str


@dataclass(frozen=True)
class ChampionRecipe:
    name: str
    kind: AssumptionOperator
    parents: tuple[str, ...]
    fallback_parent: str
    assumptions: tuple[EvolutionAssumption, ...]


@dataclass(frozen=True)
class FittedChampionPolicy:
    recipe: ChampionRecipe
    thresholds: tuple[tuple[str, float], ...]
    weights: tuple[float, ...] = ()
    overlay_alpha: float = 0.0
    correction_cap: float = 0.0
    horizon_split: float = 0.5


def preserve_parent(parent, child, *, accepted: bool):
    return child if accepted else parent
```

Validate exact built-in types, finite numeric values, identifier names, unique parents and assumptions, operator-specific arity, normalized weights, fallback membership, and canonical inactive numeric fields.

- [ ] **Step 4: Add payload round-trip and fingerprint tests**

Assert reordered input keys produce the same fingerprint, while parent order, assumptions, fitted values, lineage, or metric-policy metadata changes it.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_evolution_champion.py`

```bash
git add numerical_agent/evolution/champion.py tests/test_evolution_champion.py
git commit -m "feat(numerical): add champion contracts"
```

---

### Task 3: Execute frozen Champion operators safely

**Files:**
- Create: `numerical_agent/evolution/champion_runtime.py`
- Create: `tests/test_evolution_champion_runtime.py`

**Interfaces:**
- Consumes: `FittedChampionPolicy`, materialized forecasts, `CandidateDiagnostics`, `TaskProfile`, history, and horizon.
- Produces: `ChampionExecution(forecast, selected_names, activated_assumptions, fallback_reason)`.
- Produces: `execute_champion(...) -> ChampionExecution`.
- Never consumes: future truth, task labels, Retrieval evidence, or an LLM.

- [ ] **Step 1: Write operator and fallback RED tests**

```python
def test_independent_combined_can_win_without_toto():
    policy = fitted_policy(
        kind="select",
        parents=("combined_timesfm_seasonal",),
        fallback="combined_timesfm_seasonal",
    )
    result = execute_champion(policy, forecasts={
        "combined_timesfm_seasonal": (3.0, 4.0),
        "toto_2_0": (9.0, 9.0),
    }, diagnostics=SAFE, profile=PROFILE, history=HISTORY, horizon=2)
    assert result.forecast == (3.0, 4.0)


def test_failed_assumption_returns_exact_fallback_forecast():
    fallback = (10.0, 11.0)
    result = execute_champion(
        fitted_overlay(), forecasts={"toto_2_0": fallback, "specialist": (99.0, 99.0)},
        diagnostics=UNSAFE, profile=PROFILE, history=HISTORY, horizon=2,
    )
    assert result.forecast is fallback
    assert result.fallback_reason == "assumption_not_satisfied"


def test_horizon_route_uses_each_parent_only_in_its_segment():
    result = execute_champion(
        fitted_horizon_route(split=0.5),
        forecasts={"early": (1.0, 2.0, 3.0, 4.0), "late": (5.0, 6.0, 7.0, 8.0)},
        diagnostics=SAFE, profile=PROFILE, history=HISTORY, horizon=4,
    )
    assert result.forecast == (1.0, 2.0, 7.0, 8.0)
```

- [ ] **Step 2: Run and observe RED**

Run: `pytest -q tests/test_evolution_champion_runtime.py`

Expected: FAIL because `execute_champion` is missing.

- [ ] **Step 3: Implement all six operators**

Implement `select`, two-parent `route`, `horizon_route`, normalized `weighted`, odd/even deterministic `median`, and clipped `bounded_overlay`. Evaluate assumptions from `TaskProfile` and local diagnostics only. Validate every parent forecast is a finite tuple of exactly `horizon` values before arithmetic.

For bounded overlay use:

```python
delta = specialist[index] - fallback[index]
limit = policy.correction_cap * robust_history_scale(history)
correction = max(-limit, min(limit, policy.overlay_alpha * delta))
forecast[index] = fallback[index] + correction
```

- [ ] **Step 4: Add adversarial numeric tests**

Cover malformed sequences, strings, NaN/Inf, overflow-scale history, invalid diagnostics, missing parents, zero horizon, hostile mappings, and exact fallback identity.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_evolution_champion_runtime.py tests/test_toto_anchor_policy.py`

```bash
git add numerical_agent/evolution/champion_runtime.py tests/test_evolution_champion_runtime.py
git commit -m "feat(numerical): execute champion policies"
```

---

### Task 4: Build trusted evidence and paired scores

**Files:**
- Create: `numerical_agent/evolution/champion_evidence.py`
- Create: `tests/test_evolution_champion_evidence.py`

**Interfaces:**
- Produces: `ChampionTaskRow`, `ChampionScore`, `ChampionComparison`, `MorphologyAggregate`, and `ProposerEvidence`.
- Produces: `score_policy(rows, policy) -> ChampionScore`.
- Produces: `compare_champion(parent, child, config) -> ChampionComparison`.
- Produces: `sanitize_build_evidence(rows, comparisons) -> ProposerEvidence`.
- Consumes: canonical `drcik_point_metrics`, `TaskProfile`, materialized forecasts, and trusted future arrays inside the scorer only.

- [ ] **Step 1: Write metric authority RED tests**

```python
def test_pair_gate_rejects_srmse_regression_hidden_by_joint_mean():
    parent = score(mean_smae=1.0, mean_srmse=1.0)
    child = score(mean_smae=0.5, mean_srmse=1.4)
    result = compare_champion(parent, child, gate_config())
    assert result.accepted is False
    assert "mean_srmse" in result.failures


@pytest.mark.parametrize("metric", [
    "p90_smae_raw", "p95_smae_raw", "p90_srmse_raw", "p95_srmse_raw",
])
def test_each_raw_tail_is_an_independent_gate(metric):
    result = compare_champion(parent_score(), child_with(metric, 99.0), gate_config())
    assert result.accepted is False
    assert metric in result.failures


def test_win_count_is_reported_but_not_a_standalone_gate():
    result = compare_champion(
        parent_score(), safe_pareto_child(wins=4, losses=6), gate_config()
    )
    assert result.accepted is True
    assert result.wtl == WinTieLoss(wins=4, ties=0, losses=6)
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_evolution_champion_evidence.py`

Expected: FAIL because the evidence module is missing.

- [ ] **Step 3: Implement canonical per-task and aggregate scoring**

Compute capped and raw sMAE/sRMSE for every complete pair. Aggregate means, median, P90, P95, maxima, clipping counts, coverage, failures, and paired W/T/L. Reject duplicate task IDs and candidate/task keys before constructing maps.

- [ ] **Step 4: Implement anonymous morphology evidence**

Aggregate only reviewed features and return values shaped as:

```python
MorphologyAggregate(
    group_id="periodicity_strength:high",
    support=12,
    candidate_name="combined_timesfm_seasonal",
    mean_delta_smae=-0.08,
    mean_delta_srmse=-0.05,
    coverage=1.0,
    p95_regret_smae=0.02,
    p95_regret_srmse=0.03,
)
```

The proposer projection must contain no task IDs, truth arrays, forecasts, entity names, Dev/Public markers, or raw runtime exceptions.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_evolution_champion_evidence.py tests/test_evolution_combined_morphology.py`

```bash
git add numerical_agent/evolution/champion_evidence.py tests/test_evolution_champion_evidence.py
git commit -m "feat(numerical): score champion evidence"
```

---

### Task 5: Add strict structural proposal and host expansion

**Files:**
- Create: `numerical_agent/evolution/champion_proposal.py`
- Create: `tests/test_evolution_champion_proposal.py`

**Interfaces:**
- Produces: `propose_champion_recipes(llm, parent, inventory, evidence, *, minimum=5, maximum=10) -> tuple[ChampionRecipe, ...]`.
- Produces: `expand_recipe(recipe, build_rows) -> tuple[FittedChampionPolicy, ...]`.
- Consumes: `LLMClient`, canonical Dictionary inventory, exact Parent recipe, and `ProposerEvidence` only.

- [ ] **Step 1: Write prompt and parser RED tests**

```python
def test_prompt_has_no_labels_tasks_or_numeric_authority(recording_llm):
    propose_champion_recipes(recording_llm, PARENT, INVENTORY, EVIDENCE)
    payload = json.loads(recording_llm.messages[0][0]["content"])
    serialized = json.dumps(payload).lower()
    assert "task_" not in serialized
    assert "future" not in serialized
    assert "public" not in serialized
    assert "dev" not in serialized
    assert "weights" not in payload["output_schema"]["assumption_fields"]


def test_parser_rejects_llm_numeric_thresholds_before_expansion():
    response = valid_response()
    response["recipes"][0]["threshold"] = 0.8
    with pytest.raises(ChampionProposalError):
        parse_champion_response(response, INVENTORY)


def test_proposer_allows_tsfm_statistical_recipe_without_toto():
    recipes = parse_champion_response(timesfm_seasonal_response(), INVENTORY)
    assert recipes[0].parents == ("timesfm_2_5", "seasonal_naive")
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_evolution_champion_proposal.py`

Expected: FAIL because the proposer boundary is missing.

- [ ] **Step 3: Implement strict one-call proposal parsing**

Use one system prompt with an exact JSON schema. Reject duplicate keys, unknown fields, invalid identifiers, unknown parents/features/operators, fewer than five or more than ten recipes, duplicate recipe names, code blocks, and numeric authority. One schema retry is allowed; a second malformed response rejects the generation.

- [ ] **Step 4: Implement host-owned bounded grids**

Derive thresholds from Build feature quantiles `{0.2, 0.4, 0.5, 0.6, 0.8}` after deduplication. Use fixed grids:

```python
WEIGHT_GRID = ((0.75, 0.25), (0.5, 0.5), (0.25, 0.75))
OVERLAY_ALPHA_GRID = (0.25, 0.5, 0.75)
CORRECTION_CAP_GRID = (0.1, 0.25, 0.5, 1.0)
HORIZON_SPLIT_GRID = (0.25, 0.5, 0.75)
```

Trim grids deterministically for operator arity and reject expansion if Build has no finite value for the assumption feature.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_evolution_champion_proposal.py tests/test_evolution_combined_evolution.py`

```bash
git add numerical_agent/evolution/champion_proposal.py tests/test_evolution_champion_proposal.py
git commit -m "feat(numerical): propose champion recipes"
```

---

### Task 6: Implement Build successive halving and Parent preservation

**Files:**
- Create: `numerical_agent/evolution/champion_controller.py`
- Create: `tests/test_evolution_champion_controller.py`

**Interfaces:**
- Produces: `ChampionEvolutionConfig`, `BuildGeneration`, `BuildEvolutionResult`.
- Produces: `run_build_evolution(parent, rows, proposer, config) -> BuildEvolutionResult`.
- Consumes: Tasks 2–5 contracts, proposal adapter, host expansion, scorer, and fixed Build rows.
- Formal config fixes `build_size=64`, `calibration_size=16`, and `screen_sizes=(8, 32, 64)`; deterministic smoke config may use `build_size=8`, `calibration_size=2`, and `screen_sizes=(4, 8)` without changing production defaults.

- [ ] **Step 1: Write halving and identity RED tests**

```python
def test_successive_halving_runs_fixed_8_32_64_schedule():
    result = run_build_evolution(PARENT, ROWS_64, deterministic_proposer(), CONFIG)
    assert result.generations[0].stage_counts == (8, 32, 64)
    assert result.generations[0].full_build_children <= 3


def test_rejected_child_feedback_does_not_change_the_mutation_parent():
    parent = fixture_release("parent")
    result = run_build_evolution(parent, ROWS_64, all_rejected_proposer(), CONFIG)
    assert all(g.mutation_parent_sha256 == parent.sha256 for g in result.generations)
    assert result.active_parent is parent


def test_build_feedback_is_labeled_non_independent():
    result = run_build_evolution(PARENT, ROWS_64, deterministic_proposer(), CONFIG)
    assert result.score_label == "adaptive_train_build_diagnostic"
    assert result.independent_generalization_claim is False
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_evolution_champion_controller.py -k 'halving or mutation_parent or non_independent'`

Expected: FAIL because `run_build_evolution` is missing.

- [ ] **Step 3: Implement immutable Build evolution**

For each generation:

1. propose five through ten recipes from the exact active Parent;
2. expand numerics on Build only;
3. run fixed screen membership recorded in config;
4. remove invalid or unsafe candidates at 8 and 32 tasks;
5. score at most three on all 64;
6. retain at most three different kinds or parent sets in the shortlist;
7. create sanitized feedback from every attempted Child;
8. keep the exact Parent until Calibration.

- [ ] **Step 4: Add threshold and lineage tests**

Assert candidate minimum gain defaults to `0.005`, research target to `0.05`, both are fingerprinted, W/T/L is not a standalone gate, and the archive prefers distinct recipe kinds before a second recipe of the same kind.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_evolution_champion_controller.py`

```bash
git add numerical_agent/evolution/champion_controller.py tests/test_evolution_champion_controller.py
git commit -m "feat(numerical): evolve build challengers"
```

---

### Task 7: Enforce 64/16/20 lifecycle and durable release authority

**Files:**
- Modify: `numerical_agent/evolution/champion_controller.py`
- Modify: `tests/test_evolution_champion_controller.py`

**Interfaces:**
- Produces: `partition_train_tasks(tasks, *, build_size=64, calibration_size=16, seed) -> TrainPartitions`.
- Produces: `ChampionRunManifest`, `ChampionCheckpoint`, `ChampionEvolutionOutcome`.
- Produces: `ChampionEvolutionController.evolve(parent, train_tasks, dev_tasks) -> ChampionEvolutionOutcome`.
- Produces: `ChampionArtifactStore` with canonical atomic JSON writes.
- Consumes: `common.data.Task` so the controller can enforce entity-disjoint internal partitions before projecting label-bearing rows into the trusted scorer.

- [ ] **Step 1: Write entity-disjoint partition RED tests**

```python
def test_internal_partition_is_exact_and_entity_disjoint():
    parts = partition_train_tasks(TRAIN_80, build_size=64, calibration_size=16, seed=20260901)
    assert len(parts.build) == 64
    assert len(parts.calibration) == 16
    assert {t.entity_name for t in parts.build}.isdisjoint(
        {t.entity_name for t in parts.calibration}
    )


def test_partition_fails_when_entity_groups_cannot_make_exact_sizes():
    with pytest.raises(ChampionLifecycleError, match="entity-disjoint"):
        partition_train_tasks(UNSPLITTABLE_TASKS, build_size=3, calibration_size=1, seed=1)
```

- [ ] **Step 2: Run partition tests and verify RED**

Run: `pytest -q tests/test_evolution_champion_controller.py -k partition`

Expected: FAIL because formal partitioning is missing.

- [ ] **Step 3: Implement deterministic group subset selection**

Group by exact `entity_name`, sort groups by SHA-256 of `seed + entity_name`, and use deterministic subset-sum dynamic programming to choose exactly 16 Calibration tasks. Build receives the remaining 64. Reject duplicate task IDs, unknown entities, wrong counts, or overlap.

- [ ] **Step 4: Write one-shot Calibration and Dev RED tests**

```python
def test_calibration_is_never_returned_to_the_proposer():
    proposer = RecordingProposer()
    controller(proposer).evolve(PARENT, TRAIN_80, DEV_20)
    assert not any(task_id in proposer.serialized_inputs for task_id in CALIBRATION_IDS)
    assert "calibration" not in proposer.serialized_inputs.lower()


def test_calibration_rejection_keeps_parent_and_never_reads_dev():
    dev = ExplodingSequence("Dev must remain unopened")
    outcome = controller(calibration_accept=False).evolve(PARENT, TRAIN_80, dev)
    assert outcome.release is PARENT
    assert outcome.dev_report is None


def test_dev_rejection_preserves_release_bytes(tmp_path):
    old_bytes = canonical_release_bytes(PARENT)
    outcome = controller(dev_accept=False, root=tmp_path).evolve(PARENT, TRAIN_80, DEV_20)
    assert canonical_release_bytes(outcome.release) == old_bytes
```

- [ ] **Step 5: Implement lifecycle and checkpoint binding**

Manifest and checkpoint fingerprints must include source files, exact task IDs and entities, split manifest, Build/Calibration membership, Dictionary Python files, forecast-store identity, metric policy, proposal model/config, schedule, numeric grids, candidate threshold, and research target. Resume rejects every mismatch before task or model execution.

Persist only:

```python
ChampionCheckpoint(
    schema_version=1,
    input_fingerprint=manifest.input_fingerprint,
    completed_stage="build_generation_2",
    active_parent=parent,
    proposal_archive=archive,
    sanitized_feedback=feedback,
)
```

Calibration and Dev reports are immutable one-shot artifacts. No stored `passed` bit may open a later split; recompute the active candidate comparison from bound evidence immediately before each boundary.

- [ ] **Step 6: Run and commit**

Run: `pytest -q tests/test_evolution_champion_controller.py`

```bash
git add numerical_agent/evolution/champion_controller.py tests/test_evolution_champion_controller.py
git commit -m "feat(numerical): gate champion lifecycle"
```

---

### Task 8: Integrate the frozen Champion into Numerical runtime

**Files:**
- Modify: `numerical_agent/evolution/numerical_loop.py`
- Modify: `numerical_agent/evolution/numerical_package.py`
- Modify: `tests/test_numerical_morphology_loop.py`
- Create: `tests/test_numerical_champion_loop.py`

**Interfaces:**
- Extends: `run_numerical_loop(..., champion_release: ChampionRelease | None = None)`.
- Extends: `NumericalForecastPackage` component fingerprints with `champion_release`, `champion_recipe`, and `champion_assumptions` SHA-256 values.
- Preserves: existing behavior byte-for-byte when `champion_release is None`.

- [ ] **Step 1: Write compatibility and runtime RED tests**

```python
def test_no_champion_keeps_legacy_package_byte_identical():
    legacy = run_numerical_loop(TASK, **LEGACY_ARGS)
    explicit_none = run_numerical_loop(TASK, champion_release=None, **LEGACY_ARGS)
    assert legacy == explicit_none


def test_champion_uses_only_already_materialized_candidates():
    package = run_numerical_loop(TASK, champion_release=RELEASE, **ARGS)
    assert package.selected.name in {row.name for row in package.ranked_forecasts}
    assert RUNNER_CALLS == EXPECTED_LEAF_CALLS_ONCE


def test_champion_release_fingerprint_changes_package_identity():
    a = run_numerical_loop(TASK, champion_release=RELEASE_A, **ARGS)
    b = run_numerical_loop(TASK, champion_release=RELEASE_B, **ARGS)
    assert a.component_fingerprints["champion_release"] != b.component_fingerprints["champion_release"]
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_numerical_champion_loop.py`

Expected: FAIL because `run_numerical_loop` has no Champion parameter.

- [ ] **Step 3: Integrate after materialization and diagnostics**

Call `execute_champion` only after screening, leaves, Combined outcomes, and `CandidateDiagnostics` are frozen. Convert its selected materialized forecast into the existing `SelectionDecision` and ranked package shape. If execution rejects any condition, use the release fallback forecast and record a typed fallback reason.

- [ ] **Step 4: Add Retrieval/Decision boundary tests**

Assert the package exposes assumption IDs and hashes but no Train truth or thresholds not needed at inference. Existing two-stage Retrieval may select only one materialized ranked forecast and cannot create a new forecast vector.

- [ ] **Step 5: Run and commit**

Run: `pytest -q tests/test_numerical_champion_loop.py tests/test_numerical_morphology_loop.py tests/test_numerical_retrieval_handoff.py`

```bash
git add numerical_agent/evolution/numerical_loop.py numerical_agent/evolution/numerical_package.py tests/test_numerical_champion_loop.py tests/test_numerical_morphology_loop.py
git commit -m "feat(numerical): run frozen champion"
```

---

### Task 9: Add the formal evolution CLI and Public firewall

**Files:**
- Create: `numerical_agent/run_champion_evolution.py`
- Create: `numerical_agent/evaluate_frozen_champion.py`
- Create: `scripts/run_champion_evolution.sh`
- Create: `tests/test_champion_evolution_cli.py`
- Create: `tests/test_frozen_champion_evaluation.py`

**Interfaces:**
- Produces CLI: `python -m numerical_agent.run_champion_evolution`.
- Produces CLI: `python -m numerical_agent.evaluate_frozen_champion`.
- Evolution CLI loads only Train and optional Dev IDs with `common.data.load_tasks_by_id`.
- Frozen CLI loads only Public IDs and has no LLM, mutation, checkpoint, or release-write interface.

- [ ] **Step 1: Write CLI parser and selective-loader RED tests**

```python
def test_evolution_cli_has_no_public_option():
    parser = build_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--public" not in option_strings
    assert "--public-output" not in option_strings


def test_evolution_loader_never_decodes_public_rows(tmp_path):
    tasks = write_jsonl_with_valid_train_dev_and_invalid_public(tmp_path)
    result = main(evolution_args(tasks))
    assert result == 0


def test_frozen_public_cli_has_no_proposer_or_release_write_option():
    options = parser_options(frozen_build_parser())
    assert not {"--proposer-model", "--generations", "--checkpoint", "--release-output"} & options
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_champion_evolution_cli.py tests/test_frozen_champion_evaluation.py`

Expected: FAIL because both CLI modules are missing.

- [ ] **Step 3: Implement the evolution runner**

Parse the exact flags from the spec. Load `methods.py`, `policies.py`, `dictionary.py`, and optional `skills.py`; validate a clean Git source; construct `ForecastStore`; build the manifest before LLM or runtime calls; run the controller; close every runtime in `finally`; write canonical reports; print a concise Train, Calibration, and Dev summary.

The wrapper must forward:

```bash
--repo "$CHAMPION_REPO"
--split-file "$CHAMPION_SPLIT_FILE"
--tasks-file "$CHAMPION_TASKS_FILE"
--generations "$CHAMPION_GENERATIONS"
--proposer-model "$CHAMPION_MODEL"
--proposer-reasoning-effort "$CHAMPION_REASONING"
--candidate-minimum-gain "$CHAMPION_MIN_GAIN"
--research-target-gain "$CHAMPION_TARGET_GAIN"
--output-dir "$CHAMPION_OUTPUT_DIR"
```

- [ ] **Step 4: Implement isolated frozen Public regression**

Require a validated immutable `champion_release.json`, split-manifest fingerprint, and forecast-runtime identity. Materialize forecasts without an LLM, score Public once, and write only `public_regression_report.json` plus per-task frozen forecasts. Refuse to overwrite a completed report.

- [ ] **Step 5: Add shell and resume tests**

Run `bash -n scripts/run_champion_evolution.sh`, wrapper `--dry-run`, completed-run resume, wrong-source fingerprint, wrong split, wrong thresholds, malformed JSON, duplicate task IDs, runtime setup cleanup, and existing Public report refusal.

- [ ] **Step 6: Run and commit**

Run: `pytest -q tests/test_champion_evolution_cli.py tests/test_frozen_champion_evaluation.py`

```bash
git add numerical_agent/run_champion_evolution.py numerical_agent/evaluate_frozen_champion.py scripts/run_champion_evolution.sh tests/test_champion_evolution_cli.py tests/test_frozen_champion_evaluation.py
git commit -m "feat(numerical): add champion runner"
```

---

### Task 10: Add deterministic end-to-end smoke and documentation

**Files:**
- Create: `tests/test_champion_evolution_e2e.py`
- Modify: `numerical_agent/README.md`
- Modify: `README.md`

**Interfaces:**
- Validates the complete public production path with deterministic fake forecasts and fake LLM responses.
- Documents implementation status separately from real experimental results.

- [ ] **Step 1: Write the 8/2 end-to-end RED test**

```python
def test_fake_e2e_evolves_non_toto_champion_and_freezes_it(tmp_path):
    outcome = run_fake_champion_evolution(
        build_tasks=fixture_tasks(8),
        calibration_tasks=fixture_tasks(2),
        dev_tasks=fixture_tasks(2),
        proposal=timesfm_seasonal_recipe_without_toto(),
        screen_sizes=(4, 8),
        output_dir=tmp_path,
    )
    assert outcome.release.policy.recipe.parents == ("timesfm_2_5", "seasonal_naive")
    assert outcome.release.policy.recipe.fallback_parent == "timesfm_2_5"
    assert outcome.calibration_report.accepted is True
    assert outcome.dev_report.accepted is True
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_champion_evolution_e2e.py`

Expected: FAIL until the runner exposes its deterministic assembly function.

- [ ] **Step 3: Add rejection, resume, and Public-isolation E2E cases**

Cover malformed proposal, every Child rejected with exact Parent identity, transient proposal retry, crash after Build checkpoint, resume without duplicate Calibration or Dev, Dev rejection preserving release bytes, and invalid Public task bodies never decoded during evolution.

- [ ] **Step 4: Document exact commands and honest status**

Add sections explaining:

- Dictionary evolution remains upstream;
- Champion–Challenger evolution does not require Toto;
- assumptions are structural and host-fitted;
- 64/16 is internal to the 80 Train partition;
- 20 Dev and 99 Public are read-only;
- fake smoke proves wiring only;
- no new real 80/20 or Public-99 Champion result exists until separately run.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
pytest -q tests/test_evolution_forecast_store.py tests/test_evolution_champion.py tests/test_evolution_champion_runtime.py tests/test_evolution_champion_evidence.py tests/test_evolution_champion_proposal.py tests/test_evolution_champion_controller.py tests/test_numerical_champion_loop.py tests/test_champion_evolution_cli.py tests/test_frozen_champion_evaluation.py tests/test_champion_evolution_e2e.py
python -m compileall -q common numerical_agent
bash -n scripts/run_champion_evolution.sh
git diff --check
pytest -q
```

Expected: all focused and full tests pass; one existing optional-dependency skip may remain.

- [ ] **Step 6: Request independent review**

Ask the reviewer to verify the design acceptance criteria, exact Parent retention, paired metric authority, entity-disjoint 64/16 split, Calibration/Dev one-shot behavior, no mandatory Toto dependency, Public firewall, checkpoint binding, and frozen runtime reproducibility. Fix every confirmed Critical or Important finding with a new RED–GREEN cycle.

- [ ] **Step 7: Commit**

```bash
git add README.md numerical_agent/README.md tests/test_champion_evolution_e2e.py
git commit -m "docs(numerical): explain champion evolution"
```

## Plan Self-Review

- Spec Sections 1–4 map to Tasks 1–3.
- Candidate supply and typed assumptions map to Tasks 2 and 5.
- Build materialization and anonymous evidence map to Tasks 1 and 4.
- Successive halving and Parent preservation map to Task 6.
- 64/16/20/99 discipline, thresholds, checkpoints, and releases map to Task 7.
- Runtime inference and Retrieval/Decision handoff map to Task 8.
- CLI, frozen Public evaluation, failure behavior, and durable artifacts map to Task 9.
- Fake smoke, documentation, full verification, and independent review map to Task 10.
- All new named types and functions are introduced before their first downstream use.
- No task changes TSFM weights or treats Public-99 as unseen evaluation.
