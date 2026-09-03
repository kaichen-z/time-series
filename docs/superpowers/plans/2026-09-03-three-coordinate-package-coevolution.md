# Three-Coordinate Package Co-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and run one reproducible two-cycle Numerical → Retrieval → Decision package co-evolution lifecycle on Build-64, Calibration-16, and Dev-20, then evaluate the sealed final bundle once on Public-99.

**Architecture:** Add one immutable Numerical supply release and a runtime state that binds it to the existing frozen package registry and `HarnessPolicy`. A shared full-pipeline evaluator and successive-halving phase runner apply identical final-forecast gates to all three coordinate types, while target-specific proposers may change only their owned state. A two-cycle controller, append-only artifact store, and separate Public evaluator preserve direct lineage, cache identity, resume safety, and Public isolation.

**Tech Stack:** Python 3.11+, frozen dataclasses, SHA-256 canonical JSON identities, pytest, existing Dr-CiK `drcik_point_metrics`, `GroupFoldManifest`, `ForecastStore`, `ChampionProposerAdapter`, `NumericalForecastPackage`, `TwoStageRetrievalAgent`, `DecisionAgent`, `RetrievalGenome`, and `HarnessPolicy`.

**Spec:** `docs/superpowers/specs/2026-09-03-three-coordinate-package-coevolution-design.md`

## Global Constraints

- The registered data authority is exactly 80 Train / 20 Dev / 99 Public tasks.
- The 80 Train tasks are partitioned exactly once into 64 Build and 16 Calibration tasks using the registered group-aware authority.
- The formal coordinate order is exactly `N1 -> R1 -> D1 -> N2 -> R2 -> D2`, with early stopping only after a complete cycle accepts no coordinate.
- Each coordinate generation proposes exactly three Children and uses the exact `8 -> 32 -> 64 -> 16 -> 20` schedule, promoting at most `2 -> 1 -> 1 -> 1 -> 1` Children.
- Only one unchanged finalist per coordinate generation may access Calibration and Dev.
- Mean capped sMAE and sRMSE may not regress by more than `1e-12`; full Build, Calibration, and Dev gates require at least `0.5%` relative joint improvement.
- Build-64 requires at least four of five folds to be non-regressing, at least three folds to improve strictly, and maximum single-task joint regret no greater than `0.25`.
- Calibration and Dev reject any new single-task joint regret greater than `0.25`.
- P90 and P95 capped sMAE and sRMSE, invalid count, catastrophic count, clipped count, final-fallback count, and 100% task coverage are hard gates.
- The final accepted state must remain Pareto-non-regressing against the initial Toto bundle.
- Numerical supplies exactly one safe anchor plus no more than four materialized alternatives, with deterministic de-duplication and at most one retained member from each of Statistical, TSFM, Combined, and Atlas/overlay families when available.
- The fixed 70% Toto / 30% Atlas result is an optional materialized alternative, never the initial anchor and never an unconditional route.
- Retrieval uses the verified two-stage protocol, `strict_gain_target="final"`, and a read-only Retrieval Skill library; Skill creation and promotion are forbidden in this run.
- Retrieval Round 2 may run only for a typed high-priority Decision gap; a malformed or failed Round 2 preserves verified Round 1, while a fatal Round-1 failure preserves the Numerical safe selection.
- Decision selects exactly one already materialized Numerical alternative; it cannot create, edit, extrapolate, or blend forecast vectors.
- Agents never receive future values, ground-truth evidence labels, scorer output, task identities in feedback, or task-level Dev/Public residuals.
- Public-99 is inaccessible to proposal, fitting, selection, and coordinate acceptance; only the separate frozen evaluator may load it.
- The formal run uses fixed seed `20260903` and must never call legacy `scripts/run_co_evolution.py`.
- Public-99 is final regression evidence for this run, not a claim about a previously unseen benchmark, and its result must never feed another Child.

## File map

- `numerical_agent/evolution/specialist_atlas.py`: import the reviewed `85c561d` feature/pool base, then add the cross-fitted release and history-only routing required by package evolution; it never owns co-evolution acceptance.
- `evolving_loop/package_numerical_supply.py`: new immutable Numerical supply release, bounded package projection, release parsing, and registry construction contract.
- `evolving_loop/package_registry.py`: extend the frozen registry with exact release identity, complete task coverage validation, and deterministic manifest reconstruction.
- `evolving_loop/package_metrics.py`: new shared per-task final-pipeline score, aggregate evaluation, rank key, fold checks, Toto drift checks, and screen/full gates.
- `evolving_loop/package_pipeline_evaluator.py`: new state-aware Numerical→Retrieval→Decision evaluator that selects the registry bound to each Child and emits the shared metric rows.
- `evolving_loop/package_numerical_evolution.py`: new Numerical structural proposer, bounded host parameter fitter, package materializer, and Numerical-only state mutation.
- `evolving_loop/package_candidate_proposal.py`: new Retrieval and Decision proposal adapters that reuse existing typed parsers/mutators without running their old Train/Dev schedulers.
- `evolving_loop/package_stage_runner.py`: new target-agnostic 8/32/64/16/20 successive-halving runner with one-finalist Calibration/Dev access.
- `evolving_loop/package_coordinate_evolution.py`: extend bundle identity to the Numerical release, add bound runtime state, enforce one-principal-coordinate transitions, repeat two N-R-D cycles, and stop after a no-acceptance cycle.
- `evolving_loop/package_artifacts.py`: new immutable cache keys, append-only candidate evidence, accepted bundles, checkpoint, consumed-stage ledger, and resume validation.
- `evolving_loop/run_package_coevolution.py`: new real 80/20 package-native CLI and dependency wiring.
- `scripts/run_package_coevolution.sh`: new thin shell entry point that invokes only the package-native module.
- `evolving_loop/evaluate_frozen_package_bundle.py`: new isolated Public-99 evaluator and four-system attribution report.
- `tests/test_package_numerical_supply.py`: Numerical release, five-forecast bound, diversity, de-duplication, and registry replacement tests.
- `tests/test_package_metrics.py`: exact screen, Build, Calibration, Dev, fold, tail, regret, coverage, failure, and Toto gates.
- `tests/test_package_numerical_evolution.py`: safe Numerical proposal, fitting, materialization, and ownership tests.
- `tests/test_package_candidate_proposal.py`: typed Retrieval/Decision Child proposal and ownership tests.
- `tests/test_package_stage_runner.py`: stage counts, promotion, one-finalist holdout access, replay, and failure tests.
- `tests/test_package_coordinate_evolution.py`: bundle schema, direct lineage, six-step order, rejection byte preservation, and early-stop tests.
- `tests/test_package_artifacts.py`: immutable writes, cache mismatch, checkpoint replay, and consumed-stage tests.
- `tests/test_run_package_coevolution.py`: CLI authority, label firewall, dependency injection, resume, and zero-Public-access tests.
- `tests/test_evaluate_frozen_package_bundle.py`: sealed-bundle validation, Public-only membership, one-shot behavior, and report decomposition tests.
- `tests/test_package_coordinate_e2e.py`: deterministic fake-LLM two-cycle 80/20 end-to-end test.
- `README.md`: formal run, resume, and Public evaluation commands plus interpretation warning.

## Execution setup

Use `superpowers:using-git-worktrees` before Task 1. Do not execute this plan in the dirty main checkout and do not copy uncommitted files from `.worktrees/specialist-atlas-routing`.

```bash
cd /Users/yyoraa/time-series
git worktree add .worktrees/three-coordinate-package-coevolution \
  -b feature/three-coordinate-package-coevolution 934f7aa
cd .worktrees/three-coordinate-package-coevolution
git cherry-pick 85c561d
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_package_retrieval_evolution.py \
  tests/test_package_decision_evolution.py \
  tests/test_package_coordinate_evolution.py \
  tests/test_package_coordinate_e2e.py \
  tests/test_specialist_atlas.py \
  tests/test_evolution_forecast_store.py
```

Expected: the cherry-pick adds only `specialist_atlas.py`, its focused tests, and the cache-only `ForecastStore` support from `85c561d`; all listed baseline tests pass before new implementation begins.

---

### Task 1: Define the immutable Numerical supply and bounded package projection

**Files:**
- Create: `evolving_loop/package_numerical_supply.py`
- Modify: `evolving_loop/package_registry.py`
- Create: `tests/test_package_numerical_supply.py`

**Interfaces:**
- Consumes: `ChampionRelease`, `NumericalForecastPackage`, `RankedNumericalForecast`, `SelectionDecision`, and `FrozenNumericalPackageRegistry`.
- Produces: `NumericalAlternativeSpec(candidate_id, family, materializer_kind, recipe_payload, full_build_policy_payload, build_fold_policy_payloads, assumption_ids, failure_conditions)`.
- Produces: `NumericalSupplyRelease(schema_version, version, parent_sha256, anchor_release_payload, alternatives, atlas_release_sha256, source_fingerprints, runtime_fingerprints)`.
- Produces: `parse_numerical_supply_release(payload: object) -> NumericalSupplyRelease`.
- Produces: `bound_numerical_package(source: NumericalForecastPackage, release: NumericalSupplyRelease, materialized: Mapping[str, RankedNumericalForecast]) -> NumericalForecastPackage`.
- Produces: `build_package_registry(tasks: Sequence[ContextTask], release: NumericalSupplyRelease, package_builder: Callable[[ContextTask, NumericalSupplyRelease], NumericalForecastPackage]) -> FrozenNumericalPackageRegistry`.
- Extends: `FrozenNumericalPackageRegistry.release_sha256: str` and rejects incomplete or extra task coverage.

- [ ] **Step 1: Write failing release validation tests**

```python
def test_supply_release_binds_anchor_parent_and_four_diverse_alternatives():
    release = _supply_release(
        alternatives=(
            _alternative("seasonal_naive", "statistical"),
            _alternative("toto_2_0", "tsfm"),
            _alternative("weighted_pair", "combined"),
            _alternative("atlas_70_30", "atlas_overlay"),
        )
    )
    assert release.version == "n001"
    assert tuple(item.family for item in release.alternatives) == (
        "statistical",
        "tsfm",
        "combined",
        "atlas_overlay",
    )
    assert len(release.alternatives) == 4
    assert len(release.fingerprint) == 64


def test_supply_release_rejects_five_additional_alternatives():
    with pytest.raises(NumericalSupplyError, match="four additional"):
        _supply_release(
            alternatives=tuple(
                _alternative(f"candidate_{index}", "statistical")
                for index in range(5)
            )
        )
```

- [ ] **Step 2: Run the release tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_numerical_supply.py -k 'supply_release'`

Expected: collection fails because `evolving_loop.package_numerical_supply` does not exist.

- [ ] **Step 3: Implement the closed release types and canonical parser**

```python
NumericalSupplyFamily = Literal[
    "statistical", "tsfm", "combined", "atlas_overlay"
]
NumericalMaterializerKind = Literal[
    "dictionary", "champion", "atlas", "bounded_overlay"
]


@dataclass(frozen=True)
class NumericalAlternativeSpec:
    candidate_id: str
    family: NumericalSupplyFamily
    materializer_kind: NumericalMaterializerKind
    recipe_payload: Mapping[str, object]
    full_build_policy_payload: Mapping[str, object]
    build_fold_policy_payloads: tuple[tuple[int, Mapping[str, object]], ...]
    assumption_ids: tuple[str, ...]
    failure_conditions: tuple[str, ...]


@dataclass(frozen=True)
class NumericalSupplyRelease:
    schema_version: int
    version: str
    parent_sha256: str | None
    anchor_release_payload: Mapping[str, object]
    alternatives: tuple[NumericalAlternativeSpec, ...]
    atlas_release_sha256: str | None
    source_fingerprints: Mapping[str, str]
    runtime_fingerprints: Mapping[str, str]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()
```

Validation must require schema `1`, version format `nNNN`, canonical direct-parent SHA-256 for non-seed releases, a parseable `ChampionRelease` anchor, unique candidate IDs, no more than four alternatives, one alternative per family, aligned non-empty assumption/failure-condition tuples, exact known materializer kinds, one full-Build fitted policy, exactly one fitted policy for each Build fold `0` through `4`, and canonical source/runtime fingerprint maps. `parse_numerical_supply_release` accepts only the exact `to_payload()` keys and reconstructs frozen mappings. The `n000` seed may repeat its exact anchor policy in all five fold slots; non-seed alternatives must bind their own cross-fitted policies.

- [ ] **Step 4: Write failing bounded-package tests**

```python
def test_bounded_package_keeps_anchor_plus_one_per_family_and_deduplicates_vectors():
    source = _wide_package()
    materialized = {
        "safe_anchor": _ranked("safe_anchor", "tsfm", (1.0, 1.0)),
        "seasonal_naive": _ranked("seasonal_naive", "statistical", (2.0, 2.0)),
        "toto_2_0": _ranked("toto_2_0", "tsfm", (1.0, 1.0)),
        "weighted_pair": _ranked("weighted_pair", "combined", (3.0, 3.0)),
        "atlas_70_30": _ranked("atlas_70_30", "atlas_overlay", (4.0, 4.0)),
    }
    package = bound_numerical_package(source, _supply_release(), materialized)
    assert tuple(item.name for item in package.ranked_alternatives) == (
        "safe_anchor",
        "seasonal_naive",
        "weighted_pair",
        "atlas_70_30",
    )
    assert package.selection_decision.selected == ("safe_anchor",)
    assert package.protected_baseline.name == "safe_anchor"
    assert len(package.ranked_alternatives) <= 5


def test_fixed_atlas_blend_is_never_promoted_to_anchor():
    package = bound_numerical_package(
        _wide_package(),
        _supply_release(),
        _materialized_forecasts(),
    )
    assert package.protected_baseline.name != "atlas_70_30"
```

- [ ] **Step 5: Implement deterministic package projection**

`bound_numerical_package` must locate the source package's exact protected anchor, require it in `materialized`, then visit release alternatives in canonical family order `statistical`, `tsfm`, `combined`, `atlas_overlay`. It drops invalid forecasts, duplicate candidate IDs, and duplicate forecast-vector SHA-256 values. It constructs a single-selection host default with these exact fields:

```python
selection = SelectionDecision(
    mode="single",
    selected=(anchor.name,),
    weights=(1.0,),
    forecast=anchor.forecast,
    confidence=0.0,
    reason_codes=("package_safe_anchor",),
    rejected={},
    baseline_name=anchor.name,
    considered_candidates=tuple(item.name for item in retained),
)
```

Return an exact `NumericalForecastPackage`, not a private Champion subclass. Preserve the source morphology card, accepted/rejected assumptions, and retrieval handoff. Restrict `active_candidate_names` and `candidate_diagnostics` to retained candidates. Add `numerical_supply_release`, `numerical_supply_parent`, and `numerical_supply_runtime` SHA-256 entries to `component_fingerprints` without removing the existing Champion and task-input provenance.

- [ ] **Step 6: Write failing registry release and coverage tests**

```python
def test_registry_binds_exact_supply_release_and_task_universe():
    registry = build_package_registry(tasks, release, package_builder)
    assert registry.release_sha256 == release.fingerprint
    assert registry.task_ids == tuple(sorted(task.numeric.task_id for task in tasks))
    assert registry.manifest["release_sha256"] == release.fingerprint


def test_registry_rejects_missing_or_extra_task_packages():
    with pytest.raises(PackageRegistryError, match="complete task coverage"):
        FrozenNumericalPackageRegistry(
            entries=((tasks[0], packages[0]),),
            release_sha256=release.fingerprint,
            expected_task_ids=(tasks[0].numeric.task_id, tasks[1].numeric.task_id),
        )
```

- [ ] **Step 7: Extend the registry and run focused tests**

Add keyword-only constructor arguments `release_sha256: str` and `expected_task_ids: Sequence[str]`. Include the release digest and sorted task universe in the registry manifest. Require every package's `component_fingerprints["numerical_supply_release"]` to match it.

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_numerical_supply.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py`

Expected: all tests pass; existing registry fixtures are updated with a deterministic seed supply release rather than bypassing the new authority.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/package_numerical_supply.py evolving_loop/package_registry.py tests/test_package_numerical_supply.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py
git commit -m "feat(numerical): bind package supply release"
```

---

### Task 2: Add one shared final-pipeline metric and gate authority

**Files:**
- Create: `evolving_loop/package_metrics.py`
- Create: `evolving_loop/package_pipeline_evaluator.py`
- Modify: `evolving_loop/package_retrieval_evolution.py`
- Modify: `evolving_loop/package_decision_evolution.py`
- Create: `tests/test_package_metrics.py`
- Modify: `tests/test_package_retrieval_evolution.py`
- Modify: `tests/test_package_decision_evolution.py`

**Interfaces:**
- Produces: `PackageTaskScore` with capped/raw sMAE/sRMSE, clipping, invalid, catastrophic, fallback, selected candidate, and transitive artifact fingerprints.
- Produces: `PackageEvaluation.task_rows` plus read-only `secondary_diagnostics` for Retrieval quality, Numerical oracle gap, and Decision selection regret.
- Produces: `PackageEvaluation.from_rows(candidate_sha256: str, rows: Sequence[PackageTaskScore], expected_task_ids: Sequence[str], secondary_diagnostics: Mapping[str, float] | None = None) -> PackageEvaluation`.
- Produces: `PackageEvaluation.result_bytes() -> bytes`, excluding candidate publication metadata while retaining every forecast metric, task row, failure count, and inference artifact hash.
- Produces: `PackageGateConfig(tolerance=1e-12, minimum_relative_joint_gain=0.005, maximum_task_joint_regret=0.25)`.
- Produces: `package_screen_failures(child, parent, config) -> tuple[str, ...]`.
- Produces: `package_full_gate_failures(child, parent, config, *, stage, fold_manifest=None, initial=None) -> tuple[str, ...]`.
- Produces: `package_rank_key(evaluation) -> tuple[float, ...]`.
- Produces: `PackageRetrievalEvaluator.evaluate_package(genome, tasks, *, stage, skill_library) -> PackageEvaluation`.
- Produces: `PackageDecisionEvaluator.evaluate_package(policy, tasks, *, stage) -> PackageEvaluation`.
- Produces: `PackagePipelineEvaluator.evaluate(bundle: PackageCoordinateBundle, registry: FrozenNumericalPackageRegistry, tasks: Sequence[ContextTask], *, stage: str, cache_only: bool = False) -> PackageEvaluation`.

- [ ] **Step 1: Write failing aggregate and tail tests**

```python
def test_package_evaluation_aggregates_both_scaled_metrics_and_failure_burden():
    rows = _score_rows()
    evaluation = PackageEvaluation.from_rows(
        "a" * 64,
        rows,
        expected_task_ids=tuple(row.task_id for row in rows),
    )
    assert evaluation.task_count == 4
    assert evaluation.coverage == 1.0
    assert evaluation.mean_joint == pytest.approx(
        (evaluation.mean_smae + evaluation.mean_srmse) / 2.0
    )
    assert evaluation.p90_srmse == pytest.approx(
        linear_quantile([row.final_srmse for row in rows], 0.90)
    )
    assert evaluation.p95_srmse == pytest.approx(
        linear_quantile([row.final_srmse for row in rows], 0.95)
    )
    assert evaluation.invalid_count == 1
    assert evaluation.fallback_count == 1
    assert evaluation.clipped_count == 1


def test_full_gate_rejects_srmse_tail_regression_even_when_means_improve():
    failures = package_full_gate_failures(
        _evaluation(mean_smae=0.90, mean_srmse=0.90, p95_srmse=1.10),
        _evaluation(mean_smae=1.00, mean_srmse=1.00, p95_srmse=1.00),
        PackageGateConfig(),
        stage="calibration",
    )
    assert "p95_srmse" in failures


def test_pipeline_preserves_round1_when_typed_round2_is_malformed():
    evaluation = pipeline_evaluator.evaluate(
        bundle,
        registry,
        (round2_task,),
        stage="screen8",
    )
    assert evaluation.task_rows[0].fallback_count == 0
    assert evaluation.task_rows[0].final_retrieval_sha256 == expected_round1_sha256
    assert retrieval.calls == ("round1", "round2")
```

- [ ] **Step 2: Run metric tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_metrics.py`

Expected: collection fails because `evolving_loop.package_metrics` does not exist.

- [ ] **Step 3: Implement immutable rows and aggregates**

```python
@dataclass(frozen=True)
class PackageTaskScore:
    task_id: str
    entity_name: str
    final_smae: float
    final_srmse: float
    final_smae_raw: float
    final_srmse_raw: float
    smae_clipped: bool
    srmse_clipped: bool
    invalid_count: int
    catastrophic_count: int
    fallback_count: int
    selected_candidate_id: str
    numerical_package_sha256: str
    final_retrieval_sha256: str
    final_decision_sha256: str

    @property
    def joint(self) -> float:
        return (self.final_smae + self.final_srmse) / 2.0


@dataclass(frozen=True)
class PackageGateConfig:
    tolerance: float = 1e-12
    minimum_relative_joint_gain: float = 0.005
    maximum_task_joint_regret: float = 0.25
```

`PackageEvaluation.from_rows` rejects duplicate tasks, non-finite metrics, empty expected membership, malformed hashes, negative counts, and missing selections. It compares row IDs with `expected_task_ids`, records missing IDs, and computes coverage from the expected count rather than from successful rows. It computes capped/raw means, P90/P95 for both metrics, total clipping/failure burden, and a canonical payload/fingerprint. Define catastrophe as `smae_raw > 10.0 or srmse_raw > 10.0`; keep catastrophe and clipping as separate counts. `result_bytes()` removes only `candidate_sha256`; it does not remove any behavior or score field.

Store Numerical oracle gap, Retrieval supporting recall/distractor avoidance/exact-quote validity/temporal match/complete-chain rate, and Decision selection regret under `secondary_diagnostics`. These fields are proposer diagnostics and report fields only; neither screen nor full acceptance may substitute them for the final sMAE/sRMSE gates.

- [ ] **Step 4: Write failing screen, gain, regret, fold, and Toto tests**

```python
def test_screen_requires_pareto_safety_but_not_half_percent_gain():
    parent = _uniform_evaluation(1.0)
    child = _uniform_evaluation(0.999)
    assert package_screen_failures(child, parent, PackageGateConfig()) == ()


def test_full_gate_requires_half_percent_joint_gain():
    parent = _uniform_evaluation(1.0)
    child = _uniform_evaluation(0.996)
    assert "minimum_relative_joint_gain" in package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="calibration",
    )


def test_build_gate_requires_four_safe_and_three_improving_folds():
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="build",
        fold_manifest=five_fold_manifest,
    )
    assert "nonregressing_folds" in failures
    assert "improving_folds" in failures


def test_task_185_shaped_overlay_is_rejected_by_joint_regret():
    parent = _evaluation_with_task("task_185", smae=0.3147676772, srmse=0.5355035986)
    child = _evaluation_with_task("task_185", smae=1.1801735667, srmse=1.4480493045)
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="calibration",
    )
    assert "maximum_task_joint_regret" in failures


def test_child_must_remain_pareto_safe_against_initial_toto():
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="dev",
        initial=initial_toto,
    )
    assert "initial_toto_mean_smae" in failures
```

- [ ] **Step 5: Implement screen and full gates**

`package_screen_failures` checks exact task coverage, mean capped sMAE/sRMSE non-regression, no increase in invalid/catastrophic/clipped/fallback counts, and no Public marker. `package_full_gate_failures` adds:

```python
relative_gain = (
    (parent.mean_joint - child.mean_joint) / parent.mean_joint
    if parent.mean_joint > 0.0
    else 0.0
)
```

Require `relative_gain >= 0.005 - 1e-12`, non-regressing P90/P95 capped sMAE and sRMSE, maximum paired task joint regret `<= 0.25 + 1e-12`, and Pareto non-regression against `initial`. For Build, derive per-fold sub-evaluations from `GroupFoldManifest.task_fold_map`; a fold is non-regressing only when both capped means pass, and improving only when it is non-regressing and its joint mean is strictly lower by more than `1e-12`.

- [ ] **Step 6: Make existing package evaluators emit the shared rows**

Implement `PackagePipelineEvaluator` as the only full-pipeline scorer. For each task it resolves `registry.package_for(task)`, constructs Retrieval and Decision from `bundle.policy`, and calls `run_numerical_two_stage`. Require `registry.fingerprint == bundle.numerical_manifest_sha256` before scoring. If the inference call raises a deterministic contract error, emit the task's protected-anchor forecast with `invalid_count=1` and `fallback_count=1`; do not drop the row. Transient model errors follow the existing bounded retry policy, while deterministic contract errors are never retried.

Add `evaluate_package` to both existing evaluators and delegate their task scoring to this shared evaluator. The compatibility `evaluate` methods construct their existing `RetrievalEvaluation` or `PolicyEvaluation` from the shared result, so existing callers do not fork scoring logic.

Include `p90_srmse`, `p95_srmse`, raw metrics, clipped count, fallback count, coverage, selected candidate, and all three final artifact hashes in shared rows. Set `public_test_accessed=False` unconditionally inside evolution evaluators; the Public evaluator uses a different entry point.

- [ ] **Step 7: Run focused and compatibility tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_metrics.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py tests/test_retrieval_evolution.py`

Expected: all tests pass and legacy Retrieval defaults still use contextual strict gain outside package mode.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/package_metrics.py evolving_loop/package_pipeline_evaluator.py evolving_loop/package_retrieval_evolution.py evolving_loop/package_decision_evolution.py tests/test_package_metrics.py tests/test_package_retrieval_evolution.py tests/test_package_decision_evolution.py tests/test_retrieval_evolution.py
git commit -m "feat(evolution): unify package final gates"
```

---

### Task 3: Build the Numerical coordinate proposer and materializer

**Files:**
- Create: `evolving_loop/package_numerical_evolution.py`
- Modify: `numerical_agent/evolution/champion_controller.py`
- Modify: `numerical_agent/evolution/specialist_atlas.py`
- Create: `tests/test_package_numerical_evolution.py`
- Modify: `tests/test_specialist_atlas.py`

**Interfaces:**
- Consumes: `ChampionProposerAdapter.propose`, `ProposerEvidence`, `expand_recipe`, existing frozen Build rows, `ForecastStore`, and `build_package_registry`.
- Produces: `NumericalRecipeFit(recipe, full_build_policy, build_fold_policies, numerical_score_sha256)`.
- Produces: `fit_numerical_recipe(recipe: ChampionRecipe, build_rows: Sequence[ChampionTaskRow], fold_manifest: GroupFoldManifest, parent: ChampionRelease) -> NumericalRecipeFit`.
- Produces: `AtlasRelease`, `parse_atlas_release`, `fit_atlas_release`, `fit_atlas_oof`, and `route_atlas_task` from the committed Atlas feature/pool base.
- Produces: `NumericalCoordinateCandidate(release, registry, proposal_sha256, invalid_reason=None)`.
- Produces: `NumericalPackageMaterializer.materialize(parent_release, fit, tasks, *, version, generation) -> NumericalCoordinateCandidate`.
- Produces: `NumericalPackageProposer.propose(parent_release, parent_registry, feedback, *, generation, child_count=3) -> tuple[NumericalCoordinateCandidate, ...]`.

- [ ] **Step 1: Write failing bounded fitter tests**

```python
def test_fit_numerical_recipe_uses_host_grid_and_deterministic_rank():
    fitted = fit_numerical_recipe(recipe, build_rows, fold_manifest, parent_release)
    expanded = expand_recipe(recipe, build_rows)
    assert fitted.full_build_policy in expanded
    assert fitted.recipe == recipe
    assert tuple(fold for fold, _policy in fitted.build_fold_policies) == (0, 1, 2, 3, 4)
    assert len(fitted.numerical_score_sha256) == 64


def test_fit_numerical_recipe_never_reads_dev_or_public_rows():
    fit_numerical_recipe(recipe, build_rows, fold_manifest, parent_release)
    assert row_provider.requested_splits == ["build"]


def test_each_build_fold_policy_excludes_its_held_out_groups():
    fitted = fit_numerical_recipe(recipe, build_rows, fold_manifest, parent_release)
    for fold, policy in fitted.build_fold_policies:
        assert fitter.task_ids_used_for(policy).isdisjoint(
            task_ids_for_fold(fold_manifest, fold)
        )
```

- [ ] **Step 2: Run fitter tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_numerical_evolution.py -k 'fit_numerical_recipe'`

Expected: collection fails because `evolving_loop.package_numerical_evolution` does not exist.

- [ ] **Step 3: Extract a reviewed host-only fitted-policy helper**

Add this public helper beside `run_build_evolution` so package evolution does not invoke private controller state:

```python
def fit_champion_recipe(
    recipe: ChampionRecipe,
    rows: tuple[ChampionTaskRow, ...],
    parent: ChampionRelease,
) -> FittedChampionPolicy:
    policies = expand_recipe(recipe, rows)
    task_ids = tuple(dict.fromkeys(row.task_id for row in rows))
    scored = tuple(
        (
            policy,
            score_policy(
                _materialize_policy(
                    policy,
                    f"package_fit_{champion_fingerprint(policy)}",
                    rows,
                    task_ids,
                    executor=execute_champion,
                ),
                f"package_fit_{champion_fingerprint(policy)}",
            ),
        )
        for policy in policies
    )
    return min(
        scored,
        key=lambda item: (
            joint_scaled_error(item[1].mean_smae, item[1].mean_srmse),
            item[1].mean_srmse,
            item[1].mean_smae,
            champion_fingerprint(item[0]),
        ),
    )[0]
```

Keep `_materialize_policy` private; the new helper is the only public boundary. Validate exact input types, complete Build coverage, immutable parent/row fingerprints before and after expansion/execution, finite forecasts, and deterministic tie-breaking. Export `fit_champion_recipe` through `__all__`.

Define the cross-fit result as:

```python
@dataclass(frozen=True)
class NumericalRecipeFit:
    recipe: ChampionRecipe
    full_build_policy: FittedChampionPolicy
    build_fold_policies: tuple[tuple[int, FittedChampionPolicy], ...]
    full_build_task_ids: tuple[str, ...]
    fold_training_task_ids: tuple[tuple[int, tuple[str, ...]], ...]
    numerical_score_sha256: str
```

`fit_numerical_recipe` calls the host helper six times: once with all 64 Build rows to produce the frozen full-Build policy used on Calibration/Dev, and once per held-out fold using only the other four folds. It records exact fitting task membership beside each policy and rejects any overlap with the held-out fold.

- [ ] **Step 4: Write failing Atlas release and cross-fit routing tests**

```python
def test_atlas_release_contains_full_build_and_five_oof_models():
    release = fit_atlas_release(rows, fold_manifest, AtlasPolicy())
    assert tuple(fold for fold, _model in release.build_fold_models) == (0, 1, 2, 3, 4)
    assert release.full_build_model.training_task_count == 64


def test_atlas_oof_route_never_uses_held_out_group_labels():
    result = fit_atlas_oof(rows, fold_manifest, AtlasPolicy())
    for task_result in result.tasks:
        assert task_result.group_sha256 not in task_result.training_group_sha256s


def test_atlas_task_185_shape_falls_back_when_predicted_regret_exceeds_quarter():
    routed = route_atlas_task(
        catastrophic_overlay_case,
        atlas_release,
        fold=None,
    )
    assert routed.activated is False
    assert routed.forecast == catastrophic_overlay_case.anchor.forecast
    assert routed.fallback_reason == "predicted_regret_exceeds_limit"
```

- [ ] **Step 5: Implement the frozen Atlas release and routing boundary**

Add exact frozen types `AtlasCandidateEstimate`, `AtlasModel`, `AtlasRelease`, `AtlasTaskResult`, and `AtlasOOFResult`. `AtlasModel` stores only the normalized history-only feature scales, fitting task/group hashes, selected pool, neighbor count, and the bounded aggregate win/effect/regret targets needed for deterministic nearest-neighbor routing. `AtlasRelease` contains five held-out-fold models plus one full-Build model and canonical source/policy/fold fingerprints.

`fit_atlas_oof` trains each fold model without the held-out groups and routes only that fold. `fit_atlas_release` also fits the full-Build model before any Calibration access. `route_atlas_task(task_case, atlas_release, fold=fold_index)` uses the matching out-of-fold model for Build, where `fold_index` is an integer from `0` through `4`; `fold=None` uses the already frozen full-Build model for Calibration and Dev. Inputs are limited to `AtlasFeature`, materialized candidate/anchor forecasts, and history-only diagnostics. Enforce the Atlas policy's minimum group support, win probability, effect margin, and maximum predicted regret `0.25`; otherwise return the exact anchor.

- [ ] **Step 6: Write failing Numerical proposal and label-firewall tests**

```python
def test_numerical_proposer_returns_three_direct_lineage_states():
    children = proposer.propose(
        parent_release,
        parent_registry,
        feedback,
        generation=0,
        child_count=3,
    )
    assert len(children) == 3
    assert len({child.proposal_sha256 for child in children}) == 3
    assert all(child.release.parent_sha256 == parent_release.fingerprint for child in children)
    assert all(child.registry.release_sha256 == child.release.fingerprint for child in children)


def test_numerical_materialization_receives_history_only_tasks():
    proposer.propose(
        parent_release,
        parent_registry,
        feedback,
        generation=0,
        child_count=3,
    )
    assert materializer.seen_tasks
    assert all(task.numeric.future_values == () for task in materializer.seen_tasks)
    assert all(task.gt_evidence == () for task in materializer.seen_tasks)
    assert all(document.role is None for task in materializer.seen_tasks for document in task.documents)
```

- [ ] **Step 7: Implement `NumericalPackageMaterializer`**

Define the proposal result before the materializer:

```python
@dataclass(frozen=True)
class NumericalCoordinateCandidate:
    release: NumericalSupplyRelease
    registry: FrozenNumericalPackageRegistry
    proposal_sha256: str
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.registry.release_sha256 != self.release.fingerprint:
            raise NumericalPackageEvolutionError("candidate registry release mismatch")
```

The materializer receives all 100 evolution tasks only after creating label-free copies with `ContextTask.numeric_view()`, empty `gt_evidence`, `labels_public=False`, and document role/subtype removed. It executes the matching fold-fitted Numerical/Atlas policy for every Build task and the frozen full-Build policies for Calibration and Dev. It executes the selected Statistical/TSFM/Combined suppliers through the existing ForecastStore and `run_numerical_loop`, then calls `bound_numerical_package` and `build_package_registry` against the original host-held tasks.

An Atlas result enters `materialized` only when `route_atlas_task` produces a finite full-horizon forecast and its history-only evidence passes its own policy; otherwise the anchor remains available and the Atlas candidate is omitted for that task. The fixed 70/30 forecast is represented by an explicit `atlas_overlay` alternative spec and is never assigned as the protected anchor.

Keep Toto as the first formal run's anchor unless a non-Atlas fitted policy is explicitly proposed as `materializer_kind="champion"`, is used as the host default during the full Build/Calibration/Dev comparisons, and passes every full-pipeline gate. Merely being selected on some tasks by Decision cannot promote a candidate to anchor. Atlas and bounded-overlay alternatives are never eligible for anchor replacement in this plan.

- [ ] **Step 8: Implement `NumericalPackageProposer`**

Construct a sanitized `ProposerEvidence` containing only anonymous Build aggregates, morphology aggregates, structural identities, and gate names. Call `ChampionProposerAdapter.propose` once for the generation. Reject duplicate structures and recipes that reference candidates outside the reviewed Dictionary. Fit proposals in canonical recipe fingerprint order and return the first three valid, distinct Child states; if fewer than three are valid, return typed invalid Child records for the remaining slots so the stage trace still records exactly three proposals.

Each Child creates a `NumericalSupplyRelease` whose version is `f"n{generation * 3 + slot + 1:03d}"`, whose `parent_sha256` equals the supplied Parent release fingerprint, and whose remaining fields come from the validated anchor, fitted recipe, selected family slots, and registered source/runtime identities. It also creates a complete Train+Dev registry. Task 5 converts this pair into a bundle only through `parent.with_numerical(release, registry)`, which preserves Retrieval and Decision canonical bytes exactly.

- [ ] **Step 9: Run Numerical coordinate tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_numerical_evolution.py tests/test_package_numerical_supply.py tests/test_evolution_champion_controller.py tests/test_specialist_atlas.py`

Expected: all tests pass; the package-specific helper does not change the established Champion lifecycle behavior.

- [ ] **Step 10: Commit**

```bash
git add evolving_loop/package_numerical_evolution.py numerical_agent/evolution/champion_controller.py numerical_agent/evolution/specialist_atlas.py tests/test_package_numerical_evolution.py tests/test_package_numerical_supply.py tests/test_evolution_champion_controller.py tests/test_specialist_atlas.py
git commit -m "feat(numerical): propose package coordinate"
```

---

### Task 4: Extend the package bundle and bind it to a runtime registry state

**Files:**
- Modify: `evolving_loop/package_coordinate_evolution.py`
- Modify: `tests/test_package_coordinate_evolution.py`
- Modify: `tests/test_package_coordinate_e2e.py`

**Interfaces:**
- Extends: `PackageCoordinateTarget = Literal["numerical", "retrieval", "decision"]`.
- Extends: `PackageCoordinateBundle` with `numerical_release_payload`, `numerical_release_sha256`, `coordinate`, `acceptance_evidence_sha256`, and schema version `2`.
- Produces: `PackageCoordinateState(bundle: PackageCoordinateBundle, registry: FrozenNumericalPackageRegistry)`.
- Produces: `PackageCoordinateState.with_numerical(release, registry) -> PackageCoordinateState`.
- Produces: `PackageCoordinateState.with_policy(policy: HarnessPolicy, target: Literal["retrieval", "decision"]) -> PackageCoordinateState`.
- Extends: `PackageCoordinateStep` with `parent_registry_sha256`, `child_registry_sha256`, and `accepted_registry_sha256`.
- Preserves: canonical principal fingerprint names `numerical`, `retrieval`, and `decision`.

- [ ] **Step 1: Write failing schema and state-binding tests**

```python
def test_bundle_schema_binds_numerical_release_and_registry_manifest():
    bundle = _bundle(numerical_release=release, registry=registry)
    payload = bundle.to_payload()
    assert payload["schema_version"] == 2
    assert payload["coordinate"] == "seed"
    assert payload["numerical_release_sha256"] == release.fingerprint
    assert payload["numerical_manifest_sha256"] == registry.fingerprint


def test_coordinate_state_rejects_release_or_registry_mismatch():
    with pytest.raises(ValueError, match="registry manifest"):
        PackageCoordinateState(_bundle(registry=left), right)
    with pytest.raises(ValueError, match="Numerical release"):
        PackageCoordinateState(_bundle(numerical_release=left_release), right_registry)
```

- [ ] **Step 2: Run bundle tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_coordinate_evolution.py -k 'schema or state'`

Expected: tests fail because the bundle has no Numerical release payload and no runtime state type.

- [ ] **Step 3: Extend the canonical bundle**

```python
PackageCoordinateTarget = Literal["numerical", "retrieval", "decision"]
PackageCoordinateName = Literal["seed", "numerical", "retrieval", "decision"]


@dataclass(frozen=True)
class PackageCoordinateBundle:
    generation: int
    coordinate: PackageCoordinateName
    parent_sha256: str | None
    numerical_release_payload: Mapping[str, object]
    numerical_release_sha256: str
    numerical_manifest_sha256: str
    policy: HarnessPolicy
    runtime_fingerprints: Mapping[str, str]
    acceptance_evidence_sha256: str | None
```

Schema-2 validation parses `numerical_release_payload` with `parse_numerical_supply_release`, matches both Numerical digests, requires `coordinate="seed"` only for generation zero, and uses this exact runtime fingerprint key set: `bridge_runtime`, `numerical_runtime`, `retrieval_runtime`, `decision_runtime`, `retrieval_verifier`, `metric_policy`, `model_runtime`, and `llm_runtime`. Seed and provisional Child bundles have `acceptance_evidence_sha256=None`; every accepted non-seed bundle is sealed with the canonical evidence digest before publication. `package_principal_fingerprints` excludes acceptance metadata and hashes the Numerical release payload, registry manifest digest, bridge runtime, Numerical runtime, model runtime, and metric policy together.

Add `with_numerical(release, registry)` and `with_policy(policy, target)` methods that increment generation, set `parent_sha256` to the current bundle fingerprint, set the declared coordinate, clear prior acceptance evidence, and preserve every unowned byte. Add `seal_acceptance(evidence_sha256)` which changes only acceptance metadata and is callable once on a provisional direct Child.

- [ ] **Step 4: Implement the runtime state and ownership checks**

```python
@dataclass(frozen=True)
class PackageCoordinateState:
    bundle: PackageCoordinateBundle
    registry: FrozenNumericalPackageRegistry

    def __post_init__(self) -> None:
        if self.registry.fingerprint != self.bundle.numerical_manifest_sha256:
            raise ValueError("package state registry manifest mismatch")
        if self.registry.release_sha256 != self.bundle.numerical_release_sha256:
            raise ValueError("package state Numerical release mismatch")

    def with_numerical(
        self,
        release: NumericalSupplyRelease,
        registry: FrozenNumericalPackageRegistry,
    ) -> "PackageCoordinateState":
        return PackageCoordinateState(
            self.bundle.with_numerical(release, registry),
            registry,
        )

    def with_policy(
        self,
        policy: HarnessPolicy,
        target: Literal["retrieval", "decision"],
    ) -> "PackageCoordinateState":
        return PackageCoordinateState(
            self.bundle.with_policy(policy, target=target),
            self.registry,
        )
```

Update `PackageCoordinateStep.__post_init__` and `_step` to accept `numerical` and record all three registry hashes. A valid accepted transition has exact direct lineage and `changed_modules == (target,)`; a rejection's accepted bundle and registry fingerprints must equal the Parent exactly.

- [ ] **Step 5: Update existing fake fixtures to construct schema-2 seed bundles**

Build a deterministic `n000` supply release around the existing safe anchor and pass its fingerprint into `FrozenNumericalPackageRegistry`. Keep the original R→D test as a compatibility case but make the selected object a `PackageCoordinateState`.

- [ ] **Step 6: Run focused tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_coordinate_evolution.py tests/test_package_coordinate_e2e.py tests/test_package_numerical_supply.py`

Expected: all tests pass; changing a Numerical release, registry, Retrieval release, or Decision field changes only its matching principal fingerprint.

- [ ] **Step 7: Commit**

```bash
git add evolving_loop/package_coordinate_evolution.py tests/test_package_coordinate_evolution.py tests/test_package_coordinate_e2e.py
git commit -m "feat(evolution): bind numerical coordinate state"
```

---

### Task 5: Add typed Child proposal adapters for all three coordinates

**Files:**
- Create: `evolving_loop/package_candidate_proposal.py`
- Modify: `evolving_loop/package_numerical_evolution.py`
- Modify: `evolving_loop/retrieval_agent/evolution.py`
- Modify: `evolving_loop/package_decision_evolution.py`
- Create: `tests/test_package_candidate_proposal.py`
- Modify: `tests/test_package_numerical_evolution.py`
- Modify: `tests/test_retrieval_evolution.py`
- Modify: `tests/test_package_decision_evolution.py`

**Interfaces:**
- Produces: `PackageProposalFeedback` containing only anonymous aggregate metrics, gate names, and closed structural identities.
- Produces: `PackageCandidate(slot, target, state, proposal_sha256, invalid_reason=None)`.
- Produces: `PackageCandidateProposer.propose(parent, feedback, *, generation, child_count) -> tuple[PackageCandidate, ...]` protocol.
- Produces: `NumericalCandidateProposer`, `RetrievalCandidateProposer`, and `DecisionCandidateProposer` implementations.
- Produces: `embed_retrieval_candidate(policy, genome, skills, *, changelog) -> HarnessPolicy` without accepted-release authority.
- Extracts: `RetrievalGenomeProposer.propose(parent, *, generation, feedback, skill_library) -> tuple[RetrievalGenome | None, ...]` from the existing Retrieval engine.
- Preserves: existing `RetrievalEvolutionEngine.evolve` and `PackageDecisionEvolutionEngine.evolve` behavior through delegation.

- [ ] **Step 1: Write failing feedback-redaction and proposal-count tests**

```python
def test_package_feedback_contains_no_task_ids_values_or_dev_residuals():
    feedback = PackageProposalFeedback.from_evaluations(
        parent=parent_evaluation,
        rejected_children=rejected_evaluations,
        gate_names=("p95_srmse", "maximum_task_joint_regret"),
    )
    payload = json.dumps(feedback.to_payload(), sort_keys=True)
    assert "task_185" not in payload
    assert "future_values" not in payload
    assert "task_traces" not in payload
    assert set(feedback.gate_names) == {"maximum_task_joint_regret", "p95_srmse"}


@pytest.mark.parametrize("target", ("numerical", "retrieval", "decision"))
def test_each_proposer_returns_exactly_three_canonical_slots(target):
    children = proposers[target].propose(
        parent_state,
        feedback,
        generation=2,
        child_count=3,
    )
    assert tuple(child.slot for child in children) == (0, 1, 2)
    assert all(child.target == target for child in children)
```

- [ ] **Step 2: Run proposal tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_candidate_proposal.py`

Expected: collection fails because the proposal adapters do not exist.

- [ ] **Step 3: Implement the feedback and candidate contracts**

```python
@dataclass(frozen=True)
class PackageProposalFeedback:
    parent_summary: Mapping[str, float | int]
    rejected_summaries: tuple[Mapping[str, float | int | str], ...]
    gate_names: tuple[str, ...]
    structures: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class PackageCandidate:
    slot: int
    target: PackageCoordinateTarget
    state: PackageCoordinateState
    proposal_sha256: str
    invalid_reason: str | None = None
```

Only these aggregate keys may enter `parent_summary`: `task_count`, `mean_smae`, `mean_srmse`, `mean_joint`, `p90_smae`, `p95_smae`, `p90_srmse`, `p95_srmse`, `invalid_count`, `catastrophic_count`, `clipped_count`, `fallback_count`, and `coverage`. Reject any mapping containing a task ID, entity, forecast, truth, document, quote, residual, or per-task metric.

An invalid proposal still occupies its canonical slot, retains the Parent state exactly, and records one of the closed reason codes `invalid_schema`, `duplicate_child`, `unknown_candidate`, `materialization_failed`, or `cross_coordinate_change`.

- [ ] **Step 4: Extract the pure Retrieval proposal path**

Create `RetrievalGenomeProposer` in `evolving_loop/retrieval_agent/evolution.py` using the current `_request_child` payload, `CHILD_SCOPES`, `parse_scoped_child`, bounded retry logic, and read-only Skill catalog. Its method returns exactly the A/B/C slots and never evaluates tasks, writes Skills, publishes a release, or opens Dev.

Change `RetrievalEvolutionEngine._children_for_generation` to delegate fresh proposal creation to this class and keep its checkpoint serialization unchanged. Add a regression test comparing the old deterministic `_retrieval_proposals` fixture with the extracted proposer fingerprints.

- [ ] **Step 5: Implement the three package adapters**

- `NumericalCandidateProposer` calls the Task-3 Numerical proposer, converts each valid `NumericalCoordinateCandidate` to `parent.with_numerical(release, registry)`, and rejects any Retrieval/Decision fingerprint change.
- `RetrievalCandidateProposer` asks `RetrievalGenomeProposer` for A/B/C genomes, constructs a canonical non-authoritative `state="candidate"` payload with a new `embed_retrieval_candidate` helper, converts it with `parent.with_policy(candidate_policy, target="retrieval")`, and rejects Numerical/Decision changes. The Skill library clone is always `read_only=True`. Candidate payloads are evidence, not accepted release authority; only the trusted publisher used after Dev may create `state="accepted"`.
- `DecisionCandidateProposer` calls `PackageDecisionEvolutionEngine.mutate` three times using a `PolicyEvaluation` synthesized only from `PackageProposalFeedback.parent_summary`; it converts each policy with `parent.with_policy(child_policy, target="decision")` and rejects Numerical/Retrieval changes.

Candidate-only Retrieval manifests use `state="candidate"`; only Task 6 may publish `state="accepted"` after all gates pass.

Define `retrieval_behavior_fingerprint(genome)` over every Genome field except `version` and `parent`. Sibling candidates retain distinct provisional `vNNN` identities, but the accepted winner is rebased to the next sequential accepted version with `parent` equal to the current accepted Retrieval version. Rebase is permitted only when `retrieval_behavior_fingerprint` is unchanged.

- [ ] **Step 6: Write and run ownership regression tests**

```python
def test_retrieval_proposal_preserves_numerical_and_decision_bytes():
    child = retrieval_proposer.propose(parent, feedback, generation=0, child_count=3)[0]
    before = package_principal_fingerprints(parent.bundle)
    after = package_principal_fingerprints(child.state.bundle)
    assert before["numerical"] == after["numerical"]
    assert before["decision"] == after["decision"]
    assert before["retrieval"] != after["retrieval"]


def test_decision_proposal_cannot_change_materialized_forecasts():
    proposal_llm.responses = ['{"decision_prompt":"x","forecast":[1,2]}']
    child = decision_proposer.propose(parent, feedback, generation=0, child_count=3)[0]
    assert child.state.registry.fingerprint == parent.registry.fingerprint
    assert child.state.bundle.numerical_manifest_sha256 == parent.bundle.numerical_manifest_sha256
    assert "forecast" not in child.state.bundle.policy.to_payload()
```

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_candidate_proposal.py tests/test_package_numerical_evolution.py tests/test_retrieval_evolution.py tests/test_package_decision_evolution.py`

Expected: all tests pass, and existing standalone Retrieval/Decision evolution remains behavior-compatible.

- [ ] **Step 7: Commit**

```bash
git add evolving_loop/package_candidate_proposal.py evolving_loop/package_numerical_evolution.py evolving_loop/retrieval_agent/evolution.py evolving_loop/package_decision_evolution.py tests/test_package_candidate_proposal.py tests/test_package_numerical_evolution.py tests/test_retrieval_evolution.py tests/test_package_decision_evolution.py
git commit -m "feat(evolution): propose isolated package children"
```

---

### Task 6: Implement the shared 8/32/64/16/20 phase runner

**Files:**
- Create: `evolving_loop/package_stage_runner.py`
- Create: `tests/test_package_stage_runner.py`

**Interfaces:**
- Consumes: `PackageCandidateProposer`, `PackageEvaluation`, `PackageGateConfig`, `GroupFoldManifest`, and the Build/Calibration/Dev task tuples.
- Produces: `PackageStageSchedule(seed=20260903, screen8_ids, screen32_ids, build64_ids, calibration16_ids, dev20_ids, fold_manifest)`.
- Produces: `PackageStageEvidence(stage, parent_sha256, candidate_sha256, parent_evaluation, child_evaluation, gate_failures, opened)`.
- Produces: `PackageCoordinatePhaseOutcome(target, parent, finalist, selected, accepted, improved, reason, evidence, public_test_accessed=False)`.
- Produces: `PackageCoordinatePhaseRunner(target, proposer, evaluator, schedule, task_map, gate_config, artifact_store)`.
- Produces: `PackageCoordinatePhaseRunner.run(parent, initial, *, generation) -> PackageCoordinatePhaseOutcome`.

- [ ] **Step 1: Write failing deterministic schedule tests**

```python
def test_schedule_is_nested_registered_and_group_aware():
    schedule = PackageStageSchedule.build(train80, dev20, seed=20260903)
    assert len(schedule.screen8_ids) == 8
    assert len(schedule.screen32_ids) == 32
    assert len(schedule.build64_ids) == 64
    assert len(schedule.calibration16_ids) == 16
    assert len(schedule.dev20_ids) == 20
    assert set(schedule.screen8_ids) <= set(schedule.screen32_ids)
    assert set(schedule.screen32_ids) <= set(schedule.build64_ids)
    assert set(schedule.build64_ids).isdisjoint(schedule.calibration16_ids)
    assert schedule.fold_manifest.fold_count == 5
```

- [ ] **Step 2: Write failing one-finalist access test**

```python
def test_only_one_child_can_open_calibration_and_dev():
    outcome = runner.run(parent_state, initial_state, generation=0)
    assert evaluator.child_stage_counts("screen8") == 3
    assert evaluator.child_stage_counts("screen32") <= 2
    assert evaluator.child_stage_counts("build64") <= 1
    assert evaluator.child_stage_counts("calibration16") <= 1
    assert evaluator.child_stage_counts("dev20") <= 1
    assert len(evaluator.child_fingerprints_seen_on("dev20")) <= 1
```

- [ ] **Step 3: Run stage tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_stage_runner.py`

Expected: collection fails because the stage runner does not exist.

- [ ] **Step 4: Implement registered schedule construction**

Build the 64/16 partition with the existing registered Train partition helper and build the five folds through `build_group_fold_manifest`. Select nested screen memberships deterministically from whole groups using seed `20260903`; fail closed if exact sizes 8 and 32 cannot be formed without splitting an indivisible group. Persist the resulting payload and fingerprint before proposals run.

`PackageStageSchedule.tasks_for(stage)` is the only method that returns task objects. It accepts the host-held task map and verifies exact task-content fingerprints; proposal adapters receive no task objects through this type.

- [ ] **Step 5: Implement successive halving**

```python
promote_limits = {
    "screen8": 2,
    "screen32": 1,
    "build64": 1,
    "calibration16": 1,
    "dev20": 1,
}
```

For each stage, evaluate the Parent once and each still-active Child with the complete Numerical→Retrieval→Decision inference pipeline. Apply `package_screen_failures` on screen8/screen32 and `package_full_gate_failures` on Build/Calibration/Dev. Sort eligible Children by `package_rank_key(evaluation) + (candidate.state.bundle.fingerprint(),)` and truncate to the registered limit.

Do not open Calibration unless one unchanged finalist passes Build. Do not open Dev unless the same fingerprint passes Calibration. Reject a finalist if any stage changes its bundle bytes, registry fingerprint, proposal fingerprint, or runtime identity.

- [ ] **Step 6: Implement accepted Retrieval publication and exact replay**

If a Retrieval finalist passes Dev, rebase its unchanged behavior to the next sequential accepted `vNNN` lineage, publish that genome and read-only Skill snapshot with `_write_accepted_retrieval_release`, rebuild the selected state from the operator-loaded accepted release, and evaluate it again from lower-layer inference caches. Accept only if the behavior fingerprint is unchanged and `replayed.result_bytes()` equals `prepublication.result_bytes()`. This comparison deliberately excludes candidate-versus-accepted publication metadata while binding every task forecast, metric, failure count, selected candidate, and final Numerical/Retrieval/Decision artifact hash.

Numerical and Decision finalists also require an immediate cache-backed replay before acceptance. A cache miss at replay returns `accepted=False` with reason `replay_cache_miss`; it never reruns the model at this boundary.

After replay passes, hash the complete Build, Calibration, Dev, gate, replay, and candidate evidence payload and call `selected.bundle.seal_acceptance(evidence_sha256)`. The accepted bundle cannot be published unless `PackageArtifactStore` already contains evidence bytes with that exact digest.

- [ ] **Step 7: Add failure-path tests**

```python
@pytest.mark.parametrize(
    "failure,expected_last_stage",
    (
        ("screen_regression", "screen8"),
        ("build_fold_regression", "build64"),
        ("task_185_tail", "calibration16"),
        ("dev_regression", "dev20"),
        ("replay_cache_miss", "dev20"),
    ),
)
def test_phase_stops_at_first_failed_gate(failure, expected_last_stage):
    outcome = _runner_for_failure(failure).run(parent_state, initial_state, generation=0)
    assert outcome.accepted is False
    assert outcome.evidence[-1].stage == expected_last_stage
    assert outcome.selected.bundle.fingerprint() == parent_state.bundle.fingerprint()
```

- [ ] **Step 8: Run focused tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_stage_runner.py tests/test_package_metrics.py tests/test_package_candidate_proposal.py`

Expected: all tests pass, with exact child evaluation counts `3, <=2, <=1, <=1, <=1` and zero Public access.

- [ ] **Step 9: Commit**

```bash
git add evolving_loop/package_stage_runner.py tests/test_package_stage_runner.py
git commit -m "feat(evolution): gate package halving stages"
```

---

### Task 7: Run two ordered N-R-D cycles with direct lineage and early stopping

**Files:**
- Modify: `evolving_loop/package_coordinate_evolution.py`
- Modify: `tests/test_package_coordinate_evolution.py`
- Modify: `tests/test_package_coordinate_e2e.py`

**Interfaces:**
- Consumes: one `PackageCoordinatePhaseRunner` for each of Numerical, Retrieval, and Decision.
- Produces: `PackageCoordinateController(numerical_phase, retrieval_phase, decision_phase, *, cycles=2)`.
- Produces: `PackageCoordinateController.run(parent, initial) -> tuple[PackageCoordinateState, tuple[PackageCoordinateStep, ...]]`.
- Produces: exactly ordered phase targets and an explicit skipped Decision record when no accepted non-`v000` Retrieval release exists.

- [ ] **Step 1: Replace the old two-step controller test with a six-step order test**

```python
def test_controller_runs_two_ordered_coordinate_cycles():
    selected, trace = controller.run(seed_state, seed_state)
    assert tuple(step.target for step in trace) == (
        "numerical",
        "retrieval",
        "decision",
        "numerical",
        "retrieval",
        "decision",
    )
    assert tuple(step.generation for step in trace) == tuple(range(6))
    assert selected.bundle.generation == sum(step.accepted for step in trace)
```

- [ ] **Step 2: Add rejection, Decision skip, and early-stop tests**

```python
def test_rejected_coordinate_preserves_exact_parent_state():
    selected, trace = _controller(reject="retrieval").run(seed_state, seed_state)
    step = trace[1]
    assert step.accepted is False
    assert step.accepted_bytes_sha256 == step.parent_bytes_sha256
    assert step.accepted_registry_sha256 == step.parent_registry_sha256


def test_decision_skips_without_nonseed_retrieval_release():
    selected, trace = _controller(reject="retrieval").run(seed_state, seed_state)
    assert trace[2].target == "decision"
    assert trace[2].reason == "Decision phase requires a non-v000 accepted Retrieval release"


def test_controller_stops_after_complete_cycle_with_no_acceptance():
    selected, trace = _controller(reject="all").run(seed_state, seed_state)
    assert tuple(step.target for step in trace) == ("numerical", "retrieval", "decision")
    assert selected.bundle.fingerprint() == seed_state.bundle.fingerprint()
```

- [ ] **Step 3: Run controller tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_coordinate_evolution.py -k 'cycles or rejected or skips or stops'`

Expected: failures because the existing controller only runs Retrieval then Decision once.

- [ ] **Step 4: Implement the cycle controller**

```python
phase_order: tuple[
    tuple[PackageCoordinateTarget, PackageCoordinatePhaseRunner | None], ...
] = (
    ("numerical", self.numerical_phase),
    ("retrieval", self.retrieval_phase),
    ("decision", self.decision_phase),
)
```

For each cycle, run all three entries in order. Use the current accepted state as both the next Parent and the registry authority. Validate every phase outcome target, direct parent hash, changed principal fingerprints, registry preservation/replacement rule, zero Public access, `accepted and improved` agreement, and exact accepted replay evidence.

Numerical acceptance must change both Numerical release and registry fingerprints while preserving Retrieval/Decision fingerprints. Retrieval or Decision acceptance must preserve the registry fingerprint and change only its named principal fingerprint. After the cycle, stop only when no step in that complete cycle was accepted.

- [ ] **Step 5: Update the deterministic e2e fixture**

Use scripted Numerical, Retrieval, and Decision proposers with deterministic package evaluations. Make the first cycle accept all three coordinates and the second cycle reject all three, proving six attempted steps without accessing Public. Assert every accepted step's parent hash equals the preceding accepted bundle hash and every rejected step preserves exact Parent bytes.

- [ ] **Step 6: Run focused tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_coordinate_evolution.py tests/test_package_coordinate_e2e.py tests/test_package_stage_runner.py`

Expected: all tests pass; the controller trace is ordered, causal, and transitive.

- [ ] **Step 7: Commit**

```bash
git add evolving_loop/package_coordinate_evolution.py tests/test_package_coordinate_evolution.py tests/test_package_coordinate_e2e.py
git commit -m "feat(evolution): run two package coordinate cycles"
```

---

### Task 8: Add immutable caches, trace artifacts, and exact resume

**Files:**
- Create: `evolving_loop/package_artifacts.py`
- Create: `tests/test_package_artifacts.py`

**Interfaces:**
- Produces: `PackageCacheKey(layer, task_sha256, candidate_sha256, dependency_fingerprints)` for layers `numerical`, `retrieval`, and `decision`.
- Produces: `PackageCheckpoint(schema_version, run_sha256, schedule_sha256, initial_bundle_payload, current_bundle_payload, completed_steps, candidate_fingerprints, cache_fingerprints, consumed_stages)`.
- Produces: `PackageCheckpoint.next_coordinate_generation` and `PackageCheckpoint.next_stage` derived from the verified trace and stage ledger.
- Produces: `PackageArtifactStore(output_dir: Path)` with `write_run_manifest`, `write_schedule`, `write_candidate_evidence`, `append_coordinate_step`, `publish_accepted_bundle`, `write_checkpoint`, `load_checkpoint`, and `complete`.
- Produces: `PackageInferenceCache(root: Path, *, cache_only: bool)` with exact dependency validation and write-once entries.

- [ ] **Step 1: Write failing cache-key and mismatch tests**

```python
def test_cache_key_changes_for_each_transitive_dependency():
    base = _cache_key()
    assert base.fingerprint != _cache_key(candidate_sha256="b" * 64).fingerprint
    assert base.fingerprint != _cache_key(task_sha256="c" * 64).fingerprint
    assert base.fingerprint != _cache_key(
        dependency_fingerprints={"runtime": "d" * 64}
    ).fingerprint


def test_cache_only_mismatch_fails_closed_without_callback(tmp_path):
    calls = []
    cache = PackageInferenceCache(tmp_path, cache_only=True)
    with pytest.raises(PackageCacheMissError, match="cache-only"):
        cache.get_or_compute(_cache_key(), lambda: calls.append("called"))
    assert calls == []
```

- [ ] **Step 2: Write failing checkpoint and consumed-Dev tests**

```python
def test_resume_rejects_schedule_or_bundle_drift(tmp_path):
    store = PackageArtifactStore(tmp_path)
    store.write_checkpoint(_checkpoint())
    with pytest.raises(PackageArtifactError, match="schedule"):
        store.load_checkpoint(
            expected_run_sha256="a" * 64,
            expected_schedule_sha256="b" * 64,
        )


def test_consumed_dev_stage_cannot_be_evaluated_twice(tmp_path):
    store = PackageArtifactStore(tmp_path)
    store.claim_stage("coordinate-2-dev20", candidate_sha256="c" * 64)
    store.commit_stage("coordinate-2-dev20", evaluation_sha256="d" * 64)
    with pytest.raises(PackageArtifactError, match="already consumed"):
        store.claim_stage("coordinate-2-dev20", candidate_sha256="c" * 64)
```

- [ ] **Step 3: Run artifact tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_artifacts.py`

Expected: collection fails because the artifact and cache types do not exist.

- [ ] **Step 4: Implement canonical write-once caches**

Cache payloads use this exact envelope:

```python
{
    "schema_version": 1,
    "key": key.to_payload(),
    "key_sha256": key.fingerprint,
    "value": value,
    "value_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
}
```

Write to a uniquely named file in the cache root, flush and `fsync` it, then publish without overwriting an existing canonical entry. A present entry is reusable only when its canonical key bytes, value digest, and every dependency fingerprint match. Any mismatch is a miss; `cache_only=True` raises before invoking the compute callback.

Numerical keys bind candidate implementation, exact task input, frequency, horizon, model runtime, and source fingerprints. Retrieval keys bind task/document fingerprint, Numerical assumption/package fingerprint, `retrieval_behavior_fingerprint`, Skill snapshot, verifier, and LLM runtime; lineage version/parent remain in release provenance but do not invalidate behavior-identical inference during trusted acceptance rebase. Decision keys bind Numerical package, verified Retrieval card, Decision policy, and Decision runtime.

- [ ] **Step 5: Implement append-only artifacts and checkpoint validation**

The output layout is exact:

```text
run_manifest.json
group_folds.json
checkpoint.json
coordinate_trace.jsonl
candidate_evidence/
accepted_bundles/
final_bundle.json
evaluation_complete.json
```

`run_manifest.json` and `group_folds.json` are write-once. Candidate evidence is stored at `candidate_evidence/<coordinate>-<generation>-<proposal_sha256>.json`. Accepted bundle bytes are stored at `accepted_bundles/<bundle_sha256>.json`. `coordinate_trace.jsonl` accepts only the next expected generation and exact Parent fingerprint. The checkpoint is atomically replaced but contains the hash of every previously immutable artifact and the current trace prefix.

`load_checkpoint` verifies run/schedule identity, initial and current bundle canonical bytes, direct lineage of every completed step, exact trace prefix, candidate evidence hashes, cache identities, and stage claims. It resumes from the first stage without a committed result and returns a committed Dev result without reevaluating it.

The checkpoint never serializes live Python objects. On resume, the runner parses the embedded Numerical supply release, rebuilds its `FrozenNumericalPackageRegistry` in cache-only mode for the registered Train+Dev membership, and requires the reconstructed manifest fingerprint to equal `current_bundle_payload["numerical_manifest_sha256"]` before restoring `PackageCoordinateState`.

- [ ] **Step 6: Test interrupted-stage resume**

```python
def test_resume_continues_from_first_incomplete_stage(tmp_path):
    store = PackageArtifactStore(tmp_path)
    store.write_checkpoint(_checkpoint(completed_steps=(step0,), consumed_stages=(screen8,)))
    resumed = store.load_checkpoint(
        expected_run_sha256=run_sha256,
        expected_schedule_sha256=schedule_sha256,
    )
    assert resumed.next_coordinate_generation == 1
    assert resumed.next_stage == "screen32"
    assert resumed.current_bundle_payload == step0.accepted_bundle_payload
```

- [ ] **Step 7: Run focused tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_package_artifacts.py tests/test_package_stage_runner.py tests/test_package_coordinate_evolution.py`

Expected: all tests pass; cache identity drift, detached checkpoints, overwritten artifacts, and repeated Dev claims fail closed.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/package_artifacts.py tests/test_package_artifacts.py
git commit -m "feat(evolution): checkpoint package coevolution"
```

---

### Task 9: Add the real package-native 80/20 runner

**Files:**
- Create: `evolving_loop/run_package_coevolution.py`
- Create: `scripts/run_package_coevolution.sh`
- Modify: `evolving_loop/data.py`
- Create: `tests/test_run_package_coevolution.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `load_context_tasks_by_ids(tasks_file, allowed_task_ids) -> tuple[ContextTask, ...]` without constructing unrequested Public tasks.
- Produces: `build_parser() -> argparse.ArgumentParser` and `main(argv: Sequence[str] | None = None) -> int`.
- Consumes: explicit split, tasks, Numerical Champion release, ForecastStore, optional Atlas release, Retrieval seed release, Retrieval Skill snapshot, model/runtime config, output directory, and authority directory paths.
- Produces: one `PackageCoordinateController` wired to the shared evaluator, stage runner, artifact store, and three proposers.

- [ ] **Step 1: Write failing CLI contract and Public-firewall tests**

```python
def test_parser_has_no_public_input_or_evaluation_flag():
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "public_tasks" not in destinations
    assert "evaluate_public" not in destinations
    assert "public_output" not in destinations


def test_runner_loads_only_train_and_dev_membership(monkeypatch, tmp_path):
    seen_ids = []
    monkeypatch.setattr(
        "evolving_loop.run_package_coevolution.load_context_tasks_by_ids",
        lambda _path, ids: seen_ids.extend(ids) or _tasks_for(ids),
    )
    assert main(_valid_args(tmp_path)) == 0
    assert set(seen_ids) == set(train_ids) | set(dev_ids)
    assert set(public_ids).isdisjoint(seen_ids)
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_run_package_coevolution.py`

Expected: collection fails because the real runner does not exist.

- [ ] **Step 3: Add the ID-filtered task loader**

`load_context_tasks_by_ids` validates a non-empty unique ID set, reads only records whose `benchmark_id` is in that set, converts those records with `_to_context_task`, and returns tasks in the exact requested order. It rejects missing, duplicate, unlabeled, or unexpected records. The co-evolution runner obtains only Train and Dev IDs from the split manifest and never passes Public IDs into this loader.

- [ ] **Step 4: Implement the exact CLI surface**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--numerical-champion-release", required=True)
    parser.add_argument("--forecast-store", required=True)
    parser.add_argument("--atlas-release")
    parser.add_argument("--retrieval-seed-release", required=True)
    parser.add_argument("--retrieval-skills", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--authority-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--children-per-coordinate", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    _add_tsfm_runtime_options(parser)
    return parser
```

Formal mode rejects any value other than cycles `2`, children `3`, seed `20260903`, Train `80`, Dev `20`, Build `64`, and Calibration `16`. Smoke mode is explicitly non-formal and registers Build `8`, Calibration `2`, Dev `2`, one cycle, and one Child per coordinate; its output manifest must contain `formal_run=false`.

- [ ] **Step 5: Wire immutable inputs and the three phases**

The runner must:

1. validate the split manifest digest and exact Train/Dev membership;
2. load the Champion release, ForecastStore, optional precomputed Atlas release, seed Retrieval release, and read-only Retrieval Skill snapshot;
3. fingerprint source files, runtime settings, model settings, verifier, metric policy, and every input artifact;
4. build and freeze an Atlas release from Build-only rows when `--atlas-release` is omitted, or verify the supplied release against the same Build/fold authority;
5. build the initial `n000` supply release with Toto as the safe anchor and up to four diverse materialized alternatives;
6. materialize the initial registry for the registered Train+Dev tasks;
7. write the run manifest and schedule before any proposal;
8. instantiate the Numerical, Retrieval, and Decision proposers and one shared package evaluator/cache;
9. run `PackageCoordinateController(numerical_phase, retrieval_phase, decision_phase, cycles=2)` or resume the exact checkpoint;
10. publish `final_bundle.json` only after the controller completes; and
11. write `evaluation_complete.json` with status `complete`, the final bundle hash, accepted/rejected step counts, and `public_test_accessed=false`.

The runner passes `strict_gain_target="final"` into Retrieval configuration, clones Retrieval Skills read-only, and never calls `scripts/run_co_evolution.py`, `load_huggingface_context_tasks`, or a Public evaluator.

- [ ] **Step 6: Add resume and wrong-runtime tests**

```python
def test_resume_reuses_committed_dev_without_second_model_call(tmp_path):
    first = _run_until_checkpoint_after_dev(tmp_path)
    calls_before = first.llm_calls
    assert main(_resume_args(tmp_path)) == 0
    assert first.llm_calls == calls_before


def test_resume_rejects_runtime_fingerprint_change(tmp_path):
    _run_until_checkpoint(tmp_path)
    with pytest.raises(PackageArtifactError, match="runtime"):
        main(_resume_args(tmp_path, model="different-model"))
```

- [ ] **Step 7: Implement the thin shell wrapper and document commands**

`scripts/run_package_coevolution.sh` must contain only strict shell options, repository-root resolution, `PYTHON_BIN="${TIME_SERIES_PYTHON:-python3}"`, and `exec "$PYTHON_BIN" -u -m evolving_loop.run_package_coevolution "$@"`. The README must state that Dev is part of evolution, Public-99 is historically consumed final regression evidence, and Public output cannot be fed back into evolution.

- [ ] **Step 8: Run focused CLI tests and static checks**

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_run_package_coevolution.py tests/test_package_artifacts.py tests/test_package_coordinate_e2e.py
/Users/yyoraa/time-series/.venv/bin/python -m compileall -q evolving_loop numerical_agent
git diff --check
```

Expected: all commands exit zero, and the CLI integration tests observe zero Public task IDs.

- [ ] **Step 9: Commit**

```bash
git add evolving_loop/run_package_coevolution.py evolving_loop/data.py scripts/run_package_coevolution.sh tests/test_run_package_coevolution.py README.md
git commit -m "feat(evolution): run package coevolution"
```

---

### Task 10: Add the separate frozen Public evaluator and attribution report

**Files:**
- Create: `evolving_loop/evaluate_frozen_package_bundle.py`
- Create: `tests/test_evaluate_frozen_package_bundle.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `VerifiedPackageRun(run_manifest, schedule, initial_state, final_state, trace, runtime_fingerprints)`.
- Produces: `NamedPackageState(name, state, direct_parent_name)`.
- Produces: `verify_frozen_package_run(evolution_dir, final_bundle_path, runtime_fingerprints) -> VerifiedPackageRun`.
- Produces: `build_attribution_states(verified: VerifiedPackageRun) -> tuple[NamedPackageState, ...]` for `initial_toto`, `final_numerical_seed_context`, `final_numerical_retrieval_seed_decision`, and `final_bundle`.
- Produces: `score_frozen_states(tasks, states, evaluator) -> Mapping[str, object]`.
- Produces: Public CLI `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing sealed-run verification tests**

```python
@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing_complete", "complete evolution"),
        ("trace_gap", "canonical trace"),
        ("detached_final", "final bundle"),
        ("public_marker", "Public access"),
        ("runtime_drift", "runtime"),
    ),
)
def test_public_evaluator_rejects_unsealed_or_drifted_run(tmp_path, mutation, match):
    run = _mutated_run(tmp_path, mutation)
    with pytest.raises(FrozenPackageEvaluationError, match=match):
        verify_frozen_package_run(run.path, run.final_bundle, run.runtime_fingerprints)
```

- [ ] **Step 2: Write failing one-shot and membership tests**

```python
def test_public_evaluator_loads_exactly_99_public_tasks(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "evolving_loop.evaluate_frozen_package_bundle.load_context_tasks_by_ids",
        lambda _path, ids: seen.extend(ids) or _tasks_for(ids),
    )
    assert main(_public_args(tmp_path)) == 0
    assert seen == public_ids
    assert len(seen) == 99


def test_public_output_directory_is_one_shot(tmp_path):
    assert main(_public_args(tmp_path)) == 0
    with pytest.raises(FrozenPackageEvaluationError, match="already"):
        main(_public_args(tmp_path))
```

- [ ] **Step 3: Run Public evaluator tests and verify RED**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_evaluate_frozen_package_bundle.py`

Expected: collection fails because the separate evaluator does not exist.

- [ ] **Step 4: Implement sealed-run verification and one-shot claiming**

Define the verified immutable boundary as:

```python
@dataclass(frozen=True)
class NamedPackageState:
    name: str
    state: PackageCoordinateState
    direct_parent_name: str | None


@dataclass(frozen=True)
class VerifiedPackageRun:
    run_manifest: Mapping[str, object]
    schedule: PackageStageSchedule
    initial_state: PackageCoordinateState
    final_state: PackageCoordinateState
    trace: tuple[PackageCoordinateStep, ...]
    runtime_fingerprints: Mapping[str, str]
```

Require `evaluation_complete.json.status == "complete"`, its final hash to equal canonical `final_bundle.json`, all trace rows to form one contiguous direct-lineage chain, every accepted transition to change exactly one declared principal fingerprint, every rejected transition to preserve Parent bytes, the schedule and runtime fingerprints to match, and every evolution Public marker to be false.

The Public output directory must differ from the evolution directory. Create `evaluation_started.json` with exclusive creation before loading Public tasks. If that file or `evaluation_complete.json` already exists, reject the invocation; an interrupted evaluation requires a new output directory and may not overwrite the first claim.

The Public parser accepts `--evolution-dir`, `--final-bundle`, `--split-file`, `--tasks-file`, `--repo`, `--forecast-store`, `--output-dir`, `--model`, and `--reasoning-effort`, then adds the same `_add_tsfm_runtime_options` used by evolution. It exposes no mutation, cycle, Child-count, Retrieval publication, or resume option.

- [ ] **Step 5: Implement the four frozen attribution states**

Reconstruct all states from immutable run artifacts:

1. `initial_toto`: initial Numerical supply, seed Retrieval, seed Decision;
2. `final_numerical_seed_context`: final accepted Numerical supply, seed Retrieval, seed Decision;
3. `final_numerical_retrieval_seed_decision`: final Numerical supply, final accepted Retrieval, seed Decision; and
4. `final_bundle`: final Numerical, Retrieval, and Decision.

Do not mutate or refit any release while constructing these states. Rebuild each Public registry using only Public histories and the sealed Numerical release. Use the same runtime/model fingerprints and cache semantics registered by evolution.

- [ ] **Step 6: Implement exact reporting**

For every state, report task count, coverage, capped/raw mean sMAE/sRMSE, joint error, median, P90/P95/max, clipping, catastrophic, invalid, fallback, selected-alternative counts, and family counts. For each adjacent pair and against Toto, report paired W/T/L and mean deltas. Include per-task forecast, selected candidate, supporting document IDs, final artifact hashes, and direct coordinate attribution.

The report metadata must contain:

```python
{
    "benchmark_role": "historically_consumed_final_regression",
    "selection_used_public": False,
    "public_result_may_feed_evolution": False,
    "primary_metrics": ["smae", "srmse"],
    "probabilistic_metric_reported": False,
}
```

No sCRPS is computed or included.

- [ ] **Step 7: Run focused tests**

Run: `/Users/yyoraa/time-series/.venv/bin/pytest -q tests/test_evaluate_frozen_package_bundle.py tests/test_package_metrics.py tests/test_run_package_coevolution.py`

Expected: all tests pass, incomplete or drifted evolution runs are rejected, and a valid invocation writes only to the separate Public output directory.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/evaluate_frozen_package_bundle.py tests/test_evaluate_frozen_package_bundle.py README.md
git commit -m "feat(evaluation): score frozen package bundle"
```

---

### Task 11: Close deterministic integration and repository-wide verification

**Files:**
- Modify: `tests/test_package_coordinate_e2e.py`

**Interfaces:**
- Consumes: every package-native component built in Tasks 1-10.
- Produces: one deterministic fake-LLM 80/20 replay and proof that no legacy co-evolution or Public path is invoked.

- [ ] **Step 1: Expand the deterministic e2e assertions**

```python
def test_package_coordinate_evolution_closes_two_cycle_80_20_loop(tmp_path, monkeypatch):
    result = _run_deterministic_two_cycle_fixture(tmp_path, monkeypatch)
    assert result.schedule.counts == (8, 32, 64, 16, 20)
    assert tuple(step.target for step in result.trace) == (
        "numerical",
        "retrieval",
        "decision",
        "numerical",
        "retrieval",
        "decision",
    )
    assert all(not step.public_test_accessed for step in result.trace)
    assert result.final_state.registry.task_ids == tuple(sorted((*train_ids, *dev_ids)))
    assert result.final_state.bundle.fingerprint() == result.replayed_bundle_sha256
    assert result.public_loader_calls == 0
    assert result.legacy_runner_calls == 0
    assert result.typed_round2_request_count > 0
    assert result.malformed_round2_preserved_round1 is True
```

- [ ] **Step 2: Run every focused suite**

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_package_numerical_supply.py \
  tests/test_package_metrics.py \
  tests/test_package_numerical_evolution.py \
  tests/test_package_candidate_proposal.py \
  tests/test_package_stage_runner.py \
  tests/test_package_coordinate_evolution.py \
  tests/test_package_artifacts.py \
  tests/test_run_package_coevolution.py \
  tests/test_evaluate_frozen_package_bundle.py \
  tests/test_package_retrieval_evolution.py \
  tests/test_package_decision_evolution.py \
  tests/test_package_coordinate_e2e.py \
  tests/test_specialist_atlas.py
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the full repository verification**

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q
/Users/yyoraa/time-series/.venv/bin/python -m compileall -q common drcik_agent evolving_loop numerical_agent
git diff --check
git status --short
```

Expected: the full suite, compile check, and whitespace check exit zero. `git status --short` lists only intentional files from this plan and no generated run data.

- [ ] **Step 4: Review against the approved spec**

Use `superpowers:requesting-code-review`. The review must explicitly inspect label firewalls, Public isolation, direct lineage, exact one-coordinate ownership, stage cardinalities, cache replay, `task_185` tail rejection, Decision's materialized-selection boundary, and the Atlas-as-candidate rule. A correctness finding reopens its owning Task 1-10, uses that task's focused test and commit boundary, and then reruns Steps 2-3 before Task 11 continues.

- [ ] **Step 5: Commit any integration-only test adjustments**

```bash
git add tests/test_package_coordinate_e2e.py
git commit -m "test(evolution): close package coevolution loop"
```

This commit contains only the deterministic e2e test adjustment; implementation fixes belong to the owning earlier task's commit.

---

### Task 12: Run the real smoke, formal 80/20 evolution, and one Public-99 regression

**Files:**
- Write generated smoke artifacts only: `runs/package_coevolution/smoke_nrd_8_2_2_20260903/`
- Write generated formal artifacts only: `runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903/`
- Write generated Public artifacts only: `runs/package_coevolution_public/gpt56sol_high_nrd_g2_public99_20260903/`
- Write seed Retrieval release only: `runs/retrieval_releases/package_nrd_20260903/`

**Interfaces:**
- Consumes: reviewed code from Tasks 1-11 and existing registered data/model/cache authorities.
- Produces: one real smoke result, one canonical final 80/20 bundle, and at most one Public-99 result from that sealed bundle.

- [ ] **Step 1: Verify immutable prerequisites**

```bash
TASK_WORKTREE=/Users/yyoraa/time-series/.worktrees/three-coordinate-package-coevolution
TASK_DATA_ROOT=/Users/yyoraa/time-series
cd "$TASK_WORKTREE"
test -f "$TASK_WORKTREE/splits/drcik_public_80_20_99_v1.json"
test -d "$TASK_DATA_ROOT/external/Dr-CiK/full-download/Dr-CiK_public/tasks"
test -f "$TASK_DATA_ROOT/runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2/champion_release.json"
test -d "$TASK_DATA_ROOT/runs/champion_forecasts/gpt56sol_high_80_20_99_20260902"
test -f "$TASK_DATA_ROOT/runs/method_evolution/v001/methods.py"
git -C "$TASK_DATA_ROOT/runs/method_evolution/v001" status --short
```

Expected: every path exists and the reviewed method repository is clean. The absolute `TASK_DATA_ROOT` paths are required because untracked run/cache data is not copied into a Git worktree. If the optional Atlas release is absent, omit `--atlas-release`; the runner builds and freezes it from Build-only rows. If Atlas fitting is rejected, the runner records that family as unavailable and proceeds with the other bounded families.

- [ ] **Step 2: Create the deterministic seed Retrieval release once**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/python -c 'from pathlib import Path; from evolving_loop.retrieval_agent.policy import RetrievalGenome, write_retrieval_release; release = write_retrieval_release(Path("/Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903"), RetrievalGenome.seed()); print(release.path)'
SEED_RELEASE=/Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903/v000
test -d "$SEED_RELEASE"
```

Expected: the printed path equals `SEED_RELEASE`. Do not rerun the command after proposals begin.

- [ ] **Step 3: Run the real 8/2/2 smoke with one Child per coordinate**

```bash
TIME_SERIES_PYTHON=/Users/yyoraa/time-series/.venv/bin/python scripts/run_package_coevolution.sh \
  --repo "$TASK_DATA_ROOT/runs/method_evolution/v001" \
  --split-file "$TASK_WORKTREE/splits/drcik_public_80_20_99_v1.json" \
  --tasks-file "$TASK_DATA_ROOT/external/Dr-CiK/full-download/Dr-CiK_public/tasks" \
  --numerical-champion-release "$TASK_DATA_ROOT/runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2/champion_release.json" \
  --forecast-store "$TASK_DATA_ROOT/runs/champion_forecasts/gpt56sol_high_80_20_99_20260902" \
  --retrieval-seed-release "$SEED_RELEASE" \
  --retrieval-skills "$SEED_RELEASE/skills.json" \
  --output-dir "$TASK_DATA_ROOT/runs/package_coevolution/smoke_nrd_8_2_2_20260903" \
  --authority-dir "$TASK_DATA_ROOT/runs/package_coevolution_authority/smoke_nrd_8_2_2_20260903" \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --cycles 1 \
  --children-per-coordinate 1 \
  --seed 20260903 \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir "$TASK_DATA_ROOT/outputs/model-cache" \
  --tsfm-workers-config "$TASK_DATA_ROOT/runs/method_evolution/local_tsfm_workers.json" \
  --acknowledged-model-licenses CC-BY-NC-4.0 \
  --smoke
```

Expected: `evaluation_complete.json.status` is `complete`; registered counts are Build 8, Calibration 2, Dev 2; no model cache miss, unauthorized write, cross-coordinate mutation, missing forecast, or Public access occurs.

- [ ] **Step 4: Audit smoke artifacts before the formal run**

```bash
/Users/yyoraa/time-series/.venv/bin/python -m json.tool "$TASK_DATA_ROOT/runs/package_coevolution/smoke_nrd_8_2_2_20260903/evaluation_complete.json"
/Users/yyoraa/time-series/.venv/bin/python -m json.tool "$TASK_DATA_ROOT/runs/package_coevolution/smoke_nrd_8_2_2_20260903/final_bundle.json"
wc -l "$TASK_DATA_ROOT/runs/package_coevolution/smoke_nrd_8_2_2_20260903/coordinate_trace.jsonl"
```

Expected: completion and bundle JSON parse successfully, the trace has at most three coordinate rows because smoke runs one cycle, and every Public marker is false.

- [ ] **Step 5: Run the registered two-cycle 80/20 co-evolution**

```bash
TIME_SERIES_PYTHON=/Users/yyoraa/time-series/.venv/bin/python scripts/run_package_coevolution.sh \
  --repo "$TASK_DATA_ROOT/runs/method_evolution/v001" \
  --split-file "$TASK_WORKTREE/splits/drcik_public_80_20_99_v1.json" \
  --tasks-file "$TASK_DATA_ROOT/external/Dr-CiK/full-download/Dr-CiK_public/tasks" \
  --numerical-champion-release "$TASK_DATA_ROOT/runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2/champion_release.json" \
  --forecast-store "$TASK_DATA_ROOT/runs/champion_forecasts/gpt56sol_high_80_20_99_20260902" \
  --retrieval-seed-release "$SEED_RELEASE" \
  --retrieval-skills "$SEED_RELEASE/skills.json" \
  --output-dir "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903" \
  --authority-dir "$TASK_DATA_ROOT/runs/package_coevolution_authority/gpt56sol_high_nrd_g2_80_20_20260903" \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --cycles 2 \
  --children-per-coordinate 3 \
  --seed 20260903 \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir "$TASK_DATA_ROOT/outputs/model-cache" \
  --tsfm-workers-config "$TASK_DATA_ROOT/runs/method_evolution/local_tsfm_workers.json" \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

Expected: the run finishes or can resume with the identical command plus `--resume`; it creates one canonical `final_bundle.json`, never opens Public, and records each attempted coordinate in N-R-D order. No rejected coordinate changes the accepted Parent bytes.

- [ ] **Step 6: Verify the formal final bundle before Public**

```bash
/Users/yyoraa/time-series/.venv/bin/python -m json.tool "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903/evaluation_complete.json"
/Users/yyoraa/time-series/.venv/bin/python -m json.tool "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903/final_bundle.json"
rg -n '"public_test_accessed"[[:space:]]*:[[:space:]]*true' "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903"
```

Expected: both JSON files parse; completion status is `complete`; the `rg` command returns no matches. If any check fails, do not run Public.

- [ ] **Step 7: Run the separate Public-99 evaluator once**

```bash
/Users/yyoraa/time-series/.venv/bin/python -u -m evolving_loop.evaluate_frozen_package_bundle \
  --evolution-dir "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903" \
  --final-bundle "$TASK_DATA_ROOT/runs/package_coevolution/gpt56sol_high_nrd_g2_80_20_20260903/final_bundle.json" \
  --split-file "$TASK_WORKTREE/splits/drcik_public_80_20_99_v1.json" \
  --tasks-file "$TASK_DATA_ROOT/external/Dr-CiK/full-download/Dr-CiK_public/tasks" \
  --repo "$TASK_DATA_ROOT/runs/method_evolution/v001" \
  --forecast-store "$TASK_DATA_ROOT/runs/champion_forecasts/gpt56sol_high_80_20_99_20260902" \
  --output-dir "$TASK_DATA_ROOT/runs/package_coevolution_public/gpt56sol_high_nrd_g2_public99_20260903" \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir "$TASK_DATA_ROOT/outputs/model-cache" \
  --tsfm-workers-config "$TASK_DATA_ROOT/runs/method_evolution/local_tsfm_workers.json" \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

Expected: exactly 99 Public tasks are scored, a completion marker and final report are written only in the Public output directory, and no evolution artifact changes.

- [ ] **Step 8: Report results without feeding them back**

Report Build-64, Calibration-16, Dev-20, and Public-99 capped/raw sMAE and sRMSE, joint error, W/T/L, P90/P95/max, clipping/catastrophic/invalid/fallback/coverage counts, accepted coordinate steps, selected alternative frequencies, and the four attribution systems. Explicitly identify any `task_185`-like tail loss and state that Public-99 was historically consumed and was not used for another mutation.

Do not tune, propose a new Child, or rerun a changed final bundle after seeing Public. Any follow-up experiment requires a new approved design, a new output directory, and no claim that Public is unseen.
