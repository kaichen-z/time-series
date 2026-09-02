# Task-Local Adaptive Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen Numerical v2 policy that learns candidate supply with five-fold group-aware cross-fitting on 80 Train tasks, chooses per-task anchor-heavy weights from history-only hindcasts, accepts once on 20 Dev tasks, and then runs one mandatory report-only Public-99 regression.

**Architecture:** Add a small runtime-only tournament module that cannot see labels, plus a separate Train/Dev evolution module that owns grouping, OOF fitting, conditional-uplift evidence, release construction, and acceptance. Keep the existing Champion/Toto release as the exact fallback, reuse `ForecastStore`, screening, and `CandidateDiagnostics`, and add explicit v2 evolution/Public CLIs instead of extending the already large Champion controller.

**Tech Stack:** Python 3.11+, frozen dataclasses, canonical JSON/SHA-256 artifacts, existing Statistical/TSFM/Combined `ForecastStore`, Dr-CiK capped/raw sMAE and sRMSE, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-task-local-adaptive-ensemble-design.md`

## Global Constraints

- Do not train, fine-tune, merge, or alter any TSFM parameters.
- Use only the 80-task Train partition to fit candidate supply or evolve structure.
- Build five deterministic outer folds from the transitive union of normalized entity, exact history fingerprint, and optional source-series identity; never read future values while grouping.
- Runtime selection sees only history, `TaskProfile`, already materialized forecasts, and history-only `CandidateDiagnostics`.
- Score and accept with capped and raw Dr-CiK sMAE and sRMSE; a gain in one metric cannot hide regression in the other.
- Retain at most eight candidates including the frozen anchor and at most two specialists.
- Search only a frozen nonnegative weight grid whose weights sum to one and whose anchor weight is at least `0.5`.
- Missing, invalid, unstable, or weak specialist evidence returns the anchor forecast exactly.
- The 20-task Dev partition is opened once after OOF fitting and cannot refit the policy.
- After Dev acceptance, Public-99 is mandatory exactly once as a report-only regression; no Public result API may return to fitting, proposal, or acceptance.
- Existing single/combined Champion behavior stays byte-compatible unless the explicit task-local v2 release is supplied.
- Only Critical correctness defects block this experiment; unrelated Important/Minor hardening is recorded and deferred.

---

## File and interface map

- Create `numerical_agent/evolution/task_local_ensemble.py`: label-free group key, morphology key, frozen policy/release/result schemas, canonical parsing/fingerprints, and deterministic local weight tournament.
- Create `numerical_agent/evolution/task_local_evolution.py`: group folds, candidate-supply fitting, OOF evaluation, conditional-uplift reports, Train/Dev Pareto gates, and frozen release assembly.
- Modify `numerical_agent/evolution/numerical_loop.py`: optional task-local release execution after normal candidate materialization and before package construction.
- Modify `numerical_agent/evolution/numerical_package.py`: sealed v2 package provenance and fingerprint validation.
- Create `numerical_agent/run_task_local_ensemble_evolution.py`: formal 80/20 runner using existing repository, screening, ForecastStore, TSFM runtimes, and frozen Champion anchor.
- Create `numerical_agent/evaluate_frozen_task_local_ensemble.py`: one-shot Public-99 evaluator and Toto paired regression report.
- Create `scripts/run_task_local_ensemble_evolution.sh`: reproducible runner and dry-run surface.
- Create `tests/test_task_local_ensemble.py`: grouping and runtime tournament contracts.
- Create `tests/test_task_local_evolution.py`: OOF fitting, conditional uplift, acceptance, and leakage contracts.
- Create `tests/test_task_local_ensemble_cli.py`: deterministic 8/2 lifecycle plus Public boundary.
- Modify `tests/test_numerical_champion_loop.py`: package replay and legacy compatibility.

---

### Task 1: Label-Free Group Folds and Morphology Keys

**Files:**
- Create: `numerical_agent/evolution/task_local_evolution.py`
- Test: `tests/test_task_local_evolution.py`

**Interfaces:**
- Produces: `GroupFoldManifest`
- Produces: `build_group_fold_manifest(tasks: Sequence[common.data.Task], *, seed: int, fold_count: int = 5, source_series_ids: Mapping[str, str] | None = None) -> GroupFoldManifest`
- Produces: `task_morphology_key(profile: TaskProfile) -> str`
- Consumes: `common.data.Task`, `screening.profile_task`, canonical JSON bytes, SHA-256.

- [ ] **Step 1: Write RED tests for transitive grouping, order independence, and future blindness**

```python
def test_group_fold_manifest_keeps_transitive_entity_history_groups_together():
    tasks = (
        task("a", entity="Store A", history=(1.0, 2.0)),
        task("b", entity="store a", history=(3.0, 4.0)),
        task("c", entity="Store C", history=(3.0, 4.0)),
        task("d", entity="Store D", history=(9.0, 9.0)),
        task("e", entity="Store E", history=(8.0, 8.0)),
    )
    manifest = build_group_fold_manifest(tasks, seed=17)
    folds = manifest.task_fold_map
    assert folds["a"] == folds["b"] == folds["c"]


def test_group_fold_manifest_is_order_independent_and_does_not_use_future():
    first = build_group_fold_manifest(TASKS, seed=17)
    reversed_and_relabelled = tuple(
        replace(row, future_values=(999.0,) * row.prediction_length)
        for row in reversed(TASKS)
    )
    second = build_group_fold_manifest(reversed_and_relabelled, seed=17)
    assert first.to_payload() == second.to_payload()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_task_local_evolution.py -k 'group_fold'`

Expected: FAIL with `ModuleNotFoundError` or missing `build_group_fold_manifest`.

- [ ] **Step 3: Implement exact group normalization and union-find construction**

```python
@dataclass(frozen=True)
class GroupFoldManifest:
    schema_version: int
    seed: int
    fold_count: int
    groups: tuple[tuple[str, tuple[str, ...], int], ...]
    grouping_fingerprint: str

    @property
    def task_fold_map(self) -> Mapping[str, int]:
        return MappingProxyType({
            task_id: fold
            for _group_sha, task_ids, fold in self.groups
            for task_id in task_ids
        })


def _history_sha(task: DataTask) -> str:
    return sha256(canonical_json_bytes({"history": list(task.history_values)})).hexdigest()
```

Use a disjoint-set union keyed only by NFKC/casefold entity, `_history_sha`, and nonempty `source_series_ids[task_id]`. Sort final components by their canonical component hash, then assign whole components to the least-populated fold with a deterministic morphology-stratum tie-break. Validate unique task IDs, finite nonempty histories, `fold_count >= 2`, and no component crossing folds.

- [ ] **Step 4: Add and pass morphology-key behavior tests**

```python
def test_morphology_key_changes_for_periodic_intermittent_and_regime_profiles():
    assert task_morphology_key(PERIODIC) != task_morphology_key(INTERMITTENT)
    assert task_morphology_key(PERIODIC) != task_morphology_key(REGIME_SHIFT)
```

The key must be canonical JSON over frequency, history-length bucket, horizon/history bucket, trend bucket, periodic/nonperiodic, intermittent/dense, recent-regime/stable, and signed/nonnegative; it must exclude `task_id`.

- [ ] **Step 5: Run Task 1 verification**

Run: `pytest -q tests/test_task_local_evolution.py -k 'group_fold or morphology_key'`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add numerical_agent/evolution/task_local_evolution.py tests/test_task_local_evolution.py
git commit -m "feat(numerical): add grouped cross-fit folds"
```

---

### Task 2: Deterministic Per-Task Tournament

**Files:**
- Create: `numerical_agent/evolution/task_local_ensemble.py`
- Test: `tests/test_task_local_ensemble.py`

**Interfaces:**
- Produces: `TaskLocalTournamentPolicy`
- Produces: `TaskLocalEnsembleResult`
- Produces: `execute_task_local_ensemble(policy: TaskLocalTournamentPolicy, *, candidate_names: Sequence[str], forecasts: Mapping[str, Sequence[float]], diagnostics: Mapping[str, CandidateDiagnostics], horizon: int) -> TaskLocalEnsembleResult`
- Produces: `task_local_fingerprint(value: object) -> str`
- Consumes: `CandidateDiagnostics.fold_forecasts`, `fold_truths`, capped/raw Dr-CiK metrics, and already materialized full-horizon forecasts.

- [ ] **Step 1: Write RED tests for exact fallback and legal weights**

```python
def test_local_tournament_returns_exact_anchor_when_specialist_is_invalid():
    result = execute_task_local_ensemble(
        POLICY,
        candidate_names=("toto_2_0", "seasonal_naive"),
        forecasts={"toto_2_0": (10.0, 11.0), "seasonal_naive": (float("nan"), 9.0)},
        diagnostics=DIAGNOSTICS,
        horizon=2,
    )
    assert result.forecast == (10.0, 11.0)
    assert result.selected_names == ("toto_2_0",)
    assert result.weights == (1.0,)


def test_local_tournament_weights_are_deterministic_normalized_and_anchor_heavy():
    first = execute_task_local_ensemble(POLICY, **VALID_INPUTS)
    second = execute_task_local_ensemble(POLICY, **dict(reversed(tuple(VALID_INPUTS.items()))))
    assert first == second
    assert sum(first.weights) == pytest.approx(1.0, abs=1e-12)
    assert first.weights[first.selected_names.index("toto_2_0")] >= 0.5
    assert len(first.selected_names) <= 3
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_ensemble.py -k 'fallback or weights'`

Expected: FAIL because the runtime module does not exist.

- [ ] **Step 3: Implement closed policy/result schemas and weight enumeration**

```python
@dataclass(frozen=True)
class TaskLocalTournamentPolicy:
    schema_version: int = 1
    anchor_name: str = "toto_2_0"
    maximum_candidates: int = 8
    maximum_specialists: int = 2
    minimum_successful_folds: int = 3
    minimum_anchor_weight: float = 0.5
    weight_step: float = 0.1
    minimum_joint_improvement: float = 0.02
    maximum_worst_joint_regret: float = 0.25
    maximum_raw_smae: float = 10.0
    maximum_raw_srmse: float = 10.0


@dataclass(frozen=True)
class TaskLocalEnsembleResult:
    forecast: tuple[float, ...]
    selected_names: tuple[str, ...]
    weights: tuple[float, ...]
    activated: bool
    fold_support: int
    fallback_reason: str | None
    policy_fingerprint: str
```

Enumerate weights as exact integer tenths, require anchor units `>= 5`, at most two nonzero specialist weights, and total units `== 10`. Convert to floats only after validation.

- [ ] **Step 4: Write RED tests for dual-metric Pareto and tail safety**

```python
def test_smae_only_gain_cannot_hide_srmse_regression():
    result = execute_task_local_ensemble(POLICY, **SMAE_ONLY_GAIN)
    assert result.activated is False
    assert result.fallback_reason == "no_pareto_safe_weight"


def test_worst_fold_regression_rejects_better_median_weight():
    result = execute_task_local_ensemble(POLICY, **TAIL_REGRESSION)
    assert result.forecast == TAIL_REGRESSION["forecasts"]["toto_2_0"]
```

- [ ] **Step 5: Implement paired-fold validation and deterministic ranking**

For every candidate, require the same number of fold forecasts/truths as the anchor, exact truth equality by fold, finite equal-horizon forecasts, `successful_folds >= minimum_successful_folds`, non-dominance on median sMAE/sRMSE, and raw/worst risk budgets. Score every legal weight using the paired fold truths. Rank by `(median_joint, worst_joint, median_smae, median_srmse, -anchor_weight, canonical_selected_names)`. Activate only if the chosen weight Pareto-improves the anchor medians and improves median joint by at least `minimum_joint_improvement`; otherwise return the original anchor tuple object values exactly.

- [ ] **Step 6: Run Task 2 verification**

Run: `pytest -q tests/test_task_local_ensemble.py`

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add numerical_agent/evolution/task_local_ensemble.py tests/test_task_local_ensemble.py
git commit -m "feat(numerical): add local anchor tournament"
```

---

### Task 3: OOF Candidate Supply, Conditional Uplift, and Frozen Release

**Files:**
- Modify: `numerical_agent/evolution/task_local_evolution.py`
- Test: `tests/test_task_local_evolution.py`

**Interfaces:**
- Produces: `GroupCandidateSupply`
- Produces: `ConditionalUpliftReport`
- Produces: `TaskLocalEnsembleRelease`
- Produces: `fit_group_candidate_supply(rows: Sequence[ChampionTaskRow], *, task_ids: Sequence[str], group_keys: Mapping[str, str], anchor_name: str, maximum_candidates: int = 8) -> tuple[GroupCandidateSupply, ...]`
- Produces: `evaluate_task_local_release(release: TaskLocalEnsembleRelease, rows: Sequence[ChampionTaskRow], *, task_ids: Sequence[str]) -> ConditionalUpliftReport`
- Produces: `fit_oof_release(rows: Sequence[ChampionTaskRow], manifest: GroupFoldManifest, *, anchor_release_sha256: str, anchor_name: str, source_hashes: tuple[tuple[str, str], ...], policy: TaskLocalTournamentPolicy) -> tuple[TaskLocalEnsembleRelease, ConditionalUpliftReport]`
- Consumes: Task 1 fold/morphology keys, Task 2 tournament, `ChampionTaskRow`, and canonical metric helpers.

- [ ] **Step 1: Write RED tests for held-out fitting and bounded supply**

```python
def test_oof_supply_for_each_task_is_fitted_without_its_group():
    release, report = fit_oof_release(ROWS, MANIFEST, **RELEASE_ARGS)
    assert report.task_count == 80
    assert report.oof_task_count == 80
    assert report.group_count == len(MANIFEST.groups)
    assert all(item.candidate_names[0] == "toto_2_0" for item in release.group_supplies)
    assert all(len(item.candidate_names) <= 8 for item in release.group_supplies)
    assert report.fit_leakage_count == 0
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_evolution.py -k 'oof_supply'`

Expected: FAIL with missing `fit_oof_release`.

- [ ] **Step 3: Implement Train-only candidate supply fitting**

Compute actual per-task capped/raw sMAE and sRMSE for each successful row. For each morphology group, rank candidates on the fit task IDs by coverage, mean joint score, P90 joint score, mean sMAE, mean sRMSE, family diversity, then name. Retain the anchor first and at most seven reviewed candidates. If group support is below four tasks, use the global fit-fold supply. Never use the held-out fold when constructing its supply.

- [ ] **Step 4: Write RED tests for conditional-uplift accounting**

```python
def test_conditional_uplift_excludes_exact_anchor_fallback_ties():
    report = conditional_uplift_report((ACTIVATED_WIN, FALLBACK_TIE, ACTIVATED_LOSS))
    assert report.activation_count == 2
    assert report.activated_wtl == WinTieLoss(wins=1, ties=0, losses=1)
    assert report.fallback_count == 1


def test_proposer_feedback_contains_no_task_identity_truth_or_forecast():
    payload = REPORT.to_proposer_payload()
    forbidden = {"task_id", "truth", "future", "forecast", "public", "dev"}
    assert not forbidden & recursive_keys(payload)
```

- [ ] **Step 5: Implement strict report and release schemas**

`ConditionalUpliftReport` must store task count, activation count/rate, fallback count, activated W/T/L, capped/raw mean/median/P90/P95 sMAE and sRMSE for Parent and Child, paired mean/median deltas, maximum task/fold regret, failures, clipped counts, group support, and OOF fingerprint. `to_proposer_payload()` emits only anonymous bounded group aggregates. `TaskLocalEnsembleRelease` stores schema version, exact anchor release SHA-256, tournament policy, default supply, sorted group supplies, grouping implementation SHA-256, OOF report SHA-256, source hashes, metric-policy fingerprint, and unique lineage. Parsing accepts the exact canonical schema only.

- [ ] **Step 6: Implement OOF and acceptance gates**

OOF acceptance requires exact anchor fallback coverage, both mean capped metrics non-regressing with at least one strict gain, no capped/raw P90/P95 regression beyond policy tolerance, no clipped/failure/coverage regression, bounded maximum task/fold regret, at least four activations from at least two groups, and activated win precision `wins / (wins + losses) >= 0.6`. A failed gate returns the exact Parent release bytes and a typed rejection report.

- [ ] **Step 7: Run Task 3 verification**

Run: `pytest -q tests/test_task_local_evolution.py tests/test_task_local_ensemble.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add numerical_agent/evolution/task_local_evolution.py tests/test_task_local_evolution.py
git commit -m "feat(numerical): fit local ensemble out of fold"
```

---

### Task 4: Numerical Package Runtime Integration

**Files:**
- Modify: `numerical_agent/evolution/numerical_loop.py`
- Modify: `numerical_agent/evolution/numerical_package.py`
- Modify: `tests/test_numerical_champion_loop.py`
- Test: `tests/test_task_local_ensemble.py`

**Interfaces:**
- Extends: `run_numerical_loop(..., task_local_release: TaskLocalEnsembleRelease | None = None) -> NumericalForecastPackage`
- Produces: sealed `_TaskLocalNumericalForecastPackage`
- Consumes: `execute_task_local_ensemble`, frozen group supply, materialized forecasts, history-only diagnostics, and existing protected Champion fallback.

- [ ] **Step 1: Write RED integration tests**

```python
def test_task_local_release_materializes_no_leaf_twice_and_replays_package_exactly():
    calls = Counter()
    package = run_numerical_loop(TASK, candidate_runner=counting_runner(calls),
                                 champion_release=ANCHOR, task_local_release=LOCAL)
    assert max(calls.values()) == 1
    assert package.final_forecast == replay_selection_forecast(
        package.selection_decision,
        {item.name: item.forecast for item in package.ranked_alternatives},
    )
    assert package.component_fingerprints["task_local_release"] == LOCAL_SHA


def test_legacy_champion_package_is_unchanged_without_task_local_release():
    assert package_bytes(run_numerical_loop(TASK, champion_release=ANCHOR, **ARGS)) == LEGACY_BYTES
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_numerical_champion_loop.py -k 'task_local or legacy_champion'`

Expected: FAIL because `run_numerical_loop` has no `task_local_release` argument.

- [ ] **Step 3: Add explicit v2 runtime path**

When `task_local_release` is present, require the exact anchor Champion fingerprint, include its candidate supply in the active namespace, materialize leaves/Combined candidates through the existing memo, diagnose them from history only, call `execute_task_local_ensemble`, and create a materialized Combined alternative when more than one forecast is selected. Convert the local result into a `SelectionDecision` with reason prefix `("task_local_ensemble", ...)`, exact selected names/weights, and no LLM call.

- [ ] **Step 4: Seal package provenance**

`_TaskLocalNumericalForecastPackage.__post_init__()` must require SHA-256 component fingerprints for `champion_release`, `task_local_release`, `group_supply`, and `task_local_policy`, and must require the task-local reason prefix. Existing package classes retain their current rules.

- [ ] **Step 5: Run Task 4 verification**

Run: `pytest -q tests/test_numerical_champion_loop.py tests/test_task_local_ensemble.py tests/test_numerical_package.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add numerical_agent/evolution/numerical_loop.py numerical_agent/evolution/numerical_package.py tests/test_numerical_champion_loop.py tests/test_task_local_ensemble.py
git commit -m "feat(numerical): package local ensemble forecasts"
```

---

### Task 5: Formal 80/20 Evolution CLI and Deterministic 8/2 Smoke

**Files:**
- Create: `numerical_agent/run_task_local_ensemble_evolution.py`
- Create: `scripts/run_task_local_ensemble_evolution.sh`
- Create: `tests/test_task_local_ensemble_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces CLI: `python -m numerical_agent.run_task_local_ensemble_evolution`
- Produces artifacts: `group_folds.json`, `oof_report.json`, `dev_report.json`, `task_local_release.json`, `run_manifest.json`, `evaluation_complete.json`
- Consumes: `--repo`, `--split-file`, `--tasks-file`, `--anchor-release-dir`, `--forecast-store`, `--output-dir`, TSFM runtime flags, and `--smoke`.

- [ ] **Step 1: Write RED CLI lifecycle tests**

```python
def test_smoke_runs_8_train_2_dev_and_freezes_only_after_dev_acceptance(tmp_path):
    completed = run_fake_cli(tmp_path, train=8, dev=2)
    assert completed["train_oof_tasks"] == 8
    assert completed["dev_tasks"] == 2
    assert completed["status"] == "accepted"
    assert (tmp_path / "out/task_local_release.json").is_file()


def test_failed_dev_preserves_anchor_and_publishes_no_v2_release(tmp_path):
    completed = run_fake_cli(tmp_path, dev_regression=True)
    assert completed["status"] == "dev_rejected"
    assert not (tmp_path / "out/task_local_release.json").exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_ensemble_cli.py -k 'smoke or dev'`

Expected: FAIL because the CLI module does not exist.

- [ ] **Step 3: Implement materialization and OOF lifecycle**

Reuse the source/inventory/runtime helpers from `run_champion_evolution.py`, but do not call `ChampionEvolutionController`. Load exactly 80/20 or 8/2, build the group manifest, materialize only reviewed screening candidates through `ForecastStore`, compute one history-only `CandidateDiagnostics` per candidate/task, run five-fold OOF fitting, fit the final release on all Train, then open Dev exactly once. Write canonical artifacts with `open("x")`; a rerun against completed output fails before loading tasks or models.

- [ ] **Step 4: Add release/run authority fingerprints**

Bind exact task-content hashes, split bytes, methods/skills/dictionary/policies bytes, anchor release bytes, ForecastStore/runtime identity, screening fingerprint, grouping implementation, tournament config, metric policy, and candidate supply into `run_manifest.json`. Revalidate before Dev and before publishing `task_local_release.json`.

- [ ] **Step 5: Add shell dry run and README command**

```bash
scripts/run_task_local_ensemble_evolution.sh --dry-run
python -m numerical_agent.run_task_local_ensemble_evolution \
  --repo runs/method_evolution/v001 \
  --split-file configs/drcik_80_20_99.json \
  --tasks-file "$DRCIK_TASKS" \
  --anchor-release-dir runs/champion_evolution/latest \
  --forecast-store runs/champion_forecasts \
  --output-dir runs/task_local_ensemble/v001
```

- [ ] **Step 6: Run Task 5 verification**

Run: `pytest -q tests/test_task_local_ensemble_cli.py tests/test_task_local_evolution.py tests/test_champion_evolution_cli.py`

Run: `bash -n scripts/run_task_local_ensemble_evolution.sh && scripts/run_task_local_ensemble_evolution.sh --dry-run`

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add numerical_agent/run_task_local_ensemble_evolution.py scripts/run_task_local_ensemble_evolution.sh tests/test_task_local_ensemble_cli.py README.md
git commit -m "feat(numerical): run local ensemble evolution"
```

---

### Task 6: Mandatory Frozen Public-99 Regression

**Files:**
- Create: `numerical_agent/evaluate_frozen_task_local_ensemble.py`
- Modify: `tests/test_task_local_ensemble_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces CLI: `python -m numerical_agent.evaluate_frozen_task_local_ensemble`
- Produces once: `public_forecasts.jsonl`, `public_toto_forecasts.jsonl`, `public_regression_report.json`, `evaluation_complete.json`
- Consumes: accepted canonical v2 release/run manifest, Public-99 IDs, tasks, exact source closure, ForecastStore/runtime identity, and Toto anchor.

- [ ] **Step 1: Write RED tests for one-shot Public evaluation and no feedback path**

```python
def test_public99_runs_only_after_acceptance_and_compares_exact_toto(tmp_path):
    result = run_fake_public_cli(tmp_path, public_count=99)
    assert result["public_task_count"] == 99
    assert result["comparison"]["baseline"] == "toto_2_0"
    assert {"mean_smae", "mean_srmse", "p90_smae_raw", "p95_srmse_raw", "wtl"} <= set(result["comparison"])


def test_public_report_cannot_be_loaded_as_train_evidence_or_overwritten(tmp_path):
    run_fake_public_cli(tmp_path, public_count=99)
    with pytest.raises(ValueError, match="already completed"):
        run_fake_public_cli(tmp_path, public_count=99)
    with pytest.raises(ValueError, match="report-only"):
        load_train_feedback(tmp_path / "public_regression_report.json")
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_ensemble_cli.py -k 'public99 or report'`

Expected: FAIL because the Public evaluator does not exist.

- [ ] **Step 3: Implement pre-Public frozen authority validation**

Before loading any Public task body, validate canonical accepted release bytes, Dev acceptance record, run manifest, split fingerprint, source closure, anchor release fingerprint, ForecastStore identity, runtime identity, metric policy, and absence of an existing completion marker.

- [ ] **Step 4: Implement paired Toto regression output**

For each Public task, run the frozen v2 package and the exact Toto anchor from the same immutable ForecastStore, then compute capped/raw per-task sMAE/sRMSE. Aggregate means, medians, P90/P95/max, clipped counts, coverage/failures, paired W/T/L, mean paired deltas, and maximum regret. Write complete forecast files first through temporary paths and publish `evaluation_complete.json` last. Do not expose a function that converts Public results into proposer evidence.

- [ ] **Step 5: Run Task 6 verification**

Run: `pytest -q tests/test_task_local_ensemble_cli.py tests/test_frozen_champion_evaluation.py tests/test_numerical_champion_loop.py`

Expected: PASS.

- [ ] **Step 6: Run repository verification**

Run: `python -m compileall -q common numerical_agent`

Run: `git diff --check`

Run: `pytest -q`

Expected: PASS with only the repository's already documented skip/warnings.

- [ ] **Step 7: Commit Task 6**

```bash
git add numerical_agent/evaluate_frozen_task_local_ensemble.py tests/test_task_local_ensemble_cli.py README.md
git commit -m "feat(numerical): evaluate frozen local ensemble"
```

---

## Experiment Sequence After Implementation

1. Run the deterministic 8/2 fake smoke and verify exact anchor fallback, one-shot Dev, and canonical artifacts.
2. Reuse the existing Statistical/TSFM/Combined forecast cache wherever the runtime/source fingerprints match.
3. Run the 80-task five-fold OOF evolution. Record wall time, activation count, group support, capped/raw sMAE and sRMSE, tails, failures, and Toto paired W/T/L.
4. Only if OOF passes, open the frozen 20-task Dev partition once. If it fails, keep Toto byte-for-byte and stop this version.
5. Only if Dev accepts, freeze the release and run Public-99 exactly once against Toto.
6. Report Public-99 results honestly even if v2 loses; do not tune or mutate this release from Public output.

## Self-Review Checklist

- Spec coverage: Tasks 1–6 cover grouping, runtime tournament, OOF evidence, package integration, Dev acceptance, and mandatory Public-99.
- Leakage: grouping and runtime modules accept no future/truth; only the evolution scorer sees labeled Train/Dev rows.
- Type consistency: `TaskLocalTournamentPolicy`, `TaskLocalEnsembleResult`, `TaskLocalEnsembleRelease`, `GroupFoldManifest`, and `ConditionalUpliftReport` are defined before their consumers.
- Legacy compatibility: the existing Champion path is untouched when `task_local_release is None`.
- Scope: no Retrieval/Decision changes, no TSFM training, no horizon-segment weighting, and no arbitrary numeric LLM output.
- Completion evidence: a successful implementation is not a claimed model gain until OOF, Dev, and mandatory Public-99 reports exist.
