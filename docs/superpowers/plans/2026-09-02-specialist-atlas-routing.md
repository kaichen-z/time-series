# Specialist Atlas Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a cache-only, group-cross-fitted specialist router that selects complementary Dictionary candidates around the active Numerical Champion.

**Architecture:** A focused module converts existing `TaskLocalTaskRow` records into history-only routing features, mines a bounded complementary specialist pool from fitting folds, and uses a deterministic nearest-neighbor calibrator to filter the per-task supply before the existing task-local executor chooses safe weights. A focused CLI reuses the ForecastStore and 80/20 lifecycle, but stops after 80-task OOF unless every dual-metric gate passes.

**Tech Stack:** Python dataclasses, existing Dr-CiK sMAE/sRMSE metrics, `TaskProfile`, `CandidateDiagnostics`, `GroupFoldManifest`, `TaskLocalTournamentPolicy`, `ForecastStore`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-specialist-atlas-routing-design.md`

## Global Constraints

- Do not train, fine-tune, merge, or alter TSFM parameters.
- Fit Atlas evidence from the 80 Train tasks only with five group-aware outer folds.
- Never use held-out group labels, Dev, Public-99, task IDs, truth arrays, or forecasts as router inputs.
- Use capped and raw sMAE and sRMSE; no MASE or sMAPE may authorize selection.
- Keep the active Champion exact on missing, contradictory, unsafe, or insufficient evidence.
- Do not modify Retrieval or Decision.
- Only Critical correctness, leakage, or metric-authority defects block the experiment; defer unrelated Important/Minor hardening.

---

### Task 1: Cache-only store and Atlas feature contract

**Files:**
- Modify: `numerical_agent/evolution/forecast_store.py`
- Create: `numerical_agent/evolution/specialist_atlas.py`
- Create: `tests/test_specialist_atlas.py`
- Modify: `tests/test_evolution_forecast_store.py`

**Interfaces:**
- Consumes: `TaskLocalTaskRow`, `GroupFoldManifest`, `CandidateDiagnostics`, `TaskProfile`.
- Produces: `AtlasPolicy`, `AtlasFeature`, `AtlasTrainingRecord`, `atlas_feature`, and `select_specialist_pool`.

- [ ] **Step 1: Write failing cache-only and feature tests**

```python
def test_cache_only_store_never_executes_on_miss(tmp_path):
    store = make_store(tmp_path, cache_only=True)
    with pytest.raises(CacheMissError, match="cache-only"):
        store.forecast("naive_last", (1.0, 2.0), 1, "D")

def test_atlas_feature_is_history_only_and_task_id_free():
    left = atlas_feature(candidate_row(task_id="a"), anchor_row(task_id="a"))
    right = atlas_feature(candidate_row(task_id="b"), anchor_row(task_id="b"))
    assert left == right
    assert "task" not in repr(left).casefold()
```

Also cover missing diagnostics, mismatched fold truths, nonfinite features, exact categorical normalization, forecast-disagreement scaling, and sMAE/sRMSE-only diagnostics.

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/pytest -q tests/test_evolution_forecast_store.py tests/test_specialist_atlas.py -x
```

Expected: collection fails because `specialist_atlas` and cache-only support do not exist.

- [ ] **Step 3: Implement immutable contracts**

```python
class CacheMissError(RuntimeError):
    pass

class ForecastStore:
    # Add this keyword to the existing constructor and store it exactly.
    cache_only: bool

@dataclass(frozen=True)
class AtlasPolicy:
    schema_version: int = 1
    maximum_pool_size: int = 12
    maximum_task_candidates: int = 8
    neighbor_grid: tuple[int, ...] = (3, 5, 7, 9)
    minimum_independent_groups: int = 2
    minimum_win_probability: float = 0.60
    minimum_effect_margin: float = 0.0
    maximum_predicted_regret: float = 0.25

@dataclass(frozen=True)
class AtlasFeature:
    categorical: tuple[str, ...]
    numeric: tuple[float, ...]

@dataclass(frozen=True)
class AtlasTrainingRecord:
    candidate_name: str
    family: str
    group_sha256: str
    feature: AtlasFeature
    improvement_smae: float
    improvement_srmse: float
    regret_smae_raw: float
    regret_srmse_raw: float
```

`atlas_feature` uses only TaskProfile, paired CandidateDiagnostics, history scale, and candidate/Champion forecast disagreement. `AtlasTrainingRecord` future deltas are fitting targets and never enter `AtlasFeature`.

`select_specialist_pool` greedily minimizes fitting-fold future-aware oracle joint error while charging failures as capped 5.0. It requires support in at least two independent groups and keeps a candidate only when it adds positive complementary coverage. Output begins with the Champion and has at most 12 names.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add numerical_agent/evolution/forecast_store.py numerical_agent/evolution/specialist_atlas.py tests/test_evolution_forecast_store.py tests/test_specialist_atlas.py
git commit -m "feat(numerical): build specialist atlas"
```

---

### Task 2: Cross-fitted nearest-neighbor routing

**Files:**
- Modify: `numerical_agent/evolution/specialist_atlas.py`
- Modify: `tests/test_specialist_atlas.py`

**Interfaces:**
- Consumes: fitting-fold `AtlasTrainingRecord` values, one held-out task's rows, `TaskLocalTournamentPolicy`, and `GroupFoldManifest`.
- Produces: `AtlasCandidateEstimate`, `AtlasTaskResult`, `fit_atlas_oof`, and `evaluate_atlas_release`.

- [ ] **Step 1: Write failing group-isolation and routing tests**

```python
def test_oof_router_never_uses_held_out_or_connected_group():
    result = fit_atlas_oof(rows, manifest, atlas_policy, tournament_policy)
    assert result.fit_leakage_count == 0
    assert result.seen_fit_groups_by_task["held"] == ("other-group",)

def test_router_uses_prior_and_local_hindcast_then_calls_safe_executor():
    result = route_atlas_task(
        policy=atlas_policy,
        tournament_policy=tournament_policy,
        atlas=atlas,
        anchor=anchor,
        task_rows=task_rows,
    )
    assert result.selected_supply == ("toto_2_0", "seasonal_naive")
    assert result.ensemble.weights[0] >= 0.5
```

Also test order independence, robust scaling from fitting folds only, closed neighbor grid, sparse-support fallback, sMAE-only/sRMSE-only rejection, local/group disagreement fallback, raw-regret rejection, at most seven task specialists, exact Champion fallback, and no future-target field entering distance.

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/pytest -q tests/test_specialist_atlas.py -x
```

Expected: failures for missing estimator and OOF router APIs.

- [ ] **Step 3: Implement deterministic estimation and routing**

```python
@dataclass(frozen=True)
class AtlasCandidateEstimate:
    candidate_name: str
    independent_groups: int
    neighbor_count: int
    win_probability: float
    effect_smae: float
    effect_srmse: float
    lower_joint_effect: float
    predicted_regret: float

@dataclass(frozen=True)
class AtlasTaskResult:
    task_group_sha256: str
    selected_supply: tuple[str, ...]
    estimates: tuple[AtlasCandidateEstimate, ...]
    ensemble: TaskLocalEnsembleResult
```

Distance is the sum of frozen categorical mismatch penalties and robust-scaled numeric absolute differences. Select `k` from `(3, 5, 7, 9)` using fitting-fold inner cross-validation. Estimate win probability and median paired effects from unique nearest groups, shrink toward the global candidate prior, and compute a median-minus-MAD lower joint effect.

Qualify only dual-metric, support, probability, regret, and local-hindcast-consistent candidates. Pass the Champion plus at most seven qualified candidates into `execute_task_local_ensemble`; do not duplicate its weight search.

`fit_atlas_oof` rebuilds the pool, scaler, `k`, and evidence separately for every held-out outer fold of the 64 Build tasks. It returns a fitted Build release, exact per-task outcomes, and a `ConditionalUpliftReport`-compatible aggregate. `evaluate_atlas_release` applies that unchanged release to the 16 Calibration tasks; Calibration cannot alter the pool, scaler, `k`, or thresholds.

- [ ] **Step 4: Run focused and existing task-local tests**

```bash
.venv/bin/pytest -q tests/test_specialist_atlas.py tests/test_task_local_evolution.py tests/test_task_local_ensemble.py
```

Expected: all tests pass and existing releases remain byte-compatible.

- [ ] **Step 5: Commit Task 2**

```bash
git add numerical_agent/evolution/specialist_atlas.py tests/test_specialist_atlas.py
git commit -m "feat(numerical): route atlas specialists"
```

---

### Task 3: Cache-only 80 OOF runner and gated lifecycle

**Files:**
- Create: `numerical_agent/run_specialist_atlas_evolution.py`
- Create: `tests/test_specialist_atlas_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the same repo, split, tasks, anchor release, ForecastStore, TSFM runtime identity, and license arguments as `run_task_local_ensemble_evolution`.
- Produces: `run_manifest.json`, `group_folds.json`, `build_oof_report.json`, optional `calibration_report.json`, optional `atlas_release.json`, optional `dev_report.json`, and `evaluation_complete.json`.

- [ ] **Step 1: Write failing CLI lifecycle tests**

```python
def test_cache_only_oof_rejection_never_opens_dev(tmp_path):
    result = main(formal_args(tmp_path, cache_only=True))
    assert result == 0
    assert not (tmp_path / "dev_report.json").exists()
    assert read_complete(tmp_path)["status"] == "oof_rejected"
```

Also test acceptance opens Dev once, exact 80/20 counts, missing cache fail-closed before Dev, source/runtime/split fingerprints, no LLM construction, strict JSON, immutable output creation, and `--smoke` 8/2 behavior.

- [ ] **Step 2: Run CLI test to verify RED**

```bash
.venv/bin/pytest -q tests/test_specialist_atlas_cli.py -x
```

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement the runner using existing helpers**

The formal runner must:

1. validate a clean methods repo and exact 80/20 split;
2. construct the existing `ForecastStore` with `cache_only=True`;
3. materialize Train rows through existing `_materialize_rows` and partition the registered 80 Train IDs into the first 64 Build and final 16 Calibration IDs;
4. build the group manifest and run `fit_atlas_oof` on Build only;
5. atomically write Build OOF artifacts and stop before Calibration unless the Build gates pass;
6. evaluate the unchanged Build release on 16 Calibration tasks and stop unless the same safety gates plus at least 1% combined Train mean joint improvement pass;
7. reconstruct the frozen Atlas from all 80 Train tasks without changing the selected pool/grid, then load and evaluate 20 Dev exactly once without refitting; and
8. write a frozen release only after Dev acceptance.

Do not add a Public evaluator in this task. Use Public-99 only after an accepted Dev release exists.

- [ ] **Step 4: Run focused integration and static checks**

```bash
.venv/bin/pytest -q tests/test_specialist_atlas.py tests/test_specialist_atlas_cli.py tests/test_task_local_evolution.py tests/test_task_local_ensemble.py tests/test_evolution_forecast_store.py
python -m compileall -q common numerical_agent
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit Task 3**

```bash
git add README.md numerical_agent/run_specialist_atlas_evolution.py tests/test_specialist_atlas_cli.py
git commit -m "feat(numerical): run specialist atlas"
```

---

### Task 4: Run the cached experiment and report honestly

**Files:**
- Write generated artifacts only: `runs/specialist_atlas/cache_oof_80_20_99_20260902/`

**Interfaces:**
- Consumes: current frozen split, reviewed method repo, Champion release, and existing ForecastStore.
- Produces: one immutable 64-task Build OOF result, optional 16-task Calibration result, optional one-shot 20-task Dev result, and no Public result unless Dev accepts.

- [ ] **Step 1: Verify prerequisites without opening Dev bodies**

```bash
test -f splits/drcik_public_80_20_99_v1.json
test -f runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2/champion_release.json
test -d runs/champion_forecasts/gpt56sol_high_80_20_99_20260902
git -C runs/method_evolution/v001 status --short
```

Expected: authority inputs exist and the method repo is clean.

- [ ] **Step 2: Run cache-only 80/20 lifecycle**

```bash
.venv/bin/python -u -m numerical_agent.run_specialist_atlas_evolution \
  --repo runs/method_evolution/v001 \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --anchor-release-dir runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2 \
  --forecast-store runs/champion_forecasts/gpt56sol_high_80_20_99_20260902 \
  --output-dir runs/specialist_atlas/cache_oof_80_20_99_20260902 \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir outputs/model-cache \
  --tsfm-workers-config runs/method_evolution/local_tsfm_workers.json \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

Expected: no ForecastStore miss or model/LLM call; exactly 64 Build OOF outcomes; zero or 16 Calibration outcomes; and zero or exactly 20 Dev outcomes according to the gates.

- [ ] **Step 3: Compare against existing results**

Report Toto, task-local v1, hierarchical v3, and Atlas mean/median capped sMAE/sRMSE; raw/capped P90/P95; clipping/failures; overall and activated-only W/T/L; activation count/groups; maximum task/fold regret; and selected-specialist frequency. Label oracle results only as diagnostic ceilings.

- [ ] **Step 4: Run Public-99 only if Dev accepted**

If and only if `evaluation_complete.json` says `accepted`, adapt the frozen task-local evaluator to the Atlas release in a separate approved change, then run Public-99 exactly once. If OOF or Dev rejects, stop and report reasons without tuning from Dev or Public.
