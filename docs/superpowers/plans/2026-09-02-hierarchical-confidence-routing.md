# Hierarchical Confidence Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overly broad task-local activation rule with a Train-only, group-cross-fitted confidence gate that combines morphology priors and current-history evidence, optionally uses fixed early/late horizon regions, and preserves Toto exactly whenever evidence is insufficient.

**Architecture:** Add one focused immutable evidence module beside the existing task-local tournament. Outer Train folds build recipe-level priors without the held-out group; runtime combines the frozen prior with five-origin paired hindcasts and executes only host-owned anchor-heavy weights. Existing v1 releases remain readable, while v2 releases bind confidence evidence and optional regional arithmetic before the one-shot Dev and mandatory report-only Public-99 stages.

**Tech Stack:** Python 3.12, frozen dataclasses, canonical JSON/SHA-256, exact integer weight grids, deterministic Beta-Binomial arithmetic, median/MAD robust margins, existing `ForecastStore`, `CandidateDiagnostics`, Dr-CiK capped/raw sMAE and sRMSE, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hierarchical-confidence-routing-design.md`

## Global Constraints

- Do not train, fine-tune, merge, or alter TSFM parameters.
- Use only the 80-task Train partition to fit confidence evidence, candidate supply, or routing structure.
- Never let Dev or Public labels return to fitting, mutation, threshold selection, or acceptance of the same version.
- Keep Toto as the exact anchor and return its original forecast values whenever evidence is missing, invalid, contradictory, or unsafe.
- Continue to report and gate both capped and raw Dr-CiK sMAE and sRMSE; one metric cannot hide regression in the other.
- Retain at most eight supplied candidates, at most two specialists per recipe, nonnegative 0.1-grid weights summing to one, and Toto weight at least 0.5.
- Use fixed early/late halves only; do not learn a horizon boundary.
- Preserve schema-v1 release/runtime behavior byte-for-byte when confidence evidence is absent.
- Only Critical correctness defects block the experiment; unrelated hardening remains deferred.

---

## File and interface map

- Create `numerical_agent/evolution/task_local_confidence.py`: closed policy, recipe, evidence-record, and evidence-bank schemas; morphology hierarchy; Beta-Binomial and robust-margin calculations; strict canonical parser/fingerprint.
- Modify `numerical_agent/evolution/task_local_evolution.py`: cross-fitted evidence construction, fold-specific v2 releases, final 80-Train evidence reconstruction, v2 OOF evaluation.
- Modify `numerical_agent/evolution/task_local_ensemble.py`: confidence-qualified full/early/late recipe tournament and exact fallback.
- Modify `numerical_agent/evolution/numerical_selector.py`: replayable horizon-weighted arithmetic only.
- Modify `numerical_agent/evolution/numerical_loop.py`: provide TaskProfile/evidence to the tournament and package regional arithmetic.
- Modify `numerical_agent/evolution/numerical_package.py`: require v2 confidence/result fingerprints without changing v1 validation.
- Modify `numerical_agent/run_task_local_ensemble_evolution.py`: five-origin diagnostics, v2 artifacts, Train-only sensitivity report, unchanged Dev ordering.
- Modify `numerical_agent/evaluate_frozen_task_local_ensemble.py`: load v2 release and report regional activations on Public without returning results to evolution.
- Test `tests/test_task_local_confidence.py`: schema/math/hierarchy/leakage boundaries.
- Modify `tests/test_task_local_evolution.py`: OOF prior construction and acceptance.
- Modify `tests/test_task_local_ensemble.py`: confidence and regional tournament behavior.
- Modify `tests/test_numerical_champion_loop.py`: arithmetic/package replay and v1 compatibility.
- Modify `tests/test_task_local_ensemble_cli.py`: 8/2 lifecycle and Public isolation.

---

### Task 1: Closed confidence policy and evidence schemas

**Files:**
- Create: `numerical_agent/evolution/task_local_confidence.py`
- Create: `tests/test_task_local_confidence.py`

**Interfaces:**
- Produces: `ConfidencePolicy`
- Produces: `WeightRecipe`
- Produces: `ConfidenceEvidenceRecord`
- Produces: `HierarchicalEvidenceBank`
- Produces: `beta_win_probability(wins: int, losses: int) -> float`
- Produces: `robust_effect_margin(values: Sequence[float]) -> float`
- Produces: `exact_morphology_key(profile: TaskProfile) -> str`
- Produces: `coarse_morphology_key(profile: TaskProfile) -> str`
- Produces: `parse_hierarchical_evidence(payload: object) -> HierarchicalEvidenceBank`
- Consumes: `TaskProfile`, canonical JSON bytes, SHA-256.

- [ ] **Step 1: Write RED tests for exact schema and deterministic posterior arithmetic**

```python
def test_beta_win_probability_is_exact_deterministic_and_monotone() -> None:
    assert beta_win_probability(0, 0) == 0.5
    assert beta_win_probability(4, 0) == pytest.approx(0.96875)
    assert beta_win_probability(4, 1) > beta_win_probability(3, 1)
    assert beta_win_probability(3, 2) < beta_win_probability(4, 1)


def test_confidence_parser_rejects_unknown_fields_and_noncanonical_order() -> None:
    payload = VALID_BANK.to_payload()
    payload["records"][0]["mean_smae"] = 0.0
    with pytest.raises(ValueError, match="evidence record schema"):
        parse_hierarchical_evidence(payload)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_task_local_confidence.py -k 'beta or parser'`

Expected: FAIL because `task_local_confidence` does not exist.

- [ ] **Step 3: Implement closed host-owned policy and recipe identities**

```python
@dataclass(frozen=True)
class ConfidencePolicy:
    schema_version: int = 1
    exact_minimum_support: int = 8
    coarse_minimum_support: int = 12
    global_minimum_support: int = 20
    posterior_win_probability: float = 0.80
    minimum_paired_origins: int = 3
    robust_mad_multiplier: float = 1.0
    regional_minimum_horizon: int = 4


@dataclass(frozen=True)
class WeightRecipe:
    region: str
    names: tuple[str, ...]
    weight_units: tuple[int, ...]

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(unit / 10.0 for unit in self.weight_units)
```

Validate exact types, `region in {"full", "early", "late"}`, one anchor plus at most two specialists, unique NFKC/casefold names, positive integer units totaling ten, and first/anchor units at least five. The recipe fingerprint is canonical JSON over structural fields only.

- [ ] **Step 4: Implement evidence records and bank validation**

```python
@dataclass(frozen=True)
class ConfidenceEvidenceRecord:
    level: str
    group_key: str
    recipe: WeightRecipe
    independent_groups: int
    task_support: int
    wins: int
    ties: int
    losses: int
    posterior_win_probability: float
    robust_margin_smae: float
    robust_margin_srmse: float
    p90_regret_smae_raw: float
    p90_regret_srmse_raw: float
    failure_count: int
    clipped_smae_count: int
    clipped_srmse_count: int


@dataclass(frozen=True)
class HierarchicalEvidenceBank:
    schema_version: int
    policy: ConfidencePolicy
    records: tuple[ConfidenceEvidenceRecord, ...]
    fit_group_ids: tuple[str, ...]
    fit_group_fingerprint: str
    evidence_fingerprint: str
```

Require exact/coarse/global levels, sorted unique `(level, group_key, recipe_sha)` rows, count conservation, finite bounded values, recomputed posterior probability, and a fingerprint over every field except `evidence_fingerprint`.

- [ ] **Step 5: Implement dependency-free exact Beta posterior and robust margin**

For integer `wins` and `losses`, compute `P(p > 0.5 | Beta(wins+1, losses+1))` using the finite binomial identity:

```python
n = wins + losses + 1
probability = math.fsum(math.comb(n, k) for k in range(losses + 1, n + 1)) / (2**n)
```

This is the exact finite form for `P(p > 0.5)` under
`Beta(wins+1, losses+1)`. Implement
`median(values) - policy.robust_mad_multiplier * median(abs(x - median))`,
rejecting empty or nonfinite inputs.

- [ ] **Step 6: Add hierarchy and identity tests**

```python
def test_coarse_key_ignores_frequency_and_length_but_keeps_signal_regime() -> None:
    assert exact_morphology_key(PROFILE_A) != exact_morphology_key(PROFILE_B)
    assert coarse_morphology_key(PROFILE_A) == coarse_morphology_key(PROFILE_B)


def test_keys_are_task_identity_and_future_blind() -> None:
    assert exact_morphology_key(replace(PROFILE, task_id="other")) == exact_morphology_key(PROFILE)
```

- [ ] **Step 7: Run Task 1 verification**

Run: `pytest -q tests/test_task_local_confidence.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add numerical_agent/evolution/task_local_confidence.py tests/test_task_local_confidence.py
git commit -m "feat(numerical): add routing confidence evidence"
```

---

### Task 2: Cross-fitted hierarchical evidence construction

**Files:**
- Modify: `numerical_agent/evolution/task_local_evolution.py`
- Modify: `tests/test_task_local_evolution.py`

**Interfaces:**
- Produces: `build_hierarchical_evidence(rows: Sequence[TaskLocalTaskRow], *, task_ids: Sequence[str], group_ids: Mapping[str, str], anchor_name: str, candidate_names: Sequence[str], policy: ConfidencePolicy) -> HierarchicalEvidenceBank`
- Produces: `build_cross_fitted_evidence(rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest, *, anchor_name: str, supplies: Mapping[int, Sequence[str]], policy: ConfidencePolicy) -> Mapping[int, HierarchicalEvidenceBank]`
- Produces: `RecipeTaskScore(task_group_sha256: str, improvement_smae: float, improvement_srmse: float, regret_smae_raw: float, regret_srmse_raw: float)`
- Produces: `score_recipe_on_task(recipe: WeightRecipe, anchor: TaskLocalTaskRow, task_rows: Mapping[str, TaskLocalTaskRow], *, task_group_sha256: str) -> RecipeTaskScore | None`
- Consumes: Task 1 schemas, exact group manifest, trusted Train truths and forecasts.

- [ ] **Step 1: Write RED tests proving held-out groups never enter their prior**

```python
def test_cross_fitted_bank_excludes_every_held_out_connected_group() -> None:
    banks = build_cross_fitted_evidence(ROWS, MANIFEST, **ARGS)
    for fold, bank in banks.items():
        held_out = {
            group_sha
            for group_sha, _task_ids, group_fold in MANIFEST.groups
            if group_fold == fold
        }
        assert not held_out.intersection(bank.fit_group_ids)
```

Store `fit_group_ids` only as sorted opaque group SHA-256 values; never
serialize task IDs or group membership.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_evolution.py -k 'cross_fitted_bank'`

Expected: FAIL with missing builder.

- [ ] **Step 3: Implement trusted per-task recipe scoring**

For each legal recipe, materialize its full, early, or late slice from existing successful rows. Score Parent and Child with `drcik_point_metrics`, return paired capped/raw sMAE/sRMSE improvements where positive means Child is better, and reject missing/nonfinite/wrong-horizon rows. Do not mutate rows or invoke forecast runtimes.

- [ ] **Step 4: Implement exact/coarse/global aggregate records**

For every recipe and hierarchy bucket, aggregate only tasks in `task_ids`. Count wins by positive joint improvement with `1e-12` tolerance, retain robust joint and separate sMAE/sRMSE margins, count independent `group_ids`, and compute raw P90 regrets using `linear_quantile`. Emit a level only when its configured support and independent-group constraints pass.

- [ ] **Step 5: Add sparse hierarchy fallback and disagreement tests**

```python
def test_sparse_exact_bucket_falls_back_to_qualified_coarse_record() -> None:
    bank = build_hierarchical_evidence(SPARSE_EXACT_ROWS, **ARGS)
    record = bank.resolve(PROFILE, RECIPE)
    assert record is not None
    assert record.level == "coarse"


def test_no_qualified_level_returns_no_prior() -> None:
    assert build_hierarchical_evidence(TOO_SPARSE_ROWS, **ARGS).resolve(PROFILE, RECIPE) is None
```

- [ ] **Step 6: Wire fold-specific banks into OOF fitting**

Within `fit_oof_release`, fit candidate supply and the evidence bank from the four fit folds, execute the held-out fold with that bank, and discard the fold bank after appending OOF outcomes. After OOF evaluation, construct the deployable bank once from all 80 Train tasks. Bind the opaque fit-group fingerprint, not task memberships, into the final release.

- [ ] **Step 7: Run Task 2 verification**

Run: `pytest -q tests/test_task_local_evolution.py tests/test_task_local_confidence.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add numerical_agent/evolution/task_local_evolution.py tests/test_task_local_evolution.py
git commit -m "feat(numerical): cross-fit routing priors"
```

---

### Task 3: Confidence-qualified local and horizon-region tournament

**Files:**
- Modify: `numerical_agent/evolution/task_local_ensemble.py`
- Modify: `tests/test_task_local_ensemble.py`

**Interfaces:**
- Keeps unchanged: `execute_task_local_ensemble(...) -> TaskLocalEnsembleResult` for schema-v1 callers.
- Produces: `execute_confidence_task_local_ensemble(policy: TaskLocalTournamentPolicy, confidence_policy: ConfidencePolicy, *, candidate_names: Sequence[str], forecasts: Mapping[str, Sequence[float]], diagnostics: Mapping[str, CandidateDiagnostics], horizon: int, profile: TaskProfile, confidence_evidence: HierarchicalEvidenceBank) -> TaskLocalConfidenceResult`
- Produces: `TaskLocalConfidenceResult(forecast: tuple[float, ...], regions: tuple[TaskLocalRegionResult, ...], activated: bool, fallback_reason: str | None, policy_fingerprint: str, evidence_fingerprint: str)`
- Produces: `TaskLocalRegionResult(region: str, start: int, stop: int, selected_names: tuple[str, ...], weights: tuple[float, ...], activated: bool, posterior_win_probability: float, robust_margin_smae: float, robust_margin_srmse: float)`
- Consumes: Task 1 evidence bank and existing paired fold data.

- [ ] **Step 1: Write RED tests for prior/local agreement and exact fallback**

```python
def test_good_local_hindcast_without_qualified_group_prior_falls_back() -> None:
    result = execute_confidence_task_local_ensemble(
        POLICY_V2,
        CONFIDENCE_POLICY,
        profile=PROFILE,
        confidence_evidence=EMPTY_BANK,
        **LOCALLY_GOOD_INPUTS,
    )
    assert result.forecast is LOCALLY_GOOD_INPUTS["forecasts"]["toto_2_0"]
    assert result.fallback_reason == "confidence_prior_unavailable"


def test_good_prior_with_bad_local_hindcast_falls_back() -> None:
    result = execute_confidence_task_local_ensemble(
        POLICY_V2,
        CONFIDENCE_POLICY,
        profile=PROFILE,
        confidence_evidence=GOOD_BANK,
        **LOCALLY_BAD_INPUTS,
    )
    assert result.activated is False
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_ensemble.py -k 'prior or local_hindcast'`

Expected: FAIL because v1 executor has no confidence arguments.

- [ ] **Step 3: Implement local evidence calculation for every recipe**

Reuse exact paired folds. For each legal recipe, blend the same fold origins,
calculate paired improvements in both metrics, and require minimum origin
support, positive medians, robust joint and paired margins, bounded raw errors,
and existing worst-joint regret. A recipe is confidence-qualified only when
`bank.resolve(profile, recipe)` exists, posterior probability meets the policy
threshold, its robust joint margin is positive, neither individual prior
margin is below -0.05, and both current-task local margins are positive. Use
five hindcast origins where feasible and deterministically reduce to four or
three without fabricating folds.

- [ ] **Step 4: Implement fixed early/late region execution**

For horizons at least `regional_minimum_horizon`, split at `(horizon + 1) // 2`. Score early and late slices independently using the same fold origins and recipes. Activate a region only if it passes all confidence/Pareto/tail gates. Use exact Toto values for a rejected region. If neither region activates, attempt the full-horizon recipe; if it also fails, return the original anchor tuple unchanged.

- [ ] **Step 5: Add regional behavior tests**

```python
def test_early_specialist_and_late_anchor_are_concatenated_exactly() -> None:
    result = execute_confidence_task_local_ensemble(
        POLICY_V2, CONFIDENCE_POLICY, **EARLY_ONLY_INPUTS
    )
    assert result.forecast[:2] == EXPECTED_EARLY_BLEND
    assert result.forecast[2:] == ANCHOR[2:]
    assert tuple(region.activated for region in result.regions) == (True, False)


def test_short_horizon_uses_full_gate_without_fabricated_regions() -> None:
    result = execute_confidence_task_local_ensemble(
        POLICY_V2, CONFIDENCE_POLICY, **HORIZON_TWO_INPUTS
    )
    assert tuple(region.region for region in result.regions) == ("full",)
```

- [ ] **Step 6: Preserve exact v1 behavior**

Do not change `execute_task_local_ensemble` or `TaskLocalEnsembleResult`.
Schema-v2 callers use the new function and result type. Add a golden payload
assertion using the current v1 test fixture and exact result fingerprint.

- [ ] **Step 7: Run Task 3 verification**

Run: `pytest -q tests/test_task_local_ensemble.py tests/test_task_local_confidence.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add numerical_agent/evolution/task_local_ensemble.py tests/test_task_local_ensemble.py
git commit -m "feat(numerical): gate local recipes by confidence"
```

---

### Task 4: Replayable horizon arithmetic and package provenance

**Files:**
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Modify: `numerical_agent/evolution/numerical_loop.py`
- Modify: `numerical_agent/evolution/numerical_package.py`
- Modify: `tests/test_numerical_champion_loop.py`

**Interfaces:**
- Extends: `SelectionArithmetic` with exact `horizon_stops: tuple[int, ...]` and `region_weights: tuple[tuple[float, ...], ...]`
- Supports: `operation == "horizon_weighted"`
- Produces: bit-exact replay from the original leaf forecasts.
- Consumes: `TaskLocalConfidenceResult.regions` and v2 release/evidence fingerprints.

- [ ] **Step 1: Write RED replay tests before modifying arithmetic**

```python
def test_horizon_weighted_arithmetic_replays_early_and_late_weights() -> None:
    arithmetic = SelectionArithmetic(
        "horizon_weighted",
        inputs=(leaf("toto_2_0"), leaf("seasonal_naive")),
        horizon_stops=(2, 4),
        region_weights=((0.6, 0.4), (1.0, 0.0)),
    )
    decision = regional_decision(arithmetic)
    assert replay_selection_forecast(decision, FORECASTS) == EXPECTED
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_numerical_champion_loop.py -k 'horizon_weighted'`

Expected: FAIL because the arithmetic fields/operation are unsupported.

- [ ] **Step 3: Implement strict regional arithmetic**

Require at least one region, strictly increasing positive stops, final stop equal to every leaf forecast horizon at replay, a weight row for each region, row length equal to inputs, finite nonnegative weights summing exactly to one, and at least one positive weight per region. Reject unused residual/legacy fields. Replay each region from its leaf slices and concatenate them.

For `SelectionDecision.weights`, store the horizon-length-weighted average attribution for each selected leaf. `_arithmetic_attribution_weights()` must recompute the same aggregate from `horizon_stops` and `region_weights`.

- [ ] **Step 4: Package the v2 result and fingerprints**

Pass `profile` plus `task_local_release.confidence_evidence` to the executor. Build `SelectionArithmetic("horizon_weighted", ...)` whenever multiple region weight rows differ; otherwise retain existing `weighted`/`leaf` arithmetic. Add `task_local_confidence` and `task_local_result` SHA-256 fingerprints to `_TaskLocalNumericalForecastPackage` only for schema v2. Keep schema-v1 required fingerprint keys unchanged.

- [ ] **Step 5: Add leaf-once, replay, and compatibility tests**

```python
def test_v2_regional_package_replays_and_materializes_each_leaf_once() -> None:
    package = run_numerical_loop(TASK, task_local_release=V2_RELEASE, **ARGS)
    assert max(CALLS.values()) == 1
    assert replay_selection_forecast(package.selection_decision, ranked_map(package)) == package.final_forecast
    assert package.component_fingerprints["task_local_confidence"] == EXPECTED_BANK_SHA


def test_v1_package_bytes_are_unchanged_after_v2_support() -> None:
    assert package_bytes(run_numerical_loop(TASK, task_local_release=V1_RELEASE, **ARGS)) == V1_GOLDEN
```

- [ ] **Step 6: Run Task 4 verification**

Run: `pytest -q tests/test_numerical_champion_loop.py tests/test_task_local_ensemble.py`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add numerical_agent/evolution/numerical_selector.py numerical_agent/evolution/numerical_loop.py numerical_agent/evolution/numerical_package.py tests/test_numerical_champion_loop.py
git commit -m "feat(numerical): replay regional local weights"
```

---

### Task 5: V2 release, five-origin runner, and frozen Public boundary

**Files:**
- Modify: `numerical_agent/evolution/task_local_ensemble.py`
- Modify: `numerical_agent/evolution/task_local_evolution.py`
- Modify: `numerical_agent/run_task_local_ensemble_evolution.py`
- Modify: `numerical_agent/evaluate_frozen_task_local_ensemble.py`
- Modify: `tests/test_task_local_evolution.py`
- Modify: `tests/test_task_local_ensemble_cli.py`
- Modify: `README.md`

**Interfaces:**
- Extends: `TaskLocalEnsembleRelease` schema 2 with exact `confidence_evidence: HierarchicalEvidenceBank`
- Keeps: strict schema-1 parser for report/runtime compatibility.
- Formal evolution uses: `HindcastConfig(folds=5, min_successful_folds=3)`.
- Public evaluator remains one-shot and report-only.

- [ ] **Step 1: Write RED release round-trip and v1 compatibility tests**

```python
def test_v2_release_round_trips_exact_confidence_evidence() -> None:
    encoded = canonical_task_local_release_bytes(V2_RELEASE)
    decoded = parse_task_local_release(strict_json_loads(encoded))
    assert decoded == V2_RELEASE
    assert canonical_task_local_release_bytes(decoded) == encoded


def test_v1_release_remains_strictly_readable() -> None:
    assert parse_task_local_release(V1_PAYLOAD) == V1_RELEASE
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_task_local_evolution.py -k 'v2_release or v1_release'`

Expected: FAIL because the release accepts only schema one.

- [ ] **Step 3: Implement exact schema-dispatched parsing and release fingerprints**

Schema one must retain its exact current field set. Schema two adds only `confidence_evidence`; it requires policy/evidence types exactly, verifies the bank fingerprint, requires lineage ending in `task_local_confidence_v2`, and binds the evidence fingerprint into canonical release bytes.

- [ ] **Step 4: Switch formal materialization to five history origins**

In `_materialize_rows`, accept an exact `HindcastConfig` argument. Formal v2 passes `HindcastConfig(folds=5, min_successful_folds=3)` for Train, Dev, and Public; smoke fixtures synthesize five paired origins. Include the hindcast-config fingerprint in `run_manifest.json`, and require the same fingerprint before Public task bodies are loaded.

- [ ] **Step 5: Add CLI lifecycle tests for data authority order**

```python
def test_oof_rejection_never_loads_dev_or_public_bodies(tmp_path: Path) -> None:
    run_formal_v2(tmp_path, force_oof_rejection=True)
    assert scanner.loaded_partitions == ("train",)


def test_accepted_v2_opens_dev_once_and_public_only_in_frozen_cli(tmp_path: Path) -> None:
    run_formal_v2(tmp_path)
    assert evolution_scanner.loaded_partitions == ("train", "dev")
    run_public_once(tmp_path)
    assert public_scanner.loaded_partitions == ("public_test",)
    assert no_public_payload_reaches_evolution()
```

- [ ] **Step 6: Extend report surfaces without changing gates**

Add full/early/late activation counts and W/T/L, confidence-prior level counts, posterior ranges, and exact-fallback counts. Preserve the existing capped/raw aggregate fields and acceptance logic. Public report must compare v2 with Toto on all 99 tasks and clearly label it report-only.

- [ ] **Step 7: Update README commands and limitations**

Document the v2 80/20 command, frozen Public command, five TSFM runtime configuration, cache reuse, and the rule that an OOF or Dev rejection produces no Public run. State that confidence routing improves selection reliability but does not create new TSFM capabilities.

- [ ] **Step 8: Run Task 5 verification**

Run:

```bash
pytest -q \
  tests/test_task_local_confidence.py \
  tests/test_task_local_ensemble.py \
  tests/test_task_local_evolution.py \
  tests/test_task_local_ensemble_cli.py \
  tests/test_numerical_champion_loop.py
python -m compileall -q common numerical_agent
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 9: Commit Task 5**

```bash
git add README.md numerical_agent/evolution/task_local_ensemble.py numerical_agent/evolution/task_local_evolution.py numerical_agent/run_task_local_ensemble_evolution.py numerical_agent/evaluate_frozen_task_local_ensemble.py tests/test_task_local_evolution.py tests/test_task_local_ensemble_cli.py
git commit -m "feat(numerical): run confidence-routed evolution"
```

---

### Task 6: Train OOF, one-shot Dev, and mandatory Public-99 experiment

**Files:**
- Write generated artifacts only under: `runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902/`
- Write Public artifacts only under: `runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902_public99/`

**Interfaces:**
- Consumes: accepted Toto Champion, full reviewed Dictionary, five TSFM runtimes, existing ForecastStore, frozen split manifest.
- Produces: OOF report, optional accepted release, one-shot Dev report, and only after Dev acceptance the Public-99 paired report.

- [ ] **Step 1: Verify source/runtime prerequisites without opening Dev/Public bodies**

Run:

```bash
.venv/bin/python -m numerical_agent.run_task_local_ensemble_evolution --help
git -C runs/method_evolution/v001 status --short --branch
test -f splits/drcik_public_80_20_99_v1.json
test -f runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2/champion_release.json
```

Expected: runner imports, Dictionary repository is clean, and authority files exist.

- [ ] **Step 2: Run the formal 80/20 lifecycle with existing cache**

Run:

```bash
.venv/bin/python -u -m numerical_agent.run_task_local_ensemble_evolution \
  --repo runs/method_evolution/v001 \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --anchor-release-dir runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2 \
  --forecast-store runs/champion_forecasts/gpt56sol_high_80_20_99_20260902 \
  --output-dir runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902 \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir outputs/model-cache \
  --tsfm-workers-config runs/method_evolution/local_tsfm_workers.json \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

Expected: OOF report over exactly 80 tasks; Dev count is zero on OOF rejection or exactly 20 after OOF acceptance.

- [ ] **Step 3: Inspect the immutable completion status**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902/evaluation_complete.json")
print(json.loads(p.read_text()))
PY
```

If status is `oof_rejected` or `dev_rejected`, stop without Public evaluation and report the exact gate reasons. Do not tune from Dev.

- [ ] **Step 4: Run Public-99 exactly once only after acceptance**

Run:

```bash
.venv/bin/python -u -m numerical_agent.evaluate_frozen_task_local_ensemble \
  --repo runs/method_evolution/v001 \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --release-dir runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902 \
  --anchor-release-dir runs/champion_evolution/gpt56sol_high_80_20_99_20260902_g3_final2 \
  --forecast-store runs/champion_forecasts/gpt56sol_high_80_20_99_20260902 \
  --output-dir runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902_public99 \
  --tsfm-runtimes chronos,timesfm \
  --chronos-device-map cpu \
  --model-cache-dir outputs/model-cache \
  --tsfm-workers-config runs/method_evolution/local_tsfm_workers.json \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

Expected: exactly 99 Public records and one immutable `public_regression_complete` marker.

- [ ] **Step 5: Verify and report paired results**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("runs/task_local_ensemble/hierarchical_confidence_80_20_99_20260902_public99")
complete = json.loads((root / "evaluation_complete.json").read_text())
report = json.loads((root / "public_regression_report.json").read_text())
assert complete["public_task_count"] == 99
assert sum(1 for _ in (root / "public_forecasts.jsonl").open()) == 99
print(json.dumps(report["comparison"], indent=2, sort_keys=True))
PY
```

Report Toto and Child mean/median capped sMAE/sRMSE, raw P90/P95, clipped counts, failures, overall W/T/L, activated-only W/T/L, and activation counts by full/early/late region. Do not start another mutation from this report.

---

## Final verification checklist

- [ ] Schema-v1 releases and packages remain byte-compatible.
- [ ] Every v2 OOF task uses a prior fitted without its connected group.
- [ ] Runtime evidence is history-only and contains no task truth.
- [ ] Both group prior and local five-origin evidence are required.
- [ ] Early/late regions use a fixed midpoint and rejected regions equal Toto exactly.
- [ ] sMAE and sRMSE Pareto, raw-tail, clipping, failure, and regret gates remain strict.
- [ ] Dev is unopened after OOF rejection and never used to refit.
- [ ] Public-99 runs only after Dev acceptance and is report-only.
- [ ] Focused tests, compileall, and diff check pass before experiment claims.
