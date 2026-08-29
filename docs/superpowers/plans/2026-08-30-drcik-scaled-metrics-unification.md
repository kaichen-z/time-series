# Dr-CiK Scaled-Metric Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make capped Dr-CiK sMAE and sRMSE the only performance metrics that can affect Numerical filtering, ranking, evolution, acceptance, and release selection.

**Architecture:** Extend the existing canonical Dr-CiK metric kernel into immutable outcome and hindcast records, then migrate each consumer in dependency order: execution/cache, filtering/screening, runtime selection, Combined/Morphology evolution, and runners/releases. Use a joint score only for deterministic ordering and use a separate Pareto gate for acceptance so neither scaled metric can hide regression in the other.

**Tech Stack:** Python 3.11+, frozen dataclasses, JSON/Python-literal artifacts, pytest, existing Numerical Agent execution and evolution modules.

**Spec:** `docs/superpowers/specs/2026-08-30-drcik-scaled-metrics-unification-design.md`

## Global Constraints

- Per-task `sMAE` and `sRMSE` use `mean(abs(truth))` as the common scale and are capped independently at `5.0` before cross-task aggregation.
- `joint_scaled_error = (sMAE + sRMSE) / 2` is permitted only for deterministic ordering.
- Parent/Child acceptance requires both metrics to be non-regressing and at least one to improve strictly within the configured tolerance.
- MASE, MAE, sMAPE, and RMSSE may remain diagnostic fields but cannot influence active filtering, ranking, mutation targeting, acceptance, or release selection.
- Runtime selection scores only history-derived validation windows; Dev is trusted/read-only; Public and hidden data never feed mutation or learning.
- Statistical, TSFM, and Combined candidates use the same metric kernel and schema.
- Legacy MASE/sMAPE artifacts are explicit read-only inputs and cannot become active new-format policies.

---

## File and interface map

- `common/metrics.py`: canonical scaled-score and Pareto helpers.
- `numerical_agent/evolution/execution.py`: task-level `Outcome` and aggregate `MethodReport` production.
- `numerical_agent/evolution/cache.py`: exact scaled-metric cache serialization and validation.
- `numerical_agent/evolution/portfolio.py`: TSFM/Combined outcomes produced through the same scorer.
- `numerical_agent/evolution/filtering.py`: dictionary selection and keep/specialized/repair decisions.
- `numerical_agent/evolution/screening.py`: active-dictionary scoring and oracle-retention gates.
- `numerical_agent/evolution/screening_evolution.py`: group evidence and Train/Dev screening evolution.
- `numerical_agent/evolution/numerical_selector.py`: hindcast diagnostics and runtime Safe Selector.
- `numerical_agent/evolution/selector_evolution.py`: Train/Dev Decision-policy acceptance.
- `numerical_agent/evolution/combined_evolution.py`: Train-only Combined proposal evidence.
- `numerical_agent/evolution/morphology_credit.py`: assumption marginal credit.
- `numerical_agent/evolution/morphology_consistency.py`: assumption safety checks.
- `numerical_agent/run_*.py`, `numerical_agent/evaluate_frozen_two_stage.py`: lifecycle, reports, and schema fingerprints.

---

### Task 1: Canonical scaled score, execution outcomes, and cache schema

**Files:**
- Modify: `common/metrics.py`
- Modify: `numerical_agent/evolution/execution.py`
- Modify: `numerical_agent/evolution/cache.py`
- Modify: `numerical_agent/evolution/portfolio.py`
- Test: `tests/test_evolving_agent_metrics.py`
- Test: `tests/test_evolution_execution.py`
- Test: `tests/test_evolution_cache.py`

**Interfaces:**
- Produces: `joint_scaled_error(smae: float, srmse: float) -> float`
- Produces: `pareto_scaled_improvement(parent_smae, parent_srmse, child_smae, child_srmse, *, tolerance) -> bool`
- Produces: `Outcome.smae`, `Outcome.srmse`, `Outcome.smae_raw`, `Outcome.srmse_raw`, `Outcome.smae_clipped`, and `Outcome.srmse_clipped`
- Produces: `MethodReport.mean_smae`, `MethodReport.mean_srmse`, and characteristic-level scaled summaries
- Consumes: existing `drcik_point_metrics()`

- [ ] **Step 1: Write failing canonical-metric and Pareto tests**

```python
def test_joint_scaled_error_and_pareto_gate_keep_metrics_separate():
    assert joint_scaled_error(1.0, 3.0) == 2.0
    assert pareto_scaled_improvement(1.0, 1.0, 0.9, 1.0, tolerance=1e-12)
    assert not pareto_scaled_improvement(1.0, 1.0, 0.5, 1.01, tolerance=1e-12)
```

- [ ] **Step 2: Run the focused metric test and verify RED**

Run: `pytest -q tests/test_evolving_agent_metrics.py`

Expected: FAIL because the joint/Pareto helpers are not defined.

- [ ] **Step 3: Implement strict helpers on top of `drcik_point_metrics()`**

```python
def joint_scaled_error(smae: float, srmse: float) -> float:
    values = (float(smae), float(srmse))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("scaled errors must be finite and nonnegative")
    return math.fsum(values) / 2.0


def pareto_scaled_improvement(
    parent_smae: float,
    parent_srmse: float,
    child_smae: float,
    child_srmse: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return (
        child_smae <= parent_smae + tolerance
        and child_srmse <= parent_srmse + tolerance
        and (
            child_smae < parent_smae - tolerance
            or child_srmse < parent_srmse - tolerance
        )
    )
```

- [ ] **Step 4: Add failing execution/cache round-trip tests**

```python
def test_successful_outcome_records_capped_and_raw_scaled_metrics(tmp_path):
    outcomes, reports = run_module(METHODS, [TASK], isolated=True)
    row = outcomes[0]
    assert row.smae is not None and row.srmse is not None
    assert row.smae_raw is not None and row.srmse_raw is not None
    assert reports[0].mean_smae == row.smae
    assert reports[0].mean_srmse == row.srmse


def test_cache_rejects_success_without_both_scaled_metrics(tmp_path):
    payload = valid_cached_success_payload()
    del payload["srmse"]
    with pytest.raises(CacheError, match="scaled metrics"):
        OutcomeCache.from_payload(payload)
```

- [ ] **Step 5: Populate every Statistical, TSFM, and Combined success from the canonical scorer**

```python
point = drcik_point_metrics(truth, forecast)
return Outcome(
    method=name,
    task_id=task.task_id,
    status=SUCCESS,
    smae=float(point["smae"]),
    srmse=float(point["srmse"]),
    smae_raw=float(point["smae_raw"]),
    srmse_raw=float(point["srmse_raw"]),
    smae_clipped=bool(point["smae_clipped"]),
    srmse_clipped=bool(point["srmse_clipped"]),
    forecast=tuple(forecast),
    # Existing values below are diagnostic only.
    mae=mae(truth, forecast),
    mase=mase(truth, forecast, history),
    smape=smape(truth, forecast),
)
```

- [ ] **Step 6: Bump cache schema and reject incomplete/new-policy legacy rows**

Bind the cache key and serialized record to the scaled-metric schema version and cap. Preserve an explicit legacy report reader, but require both scaled metrics when reconstructing active outcomes.

- [ ] **Step 7: Run Task 1 verification**

Run: `pytest -q tests/test_evolving_agent_metrics.py tests/test_evolution_execution.py tests/test_evolution_cache.py tests/test_evolution_portfolio.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add common/metrics.py numerical_agent/evolution/execution.py numerical_agent/evolution/cache.py numerical_agent/evolution/portfolio.py tests/test_evolving_agent_metrics.py tests/test_evolution_execution.py tests/test_evolution_cache.py tests/test_evolution_portfolio.py
git commit -m "feat(numerical): add canonical scaled outcomes"
```

---

### Task 2: Dictionary filtering and task-conditioned Screening

**Files:**
- Modify: `numerical_agent/evolution/filtering.py`
- Modify: `numerical_agent/evolution/diagnostics.py`
- Modify: `numerical_agent/evolution/screening.py`
- Modify: `numerical_agent/evolution/screening_evolution.py`
- Modify: `numerical_agent/run_filter_evolution.py`
- Modify: `numerical_agent/run_task_conditioned_screening.py`
- Test: `tests/test_evolution_filtering.py`
- Test: `tests/test_evolution_screening.py`
- Test: `tests/test_evolution_screening_evolution.py`
- Test: `tests/test_run_filter_evolution.py`
- Test: `tests/test_task_conditioned_screening_script.py`

**Interfaces:**
- Consumes: scaled fields from Task 1
- Produces: `FilterScore.mean_smae`, `mean_srmse`, `median_smae`, `median_srmse`
- Produces: screening oracle identity determined by `(joint_scaled_error, smae, srmse, name)`
- Produces: Train group evidence containing `delta_smae` and `delta_srmse`

- [ ] **Step 1: Write failing disagreement and no-MASE-influence tests**

```python
def test_filter_uses_joint_scaled_error_not_mase():
    rows = (
        success("a", smae=0.8, srmse=0.8, mase=100.0),
        success("b", smae=0.9, srmse=0.9, mase=0.01),
    )
    score = evaluate_filter(DICTIONARY, rows, TASKS, reference_outcomes=rows)
    assert score.selected[TASK_ID] == "a"


def test_screening_retains_all_pareto_equivalent_oracles():
    rows = (
        success("low_mae", smae=0.7, srmse=1.0),
        success("low_rmse", smae=1.0, srmse=0.7),
    )
    score = evaluate_screening(POLICY, rows, TASKS)
    assert set(score.oracle_names[TASK_ID]) == {"low_mae", "low_rmse"}
```

- [ ] **Step 2: Run the filtering/screening suites and verify RED**

Run: `pytest -q tests/test_evolution_filtering.py tests/test_evolution_screening.py tests/test_evolution_screening_evolution.py`

Expected: failures showing MASE-dependent selection/evidence.

- [ ] **Step 3: Replace MASE score vectors with scaled pairs**

Use `(joint_scaled_error(row.smae, row.srmse), row.smae, row.srmse, row.method)` for deterministic ordering. Penalties become `(5.0, 5.0)` plus explicit failure counters rather than synthetic MASE values.

- [ ] **Step 4: Migrate keep/specialized/repair evidence and prompts**

```python
evidence = {
    "mean_smae": mean(row.smae for row in successes),
    "mean_srmse": mean(row.srmse for row in successes),
    "delta_smae": mean(method_smae - baseline_smae),
    "delta_srmse": mean(method_srmse - baseline_srmse),
    "coverage": coverage,
    "failure_rate": failure_rate,
}
```

Remove prompt language that treats poor MASE as the performance criterion. Retain explicit warnings that specialist methods are judged only on applicable tasks and that crashes/invalid results are not NotApplicable.

- [ ] **Step 5: Apply Pareto gates to Train/Dev Screening acceptance**

Child acceptance must call the Task 1 helper and separately preserve oracle retention, candidate-pool constraints, dictionary diversity, coverage, and failure exposure.

- [ ] **Step 6: Update runner reports and schema fingerprints**

Lead tables with sMAE/sRMSE. Mark old metrics as `diagnostic_only`. Include metric schema/cap/objective in every filter/screening manifest hash.

- [ ] **Step 7: Run Task 2 verification**

Run: `pytest -q tests/test_evolution_filtering.py tests/test_evolution_screening.py tests/test_evolution_screening_evolution.py tests/test_run_filter_evolution.py tests/test_task_conditioned_screening_script.py`

Expected: PASS, including tests that vary only MASE/sMAPE without changing a decision.

- [ ] **Step 8: Commit Task 2**

```bash
git add numerical_agent/evolution/filtering.py numerical_agent/evolution/diagnostics.py numerical_agent/evolution/screening.py numerical_agent/evolution/screening_evolution.py numerical_agent/run_filter_evolution.py numerical_agent/run_task_conditioned_screening.py tests/test_evolution_filtering.py tests/test_evolution_screening.py tests/test_evolution_screening_evolution.py tests/test_run_filter_evolution.py tests/test_task_conditioned_screening_script.py
git commit -m "feat(numerical): scale filtering objectives"
```

---

### Task 3: Hindcast diagnostics and runtime Safe Selector

**Files:**
- Modify: `numerical_agent/evolution/numerical_selector.py`
- Modify: active policy parsing/defaults in `numerical_agent/evolution/numerical_selector.py`; do not overwrite historical frozen releases under `runs/`
- Test: `tests/test_evolution_numerical_selector.py`
- Test: `tests/test_numerical_selector_script.py`
- Test: `tests/test_numerical_morphology_loop.py`

**Interfaces:**
- Produces: scaled `HindcastFold` and `CandidateDiagnostics`
- Produces: `DecisionPolicy.ranking_order` restricted to scaled fields and non-error safety features
- Consumes: `joint_scaled_error()` and scaled Outcome schema

- [ ] **Step 1: Add failing hindcast/ranking/Safe-Anchor tests**

```python
def test_selector_ignores_mase_when_scaled_metrics_disagree():
    diagnostics = {
        "anchor": synthetic(smae=1.0, srmse=1.0, mase=0.01),
        "challenger": synthetic(smae=0.8, srmse=0.8, mase=100.0),
    }
    result = select_numerical_forecast(..., diagnostics=diagnostics)
    assert result.selected_candidates == ("challenger",)


def test_safe_anchor_blocks_one_metric_tail_regression():
    challenger = synthetic(smae=0.6, srmse=1.2, worst_srmse=2.0)
    result = select_protected_safe_anchor(...)
    assert result.selected_candidates == ("toto_2_0",)
```

- [ ] **Step 2: Run selector tests and verify RED**

Run: `pytest -q tests/test_evolution_numerical_selector.py tests/test_numerical_selector_script.py tests/test_numerical_morphology_loop.py`

Expected: failures on missing scaled hindcast fields and MASE-based ranking.

- [ ] **Step 3: Migrate fold and candidate records**

Add capped/raw sMAE/sRMSE and clipping flags to each fold. Replace `median_mase`, `recent_mase`, `worst_mase`, and `mase_mad` in active ranking with scaled counterparts and joint summaries. Preserve legacy deserialization only behind an explicit legacy flag.

- [ ] **Step 4: Replace active DecisionPolicy defaults and validation**

```python
ranking_order = (
    "median_joint_scaled_error",
    "recent_joint_scaled_error",
    "worst_joint_scaled_error",
    "median_smae",
    "median_srmse",
    "normalized_bias",
)
```

Replace `catastrophic_mase` with independent sMAE and sRMSE thresholds. Validate that active policies contain no MASE/sMAPE ranking fields.

- [ ] **Step 5: Apply separate baseline/tail/long-horizon guards**

A challenger may replace the Safe-Anchor only if neither scaled metric exceeds its configured regret on every required fold and the long-horizon audit. Ensemble acceptance uses the same rule after materializing its fold predictions.

- [ ] **Step 6: Run Task 3 verification**

Run: `pytest -q tests/test_evolution_numerical_selector.py tests/test_numerical_selector_script.py tests/test_numerical_morphology_loop.py`

Expected: PASS with deterministic ties and unchanged raw forecasts.

- [ ] **Step 7: Commit Task 3**

```bash
git add numerical_agent/evolution/numerical_selector.py tests/test_evolution_numerical_selector.py tests/test_numerical_selector_script.py tests/test_numerical_morphology_loop.py
git commit -m "feat(numerical): scale runtime selection"
```

---

### Task 4: Combined and Morphology Train-only evolution

**Files:**
- Modify: `numerical_agent/evolution/combined_evolution.py`
- Modify: `numerical_agent/evolution/morphology_credit.py`
- Modify: `numerical_agent/evolution/morphology_consistency.py`
- Modify: `numerical_agent/evolution/selector_evolution.py`
- Modify: `numerical_agent/evolution/numerical_loop.py`
- Test: `tests/test_evolution_combined_evolution.py`
- Test: `tests/test_evolution_combined_morphology.py`
- Test: `tests/test_evolution_morphology_consistency.py`
- Test: `tests/test_evolution_selector_evolution.py`
- Test: `tests/test_numerical_morphology_loop.py`

**Interfaces:**
- Consumes: scaled hindcast diagnostics from Task 3
- Produces: Train-only group evidence with winsorized delta-sMAE/delta-sRMSE
- Produces: dual-metric assumption credit and dual-metric Child gate

- [ ] **Step 1: Add failing single-metric-regression and assumption-bypass tests**

```python
def test_combined_child_is_rejected_when_srmse_regresses():
    parent = score(smae=1.0, srmse=1.0)
    child = score(smae=0.8, srmse=1.1)
    assert not accept_combined_child(parent, child)


def test_assumption_cannot_bypass_srmse_safe_anchor_guard():
    grounded = assumption_for("challenger")
    diagnostics = {"challenger": synthetic(smae=0.7, srmse=1.4)}
    assert check_assumption_consistency(...).accepted is False
```

- [ ] **Step 2: Run affected Combined/Morphology tests and verify RED**

Run: `pytest -q tests/test_evolution_combined_evolution.py tests/test_evolution_combined_morphology.py tests/test_evolution_morphology_consistency.py tests/test_evolution_selector_evolution.py tests/test_numerical_morphology_loop.py`

Expected: failures where one scaled metric is ignored or a MASE field remains required.

- [ ] **Step 3: Standardize Combined proposal diagnostics**

Keep the existing typed `winsorized_smae_delta` and `winsorized_srmse_delta` fields, but ensure every producer uses the canonical scorer and cap. Reject incomplete pairs. Remove remaining MASE/sMAPE performance fields from mutation prompts.

- [ ] **Step 4: Migrate assumption credit and consistency checks**

```python
credit = {
    "smae_improvement": baseline.smae - current.smae,
    "srmse_improvement": baseline.srmse - current.srmse,
    "joint_improvement": joint_scaled_error(baseline.smae, baseline.srmse)
        - joint_scaled_error(current.smae, current.srmse),
}
```

Require complete scaled folds, independent catastrophe checks, and Safe-Anchor regret protection for both metrics.

- [ ] **Step 5: Migrate Train screening and Dev acceptance**

Successive-halving may rank Children by joint error, but promotion and final acceptance call the Pareto helper and preserve coverage, clipping, and tail constraints. Dev remains read-only and rejection returns the exact Parent object.

- [ ] **Step 6: Run Task 4 verification**

Run: `pytest -q tests/test_evolution_combined_evolution.py tests/test_evolution_combined_morphology.py tests/test_evolution_morphology_consistency.py tests/test_evolution_selector_evolution.py tests/test_numerical_morphology_loop.py`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add numerical_agent/evolution/combined_evolution.py numerical_agent/evolution/morphology_credit.py numerical_agent/evolution/morphology_consistency.py numerical_agent/evolution/selector_evolution.py numerical_agent/evolution/numerical_loop.py tests/test_evolution_combined_evolution.py tests/test_evolution_combined_morphology.py tests/test_evolution_morphology_consistency.py tests/test_evolution_selector_evolution.py tests/test_numerical_morphology_loop.py
git commit -m "feat(numerical): unify evolution metrics"
```

---

### Task 5: Lifecycle runners, release schemas, and legacy boundary

**Files:**
- Modify: `numerical_agent/run_evolution.py`
- Modify: `numerical_agent/run_selector_evolution.py`
- Modify: `numerical_agent/run_morphology_smoke.py`
- Modify: `numerical_agent/run_task_conditioned_audit_experiment.py`
- Modify: `numerical_agent/evaluate_frozen_two_stage.py`
- Modify: `numerical_agent/rescore_point_forecasts.py`
- Modify: `numerical_agent/config.py`
- Modify: `common/evolution_core/contracts.py`
- Test: `tests/test_run_morphology_smoke.py`
- Test: `tests/test_frozen_two_stage_evaluation.py`
- Test: `tests/test_evolving_cli.py`
- Test: `tests/test_numerical_agent_cli.py`
- Test: `tests/test_numerical_selector_script.py`

**Interfaces:**
- Consumes: Tasks 1–4 active metric contract
- Produces: metric-policy fingerprint and versioned release/checkpoint schema
- Produces: reports led by sMAE/sRMSE with legacy values marked diagnostic-only

- [ ] **Step 1: Add failing active-policy and legacy-boundary tests**

```python
def test_active_release_rejects_legacy_mase_ranking(tmp_path):
    payload = legacy_release_payload(ranking_order=["median_mase"])
    with pytest.raises(ValueError, match="legacy metric policy"):
        load_active_release(payload)


def test_report_declares_old_metrics_diagnostic_only(tmp_path):
    report = run_report(...)
    assert report["primary_metrics"] == ["smae", "srmse"]
    assert set(report["diagnostic_only"]) >= {"mase", "mae", "smape", "rmsse"}
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run: `pytest -q tests/test_run_morphology_smoke.py tests/test_frozen_two_stage_evaluation.py tests/test_evolving_cli.py tests/test_numerical_agent_cli.py tests/test_numerical_selector_script.py`

Expected: failures on old defaults, schema, or report metadata.

- [ ] **Step 3: Version and fingerprint the metric contract**

Every active run manifest, checkpoint, cache, policy, and release binds this canonical payload:

```python
METRIC_POLICY = {
    "schema_version": 2,
    "primary": ("smae", "srmse"),
    "cap": 5.0,
    "ordering": "mean_pair",
    "acceptance": "pareto_non_regression",
}
```

Legacy rows may be rendered in historical reports but cannot seed evolution or runtime selection.

- [ ] **Step 4: Update CLI defaults and reports**

Remove active `smape`/`mase` metric defaults. Do not add a CLI mode that silently restores them. Report sMAE/sRMSE means, medians, standard errors, tails, clipping, coverage, and paired joint-objective wins/ties/losses first.

- [ ] **Step 5: Run Task 5 verification**

Run: `pytest -q tests/test_run_morphology_smoke.py tests/test_frozen_two_stage_evaluation.py tests/test_evolving_cli.py tests/test_numerical_agent_cli.py tests/test_numerical_selector_script.py tests/test_task_conditioned_screening_script.py`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add numerical_agent/run_evolution.py numerical_agent/run_selector_evolution.py numerical_agent/run_morphology_smoke.py numerical_agent/run_task_conditioned_audit_experiment.py numerical_agent/evaluate_frozen_two_stage.py numerical_agent/rescore_point_forecasts.py numerical_agent/config.py common/evolution_core/contracts.py tests/test_run_morphology_smoke.py tests/test_frozen_two_stage_evaluation.py tests/test_evolving_cli.py tests/test_numerical_agent_cli.py tests/test_numerical_selector_script.py
git commit -m "feat(numerical): version scaled metric releases"
```

---

### Task 6: Migration audit, deterministic fake smoke, and real one-task smoke

**Files:**
- Modify: `numerical_agent/README.md`
- Modify: `docs/SELF_EVOLUTION_FRAMEWORK.md`
- Modify: `docs/forecasting_pipeline_full_2026-08-26.html`
- Modify: `docs/forecasting_pipeline_full_2026-08-27_en.html`
- Test: `tests/test_run_morphology_smoke.py`
- Test: add or modify the repository's HTML semantic test file if present

**Interfaces:**
- Consumes: the complete active pipeline from Tasks 1–5
- Produces: one deterministic fake smoke artifact and one small real Numerical smoke artifact using the new metric fingerprint

- [ ] **Step 1: Add an active-path metric dependency audit test**

```python
def test_active_numerical_paths_do_not_rank_or_accept_with_legacy_metrics():
    forbidden = ("median_mase", "mean_mase", "primary=smape", '"smape" in ranking_order')
    for path in ACTIVE_NUMERICAL_POLICY_PATHS:
        text = path.read_text()
        assert not any(token in text for token in forbidden)
```

The audit allowlist may include explicit diagnostic rendering and legacy readers only.

- [ ] **Step 2: Run the audit and verify RED before documentation changes**

Run: `pytest -q tests/test_run_morphology_smoke.py -k scaled_metric_contract`

Expected: FAIL until every active dependency is migrated.

- [ ] **Step 3: Update English and Chinese documentation**

Describe one rule consistently: all performance-based stages use sMAE+sRMSE; MASE/MAE/sMAPE/RMSSE are diagnostic only. Mark prior 99-task results as legacy results produced by their historical objective rather than rewriting them.

- [ ] **Step 4: Run deterministic fake smoke**

Run the existing smoke CLI with its fake model/runtime fixtures on one task and assert:

```python
assert result["metric_policy"]["primary"] == ["smae", "srmse"]
assert result["selection"]["decision_metrics"] == ["smae", "srmse"]
assert result["evolution"]["dev_read_only"] is True
```

- [ ] **Step 5: Run one small real Numerical smoke**

Use `task_42`, existing locally attested TSFM workers, at most two hindcast folds, and no 80/20/Public evaluation. Record successful Statistical, TSFM, and Combined candidates, selected forecast, sMAE, and sRMSE. This smoke verifies execution only; it is not a benchmark claim.

- [ ] **Step 6: Run focused and full verification**

Run:

```bash
pytest -q \
  tests/test_evolving_agent_metrics.py \
  tests/test_evolution_execution.py \
  tests/test_evolution_cache.py \
  tests/test_evolution_filtering.py \
  tests/test_evolution_screening.py \
  tests/test_evolution_screening_evolution.py \
  tests/test_evolution_numerical_selector.py \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_combined_morphology.py \
  tests/test_evolution_morphology_consistency.py \
  tests/test_evolution_selector_evolution.py \
  tests/test_numerical_morphology_loop.py \
  tests/test_run_morphology_smoke.py \
  tests/test_frozen_two_stage_evaluation.py
pytest -q
python -m compileall -q common numerical_agent
git diff --check
```

Expected: all commands PASS; full suite retains only pre-existing documented skips/warnings.

- [ ] **Step 7: Commit Task 6**

```bash
git add numerical_agent/README.md docs/SELF_EVOLUTION_FRAMEWORK.md docs/forecasting_pipeline_full_2026-08-26.html docs/forecasting_pipeline_full_2026-08-27_en.html tests/test_run_morphology_smoke.py
git commit -m "docs(numerical): document scaled metric pipeline"
```

---

## Final completion criteria

- Active Numerical decisions never read MASE, MAE, sMAPE, or RMSSE.
- Every successful candidate has complete capped/raw sMAE and sRMSE fields.
- Statistical, TSFM, and Combined candidates share the same scoring function.
- Screening, selector, Combined, Morphology, and Child acceptance use the agreed joint-ordering plus Pareto-gate contract.
- Dev remains read-only and Public/hidden remain outside evolution.
- Legacy artifacts are explicitly historical/read-only.
- Deterministic fake smoke, small real smoke, focused tests, full tests, compileall, and diff checks pass.
