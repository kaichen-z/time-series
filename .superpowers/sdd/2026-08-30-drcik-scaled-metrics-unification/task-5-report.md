# Task 5 Report: Lifecycle runners, release schemas, and legacy boundary

## Status

Complete on `feature/morphology-single-smoke`.

Commit: `36729fc feat(numerical): version scaled metric releases`

## Changes

- Added the single canonical schema-v2 capped sMAE/sRMSE metric-policy payload, its deterministic SHA-256 fingerprint, report-role metadata, and fail-closed active-release validation.
- Bound active evolution and curation configs, run manifests, selector hindcast cache rows and keys, generation results, screening/selector releases, morphology results, frozen evaluation rows/completion markers, and rescore reports to that policy.
- Rejected missing, forged, or legacy MASE/MAE/sMAPE/RMSSE active controls while retaining an explicit historical report-only forecast reader.
- Replaced the active selector global ranking with the joint capped sMAE/sRMSE ordering and a `(5, 5)` failure exposure; legacy successful outcomes without the pair reject.
- Made frozen and rescore reports lead with sMAE/sRMSE means, medians, standard errors, raw tails, clipping, coverage, and joint paired W/T/L; legacy metrics are declared diagnostic-only.
- Updated generated curation configs and fixtures from sMAPE to sMAE and required exact policy fields on active config loads.
- Preserved Train-only mutation evidence, exact Parent behavior on Dev rejection, and read-only Public/Hidden evaluation boundaries.

## TDD evidence

- Initial lifecycle RED: `15 failed, 181 passed`; failures were the intended old defaults, missing schemas, missing report metadata, and legacy ranking seams.
- Producer/type closure RED: four failures, followed by focused green.
- Config binding RED: missing and legacy active config loads plus generated sMAPE defaults failed (`3 failed`), then passed after the reader/default fix.
- MAE boundary RED: the canonical binding initially allowed an explicit `mae` control (`1 failed, 1 passed`); token-aware legacy detection closed it (`2 passed`).
- Cache mutation check: removing active cache validation made the legacy-row regression fail; restoring it passed.

## Verification

- Required Task 5 command: `220 passed in 4.39s`.
- Expanded final focused command: `236 passed in 5.32s`.
- Final full suite: `2182 passed, 1 skipped in 57.23s`.
- `python -m compileall -q ...`: passed.
- `git diff --check`: passed.

## Self-review

- Confirmed canonical producers and consumers use the same payload and fingerprint and active loads reject schema/policy drift.
- Confirmed remaining MASE/MAE/sMAPE/RMSSE fields in scoped production paths are diagnostics or the explicit historical rescore reader, not ranking or acceptance inputs.
- Confirmed reports expose both scaled metrics first and retain raw tail/clipping evidence.
- Confirmed no Retrieval/Decision implementation, historical `runs/` artifact, documentation, or final smoke contract was changed.

## Concerns

None. The pre-existing untracked `runs/numerical_morphology/` directory remains untouched and unstaged.

---

## Fix round 1

### Status and changes

Complete in the fix-round changeset. Closed every Critical/Important review seam: all active batch, targetwise, TSFM/Combined, selector, filtering, screening, and curation decisions now consume the capped sMAE/sRMSE pair with joint ordering and Pareto non-regression. The canonical policy authority is deeply immutable; checkpoints, both outcome caches, configs, seeds, hindcasts, releases, and reports validate or persist its exact payload and recomputed fingerprint. Existing malformed active cache rows fail closed. Legacy seeds and `catastrophic_mase` are active-path errors, with explicit opt-in report-only Decision parsing retained. Lifecycle reports now include both scaled aggregates, raw tails, clipping, coverage, paired joint W/T/L, and policy/report-role metadata; standards JSON represents raw infinities with typed sentinels.

### TDD RED evidence

1. Policy authority/checkpoints:

   `pytest -q tests/test_frozen_two_stage_evaluation.py::test_exported_metric_policy_cannot_mutate_canonical_authority tests/test_frozen_two_stage_evaluation.py::test_active_release_recomputes_and_rejects_a_forged_fingerprint tests/test_evolution_core_persistence.py::test_json_store_round_trips_checkpoint_and_artifact tests/test_evolution_core_persistence.py::test_json_store_rejects_missing_or_wrong_checkpoint_policy tests/test_evolution_core_persistence.py::test_json_store_checkpoint_persists_exact_canonical_fingerprint`

   Output: `5 failed, 1 passed in 0.16s`.

2. Pair-only evolution and curation:

   `pytest -q tests/test_evolution_loop.py::test_active_batch_prompts_ignore_legacy_metric_variation tests/test_evolution_loop.py::test_batch_validation_rejects_srmse_regression_despite_smae_improvement tests/test_targetwise_evolution.py::test_targetwise_gate_rejects_srmse_regression_and_ignores_legacy_gain tests/test_evolution_policy_targetwise.py::test_policy_gate_rejects_srmse_regression_despite_legacy_mase_gain tests/test_dictionary_curation_adapter.py::test_curation_config_replaces_unbounded_scalar_error_thresholds tests/test_dictionary_curation_adapter.py::test_curation_pair_gate_rejects_srmse_regression tests/test_dictionary_curation_adapter.py::test_curation_evaluator_produces_both_scaled_aggregates tests/test_dictionary_curation_adapter.py::test_scaled_pair_classification_preserves_all_status_distinctions`

   Output: `8 failed in 0.26s`.

3. Active config/cache/hindcast/JSON/Decision boundaries:

   `pytest -q tests/test_numerical_agent_cli.py::test_active_evolution_config_load_fails_closed_without_metric_policy tests/test_numerical_agent_cli.py::test_active_evolution_config_round_trips_exact_metric_policy tests/test_evolution_cache.py::test_corrupt_existing_cache_entry_fails_closed tests/test_evolution_cache.py::test_cache_record_copied_under_another_key_fails_closed tests/test_evolution_portfolio.py::test_policy_cache_existing_wrong_policy_binding_fails_closed tests/test_numerical_selector_script.py::test_forecast_store_identity_binds_skills_runtime_and_checkpoint tests/test_numerical_selector_script.py::test_forecast_store_existing_malformed_or_noncanonical_row_fails_closed tests/test_frozen_two_stage_evaluation.py::test_raw_infinite_tail_uses_an_explicit_standards_json_sentinel tests/test_evolution_selector_evolution.py::test_active_decision_payload_omits_catastrophic_mase_and_legacy_is_opt_in`

   Output: `11 failed in 1.04s`.

4. Seed CLI boundary:

   `pytest -q tests/test_task_conditioned_screening_script.py::test_screening_cli_has_train_dev_but_no_public_test_option`

   Output: `1 failed in 0.13s` (`seed_manifest` was absent; the parser still exposed the legacy seed policy).

### GREEN and compatibility evidence

- Policy/checkpoint focused rerun plus active defaults: `8 passed in 0.19s`.
- Pair-only evolution/curation focused rerun: `8 passed in 0.14s`.
- Config/cache/hindcast/JSON/Decision focused rerun: `11 passed in 0.18s`.
- First compatibility sweep exposed only stale expectations and one rejection-message key (`12 failed, 80 passed in 9.10s`); after exact-Parent fixture/message migration, the five isolated failures passed (`5 passed in 0.90s`).
- Compile plus selector/filter/screening focused rerun: `46 passed in 0.61s`.

### Final verification

- `python -m compileall -q common numerical_agent && pytest -q tests/test_run_morphology_smoke.py tests/test_frozen_two_stage_evaluation.py tests/test_evolving_cli.py tests/test_numerical_agent_cli.py tests/test_numerical_selector_script.py tests/test_task_conditioned_screening_script.py tests/test_run_filter_evolution.py tests/test_evolution_cache.py tests/test_evolution_portfolio.py tests/test_evolution_loop.py tests/test_targetwise_evolution.py tests/test_evolution_policy_targetwise.py tests/test_dictionary_curation_adapter.py tests/test_dictionary_curation_script.py tests/test_numerical_tsfm_integration.py tests/test_evolution_selector_evolution.py tests/test_evolution_numerical_selector.py tests/test_evolution_filtering.py tests/test_evolution_screening_evolution.py tests/test_evolution_core_persistence.py tests/test_prewarm_frozen_hindcasts.py`: `571 passed in 14.78s`.
- Exact Task 5 lifecycle command, including screening: `232 passed in 4.37s`.
- `git diff --check`: passed.
- Per fix-round instruction, the full suite was not rerun; the prior Task 5 full-suite result above remains the latest full-suite evidence.

### Files and self-review

- Authority/persistence: `common/evolution_core/{contracts,acceptance,persistence}.py`, `common/payload.py`, both outcome caches, experiment/config readers.
- Pair lifecycle: batch/targetwise/policy evolution, curation, filtering, screening, selector evaluation and their runners.
- Release boundaries: seed manifests, hindcast identity/rows, frozen curation/two-stage and audit outputs, shell defaults, fixtures, and focused regression tests.
- Verified no active MASE/sMAPE reference remains in the five reviewed evolution rank/gate/prompt files; legacy metrics remain diagnostics or explicit opt-in report readers only.
- Verified Dev remains read-only/exact Parent on rejection, Public/Hidden do not mutate, shared Statistical/TSFM/Combined kernels and Task 4 Train evidence remain intact.
- `runs/numerical_morphology/` remains untracked, untouched, and will not be staged. No Retrieval/Decision or Task 6 docs/final-smoke changes were made.

### Concerns

Active filter/screening launches now require a schema-v2 `seed_manifest.json` (or explicit `--seed-manifest`) with exact canonical policy/fingerprint and source hashes. This is intentional fail-closed behavior; old unversioned or `--seed-policy legacy` launches must be migrated explicitly, never normalized at runtime.
