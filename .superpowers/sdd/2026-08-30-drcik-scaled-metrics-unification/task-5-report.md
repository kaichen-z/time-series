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

---

## Fix round 2

### Status and changes

Complete in the fix-round-2 changeset. Batch and targetwise acceptance now use only the canonical mean sMAE/sRMSE Pareto gate; median aggregates remain reporting evidence. Curation child ranking uses deterministic joint-pair ordering and final pair acceptance. Active ToolDictionary/config/checkpoint/cache/generic artifact/method-evaluation/generation artifacts require exact integer schemas and exact metric-policy binding; standalone materialized dictionaries and decision cases are schema-v2 envelopes. Hindcast cache corruption raises a distinct integrity exception through direct and Combined paths. Active Decision parsing requires every canonical field and rejects all legacy members; only explicit `allow_legacy=True` report readers normalize old payloads. Active `HindcastConfig` no longer contains `catastrophic_mase`. Strict JSON rejects duplicate keys at every nesting level and nonfinite constants. Filter, screening, selector, frozen, and rescore reports expose the complete scaled pair, raw tails, clipping, coverage, and conservation-safe W/T/L/Missing/Unscored counts; infinities use explicit typed status sentinels in JSON and reports.

### TDD RED evidence

1. Mean-pair gates and joint child ranking:

   `pytest -q tests/test_evolution_loop.py::test_batch_validation_rejects_median_only_improvement tests/test_targetwise_evolution.py::test_targetwise_gate_rejects_median_only_improvement tests/test_evolution_core_controller.py::test_engine_ranks_train_children_by_joint_scaled_pair_then_name`

   Output: `3 failed in 0.18s`.

2. Dictionary, persistence, strict schema, and strict JSON boundary:

   `pytest -q tests/test_numerical_dictionary_contracts.py::test_active_dictionary_rejects_legacy_unbound_or_defaulted_fields tests/test_numerical_dictionary_contracts.py::test_legacy_dictionary_reader_is_explicit_and_report_only tests/test_evolution_core_persistence.py::test_json_store_round_trips_checkpoint_and_artifact tests/test_evolution_core_persistence.py::test_json_store_appends_one_trace_object_per_line tests/test_evolution_core_persistence.py::test_active_checkpoint_rejects_noninteger_schema_aliases tests/test_evolution_core_persistence.py::test_generic_artifact_rejects_incomplete_active_envelope tests/test_frozen_two_stage_evaluation.py::test_active_release_rejects_noninteger_envelope_schema_aliases tests/test_common_payload.py`

   Output: `14 failed, 4 passed in 0.18s`.

3. Active CLI dictionary/config fail-closed boundary:

   `pytest -q tests/test_numerical_agent_cli.py::test_active_config_sections_require_exact_integer_schema tests/test_numerical_agent_cli.py::test_curate_rejects_unbound_dictionary_before_creating_output tests/test_numerical_agent_cli.py::test_frozen_rejects_unbound_dictionary_before_reading_public_data`

   Output: `5 failed in 0.13s`.

4. Generation/method-evaluation binding: focused curation CLI/transcript assertions failed before envelope binding (`2 failed in 0.45s`). Cache schema alias probes failed before exact-type validation (`3 failed, 3 passed`). Corrupt direct/Combined cache probes showed swallowed integrity faults (`4 failed, 1 passed`).

5. Strict Decision/Hindcast legacy split:

   `pytest -q tests/test_evolution_numerical_selector.py::test_active_policy_rejects_legacy_error_ranking_fields tests/test_evolution_numerical_selector.py::test_active_policy_parser_requires_explicit_legacy_migration_flag tests/test_evolution_numerical_selector.py::test_active_policy_payload_requires_every_canonical_field tests/test_evolution_numerical_selector.py::test_hindcast_config_has_no_active_catastrophic_mase_and_legacy_is_explicit tests/test_evolution_selector_evolution.py::test_task_conditioned_long_horizon_route_round_trips_and_legacy_defaults_are_safe tests/test_evolution_selector_evolution.py::test_policy_parser_accepts_legacy_source_without_soft_overlay_weight tests/test_evolution_selector_evolution.py::test_legacy_policy_source_defaults_to_flat_selection_without_assumptions tests/test_evolution_selector_evolution.py::test_pre_combined_policy_source_remains_backward_compatible`

   Output after correcting a test import: `7 failed, 1 passed in 0.25s`.

6. Exact report fields and paired-count conservation:

   `pytest -q tests/test_run_filter_evolution.py::test_filter_report_and_manifest_lead_with_bound_scaled_objective tests/test_run_filter_evolution.py::test_filter_paired_counts_conserve_tasks_with_both_missing_unscored tests/test_task_conditioned_screening_script.py::test_report_exposes_task_conditioning_and_family_coverage tests/test_task_conditioned_screening_script.py::test_screening_paired_counts_conserve_both_missing_tasks tests/test_numerical_selector_script.py::test_selector_report_leads_with_drcik_point_metrics tests/test_numerical_selector_script.py::test_selector_paired_counts_conserve_both_missing_tasks`

   Output: `6 failed in 0.21s`.

7. Remaining artifact/frozen/rescore probes: targetwise and policy generations plus materialized dictionaries/cases were initially unbound (`4 failed in 1.25s`); frozen W/T/L/M/U surfaces were incomplete (`3 failed in 0.27s`); rescore rendered infinity as `n/a` (`1 failed in 0.12s`).

### GREEN and compatibility evidence

- Mean-pair/ranking focused rerun: `3 passed in 0.11s`; batch/targetwise/controller compatibility: `53 passed in 6.53s`.
- Dictionary/persistence/strict-JSON rerun: `18 passed in 0.12s`; dictionary/config compatibility: `48 passed in 0.54s`; active CLI config/dictionary boundary: `6 passed in 0.07s`.
- Generation/method-evaluation binding: `2 passed in 0.44s`; exact cache schemas: `6 passed in 0.50s`; cache-integrity propagation including ordinary provider behavior: `7 passed in 0.47s`.
- Decision/Hindcast focused rerun: `8 passed in 0.11s`; full selector policy compatibility files: `121 passed in 0.47s`.
- Report/paired-count focused rerun: `6 passed in 0.11s`; complete filter/screening/selector script files: `66 passed in 1.43s`.
- Targetwise/policy/case/materialized-dictionary binding: `4 passed in 1.13s`; frozen W/T/L/M/U: `3 passed in 0.41s`; rescore sentinel/text surface: `3 passed in 0.10s`.
- Expanded focused suite first exposed one stale rescore W/T/L expectation (`1 failed, 614 passed in 14.56s`); after updating the schema expectation, final expanded verification passed: `615 passed in 15.36s`.
- Final post-self-review revalidation, including compile and diff checks: `615 passed in 14.30s`.
- `python -m compileall -q common numerical_agent`: passed. `git diff --check`: passed.

### Files and self-review

- Mean-pair lifecycle: `common/evolution_core/controller.py`, batch/targetwise/policy evolution and their tests.
- Schema/legacy boundary: `common/{payload,evolution_core/contracts,evolution_core/persistence}.py`, ToolDictionary, config/main/experiment, both caches, Decision/Hindcast parsers, frozen/rescore readers and tests.
- Release/report surfaces: filter, screening, selector, audit, frozen evaluation, rescore, targetwise generation artifacts, fixtures, and focused lifecycle tests.
- Confirmed active parsers never default missing schema/policy fields; bool/float schema aliases reject; legacy conversion is explicit and report-only; cache corruption cannot fall through Combined fallback; ordinary provider failures retain outcome classification.
- Confirmed mean-only Pareto acceptance, joint pair ordering, raw safety tails, complete W/T/L/M/U conservation, exact-Parent Dev rejection, no Public/Hidden mutation, and no active `catastrophic_mase` field.
- Confirmed no historical file under `runs/` was opened or rewritten, no Retrieval/Decision or Task 6 documentation/final-smoke implementation changed, and untracked `runs/numerical_morphology/` remains unstaged.

### Concerns

None.

---

## Fix round 3

### Status and changes

Complete in the fix-round-3 changeset. Active schema-v2 ToolDictionary reads now require the exact canonical nested MethodRecord/definition/candidate shape, exact field sets and scalar types, finite train summaries, explicit record state, and matching definition/candidate method IDs. Flat/coercive/defaulted dictionary rows remain available only through the explicitly named report-only legacy reader. Frozen paired comparisons now consume the exact expected task-ID universe, reject duplicate or unexpected identities, and conserve every task across win/tie/loss/missing/unscored. Historical point-forecast rescoring now uses the shared strict JSON decoder, exact row schemas and types, finite oracle values, and explicit `allow_legacy=True` report-only parsing.

### TDD RED evidence

1. Exact active dictionary and paired-universe boundary:

   `pytest -q tests/test_numerical_dictionary_contracts.py::test_active_dictionary_requires_exact_canonical_nested_records tests/test_numerical_dictionary_contracts.py::test_legacy_dictionary_reader_is_explicit_and_report_only tests/test_frozen_two_stage_evaluation.py::test_paired_counts_compare_the_joint_scaled_metric_pair tests/test_frozen_two_stage_evaluation.py::test_paired_counts_conserve_the_exact_expected_task_universe tests/test_frozen_two_stage_evaluation.py::test_paired_counts_reject_unexpected_task_identity tests/test_frozen_two_stage_evaluation.py::test_paired_counts_reject_duplicate_expected_or_observed_ids tests/test_rescore_point_forecasts.py::test_rescore_rejects_duplicate_keys_and_nonstandard_constants_before_scoring tests/test_rescore_point_forecasts.py::test_rescore_legacy_reader_requires_exact_finite_row_schema tests/test_rescore_point_forecasts.py::test_rescore_uses_cached_forecasts_without_probabilistic_metrics`

   Output: `22 failed, 5 passed in 0.22s`.

2. Explicit rescore legacy opt-in:

   `pytest -q tests/test_rescore_point_forecasts.py::test_legacy_forecast_reader_requires_explicit_report_only_opt_in`

   Output: `1 failed in 0.12s`.

### GREEN and compatibility evidence

- Exact dictionary/paired/strict-rescore focused rerun: `30 passed in 0.14s`.
- Complete rescore file after the explicit legacy-reader split: `14 passed in 0.11s`.
- Dictionary, frozen, rescore, curation, selector, and screening compatibility sweep: `185 passed in 2.87s`.
- Expanded focused lifecycle suite: `329 passed in 5.96s`; final fresh rerun after self-review: `329 passed in 5.60s`.
- `python -m compileall -q numerical_agent`: passed. `git diff --check`: passed.

### Files and self-review

- Active/legacy dictionary boundary: `numerical_agent/dictionary.py` and `tests/test_numerical_dictionary_contracts.py`.
- Exact paired task universe: `numerical_agent/evaluate_frozen_two_stage.py` and `tests/test_frozen_two_stage_evaluation.py`.
- Strict historical rescore reader: `numerical_agent/rescore_point_forecasts.py` and `tests/test_rescore_point_forecasts.py`.
- Confirmed canonical nested dictionary round trips still pass while flat rows, missing/unknown fields, string numerics, bool integers, and mismatched method IDs fail active parsing.
- Confirmed W/T/L/M/U sums to the expected task count across both-missing and one-sided-missing cases, with unexpected/duplicate identities rejected; rescore exercises the same counter.
- Confirmed duplicate keys at both JSON levels, NaN/Infinity constants, nonfinite oracle values, unknown fields, and type coercions reject before scoring; valid finite historical rows work only through explicit report-only opt-in.
- Confirmed the diff does not touch Retrieval/Decision, docs/final smoke, or historical `runs/`; untracked `runs/numerical_morphology/` remains untouched and unstaged.

### Concerns

None.
