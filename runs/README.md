# Run Artifact Inventory

Last audited: 2026-08-29

Repository: `time-series`

Primary split used by the Numerical Agent experiments: 80 Train / 20 Dev / 99 Public Regression tasks from Dr-CiK.

This directory contains both reproducible experiment artifacts and local execution caches. It is
not a flat collection of equally important results. This document identifies the producer for each
run family, whether that producer is still useful, what the run established, and what should be
kept.

## Status labels

| Label | Meaning |
|---|---|
| **Canonical** | A complete, reproducible result that is still used as a main comparison. Keep it. |
| **Reference** | A useful pilot, diagnostic, or runtime proof, but not the current final system. Keep the compact report/artifacts. |
| **Historical** | An intermediate ablation superseded by a later run. Keep only while its provenance is useful. |
| **Disposable** | Cache, transcript, log, virtual environment, or incomplete smoke output. Regenerate rather than commit it. |

The 99-task Public Regression split has already been used repeatedly for engineering comparisons.
It is a regression set, not a fresh sealed test set. The official hidden 80-task Dr-CiK labels are
not available locally, so local hidden inference artifacts do not contain an official score.

## What matters most

The current compact set worth retaining is:

1. `method_filtering/combined103_full_80_20_99_20260823/`: the frozen 103-candidate dictionary and
   its four-generation filtering report.
2. `task_conditioned_screening/formal_80_20_all103_20260823/`: the original frozen screening
   policy used by the baseline and Safe-Anchor evaluation chains.
3. `numerical_selector/formal_80_20_fallback_20260823/`: the original accepted selector.
4. `numerical_selector/smae_train_only_combined_v3_80_20_20260825/`: the Safe-Anchor Combined
   selector used in the best Public-99 sMAE result to date.
5. `frozen_two_stage/public_test_99_20260823/`: the original two-stage Public-99 baseline.
6. `frozen_two_stage/public_regression_99_safe_anchor_combined_20260825/`: the strongest completed
   guarded result by mean sMAE.
7. `task_conditioned_screening/conditional_v11_compiled_specialists_train_only_80_20_20260825/`,
   `numerical_selector/change_aware_guard_80_20_20260826/`, and
   `frozen_two_stage/change_aware_guard_public99_20260826/`: the later change-aware diagnostic
   chain. It is informative, but it did not beat the Toto reference overall.
8. `method_evolution/combined103_20_valid_20260822/`: the 20-task executable-method evolution
   pilot, including its accepted Python repairs.
9. `fresh30_four_method_20260815/`: the compact four-method 30-task historical comparison.
10. `tsfm_smoke/persistent_workers_20260826/`: proof that three persistent TSFM workers produced
    finite CPU forecasts with pinned checkpoints.

## Canonical experiment chains

All final rows below are frozen evaluations: `llm_calls = 0` and `mutation_calls = 0`.

| Chain | Screening run | Selector run | Public-99 run | Main result | Status |
|---|---|---|---|---|---|
| Original two-stage baseline | `formal_80_20_all103_20260823` | `formal_80_20_fallback_20260823` | `public_test_99_20260823` | Full system mean MASE `3.1846` versus Toto `2.8830`; 41 wins / 7 ties / 51 losses. It exposed selector tail risk. | **Canonical baseline** |
| Safe-Anchor Combined | `formal_80_20_all103_20260823` | `smae_train_only_combined_v3_80_20_20260825` | `public_regression_99_safe_anchor_combined_20260825` | Mean sMAE `0.4550` versus Toto `0.4607`; mean sRMSE `0.6839` versus Toto `0.6806`; 27 wins / 55 ties / 17 losses. It slightly improves sMAE, but not sRMSE. | **Canonical guarded reference** |
| Change-aware guard | `conditional_v11_compiled_specialists_train_only_80_20_20260825` | `change_aware_guard_80_20_20260826` | `change_aware_guard_public99_20260826` | Full system mean sMAE `0.4706` and sRMSE `0.7004`, both worse than Toto (`0.4607`, `0.6806`). | **Reference / negative result** |

The Safe-Anchor result is the most encouraging completed regression result, but it is not an
official hidden-test result and should not be described as final generalization evidence.

## Producer scripts

| Script or entry point | Produces | Still useful? | Notes |
|---|---|---|---|
| `scripts/bootstrap_method_evolution.sh` | A Git-tracked method repository such as `method_evolution/v001/` | **Yes** | Run once to seed `methods.py`, `skills.py`, and `policies.py`. Do not reseed an existing evolution repository. |
| `scripts/run_method_evolution.sh` / `python -m numerical_agent.run_evolution` | `method_evolution/*` and, with the filtering strategy, `method_filtering/*` | **Yes** | Current executable Python/TSFM/Combined evolution entry point. Requires a task path, split manifest, model backend, and any requested TSFM worker environments. |
| `scripts/run_task_conditioned_screening.sh` / `python -m numerical_agent.run_task_conditioned_screening` | `task_conditioned_screening/*` | **Yes** | Learns history-only task-conditioned active dictionaries. Defaults are a starting configuration, not a guarantee that an old run will be reproduced under a different local environment. |
| `scripts/run_numerical_selector_evolution.sh` / `python -m numerical_agent.run_selector_evolution` | `numerical_selector/*` | **Yes** | Evolves the history-only numerical Decision policy on Train and accepts on read-only Dev. Use a new output directory for a new experiment. |
| `scripts/evaluate_frozen_two_stage.sh` / `python -m numerical_agent.evaluate_frozen_two_stage` | `frozen_two_stage/*` | **Yes** | Frozen, write-free evaluation. The shell defaults reproduce the original baseline chain; override `SCREENING_DIR`, `SELECTOR_DIR`, and `OUTPUT_DIR` for another frozen policy. |
| `scripts/run_tsfm_checkpoint_smoke.sh` / `python -m numerical_agent.tsfm.smoke` | `tsfm_smoke/*` | **Yes** | Verifies one manifest/checkpoint/environment and finite output. It is not an accuracy benchmark. |
| `scripts/run_fresh30_four_method_eval.sh` | `fresh30_four_method_20260815/*` | **Reference only** | Reproduces the older 30-task comparison if all old policies/caches are available. It is not part of the current 103-candidate pipeline. |
| `scripts/run_llm_only_evolutions.sh`, `scripts/run_co_evolution.py`, `scripts/run_coevolution_pilot30.sh` | `evolving/*` and older co-evolution runs | **Legacy/reference** | Earlier Prompt/Genome/Source harness experiments. Useful for architecture comparison, not the current Numerical dictionary/screening/selector chain. |
| `scripts/run_dictionary_curation.sh`, `scripts/run_dictionary_frozen_test.sh` | Older JSON `ToolDictionary` curation/evaluation outputs | **Legacy/reference** | The current executable statistical methods live in a Git-tracked Python module. These scripts remain relevant only to the older JSON dictionary path. |
| `runs/posthoc_ablation/screening_topk_safety_analysis.py` | `posthoc_ablation/no_toto_public99_20260823/*` | **Diagnostic only** | Post-hoc rescore using cached outcomes. It cannot create a new sealed-test result. |
| `numerical_agent/plot_forecasts.py` | Local forecast plots below a method-evolution repository | **Yes, diagnostic** | Visual debugging only; plots do not decide acceptance. |

## Run-family inventory

### `method_evolution/`

Producer: `scripts/bootstrap_method_evolution.sh`, then `scripts/run_method_evolution.sh`.

| Run | Result | Status / recommendation |
|---|---|---|
| `v001/` | Seed Git repository containing the reusable executable method, skill, and policy modules. | **Reference input.** Keep the small source/history; caches and its local virtual environment are disposable. |
| `combined103_20_valid_20260822/` | One generation on 16 Train + 4 mini-Dev tasks over 93 Python + 5 TSFM + 5 Combined candidates. Accepted repairs to `scinet` and `itransformer`; no TSFM/Combined policy mutation passed Dev. Runtime was about 34.6 minutes. | **Reference.** Best documented method-evolution pilot; see `docs/COMBINED103_20_TASK_PILOT.md`. |
| `combined103_20_20260822/` | Earlier incomplete/invalid attempt preceding the validated pilot. | **Historical.** Prefer `combined103_20_valid_20260822/`. |
| `gpt54_probe_20260822/` | Small GPT-5.4 implementation probe. | **Historical.** Keep only if model-behavior provenance is needed. |
| `gpt56_identity_20_20260822/` | Tested identity-preserving repair validation. | **Historical engineering run.** |
| `gpt56_judge_halving_20_20260822/` | Tested Judge-assisted successive halving. | **Historical engineering run.** |
| `gpt56_targetwise_20_20260822/` | Tested target-wise method mutation. | **Historical engineering run.** |
| `gpt56_two_stage_20_20260822/` | Parent/Child two-stage comparison on 20 tasks. | **Historical engineering run.** |
| `flagship5_single_smoke_cache/` | Cached single-task results for five flagship TSFMs. | **Disposable cache.** Do not treat it as a benchmark report. |
| `.venv/` | Local environment, roughly the largest non-result item in this family. | **Disposable and local-only.** Recreate from dependencies; never commit. |

### `method_filtering/`

Producer: the filtering mode of the method-evolution pipeline. These directories are intentionally
local/ignored because they include large outcome caches; the frozen Python dictionary and compact
report are the valuable parts.

| Run | Result | Status / recommendation |
|---|---|---|
| `combined103_10_smoke_20260823/` | Ten-task development smoke for the Filter Agent and keep/specialize/repair/quarantine states. | **Historical smoke.** |
| `combined103_full_80_20_99_20260823/` | Four filter generations over 103 candidates. Three accepted generations raised selectable success from `45.98%` to `91.51%` and reduced Crash/Invalid exposure from `46.15%` to `7.73%`; forecast metrics stayed unchanged because both paths selected Toto. | **Canonical dictionary input.** Keep `frozen_dictionary.py`, reports, target batches, bootstrap/provenance, and compact result JSON. Caches/logs/transcripts are disposable. |

### `task_conditioned_screening/`

Producer: `scripts/run_task_conditioned_screening.sh`. All formal runs use 80 Train / 20 Dev;
`smoke_8_2_20260823` uses 8 / 2. The table reports Dev active-candidate success, mean active
candidate count, and mean pairwise Jaccard where present.

| Run | Dev result | Status |
|---|---|---|
| `smoke_8_2_20260823` | success `92.93%`, mean active `49.5` | **Disposable smoke** |
| `formal_80_20_20260823` | success `92.73%`, mean active `41.25` | **Historical**; smaller original candidate set |
| `formal_80_20_all103_20260823` | success `85.10%`, mean active `56.05` | **Canonical** for the original and Safe-Anchor chains |
| `conditional_v2_80_20_20260824` | `57.16%`, `83.45`, Jaccard `97.77%` | **Historical** |
| `conditional_v3_80_20_20260824` | `60.08%`, `79.40`, Jaccard `99.07%` | **Historical** |
| `conditional_v4_batch8_80_20_20260824` | `66.95%`, `71.25`, Jaccard `99.34%` | **Historical** |
| `conditional_v5_batch8_salvage_80_20_20260824` | `79.80%`, `58.90`, Jaccard `95.22%` | **Historical** |
| `conditional_v6_refinement_80_20_20260824` | `81.95%`, `56.50`, Jaccard `93.65%` | **Historical** |
| `conditional_v7_finalized_80_20_20260824` | `81.95%`, `56.50`, Jaccard `93.65%` | **Reference milestone** |
| `conditional_v7_experimental_promotion_20260825` | same frozen metrics as v7 | **Historical promotion probe** |
| `conditional_v8_activation_aware_80_20_20260824` | `80.02%`, `58.05`, Jaccard `94.73%` | **Historical** |
| `conditional_v9_revisable_conditions_80_20_20260824` | same aggregate metrics as v8 | **Historical** |
| `conditional_v10_joint_evidence_train_only_80_20_20260825` | `59.63%`, `80.00`, Jaccard `100%` | **Rejected/negative result** |
| `conditional_v11_compiled_specialists_train_only_80_20_20260825` | `78.56%`, `47.35`, Jaccard `97.31%` | **Reference** for the change-aware chain |
| `conditional_v12_failure_evidence_train_only_80_20_20260825` | `61.04%`, `78.15`, Jaccard `99.63%` | **Rejected/negative result** |
| `conditional_v13_failure_burden_train_only_80_20_20260825` | `59.16%`, `78.35`, Jaccard `98.36%` | **Rejected/negative result** |

### `numerical_selector/`

Producer: `scripts/run_numerical_selector_evolution.sh`. The named 80/20 runs evolve on Train and
use Dev only for acceptance. Directories ending in `train80` are Train-only audits or mutation
proposals; they are not independently accepted Dev results. Shared `hindcast-cache*` directories
are execution accelerators, not experiments.

| Runs | What they tested / result | Status |
|---|---|---|
| `smoke_8_2_20260823` | 8/2 selector smoke; mean MASE `0.9120` on two Dev tasks. | **Disposable smoke** |
| `formal_80_20_20260823` | Parent selector; Dev mean MASE `4.5495`, catastrophic rate `15.79%`; no accepted generation. | **Historical** |
| `formal_80_20_fallback_20260823` | Accepted fallback generation; Dev mean MASE `3.3753`, catastrophic rate `10%`. | **Canonical baseline selector** |
| `part2_v7_conditioned_specialists_80_20_20260825`, `part2_v7_profile_routing_80_20_20260825`, `part2_v7_train_only_combined_80_20_20260825` | Early Part-2 routing/Combined variants; Dev mean MASE about `3.20–3.29`. | **Historical ablations** |
| `part2_v7_canonical_cache_combined_80_20_20260825`, `part2_v7_controlled_attribution_80_20_20260825` | Cache/attribution diagnostics without a formal selector manifest. | **Disposable or historical diagnostics** |
| `part2_v11_compiled_specialists_80_20_20260825`, `part2_v11_crossfold_safe_80_20_20260825`, `part2_v11_safe_anchor_rerun_80_20_20260825` | Conservative variants converged to Dev mean MASE `3.2141`; no accepted generation. | **Historical** |
| `part2_v11_minimax_crossfold_80_20_20260825` | Accepted one generation, but Dev mean MASE `3.3265`. | **Historical**; evaluated on Public-99 and did not beat Toto |
| `part2_v13_failure_burden_80_20_20260825` | Failure-burden screening; Dev mean MASE `3.3023`. | **Historical** |
| `assumption_topk_80_20_20260825`, `assumption_topk_observed_80_20_20260825`, `assumption_topk_crossed_80_20_20260825`, `assumption_topk_combined_guard_80_20_20260825` | Assumption + Top-k ranking/guard variants; Dev mean MASE `3.21–3.26`. Public-99 variants improved some median/task-level behavior but not the main scaled-error baseline. | **Historical ablations** |
| `smae_guarded_combined_80_20_20260825`, `smae_guarded_combined_v2_80_20_20260825` | Early scaled-MAE guarded Combined designs; Dev mean sMAE `0.3371`. | **Historical** |
| `smae_safe_anchor_smoke_80_20_20260825` | Safe-Anchor smoke; Dev mean sMAE `0.2783`. | **Reference smoke** |
| `smae_train_only_combined_v3_80_20_20260825` | Frozen Safe-Anchor Combined selector used by the best completed Public-99 sMAE chain. | **Canonical guarded selector** |
| `target_horizon_audit_80_20_20260825`, `target_horizon_soft_penalty_80_20_20260825`, `task_conditioned_audit_80_20_20260825`, `task_conditioned_audit_activation_gate_80_20_20260825`, `context_preserving_long_horizon_80_20_20260825` | Analysis-only policy audits; no formal selector manifest. | **Historical diagnostics** |
| `protected_topk_80_20_20260826`, `protected_topk_parent_clean_80_20_20260826`, `protected_topk_parent_preserving_80_20_20260826` | Parent-protected Top-k experiments; one accepted generation, Dev mean sMAE `0.2539`, but not promoted as a final Public-99 system. | **Reference / development** |
| `conservative_tsfm_router_80_20_20260826`, `conservative_tsfm_overlay_80_20_20260826`, `conservative_tsfm_soft_overlay_80_20_20260826`, `conservative_tsfm_soft_overlay_v2_80_20_20260826`, `conservative_tsfm_soft_overlay_v3_80_20_20260826`, `conservative_tsfm_dual_metric_80_20_20260826` | Conservative TSFM routing/overlay attempts; all retained the parent Dev metrics (`MASE 3.2141`) and accepted no generation. | **Historical negative results** |
| `conservative_tsfm_adaptive_train80_20260826`, `conservative_combined_train80_20260826`, `conservative_combined_margin_train80_20260826`, `conservative_conditioned_combined_train80_20260826`, `conservative_conditioned_combined_final_train80_20260826` | Train-only proposals for safer TSFM/statistical combination. | **Historical development artifacts** |
| `joint_tsfm_statistical_train80_20260826`, `joint_tsfm_statistical_v2_train80_20260826`, `joint_tsfm_statistical_v3_train80_20260826`, `joint_tsfm_statistical_v4_train80_20260826` | Train-only joint TSFM/statistical search iterations. | **Historical development artifacts** |
| `protected_r1_r2_r3_train80_20260826` | Train-only staged protection experiment. | **Historical development artifact** |
| `change_aware_guard_80_20_20260826` | Change-aware guard paired with v11 screening. Dev retained the parent (`MASE 3.2141`); Public-99 full system sMAE `0.4706` was worse than Toto `0.4607`. | **Reference negative result** |
| `hindcast-cache/`, `hidden80-safe-anchor-hindcast-cache*` | Reusable historical-cutoff forecasts. No policy and no official hidden labels. | **Disposable/local cache** |

### `frozen_two_stage/`

Producer: `scripts/evaluate_frozen_two_stage.sh`. Every directory below contains a completion
marker, aggregate JSON, per-task JSONL, and generated report.

| Run | Full two-stage result on Public-99 | Status |
|---|---|---|
| `public_test_99_20260823` | mean MASE `3.1846` vs Toto `2.8830`; 41/7/51 | **Canonical baseline** |
| `public_regression_99_assumption_topk_20260825` | mean sMAE `0.4755`, sRMSE `0.7155`; 29/50/20 | **Historical ablation** |
| `public_regression_99_assumption_topk_combined_guard_20260825` | mean sMAE `0.4632`, sRMSE `0.6922`; 30/52/17 | **Historical ablation** |
| `public_regression_99_part2_v11_minimax_20260825` | mean sMAE `0.4782`, sRMSE `0.7094`; 34/44/21 | **Historical ablation** |
| `public_regression_99_safe_anchor_combined_20260825` | mean sMAE `0.4550`, sRMSE `0.6839`; 27/55/17 | **Canonical guarded reference** |
| `change_aware_guard_public99_20260826` | mean sMAE `0.4706`, sRMSE `0.7004`; 28/55/16 | **Reference negative result** |

Win/tie/loss counts in this section compare each full two-stage system with configuration A, the
frozen Toto/global reference used in the same artifact.

### Other top-level families

| Directory | Producer | Result | Status / recommendation |
|---|---|---|---|
| `fresh30_four_method_20260815/` | `scripts/run_fresh30_four_method_eval.sh` | On 30 public-development tasks, exploratory `retry2_v003` mean MAE `23.06` vs `v000` `33.07`, a `30.27%` reduction; v003 won/tied/lost `14/6/10`. Codex-Direct mean MAE was `64.26`, Codex-Contract `50.19`, and Chronos `51.57`. Its `manifests/` and `policies/` subdirectories are required compact inputs, not separate experiments. | **Reference historical comparison** |
| `posthoc_ablation/no_toto_public99_20260823/` | `runs/posthoc_ablation/screening_topk_safety_analysis.py` | Removed Toto after the fact; mean MASE `3.2571`, 41/4/54 versus original Toto. The artifact explicitly says `valid_as_new_sealed_test: false`. | **Diagnostic only** |
| `tsfm_smoke/persistent_workers_20260826/` | `scripts/run_tsfm_checkpoint_smoke.sh` | Granite TTM R2, Moirai 2.0, and Toto 2.0 each produced four finite CPU values from 96 history points; measured latency was about `9.70s`, `8.82s`, and `8.70s`. | **Reference runtime proof**, not accuracy evidence |
| `evolving/` | older `evolving-agent` CLI and co-evolution scripts | Prompt/Genome/Source evolution experiments and Codex response cache. Current visible contents are cache-only. | **Legacy; cache is disposable** |
| `hidden_test/safe_anchor_80_20260827/` | local hidden-export/inference work | Prediction/deployment artifact only; no locally available labels and therefore no official Dr-CiK score. | **Local WIP/reference**, never report as evaluated |

The local directories `numerical_selector/hidden80-safe-anchor-hindcast-cache/`,
`numerical_selector/hidden80-safe-anchor-hindcast-cache-v2-noscorer/`, and
`numerical_selector/hidden80-safe-anchor-hindcast-cache-v3-deployment-bound/` are three iterations
of hidden-inference hindcast caching. They contain no hidden labels or official score and are all
**Disposable/local cache** artifacts.

## Files that are usually leftovers

The following are useful while a run is active but are not durable scientific results:

- `.venv/` and other per-run environments;
- `hindcast-cache/`, `outcome-cache*`, `policy-outcome-cache*`, and LLM response caches;
- `agent-cache/`, `filter-agent-cache/`, `codex-cache/`;
- `transcripts/`, `filter-transcripts/`, raw execution logs, and temporary retry state;
- generated forecast plots;
- incomplete smoke directories that have no manifest, completion marker, or final report.

These paths are mostly excluded by `.gitignore`. They explain why a local `runs/` directory can be
several gigabytes while the tracked compact artifacts are only tens of megabytes. Do not delete an
active cache while its process is running; otherwise it can be regenerated when needed.

## Rules for adding a new run

A new durable run directory should contain or document:

1. the exact producer command or shell script;
2. the task split and task count;
3. source/dictionary, screening-policy, and Decision-policy fingerprints;
4. model/checkpoint and worker-environment identities where applicable;
5. a machine-readable result plus a completion marker;
6. whether Train, Dev, Public Regression, or official hidden data was accessed;
7. the primary metrics (`sMAE` and `sRMSE` for Dr-CiK comparison, with MASE/MAE/sMAPE as useful
   diagnostics);
8. whether the result is accepted, rejected, historical, or diagnostic;
9. an explicit statement when a result is post-hoc and therefore not a new sealed-test result.

Large caches, environments, checkpoints, transcripts, and logs should remain local or in external
artifact storage rather than being added to Git.
