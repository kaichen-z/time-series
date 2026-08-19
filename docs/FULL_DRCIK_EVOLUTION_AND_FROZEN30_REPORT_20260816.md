# Full Dr-CiK Evolution and Frozen-30 Evaluation

**Date:** 2026-08-16

**Repository:** `time-series-setting2`

**Branch:** `codex/setting2-domain-knowledge`
**Evaluation model:** Codex CLI, `gpt-5.6-sol`, reasoning effort `high`

## Executive summary

This iteration expanded Setting 2 from a fixed external forecasting library into a two-level learning system:

1. a static, source-backed time-series knowledge base supplies general forecasting principles selected from numeric diagnostics;
2. resolved Dr-CiK training outcomes create persistent Coding, Retrieval, and Decision skills;
3. a role-level co-evolution loop proposes prompt mutations, evaluates them with raw MAE, and accepts a policy only when it passes train, aggregate-dev, and per-dev-task guards.

The official Dr-CiK release was pinned and inventoried before training. It contains **279 tasks and 10,342 documents**: 199 public labeled tasks with 7,302 documents and 80 hidden unlabeled tasks with 3,040 documents. The hidden tasks were not used. After excluding 50 previously observed tasks and all 45 associated entities, two disjoint curricula used 24 new public tasks: 16 for training and 8 for held-out development. Task selection used only metadata, not future values, ground-truth evidence, or document roles.

The static knowledge base now contains **90 executable forecasting rules across 28 categories, grounded in 48 sources**. The two curricula added persistent outcome-derived skills, growing the runtime libraries from zero to **6 Coding, 12 Retrieval, and 5 Decision skills**. No prompt mutation passed all conservative deployment gates, so the frozen policy remained `v000`; the useful evolution in this run was the accumulated role skills.

The resulting Stage-2-seeded system was then frozen and evaluated once on the existing 30-task regression suite. It achieved:

- mean MAE **87.6564** and median MAE **4.9820**;
- mean sMAPE **36.8049**;
- versus Setting 1: **18.05% lower mean MAE**, 14 wins and 16 losses, with a paired-bootstrap probability of **98.21%** that mean MAE is lower;
- versus Codex-Contract: **43.94% lower mean MAE**, 18 wins and 12 losses;
- versus the previous best Setting 2 v4: **82.33% higher mean MAE**, 12 wins, 1 tie, and 17 losses.

The regression relative to v4 is dominated by one failure: task 123 contributes **57.59%** of the new system's total MAE. Excluding that task descriptively, the new mean is 38.4602 versus 43.0156 for v4. This does not remove the failure from the official result; it identifies the next technical bottleneck. The evidence-conditioned route transformed a reasonable Coding candidate into a very poor final mixture, while the current role credit assignment measured candidate-selection regret but not post-selection routing regret.

The user's desired gate is not fully met. The strict win rate is at least 40% against both Setting 1 and v4, and mean MAE falls substantially against Setting 1, but there are harmed tasks and mean MAE does not beat v4. Following the pre-declared stop instruction, no further tuning was performed on the frozen 30 tasks. The previous Setting 2 v4 remains the stronger frozen-30 checkpoint by primary mean MAE.

## 1. What was changed

### 1.1 MAE became the sole deployment objective

The earlier outer loop combined forecast sMAPE and retrieval quality. That could reward a policy for finding more supporting documents even when its numerical forecast became worse. This iteration changed system utility to:

```text
utility = -mean(final MAE)
```

Retrieval precision, supporting-document recall, distractor avoidance, sMAPE, and role-level regret remain diagnostics. They no longer contribute positive deployment reward. Parent selection, train rejection, dev acceptance, task-specialist retention, and merge ranking now use the same raw-MAE objective.

The mutation prompt also states explicitly that MAE is the sole deployment objective and that retrieval improvements cannot compensate for worse MAE.

### 1.2 Failure traces became forecast-utility traces

Each resolved task now records:

- final MAE and sMAPE;
- the best available Coding-candidate MAE;
- selected-candidate MAE regret;
- resolved MAE for every candidate;
- the selected candidate, route baseline, and route weight;
- retrieval precision, supporting recall, and distractor avoidance;
- supporting and retrieved document IDs.

This makes it possible to distinguish four different failures: no good numerical candidate, poor document selection, poor candidate selection, and harmful final routing.

### 1.3 Learned skills are now materialized

Previously, the search harness could learn task-local skills inside isolated evaluation instances and then discard them. After policy selection, the chosen policy is now rerun on training tasks with the persistent libraries enabled. This writes accepted, outcome-derived skills to the Coding, Retrieval, and Decision JSON libraries. The frozen evaluation therefore uses both the selected policy and the skills actually learned under that policy.

Only public resolved training outcomes can write skills. Dev, frozen-30, hidden, and unresolved tasks do not write to memory.

### 1.4 Exact manifests and pinned data

The command-line interface now supports exact ordered task manifests. A manifest may define explicit train and dev partitions. The host rejects overlapping task IDs, overlapping train/dev entities, missing IDs, and ambiguous use of `--limit` together with explicit partitions.

The Dr-CiK Hugging Face loader is pinned to revision:

```text
00fbe820ff7a221e4aca71883219ef27f8223050
```

This prevents a later dataset refresh from silently changing the evaluation population.

### 1.5 Operational reliability

All Codex calls used a 900-second timeout, 12 capacity retries, and a 30-second retry delay. Calls ran through the authenticated local Codex CLI; MaaS and QS were not used. The two curriculum stages and four frozen shards completed without capacity errors, timeouts, parser failures, or fallbacks.

## 2. Dr-CiK data inventory and curriculum

### 2.1 Pinned release

| Partition | Tasks | Entities | Documents | Labels used here |
|---|---:|---:|---:|---|
| Public development | 199 | 113 | 7,302 | Train/dev evaluation only |
| Hidden test | 80 | 64 | 3,040 | No |
| **Total** | **279** | — | **10,342** | — |

The 199 public tasks cover five frequencies: 114 hourly, 51 daily, 15 minutely, 10 five-minute, and 9 second-level tasks.

The local data blobs were recorded by SHA-256:

- tasks: `deb965ffba9fd5ddd46f8216be993ad1b6991ba7a0203b62385cf850389369f3`;
- documents: `c6ef369b7add313d009e9e1539b31708923abebe13a87da877d629155d44ea06`;
- task-document links: `2f251f8124c3b0c8a3d24348264db8767ec699eb1c6d33618e8a9c7b6adc4e48`.

### 2.2 Exclusion and selection

Before curriculum construction, 50 tasks used in prior pilots, development, or frozen evaluation were excluded. Their 45 entities were also excluded, preventing a differently numbered task about the same entity from crossing the boundary. This left 101 eligible public tasks.

From that pool, 24 tasks with 24 distinct entities were selected using label-free metadata. The selection procedure did not inspect `future_values`, `gt_evidence`, or document labels. The 24 tasks were divided into two sequential stages, each with eight train and four entity-disjoint dev tasks.

| Stage | Train task IDs | Dev task IDs |
|---|---|---|
| 1 | 237, 81, 80, 228, 240, 231, 195, 83 | 209, 152, 88, 61 |
| 2 | 76, 56, 75, 208, 200, 204, 224, 238 | 233, 71, 60, 70 |

Stage 1 covers temperature, speed, sales, electricity, traffic, and CPU behavior across all five released frequencies. Stage 2 deliberately adds a more concentrated retail/sales curriculum, plus a second-level speed task, to stress transfer under repeated but entity-disjoint business mechanisms.

## 3. Knowledge available to Setting 2

### 3.1 Static source-backed library

The external library contains 90 entries and 48 sources. Every entry has:

- a stable ID and category;
- a forecasting principle;
- explicit `use_when` and `avoid_when` conditions;
- executable implementation guidance;
- applicability tags and priority;
- one or more source IDs linked to citations and URLs.

The 28 categories are:

| Category | Entries | Category | Entries |
|---|---:|---|---:|
| seasonality | 6 | ETS/ARIMA | 6 |
| evaluation | 5 | advanced uncertainty | 5 |
| neural prior | 5 | baseline | 4 |
| level/trend | 4 | model selection | 4 |
| advanced seasonality | 4 | advanced combination | 4 |
| regime | 3 | autocorrelation | 3 |
| Bayesian process | 3 | TSFM | 3 |
| multistep | 3 | advanced regime | 3 |
| advanced intermittency | 3 | advanced transformation | 3 |
| robustness | 2 | transformation | 2 |
| intermittency | 2 | constraints | 2 |
| combination | 2 | nonparametric | 2 |
| nonparametric safety | 2 | time aggregation | 2 |
| advanced constraints | 2 | uncertainty | 1 |

A deterministic diagnostic pass computes history length, horizon ratio, autocorrelation, seasonal autocorrelation, trend effect, recent level and trend changes, outlier and zero fractions, variance shift, and candidate lags. These values become applicability tags. The retriever then selects at most ten relevant entries, normally no more than two per category. TSFM-specific entries are suppressed when no TSFM is available. The prompt instructs the Coding Agent to treat entries as falsifiable priors and cite their IDs, not obey them blindly.

Across the frozen 30 tasks, 47 distinct knowledge entries were actually selected. The final candidate sources were 11 built-in library programs, 5 open candidates, 5 open mutations, 3 knowledge-conditioned programs, 4 knowledge-conditioned mutations, and 2 evidence-adjusted candidates.

### 3.2 Outcome-derived role skills

The second knowledge layer is learned from resolved train outcomes. It does not replace the static library.

After Stage 1, the persistent libraries contained:

- 4 Coding skills, including `validated_multiscale_analog`, `accelerating_cycle_copy`, `macro_regime_recurrence`, and `rolling_ets_long_bias_guard`;
- 8 Retrieval skills, including separation of observation error from process effects, contradiction-aware retrieval, and boundary/mechanism triangulation;
- 3 Decision skills, including evidence-conditioned selection and bounded-window overrides.

After Stage 2, they grew to:

- **6 Coding skills**, adding `eight_step_damping_freeze` and `transition_onset_boundary`;
- **12 Retrieval skills**, adding sensor/dynamics separation, boundary-regime synthesis, evidence reconciliation, and expansion-effect disambiguation;
- **5 Decision skills**, adding normalized-analogue continuation and horizon-aligned causal selection.

During frozen-30 inference, `separate_observation_and_process_effects` and `contradiction_aware_regime_retrieval` were each invoked on 26 tasks. `evidence_conditioned_candidate_selection` was invoked on 28 tasks. This confirms that the learned libraries were not only written to disk but were injected and used in later inference.

## 4. Two-stage co-evolution result

### 4.1 Acceptance protocol

For each stage, the incumbent and every child were evaluated on the exact train manifest. A train-regressing child was stopped before dev. A train-safe child was evaluated on the exact entity-disjoint dev tasks. Deployment required all of the following:

1. no mean train-MAE regression;
2. lower aggregate dev MAE;
3. no individual dev-task MAE regression;
4. a material, role-owned prompt mutation.

Every generation proposed one Retrieval, one Decision, and one Coding child. The evaluator, data boundary, metric, forecast sandbox, and exact-quote verifier were immutable.

### 4.2 Stage 1

| Policy | Role | Train mean MAE | Dev mean MAE | Decision |
|---|---|---:|---:|---|
| `v000` | incumbent | 195.9867 | 2.4787 | retained |
| `v001` | Retrieval | 31.9828 | 2.4832 | rejected: dev mean did not improve |
| `v002` | Decision | 196.0454 | — | rejected: train regression |
| `v003` | Coding | 31.3259 | **1.2631** | rejected: task 209 regressed |

The Coding child delivered a large aggregate improvement but violated the no-harm guard on one held-out task. The Retrieval child greatly improved train MAE but did not generalize even at the aggregate dev level. Therefore the prompt policy stayed at `v000`. Training outcomes were still used to materialize the 4/8/3 role skills described above.

### 4.3 Stage 2

| Policy | Role | Train mean MAE | Dev mean MAE | Decision |
|---|---|---:|---:|---|
| `v000` | incumbent | 2,772.6250 | 139.9295 | retained |
| `v001` | Retrieval | 5,240.0158 | — | rejected: train regression |
| `v002` | Decision | 2,801.4334 | — | rejected: train regression |
| `v003` | Coding | 5,325.4869 | — | rejected: train regression |

All three Stage-2 mutations failed the first gate, so none saw dev and no merge was created. Task 200 alone accounts for approximately 98.6% of the incumbent's Stage-2 train absolute error and became substantially worse under the Retrieval and Coding children. This exposes a cost of using unscaled mean MAE exactly as requested: high-scale tasks correctly dominate aggregate absolute error, but they can also make small curricula unstable. The final prompt policy again stayed at `v000`, while resolved train outcomes expanded the persistent skill libraries to 6/12/5.

## 5. Frozen-30 protocol

The older 30-task manifest was preserved in its exact original order. It was split into four execution shards only for operational parallelism and then mechanically reassembled in manifest order. All shards used identical settings:

- Setting: statistics, with no external Chronos dependency;
- model: `gpt-5.6-sol`, reasoning effort `high`;
- three initial programs and one mutation per parent;
- three ordinary validation folds plus the host's deployment-scale validation behavior;
- Stage-2 `v000` policy and Stage-2 6/12/5 role skills;
- outcome learning disabled;
- no frozen-task feedback, policy update, or skill write.

All 30 tasks produced valid outputs. There were zero capacity failures, timeouts, parser failures, or fallbacks.

The old suite has been evaluated before and is therefore a regression/development suite, not a pristine hidden-test estimate. This run answers whether the newly evolved system preserves or improves prior performance on exactly the same 30 tasks.

## 6. Frozen-30 results

### 6.1 Aggregate comparison

| System | Mean MAE | Median MAE | Mean sMAPE | Strict W/T/L of new system |
|---|---:|---:|---:|---:|
| **New Stage-2-seeded Setting 2** | **87.6564** | **4.9820** | **36.8049** | — |
| Setting 1 | 106.9631 | 7.6947 | 38.0500 | 14 / 0 / 16 |
| Previous Setting 2 v4 | **48.0744** | 8.3134 | 37.8262 | 12 / 1 / 17 |
| Codex-Contract | 156.3736 | 7.7733 | — | 18 / 0 / 12 |

The new system has the best median MAE and the best mean sMAPE in this table, but raw mean MAE is the declared primary metric. By that metric it improves over Setting 1 and Codex-Contract and is 82.33% worse than Setting 2 v4.

### 6.2 Paired statistics

Task-paired bootstrap used 100,000 resamples with seed `20260816`.

| Baseline | Mean MAE delta (new − baseline) | Relative reduction | Strict win rate | 95% bootstrap CI for mean delta | P(new mean MAE lower) |
|---|---:|---:|---:|---:|---:|
| Setting 1 | −19.3067 | 18.05% | 46.67% | [−48.1013, −0.5025] | 98.21% |
| Setting 2 v4 | +39.5820 | −82.33% | 40.00% | [−11.0446, 131.8818] | 31.91% |
| Codex-Contract | −68.7172 | 43.94% | 60.00% | [−155.2799, −1.7399] | 98.94% |

The 95% Wilson interval for strict task-win rate is [30.23%, 63.86%] against Setting 1, [24.59%, 57.68%] against v4, and [42.32%, 75.41%] against Codex-Contract.

### 6.3 User-defined success gate

| Requirement | Versus Setting 1 | Versus previous v4 |
|---|---|---|
| Mean MAE substantially lower | Pass: −18.05% | Fail: +82.33% |
| Strict wins at least 40% | Pass: 46.67% | Pass at boundary: 40.00% |
| No harmed task | Fail: 16 losses | Fail: 17 losses |
| Overall conclusion | Partial | Not met |

Because the complete gate was not met, this branch is not promoted over v4 and was not tuned again on frozen-30.

## 7. Case studies

### 7.1 task 67: corrupted historical recurrence was rejected

| System | MAE |
|---|---:|
| New evolved Setting 2 | **49.9400** |
| Previous Setting 2 v4 | 163.1066 |
| Setting 1 | 197.1700 |
| Codex-Contract | 900.8084 |

The new system selected `macro_lag_alias_guard`. Retrieved evidence established that the apparent 168–190-step recurrence overlapped a synchronization-corrupted historical region. The host default treated this lag as a clean latent-process recurrence; the new candidate rejected that assumption and produced a smoother trajectory consistent with the documented post-remediation regime. This reduced MAE by 113.17 relative to v4 and by 147.23 relative to Setting 1.

This case combines the two knowledge layers: general analogue and seasonal rules from the static library, plus learned Retrieval skills for boundary-aware mechanisms, observation/process separation, contradiction handling, and telemetry reconciliation.

### 7.2 task 120: a hard forecast-window constraint overrode the default

| System | MAE |
|---|---:|
| New evolved Setting 2 | **17.7500** |
| Previous Setting 2 v4 | 39.6798 |
| Setting 1 | 69.1958 |
| Codex-Contract | 66.4739 |

A verified document constrained every October 16 direct-normal-irradiance value to below 60 W/m². That falsified the host default and every executed nonnegative candidate exceeding the ceiling. The Decision Agent selected `eight_step_damping_freeze`, the lowest-hindcast-error candidate satisfying the constraint, while explicitly avoiding the unsupported claim that irradiance must be exactly zero. The result improves MAE by 21.93 versus v4.

### 7.3 task 91: maintenance evidence prevented false recurring shutdowns

| System | MAE |
|---|---:|
| New evolved Setting 2 | **5.1955** |
| Previous Setting 2 v4 | 7.8980 |
| Setting 1 | 7.2460 |
| Codex-Contract | 8.4233 |

The numeric host default repeated near-zero collapses as a fixed cycle. Documents identified the earlier collapse as a seven-day shutdown and stated that it was the final maintenance cycle, with later recurring downtime suppressed. The Decision Agent therefore selected a positive-service `macro_regime_recurrence` trajectory. The adjustment is conservative: an undated projection above 46 was not used as a quantitative override.

### 7.4 task 123: bounded evidence adjustment plus routing caused a catastrophic miss

| System | MAE |
|---|---:|
| Previous Setting 2 v4 | **194.7801** |
| New evolved Setting 2 | **1,514.3454** |
| Codex-Contract | 1,822.9243 |
| Setting 1 | 1,871.6859 |

The system retrieved a coherent heat-event chain and produced a three-hour evidence adjustment. However, its best available Coding candidate had an oracle MAE of 208.1287, while the final routed forecast had MAE 1,514.3454. The final route used weight 0.2 with `unconditioned_top3_median`; the resulting mixture preserved too much of the poor route baseline.

The current `decision_selection_mae_regret` was zero because the selected evidence-adjusted candidate was the best candidate, but that diagnostic is calculated before the final routing mixture. It therefore failed to assign blame to the harmful router. This task alone contributes 57.59% of total MAE and explains the aggregate reversal relative to v4.

For diagnosis only, removing task 123 yields mean MAE 38.4602 for the new system versus 43.0156 for v4, 46.1105 for Setting 1, and 98.9063 for Codex-Contract. The official result still includes task 123.

### 7.5 task 214 and task 118: additional routing/selection regret

Task 214 finished at MAE 28.4939 even though the Coding oracle was 1.5080. Task 118 finished at 143.8500 with a Coding oracle of 57.9186 and candidate-selection MAE regret of 73.2064. Both use route weight 0.0, preserving the route baseline. These cases reinforce the same conclusion: future evolution needs explicit post-routing credit assignment, not only better candidate generation or retrieval.

## 8. Interpretation

### What worked

- The full official dataset is now pinned, inventoried, and selectable through reproducible manifests.
- Search and deployment optimize the same primary metric, raw MAE.
- Skills learned from resolved training outcomes persist into later stages.
- The system uses both general forecasting knowledge and entity-specific retrieved evidence.
- Several difficult tasks improved substantially, and the final median MAE is the best among the compared systems.
- Capacity handling was operationally reliable across both curricula and all four frozen shards.

### What did not work

- No prompt mutation passed the conservative no-regression gates; the policy itself did not advance beyond `v000`.
- A small raw-MAE curriculum was dominated by one high-scale task in each stage.
- Frozen performance is heavy-tailed: one routing failure erased gains on many smaller tasks.
- Role credit assignment stops at candidate selection and does not measure the additional regret introduced by the final route/blend.
- The no-harmed-task requirement was not achieved.

## 9. Decision and bounded next step

This frozen run is complete. The new system should not replace Setting 2 v4 on the basis of mean MAE. Per the stop rule, no additional frozen-30 prompt or knowledge tuning was performed.

The most direct future change, if a new training-only iteration is authorized, is not to enlarge the framework. It is to add one scalar diagnostic and gate:

```text
route_regret_mae = final_mae - selected_candidate_mae
```

The route should preserve a no-op candidate and be allowed only when causal backtests or bounded evidence edits predict lower absolute error. Evolution should then receive route regret as Decision/Router credit, while the frozen suite remains untouched. A larger entity-disjoint training curriculum should also report per-task and scale-stratified MAE alongside global raw MAE so that a single high-scale task is visible during selection.

## 10. Reproducibility artifacts

| Artifact | Path |
|---|---|
| Dataset inventory | `runs/full_dataset_evolution_20260816/dataset_inventory.json` |
| Stage-1 manifest | `runs/full_dataset_evolution_20260816/stage1_manifest.json` |
| Stage-1 trace | `runs/full_dataset_evolution_20260816/stage1_evolution_trace.json` |
| Stage-1 policy/archive | `runs/full_dataset_evolution_20260816/stage1_best_policy.json`, `stage1_policy_archive.json` |
| Stage-1 skills | `stage1_coding_skills.json`, `stage1_retrieval_skills.json`, `stage1_decision_skills.json` |
| Stage-2 manifest | `runs/full_dataset_evolution_20260816/stage2_manifest.json` |
| Stage-2 trace | `runs/full_dataset_evolution_20260816/stage2_evolution_trace.json` |
| Stage-2 policy/archive | `runs/full_dataset_evolution_20260816/stage2_best_policy.json`, `stage2_policy_archive.json` |
| Stage-2 skills | `stage2_coding_skills.json`, `stage2_retrieval_skills.json`, `stage2_decision_skills.json` |
| Frozen shard outputs | `runs/full_dataset_evolution_20260816/frozen30_shard_{1,2,3,4}.jsonl` |
| Ordered combined output | `runs/full_dataset_evolution_20260816/frozen30_evolved.jsonl` |
| Exact statistics and per-task records | `runs/full_dataset_evolution_20260816/frozen30_statistics.json` |

`frozen30_statistics.json` contains the source SHA-256 hashes, all paired statistics, all 30 task-level comparisons, selected candidates, route settings, retrieval metrics, used knowledge IDs, used learned skills, and decision rationales.
