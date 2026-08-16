# Baseline Methods and Results Catalogue

**Repository:** <https://github.com/kaichen-z/time-series>

**Audit date:** August 13, 2026

This document consolidates the baseline and diagnostic runs found in the local experiment archive.
The audit inspected **68 `summary.json` artifacts**, the frozen 30-task manifest and report, the
Coding-evolution trace, and the newer three-level evolution traces. Many of the 68 directories are
reruns, shards, parser/safety patches, or one-task debugging checks. This catalogue reports each
distinct method family once and preserves notable failure trajectories separately.

Machine-readable catalogue: [`results/baseline_results_catalogue.json`](results/baseline_results_catalogue.json)

All currently executable methods use one entrypoint:

```bash
python -m evolving_loop --list-methods
python -m evolving_loop --baseline <name> [data options]
python -m evolving_loop --evolution <prompt|genome|source> [data options]
```

The baseline names distinguish fixed methods and never trigger evolution. The evolution names run
train/entity-held-out-dev search and apply their respective acceptance gate.

## 1. Metric and comparability rules

- **MAE and RMSE:** lower is better. These are directly comparable only when the task set and
  numerical backbone are the same.
- **sMAE, sRMSE, and sCRPS:** lower is better. Values emitted by the main agent runner are local
  development proxies, not official hidden-test Dr-CiK scores.
- **Retrieval precision, supporting-document recall, and distractor avoidance:** higher is better.
- **System reward:** higher is better. This metric belongs to the new evolution harness and must not
  be compared numerically with MAE or the Dr-CiK scaled metrics.
- Three-task and one-task results are mechanism diagnostics, not ranking evidence.
- Oracle-context runs use public ground-truth evidence and are ceiling diagnostics, not deployable
  baselines.

## 2. Method families

### 2.1 Numerical-only baselines

| Method | Numerical model | Context use |
|---|---|---|
| Seasonal naive | Repeat the configured seasonal lag | None |
| Drifted seasonal naive | Seasonal repeat plus a local drift term | None |
| Chronos-Bolt | `amazon/chronos-bolt-small` or the cached local Bolt checkpoint | None |
| Generated statistical programs | LLM-generated or hand-available statistical programs, executed and hindcast | Numbers only |

### 2.2 Contextual revision baselines

| Method | Description |
|---|---|
| One-pass statistical | Statistical baseline plus one retrieval/evidence pass; no accepted numerical revisions in the recorded sample run |
| Iterative statistical | Gap diagnosis, repeated retrieval, evidence verification, and a restricted revision workspace |
| Unsafe Chronos revision | Early Chronos agent that applied contextual revisions without a sufficiently conservative acceptance gate |
| Safe Chronos gate | Chronos plus conservative bounded revision and fallback-to-baseline behavior |
| Oracle evidence | Same revision machinery supplied with public ground-truth evidence; diagnostic only |
| Regime retrieval | Retrieval identifies observation errors, temporary events, normal regimes, and future drivers, then opens compatible numerical candidates |
| Regime-table retrieval | Regime retrieval plus a structured mechanism table and deterministic evidence-to-impact mapping |

### 2.3 LLM-assisted decision baselines

| Method | Description |
|---|---|
| Codex Direct | Codex emits a contract/decision from context but does not generate and compare a rich executable candidate set |
| Codex Contract | Codex produces a structured causal contract; Python generates and validates compatible numerical candidates |
| Codex Contract + explicit points | Contract system additionally permits independently validated, horizon-aligned explicit future values |
| Rules Triad | Deterministic Coding, retrieval, verification, and Decision roles |
| Codex Triad | Codex performs Coding hypothesis formation, Retrieval, and Decision; Chronos/Python still produce executable numbers |
| Coding self-evolution | Codex generates Python forecasters, runs historical hindcasts, mutates a candidate, and keeps only a backtest improvement |

### 2.4 Evolution baselines

| Method | Mutable object |
|---|---|
| Prompt-only | Exactly one role prompt |
| Harness Genome | Prompts, Coding budgets, hindcast configuration, workflow/topology, evidence policy, and aggregation |
| Source evolution | Audited Coding/Retrieval/Decision/Harness Python in isolated Git worktrees |

## 3. Public-development aggregate results

### 3.1 Full 199-task Chronos family

These runs share the same public-development task set and Chronos baseline.

| Method | Tasks | MAE | RMSE | sMAE proxy | sRMSE proxy | sCRPS proxy | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Chronos baseline, revisions disabled | 199 | 797.6268 | 1121.9784 | 2.6981 | 3.2032 | 2.5370 | Reference numerical baseline |
| Early unsafe contextual revision | 199 | 1289.7688 | 1591.6977 | 2.8350 | 3.2252 | 2.6919 | MAE worsened by 61.70%; 14.57% of tasks were harmed |
| Safe contextual gate, final | 199 | 797.1785 | 1121.4969 | 2.6724 | 3.1789 | 2.5117 | MAE improved by 0.056%; one harmed task |
| Oracle public evidence | 199 | 797.1506 | 1121.4600 | 2.6641 | 3.1650 | 2.5038 | MAE improved by 0.060%; not deployable |

Retrieval quality for the non-oracle 199-task runs was unchanged because they reused the same
retrieval output:

- retrieval precision: **0.3154**;
- supporting-document recall: **0.2210**;
- distractor avoidance: **0.3154**.

The main lesson is safety rather than broad gain: the early revision mechanism could be
catastrophically harmful, while conservative gating reduced the system to sparse, small changes.
Even oracle evidence produced only a small aggregate improvement, showing that evidence-to-number
translation and candidate coverage remain bottlenecks after retrieval.

### 3.2 Frozen 30-task Codex Contract evaluation

This is the strongest currently citable aggregate agent result in the repository. The frozen subset
excluded `task_42`, `task_163`, and `task_201`, which had been used during development.

| Metric | Chronos baseline | Codex Contract | Change |
|---|---:|---:|---:|
| Mean MAE | 162.626963 | 156.362114 | -3.85% |
| Mean RMSE | 301.011687 | 296.465854 | -1.51% |

Additional results:

- final CRPS: **149.194873**; no directly comparable baseline CRPS was stored;
- improved / unchanged / harmed tasks: **2 / 28 / 0**;
- retrieval precision: **0.7791**;
- supporting-document recall: **0.4120**;
- distractor avoidance: **0.7791**;
- Codex failures: **0 / 30**.

The two accepted revisions were `task_67` (17.26% MAE reduction) and `task_213` (2.94%). Most of the
aggregate MAE reduction came from `task_67`; the median task was unchanged.

### 3.3 Frozen 30-task Codex Triad diagnostic

The three ten-task shards cover the same frozen 30-task manifest. Weighted aggregation gives:

| Metric | Chronos baseline | Codex Triad | Change |
|---|---:|---:|---:|
| Mean MAE | 162.974946 | 219.754773 | +34.84% |
| Mean RMSE | 301.028082 | 335.928667 | +11.59% |

Other aggregate diagnostics:

- retrieval precision: **0.4467**;
- supporting-document recall: **0.2805**;
- distractor avoidance: **0.8467**;
- harmed tasks: **11 / 30**;
- recorded Codex stage calls: **683**;
- recorded Codex stage failures: **252**.

This run is a negative baseline. Candidate generation sometimes improved coverage, but Decision and
runtime reliability were insufficient. It motivated the later citation-constrained Decision logic,
fallback rules, and the current evolution framework.

### 3.4 Separate Dr-CiK reproduction package

The local project archive also contains results from a second, isolated `dr-cik/` reproduction.
That package is not part of the current branch. Its metrics follow its own implementation of the
Dr-CiK formulas and should not be merged with the main runner's development proxies; they are
preserved here so that the earlier experiments are not lost.

| Dataset | Method | sMAE | sRMSE | sCRPS |
|---|---|---:|---:|---:|
| Public dev, 199 tasks | DRBench + Qwen2.5-14B + Chronos-Bolt-base | 0.542 | 0.748 | 0.421 |
| Public sample, 3 tasks | Chronos-Bolt-base | 0.134 | 0.179 | 0.263 |
| Public sample, 3 tasks | Direct Prompt, Qwen3.5-4B | 0.178 | 0.245 | 0.131 |
| Public sample, 3 tasks | Direct Prompt, Qwen3.5-9B | 0.161 | 0.207 | 0.113 |

For the 199-task DRBench run, the recorded evidence-recall proxy was **0.295**; supporting-document
recall was **0.058 cited / 0.149 retrieved**, and distractor avoidance was **0.180 cited / 0.222
retrieved**. The Direct-Prompt sample methods consumed context from a prior DRBench run. They used
real model outputs without fallback markers, but three tasks remain too small for ranking claims.

### 3.5 Legacy intermediate aggregates

Two earlier checkpoints are retained for completeness but should not be used as headline results:

| Run | Tasks | MAE | RMSE | Interpretation |
|---|---:|---:|---:|---|
| Initial `official-sample` pipeline | 3 | 42.1395 | 56.0720 | Early end-to-end sample before the later statistical/Chronos revisions |
| Chronos contextual public subset | 20 | 50.0577 | 84.9578 | Baseline MAE 49.9293; contextual revision worsened MAE by 0.26% and harmed 1/20 tasks |

The 20-task checkpoint helped motivate the conservative full-199 safety gate. The remaining saved
20/30-task shard directories are either components of the aggregate tables above, smoke tests, or
reruns created while repairing parsing, grounding, and safety behavior.

## 4. Three-task sample development ladder

These runs use the official public sample (`task_42`, `task_163`, and `task_201`) but not always the
same backbone. Results are grouped by comparable baseline.

### 4.1 Drifted-seasonal statistical baseline

| Method | MAE | RMSE | Retrieval precision | Supporting recall | Result |
|---|---:|---:|---:|---:|---|
| One-pass statistical | 9.4101 | 17.9559 | 0.3333 | 0.2013 | No numerical revision |
| Iterative statistical | 9.4101 | 17.9559 | 0.4180 | 0.2573 | Better retrieval, same forecast |

The iterative loop found more relevant support but could not translate it into an accepted numeric
change. This is an early example of retrieval improvement not automatically producing forecasting
improvement.

### 4.2 Chronos contextual revisions

| Method | Baseline MAE | Final MAE | Final RMSE | Retrieval precision | Result |
|---|---:|---:|---:|---:|---|
| Chronos, fixed retrieval, no accepted change | 27.7416 | 27.7416 | 39.4859 | 0.4180 | Unchanged |
| Regime retrieval | 27.7416 | 14.0457 | 23.2439 | 0.4180 | 49.37% MAE reduction |
| Regime-table retrieval | 27.7416 | 13.5600 | 22.3424 | 0.4180 | 51.12% MAE reduction |
| Oracle regime evidence | 27.7416 | 13.6800 | 22.5559 | N/A | Diagnostic ceiling |

These large gains are sample-specific and were a major motivation for representing whether evidence
acts on the observation layer, latent process, future driver, or regime.

### 4.3 Codex Contract variants

| Method | Baseline MAE | Final MAE | Final RMSE | Retrieval precision | Improved / unchanged / harmed |
|---|---:|---:|---:|---:|---:|
| Codex Contract | 27.7416 | 14.0457 | 23.2439 | 0.9048 | 1 / 2 / 0 |
| Contract + validated explicit points | 27.7416 | 13.6800 | 22.5559 | 0.9048 | 2 / 1 / 0 |

The explicit-points variant accepted four independently validated points and improved one additional
sample task. This remains a three-task development result.

## 5. `task_42` development history

All rows below are one-task diagnostics and are strongly overfit to the development example. They
are retained because they show why the system architecture changed.

| Method/version | MAE | RMSE | Retrieval precision | Decision behavior |
|---|---:|---:|---:|---|
| Chronos / Codex Direct | 72.7346 | 89.3146 | 1.0000 | Preserved baseline; context was not converted into a useful candidate |
| Initial Codex Triad, all calls failed | 72.7346 | 89.3146 | 0.0000 | Full fallback |
| Codex Triad v1, calls repaired | 72.7346 | 89.3146 | 0.6667 | Better retrieval, still selected baseline |
| Codex Triad v2 | 94.2149 | 122.0032 | 0.8333 | Harmful contextual selection |
| Codex Triad v3 / Codex Contract | 31.6472 | 40.5885 | 0.8333 to 1.0000 | Executable normal-regime candidate selected |
| New evolving-agent parent replay | 104.4267 | N/A in this metric family | 0.7500 | Different implementation and reduced Coding budget; final sMAPE 23.6486 |

The v3 improvement is the origin of the frequently cited `72.73 -> 31.65` MAE result. It should not
be interpreted as a general benchmark result. The newer evolving-agent replay uses a different
candidate system and reports sMAPE as its primary metric, so its MAE is not evidence that the new
system is categorically worse.

## 6. Coding self-evolution result

On `task_42`, Codex generated three initial executable statistical programs and one mutation. All
programs were evaluated by three historical hindcast folds.

| Quantity | Value |
|---|---:|
| Initial best future MAE | 47.3090 |
| Selected future MAE after one mutation round | 47.3090 |
| Future MAE improvement | 0.0000 |

The mutation did not improve historical backtesting, so the system retained the parent program.
This validates the inner acceptance gate but does not demonstrate a performance gain.

## 7. Three-level evolution results

| Evolution mode | Parent train reward | Child train reward | Parent dev reward | Child dev reward | Accepted? |
|---|---:|---:|---:|---:|---|
| Prompt-only | N/A | N/A | N/A | N/A | Not yet run |
| Harness Genome | 0.659675 | 0.850589 | 0.859456 | 0.870863 | Yes |
| Source evolution | 0.898000 | 0.874556 | 0.859363 | 0.847298 | No |

The Genome child introduced `retrieve -> decide -> retrieve -> decide` and improved held-out reward.
The Source child passed static audit and all 163 tests but degraded both train and development
reward, so its patch was rejected.

## 8. Other one-task and targeted diagnostics

These experiments are useful for debugging but should not appear in a headline comparison:

| Run | Tasks | Baseline MAE | Final MAE | Interpretation |
|---|---:|---:|---:|---|
| Rules `task_116` | 1 | 15.8733 | 15.8733 | Rule system rejected all changes |
| Codex hard `task_116` | 1 | 15.8733 | 12.0222 | 24.26% improvement after a two-step retrieval fix |
| Rules `task_117` | 1 | 228.8206 | 228.8206 | Rule system rejected all changes |
| Codex hard `task_117` | 1 | 228.8206 | 222.7202 | 2.67% improvement after retrieval fix |
| Decision-fix targeted set | 5 | 455.7398 | 445.8512 | 3 improved, 1 unchanged, 1 harmed; not frozen |
| Task 191 safety check | 1 | 1.7233 | 1.7233 | Gate preserved baseline |

## 9. Consolidated conclusions

1. **Chronos is a stronger starting point than an unrestricted contextual reviser.** Early context
   application increased 199-task MAE by 61.70%.
2. **A conservative gate is essential.** It converted catastrophic degradation into a small,
   sparse 199-task improvement.
3. **Retrieval quality alone is insufficient.** The statistical sample loop improved retrieval but
   not forecast values; oracle evidence also produced only a small 199-task gain.
4. **Structured contracts are the strongest current aggregate baseline.** On the frozen 30-task
   subset, Codex Contract improved MAE by 3.85% with no harmed tasks.
5. **The earlier free-form Codex Triad was unstable.** It worsened frozen 30-task MAE by 34.84% and
   motivated executable-candidate constraints and conservative Decision logic.
6. **Current evolution gates behave correctly.** Genome evolution accepted a held-out improvement;
   Source evolution rejected an executable but empirically weaker architecture.

## 10. Recommended benchmark table for the next paper draft

The next clean experiment should evaluate, on one frozen entity-disjoint split and identical
budgets:

1. Chronos only;
2. one-pass statistical;
3. safe contextual Chronos;
4. Codex Contract;
5. fixed three-agent harness;
6. Prompt-evolved harness;
7. Genome-evolved harness; and
8. Source-evolved harness.

Report MAE, RMSE, sCRPS, retrieval precision/recall, Decision regret, harmful-revision rate, token
cost, latency, and multiple random seeds. Until that experiment is complete, results from different
task sets and metric implementations must remain in separate tables as they are here.
