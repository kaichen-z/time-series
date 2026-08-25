# Dr-CiK Metrics, Public Results, and the Position of Our System

**Date:** 2026-08-25
**Repository state inspected:** `d94ed50953a7a0bae2edc055194ce82be3298606` plus the metric corrections described in this report
**Official Dr-CiK revision inspected:** `4acbafe11f2e7caec792277caed606001abaf12c`
**Dataset revision used for local recomputation:** `00fbe820ff7a221e4aca71883219ef27f8223050`

## Executive summary

1. The current Dr-CiK release has **279 tasks**: 199 synthetic public-development tasks and 80 human-authored hidden-test tasks. The paper evaluated an older 240-task release. These protocols are not interchangeable.
2. The official hidden-80 leaderboard is currently **empty**. There is therefore no verified hidden-test SOTA. The scores shown on the project page are reference results from the original paper, not entries on the current leaderboard.
3. Forecasting is evaluated with future-value-scaled `sMAE`, `sRMSE`, and empirical `sCRPS`; `sCRPS` is the primary forecasting column. Each task score is capped at 5 before reporting the cross-task mean and standard error.
4. Deep research is evaluated with Evidence Recall, Supporting Document Recall, and Distractor Avoidance; Evidence Recall is the primary deep-research column. These are not ordinary unique-document precision/recall metrics.
5. No independent paper or project reporting a Dr-CiK result was found. Within the search scope described in Section 5, every public Dr-CiK score found came from the benchmark authors' evaluation. In particular, TimeClaw reports results on CiK, not Dr-CiK.
6. Recomputing the same historical 30 public tasks under the official forecasting formula gives our best historical Setting 2 v4 `sMAE = 0.4581 +/- 0.1089 SE` and `sRMSE = 0.6796 +/- 0.1602 SE`, versus Setting 1 at `0.5234 / 0.7477`. This is a real internal improvement, but it is not a leaderboard result.
7. Setting 2 v4 is point-valued. Repeating its point forecast can satisfy the output shape and sample-count requirement, but it gives a degenerate `sCRPS = 0.4581`, zero interval width, and only 5.84% empirical 90% coverage. It is not a meaningful or calibrated probabilistic forecast.

## 1. Benchmark versions and evaluation protocols

Three populations appear in current discussions and must remain separate.

| Protocol | Tasks | Labels | Proper use |
|---|---:|---|---|
| Original paper | 240 = 199 synthetic + 41 expert | Available to the benchmark authors | Paper Tables 1, 2, 6, and 7 |
| Current public development | 199 synthetic | Public | Development and reproducible local analysis |
| Current official test | 80 human-authored | Withheld | Maintainer-scored official leaderboard |

The current release contains 10,342 documents: 3,367 supporting and 6,975 distractor documents. Hidden tasks expose history, future timestamps, task metadata, and the document corpus, but withhold `future_values` and `gt_evidence`.

The protocol allows submission to one or both tracks. A forecasting submission must provide at least 100 sample trajectories for every hidden task. A deep-research submission must provide cited document IDs and synthesized evidence for every hidden task. Whichever track is submitted must cover all 80 tasks, and maintainers run the corresponding private scorer before publishing a verified entry.

The paper used 25 trajectories per task. The current submission protocol raises this to at least 100, so even a faithful reproduction of the paper protocol is not automatically a current-compliant submission.

## 2. Exact forecasting metrics

For one task, let the forecast horizon be $T$, the number of forecast trajectories be $S$, the target be $y_t$, and sample $s$ at step $t$ be $\hat y_{s,t}$.

The task scale is based on the **future target itself**:

$$
a = \left(\frac{1}{T}\sum_{t=1}^{T}|y_t|\right)^{-1}.
$$

This is not MASE, a history-naive scale, or a scale derived from the training series.

The point forecast is the sample mean:

$$
\bar y_t = \frac{1}{S}\sum_{s=1}^{S}\hat y_{s,t}.
$$

The two point metrics are:

$$
\mathrm{sMAE} = a\frac{1}{T}\sum_{t=1}^{T}|\bar y_t-y_t|,
$$

$$
\mathrm{sRMSE} = a\sqrt{\frac{1}{T}\sum_{t=1}^{T}(\bar y_t-y_t)^2}.
$$

At each horizon step, empirical CRPS is:

$$
\mathrm{CRPS}_t =
\frac{1}{S}\sum_s|\hat y_{s,t}-y_t|
-\frac{1}{2S^2}\sum_s\sum_{s'}|\hat y_{s,t}-\hat y_{s',t}|.
$$

The scaled distribution metric is:

$$
\mathrm{sCRPS}=a\frac{1}{T}\sum_{t=1}^{T}\mathrm{CRPS}_t.
$$

Lower is better for all three metrics. If all trajectories are identical, the dispersion term is zero and `sCRPS = sMAE`; this is mathematically defined but represents no uncertainty.

### 2.1 Cross-task aggregation

For each task and each of `sMAE`, `sRMSE`, and `sCRPS`, Dr-CiK independently applies:

```text
task_score = min(task_score, 5.0)
```

It then reports the arithmetic mean and standard error across valid tasks. The paper's Table 7 allows model-specific failures and reports a `Fail` count, so cells may cover different task sets. Table 6 instead uses the common 225-task all-model-valid intersection and reports mean plus sample standard deviation. The two tables must not be compared as if they had the same population or uncertainty statistic.

The paper also reports per-task average ranks:

- point average rank is based on `sRMSE`;
- distribution average rank is based on `sCRPS`.

The current official webpage marks `sCRPS` as the primary forecasting column, but the submission guide does not publish a hidden-board tie-break rule.

## 3. Exact deep-research metrics

For task $t$, let $E_t$ be its ground-truth evidence items. For evidence item $e$, let $R_e$ be the required supporting documents. The evaluator examines only the agent's top $K_t=|E_t|+5$ synthesized evidence items. Let $M_e$ indicate whether at least one of those items semantically matches $e$, and let $C_e$ be the union of source documents cited by the matched synthesized item or items. Let $Q_t$ be all resolved citations in the report and $D_t$ be distractor documents.

### 3.1 Evidence Recall

$$
\mathrm{EvidenceRecall} =
\frac{1}{\sum_t|E_t|}
\sum_t\sum_{e\in E_t}
M_e\frac{|R_e\cap C_e|}{|R_e|}.
$$

This metric requires both semantic recovery of the evidence and citation of its required documents. It is an evidence-item micro-average, not ordinary span recall. The paper uses `google/gemini-3-flash-preview` as the semantic judge.

### 3.2 Supporting Document Recall

$$
\mathrm{SupportingDocRecall} =
\frac{1}{\sum_t|E_t|}
\sum_t\sum_{e\in E_t}
\frac{|R_e\cap Q_t|}{|R_e|}.
$$

This is also weighted by ground-truth evidence items. A conventional unique-supporting-document recall implementation is only a proxy.

### 3.3 Distractor Avoidance

For a task with at least one resolved citation:

$$
\mathrm{DistractorAvoidance}_t = 1-\frac{|Q_t\cap D_t|}{|Q_t|}.
$$

The official score averages this value over tasks with at least one resolved citation. It divides by cited documents, not by all distractors available in the corpus. Higher is better. The official page marks Evidence Recall as the primary deep-research column.

## 4. What the official paper reports

### 4.1 End-to-end primary result: sCRPS

Paper Table 1 reports `sCRPS` on the original 240 tasks for three representative forecasters under several context sources.

| Context source | Aurora | Direct-Prompt Gemini | MoiraiAgent |
|---|---:|---:|---:|
| No context | 0.483 +/- 0.058 | 0.319 +/- 0.034 | 0.338 +/- 0.033 |
| Original context (oracle/source context) | 0.487 +/- 0.058 | 0.233 +/- 0.032 | **0.206 +/- 0.030** |
| Bench2Future | 0.481 +/- 0.057 | 0.631 +/- 0.065 | 0.521 +/- 0.058 |
| DrBench | 0.483 +/- 0.058 | 0.567 +/- 0.062 | 0.483 +/- 0.055 |
| Retrieval | 0.479 +/- 0.057 | 0.586 +/- 0.062 | 0.515 +/- 0.059 |
| OpenDR | 0.482 +/- 0.058 | 0.582 +/- 0.061 | 0.415 +/- 0.042 |
| Codex GPT-5.5 High | 0.483 +/- 0.058 | **0.326 +/- 0.033** | **0.310 +/- 0.035** |

The strongest autonomous deep-research pairing in this table is Codex evidence plus MoiraiAgent at `0.310`. The overall `0.206` cell uses original source context and is an oracle-context upper reference, not autonomous retrieval.

### 4.2 Deep-research quality

| Agent | Evidence Recall | Supporting-doc Recall | Distractor Avoidance |
|---|---:|---:|---:|
| **Codex / GPT-5.5 High** | **38.5%** | **48.9%** | **41.0%** |
| OpenDR | 4.8% | 9.9% | 23.5% |
| Retrieval | 4.3% | 10.0% | 20.4% |
| DrBench | 3.9% | 9.2% | 29.0% |
| Bench2Future | 3.8% | 7.5% | 22.7% |

Codex is far ahead on evidence recovery. Because Distractor Avoidance is averaged per task, 41.0% corresponds to an average per-task distractor share of approximately 59% among tasks with resolved citations; it does not imply that exactly 59% of all pooled citations are distractors.

### 4.3 Best Table 7 reference cells by forecasting metric

The best overall cells use original context and MoiraiAgent:

| Metric | Best original-context reference |
|---|---:|
| sMAE | **0.242 +/- 0.031** |
| sRMSE | **0.343 +/- 0.042** |
| sCRPS | **0.206 +/- 0.030** |

Within actual Codex-synthesized context, different forecasters win different metrics:

| Metric | Best Codex-context cell | Score | Failures |
|---|---|---:|---:|
| sMAE | Gemini 3.1 Flash-Lite, medium reasoning | **0.370 +/- 0.035** | 0 |
| sRMSE | MoiraiAgent | **0.516 +/- 0.051** | 1 |
| sCRPS | Qwen3.5-9B | **0.297 +/- 0.027** | 11 |
| Point average rank | MoiraiAgent | **8.90** | 1 |
| Distribution average rank | MoiraiAgent | **9.62** | 1 |

If full coverage is required, the lowest zero-failure Codex-context `sCRPS` in Table 7 is Gemini medium at `0.314 +/- 0.033`; its `sRMSE` is `0.518 +/- 0.047`. The `0.297` Qwen result cannot be treated as a fully covered current submission because it failed on 11 paper tasks.

The controlled Figure 8 result `sCRPS = 0.105` for ground-truth evidence is an oracle ablation with a particular Direct-Prompt Gemini setup. It is not a leaderboard score and should not be called SOTA.

## 5. Are there independent Dr-CiK results?

No independent result was found as of 2026-08-25.

The search covered:

- the official hidden leaderboard and `submissions/` directory;
- open and merged pull requests in the official repository;
- exact searches for `Dr-CiK`, `sCRPS`, benchmark results, and submissions;
- OpenAlex citation records and papers citing or discussing Dr-CiK;
- project pages for related contextual forecasting agents.

The official leaderboard arrays for both forecasting and deep research remain empty, and the official repository contains only a submission template. OpenAlex reported zero citations for the benchmark paper at the time of inspection. Several reviews and surveys discuss the benchmark, but they do not report new evaluated results.

Important name disambiguation:

- **TimeClaw** reports Context-is-Key (`CiK`) results, not Dr-CiK.
- Papers titled around “contextualized time series” may also use CiK or GIFT-CTX rather than Dr-CiK.
- Results from such benchmarks are useful design references but cannot appear in a Dr-CiK leaderboard comparison table.

## 6. Recomputing our archived results under the official formula

All rows below use the same 30 synthetic public tasks and the same current public labels. The recomputation reads each original forecast trajectory, applies the official future-target scale, independently caps each task metric at 5, and reports mean plus standard error.

| System | Raw mean MAE | Median MAE | sMAE | sRMSE | sCRPS | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Setting 1 | 106.9631 | 7.6947 | 0.5234 +/- 0.1471 | 0.7477 +/- 0.1873 | 0.5234 +/- 0.1471† | 8.53% |
| Setting 2 v2 | 56.0935 | 7.2793 | 0.5661 +/- 0.1592 | 0.7689 +/- 0.1925 | 0.5661 +/- 0.1592† | 6.57% |
| Setting 2 v3 | 52.1090 | 8.6992 | 0.5521 +/- 0.1479 | 0.7687 +/- 0.1917 | 0.5521 +/- 0.1479† | 6.49% |
| **Setting 2 v4** | **48.0744** | 8.3134 | **0.4581 +/- 0.1089** | **0.6796 +/- 0.1602** | **0.4581 +/- 0.1089**† | 5.84% |
| Full-curriculum system | 87.6564 | **4.9820** | 0.4662 +/- 0.1021 | 0.7148 +/- 0.1445 | 0.4662 +/- 0.1021† | 7.53% |
| Codex-Contract | 157.3252 | 7.6470 | 0.7822 +/- 0.2065 | 1.0140 +/- 0.2403 | 0.6741 +/- 0.1841 | 42.01% |

`†` The Setting 1/2 systems emitted one point trajectory. Their displayed `sCRPS` is the exact score of a repeated-point degenerate distribution, not a calibrated probabilistic result. Their scaled interval width is zero. Codex-Contract emitted 100 non-identical trajectories.

One Setting 1 task and one task in each of v2/v3 reached the `sRMSE` cap; v4 and the full-curriculum system had no capped tasks. The Codex-Contract row had two capped `sRMSE` tasks.

### 6.1 What changed after using the correct metric

Setting 2 v2 and v3 appear much better than Setting 1 under raw mean MAE because a few high-scale tasks dominate absolute error. Under the official per-task normalized aggregate, both are worse than Setting 1. Setting 2 v4 is the first archived version that improves both objectives:

- versus Setting 1, v4 lowers official sMAE by **12.48%**;
- it lowers official sRMSE by **9.11%**;
- it lowers raw mean MAE by **55.05%**.

This is why future development must record the official scaled metrics rather than inferring benchmark progress from raw MAE alone.

### 6.2 Correcting earlier local retrieval numbers

The archived evolving-loop code calculated:

```text
1 - retrieved_distractors / all_available_distractors
```

and called it Distractor Avoidance. That is not the official definition. The correct task denominator is all cited documents. Therefore the historical stored value around `0.841` cannot be compared with the paper's `0.410` Codex result. Historical retrieval precision around `0.535` is conceptually closer, but it is still not a formal official score.

Likewise, the historical `supporting_recall` uses unique supporting document IDs rather than evidence-item-weighted required-document coverage. We did not compute official Evidence Recall because that additionally requires ground-truth evidence matching, top-`|E|+5` truncation, required-document mappings, and the specified semantic judge.

## 7. Where our best historical system stands

On its own 30-task development slice, v4 is clearly better than our Setting 1 and Codex-Contract baselines in point accuracy. Relative to Codex-Contract, it reduces sMAE by 41.44% and sRMSE by 32.98%.

Against the paper's best actual Codex-context reference cells, the numerical gaps are:

| Metric | Our v4 public-30 | Paper Codex-context reference | Absolute gap | Relative gap |
|---|---:|---:|---:|---:|
| sMAE | 0.4581 | 0.370, Gemini medium, 0 fail | +0.0881 | +23.8% |
| sRMSE | 0.6796 | 0.516, MoiraiAgent, 1 fail | +0.1636 | +31.7% |
| sCRPS | 0.4581, degenerate | 0.297, Qwen9B, 11 fail | +0.1611 | +54.2%, descriptive only |

These are descriptive cross-protocol differences, not a rank or an isolated method effect. The comparisons differ in task population, repeated development exposure, model/backbone, context pipeline, forecast distribution, and model-specific valid-task subsets. The strongest defensible statement is:

> Setting 2 v4 makes material internal progress over our comparable baselines, but remains roughly 24-32% worse in point metrics than the strongest Codex-context paper cells, and it lacks a meaningful probabilistic forecast altogether.

Formal positioning requires two additional experiments:

1. run a frozen system once on all 199 current public-development tasks and report official metrics with full coverage;
2. export at least 100 genuine trajectories for every hidden task and submit all 80 tasks for maintainer scoring.

### 7.1 Post-research v004 recheck

The conservative v004 implementation was subsequently run on the same 30 public tasks. It obtained `sMAE 0.6572 +/- 0.1543 SE`, `sRMSE 0.8985 +/- 0.1867 SE`, and degenerate deterministic `sCRPS 0.6572 +/- 0.1543 SE`. This is worse than both Setting 1 and historical Setting 2 v4 on the official point aggregates. Its raw median MAE is lower at 6.7028, but raw mean MAE rises to 218.6567 because of severe tail failures. The best historical internal result therefore remains Setting 2 v4. Full diagnostics and cases are reported in `DRCIK_STRONG_AGENTS_AND_CONSERVATIVE_UPGRADE_20260825.md`.

## 8. Metric implementation correction

This work added the paper's exact public forecasting formulas to `common.metrics` and surfaced them in both the Dr-CiK agent and frozen evolving-loop reports. The implementation:

- uses the future-target mean absolute value as the denominator;
- derives the point forecast from the sample mean;
- computes empirical CRPS in `O(S log S)` per horizon step;
- caps each task metric independently at 5;
- reports mean and sample standard error;
- raises an explicit error for an all-zero future target because the public paper does not specify a fallback;
- retains the old history-scaled values under explicit `*_proxy` names where backward compatibility is needed.

An independent NumPy implementation was compared against the new shared scorer on all 30 v4 tasks. The maximum absolute difference over `sMAE`, `sRMSE`, and `sCRPS` was `4.44e-15`.

## 9. Sources

- [Dr-CiK paper](https://arxiv.org/abs/2605.27904)
- [Official repository](https://github.com/ServiceNow/Dr-CiK)
- [Official submission protocol at the inspected revision](https://github.com/ServiceNow/Dr-CiK/blob/4acbafe11f2e7caec792277caed606001abaf12c/SUBMISSION.md)
- [Official leaderboard data at the inspected revision](https://github.com/ServiceNow/Dr-CiK/blob/4acbafe11f2e7caec792277caed606001abaf12c/docs/static/data/leaderboard.js)
- [Live leaderboard](https://servicenow.github.io/Dr-CiK/#leaderboard)
- [Official Hugging Face dataset](https://huggingface.co/datasets/ServiceNow/Dr-CiK)
- [MoiraiAgent research implementation](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-agent)
- [TimeClaw, a related CiK rather than Dr-CiK system](https://arxiv.org/abs/2606.05404)
