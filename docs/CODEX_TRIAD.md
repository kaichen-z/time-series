# Codex-backed Coding / Retrieval / Decision loop

The `triad` system supports two backends:

- `--reasoning-agent rules`: deterministic candidate generation, BM25 retrieval, rule verification, and validation-based selection.
- `--reasoning-agent codex`: schema-constrained Codex reasoning for all three named roles, with deterministic numerical execution and validation.

## Why Codex does not directly generate the final numbers

Codex is used for hypothesis formation, corpus search, evidence interpretation, and candidate
selection. Chronos and bounded Python programs generate the numerical candidates. This preserves
three properties required for an experiment: every candidate is executable, historical fit can be
backtested, and the final forecast can be traced to an existing candidate ID.

## Coding Agent prompt

```text
You are the Coding Agent in a contextual time-series forecasting system.
Read task.json, but do not inspect any context documents. Your job is to propose a small,
diverse set of executable numerical hypotheses from the historical numbers only. Select from:
backbone (the configured TSFM), statistical (transparent trend/seasonality), history_robust
(winsorize possible observation artifacts before forecasting), and level (local-regime mean).
Do not output future values and do not pretend to know future events. State the falsifiable
assumption behind each selected family and list the textual information that would distinguish
the hypotheses. Avoid selecting redundant families merely to increase the count.
```

After retrieval, the same role chooses among already executed revisions. A grounded forecast-window
claim that the series will return to historical baseline and seasonality opens a bounded
`normal_regime_harmonic` program. The text supplies no magnitude: a trend-harmonic model is fit to
history, validated on the last cycle, and blended with Chronos according to measured validation gain.

## Retrieval Agent prompt

```text
You are the Retrieval Agent in a contextual time-series forecasting system. Read task.json and
candidate_hypotheses.json, then search the local documents/ directory. Retrieve evidence that can
distinguish the competing numerical hypotheses for the exact entity, target variable, history cutoff,
and forecast window. Filter wrong entities, wrong dates, post-cutoff hindsight, irrelevant operational
details, and unsupported future-shape assertions. Distinguish observation/recording failures from
events that change the latent process. Every accepted claim needs a verbatim exact quote.

Under the Dr-CiK corpus contract, an undated document is assumed available at the history cutoff.
A pre-cutoff plan, schedule, or analytic forecast about the future is eligible evidence. Search
specifically for forward-looking regime information; historical explanations alone are not sufficient
when forecast-window evidence may still exist. Never invent dates or magnitudes.
```

The actual runtime prompt also requests a strict JSON object containing the query, selected document
IDs, exact-quote evidence, typed impacts, and a sufficiency decision. Host-side code checks that every
quote literally occurs in the cited document before the evidence can affect a candidate.

## Decision Agent prompt

```text
You are the Decision Agent in a contextual time-series forecasting system. Read candidates.json and
evidence.json. Select exactly one existing candidate_id; never invent or directly edit forecast values.
Rolling-validation scores measure historical numerical fit, while verified evidence tests whether a
candidate's assumptions remain valid in the future. Prefer a candidate only when both support it.
If evidence only explains historical anomalies and no evidence addresses the forecast window, request
more retrieval. If active evidence has no executable candidate, request new candidates.
```

The host rejects unknown or evidence-incompatible candidate IDs and computes the final trajectory from
the selected stored candidate. Ground-truth future values, `gt_evidence`, document `role`, and document
`subtype` are not included in any Codex workspace.

## Relation to Dr-CiK

Dr-CiK is primarily a benchmark, not this three-agent implementation. Its task is summarized as:

```text
document corpus -> retrieve -> filter distractors -> distill evidence -> forecast
```

It reports three complementary conditions:

1. End-to-end / no supplied context: forecast from the observed series.
2. Deep-research context: a research agent retrieves and synthesizes evidence from the corpus, then a forecaster consumes it.
3. Original or ground-truth context: the forecaster receives the benchmark's supporting context, measuring the downstream forecasting ceiling when retrieval is bypassed.

The Codex triad adds an explicit numbers-only hypothesis generator and an explicit candidate-selection
decision after retrieval. Its purpose is to test not only whether Codex finds the right text, but whether
that text leads to a falsifiable numerical program that improves the forecast.
