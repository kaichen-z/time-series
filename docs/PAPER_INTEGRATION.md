# Paper-to-System Integration

This document records the design boundary of the Dr-CiK system. The online loop is
leakage-safe: `future_values`, `gt_evidence`, document `role`, and document `subtype`
never enter retrieval, grounding, or revision. Resolved outcomes are available only to
offline training and post-hoc chronological memory.

The research target is **foresight-driven retrieval**: rank evidence by its expected
downstream forecast value, not just by text similarity.

## Online inference architecture

### 1. Numerical prior and forecast workspace

Sources: *PostTime* and *Bridging the Last Mile of Time Series Forecasting with LLM
Agents*.

Chronos-Bolt generates `y_baseline` exactly once from historical values. The baseline is
stored as an immutable tuple. Contextual reasoning modifies only `y_final` through
validated `preserve`, `multiply`, `add`, `clip`, or `override` actions. Every accepted or
rejected action is written to the audit trace.

### 2. Structured sufficiency and gap controller

Sources: S2G-RAG and ReflectiveRAG.

The controller maintains explicit `ForecastGap` objects and returns a
`SufficiencyDecision` before every retrieval:

```text
sufficient
resolved_gap_ids
unresolved_gap_ids
selected_gap_id
next_query
expected_information_gain
stop_reason
```

Initial gaps cover historical regime, future drivers, and future regime. Follow-up gaps
are evidence-dependent: anomaly evidence creates a resolution/recurrence gap, while an
unquantified future event creates an effect-magnitude gap. This replaces the previous
fixed four-query schedule.

The current judge is deterministic. Its interface is intended for a distilled small
model trained from chronological retrieval traces.

### 3. Forecast-utility retrieval

Sources: Agentic-R and *From Long News to Accurate Forecast*.

BM25 generates a candidate pool. `ForecastUtilityRetriever` then combines lexical
retrieval with gap alignment, entity/target relevance, causal content, temporal
alignment, novelty, redundancy, and token cost. The default scorer is explicitly named
`label_free_proxy`.

The retriever accepts an injected `ForecastUtilityScorer`. A learned scorer should be
trained offline with labels of the form:

```text
forecast_gain = error_before_document - error_after_document

net_utility = forecast_gain
              - latency_cost
              - redundancy_cost
              - token_cost
```

`ForecastUtilityLabeler` implements this leakage-sensitive label construction but is not
imported by the online loop.

### 4. Grounded Evidence State

Sources: S2G-RAG, ReflectiveRAG, and BLF.

Accepted documents are reduced to sentence-level claims and stored with document IDs,
gap IDs, entity, target variable, publication date, occurrence dates, direction,
magnitude, persistence, verbatim evidence quote, and confidence. Hard gates reject
post-cutoff publication, wrong entity, wrong target, and contradictions with strong
numerical seasonality. Raw documents do not accumulate in the belief state.

The BLF-inspired linguistic belief records compact supporting and counter-evidence for
each gap. It is an evidence-sufficiency state, not a binary forecast probability.

### 5. Importance-aware context and macro/micro outlooks

Sources: *From Long News to Accurate Forecast* and NEXUS.

The context module allocates a shared character budget according to retrieval utility
and prioritizes sentences containing the entity, target, causal language, dates,
quantified effects, and forecast-horizon information. NEXUS-style macro and micro views
remain separate structured inputs:

- macro: numerical trend, baseline model, seasonality, confidence;
- micro: local events, dates, direction, permanence, magnitude type, sources.

They do not independently emit competing full forecasts.

### 6. Contextual revision and restricted execution

Sources: PostTime and Last-Mile Forecasting.

The current deterministic revision policy decides whether each proposal has enough
support to revise or preserve the Chronos prior. It is a baseline—not a trained PostTime
model. The target replacement is a compact reviser trained with verified forecast-time
traces and a reward relative to the selected numerical baseline.

Regardless of the reviser implementation, the Last-Mile executor remains the final
safety boundary.

## Offline-only architecture

### Forecast-utility retriever training

For each resolved training task, run controlled counterfactual retrieval ablations:

```text
forecast without document d -> error_before
forecast with document d    -> error_after
```

Use positive-gain passages as positives. Passages with high semantic similarity but
negative forecast gain are hard negatives. Following Agentic-R, later iterations should
co-train the query-producing controller and retriever using improved multi-turn traces.

### PostTime-style reviser training

Generate several forecast-time revision candidates without showing the generator the
future target. Use the resolved future only to retain candidates that improve the
Chronos prior; insert preserve targets when none improve. SFT supplies the initial
revision policy, followed by baseline-relative RLVR.

### Continuous uncertainty and calibration

BLF's logit averaging and Platt scaling are binary-specific and are not applied to a
continuous trajectory. The target design runs independent revision trials, aggregates
trajectories or quantiles, and uses horizon/domain-aware conformal or quantile
calibration. Dr-CiK sCRPS is the primary probabilistic metric.

### CORAL-style evolution

CORAL belongs outside hidden-test inference. Isolated agents may propose controller,
retriever, grounding, compression, or reviser changes on development tasks. A separate
evaluator accepts changes only when held-out forecast, retrieval, and cost metrics
improve. Accepted lessons enter shared strategy memory; one frozen policy is deployed to
test.

## Required experiment matrix

Keep splits chronological and compare both retrieval and forecasting components:

| Context source | Forecast policy |
|---|---|
| none | Chronos only |
| oracle supporting evidence | Chronos + reviser |
| one-pass BM25 | Chronos + reviser |
| gap controller + BM25 | Chronos + reviser |
| gap controller + learned forecast-utility retriever | Chronos + reviser |
| utility retriever + importance compression | trained PostTime-style reviser |

Report:

- retrieval: supporting recall/precision, distractor citation rate, gap coverage,
  forecast-utility ranking quality;
- forecast: sCRPS, MASE, MAE/RMSE, interval coverage, harmful-revision rate;
- system: turns, documents inspected, tokens, latency, fallback rate, invalid actions.

Oracle context is essential: if oracle evidence does not improve the baseline, the
bottleneck is the reviser rather than retrieval.

## Current implementation status

Implemented now:

- structured gaps and sufficiency traces;
- gap-derived queries and dynamic follow-up gaps;
- injectable forecast-utility scorer plus transparent proxy;
- offline forecast-utility label construction;
- sentence-level grounded evidence schema;
- importance-aware compression;
- Chronos immutable prior and Last-Mile workspace;
- deterministic macro/micro reasoning and revise/preserve baseline;
- chronological post-outcome memory.

Not yet implemented or trained:

- dense retriever and learned forecast-utility checkpoint;
- distilled S2G-style controller;
- LLM entailment verifier;
- PostTime SFT/RLVR reviser;
- continuous multi-trial calibration;
- autonomous CORAL development runner.
