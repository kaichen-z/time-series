# Paper-to-System Integration

This document records which ideas are used in the Dr-CiK system and where they belong.
The primary constraint is leakage-safe forecasting: task outcomes and Dr-CiK labels are
never available to an online agent.

## Selected online agents

### 1. Retrieval Process Reward Agent

Source: *From Long News to Accurate Forecast*.

The BM25 retriever first returns a larger candidate pool. A frozen utility ranker scores
each document using retrieval relevance, entity/target alignment, causal content,
temporal alignment, and novelty relative to the current linguistic belief. It then sends
only the highest-utility candidates to the verifier.

The current implementation is a deterministic proxy for a learned PRM. It is explicitly
named as a proxy because it has not been trained on forecast-error trajectories.

### 2. Importance-Aware Context Agent

Source: *From Long News to Accurate Forecast* and the NEXUS historical context agent.

After verification, the agent allocates a shared character budget across accepted
documents using their retrieval utility. Within each document it prioritizes sentences
that preserve the entity, target, causal language, dates, quantified effects, forecast
language, and relevant time-series regime information. The original document ID is
preserved for auditing.

### 3. Bayesian Linguistic Belief Updater

Source: BLF.

Every retrieval question owns a compact state:

- evidence-sufficiency probability;
- short supporting evidence summaries;
- short counterevidence/rejection summaries;
- update count.

The probability is updated in log-odds space after each verifier step. It represents
whether the information need is sufficiently resolved, not the probability of a binary
world event. Raw documents therefore do not accumulate indefinitely in the belief state.

### 4. Macro and Micro Reasoning Agents

Source: NEXUS.

The macro agent summarizes the numerical trajectory, slope, periodicity, baseline model,
and confidence. The micro agent summarizes localized events, their windows, directions,
permanence, explicit magnitude type, sources, and confidence. Both are structured inputs
to the revision gate and are saved in the forecast workspace.

### 5. Revision Utility Agent

Sources: PostTime and *Bridging the Last Mile of Time Series Forecasting with LLM Agents*.

The agent does not forecast from scratch. It evaluates whether a proposed edit has enough
evidence to improve a strong numerical prior. Its score uses evidence confidence, belief
sufficiency, source corroboration, magnitude specificity, horizon validity, prior memory,
and weak macro/micro conflicts. A proposal below the threshold becomes an explicit
`preserve` action rather than a guessed numerical intervention.

All accepted edits still pass through the restricted workspace executor.

## Offline-only ideas

### PostTime SFT and RLVR

The repository now produces the ingredients needed for future training: baseline,
context, reasoning state, revision proposal, revise/preserve decision, final forecast,
and improvement over baseline. Actual SFT/RLVR is not simulated by heuristics. It should
be trained separately using chronological splits and an improvement-ratio reward.

### Long-news reward model and PRM training

Historical resolved tasks can be used to label article retention utility and step-wise
retrieval gain. The resulting models must be frozen before test-time deployment. Ground
truth must never be used to rank documents for the same unresolved task.

### CORAL autonomous evolution

CORAL is appropriate for searching over agent policies, prompts, thresholds, query
strategies, and compression budgets on an isolated development environment. It is not an
online forecasting agent. A long-running CORAL loop on a benchmark task could repeatedly
optimize against its evaluator and compromise the intended forecast cutoff.

The safe future design is:

```text
development tasks -> isolated policy proposals -> evaluator -> shared strategy memory
                                                        |
                                                        v
                                      select and freeze one policy
                                                        |
                                                        v
                                              hidden-test inference
```

## Recommended ablations

Keep the numerical backbone fixed and compare:

1. backbone only;
2. BM25 retrieval without utility reranking;
3. utility reranking without compression;
4. reranking plus importance-aware compression;
5. add linguistic belief updates;
6. add macro/micro reasoning;
7. add the revise-or-preserve gate;
8. oracle context as an upper bound.

Report retrieval precision/recall, evidence recall, baseline MAE, final MAE, revision
gain, harmful-revision rate, fallback rate, context retention, and latency.
