# Dr-CiK Last-Mile Forecasting Agent

Repository: <https://github.com/kaichen-z/time-series>

This repository contains a dependency-free, auditable agent loop for the
[Dr-CiK](https://github.com/ServiceNow/Dr-CiK) contextual time-series forecasting
benchmark. The forecasting architecture now follows the central idea of
[Bridging the Last Mile of Time Series Forecasting with LLM Agents](https://arxiv.org/pdf/2606.02497):
a numerical forecasting backbone creates an immutable baseline, while an agent may
revise only the future forecast through a small, evidence-backed action language.

The paper's forecast workspace and restricted actions are combined with this project's
existing iterative Dr-CiK retrieval, evidence verification, and evidence-to-impact
translation. All components remain deterministic so every retrieval decision and
numerical revision can be inspected before substituting LLM or TSFM components.

## System flow

```text
historical values + noisy document corpus
          |
          +--> TS diagnosis --> forecast backbone --> immutable y_baseline
          |                                           copy to y_final
          |
          +--> Query Planner --> BM25 Retrieval --> Evidence Verifier
                                      ^                    |
                                      |                    v
                              persistent belief <-- Belief Updater
                                                           |
                                                           v
                                                  Evidence-to-Impact
                                                           |
                                                           v
                                                Structured Proposals
                                                           |
                                                           v
                              Forecast Workspace + Restricted Action Executor
                              preserve | multiply | add | clip | point override
                                                           |
                                                           v
                                            y_final + uncertainty samples
                                                           |
                                                           v
                                              Forecast Critic --> next query/stop

actual outcomes (after resolution only) --> post-hoc memory --> later tasks
```

The baseline is generated exactly once. Retrieval cannot rewrite historical values or
`y_baseline`; it can only propose changes to `y_final`. This makes it possible to ask
whether context improved the forecast instead of hiding the backbone and the contextual
revision inside one opaque prompt.

Each loop iteration asks one forecast-relevant question:

1. What caused historical anomalies or regime changes?
2. Was the disruption resolved, and was its effect temporary or permanent?
3. Which external events changed the target and over what time window?
4. Which trend or seasonal regime should govern the forecast horizon?

The verifier checks entity identity, temporal alignment, target relevance, whether the
document answers the current question, and whether textual claims conflict with the
observed numerical pattern. A document rejected only because it answers a different
question remains available to later iterations.

Accepted evidence is translated into structured forecast impacts containing the event
window, direction, permanence, forecast-horizon overlap, magnitude, confidence, and an
auditable adjustment rule. Explicit effects such as `increase by 20%` or `2 times the
usual demand` change the affected forecast steps directly. An event that ended before
the horizon produces a `return_to_baseline` instruction and is not extrapolated. A
future directional claim without a magnitude receives only a conservative
quarter-residual adjustment, rather than an invented large effect.

Each impact then becomes a proposal with an event type, affected range, action type,
value, source documents, confidence, rationale, and any retrieved memory IDs. The
workspace executor accepts only these actions:

| Action | Meaning |
|---|---|
| `preserve` | Keep the baseline when an event ended, is already reflected in history, or lacks a defensible magnitude |
| `multiply` | Apply an explicit multiplier or percentage to an evidence-supported range |
| `add` | Apply an explicit absolute change or a conservative residual-scaled change |
| `clip` | Enforce an explicit lower or upper bound |
| `override` | Revise a specific point when verified context provides an explicit future value |

Unsupported edits, out-of-horizon ranges, low-confidence changes, unsafe multipliers,
and duplicate actions are rejected and recorded. Corroborating documents therefore do
not multiply the same event effect twice.

Dr-CiK's `role`, `subtype`, `future_values`, and `gt_evidence` fields are never exposed
to the inference loop. Public labels are used only after a run to calculate development
metrics.

Post-hoc memory follows the same separation. A run never reads its own future values.
Only after an outcome is explicitly recorded can the system compare `y_baseline`,
`y_final`, and the actual series, store whether a revision helped, and use that lesson
as a shrinkage prior for later matching events.

## Quick start

Clone Dr-CiK and run its three official sample tasks:

```bash
git clone https://github.com/ServiceNow/Dr-CiK.git external/Dr-CiK
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

The same run works without installation:

```bash
PYTHONPATH=src python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

Useful loop controls:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop \
  --system iterative \
  --max-steps 10 \
  --top-k 5 \
  --max-no-progress 4 \
  --convergence-tolerance 0.002
```

Optional outcome memory for sequential research runs:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/with-memory \
  --memory-file outputs/forecast-memory.jsonl \
  --learn-from-public-outcomes
```

`--learn-from-public-outcomes` is intentionally opt-in and is rejected for the hidden
test split. For a clean benchmark comparison, keep it off and record outcomes only in a
separate, chronologically valid backtest.

`--top-k` is the number of new documents inspected at each iteration. The original
one-pass baseline is preserved for ablations:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/one-pass \
  --system one-pass \
  --top-k 8
```

## Outputs

- `forecasts.jsonl`: Dr-CiK forecasting submission format with 100 trajectories per task.
- `deep_research.jsonl`: accepted document IDs and extracted evidence.
- `loop_trace.jsonl`: every query, candidate, verifier verdict, structured evidence
  impact, revision proposal, accepted/rejected action, belief update, forecast summary,
  and stop decision.
- `run_report.jsonl`: per-task diagnosis, belief state, development metrics, and the full
  forecast workspace containing historical observations, immutable `y_baseline`, editable
  `y_final`, proposals, action results, and memory references.
- `summary.json`: aggregate development metrics.

Local `sMAE`, `sRMSE`, and `sCRPS` values are explicitly development proxies. Official
hidden-test scores are calculated by the Dr-CiK maintainers. When a workspace is used,
the report also includes `baseline_mae`, `revision_value_mae`,
`relative_revision_gain`, and `harmful_revision` to measure whether the last-mile agent
actually improved the forecasting backbone.

## Public development split

```bash
pip install -e '.[huggingface]'
drcik-agent run-hf --public-dev --output-dir outputs/public-dev
```

Use `--limit 5` for a short development run. `--hidden-test` creates submission files
without local forecast scores because the hidden labels are unavailable.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Current scope

| Component | Current implementation | Natural next experiment |
|---|---|---|
| Diagnosis | Trend, robust residual scale, conservative seasonality inference | Specialized TSFM diagnostics |
| Planning | Four explicit unresolved information needs | LLM query planner |
| Retrieval | Iterative BM25 with per-question document memory | Agentic multi-hop or hybrid retrieval |
| Verification | Entity/time/question/numerical consistency checks | LLM verifier plus cross-document corroboration |
| Working memory | Structured belief state plus unified forecast workspace | BLF-style linguistic probability beliefs |
| Evidence impact | Event window, direction, permanence, explicit magnitude, and conservative fallback | LLM causal-impact estimator with calibrated uncertainty |
| Forecast backbone | Drifted seasonal or trend baseline with uncertainty samples | TimesFM/Chronos/Nexus macro-micro backbone |
| Last-mile revision | Immutable baseline, structured proposals, restricted actions, duplicate and safety checks | Learned proposal ranking and counterfactual revision scoring |
| Outcome memory | Optional post-resolution calibration lessons in JSONL | Event embeddings and leakage-safe chronological retrieval |
| Control | Convergence, no-progress, exhaustion, and step-budget stopping | CORAL-style reflection and strategy evolution |

The first controlled comparison should be `backbone only` vs. `oracle context` vs.
`one-pass retrieval` vs. `iterative retrieval + unrestricted revision` vs. `iterative
retrieval + restricted workspace revision`, all using the same backbone. Report both
forecast accuracy and revision value (`baseline error - final error`) so retrieval gains
are separated from backbone quality and harmful context edits.
