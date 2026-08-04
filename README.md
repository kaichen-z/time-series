# Dr-CiK Iterative Forecasting Agent

Repository: <https://github.com/kaichen-z/time-series>

This repository contains a dependency-free, auditable agent loop for the
[Dr-CiK](https://github.com/ServiceNow/Dr-CiK) contextual time-series forecasting
benchmark. It is a small research baseline: the components are deterministic so that
retrieval, verification, memory updates, forecasting, and stopping behavior can be
inspected before LLM-based components are introduced.

## System flow

```text
historical values + noisy document corpus
                  |
                  v
         Time-Series Diagnosis
                  |
                  v
     +------ persistent belief state <------------------+
     |                                                   |
     v                                                   |
Query Planner -> BM25 Retrieval -> Evidence Verifier     |
                                      |                  |
                                      v                  |
                               Belief Updater            |
                                      |                  |
                                      v                  |
                          Probabilistic Forecaster       |
                                      |                  |
                                      v                  |
                              Forecast Critic -----------+
                                      |
                              stop when resolved,
                              converged, exhausted,
                              or budget reached
```

Each loop iteration asks one forecast-relevant question:

1. What caused historical anomalies or regime changes?
2. Was the disruption resolved, and was its effect temporary or permanent?
3. Which external events changed the target and over what time window?
4. Which trend or seasonal regime should govern the forecast horizon?

The verifier checks entity identity, temporal alignment, target relevance, whether the
document answers the current question, and whether textual claims conflict with the
observed numerical pattern. A document rejected only because it answers a different
question remains available to later iterations.

Dr-CiK's `role`, `subtype`, `future_values`, and `gt_evidence` fields are never exposed
to the inference loop. Public labels are used only after a run to calculate development
metrics.

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
- `loop_trace.jsonl`: every query, candidate, verifier verdict, belief update, forecast
  summary, and stop decision.
- `run_report.jsonl`: per-task diagnosis, belief state, forecast, and development metrics.
- `summary.json`: aggregate development metrics.

Local `sMAE`, `sRMSE`, and `sCRPS` values are explicitly development proxies. Official
hidden-test scores are calculated by the Dr-CiK maintainers.

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
| Memory | Structured persistent belief state and audit trail | BLF-style linguistic probability beliefs |
| Forecast | Drifted seasonal or trend baseline with uncertainty samples | Nexus-style macro/micro synthesis |
| Control | Convergence, no-progress, exhaustion, and step-budget stopping | CORAL-style reflection and strategy evolution |

The first controlled comparison should be `no context` vs. `oracle context` vs.
`one-pass retrieval` vs. `iterative retrieval`, all using the same forecaster. This
separates retrieval gains from forecasting-model gains.
