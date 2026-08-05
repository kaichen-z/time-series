# Foresight-Driven Retrieval for Time-Series Forecasting

Repository: <https://github.com/kaichen-z/time-series>

This repository contains an auditable, gap-guided agent loop for the
[Dr-CiK](https://github.com/ServiceNow/Dr-CiK) contextual time-series forecasting
benchmark. Its research hypothesis is that a passage should be retrieved because it is
expected to improve the downstream forecast—not merely because it is lexically similar
to a query. A numerical forecasting backbone creates an immutable baseline; a structured
retrieval controller fills explicit information gaps; and a reviser may change only the
future forecast through a small, evidence-backed action language.

The numerical backbone is now Google Research TimesFM 2.5 by default. The paper's
forecast workspace and restricted actions are combined with this project's
structured gap judging, forecast-utility retrieval, evidence grounding, and
evidence-to-impact translation. The current runtime remains deterministic and exposes
interfaces for learned retrievers, judges, and PostTime-style revisers. It does not claim
that its label-free utility proxy is an already trained PRM.

## System flow

```text
history ──> diagnosis ──> TimesFM 2.5 ──> immutable y_baseline
   │
documents ──> gap judge ──> next query ──> BM25 candidate pool
                                │                    │
                                │                    v
                                │        forecast-utility reranker
                                │                    │
                                │                    v
                                └──── insufficient ─ evidence grounding
                                                     │
                                                     v
                                       sentence-level Evidence State
                                                     │
                                          sufficient / missing gaps
                                                     │
                                                     v
                       importance compression ──> macro + micro outlook
                                                     │
                                                     v
                                      PostTime-style revise/preserve
                                                     │
                                                     v
                                     Last-Mile restricted workspace
                                                     │
                                                     v
                                    y_final + trajectories + audit trace

resolved training tasks ──> forecast-utility labels ──> frozen learned scorer
realized outcomes ──> post-hoc memory ──> later chronological tasks
```

The complete English architecture diagram, including the online inference path and
offline learning/evolution path, is available in two standalone formats:
[`docs/architecture.html`](docs/architecture.html) and
[`docs/architecture.svg`](docs/architecture.svg).

The TimesFM baseline is generated exactly once. Retrieval cannot rewrite historical values or
`y_baseline`; it can only propose changes to `y_final`. This makes it possible to ask
whether context improved the forecast instead of hiding the backbone and the contextual
revision inside one opaque prompt.

## Paper-derived design

The runtime is deliberately a synthesis rather than many complete frameworks stacked
together:

| Work | What is integrated now | What is not placed in online inference |
|---|---|---|
| [Last-Mile Forecasting](https://arxiv.org/pdf/2606.02497) | Immutable baseline, forecast workspace, evidence-backed restricted actions | None of its case-specific prompts are required |
| [PostTime](https://arxiv.org/pdf/2605.29401) | LLM-as-reviser role, explicit revise-or-preserve gate, improvement-over-baseline metrics, hard-case fallback behavior | SFT/RLVR weight training requires a separate training corpus and GPUs |
| [From Long News to Accurate Forecast](https://arxiv.org/abs/2606.03097) | Candidate-pool utility reranking and forecast-aware long-document compression | The current scorer is a frozen label-free proxy; learned RM/PRM training is an offline next step |
| [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/) | Explicit evidence sufficiency, structured missing gaps, and gap-guided queries | Its QA-specific gap labels are replaced with forecast gaps |
| [ReflectiveRAG](https://aclanthology.org/2026.eacl-industry.27/) | Adaptive stopping and relevance-minus-redundancy filtering | Its QA answer controller is not reused as a forecast judge |
| [Agentic-R](https://aclanthology.org/2026.findings-acl.785/) | Trainable retriever interface combining local relevance with global task utility | Global answer correctness becomes downstream forecast improvement |
| [BLF](https://arxiv.org/pdf/2604.18576) | Compact linguistic belief state updated in log-odds space instead of accumulating raw text | Binary Platt calibration and logit aggregation do not directly apply to continuous trajectories |
| [NEXUS](https://arxiv.org/pdf/2605.14389) | Separate macro numerical outlook and micro event outlook before final synthesis | LLM prompts can replace the deterministic agents after controlled ablations |
| [CORAL](https://arxiv.org/pdf/2604.01658) | Shared persistent artifacts and evaluator separation inform the architecture | Long-running autonomous evolution belongs outside task inference to prevent leakage and uncontrolled benchmark search |

See [`docs/PAPER_INTEGRATION.md`](docs/PAPER_INTEGRATION.md) for the module mapping and
recommended ablations, and [`docs/TIMESFM_BACKBONE.md`](docs/TIMESFM_BACKBONE.md) for
backbone configuration.

Each loop iteration begins with a structured sufficiency decision. The controller
selects the highest-value unresolved gap, constructs the next query, and can create new
follow-up gaps from grounded evidence. For example, finding an anomaly dynamically
creates a `resolution_permanence` gap; finding an unquantified future event creates an
`event_magnitude` gap. The loop stops when gaps are resolved, the corpus is exhausted,
progress stalls, or expected information gain falls below the configured cost threshold.

The verifier checks entity identity, temporal alignment, target relevance, whether the
document answers the current question, and whether textual claims conflict with the
observed numerical pattern. A document rejected only because it answers a different
question remains available to later iterations.

Accepted evidence is translated into structured forecast impacts containing the event
window, direction, permanence, forecast-horizon overlap, magnitude, confidence, and an
auditable adjustment rule. Explicit effects such as `increase by 20%` or `2 times the
usual demand` change the affected forecast steps directly. An event that ended before
the horizon produces a `return_to_baseline` instruction and is not extrapolated. A
future directional claim without a magnitude produces a conservative candidate, but the
revision utility gate normally falls back to the numerical prior unless other evidence
raises its predicted utility.

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
pip install -e '.[timesfm]'

drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

The project entrypoint can run directly from source after installing the TimesFM runtime:

```bash
pip install 'timesfm[torch]>=2.0.2'
PYTHONPATH=src python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

The package and the `google/timesfm-2.5-200m-pytorch` checkpoint are loaded lazily on
the first forecast. `timesfm[torch]` installs PyTorch; the checkpoint is downloaded from
Hugging Face unless it is already cached.

Useful loop controls:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop \
  --system iterative \
  --max-steps 10 \
  --top-k 5 \
  --max-no-progress 4 \
  --convergence-tolerance 0.002 \
  --candidate-multiplier 3 \
  --context-character-budget 12000 \
  --min-information-gain 0.05 \
  --revision-threshold 0.60
```

TimesFM controls:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --backbone timesfm \
  --timesfm-model-id google/timesfm-2.5-200m-pytorch \
  --timesfm-max-context 4096 \
  --timesfm-max-horizon 1024 \
  --timesfm-cache-dir outputs/model-cache
```

TimesFM is the default and failure is explicit. The system does not silently switch to
the old statistical model. For an intentional degraded-mode run, add
`--allow-statistical-fallback`; the recorded baseline method will begin with
`statistical_fallback:`. For a clean ablation, use `--backbone statistical`.

`--top-k 5 --candidate-multiplier 3` retrieves 15 candidates, scores all 15 for
forecasting utility, and sends only the best 5 to the verifier. The ranking and
compression modules never see `future_values`.

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
  utility score, compression decision, macro/micro outlook, revision decision,
  accepted/rejected action, belief update, forecast summary, and stop decision.
- `run_report.jsonl`: per-task diagnosis, belief state, development metrics, and the full
  forecast workspace containing historical observations, immutable `y_baseline`, editable
  `y_final`, proposals, action results, and memory references.
- `summary.json`: aggregate development metrics.

Local `sMAE`, `sRMSE`, and `sCRPS` values are explicitly development proxies. Official
hidden-test scores are calculated by the Dr-CiK maintainers. When a workspace is used,
the report also includes `baseline_mae`, `revision_value_mae`,
`relative_revision_gain`, and `harmful_revision` to measure whether the last-mile agent
actually improved the forecasting backbone. It also reports `revision_accept_rate`,
`revision_fallback_rate`, `mean_predicted_revision_utility`,
`context_retention_ratio`, `mean_belief_sufficiency`, `gap_coverage`,
`mean_expected_information_gain`, `retrieval_turns`, and `documents_inspected`.

## Public development split

```bash
pip install -e '.[timesfm,huggingface]'
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
| Control | Structured sufficiency decision, dynamic forecast gaps, gap-guided query, marginal-gain stopping | Distill a lightweight S2G-style judge from chronological traces |
| Retrieval | BM25 candidate pool plus injectable forecast-utility scorer; label-free proxy is the default ablation | Train the Agentic-R/long-news scorer on chronological forecast-gain labels |
| Grounding | Entity/time/target checks plus sentence-level claims with provenance, magnitude, and persistence | LLM entailment and cross-document corroboration |
| Context | Importance-aware sentence retention under a global character budget | Learned article reward model and pairwise fusion |
| Working memory | BLF-inspired linguistic belief state plus unified forecast workspace | Multi-trial continuous-trajectory aggregation |
| Evidence impact | Event window, direction, permanence, explicit magnitude, and conservative fallback | LLM causal-impact estimator with calibrated uncertainty |
| Reasoning | NEXUS-style macro numerical and micro event outlooks | LLM outlook agents with schema-constrained outputs |
| Forecast backbone | TimesFM 2.5 200M point forecast; statistical backbone retained only for explicit ablation/fallback | Use TimesFM quantiles directly for calibrated trajectory sampling |
| Last-mile revision | PostTime-style revise/preserve gate plus restricted workspace actions | Post-train a compact reviser with SFT and improvement-ratio RLVR |
| Outcome memory | Optional post-resolution calibration lessons in JSONL | Event embeddings and leakage-safe chronological retrieval |
| Offline evolution | Evaluator-separated memory and outcome label interfaces | CORAL-style policy evolution on isolated development runs |

The first controlled comparison should be `backbone only` vs. `oracle context` vs.
`one-pass retrieval` vs. `iterative retrieval + unrestricted revision` vs. `iterative
retrieval + restricted workspace revision`, all using the same backbone. Report both
forecast accuracy and revision value (`baseline error - final error`) so retrieval gains
are separated from backbone quality and harmful context edits.
