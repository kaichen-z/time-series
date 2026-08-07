# dr-cik

An independent reproduction of baselines from the **Dr-CiK** benchmark
([arXiv 2605.27904](https://arxiv.org/abs/2605.27904), ServiceNow) — contextual time-series
forecasting over a noisy document corpus.

Dr-CiK asks two questions separately, and so does this package:

| Question | Answered by | Scored by |
|---|---|---|
| Did we find the right evidence in a corpus full of distractors? | `agents/` — OpenDR, DRBench | EvidenceRecall, SuppDocRecall, DistractorAvoidance |
| Did we produce good numbers? | `forecasters/` — Chronos, Direct-Prompt | sMAE, sRMSE, sCRPS |

This is a **baseline reproduction**, deliberately standalone (no imports from the parent
repo's `src/drcik_agent/`), so its numbers stay comparable to the paper's own table rather
than entangled with in-house method changes.

## Install

```bash
cd dr-cik
pip install -e '.[chronos,gemini,qwen,plots,dev]'
```

Extras are separable on purpose: `chronos` (numeric forecaster), `gemini` (hosted LLM),
`qwen` (local LLM + GPU), `plots` (matplotlib), `dev` (pytest). The offline test suite
needs none of the model extras.

Put credentials in a `.env` at the repo root (gitignored; see `.env.example`):

```
GEMINI_API_KEY=...   # only needed for --llm-backend gemini
```

## Quick start

The official 3-task sample bundle needs no download and is the fastest way to see the
whole pipeline work:

```bash
SAMPLE=/raid/home/air/khoutaibi/external/Dr-CiK/sample

# 1. Deep-research agent + Chronos forecaster
dr-cik run --agent drbench --llm-backend qwen \
  --sample-dir $SAMPLE --output-dir outputs/drbench-sample

# 2. Direct-Prompt baseline, reusing step 1's evidence as context
dr-cik direct-prompt --sample-dir $SAMPLE \
  --from-run-dir outputs/drbench-sample \
  --model-id Qwen/Qwen3.5-4B \
  --output-dir outputs/dp-qwen3.5-4b-sample

# 3. Overlay both on the same axes
dr-cik plot-compare --sample-dir $SAMPLE \
  --series "Chronos=outputs/drbench-sample/forecasts.jsonl" \
  --series "Qwen3.5-4B=outputs/dp-qwen3.5-4b-sample/forecasts.jsonl" \
  --output-dir outputs/compare
```

## Full dataset

```bash
dr-cik download-data     # ~50 MB -> /raid/home/air/khoutaibi/time_series_dataset/Dr-CiK
dr-cik download-models   # Chronos checkpoint -> /raid/home/air/khoutaibi/models

dr-cik run --agent drbench --llm-backend qwen \
  --data-dir /raid/home/air/khoutaibi/time_series_dataset/Dr-CiK \
  --split public-dev --output-dir outputs/drbench-dev
```

279 tasks total: 199 `public-dev` (labelled, scored locally) and 80 `hidden-test`
(unlabelled — ground-truth metrics come back `null`, by design). `--limit N` and
repeatable `--task-id ID` work on any split.

## Repository map

```
src/dr_cik/
├── models.py         Core types. Enforces the leakage split (see below).
├── data.py           Loaders: official sample dir, and the full HF dataset.
├── retrieval.py      Dependency-free BM25 over a task's corpus.
├── llm.py            LLMClient protocol + GeminiClient + FakeLLMClient (tests).
├── local_llm.py      QwenClient: local GPU inference, batched sampling.
├── agents/           Evidence producers.
│   ├── opendr.py     Plan -> ReAct search/finish loop -> report.
│   └── drbench.py    Search -> per-document brief -> synthesize.
├── forecasters/      Number producers.
│   ├── chronos.py    Numeric foundation model, zero-shot. Text-blind.
│   └── direct_prompt.py  An LLM writing the numbers itself, from text context.
├── evaluation.py     The paper's metrics, plus an LLM-judge proxy.
├── pipeline.py       Orchestration + submission-shaped output writing.
├── plotting.py       Per-run and multi-run comparison figures.
└── cli.py            The `dr-cik` command.
```

Outputs per run: `forecasts.jsonl` and `deep_research.jsonl` (the exact shapes
`SUBMISSION.md` requires), plus `run_report.jsonl` (full agent trace, for auditing what
the agent actually did) and `summary.json` (mean metrics).

## Leakage safety

An agent must never see the labels it is being scored against. That is enforced
*structurally*, not by convention: `TaskView`/`AgentDocument` are separate dataclasses
from `ForecastTask`/`Document`, so `future_values`, `gt_evidence`, `role`, and `subtype`
are not nulled-out fields an agent could read — they do not exist on the objects agents
receive, and touching them is an `AttributeError`.

`tests/test_models_leakage.py` backs this with an AST scan asserting that no
inference-time module (every agent, both forecasters, retrieval, both LLM clients) so much
as *names* the labelled types. It runs on every `pytest` invocation.

## Results

See **[docs/RESULTS.md](docs/RESULTS.md)** for measured numbers (`outputs/` is gitignored,
so results are recorded there rather than only on disk).

## What is faithful to the paper, and what is not

Some parts are the paper's, some are our reconstruction, and the difference matters when
citing these numbers. **[docs/METHODS.md](docs/METHODS.md)** states exactly which is which.

## Tests

```bash
pytest tests/          # offline: no API key, no GPU, no downloads
```

Every external dependency — Chronos, Gemini, Qwen, matplotlib — is faked in the offline
suite, including both branches of Chronos's two different prediction APIs. Tests marked
`live` (real network/GPU) are excluded by default.
