# How dr-cik works

A walkthrough of the system's mechanics: what runs, in what order, and why. For which parts
are faithful to the Dr-CiK paper versus our own reconstruction, see
[METHODS.md](METHODS.md). For measured numbers, see [RESULTS.md](RESULTS.md).

## The two pipelines

Dr-CiK asks two separate questions, and this package answers them with two separate
pipelines that share a task format but never share code paths:

```
dr-cik run            agent (OpenDR|DRBench) -> evidence  +  Chronos -> numbers
dr-cik direct-prompt   evidence (from a prior `run`) -> an LLM -> numbers directly
```

`run` measures **retrieval**: did the agent find the right evidence in a corpus full of
distractors? `direct-prompt` measures a **different forecaster**: can an LLM, reading that
evidence as text, produce good numbers without a numeric foundation model at all?
`direct-prompt` depends on `run` having already produced a `run_report.jsonl` to read
context from (`--from-run-dir`) — it is always the second step, never standalone.

## Leakage safety, enforced structurally

An agent must never see the labels it will be scored against. This isn't a convention an
agent could accidentally violate — it's structural. `models.py` defines two families of
type:

```
Document / ForecastTask        full labeled records — loaders and scorers only
AgentDocument / TaskView       what an agent is allowed to see — id/text only, no role,
                                no subtype, no future_values, no gt_evidence
```

`ForecastTask.agent_view()` and `Document.agent_view()` produce the stripped versions.
Because these are genuinely different dataclasses (not the same class with fields
nulled-out), an agent reaching for `.future_values` or `.role` gets an `AttributeError`,
not a silent `None`. `tests/test_models_leakage.py` backs this with an AST scan asserting
that no inference-time module (agents, retrieval, both forecasters, both LLM clients) so
much as *names* `Document` or `ForecastTask`.

## Retrieval: `retrieval/`

Both agents search a per-task corpus (`TaskView.documents`, 30-74 documents, mean ~37) via
one shared interface, `build_index(documents, retriever="bm25"|"dense")`:

- **`bm25.py`** (default) — classic lexical BM25, dependency-free, scores only chunks
  sharing at least one query term.
- **`dense.py`** — SentenceTransformer embeddings + cosine similarity, ported from the real
  `ServiceNow/drbench`'s own retriever (see METHODS.md). Measured *worse* than BM25 on this
  benchmark's invented-proper-noun corpus, which is why BM25 stays the default.

Both return `top_k` `(Chunk, score)` pairs; agents dedupe by `document_id` since a
document can produce several chunks. `top_k` is always a fraction of the corpus (default
16, see `--drbench-top-k` / `OpenDRConfig.max_search_results`) — the point is that
retrieval quality matters, not that everything gets read.

## Agents: `agents/`

Both take an `LLMClient` and a `TaskView`, and return an `AgentResult` (`report`,
`steps` — a full audit trail, `stop_reason`, `llm_call_count`).

**`opendr.py` — plan, then a ReAct search/finish loop:**

```
1. plan          1 LLM call  -> sub-questions + initial search queries
2. react loop     up to max_steps turns, each either:
                    {"action": "search", "args": {"query": ...}}   -> Python runs BM25/dense search
                    {"action": "finish", "args": {"report", "evidence"}}  -> done
3. fallback       step budget exhausted -> one forced "finish now" call;
                   two consecutive unparseable turns -> degraded report, no fabricated evidence
```

The model only ever *decides* to search; Python executes the actual retrieval. Repeated
identical queries get redirected instead of re-searched.

**`drbench.py` — a deterministic, three-stage cascade, no branching:**

```
1. search         0 LLM calls   query built from target/entity/trend -> top_k retrieval
2. brief          1 LLM call per retrieved document -> {"relevant": bool, "brief", "key_claims"}
                   (the prompt includes the task brief, not just the raw document text —
                   without it the model has nothing to judge relevance against)
3. synthesize     1 LLM call, given only the *relevant* briefs -> final report + evidence
```

Call budget is bounded and known in advance (`top_k + 1`), unlike OpenDR's open-ended loop.

Both agents share `agents/common.py`: the system preamble, `render_task_brief()` (which
exposes document *ids* and counts but never document *text* — an agent must actually
search to read anything), and `parse_evidence_list()` (drops citations to document ids
outside the corpus).

## Forecasters: `forecasters/`

**`chronos.py`** — Amazon's Chronos, zero-shot, text-blind by construction (it only ever
sees `TaskView.history_values`). Two API branches depending on checkpoint family: classic
`chronos-t5-*` samples natively (tagged `mc-samples`); the default `chronos-bolt-base` is
quantile-regression only, so pseudo-samples are built from evenly spaced quantile
trajectories (tagged `quantile-ensemble`) — the two are never silently mixed in
`Forecast.method`.

**`direct_prompt.py`** — an LLM forecasts the numbers itself, reading history *and*
research context (no numeric model at all). Samples **S independent temperature-sampled
calls, one trajectory per call** — this matches the literature's actual protocol
(Williams et al. 2025, cited by Dr-CiK for Direct-Prompt) and replaced an earlier, buggier
design that asked one call to invent all S arrays at once.

```
forecast(task_view, context_text):
  budget = horizon * tokens_per_number + json_overhead   (+ thinking budget if enabled)
  rows = S parallel sampled calls, each asked for one {"forecast": [v1..vH]}
  missing rows -> one retry pass with a stricter "JSON only" reminder
  still missing -> pad by resampling from the valid rows (tagged :padded(...))
  zero valid rows -> last-value persistence + volatility-scaled jitter (tagged :degraded-fallback)
```

`enable_thinking` (Qwen3+ reasoning) exists as a config knob but defaults off: measured
live, Qwen3.5-4B did not reliably reach a JSON answer even inside a 3584-token reasoning
budget, so turning it on costs a lot of wall-clock for an unproven accuracy gain. JSON
extraction (`llm.py:parse_json_object`) is robust either way — it strips a closed
`<think>...</think>` block if present, and falls back to slicing the outermost `{...}`
span if a response has other unmarked prose (reasoning or otherwise) around the JSON.

## Evaluation: `evaluation.py`

Forecast accuracy — `smae`, `srmse`, `scrps` — transcribed from the paper's Appendix H,
winsorized at 5.0 per task. Retrieval quality — `supp_doc_recall`, `distractor_avoidance`
— pure document-id set comparisons against `ForecastTask.documents[].role` (one of the few
places allowed to import the labeled types, since it's post-hoc scoring). `evidence_recall`
is an LLM-judge proxy (the paper's official scorer is private) and is always labelled as
such in `summary.json`.

## Orchestration: `pipeline.py`

`DrCikPipeline.run_many(tasks)` drives the `run` command: agent -> forecaster -> metrics,
per task, writing `forecasts.jsonl` / `deep_research.jsonl` / `run_report.jsonl` /
`summary.json`. `run_direct_prompt(...)` is the parallel entrypoint for the `direct-prompt`
command. Both accept an optional `plot_dir`, writing each task's PNG right after its own
forecast — so a long dev-split run's plots accumulate as it goes rather than only
appearing once the whole run finishes.

A fallback forecast is never presented as a real one: `Forecast.method` always carries a
`:degraded-fallback` or `:padded(...)` suffix when something went wrong, and
`summary.json`'s `methods` field lists every distinct method string that appears in a run
— check it before trusting any Direct-Prompt number.

## Logging

Every stage logs through the standard `logging` module (`logging.getLogger(__name__)` per
file), configured once in `cli.py`. `--log-level` (default `INFO`) shows task/step progress
— a hashed banner per task, agent steps by name, retrieved document ids, generation timing
— `DEBUG` additionally shows prompt/response text. `--log-file` defaults to
`./logs/<output-dir-name>.log`, so console and file always carry the same content.

## Running it

```
dr-cik run --agent drbench --llm-backend qwen --data-dir .../Dr-CiK \
  --split public-dev --output-dir results/drbench-dev --plot

dr-cik direct-prompt --data-dir .../Dr-CiK --split public-dev \
  --from-run-dir results/drbench-dev --model-id Qwen/Qwen3.5-9B \
  --output-dir results/dp-qwen3.5-9b-dev --plot

dr-cik plot-compare --data-dir .../Dr-CiK --split public-dev \
  --series "Chronos=results/drbench-dev/forecasts.jsonl" \
  --series "Qwen3.5-9B=results/dp-qwen3.5-9b-dev/forecasts.jsonl" \
  --output-dir results/compare
```

See `dr-cik <command> --help` for the full flag list; the README's Quick Start covers the
3-task sample bundle, which needs no download.
