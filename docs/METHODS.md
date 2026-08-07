# Methods: what is faithful, and what is our reconstruction

These numbers are only citable if it is clear which parts come from the Dr-CiK paper
(arXiv 2605.27904) and which are our own filling-in. This document draws that line.

## Taken from the paper

**Metrics.** The sMAE / sRMSE / sCRPS definitions in `evaluation.py` are transcribed from
Appendix H.2, including the scale normaliser `a = (1/T · Σ|y_t|)⁻¹` and the closed-form
CRPS identity:

```
CRPS_t = (1/S)Σ|ŷ_s,t − y_t| − (1/(2S²))ΣΣ|ŷ_s,t − ŷ_s',t|
```

**Agent descriptions.** OpenDR is described as decomposing research into "planning,
retrieval through ReAct-style tool calls, and report writing"; DRBench as a
"search–summarize–synthesize cascade" that "compresses retrieved documents into
per-document briefs." Our two agents implement those control flows.

**Forecaster choice.** Chronos is the paper's own pretrained-TSFM baseline, used zero-shot.
TimesFM does not appear in the paper and is deliberately not used here.

**Direct-Prompt sampling protocol.** S = 25 sampled trajectories per task, and the
strategy is cited to Williams et al. 2025 ("Context is Key," ICML) rather than defined by
Dr-CiK. We follow the cited protocol: **S independent temperature-sampled calls, one
trajectory each.**

**Context conditions.** The paper evaluates Direct Prompt under eight interchangeable
text-context conditions. We implement one: **DR-synthesized** — the report and evidence a
deep-research agent produced, loaded from a prior run via `--from-run-dir`.

## Our reconstruction (not verbatim)

**Agent prompts.** The paper describes control flow, not prompt text. Every prompt in
`agents/` and `forecasters/direct_prompt.py` is ours. Behaviour should be
directionally comparable; exact numbers will not be.

**OpenDR is reimplemented, not wrapped.** We do not depend on
`langchain-ai/open_deep_research`; we implement the described pattern in-house, so this is
a reproduction of the *method*, not of that repository's exact behaviour.

**Agent retrieval strategy.** The Dr-CiK paper does not pin a retriever for the agents'
internal search tool, so the choice is ours — but it is not arbitrary, see below.

**EvidenceRecall is a proxy.** The official scorer is private. `evaluation.py` approximates
it with our own LLM-judge prompt asking whether any predicted claim conveys each
ground-truth evidence item. It is labelled a proxy in every `summary.json` it appears in,
and it is not comparable to the leaderboard's EvidenceRecall column.

**Winsorisation.** `SUBMISSION.md` specifies winsorisation without a per-metric breakdown;
we apply a cap of 5.0 uniformly across sMAE/sRMSE/sCRPS. This is a documented reading, not
a quoted rule.

## Ported from the real DRBench implementation

Dr-CiK's baselines are unpublished, but the *upstream* system its DRBench baseline adapts
is public: [`ServiceNow/drbench`](https://github.com/ServiceNow/drbench) (Apache-2.0,
arXiv 2510.00172). Its agent retrieves with **dense embeddings**, not lexical matching.

`retrieval/dense.py` reproduces that method from
`drbench/agents/drbench_agent/vector_store.py`:

- **Encoder** — a SentenceTransformer, via their `_get_embeddings_local` path. We default
  to `all-MiniLM-L6-v2` so retrieval stays offline and key-free; their other default,
  `text-embedding-ada-002`, needs an OpenAI key.
- **Ranking** — cosine similarity then `argsort(similarities)[::-1][:top_k]`, matching
  their `semantic_search` exactly.
- **Threshold** — they apply a `0.7` similarity floor, tuned for ada-002 whose
  similarities sit high. MiniLM similarities spread much lower, so 0.7 would discard
  nearly every hit; we default to `0.0` and leave the knob exposed. **This is the one
  place we knowingly diverge from their algorithm**, and it is a consequence of the
  encoder swap, not a disagreement with the method.

Select it with `--retriever dense`. BM25 (`retrieval/bm25.py`) remains the default because
it needs no extra dependency and no model download; it is *our* choice, not the paper's or
DRBench's.

**What this does and does not buy.** It removes a method-level divergence, so retrieval
quality can be measured under the literature's own approach rather than only under ours.
Measured, dense retrieval turns out **worse** than BM25 on this benchmark (0.064 vs 0.149
SuppDocRecall — see [RESULTS.md](RESULTS.md)), most likely because Dr-CiK's invented proper
nouns favour lexical matching and because we substituted a small local encoder for
DRBench's `text-embedding-ada-002`. BM25 therefore remains the default, now on evidence.

It does **not** reproduce the paper's table. Dr-CiK's own OpenDR/DRBench adaptations — the
corpus-as-search-tool wiring, the prompts, the model configuration — are unpublished, and
`ServiceNow/drbench` is a different benchmark with its own task format that cannot be
installed here anyway (it requires Python >=3.12; this project targets >=3.10).

## Deliberate deviations

**Sample count differs between the file and the score.** `SUBMISSION.md` requires ≥100
samples in `forecasts.jsonl`; the paper's CRPS formula uses S = 25. Rather than silently
picking one, we do both: `--num-samples` (default 100) controls what is written to the
submission file, and `--crps-sample-size` (default 25) controls the local sCRPS proxy.

**LLM backend.** The paper runs its DR agents on Gemini-3 Flash. We support that
(`--llm-backend gemini`) but default our own runs to a local Qwen (`--llm-backend qwen`)
for cost and rate-limit reasons. The backend is a flag, not a hardcoded assumption.

**Chronos-Bolt has no native sampling.** The default checkpoint
(`amazon/chronos-bolt-base`) is a quantile-regression model. We build pseudo-samples from
evenly spaced quantile trajectories, with levels held inside [0.1, 0.9] because Bolt clamps
its extreme tails. This is tagged `quantile-ensemble` in `Forecast.method`, versus
`mc-samples` for classic `chronos-t5-*` checkpoints, so the two are never silently mixed.

**Model subset.** The paper's Direct-Prompt table includes Gemini-3.1-flash-lite,
Mistral-medium-3.1, Qwen-3.5 (4B/9B/27B), Llama-3.2-3B, and Phi-4-mini. We run
Qwen3.5-4B and Qwen3.5-9B. Mistral-medium-3.1 has no open weights (the paper reaches it
through OpenRouter) and is not implemented.

## Degradation is recorded, never hidden

An LLM asked for a numeric array does not always return one. When a response fails to
parse, the forecaster retries once for the missing count; if trajectories are still
missing it resamples from the valid ones, and if none parsed at all it falls back to
last-value persistence with volatility-scaled jitter.

**A fallback forecast is not a model forecast**, so it is never presented as one:
`Forecast.method` carries `:degraded-fallback` or `:padded(...,model_rows=N)`, and
`summary.json` lists every distinct method under `methods`. Check that field before
reporting any Direct-Prompt number.

This mechanism exists because the failure is real: an earlier single-call design (asking
one response to contain all 25 trajectories) silently produced fallback data for a
90-step-horizon task while reporting plausible-looking metrics. Switching to the cited
per-sample protocol fixed the cause; the `methods` field makes any recurrence visible.
