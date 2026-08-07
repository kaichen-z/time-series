# Results

Measured with this package. `results/` is gitignored, so numbers are recorded here.

All three forecast metrics are **lower-is-better**. Read
[METHODS.md](METHODS.md) before citing any of these — sCRPS uses S=25 per the paper's
formula, and `evidence_recall` is our own judge proxy, not the official scorer.

## Public dev split (199 tasks)

DRBench agent on local Qwen2.5-14B-Instruct, Chronos-Bolt-base forecaster.

| Metric | Value |
|---|---|
| sMAE | 0.542 |
| sRMSE | 0.748 |
| sCRPS | 0.421 |
| EvidenceRecall *(proxy)* | 0.295 |
| SuppDocRecall (cited / retrieved) | 0.058 / 0.149 |
| DistractorAvoidance (cited / retrieved) | 0.180 / 0.222 |

Retrieval is the visible bottleneck: the agent cites supporting documents ~6% of the time,
and most of what it does cite is distractor material. That is the failure mode Dr-CiK was
built to expose, and it is where method work has the most room.

## Retrieval: BM25 vs dense embeddings

DRBench's own agent retrieves with dense embeddings rather than lexical matching, so the
obvious hypothesis was that BM25 explains the poor recall above. **It does not.**

Measured over all 199 public-dev tasks. This isolates the retriever exactly — DRBench's
search step uses no LLM, so no model calls are involved — and the BM25 row reproduces the
full pipeline's `supp_doc_recall_retrieved` of 0.1488 exactly, confirming the harness is
faithful.

| Retriever | SuppDocRecall (retrieved) | DistractorAvoidance (retrieved) | Time |
|---|---|---|---|
| BM25 (lexical) | **0.1488** | **0.2119** | 1 s |
| Dense (`all-MiniLM-L6-v2`) | 0.0643 | 0.0985 | 220 s |

Dense is **less than half as good on both metrics**, and ~200× slower. A chunk-concentration
confound was checked and ruled out: both return ~7.5 unique documents from 8 chunks, so
dense is genuinely surfacing worse documents, not merely clustering them.

The likely cause is the corpus, not the method. Dr-CiK's documents are built around
invented proper nouns — "Arid Heights Research Annex", "the Federative Republic of Althea".
Rare invented tokens are exactly where BM25 is strongest (exact match, high IDF) and where
a small general-purpose sentence encoder is weakest (out-of-distribution, no learned
semantics). DRBench's own default encoder is `text-embedding-ada-002` (1536-d), far
stronger than the 384-d MiniLM we substituted to keep retrieval offline and key-free;
testing that would need an OpenAI key and would change the conclusion's scope.

**Consequences:** BM25 stays the default — now an empirically justified choice rather than
an arbitrary one. And the retrieval bottleneck is *not* fixable by swapping the retriever;
whatever closes that gap has to be a method change, not a component swap.

## Sample bundle (3 tasks)

The official 3-task sample. Too small to rank methods — it is a smoke test, and the
per-task plots below are more informative than the means.

| Method | sMAE | sRMSE | sCRPS |
|---|---|---|---|
| Chronos-Bolt-base (via DRBench run) | **0.134** | **0.179** | 0.263 |
| Direct-Prompt · Qwen3.5-4B | 0.178 | 0.245 | 0.131 |
| Direct-Prompt · Qwen3.5-9B | 0.161 | 0.207 | **0.113** |

Both Direct-Prompt runs used genuine model output on all three tasks — no
`:degraded-fallback`, no `:padded` (verified against the fallback generator directly).

Two things worth noting rather than over-reading:

- **9B > 4B** on all three metrics, the expected ordering.
- **Chronos wins on point error, loses on sCRPS.** Chronos-Bolt's pseudo-samples come from
  quantile trajectories, which spread wider than the LLMs' sampled draws; sCRPS rewards the
  tighter, better-calibrated LLM spread while sMAE/sRMSE only see the mean. The two metric
  families genuinely disagree here, which is the interesting part.

## Reproducing

```bash
SAMPLE=/raid/home/air/khoutaibi/external/Dr-CiK/sample

dr-cik run --agent drbench --llm-backend qwen --sample-dir $SAMPLE \
  --output-dir results/drbench-qwen-sample

for M in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  dr-cik direct-prompt --sample-dir $SAMPLE \
    --from-run-dir results/drbench-qwen-sample \
    --model-id $M --output-dir "results/dp-$(basename $M)-sample"
done
```

`--seed` (default 7) seeds torch once at model load, so a fixed sequence of sampled calls
is reproducible on the same hardware. Exact float reproducibility across different GPU
models or transformers versions is not guaranteed.

## Caveats

- The hidden-test split has no public labels; those runs report `null` for every
  ground-truth metric by design.
- The Direct-Prompt agents consume evidence from a *prior* DRBench run, so their forecast
  quality is bounded by that run's retrieval quality — the 199-task retrieval numbers
  above are the ceiling.
- Sample-bundle results are 3 tasks. Do not rank methods on them.
