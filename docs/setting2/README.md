# Setting 2: Source-Backed Forecasting Knowledge

This module ports the Setting 2 external-knowledge component onto the current upstream architecture
at commit `8f0ace8`. It does not restore the deleted `evolving_agent/` package or replace the new
`common/`, `evolving_loop/`, or `numerical_agent/` structure.

## What this branch adds

### A compact forecasting knowledge base

`evolving_loop/knowledge/time_series.json` contains 90 operational forecasting rules in 28
categories, grounded in 48 primary or authoritative sources. Every entry records:

- a stable ID and category;
- a forecasting principle;
- explicit use and avoidance conditions;
- executable implementation guidance;
- diagnostic applicability tags and priority;
- traceable source identifiers, citations, and URLs.

The collection covers the main operational decisions needed by the Coding Agent: baseline choice,
trend and seasonality, robustness, regime changes, analogues, intermittent series, transformations,
combination, validation, uncertainty, constraints, and foundation-model use.

This library is complementary to `numerical_agent/datasets/forecast_method_dataset_v001.json`.
The upstream 166-method dataset is a canonical method-and-provenance catalog. The Setting 2
library is a compact set of operational rules retrieved from diagnostics for one observed series.

### Deterministic series diagnostics

`TimeSeriesKnowledgeBase` derives a label-free profile from observed history:

- history length and forecast-to-history ratio;
- lag-1 and seasonal autocorrelation;
- recent trend effect, level shift, and trend change;
- outlier and zero fractions;
- recent-to-early variance ratio;
- candidate cycle lags;
- tags such as long horizon, intermittency, count-like data, multiple cycles, regime change,
  heteroscedasticity, and weak seasonal evidence.

It selects at most ten rules, normally limiting any category to two entries. TSFM-specific entries
remain disabled unless the caller uses the combined setting.

### Opt-in knowledge-conditioned Coding candidates

The feature is deliberately opt-in:

```bash
python -m evolving_loop run \
  --tasks-file /path/to/tasks.jsonl \
  --setting statistics \
  --setting2-knowledge \
  --results-path runs/setting2/results.jsonl
```

Without `--setting2-knowledge`, existing upstream behavior is unchanged. With the flag enabled,
the Coding Agent generates its normal candidates and an additional knowledge-conditioned branch.
The normal branch is preserved so external knowledge can add coverage without erasing the Setting 1
search space. Knowledge IDs returned by the LLM are accepted only when they belong to the selected
rule set. All programs still pass through the existing sandbox and rolling hindcast evaluation.

The result JSONL records the knowledge-base version, diagnostic profile, selected entry IDs, and
per-candidate cited knowledge IDs.

## System integration

The implementation extends the existing Coding evolution loop instead of introducing a parallel
agent framework:

1. `--setting2-knowledge` is parsed by `evolving_loop/cli.py` and becomes
   `CodingEvolutionConfig.use_external_knowledge`. The default remains `False`.
2. At task start, `TimeSeriesKnowledgeBase` computes deterministic diagnostics from observed
   history, horizon, frequency, and the declared seasonal period. It retrieves a small, diverse
   rule set without reading future values, task documents, or relevance labels.
3. The original library and unconstrained generation path still runs. Setting 2 adds a second
   generation call conditioned on the retrieved rules, so knowledge can expand the candidate pool
   without replacing Setting 1 candidates.
4. Returned knowledge IDs are filtered against the retrieved allowlist and confidence values are
   bounded to `[0, 1]`. Generated code then follows the existing execution path: sandbox execution,
   causal rolling hindcasts, candidate ranking, mutation, and final selection.
5. A winning knowledge-conditioned program can enter the existing skill library only when it beats
   the repeat-last baseline under the configured threshold. No separate memory store is added.
6. Each result records the knowledge-base version, diagnostic profile, retrieved rule IDs, and
   candidate citations. The JSON file is included as package data, making installed runs traceable
   to the same rule version.

This design is complementary to the upstream Numerical Agent method catalog. The catalog supplies
canonical executable methods for dictionary curation; Setting 2 supplies task-specific operational
priors to the existing candidate-generation and validation loop.

## Evolution update (2026-08-22)

The latest `main` branch adds Git-backed Numerical method evolution, a versioned 166-method catalog,
MASE reporting, and isolated TSFM runtimes. Setting 2 continues to use the existing
`evolving_loop` contracts and now improves both levels of that evolution path without duplicating
the new runtime infrastructure.

### Single-agent evolution: preserve both lineages

Previously, knowledge affected only generation zero. The inner loop pooled ordinary and
knowledge-conditioned candidates, selected one global parent, and mutated only that parent. This
made one branch disappear immediately and dropped knowledge provenance during revision.

The loop now keeps two independently validated elites when available:

1. the best ordinary or library-derived candidate;
2. the best knowledge-conditioned candidate.

Each elite receives its own hindcast-driven revision call in every generation. A knowledge-lineage
revision receives only the diagnostic profile and rule IDs cited by its parent. Returned IDs are
checked against that allowlist, and the revised confidence remains bounded. Final selection still
uses the same pooled causal-hindcast ranking, so the extra lineage adds search diversity without a
privileged score. Results separately record retrieved rules and the rules actually cited by the
selected program.

### Multi-agent evolution: attribute the failure before mutation

The outer Meta-Harness already computed candidate-coverage, selection-regret, retrieval, and
hindcast-ranking diagnostics, but its mutation prompt did not receive them. It now receives those
aggregate diagnostics plus task-level candidate source, knowledge IDs, prior confidence, resolved
error, retrieval precision/recall/avoidance, and selection regret. This lets it distinguish:

- poor best-of-k forecasts: improve Coding candidate coverage;
- good best-of-k forecasts but high regret: improve Decision selection;
- weak evidence precision, recall, or distractor avoidance: improve Retrieval.

Successive-halving also now selects the full-evaluation child by Train reward before using held-out
Dev only for the final parent-versus-child acceptance decision. This fixes a path that previously
named the candidate `train_best` while actually selecting it by Dev reward.

## Historical experiment

The original Setting 2 work was developed before upstream reorganized the packages. That
experimental branch also explored safer multi-scale numerical evolution, evidence ledgers,
typed contextual effects, role-attributed prompt co-evolution, exact curricula, MAE-only outer
selection, and persistent outcome skills. The English reports are included for traceability, but
this current-main port intentionally limits executable changes to the non-conflicting knowledge
layer.

### Best frozen-30 checkpoint

The strongest historical Setting 2 v4 checkpoint achieved mean MAE **48.0744** on the same 30-task
development suite, compared with **106.9631** for Setting 1 and **156.3736** for Codex-Contract. It
produced 16 wins, 2 ties, and 12 losses relative to Setting 1.

### Later curriculum result

A later two-stage curriculum materialized 6 Coding, 12 Retrieval, and 5 Decision skills. Its frozen
run achieved mean MAE **87.6564**, median MAE **4.9820**, and mean sMAPE **36.8049**. It reduced mean
MAE by 18.05% relative to Setting 1 and by 43.94% relative to Codex-Contract, but did not beat v4.
Task 123 contributed 57.59% of its total error because a harmful post-selection route converted a
reasonable candidate into a poor final mixture. No additional frozen-30 tuning was performed.

These numbers describe the historical full Setting 2 system, not a new evaluation of this minimal
current-main port.

## Files

| Path | Purpose |
|---|---|
| `evolving_loop/knowledge_base.py` | Diagnostics, validation, retrieval, and prompt rendering |
| `evolving_loop/knowledge/time_series.json` | 90 rules and 48 sources |
| `evolving_loop/coding_agent/evolution.py` | Opt-in conditioned candidate branch |
| `evolving_loop/cli.py` | `--setting2-knowledge` and result provenance |
| `tests/test_knowledge_base.py` | Library, diagnostics, retrieval, and non-finite-history checks |
| `docs/setting2/SETTING2_DOMAIN_KNOWLEDGE.md` | Knowledge design and research basis |
| `docs/setting2/EVOLUTION_UPGRADE_AND_COEVOLVE_REPORT_20260816.md` | Historical inner/outer evolution report |
| `docs/setting2/FULL_DRCIK_EVOLUTION_AND_FROZEN30_REPORT_20260816.md` | Historical curriculum and final statistics |

## Data boundary

Knowledge selection uses only numeric history, horizon, frequency, and declared seasonal period.
It does not receive future values, task documents, document relevance roles, or ground-truth
evidence. The Dr-CiK revision used in the historical experiment was
`00fbe820ff7a221e4aca71883219ef27f8223050`.

## Validation

```bash
pytest -q
git diff --check
```

The knowledge JSON is included as package data so installed builds retain the same versioned rules.
