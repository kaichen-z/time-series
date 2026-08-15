# Tool Dictionary Curation Self-Harness Design

## Scope

Phase 1 builds and validates a reusable numerical forecasting tool dictionary with one
self-evolving Numerical Agent. It does not use context documents, Retrieval Agent, Decision
Agent, multi-agent co-evolution, or LLM weight training.

The required workflow is:

1. Construct one raw JSON dictionary containing statistical, time-series foundation-model, and
   combined forecasting methods.
2. Let one Numerical Agent implement every method behind a common executable interface.
3. Execute and test each implementation on historical time-series tasks.
4. Keep, revise, quarantine, or discard methods based on trusted performance evidence.
5. Accept an evolved dictionary generation only when it improves on held-out Dev tasks.

The output is a vetted working dictionary, not merely a ranked list of method descriptions.

## Terminology

- **Method specification**: a human- or collaborator-authored description of a forecasting
  method.
- **Tool implementation**: executable code or a validated wrapper that realizes a method.
- **Raw dictionary**: the initial collection of unvalidated method specifications.
- **Working dictionary**: methods with executable implementations and validation records.
- **Dictionary generation**: a versioned Parent or Child dictionary artifact.
- **Numerical Agent**: the single LLM role that implements and revises methods.
- **Trusted evaluator**: deterministic Python code that executes tools and calculates metrics.

## Information Boundary

The Numerical Agent may receive only:

- historical timestamps and values;
- frequency and prediction horizon;
- public numerical target metadata;
- one method specification at a time;
- sanitized implementation errors and historical hindcast diagnostics;
- summaries of previously validated numerical tools.

It must not receive:

- context documents or document roles;
- Retrieval or Decision Agent output;
- GT evidence;
- unresolved future values;
- Public-Test or Hidden-Test labels.

Resolved Train labels are visible only to the trusted evaluator after forecasts are frozen. Dev
labels are read-only for generation acceptance and cannot be used to revise individual tools.

## Raw Dictionary Schema

The repository stores one canonical raw dictionary:

```json
{
  "schema_version": 1,
  "dictionary_id": "forecast_tools_raw_v000",
  "tools": [
    {
      "tool_id": "damped_trend",
      "name": "Damped trend",
      "family": "statistical",
      "description": "Extrapolate a trend whose future slope decays over the horizon.",
      "assumptions": ["a recent trend exists", "trend persistence is uncertain"],
      "failure_conditions": ["abrupt regime shift", "dominant unmodeled seasonality"],
      "implementation_kind": "generated_python",
      "implementation_hint": "Use robust slope estimation and exponential damping.",
      "dependencies": [],
      "source_ids": [],
      "status": "unimplemented"
    }
  ]
}
```

Allowed families are:

- `statistical`: executable Python forecasting methods;
- `foundation`: wrappers around available TSFM backends such as Chronos or TimesFM;
- `combined`: deterministic selection, ensemble, preprocessing, or residual-correction methods
  combining statistical tools and/or TSFMs.

Allowed implementation kinds are `generated_python`, `builtin`, `tsfm_wrapper`, and
`composition`. Dictionary loading rejects duplicate IDs, malformed fields, unknown families, and
invalid dependency references.

## Common Executable Contract

Every accepted method must expose the same host interface:

```python
def forecast(
    history: list[float],
    horizon: int,
    frequency: str,
) -> list[float]:
    ...
```

Statistical and composition methods run through the existing forecasting sandbox. Foundation
methods use host-owned adapters; the agent generates or revises wrapper configuration and
composition logic, not foundation-model weights or external package internals.

An implementation is invalid when it fails static validation, times out, returns non-finite
values, changes output length, mutates its inputs, accesses forbidden data, or requires an
unavailable backend.

## Method-Level Self-Evolution

Each raw method is processed independently by the same Numerical Agent role:

1. **Implement**: translate the specification into executable code or a backend wrapper.
2. **Smoke test**: validate syntax, safety, determinism, output shape, and finite values.
3. **Hindcast**: evaluate on historical rolling cutoffs from Train tasks.
4. **Diagnose**: expose sanitized fold errors, execution failures, and failure categories.
5. **Revise**: allow a bounded number of implementation or applicability revisions.
6. **Classify**: assign `accepted`, `specialized`, `quarantined`, `unavailable`, or `discarded`.

The LLM proposes implementations and revisions. It cannot assign its own final status or score.
The trusted evaluator owns all metrics and status transitions.

## Status Policy

- `accepted`: valid and competitive on a sufficiently broad applicable task set.
- `specialized`: valid and materially useful on a coherent subset despite weak global averages.
- `quarantined`: valid or repairable, but evidence is insufficient or performance is currently
  weak.
- `unavailable`: requires a backend or dependency absent from the evaluation environment.
- `discarded`: irreparably invalid, duplicate and dominated, unsafe, or consistently inferior
  without a detectable applicability region.

Low global average performance alone is not sufficient for permanent discard. This prevents a
rare but valuable seasonal, intermittent-demand, or regime-shift method from being deleted.

## Method Evaluation

Historical evaluation uses the same deterministic folds and budget for all comparable methods.
The evaluator records:

- MAE and sMAPE per fold;
- mean, median, and worst-fold error;
- execution success rate;
- rank and win rate against simple frozen baselines;
- task characteristics associated with wins and failures;
- runtime and backend availability;
- whether a revision improved over its Parent implementation.

Method status uses Train evidence only. Method selection must consider conditional performance by
task characteristics, not only the raw mean across heterogeneous scales.

## Dictionary-Level Self-Evolution

A dictionary generation contains executable tools, learned applicability metadata, status, and
provenance:

```json
{
  "dictionary_id": "forecast_tools_v001",
  "parent_dictionary_id": "forecast_tools_v000",
  "generation": 1,
  "tools": [
    {
      "tool_id": "damped_trend",
      "family": "statistical",
      "implementation_ref": "generated/damped_trend_v002.py",
      "status": "accepted",
      "applicability": ["stable positive or negative local slope"],
      "failure_conditions": ["recent change point"],
      "train_summary": {
        "mean_smape": 18.4,
        "median_smape": 11.2,
        "win_rate": 0.61,
        "successful_tasks": 128
      },
      "version": 2,
      "parent_version": 1
    }
  ]
}
```

For each generation:

1. Evaluate the Parent dictionary on Train.
2. Identify invalid methods, duplicate methods, weak applicability metadata, and failure clusters.
3. Generate bounded Child changes: method revisions, applicability revisions, status changes,
   deduplication, or safe compositions.
4. Re-evaluate changed methods on Train.
5. Freeze the Child dictionary.
6. Evaluate Parent and Child dictionaries with the same router, budget, and tasks on read-only
   Dev.
7. Accept the Child only when the primary Dev metric improves and safety/coverage gates pass.

The initial implementation uses mean Dev sMAPE as the primary acceptance metric. Secondary gates
prevent apparent improvements caused by excessive method removal, backend failure, or severe
regressions on a small subset.

## Single-Agent Boundary

This remains a single-agent experiment even though the Numerical Agent is invoked at multiple
stages. `Implementer`, `Diagnoser`, and `Reviser` are prompt roles or calls of the same agent, not
separate communicating agents.

Python orchestration is responsible for:

- task loading and split enforcement;
- method scheduling;
- sandbox and backend execution;
- metrics and failure records;
- Parent/Child comparison;
- persistence and checkpointing;
- acceptance and rollback.

## Data Split

The canonical experiment uses `splits/drcik_public_v1.json`:

- Train: 139 public labeled tasks;
- Dev: 30 public labeled tasks;
- Public Test: 30 public labeled tasks;
- Hidden Test: excluded from local training and scoring.

Method implementation and revision use Train only. Dev accepts dictionary generations. Public
Test is accessed once after the dictionary and all budgets are frozen.

## Artifacts

The curation run writes:

- `forecast_tools_raw.json`: immutable input specifications;
- `forecast_tools_working.json`: latest fully materialized dictionary;
- `best_dictionary.json`: accepted dictionary generation;
- `method_evaluations.jsonl`: trusted per-method/per-task results;
- `dictionary_evolution_trace.json`: Parent/Child proposals and acceptance decisions;
- `checkpoint.json`: resumable progress;
- `generated/`: versioned validated method implementations;
- `quarantine.json`: rejected or unavailable implementations with reasons;
- `dev_evaluation.json`: read-only generation comparison.

No generated implementation or skill is learned from Dev or Public Test.

## Package Structure

```text
numerical_agent/
  dictionary.py             Dictionary and tool schemas.
  method_contract.py        Common executable contract and validation.
  method_implementer.py     Numerical Agent implementation/revision calls.
  method_executor.py        Sandbox and TSFM adapter execution.
  method_evaluation.py      Historical folds and method diagnostics.
  dictionary_curator.py     Status, deduplication, and Child construction.
  dictionary_evolution.py   Train/Dev Parent-Child lifecycle.
  persistence.py            Versioned artifacts and checkpoints.
  main.py                   CLI.
```

Existing primitives from `evolving_agent` are reused for LLM access, sandboxing, task loading,
metrics, retries, and trace persistence. Existing Coding Program Evolution supplies the inner
implementation/revision pattern; it is not copied wholesale.

## CLI

```bash
python -m numerical_agent curate \
  --dictionary numerical_agent/forecast_tools_raw.json \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --split-manifest splits/drcik_public_v1.json \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --method-revisions 1 \
  --generations 1 \
  --children 2 \
  --output-dir runs/dictionary_curation
```

A deterministic fake-LLM and a small fixture dictionary support offline unit and smoke tests.

## Acceptance Criteria

The first implementation is complete when it can:

1. Load and validate a mixed-family raw dictionary.
2. Implement at least one statistical method, one foundation wrapper, and one composition method.
3. Reject unsafe or malformed implementations before evaluation.
4. Execute deterministic historical hindcasts and persist per-method diagnostics.
5. Revise a failed method once using sanitized feedback.
6. Assign all five statuses through trusted rules.
7. Resume an interrupted curation run without repeating completed method evaluations.
8. Produce a complete versioned working dictionary and quarantine artifact.
9. Accept an improving Child and reject a non-improving Child on read-only Dev.
10. Preserve the existing `evolving_agent` regression suite and future-label firewall.

## Deferred Work

The following are explicitly outside Phase 1:

- Retrieval Agent and context documents;
- Decision Agent;
- multi-agent co-evolution;
- Prompt/Genome/Source Meta-Harness evolution of the full forecasting system;
- LLM weight training;
- online learning from Hidden Test or unresolved future outcomes.
