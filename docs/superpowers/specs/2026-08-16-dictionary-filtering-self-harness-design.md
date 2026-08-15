# Dictionary Filtering Self-Harness Design

## Goal

Build a single-agent self-evolution experiment that learns how to filter and rank a fixed
dictionary of numerical forecasting tools. The experiment must isolate numerical reasoning from
Dr-CiK documents, Retrieval, and Decision, and accept an evolved filtering policy only when it
improves an entity-disjoint development split.

## Scope

The first version evolves the Numerical Agent's filtering policy, not the mathematical
implementation of dictionary tools. It supports statistical tools, time-series foundation-model
adapters, and hybrid tools through one JSON schema. Tool-code mutation and multi-agent
co-evolution remain separate later experiments.

The implementation reuses the existing `evolving_agent` infrastructure for Dr-CiK loading, numeric
task views, LLM clients, sMAPE, deterministic split manifests, tracing conventions, and label
firewalls. The collaborator-created top-level `numerical_agent/` directory is the owner of the new
dictionary-filtering experiment.

## Information Boundary

During inference, the Filter Agent may receive only:

- historical numerical values;
- frequency and prediction horizon;
- public metadata for the numerical target;
- tool descriptions, applicability conditions, failure conditions, availability, and historical
  hindcast diagnostics.

It must never receive:

- future values;
- GT evidence;
- documents or document labels;
- Retrieval or Decision output;
- resolved Train/Dev scores from the current task before its filtering decision is frozen.

The trusted Python evaluator executes and freezes all tool forecasts before exposing future labels
for scoring.

## Tool Dictionary

The canonical dictionary is a versioned JSON object:

```json
{
  "schema_version": 1,
  "dictionary_id": "numerical_tools_v000",
  "tools": [
    {
      "tool_id": "robust_local_level",
      "name": "Robust local level",
      "family": "statistical",
      "description": "Forecast a robust estimate of the recent level.",
      "applicability": ["locally stable level", "outlier-contaminated history"],
      "failure_conditions": ["strong recurring seasonality", "persistent trend"],
      "backend": {"kind": "builtin", "name": "robust_local_level"},
      "enabled": true,
      "cost": 1.0
    }
  ]
}
```

Required families are `statistical`, `foundation`, and `hybrid`. Backend kinds are initially
`builtin` and `tsfm`. Unknown or unavailable backends are reported as unavailable and cannot be
selected; they are not counted as forecasting failures.

Dictionary loading rejects duplicate tool IDs, missing required fields, invalid families, invalid
backend records, and an empty enabled tool set.

## Filter Policy and Agent Output

A versioned Filter Policy contains:

- policy ID and parent policy ID;
- complete filter prompt;
- maximum selected-tool count;
- aggregation strategy;
- changelog and mutation hypothesis.

For each task, the Filter Agent returns:

```json
{
  "selected_tools": [
    {
      "tool_id": "robust_local_level",
      "rank": 1,
      "confidence": 0.72,
      "reason": "The local-level hindcasts are stable."
    }
  ],
  "rejected_tools": [
    {
      "tool_id": "damped_trend",
      "reason": "Recent slope estimates disagree across folds."
    }
  ],
  "forecast_strategy": "best_hindcast"
}
```

The Python host validates tool IDs, removes duplicates, enforces the selection budget, and applies
a deterministic conservative fallback if model output is invalid or empty. The LLM never supplies
forecast numbers.

## Execution and Historical Diagnostics

Before filtering, the trusted host runs each available tool on rolling historical cutoffs. The
agent sees only label-free historical diagnostics such as median fold sMAPE, worst-fold sMAPE,
execution availability, and fold count.

After the filtering decision is frozen, the host executes every available dictionary tool for the
future horizon. This provides:

- forecasts for selected tools;
- a trusted post-resolution oracle tool for diagnostics;
- evidence of whether the best available tool was filtered out.

The final numerical forecast uses the policy's deterministic aggregation rule. The first version
supports `best_hindcast`, `mean`, and `median`.

## Reward and Failure Attribution

Per-task resolved diagnostics are:

- final sMAPE;
- selected tool IDs;
- oracle tool ID and oracle sMAPE;
- selection regret: final sMAPE minus oracle sMAPE, floored at zero;
- oracle-retained flag;
- best-tool-filtered-out flag;
- selected-tool count;
- unavailable and failed tool IDs.

The policy's primary score is mean final sMAPE across tasks, where lower is better. Mean selection
regret, oracle-retention rate, and selected-tool count are diagnostics only. They inform mutation
but cannot override worse development sMAPE.

## Self-Evolution

The same model may serve as Solver and Evolver, but they run in separate calls with separate
prompts. The Evolver receives:

- the current Filter Policy;
- aggregate Train diagnostics;
- up to five worst sanitized failure trajectories;
- the dictionary's tool metadata;
- a diversity instruction for the current child.

It returns a complete child Filter Policy. It may change the filter prompt, selection budget, and
aggregation strategy. It cannot change the dictionary, tool implementations, scorer, split,
future-label boundary, or resource limits.

For each generation:

1. Evaluate the parent on Train.
2. Generate multiple child Filter Policies from Train failures.
3. Evaluate children on Train and retain the Train-best child that improves the parent.
4. Evaluate the retained child and parent on read-only Dev.
5. Accept the child only when its mean Dev sMAPE is strictly lower.
6. Otherwise preserve the parent.
7. Save a checkpoint and append an evolution trace.

Dev tasks never generate failure feedback, mutate policies, or update a dictionary. Public Test is
untouched until the accepted policy is frozen.

## Data Split

The canonical experiment consumes `splits/drcik_public_v1.json`:

- Train: 139 public labeled tasks;
- Dev: 30 public labeled tasks;
- Public Test: 30 public labeled tasks;
- Hidden Test: excluded from training and local scoring.

The loader verifies the manifest digest, exact task membership, and entity disjointness before an
evolution run. Earlier fraction-based split generation is not used for this experiment.

## Artifacts

Each run writes:

- `best_filter_policy.json`;
- `filter_evolution_trace.json`;
- `checkpoint.json`;
- `train_evaluation.json`;
- `dev_evaluation.json`;
- a frozen Public-Test prediction file only when explicitly requested.

The fixed input dictionary remains unchanged. A later dictionary-cleaning experiment may write a
separate `filtered_numerical_tools.json`, but the first experiment attributes improvement only to
the filtering policy.

## Package Structure

```text
numerical_agent/
  __init__.py          Public types and version.
  __main__.py          `python -m numerical_agent` entrypoint.
  main.py              CLI construction and command dispatch.
  dictionary.py        Tool schema, loading, validation, and availability.
  tools.py             Builtin statistical tools and injectable TSFM adapters.
  diagnostics.py       Rolling historical tool diagnostics.
  filter_agent.py      Filter prompt, structured output, and host validation.
  evaluation.py        Frozen inference, trusted scoring, and failure traces.
  evolution.py         Parent/child Train/Dev acceptance and persistence.
  policy.py            Filter Policy schema and JSON serialization.
```

The implementation imports shared primitives from `evolving_agent`; it does not copy the existing
LLM, metric, task-loading, or split logic.

## CLI

The primary command is:

```bash
python -m numerical_agent evolve-filter \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --dictionary numerical_agent/numerical_tools.json \
  --split-manifest splits/drcik_public_v1.json \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --generations 1 \
  --children 2 \
  --output-dir runs/numerical_filter
```

A deterministic fake-LLM smoke command is available for tests. Real TSFM tools are optional and
must report unavailable when their dependency or model is absent rather than causing the entire
run to fail.

## Testing

Tests cover:

- valid and invalid dictionary schemas;
- duplicate and unavailable tools;
- label-free Filter Agent prompts;
- malformed, unknown, duplicate, over-budget, and empty selections;
- rolling historical diagnostics;
- deterministic tool execution and aggregation;
- oracle retention and selection regret computed only after resolution;
- child acceptance and rejection on held-out Dev;
- no Dev mutation and no Public-Test access during evolution;
- checkpoint and JSON round trips;
- CLI smoke execution;
- compatibility with the existing `evolving_agent` test suite.

