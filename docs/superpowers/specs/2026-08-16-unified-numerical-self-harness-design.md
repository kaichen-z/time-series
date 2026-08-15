# Unified Numerical Self-Harness Design

## Goal

Unify the existing Coding Program Evolution loop and the planned Dictionary Filtering loop into
one Numerical Self-Harness with three controlled action modes: `dictionary`, `program`, and
`hybrid`. All modes share the same task splits, evaluation, Parent/Child lifecycle, held-out Dev
acceptance, checkpoints, and traces so improvements can be attributed to the numerical action
space rather than duplicated orchestration.

## Research Questions

The three modes answer distinct questions inside one implementation:

- `dictionary`: can an LLM learn to select and combine supplied numerical tools?
- `program`: can an LLM invent and revise executable numerical forecasters without supplied tools?
- `hybrid`: does tool routing plus open-ended program generation outperform either action space
  alone?

The modes remain separate experiment conditions even though they use the same harness.

## Shared Information Boundary

During numerical inference, the harness may expose only:

- historical numerical values;
- frequency and prediction horizon;
- public numerical-target metadata;
- reusable numerical skill summaries permitted by the selected mode;
- historical rolling-hindcast diagnostics.

It must never expose documents, retrieved evidence, GT evidence, future values, or Retrieval and
Decision output. The trusted host freezes all forecasts before resolved labels are used for Train,
Dev, or Public-Test scoring.

## Shared Numerical Policy

Every generation produces a versioned `NumericalPolicy` artifact:

```json
{
  "policy_id": "numerical_v001",
  "parent_policy_id": "numerical_v000",
  "mode": "hybrid",
  "filter_prompt": "...",
  "program_generation_prompt": "...",
  "program_revision_prompt": "...",
  "selection_budget": 3,
  "generation_budget": 1,
  "mutation_rounds": 1,
  "mutation_children": 1,
  "validation_folds": 3,
  "validation_horizon": 8,
  "aggregation": "best_hindcast",
  "changelog": "...",
  "mutation_hypothesis": "..."
}
```

Mode-specific validation freezes irrelevant fields. A `dictionary` child cannot increase program
generation budgets or alter program prompts. A `program` child cannot use dictionary tools or
filter prompts. A `hybrid` child may modify both numerical action paths.

## Mode 1: Dictionary

### Fixed Tool Dictionary

The canonical dictionary is a versioned JSON object whose enabled tools belong to
`statistical`, `foundation`, or `hybrid` families:

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

Backend kinds are initially `builtin` and `tsfm`. Unknown or unavailable backends are marked
unavailable and cannot be selected. Dictionary loading rejects duplicate IDs, invalid families,
malformed backend records, and an empty enabled set.

### Filter Output

The LLM returns tool IDs, not forecast values:

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
  ]
}
```

The host validates IDs, removes duplicates, enforces the budget, and supplies a deterministic
fallback for invalid or empty output. The fixed dictionary is not mutated in this experiment.

## Mode 2: Program

This mode migrates the current `CodingEvolutionAgent` behavior behind the unified policy:

1. Generate multiple executable forecasting programs.
2. Execute them in the existing sandbox.
3. Evaluate rolling historical cutoffs.
4. Revise the strongest failed Parent Program from fold scores and execution errors.
5. Retain valid descendants and produce numerical candidates.

Programs must implement:

```python
def forecast(history: list[float], horizon: int, frequency: str) -> list[float]:
    ...
```

The mode does not receive dictionary tools. Existing saved Program Skills can be loaded only when
the experiment explicitly enables a frozen Train-produced skill snapshot.

## Mode 3: Hybrid

Hybrid mode evaluates fixed dictionary tools and generated programs under the same historical
diagnostics and final numerical metric:

1. Compute historical diagnostics for available dictionary tools.
2. Filter the dictionary to a bounded tool subset.
3. Generate a bounded number of program challengers.
4. Validate all selected tools and generated programs identically.
5. Aggregate or select candidates according to the Numerical Policy.

Generated programs remain in a separate provisional Program Skill Library. They do not silently
modify the canonical tool dictionary. Dev and Public Test read a frozen snapshot learned on Train.

## Shared Execution and Aggregation

All modes produce a common `NumericalCandidate`:

```json
{
  "candidate_id": "...",
  "source_kind": "dictionary_tool|generated_program|saved_program_skill",
  "source_id": "...",
  "assumption": "...",
  "failure_condition": "...",
  "forecast": [1.0, 2.0],
  "hindcast_smape": 12.3,
  "fold_scores": [10.0, 14.6]
}
```

The first implementation supports `best_hindcast`, `mean`, and `median` aggregation. Every mode
uses the same deterministic fallback when no complex candidate is valid.

## Shared Reward and Failure Attribution

After inference is frozen, the trusted evaluator computes:

- final sMAPE and MAE;
- best available candidate and its sMAPE;
- selection regret;
- candidate count by source kind;
- mean and worst historical fold error;
- hindcast-to-future rank correlation;
- unavailable and failed source IDs;
- for dictionary and hybrid modes, oracle-tool retention and best-tool-filtered-out;
- for program and hybrid modes, best-generated-program coverage.

Mean final sMAPE is the sole acceptance metric and is minimized. The other fields diagnose which
action path failed and shape mutation context without overriding worse Dev forecasting.

## Unified Self-Evolution

The Solver and Evolver may use the same underlying LLM in separate calls and roles. The Evolver
receives the current policy, aggregate Train diagnostics, up to five sanitized failure
trajectories, and a mode-specific mutation boundary. It returns a complete child
`NumericalPolicy`.

For each generation:

1. Evaluate the Parent on Train.
2. Generate multiple mode-valid Child policies.
3. Validate each Child's mutation scope.
4. Screen Children on deterministic Train/Dev prefixes when successive halving is enabled.
5. Fully evaluate promising Children on Train.
6. Evaluate only the Train-best improving Child on read-only Dev.
7. Accept the Child only when mean Dev sMAPE is strictly lower than the Parent's.
8. Save the accepted policy, frozen Train-produced skills, checkpoint, and trace.

Train outcomes may update provisional Program Skills. Dev cannot create, revise, or persist Skills.
Public Test remains untouched until the policy is frozen.

## Data Split

The canonical experiments consume `splits/drcik_public_v1.json`:

- Train: 139 public labeled tasks;
- Dev: 30 public labeled tasks;
- Public Test: 30 public labeled tasks;
- Hidden Test: excluded from training and local scoring.

The loader verifies digest, exact membership, and entity disjointness. All three numerical modes
use the same manifest, model, token budget, generations, children, and evaluation metric.

## Package Structure

```text
numerical_agent/
  __init__.py             Public interfaces and version.
  __main__.py             `python -m numerical_agent` entrypoint.
  main.py                 CLI construction and command dispatch.
  policy.py               Unified NumericalPolicy schema and mode validation.
  candidates.py           Common NumericalCandidate representation.
  dictionary.py           Tool dictionary schema and validation.
  tools.py                Builtin tools and injectable TSFM adapters.
  program_adapter.py      Existing Program generation, sandbox, and revision adapter.
  dictionary_adapter.py   Dictionary filtering and execution adapter.
  hybrid_adapter.py       Bounded composition of both action paths.
  evaluation.py           Frozen numerical inference and trusted resolved scoring.
  evolution.py            Shared Parent/Child Train/Dev controller.
  persistence.py          Policy, checkpoint, skill snapshot, and trace artifacts.
```

Shared primitives are imported from `evolving_agent` rather than copied. Existing
`evolving_agent.coding_agent` public behavior remains available during migration and becomes a
compatibility wrapper around Program mode after parity tests pass.

## Artifacts

Every run writes:

- `best_numerical_policy.json`;
- `numerical_evolution_trace.json`;
- `checkpoint.json`;
- `train_evaluation.json`;
- `dev_evaluation.json`;
- `generated_program_skills.json` when Program generation is enabled;
- Public-Test predictions only through an explicit frozen-inference command.

## CLI

```bash
python -m numerical_agent evolve \
  --mode dictionary \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --dictionary numerical_agent/numerical_tools.json \
  --split-manifest splits/drcik_public_v1.json \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --generations 1 \
  --children 2 \
  --output-dir runs/numerical_dictionary
```

`--mode program` does not require `--dictionary`. `--mode hybrid` requires it. A deterministic
fake-LLM smoke path supports tests without network access.

## Experimental Matrix

The code is unified while the conditions remain separate:

| Condition | Dictionary tools | Program generation | Self-evolution |
|---|:---:|:---:|:---:|
| Frozen numerical baseline | Configured | Configured | No |
| Dictionary | Yes | No | Filter Policy |
| Program | No | Yes | Program Policy |
| Hybrid | Yes | Yes | Unified Numerical Policy |

This matrix separates tool-routing gains, open-ended coding gains, and their interaction.

## Safety and Testing

The shared controller cannot mutate the scorer, split, future-label firewall, sandbox, tool
implementations, or resource limits. Program code remains subject to the existing sandbox.
Dictionary IDs and backends are host-validated. TSFM dependency failures mark tools unavailable
instead of failing a policy.

Tests cover policy mode boundaries, dictionary schemas, program parity, common candidates,
malformed filtering output, unavailable tools, hybrid candidate composition, aggregation, resolved
diagnostics, mutation-scope rejection, Train/Dev acceptance and rejection, successive halving,
checkpoint round trips, no Dev learning, no Public-Test access, CLI smoke execution, and the full
existing `evolving_agent` regression suite.

