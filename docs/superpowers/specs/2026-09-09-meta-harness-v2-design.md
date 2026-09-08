# Meta-Harness V2 Design

## Goal

Make the complete three-agent `HarnessPolicy` the inherited Parent/Child unit,
while preserving Host-owned evaluation and safety boundaries. Every generation
must compare role-isolated ablations with at least one coordinated multi-role
Child, so cross-agent improvements are discoverable and attributable.

This first runnable slice deliberately uses the existing `CoEvolutionEngine` and
`EvolvingForecastHarness`. It does not relabel the newer Dictionary/Champion
package runner as integrated. The Numerical Dictionary remains an external
forecasting supply until a later adapter connects it to `HarnessPolicy`.

## Parent and Child

The Parent is one complete `HarnessPolicy` containing:

- Coding generation and revision prompts;
- Retrieval and Decision prompts;
- Coding candidate and hindcast budgets;
- the retrieve/decide workflow;
- evidence-adjustment controls;
- Decision aggregation; and
- accepted Skill snapshots and, when present, an authenticated Retrieval release.

One generation requests four distinct Child shapes:

1. Coding-only ablation;
2. Retrieval-only ablation;
3. Decision-only ablation; and
4. coordinated multi-role Child changing at least two role scopes.

If more than four Children are requested, later slots repeat the coordinated
shape but must remain structurally distinct. If fewer than four are requested,
Meta-Harness V2 rejects the configuration before any model call.

## Proposal interface

The LLM receives:

- the complete mutable Parent policy, excluding Skill bodies and authenticated
  Retrieval release internals;
- the exact requested Child shape;
- label-free interface failure categories;
- Skill names only;
- the last three generations of Train-only evolution memory; and
- an instruction to return every field.

The reply is one exact JSON object with these fields:

```json
{
  "mutation_scope": ["coding", "retrieval"],
  "interaction_hypothesis": "Changing candidate generation and evidence search together should improve coverage.",
  "coding_generation_prompt": "...",
  "coding_revision_prompt": "...",
  "retrieval_prompt": "...",
  "decision_prompt": "...",
  "coding_initial_programs": 3,
  "coding_mutations": 1,
  "coding_mutation_children": 1,
  "coding_validation_folds": 3,
  "coding_validation_horizon": 8,
  "workflow": ["retrieve", "decide"],
  "enable_evidence_adjustments": true,
  "max_evidence_adjustments": 3,
  "decision_aggregation": "last",
  "changelog": "..."
}
```

Unknown, missing, duplicate, malformed, or out-of-budget fields reject the Child.
The Host recomputes the actual changed scopes and requires exact equality with
`mutation_scope`. Role-isolated slots may change only their named scope. The
joint slot must change at least two of `coding`, `retrieval`, and `decision`;
workflow changes are allowed only in the joint slot and count as coordination,
not as a substitute for two changed roles.

An authenticated typed Retrieval release remains immutable to this generic
Meta-Harness proposal. Retrieval changes in that case require the dedicated
Retrieval evolution path; the Retrieval ablation is recorded as unavailable,
not silently accepted.

## Evolution memory

No independent memory Agent is added. The existing immutable evolution trace and
checkpoint are the memory authority. A bounded Train-only projection records:

- generation and Child shape;
- actual changed role scopes;
- the LLM's interaction hypothesis;
- Parent and Child Train sMAE/sRMSE; and
- one closed status: `train_improved`, `train_rejected`, or `invalid`.

It excludes task IDs, entities, forecasts, future values, document text, raw
failure traces, Dev metrics, and Public information. Only the last three
generations enter the next mutation prompt. The same canonical memory is stored
in the checkpoint and revalidated on resume.

## Evaluation

The existing label firewall remains authoritative:

```text
Parent + four Children
  -> Train-only successive-halving screen
  -> full Train comparison using sMAE and sRMSE Pareto improvement
  -> choose one Train finalist
  -> evaluate Parent and finalist on disjoint Dev
  -> accept complete Child or preserve exact Parent
```

Dev is read-only: its values never enter evolution memory or a later prompt.
Public/Holdout is not opened by evolution.

## Mutable and immutable boundaries

Meta-Harness may change all `HarnessPolicy` fields listed in the proposal schema,
subject to the Child-shape and resource bounds. It may therefore coordinate all
three prompts, numerical candidate budgets, workflow, evidence use, and Decision
aggregation across generations.

Meta-Harness may not change:

- task loading, Train/Dev/Holdout membership, or label visibility;
- sMAE/sRMSE definitions or Pareto acceptance;
- Retrieval citation/quote/target/window verification;
- the code sandbox;
- model/runtime/checkpoint authority;
- Skill bodies or authenticated Retrieval release contents through this path;
- resource ceilings; or
- exact Parent rollback.

## CLI and experiment

`evolving_loop.cli evolve --meta-harness-v2` enables this contract. It requires:

- `--evolution-mode genome`;
- `--evolve-target auto`;
- at least four Children; and
- a disjoint Dev split.

A deterministic fake test proves the four proposal shapes, Train-only memory,
checkpoint round trip, and exact rollback. The real experiment uses one
generation, four Children, a small Train/Dev split, `gpt-5.6-sol` with medium
reasoning, and successive halving. Its report must distinguish implementation
success, accepted/rejected policy outcome, and any unavailable model/runtime.

## Success criteria

- all four Child shapes are proposed and Host-validated;
- the coordinated Child can change multiple role scopes atomically;
- role-isolated and joint Children are compared under the same evaluator;
- the next generation receives only bounded Train memory;
- Dev information never enters a prompt or checkpoint memory record;
- legacy prompt/genome/source/retrieval modes remain unchanged when the flag is
  absent; and
- one local evolve command completes and publishes its trace and explicit metric
  comparison without opening Public/Holdout.
