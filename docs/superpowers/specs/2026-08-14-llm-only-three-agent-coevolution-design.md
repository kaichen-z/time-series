# LLM-Only Three-Agent Co-Evolution Design

## Objective

Make the existing Dr-CiK harness demonstrate measurable, leakage-safe improvement while keeping
the Coding Agent strictly LLM-only. Retrieval and Decision must also evolve, but their changes must
remain attributable and must pass a disjoint development gate before inheritance.

## Scientific boundary

- Coding receives historical numeric values, horizon, frequency, optional seasonal metadata,
  reusable numeric skills, and historical hindcast diagnostics only.
- Coding receives no documents, retrieved evidence, GT evidence, future target values, supplied
  statistical method dictionary, or TSFM trajectory when `setting=llm_only`.
- Retrieval receives documents, task metadata, and Coding assumptions, but no resolved labels.
- Decision receives executed candidates, historical validation diagnostics, and verified evidence;
  it cannot invent an unexecuted forecast.
- Future values and public Dr-CiK annotations are available only after inference, in the trusted
  evaluator and skill-promotion path.
- Entity-disjoint train, development, and holdout splits remain immutable during a run.

## Evolution schedule

The system supports four target policies: `coding`, `retrieval`, `decision`, and `auto`.

1. A targeted generation mutates only the selected role. The other two roles and workflow remain
   byte-for-byte identical to the parent policy.
2. `auto` computes module rewards, mutates only the weakest role for prompt evolution, and may use
   a bounded joint Genome mutation only after targeted modes have established working parents.
3. Children are ranked on training reward. Only the best training child reaches development.
4. A child is inherited only when its development system reward is strictly greater than the
   parent. The sealed holdout is never used for selection.
5. Source evolution is last-stage exploration. A Source Engineer must edit the isolated worktree
   directly, may not ask for permission, and is rejected when it produces no changed files.

## Role-specific evolution

### Coding

Prompt mode may replace only `coding_generation_prompt` or `coding_revision_prompt`. Coding-target
Genome mode may also change initial candidate count, mutation count, mutation children, validation
folds, and validation horizon. It may not alter Retrieval, Decision, evidence adjustment, or
workflow. The Coding reward is candidate-coverage quality derived from resolved outcomes; final
system reward remains the inheritance gate so a locally improved generator cannot harm the system.

### Retrieval

Prompt mode may replace only `retrieval_prompt`. Retrieval-target Genome mode may additionally
change retrieval/decision round topology only when required to issue a gap query, while preserving
Coding fields. Retrieval is diagnosed by supporting precision/recall and distractor avoidance.

### Decision

Prompt mode may replace only `decision_prompt`. Decision-target Genome mode may additionally alter
decision aggregation and bounded evidence-adjustment policy, while preserving Coding generation.
Decision is diagnosed by selection regret and harmful-revision behavior.

## Initial success criterion

The bounded pilot succeeds when all tests pass and at least one generated child is accepted because
it strictly improves entity-disjoint development reward on the frozen 30-task public manifest.
The result must report parent and child train/dev rewards, per-module rewards, accepted policy,
failure count, and the untouched holdout manifest. A smoke result is not a paper claim; the next
step is the full 199-task public split with multiple seeds.
