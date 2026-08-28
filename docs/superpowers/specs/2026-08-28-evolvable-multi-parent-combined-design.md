# Evolvable Multi-Parent Combined Forecasts Design

**Date:** 2026-08-28  
**Status:** Approved in chat; implementation pending  
**Scope:** Numerical Agent only

## Goal

Replace the fixed five two-parent Combined policies with a typed, auditable,
history-only combination graph that can represent and evolve:

- TSFM + TSFM combinations;
- TSFM + Statistical combinations;
- multi-parent TSFM + Statistical combinations.

The system must not train or modify LLM or TSFM weights. Evolution changes only
combination structure, parameters, applicability, and source-controlled policy
artifacts.

## Current Limitation

`CombinedPolicy` currently stores exactly one `tsfm_parent` and one
`statistical_parent`. It supports only a two-way fixed-weight blend or a
single-signal binary route. `PolicyPortfolio` also requires exactly five fixed
Combined names and forbids parent changes during repair. As a result, the
Meta-Harness cannot add a TSFM ensemble, choose a different parent, or create a
three-model combination.

The reviewed method catalog already contains 24 Combined method specifications,
but only five are executable. This change upgrades the executable contract; it
does not claim that every collected specification is implemented.

## Chosen Approach

Use a typed combination DSL for ordinary evolution, with source-level evolution
reserved for genuinely new operators.

This is preferred over:

1. **More fixed hand-written pairs.** Easy to implement, but retains the current
   structural bottleneck and cannot discover new parent sets.
2. **Unrestricted generated Python for every child.** Expressive, but expensive
   to audit and unnecessary for median, weighted, trimmed, or routed ensembles.

The typed DSL lets the Agent generate useful new combinations without executing
arbitrary code. A later source-evolution phase may add a reviewed operator to the
DSL only after static checks, tests, Train screening, and read-only Dev acceptance.

## Policy Contract

### Parent references

`CombinedPolicy.parents` is an ordered tuple of two to five existing leaf
candidate names. A leaf is either:

- one of the manifest-bound `TSFMPolicy` names; or
- one executable Statistical method in `methods.py`.

Combined policies cannot initially reference other Combined policies. This keeps
evaluation acyclic and avoids recursion, hidden duplicate work, and dependency
ordering ambiguity. At least one parent must be a TSFM. Therefore this version
supports TSFM--TSFM and TSFM--Statistical, but intentionally excludes purely
Statistical ensembles from the non-Python portfolio.

The parent family is resolved by Python from the reviewed portfolio and parsed
method module. The Agent does not supply or override a family label.

### Operators

The initial reviewed operators are:

- `weighted_mean`: two to five parents, one nonnegative normalized weight per
  parent;
- `median`: two to five parents, pointwise median;
- `trimmed_mean`: three to five parents, pointwise mean after removing one minimum
  and one maximum value;
- `route`: exactly two parents, selected by one reviewed history-only signal and
  threshold.

`weighted_mean` weights must be finite, nonnegative, and sum to one within a
strict numeric tolerance. The other operators do not accept weights. `route`
stores explicit `above_parent` and `below_parent` names rather than an implicit
TSFM direction.

Every policy names one `fallback_parent`, which must occur in `parents`. The
fallback is used only if the selected operator cannot run because a non-fallback
parent is unavailable, crashed, invalid, or not applicable. The fallback itself
must have a successful, finite, horizon-length forecast. Fallback use remains
visible in the `Outcome.detail` field.

### Applicability

Combined policies retain the reviewed history-only applicability and signal
vocabulary. No documents, retrieved evidence, future values, GT evidence, or
task-role labels enter combination execution.

### Canonical payload

New policy artifacts use this shape:

```python
{
    "name": "combined_tsfm_statistical_v1",
    "parents": ("toto_2_0", "timesfm_2_5", "seasonal_naive"),
    "operator": "weighted_mean",
    "weights": (0.50, 0.30, 0.20),
    "signal": "periodicity_strength",
    "threshold": 0.45,
    "above_parent": "toto_2_0",
    "below_parent": "seasonal_naive",
    "fallback_parent": "toto_2_0",
}
```

Fields unused by an operator retain canonical neutral values so rendering and
hashing remain deterministic.

## Backward Compatibility

The parser accepts the legacy five-policy payload once and migrates it in memory:

- legacy `blend` becomes `weighted_mean` with `(weight, 1 - weight)`;
- legacy `route` becomes `route` with explicit above/below parent names;
- the legacy TSFM parent becomes the fallback.

Rendering always emits the new canonical schema. The five existing policies keep
their names and numerical behavior after migration. Frozen artifacts remain
readable, but any newly written policy file uses the new schema.

## Portfolio Mutation

`PolicyPortfolio` continues to freeze the five reviewed TSFM identities and their
manifest bindings. Combined policies become a bounded variable-length tuple:

- minimum: one;
- maximum: 32;
- unique policy names;
- unique parent names inside each policy;
- all parents must resolve to reviewed TSFM or executable Statistical leaves;
- at least one TSFM parent;
- no Combined-to-Combined references.

The portfolio API gains explicit atomic operations:

- `add_combined(policy)`;
- `replace(name, policy)` for repair, including parent changes;
- `remove_combined(name)`, while refusing to remove the final Combined policy;
- `fork_combined(source, child)` with a new unique name.

An invalid operation leaves the Parent portfolio unchanged.

## Agent Evolution Boundary

The Meta-Harness may propose only typed Combined operations. It may choose:

- parent set and order;
- reviewed operator;
- weights;
- route signal and threshold;
- above/below parent;
- fallback parent.

It may not change TSFM method IDs, checkpoints, adapters, runtime options,
licenses, scorers, task splits, caches, or evaluation labels.

The first implementation exposes the portfolio operations and schemas. Wiring a
new LLM mutation prompt into the formal 80/20 evolution command is a separate
follow-up, because mutation/search policy and executable representation have
independent acceptance criteria.

## Execution Semantics

For each task:

1. Execute and score all Statistical leaf methods.
2. Execute each TSFM leaf once, using the existing content-addressed cache.
3. Resolve each Combined policy's parent outcomes.
4. If every required parent succeeds, apply the reviewed operator pointwise.
5. If a required parent fails and the fallback succeeds, return the fallback
   forecast and record the degraded path.
6. Otherwise return the strongest applicable failure status without fabricating a
   forecast.
7. Validate exact horizon length and finite values before scoring.

Combined policies never invoke parent runtimes themselves; they consume already
materialized parent outcomes. This prevents duplicate TSFM calls.

## Acceptance and Safety

Generated Combined children follow the existing no-training evaluation pattern:

```text
typed child proposal
-> schema and parent validation
-> deterministic unit tests
-> small Train screen
-> complete 80 Train
-> exactly-once read-only 20 Dev
-> accept only if forecast and tail-risk gates improve
```

The 99-task Public Regression set is not used for mutation or acceptance.

Required forecast gates remain:

- 100% final forecast coverage through an explicit safe fallback;
- no increase in clipped sMAE/sRMSE count;
- no material sRMSE regression;
- bounded P90/P95 tail risk;
- lower Train and Dev clipped sMAE;
- no increase in active-oracle regret.

## Testing

Focused tests must prove:

- two-TSFM weighted and median combinations execute correctly;
- TSFM--Statistical and three-parent combinations execute correctly;
- trimmed mean is robust to one high and one low parent;
- route uses only reviewed history signals;
- fallback is explicit and does not trigger duplicate TSFM calls;
- invalid weights, duplicate parents, unknown parents, all-Statistical parents,
  cycles, and oversized portfolios are rejected;
- legacy five-policy source parses and re-renders canonically without changing
  forecasts;
- add, repair, remove, and fork are atomic;
- TSFM checkpoint identities remain immutable;
- the existing portfolio and selector suites remain green.

## Non-Goals

- No LLM SFT, RL, GRPO, or weight updates.
- No TSFM fine-tuning, merging, or checkpoint modification.
- No learned gate or meta-regressor.
- No contextual Retrieval input inside the Numerical Agent.
- No Combined-to-Combined recursive graph in this version.
- No automatic Public or hidden-set evaluation during evolution.
