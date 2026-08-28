# Task 4 Report: Agent-Ready Structured Combined Operations

## Implementation

- Added `combined_evolution.py`, a typed proposal boundary with exact JSON
  schemas for `add`, `repair`, `fork`, and `remove` operations.
- Every operation uses canonical `CombinedPolicy` fields only. Batches contain
  no more than eight unique mutation targets. A fork reads its source but adds a
  distinct child name, leaving the source policy unchanged.
- Batch application is functional: it builds a local candidate, validates the
  final Statistical/TSFM/Combined namespace only after all operations, and
  raises `CombinedEvolutionError` on any failure. The caller still owns the
  unchanged Parent object.
- `propose_combined_child()` sends the LLM only canonical current Combined
  policies, reviewed Statistical names, bounded sanitized aggregate diagnostics,
  and an explicit operation schema. It never evaluates, scores, or accepts a
  child. Rejected proposals return the exact Parent and a generic sanitized
  reason.

## TDD evidence

RED began with the required missing-module test:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py
16 failed in 0.12s
```

All failures were the expected `ModuleNotFoundError` for
`numerical_agent.evolution.combined_evolution`.

GREEN after implementation and boundary tightening:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py
16 passed in 0.09s
```

Focused compatibility verification:

```text
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolution_policy_targetwise.py \
  tests/test_targetwise_evolution.py \
  tests/test_evolving_agent_llm.py
93 passed in 6.69s
```

## Safety review

- No training, weight/checkpoint, TSFM binding/order, runtime, or evaluation
  changes were made.
- The adapter does not receive task futures, documents, ground-truth evidence,
  task roles/subtypes, runtime secrets, or Dev/Public/hidden diagnostics.
- Policy source remains literal-only; the JSON response is parsed through the
  shared JSON-object parser and instantiated as validated immutable policies.

## Concern

The adapter deliberately requires the caller to supply the complete reviewed
Statistical namespace. This is necessary to validate the pre-existing Combined
policies as well as the candidate atomically; formal 80/20 wiring remains a
later controller concern.
