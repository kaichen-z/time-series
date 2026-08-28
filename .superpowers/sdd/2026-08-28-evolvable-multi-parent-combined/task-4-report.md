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

## Fix round 1: strict JSON and bounded diagnostics

### Implementation

- Replaced permissive object decoding at the proposal boundary with a strict
  local one-object parser that preserves the shared parser's think-tag,
  fenced-JSON, and embedded-object handling while rejecting duplicate keys at
  every object level and `NaN`/`Infinity` constants.
- Added finite-number preflight and `OverflowError` translation around canonical
  policy construction, so enormous integer thresholds/weights cannot escape the
  boundary as raw exceptions.
- Reworked diagnostics sanitization around one monotonic traversal depth and one
  shared node/item/byte budget across mappings and sequences. Forbidden keys are
  filtered recursively, including within nested sequence members.

### TDD evidence

RED:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py \
  -k 'duplicate_json_keys or nonfinite_json_constants or invalid_numeric_literals or bounds_deep_diagnostics'
6 failed, 5 passed, 16 deselected in 0.16s
```

The failures reproduced silent last-write-wins duplicate handling, raw
`OverflowError` for huge policy numbers, and diagnostics exceeding the bounded
prompt size through nested-map depth resets.

GREEN:

```text
11 passed, 16 deselected in 0.07s
```

Focused verification:

```text
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolving_agent_llm.py
87 passed in 2.32s
```

## Fix round 2: fail-closed diagnostics and direct-operation validation

### Implementation

- Diagnostics now carry only finite numeric, boolean, and null aggregate values.
  All freeform strings are dropped, including values under otherwise allowed
  keys, so diagnostic text cannot carry a secret into the LLM prompt.
- Mapping and sequence containers are rejected when over the hard item cap
  before sorting, iteration, or nested traversal. Key length is bounded before
  regex checks; oversized raw strings are dropped before any full scan. The
  monotonic depth/budget handling remains cycle-safe.
- Directly constructed `CombinedOperation` values receive the same structural
  checks as parsed operations before any portfolio method runs. Missing policies,
  impossible field combinations, unknown operations, and invariant failures are
  translated to `CombinedEvolutionError` without mutating the Parent.

### TDD evidence

RED:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py \
  -k 'malformed_direct_operations or secret_strings or oversized_diagnostics or cyclic_diagnostics'
3 failed, 1 passed, 27 deselected in 0.13s
```

The failures exposed the raw `AssertionError`, forwarding of an allowed-key
sentinel secret string, and an oversized mapping being iterated before its cap
was checked.

GREEN:

```text
4 passed, 27 deselected in 0.07s
```

Focused verification:

```text
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolving_agent_llm.py
91 passed in 0.67s
```

## Fix round 3: hostile container rejection

### Implementation

- Diagnostics accept only exact built-in `dict` mappings and exact built-in
  `list`/`tuple` sequences. Mapping and sequence subclasses are rejected before
  calling `len()`, iterating keys/items, converting keys to strings, applying a
  regex, or sorting.
- Built-in dictionary keys are first filtered to bounded raw built-in strings,
  then regex-filtered and sorted. This prevents a hostile key object or
  lying/infinite custom mapping from reaching any traversal work.
- A rejected diagnostic container causes `propose_combined_child()` to return the
  exact Parent with its generic rejection reason before an LLM completion call.

### TDD evidence

RED:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py \
  -k 'oversized_diagnostics or lying_length_mapping'
2 failed, 31 deselected in 0.14s
```

The prior sanitizer trusted custom `len()` values, sorted the lying mapping, and
called the LLM. The infinite-yield case was added with the same no-iteration
contract and run only after the fail-closed guard existed to avoid deliberately
blocking a test worker.

GREEN:

```text
3 passed, 30 deselected in 0.10s
```

Focused verification:

```text
../../.venv/bin/python -m pytest -q \
  tests/test_evolution_combined_evolution.py \
  tests/test_evolution_portfolio.py \
  tests/test_evolving_agent_llm.py
93 passed in 0.63s
```

## Fix round 4: collision-safe built-in dict traversal

### Implementation

- Built-in diagnostic dictionaries are traversed once with `.items()`. Only
  pairs with exact bounded string keys are retained; retained pairs are sorted
  by that trusted key and their captured values are sanitized directly.
- The sanitizer never performs a second `dict[key]` lookup, stringifies, or
  renders rejected key objects. This prevents an untrusted hash-collision key
  from running `__eq__` after filtering.

### TDD evidence

RED:

```text
../../.venv/bin/python -m pytest -q tests/test_evolution_combined_evolution.py \
  -k collision_key_lookups
1 failed, 33 deselected in 0.09s
```

The old second lookup invoked the armed collision key's `__eq__`.

GREEN and focused verification:

```text
1 passed, 33 deselected in 0.07s
94 passed in 3.18s
```
