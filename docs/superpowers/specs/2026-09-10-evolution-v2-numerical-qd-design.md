# Evolution V2 Numerical Quality-Diversity Design

## Status

Project 2 design addendum to
`2026-09-10-unified-evolution-v2-design.md`. The user approved the sequential
delivery order and the hybrid proposer boundary on 2026-09-10.

This project implements a real, resumable Numerical Dictionary/Supply
self-evolution loop on top of the completed V2 Kernel. It does not implement
Retrieval/Decision cooperative evolution, DGM source evolution, Public scoring,
or L1 protocol migration.

## Objective

Project 2 must replace the deterministic fake Numerical transition with a real
quality-diversity search that:

1. retains executable methods for distinct history-only task niches rather than
   converging to one global Champion;
2. evolves Numerical methods, typed combination/routing structure, applicability,
   mutation-policy weights, and the proposer prompt across generations;
3. evaluates every candidate through existing Numerical execution adapters;
4. allocates Train tasks through checkpointed Hyperband;
5. uses constrained NSGA-II within bounded MAP-Elites cells;
6. opens Dev only for the active Parent and final Train winner;
7. freezes an existing `NumericalSupplyRelease` plus
   `FrozenNumericalPackageRegistry`; and
8. promotes the resulting atomic release/registry pair only through the V2
   Kernel.

The self-evolution loop is:

```text
Parent Numerical Evolution State
  -> select an archive Parent or stepping stone
  -> propose typed Children from Train-only feedback
  -> validate grammar and executable ownership
  -> evaluate Hyperband rungs on Train
  -> constraint filter and NSGA-II selection
  -> update MAP-Elites cells and Train-only proposal credit
  -> compare final Train winner with Parent on read-only Dev
  -> accept an atomic Numerical release/registry Child or preserve Parent bytes
  -> repeat from the accepted state
```

## Scope and Trust Boundary

### Evolvable in Project 2

- Statistical method implementations stored as content-addressed candidate
  sources;
- immutable TSFM invocation and preprocessing policies, never model weights;
- typed Combined structures;
- history-only applicability and specialization rules;
- the distribution over mutation operators;
- the bounded LLM proposer prompt; and
- the selected frozen Numerical Supply.

### Fixed for a Project 2 run

- task and Train/Dev/Public split commitments;
- history-only descriptor thresholds;
- canonical sMAE/sRMSE and constraint policy;
- candidate parser, sandbox, timeout, and resource enforcement;
- Hyperband semantics and acceptance rules;
- V2 serialization, archive, budget, evidence, promotion, and rollback; and
- runtime/model/license identities.

Project 2 evolves Numerical artifacts, not the Python implementation of the
Evolver or Harness. DGM-lite may evolve proposer, scheduler, mutation-operator,
and Harness source only in Project 4, under a separate source archive and L0
evaluation. No candidate may edit the authority evaluating that same candidate.

## Package Layout

Add a focused package without rewriting legacy Numerical production modules:

```text
evolving_loop/v2/numerical_qd/
  __init__.py
  contracts.py
  descriptors.py
  mutation.py
  proposers.py
  nsga2.py
  map_elites.py
  hyperband.py
  adapters.py
  persistence.py
  runner.py
```

Add Project 2 tests under `tests/test_evolution_v2_numerical_*.py` and strict
profiles under `configs/evolution_v2/numerical_qd/`. Existing modules are
consumed behind adapters:

- `evolving_loop.package_numerical_supply`;
- `evolving_loop.package_registry`;
- `evolving_loop.package_numerical_evolution`;
- `numerical_agent.evolution.execution`;
- `numerical_agent.evolution.portfolio`;
- reviewed history-only analysis/morphology functions; and
- existing cache, metric, runtime, and split-manifest code.

An adapter may normalize a legacy artifact into a new V2 envelope, but it never
rewrites its source bytes or changes a legacy module's behavior.

## Closed Artifacts

All artifacts use strict finite canonical V2 JSON. Unknown fields, booleans in
integer positions, duplicate keys, NaN/Infinity, paths, callbacks, clients, and
live Python objects are rejected.

### `NumericalGenomeV2`

One genome describes an executable Supply candidate:

```text
schema_version
generation
parent_genome_sha256s
mutation_operator
inventory_sha256
screening_policy_sha256
combined_policy_sha256
mutation_policy_sha256
proposer_prompt_sha256
runtime_fingerprints
protocol_fingerprint
```

The inventory references typed Statistical, TSFM, and Combined member records.
Candidate Python source is stored by content digest outside JSON; the genome
contains only the source SHA and declared callable inventory. A Host parser
derives the executable inventory from the source and requires an exact match.

### `NumericalMutationPolicyV2`

The mutation policy stores, per typed operator, non-negative integer credits,
attempts, feasible Children, Hyperband promotions, and archive insertions. The
Host converts integer credits to sampling weights deterministically. It updates
credit only from Train feasibility, Train objectives, and QD insertion. Dev
acceptance never changes mutation memory.

### `NumericalProposerPromptV2`

The prompt artifact is a bounded template with an exact schema version,
allowed response schema, maximum response bytes, allowed mutation operators,
and Parent prompt identity. Prompt variants may be evaluated through capped
Train-only proposal episodes. A prompt earns credit only when its parsed
Children become feasible or enter a QD cell. Raw Dev values, evaluator labels,
and Public information never enter prompt text or prompt-selection memory.

### `NumericalEvaluationV2`

Each immutable evaluation record binds:

- genome, Supply, registry, task-subset, split, metric, runtime, and protocol
  identities;
- Hyperband bracket/rung and committed task IDs;
- per-task status without future values;
- finite aggregate objective vector;
- history-only descriptor cells;
- feasibility violations;
- resource use and cache identities; and
- Train-only diagnostic categories.

Forecast arrays and raw model responses are content-addressed separately. Dev
comparisons follow the Kernel rule: raw values appear only in sealed decision
evidence and never in QD records, scheduler state, prompts, or proposal memory.

### `NumericalQDStateV2`

The resumable state contains exactly:

```text
schema_version
generation
active_bundle_sha256
active_genome_sha256
qd_archive_snapshot_sha256
mutation_policy_sha256
proposer_prompt_sha256
hyperband_state_sha256
random_stream_counter
budget_checkpoint_sha256
completed_operation_sha256s
```

Randomness comes from a counter-based SHA-256 stream derived from the run seed,
stream name, and counter. Python `random` implementation state is not persisted.

### Frozen Supply

The final archive projection produces the existing canonical
`NumericalSupplyRelease` and `FrozenNumericalPackageRegistry` types. The Supply
retains its safe anchor and at most four executable alternatives, matching the
existing runtime contract. A deterministic coverage-first projection selects
alternatives from distinct occupied cells, then breaks ties by constrained
rank, crowding, normalized cost, and artifact SHA.

The Supply `source_fingerprints` bind the full QD archive snapshot, accepted
genome, screening/Combined policies, mutation policy, proposer prompt, and every
candidate source used by the projection. The V2 Bundle changes
`(numerical_release_sha256, numerical_registry_sha256)` atomically.

## History-Only Descriptors

The immutable Project 2 descriptor component computes only from task history,
horizon, and declared frequency:

- trend strength: `low`, `medium`, `high`;
- seasonality: `none`, `short`, `long`;
- intermittency: `low`, `high`;
- regime behavior: `stable`, `shift`;
- horizon: `short`, `medium`, `long`; and
- method family: `statistical`, `tsfm`, `combined`, `program`.

Every numeric threshold and frequency-to-period mapping is explicit in a
versioned descriptor policy whose fingerprint is part of the protocol binding.
Descriptor tests replace future values and evaluator labels independently and
require byte-identical output.

A candidate can occupy more than one task-morphology cell. Each cell entry is
candidate-and-cell specific and contains only objectives aggregated over the
cell's committed Train tasks.

## Typed Mutation Grammar

The only legal operations are:

- `add`: add one bounded executable Statistical/program member;
- `repair`: replace a failed member with a direct lineage Child;
- `fork`: change one assumption or applicability boundary;
- `combine`: create a typed multi-parent operator;
- `route`: choose members from history-only signals;
- `specialize`: narrow applicability to declared cells;
- `crossover`: combine compatible structures from two archived lineages;
- `remove`: remove one redundant member while preserving required coverage;
- `quarantine`: retain provenance but exclude execution; and
- `policy_tune`: propose a bounded prompt variant or Train-credit weight update.

Each operation has an exact tagged-union schema, maximum parent count, maximum
inventory growth, and allowed target fields. The Host fits numeric thresholds
and weights. An LLM response is an untrusted proposal: it is persisted for
audit, parsed once, normalized into the tagged union, and rejected before task
access if malformed or out of scope.

## Hybrid Proposer

The proposer interface accepts only canonical primitive data:

```text
Parent genome payload
selected archive-cell summary
sanitized Train-only feedback
remaining proposal budget
allowed mutation schema
```

It never receives Store, Kernel, archive handles, filesystem paths, callbacks,
raw forecasts, task futures, Dev values, or Public identifiers.

The production proposer has two providers:

1. an adapter around the existing repository `LLMClient`, requiring a strict
   JSON response and enforcing token/call budgets; and
2. a deterministic Host proposer that emits legal repair, fork, combine,
   route, specialize, remove, and policy-credit Children from the current
   inventory and archive diagnostics.

If the LLM is unconfigured, unavailable, times out, or returns no legal Child,
the event is closed and charged, then the deterministic provider continues.
The fallback therefore preserves a real self-evolution loop rather than
turning the run into a no-op. No network call is required by deterministic CI.

## MAP-Elites

One cell is an exact descriptor tuple. Each occupied cell retains a constrained
Pareto front of at most four entries by default.

Parent-category sampling uses the master proportions:

- 40% underexplored occupied cells;
- 30% high-performing elites;
- 20% failure-matched specialists; and
- 10% non-elite lineage stepping stones.

Empty categories are removed and the remaining integer weights are normalized.
The SHA-256 counter stream chooses the category and the sorted member. Sampling
state is checkpointed before proposal dispatch, so a crash cannot resample a
different Parent.

Archive append and cell-snapshot publication are immutable. Capacity selection
uses constrained NSGA-II; no scalar weighted score decides survival.

## Constrained NSGA-II

Constraint filtering precedes dominance. A candidate is infeasible for any of:

- missing required task coverage;
- non-finite forecast or wrong horizon;
- a new invalid/catastrophic task relative to Parent;
- excessive per-task joint regret;
- protocol/runtime/cache mismatch;
- ownership/scope violation; or
- future-label, Dev, document-role, or Public leakage.

Feasible candidates minimize this exact objective order:

1. mean capped sMAE;
2. mean capped sRMSE;
3. P95 capped sRMSE;
4. mean raw joint error; and
5. normalized execution cost.

The implementation provides deterministic constraint dominance, non-dominated
front construction, per-objective crowding distance, and final SHA tie-breaks.
Descriptor dimensions are cell coordinates, never extra weighted objectives.
Finite metric floats are compared only under the bound Python/metric runtime
fingerprints.

## Hyperband

Resource is the number of committed unique Train task evaluations. Supported
brackets are:

```text
explore: 8 -> 32 -> 80
confirm:     32 -> 80
replay:            80
```

For three Children the default promotions are `3 -> 2 -> 1`. Larger
populations use the configured reduction factor. Entity groups are kept
together, and deterministic task manifests fix every rung before execution.

The scheduler chooses a bracket before reserving work using remaining wall
time, available exact cache coverage, candidate count, and finalization reserve.
A cache key binds candidate, task content, split, metric, descriptor, runtime,
protocol, and execution-adapter identities. Missing or mismatched rows are
ordinary charged cache misses. Runtime failure is a closed invalid result and
is never cached as success.

Every rung records reservation, completed task IDs, results, survivors, and
post-rung budget. Dev20 is not a Hyperband resource. It opens once for the
active Parent and final Train winner after all Train mutation and policy credit
is sealed.

## Kernel Transition and Acceptance

At the end of a generation, the runner projects the best feasible archive state
to a frozen Supply and registry, constructs a Numerical-only provisional Bundle
Child, and requests an issuer-owned Kernel evaluation permit.

Train evidence contains the winner's aggregate objectives, cells, constraints,
and resource use. The trusted Dev adapter compares the exact active Parent and
winner. The Kernel decides:

- accept only when scope, protocol, runtime, accounting, Train status, Dev
  comparison, evidence, and archive bindings all pass; or
- reject and return the exact Parent object and bytes.

An accepted Bundle supplies the next generation's active Numerical Parent.
Useful non-winning and non-accepted candidates remain immutable QD stepping
stones, but they never become the active Bundle without Kernel acceptance.

## Checkpoint and Recovery

Project 2 adds one runner checkpoint beneath the run, cross-bound to the
existing Kernel and budget checkpoints. A checkpoint references only completed
immutable operations. The write order is:

1. persist source/proposal and candidate artifact;
2. reserve budget and persist the fixed rung manifest;
3. persist each completed task result and resource charge;
4. persist the immutable rung result;
5. append QD entries and publish an immutable QD snapshot;
6. persist mutation-policy/prompt state;
7. write and reread the exact runner checkpoint; and
8. begin the next operation.

Resume rereads and verifies every referenced byte, recomputes the archive,
replays budget closure identities, and checks the counter stream. A completed
operation is never executed twice. An unreferenced atomic temporary file may be
ignored. A partial authoritative operation, changed cache row, missing source,
open reservation, or mismatched checkpoint fails closed and requires a new
epoch from the last verified frozen Bundle; no heuristic repair is permitted.

Deadline interruption finishes accounting for already executed work and seals
only a fully completed rung. It does not publish a partial QD snapshot.

## CLI and Configuration

Project 2 adds an explicit intermediate command:

```text
python -m evolving_loop.v2 numerical-evolve \
  --config configs/evolution_v2/numerical_qd/smoke.json \
  --seed-supply <canonical-supply.json> \
  --task-manifest <train-dev-manifest.json> \
  --output-dir <new-or-exact-resumable-dir>
```

The config is a new exact `NumericalQDConfigV2` envelope, rather than silently
adding fields to the Project 1 config schema. It contains the V2 protocol and
budget profile plus descriptor policy, mutation limits, QD capacity, parent
sampling weights, objective/constraint policy, Hyperband brackets, proposal
budget, provider settings, and fixed seed Retrieval/Decision/Harness/scheduler
identities. The adapter materializes the seed Numerical registry, then constructs
the complete generation-zero V2 Bundle from those fixed identities and the
Supply/registry pair. Operator-supplied inputs are hashed into the run manifest
before output creation.

Profiles:

- `smoke`: deterministic provider, small fixture tasks, all algorithm paths;
- `pilot`: real adapters, 7,200-second limit, hybrid proposer; and
- `formal`: real adapters, exact 14,400-second limit with 0.2 finalization
  reserve, hybrid proposer.

Completion status is `numerical_qd_complete`. It explicitly reports whether an
LLM provider was used, occupied-cell count, frozen Supply/registry/Bundle
identities, Train/Dev access state, budget use, and `public_test_accessed:false`.
The completion is the V2 run's root `evaluation_complete.json`; the
`numerical_qd/` subtree has its own manifest and checkpoint but no competing
completion authority.
The existing full `evolve` production path remains unavailable until Project 3
connects Retrieval and Decision.

## Failure Semantics

- Malformed or out-of-scope proposal: charge proposal use, persist rejection,
  do not open task data.
- LLM unavailable or invalid: close provider attempt and continue deterministic
  fallback if proposal budget remains.
- Method/runtime crash: closed invalid evaluation; never success-cache it.
- Cache omission/mismatch: charged cache miss, never guessed repair.
- No feasible Child: preserve Parent and advance the Train-only policy state.
- No Dev improvement: preserve exact Parent Bundle; Dev does not update QD or
  proposal memory.
- Budget overrun: charge full actual use, close stage as overrun, forbid new
  stages, and finalize verified artifacts.
- Deadline during a rung: retain completed task rows but do not promote the
  incomplete rung or publish a new cell snapshot.
- Artifact/checkpoint/authority corruption: fail closed without overwriting the
  damaged bytes.

## Test Strategy

### Contract and safety tests

- strict schemas, immutability, finite numbers, stable cross-process hashes;
- source digest and declared callable inventory binding;
- every mutation ownership violation and grammar limit;
- future/Dev/Public/document-label sentinels absent from proposer requests,
  QD state, policy credit, prompts, progress, and archives; and
- candidate source cannot import/write Kernel modules through the Project 2
  execution boundary.

### Deterministic algorithm tests

- every descriptor bin and history-only invariance;
- multi-cell placement and capacity;
- 40/30/20/10 category sampling, empty-category normalization, and counter
  resume;
- constraint dominance, fronts, crowding, and artifact-SHA ties;
- every typed mutation and invalid union;
- mutation-policy Train credit and prompt-variant selection;
- all Hyperband brackets, promotion counts, entity grouping, exact cache keys,
  cache misses, invalid results, budget exhaustion, and rung resume.

### Integration tests

- deterministic self-evolution occupies multiple cells and freezes an
  executable Supply;
- at least one accepted Numerical Child and one rejected Child reproduce
  byte-for-byte across uninterrupted and resumed runs;
- deterministic fallback completes without LLM configuration;
- a scripted LLM produces one accepted legal proposal and several rejected
  hostile/malformed proposals;
- existing real Statistical/TSFM/Combined adapter seams materialize candidates
  on a small Train/Dev fixture without Public access;
- the accepted Supply and exact registry are executable by the existing package
  Numerical runtime; and
- all Project 1 and named legacy regression suites remain green without legacy
  artifact changes.

Network/model availability is not required in CI. A real hybrid pilot uses the
configured existing `LLMClient` and runtime manifests, but its absence cannot
invalidate deterministic algorithm correctness.

## Project 2 Exit Gate

Project 3 may begin only when a fresh verification run proves:

- every named Project 2 algorithm is implemented, not represented by intent
  config only;
- deterministic and hybrid proposer paths both execute through the same strict
  mutation parser;
- Numerical evolution updates its Supply, prompt, and mutation policy across
  generations using Train-only feedback;
- a bounded MAP-Elites archive retains multiple executable niches;
- all Hyperband brackets resume without duplicate work or deadline leakage;
- constrained NSGA-II deterministically selects per-cell survivors;
- a frozen `NumericalSupplyRelease` and exact
  `FrozenNumericalPackageRegistry` execute through the legacy adapter;
- Kernel acceptance promotes an atomic release/registry pair and rejection
  preserves Parent bytes;
- no Dev or Public value reaches mutation memory;
- a small real-adapter pilot finishes with `public_test_accessed:false`; and
- Project 1 plus legacy compatibility gates pass unchanged.

Project 2 does not claim complete Evolution V2. Project 3 consumes this exact
frozen Supply in Numerical/Retrieval/Decision cooperative and joint Bundle
evolution. Project 4 subsequently evolves Evolver/Harness source under the
fixed Kernel.
