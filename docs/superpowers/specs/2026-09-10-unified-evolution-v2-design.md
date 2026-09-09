# Unified Evolution V2 Design

## Status

This document specifies a parallel, backward-compatible Evolution V2 control
plane. It unifies Numerical Dictionary self-evolution with end-to-end
Numerical/Retrieval/Decision co-evolution while retaining the existing runtime,
experiment artifacts, and command paths. It also adds bounded DGM-style source
self-improvement without allowing a candidate to rewrite the authority that
evaluates or promotes it.

The design decisions confirmed for this version are:

- preserve the existing system and add V2 alongside it;
- run both Numerical Supply self-evolution and three-Agent co-evolution;
- include MAP-Elites, constrained grammar mutation/crossover, NSGA-II,
  Hyperband, cooperative and joint co-evolution, UCB and Thompson schedulers,
  and a DGM-lite branching source archive;
- cap each resumable formal epoch at four hours;
- allow automatic source-Child promotion without human approval; and
- keep only a minimal external trust kernel immutable, while permitting
  infrastructure changes through a separately evaluated protocol migration.

## Goals

Evolution V2 must:

1. build and retain a diverse, executable Numerical Dictionary rather than
   converging only to one global Champion;
2. freeze a versioned Numerical Supply before it is consumed by Retrieval and
   Decision;
3. evolve Numerical, Retrieval, and Decision as cooperating modules whose
   fitness is measured through the complete forecast pipeline;
4. support attributable single-coordinate Children and coordinated joint
   Children;
5. allocate expensive evaluation resources adaptively and terminate within a
   four-hour epoch;
6. retain useful non-Champion stepping stones in both artifact and source
   archives;
7. permit the Evolver and Harness implementation to improve automatically; and
8. preserve label isolation, reproducibility, exact rollback, and frozen Public
   evaluation.

## Non-goals

Evolution V2 does not:

- train or fine-tune foundation-model weights;
- mutate an artifact during task inference;
- use Public-99 or hidden outcomes for mutation, scheduling, archive ranking,
  or promotion;
- silently reinterpret legacy artifacts under the V2 schema;
- permit a candidate to modify the kernel that scores and promotes that same
  candidate; or
- delete or rewrite existing experiment results.

## System Model

The system has one immutable authority boundary and three evolvable layers.

```text
L0  Immutable Evolution Kernel
    task/split commitments, label firewall, canonical primary scoring,
    sandbox/resource enforcement, artifact signatures, promotion, rollback

L1  Infrastructure Evolution
    loader adapters, forecast backbones, verifier strategies, diagnostic
    metrics, schema migrations, evaluation schedules

L2  Module and Bundle Evolution
    Numerical Dictionary/Supply, Retrieval release, Decision policy,
    prompts, skills, tools, budgets, and communication topology

L3  Evolver Evolution
    DGM-lite source variants of V2 proposers, schedulers, mutation operators,
    archive policies, and Harness orchestration
```

L2 and L3 may promote automatically. L1 changes use a protocol-migration epoch
evaluated by L0 against the old and proposed protocol. L0 is outside every
candidate's editable source root.

## Two Coupled Evolution Loops

### Numerical Supply Self-Evolution

The upstream loop produces a frozen, diverse supply of executable forecasts.

```text
Seed methods and releases
  -> classify history-only task morphology
  -> sample archive parents
  -> grammar-constrained add/repair/fork/combine/route/specialize/remove
  -> sandbox execution and historical hindcasting
  -> Hyperband resource allocation
  -> constraint filtering and NSGA-II selection
  -> update MAP-Elites cells
  -> freeze NumericalSupplyRelease + FrozenNumericalPackageRegistry
```

The Dictionary is a set-valued artifact. A method may be retained for a niche
even when it is not the best method globally. No Dictionary mutation occurs
during a task forecast or a Parent/Child pipeline comparison.

### Three-Agent Bundle Co-Evolution

The downstream loop consumes an exact Numerical Supply snapshot and optimizes
the full pipeline.

```text
Frozen Numerical Supply
  -> Numerical executes candidates and emits assumptions
  -> Retrieval produces host-verified evidence
  -> Decision selects an executed candidate
  -> final forecast is scored by the Host
```

Each accepted state is a complete `EvolutionBundleV2` containing Numerical,
Retrieval, Decision, scheduler, archive, protocol, and lineage identities. A
module Child changes exactly one principal component. A joint Child changes at
least two principal components atomically. All Children are scored through the
same complete pipeline.

## Immutable Kernel

The V2 kernel owns the following operations:

- canonical task materialization and task-content hashes;
- exact Train/Dev/Public membership commitments;
- removal of future labels and evaluator-only document labels before inference;
- canonical capped and raw sMAE/sRMSE computation;
- sandbox and process/resource limits;
- canonical serialization and SHA-256 artifact identities;
- cache-key validation and deterministic replay comparison;
- stage opening and Public-access prohibition;
- acceptance-evidence sealing;
- atomic active-version publication; and
- canary rollback.

The kernel imports candidate artifacts through data-only interfaces. Candidate
source code cannot import, monkey-patch, replace, or write kernel modules. The
kernel runs in a separate process with a read-only code root and a fresh output
directory.

## Core Interfaces

### Evolution Artifact

Every evolvable object implements the following semantic contract:

```python
class EvolutionArtifact(Protocol):
    @property
    def artifact_kind(self) -> str: ...
    @property
    def schema_version(self) -> int: ...
    def to_payload(self) -> Mapping[str, object]: ...
    def canonical_bytes(self) -> bytes: ...
    def fingerprint(self) -> str: ...
```

Canonical bytes contain no timestamps, filesystem-dependent paths, open file
handles, clients, callbacks, or secrets.

### Module Evolver

```python
class ModuleEvolver(Protocol):
    target: Literal["numerical", "retrieval", "decision", "joint"]
    def propose(
        self,
        parent: EvolutionBundleV2,
        archive: EvolutionArchive,
        feedback: SanitizedEvolutionFeedback,
        budget: EvolutionBudget,
    ) -> tuple[CandidateArtifact, ...]: ...
```

Proposers cannot evaluate or promote their own Children. The kernel recomputes
changed scopes and rejects ownership violations.

### Archive

`EvolutionArchive` is append-only and content addressed. It stores:

- artifact fingerprint and canonical payload;
- parent fingerprints and mutation operator;
- protocol and runtime fingerprints;
- Train-only behavior descriptors and objective vectors;
- closed evaluation status;
- cost and resource use;
- accepted-release references; and
- source lineage for DGM variants.

Dev values are stored in sealed acceptance evidence but are never exposed to
proposers, scheduler state, MAP-Elites descriptors, DGM prompts, or mutation
memory.

### Evolution Bundle V2

The bundle contains:

```text
schema_version
generation
parent_bundle_sha256
numerical_release_sha256
numerical_registry_sha256
retrieval_release_sha256
decision_policy_sha256
harness_policy_sha256
archive_snapshot_sha256
scheduler_state_sha256
protocol_fingerprint
runtime_fingerprints
acceptance_evidence_sha256
```

The payload embeds or content-addresses every dependency needed for frozen
replay. A rejected transition preserves the Parent bytes exactly.

## Numerical Quality-Diversity Search

### History-Only Descriptors

The kernel computes morphology descriptors only from observed history:

- trend strength: `low`, `medium`, `high`;
- seasonality: `none`, `short`, `long`;
- intermittency: `low`, `high`;
- regime behavior: `stable`, `shift`;
- forecast horizon: `short`, `medium`, `long`; and
- method family: `statistical`, `tsfm`, `combined`, `program`.

Bin thresholds are versioned kernel configuration derived from Train history
only. Descriptor computation never reads future values or document labels.

### MAP-Elites Archive

One archive cell represents one descriptor tuple. Each cell retains a bounded
Pareto front rather than only one scalar Champion. The default capacity is four
artifacts per occupied cell. When capacity is exceeded, constrained NSGA-II
selection chooses survivors.

Archive parent sampling mixes:

- 40% underexplored occupied cells;
- 30% high-performing elites;
- 20% failure-matched specialists; and
- 10% lineage stepping stones that are not current elites.

Percentages are normalized when a category is empty. The random stream is
seeded and checkpointed.

### Mutation Grammar

Numerical mutations use typed operations:

- `add`: introduce one executable method;
- `repair`: preserve identity while correcting a diagnosed failure;
- `fork`: change an assumption or applicability boundary;
- `combine`: build a validated multi-parent operator;
- `route`: select parents by history-only morphology or horizon;
- `specialize`: narrow a method to an archive niche;
- `crossover`: combine compatible structures from two archived lineages;
- `remove`: remove a redundant or unsafe member; and
- `quarantine`: retain provenance while excluding execution.

LLMs may propose symbolic structure and source code. Host code fits numeric
thresholds or weights, enforces namespaces and parent limits, and executes the
result in the sandbox.

## Constrained NSGA-II

Constraint handling precedes non-dominated sorting. A candidate is infeasible
when it violates any of:

- exact task coverage;
- finite forecast and exact horizon;
- no added invalid or catastrophic tasks;
- maximum per-task joint regret;
- frozen runtime and protocol identity;
- module ownership; or
- label, document-role, or Public-access boundaries.

Feasible candidates are ranked by minimizing:

- mean capped sMAE;
- mean capped sRMSE;
- P95 capped sRMSE;
- mean raw joint error; and
- normalized execution cost.

Crowding distance preserves objective diversity. MAP-Elites descriptors are
not added as weighted objectives; they define the cells within which NSGA-II
operates. Deterministic artifact fingerprint breaks exact ties.

## Hyperband Evaluation

V2 generalizes the existing successive-halving stages into checkpointed
Hyperband brackets whose resource is the number of committed task evaluations.
The formal resource levels are 8, 32, and 80 entity-group-respecting Train
tasks. Dev20 is not a Hyperband resource and is opened only for the final Parent
and Train winner.

Supported brackets are:

```text
explore: 8 -> 32 -> 80
confirm:     32 -> 80
replay:            80
```

The scheduler chooses a bracket before opening any stage based on remaining
wall-clock budget, cached coverage, and candidate count. Cached results must
match all artifact, task, runtime, metric, and protocol fingerprints. A cache
miss consumes normal budget; it never silently changes the stage universe.

The default promotion limits for three Children are `3 -> 2 -> 1`. Larger
DGM-generated populations use the bracket's reduction factor while always
leaving exactly one Train finalist. Partial evaluation cannot produce an
accepted Child.

## Cooperative Co-Evolution

Numerical, Retrieval, and Decision are cooperating species. A candidate from
one species is evaluated with the current accepted collaborators from the other
species. For robustness against collaborator overfitting, a Train-stage
candidate is also evaluated with at most one recent accepted collaborator
bundle when such a replay is already cached. Only the current Parent
collaboration can reach Dev or be promoted.

The normal mutation scopes are:

- Numerical-only;
- Retrieval-only;
- Decision-only; and
- joint, changing at least two of the three.

The evaluator records counterfactual module rewards for diagnosis, but only
end-to-end final forecast objectives govern inheritance.

## Adaptive Coordinate Scheduling

The scheduler has four arms: `numerical`, `retrieval`, `decision`, and `joint`.
The default policy is cost-aware discounted UCB. A Thompson-sampling policy is
also implemented and selectable in configuration so both requested scheduling
methods are available under the same state contract.

An arm reward is a Train-only vector projected to a scalar scheduler reward:

```text
relative joint improvement
- normalized evaluation cost
- catastrophe penalty
- invalid-candidate penalty
```

This scalar is used only for allocating search budget; it cannot override the
constrained NSGA-II or Dev acceptance gates. Rewards decay so old module behavior
does not dominate after collaborator changes.

The first epoch evaluates every arm at least once through a cheap bracket or a
structural dry run. Later four-hour epochs normally select one arm. The joint arm
receives a forced trial after two completed epochs without an accepted Child.

## DGM-Lite Source Evolution

### Editable Scope

Normal DGM epochs may edit only V2 implementations of:

- proposers and mutation operators;
- archive parent selection;
- UCB and Thompson schedulers;
- Hyperband bracket policy;
- failure diagnosis and sanitized feedback;
- agent prompts and tools; and
- Harness workflow and orchestration.

Backbone implementations, loader adapters, and verifier strategies can be
evolved only as L1 infrastructure artifacts. Primary metrics, canonical task
materialization, artifact validation, promotion, rollback, and kernel process
isolation are never editable by DGM candidates.

### Open-Ended Archive

The DGM archive is a branching content-addressed tree. Parent sampling considers
quality, novelty, cost, lineage depth, and number of existing children. A source
variant is retained as a stepping stone after passing static and unit-test gates
even if it does not become active. Failed safety or integrity variants are
recorded as terminal lineages and are never sampled.

### Automatic Promotion

No human approval is required. Promotion is:

```text
editable-path audit
  -> static security and label audit
  -> deterministic unit and integration tests
  -> Train meta-cross-validation
  -> read-only meta-validation gate
  -> cache-backed deterministic replay
  -> signed candidate publication
  -> one-epoch canary
  -> active publication or atomic rollback
```

The DGM process cannot write the active pointer. It submits a content-addressed
candidate to the kernel, which performs promotion in a separate process. Canary
failure automatically restores the previous active fingerprint and closes the
lineage.

## Protocol Migration for L1 Infrastructure

L1 changes are versioned proposals, not in-place mutations. Examples include a
new loader adapter, backbone runtime, verifier strategy, diagnostic metric, or
artifact schema. The kernel evaluates old and new protocols on the same
committed compatibility corpus. A proposal may become a new protocol version
only when it:

- reproduces canonical task hashes or declares an explicit data-version break;
- preserves label and Public isolation;
- reproduces all existing valid artifact identities or supplies a deterministic
  one-way migration;
- passes adversarial verifier fixtures;
- re-evaluates the relevant archive instead of comparing scores across metric
  definitions; and
- leaves the previous protocol runnable for frozen reproduction.

Primary metric changes start a new experiment epoch and cannot inherit a
Champion selected by a different primary metric definition.

## Four-Hour Budget Model

Every formal invocation is a resumable epoch with a hard four-hour wall-clock
limit. The kernel reserves the final 20% of the epoch for finalist completion,
Dev comparison, deterministic replay, checkpoint sealing, and clean shutdown.
If insufficient time remains to complete both sides of a stage, that stage is
not opened.

Budget profiles are:

| Profile | Data | Search | Hard limit |
|---|---|---|---:|
| `smoke` | deterministic fake 8/2 | one Child per enabled arm | 10 minutes |
| `pilot` | real 8 Train / 2 Dev | one generation, 4-8 Children | 2 hours |
| `formal` | 80 Train / 20 Dev | one scheduler-selected arm | 4 hours |
| `public` | frozen Public-99 | no mutation or learning | evaluation only |

The epoch records separate ceilings and use for wall time, task executions,
LLM calls, input/output tokens, GPU-seconds, subprocesses, and artifact bytes.
The first implementation derives conservative numeric ceilings from a dry-run
plan and aborts before execution when the configured resources cannot cover the
minimum bracket plus reserved finalization budget.

Co-evolution proceeds across epochs:

```text
epoch N     -> scheduler-selected module or joint search -> checkpoint
epoch N + 1 -> resume archive and scheduler state        -> checkpoint
```

No requirement forces all coordinates to run in one four-hour epoch.

## Failure Handling

- Structurally invalid proposals consume proposal budget but do not open task
  data.
- Runtime failure produces a closed invalid evaluation and cannot be cached as
  success.
- Missing, mismatched, or partial cache state is a cache miss.
- Deadline interruption checkpoint-seals completed immutable stages only.
- A Child that has seen Dev cannot return to Train mutation or scheduler memory.
- No accepted Retrieval release means Decision evolution requiring contextual
  behavior is skipped with a closed reason.
- DGM safety, integrity, or canary failure closes that source lineage and
  restores the prior active version.
- Artifact or authority corruption fails closed and requires starting from the
  last verified checkpoint; V2 never repairs authoritative state heuristically.

## Compatibility and Migration

V2 is implemented in new modules and commands. Existing `evolving_loop`,
Package Co-Evolution, Retrieval evolution, Numerical Champion, and legacy
prompt/genome/source commands remain unchanged during migration.

Adapters import existing frozen artifacts into a V2 seed bundle without
rewriting the source files. Imported artifacts retain their original bytes and
receive a V2 envelope containing the legacy schema and fingerprint. V2 outputs
use distinct directories and cannot overwrite legacy runs.

The migration order is:

1. introduce contracts, canonical artifacts, budgets, and deterministic fake
   fixtures;
2. add MAP-Elites plus constrained NSGA-II around existing Numerical execution;
3. add Hyperband brackets while retaining the existing fixed schedule adapter;
4. add cooperative module evolvers and UCB/Thompson scheduling;
5. connect the real frozen Numerical Supply to Retrieval and Decision;
6. add joint Bundle Children;
7. add the DGM-lite source archive and automatic canary promotion;
8. add L1 protocol-migration evaluation; and
9. run pilot and formal comparisons before deprecating any legacy controller.

## Delivery Decomposition

This architecture is intentionally larger than one implementation plan. It is
delivered as five independently reviewable projects, each with its own focused
design addendum, implementation plan, tests, and compatibility checkpoint:

1. **V2 Kernel and contracts**: canonical artifacts, bundle schema, append-only
   archive, budget accounting, immutable promotion boundary, fake fixtures, and
   the parallel CLI skeleton.
2. **Numerical quality-diversity**: history-only descriptors, MAP-Elites,
   constrained NSGA-II, typed mutation grammar, Hyperband, and frozen Numerical
   Supply export.
3. **Cooperative Bundle evolution**: Numerical/Retrieval/Decision adapters,
   UCB and Thompson schedulers, single-coordinate and joint Children, complete
   pipeline evaluation, and four-hour resumable epochs.
4. **DGM-lite source evolution**: editable-source isolation, branching source
   archive, automated meta-validation, canary activation, and rollback.
5. **Infrastructure protocol evolution**: versioned L1 backbone, loader,
   verifier-strategy, diagnostic-metric, and schema-migration proposals.

Project 1 is the only valid starting point because every later project depends
on its authority, artifact, archive, and budget contracts. A later project may
begin only after the preceding project's deterministic compatibility gate
passes. This master specification fixes the cross-project architecture; the
focused addenda fix project-local APIs and file-level changes before code is
written for that project.

## CLI Surface

The V2 entry point is separate:

```text
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/formal.json \
  --output-dir runs/evolution_v2/<run-id>

python -m evolving_loop.v2 public-evaluate \
  --bundle runs/evolution_v2/<run-id>/accepted_bundle.json \
  --output-dir runs/evolution_v2_public/<run-id>
```

The configuration selects budget profile, scheduler (`ucb` or `thompson`),
enabled mutation scopes, archive capacities, Hyperband policy, runtime bindings,
and seed. Formal and Public commands reject overlapping output directories.

## Artifacts

Each V2 run writes:

```text
run_manifest.json
budget_plan.json
checkpoint.json
progress.jsonl
archive/index.jsonl
archive/objects/<sha256>.json
candidates/<sha256>/proposal.json
evaluations/<sha256>/<stage>.json
acceptance/<sha256>.json
accepted_bundle.json
canary/<source-sha256>.json
evaluation_complete.json
```

All authoritative JSON rejects NaN and Infinity and is written atomically.
Large forecast and model-response payloads are content addressed and referenced
by digest.

## Testing Strategy

### Contract Tests

- canonical serialization and stable fingerprints;
- exact scope ownership and joint-scope validation;
- rejected-Child byte equality with Parent;
- archive append-only behavior and lineage reconstruction;
- schema and protocol mismatch failures; and
- budget accounting and deadline reservation.

### Algorithm Tests

- MAP-Elites cell placement, capacity, and deterministic parent sampling;
- constrained dominance and NSGA-II crowding/tie behavior;
- all Hyperband brackets, cache hits, pruning, and resume;
- UCB exploration/discounting and Thompson seeded reproducibility;
- cooperative collaborator selection and counterfactual diagnostics; and
- DGM stepping-stone retention, terminal lineages, canary promotion, and
  rollback.

### Safety Tests

- future values, Public IDs, evaluator-only document labels, and Dev metrics
  never reach proposers or scheduler memory;
- candidate source cannot import or write kernel modules;
- metric, split, verifier-kernel, validator, and promotion mutations fail;
- fabricated cache, lineage, signature, and acceptance evidence fail closed;
- expired budget cannot open a new stage; and
- Public evaluation cannot mutate archives, skills, policies, or active state.

### Integration Tests

- deterministic Numerical Dictionary self-evolution creates a diverse frozen
  Supply;
- the Supply is consumed by the complete three-Agent runtime;
- Numerical-, Retrieval-, Decision-, and joint-Child paths all execute;
- one accepted Child and one rejected Child reproduce after checkpoint resume;
- a DGM source Child promotes automatically through canary and another rolls
  back; and
- legacy commands and representative frozen artifacts remain unchanged.

### Real Validation

After deterministic tests pass:

1. run a two-hour real 8/2 pilot without Public access;
2. inspect complete artifact and budget accounting;
3. run one four-hour formal epoch on 80/20;
4. resume for additional four-hour epochs only from verified checkpoints; and
5. run Public-99 once on an explicitly frozen accepted Bundle.

## Acceptance Criteria

The implementation is complete only when:

- all named algorithms are implemented and exercised by deterministic tests;
- Numerical self-evolution produces a frozen MAP-Elites-backed Supply;
- the exact Supply is consumed by end-to-end three-Agent evaluation;
- single-coordinate and joint Children obey ownership rules;
- UCB and Thompson schedulers both resume deterministically;
- all Hyperband brackets enforce the four-hour budget contract;
- DGM-lite source variants form a branching archive and promote or roll back
  automatically without human approval;
- L0 rejects every attempted authority mutation in the safety fixtures;
- no Dev or Public information reaches mutation memory;
- legacy entry points and frozen-artifact reproduction still pass; and
- a real pilot finishes with an explicit accepted/rejected result and no Public
  access.
