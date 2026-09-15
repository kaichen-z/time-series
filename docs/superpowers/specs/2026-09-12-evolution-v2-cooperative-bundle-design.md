# Evolution V2 Cooperative Bundle Research Prototype Design

## Status

Project 3 design addendum to
`docs/superpowers/specs/2026-09-10-unified-evolution-v2-design.md`.
The user explicitly chose a runnable research prototype rather than another
production-hardening project on 2026-09-12. This addendum therefore keeps the
scientific invariants that affect results and removes defenses and acceptance
work that do not help the first cooperative-evolution experiment.

Project 3 starts from the completed Project 1 Kernel contracts and the complete
schema-v2 Numerical Dictionary produced by Project 2. It adds a complete
Numerical -> Retrieval -> Decision evaluation loop, adaptive coordinate
scheduling, single-coordinate and joint Children, checkpoint resume, and the
unified `evolve` command. It does not add DGM-lite source evolution, L1
protocol evolution, or Public evaluation.

## Prototype Objective

The prototype must answer one research question reproducibly:

> Given one complete schema-v2 Numerical Dictionary and mutable Retrieval and Decision
> artifacts, can a seeded UCB or Thompson scheduler generate and evaluate
> attributable module and joint Children through the real three-Agent pipeline,
> preserve the better complete Bundle, and resume to byte-identical results?

Success means both scheduler modes execute, all four mutation scopes execute,
the accepted Bundle can change across generations, a rejected Child preserves
the Parent, and an interrupted smoke run resumes to the same terminal summary
as an uninterrupted run.

## Deliberate Scope Reduction

This addendum narrows the master design only for the Project 3 research
prototype:

- `smoke` uses deterministic 4 Train / 1 Dev tasks; `pilot` uses 8 Train / 2
  Dev tasks.
- One invocation runs at most four scheduled steps with one Child per step.
- The 80/20 and four-hour profiles remain optional experiment configurations;
  they are not CI gates and are not required for Project 3 completion.
- Persistence occurs once per candidate evaluation and once per checkpoint,
  not once per forecast. No per-forecast `fsync` is added.
- Project 1 canonical JSON, digest validation, Kernel permits, immutable
  promotion, and output-directory separation are reused.
- New Project 3 code performs ordinary canonical readback and exact digest
  checks. It does not add new symlink, TOCTOU, inode-retention, crash-window,
  forged-receipt, or hostile-filesystem machinery.
- Test fixtures use deterministic in-process Retrieval and Decision agents.
  Real model clients remain dependency-injected pilot adapters, not CI
  requirements.

These reductions do not relax Train/Dev/Public isolation. Public tasks are not
accepted by the Project 3 runner at all. Future values are available only to
the Host evaluator and never enter proposal feedback or scheduler state.

## Reused Authority and Public APIs

Project 3 reuses these existing interfaces rather than building replacements:

- `EvolutionBundleV2.provisional_child(target, changes)` and
  `validate_child_scope(...)` for exact single/joint ownership;
- `EvolutionKernel.reserve_evaluation`, `close_evaluation`,
  `evaluate_transition`, `active_bundle`, and `finalize` for budgeted
  acceptance and Parent preservation;
- `EvolutionArchive` and `V2RunStore` for content-addressed Bundle lineage;
- `BudgetPlan`, `BudgetLedger`, and `ResourceUse` for bounded work;
- `FrozenNumericalArtifactsV2`, `NumericalSupplyRelease`, and
  `FrozenNumericalPackageRegistry` from Project 2;
- `RetrievalGenome`, `RetrievalSkillLibrary`, `HarnessPolicy`,
  `TwoStageRetrievalAgent`, and `DecisionAgent` from the legacy runtime; and
- `PackagePipelineEvaluator._evaluate_components(...)` and
  `PackageEvaluation` as the existing complete-pipeline scoring boundary.

Project 3 does not call the legacy coordinate controller, its 8/32/64/16/20
stage runner, or its Public evaluator.

## Package Layout

```text
evolving_loop/v2/cooperative/
  __init__.py       public Project 3 API
  contracts.py      canonical module, candidate, scheduler, and checkpoint artifacts
  schedulers.py     discounted UCB and seeded Thompson selection
  adapters.py       Numerical, Retrieval, Decision, and pipeline adapters
  proposals.py      deterministic single-coordinate and joint Child construction
  runner.py         bounded generation loop, Kernel transition, and resume
```

The unified CLI remains `python -m evolving_loop.v2 evolve`. A cooperative
configuration dispatches to the Project 3 runner; existing Project 1 fake
configuration continues to run unchanged.

## Canonical Artifacts

All new artifacts are finite canonical JSON values using
`canonical_v2_bytes` and `fingerprint_payload`. They reject unknown fields,
NaN, Infinity, and malformed SHA-256 identities. They contain no clients,
callbacks, paths, labels, or forecast arrays.

### `RetrievalModuleV2`

```text
schema_version = 1
source_release_sha256
genome_payload
skills_payload
```

`source_release_sha256` binds the imported legacy release bytes.
`genome_payload` parses as a real `RetrievalGenome`, and `skills_payload`
contains its unchanged Skill snapshot. Project 3 may mutate the Genome but not
the Skills; active Skill IDs must agree with the supplied verified runtime
library. This small V2 envelope avoids rewriting a legacy release manifest
merely to evaluate a candidate. The envelope fingerprint becomes
`EvolutionBundleV2.retrieval_release_sha256`.

### `DecisionModuleV2`

```text
schema_version = 1
prompt
skills
enable_evidence_adjustments
max_evidence_adjustments
aggregation
```

This is the existing Decision-owned slice of `HarnessPolicy`. It is sufficient
to reconstruct the exact `DecisionAgent` used by a pipeline evaluation. Its
fingerprint becomes `EvolutionBundleV2.decision_policy_sha256`.

### `BundleCandidateV2`

```text
schema_version = 1
target                       numerical | retrieval | decision | joint
parent_bundle_sha256
operator
numerical_release_sha256     optional
numerical_registry_sha256    optional
retrieval_release_sha256     optional
decision_policy_sha256       optional
```

The Host derives `changes` from the non-null fields and calls
`parent.provisional_child(target, changes)`. A single-coordinate candidate must
change exactly its named principal scope. A joint candidate must change at
least two principal scopes atomically. The candidate never supplies harness,
archive, scheduler, protocol, runtime, or acceptance identities.

### `CooperativeSchedulerStateV2`

The state contains the selected mode, seed, next draw counter, completed step,
discount factor, and one `SchedulerArmStateV2` for every enabled arm. Each arm
stores attempts, acceptances, discounted Train reward sum, and discounted cost
sum. No Dev value, task ID, document, forecast, or per-task residual is legal.

### `CooperativeCheckpointV2`

The checkpoint binds:

```text
schema_version
config_sha256
input_sha256s
active_bundle_sha256
scheduler_state
scheduler_state_sha256
next_step
accepted_steps
rejected_steps
completed_candidate_sha256s
kernel_checkpoint_sha256
checkpoint_sha256
```

It is rewritten atomically only at a closed candidate boundary. Resume verifies
the configuration, every operator input digest, the scheduler fingerprint, the
Kernel checkpoint, and the active Bundle. Project 3 does not resume a partially
executed forecast; the candidate is simply not checkpointed until its complete
aggregate evaluation closes.

## Adapters

### Numerical Coordinate Adapter

The Numerical adapter consumes the complete P2 Dictionary and applies the
history-only P3 Selector. It filters to at most eight safe candidates per task,
then selects Anchor plus zero, one, or two specialists with Anchor weight at
least 0.5. It never runs Numerical self-evolution inside Project 3. This makes
the active boundary explicit:

```text
Project 2 self-evolve -> complete Dictionary -> Project 3 Selector/co-evolve
```

The smoke fixture provides one alternate Supply/registry pair so the numerical
arm is executable. A pilot may point at several Project 2 outputs.

### Retrieval Coordinate Adapter

The Retrieval adapter parses `RetrievalModuleV2` into a real
`RetrievalGenome` plus a read-only `RetrievalSkillLibrary`. The smoke artifact
has no active Skills; a pilot with active Skills must inject the verified
read-only source library matching `skills_payload`. Its deterministic
proposal operator changes one legal typed field: round-one strategy,
round-two strategy, trigger, or one bounded evidence budget. It never weakens
the three Host verification booleans and never creates or promotes Skills.

A pilot may inject a callable proposal provider with the same typed output.
Raw model text is outside this prototype and is never accepted directly by the
runner.

### Decision Coordinate Adapter

The Decision adapter reconstructs a `HarnessPolicy` Decision slice and a real
`DecisionAgent`. Its deterministic operator selects one complete prompt from a
small configured prompt tuple or changes one bounded Decision-owned setting.
It cannot change Retrieval release fields or Numerical identities.

### Complete Pipeline Adapter

`CooperativePipelineAdapter.evaluate(bundle, tasks, stage)` resolves all three
artifacts by the exact SHA values in the Bundle, reconstructs the read-only
legacy factories, and calls
`PackagePipelineEvaluator._evaluate_components(...)`. The call supplies the
expected Retrieval genome and Decision prompt fingerprints, so evaluation
fails if a factory changes the bound module.

Only the Host sees labeled `ContextTask` objects. Proposal adapters receive a
`SanitizedEvolutionFeedback` containing aggregate Train categories and scalar
improvement only. The runner rejects tasks or manifests carrying Public
membership before proposing a Child.

## Candidate Construction

Each scheduled step creates one candidate.

- `numerical`: replace the atomic `(release, registry)` pair.
- `retrieval`: replace only the Retrieval module.
- `decision`: replace only the Decision module.
- `joint`: take the next two available distinct single-coordinate proposals in
  canonical scope order `numerical`, `retrieval`, `decision`, combine them in
  one `BundleCandidateV2`, and validate the actual Bundle diff as `joint`.

If an arm has no legal changed artifact, the runner records a closed
`no_candidate` step with a negative Train-only scheduler reward and continues.
No Dev task is opened for that step.

## Adaptive Scheduling

Both schedulers implement one pure interface:

```python
select_arm(state: CooperativeSchedulerStateV2) -> str
record_outcome(
    state: CooperativeSchedulerStateV2,
    arm: str,
    *,
    train_reward: float,
    normalized_cost: float,
    accepted: bool,
) -> CooperativeSchedulerStateV2
```

Untried enabled arms are selected first in canonical scope order. Thereafter:

- discounted UCB uses mean discounted reward minus mean normalized cost plus
  `sqrt(2 * log(total_attempts + 1) / attempts)`;
- Thompson samples a Beta posterior with `alpha = 1 + acceptances` and
  `beta = 1 + attempts - acceptances`, then subtracts mean normalized cost.

Thompson draws use a fresh `random.Random` seeded from SHA-256 of
`(run_seed, draw_counter, arm)`. Incrementing and checkpointing the draw
counter makes uninterrupted and resumed selection identical without storing
Python RNG state.

The scheduler reward is Train-only:

```text
relative improvement in mean capped joint error
- normalized task cost
- 1.0 for an invalid candidate
- 0.5 per catastrophic outcome rate
```

Dev acceptance and Dev metric values never update the scheduler.

## Complete Evaluation and Acceptance

For one candidate the runner evaluates the Parent and Child on the same Train
task tuple through the complete three-Agent pipeline. Parent results are cached
as one aggregate artifact per `(bundle, split, task-universe)` key. The final
Train objective contains aggregate capped sMAE, capped sRMSE, joint error,
invalid count, catastrophe count, fallback count, and normalized task cost.

A Train-eligible Child must:

- cover every committed task;
- contain finite aggregate metrics;
- not increase invalid or catastrophic counts; and
- not regress either mean capped sMAE or mean capped sRMSE by more than
  `1e-12`.

Only a Train-eligible Child is evaluated with its Parent on Dev. Dev passes
when both capped means are non-regressing within `1e-12` and their equal-weight
joint mean improves strictly beyond `1e-12`. The unchanged Child is then sent
to `EvolutionKernel.evaluate_transition`; otherwise the Kernel returns the
exact Parent.

The runner reserves one candidate stage before complete evaluation and closes
it with actual task count, elapsed wall time, and aggregate artifact bytes.
There is no per-forecast receipt or persistence event.

## Minimal Kernel Extension

Scheduler state changes after every Train result, including rejections, while
a rejected principal Child must preserve Parent bytes. Therefore the live
scheduler state remains in the Project 3 checkpoint after a rejection. On an
accepted transition, the Kernel must seal the latest Host-owned scheduler
fingerprint into the accepted Bundle.

Project 3 adds one backward-compatible keyword to
`EvolutionKernel.evaluate_transition`:

```python
host_scheduler_state_sha256: str | None = None
```

`None` retains current Project 1/2 bytes and behavior. A supplied value must be
a canonical SHA-256 and is passed only to Kernel-owned acceptance sealing and
evidence. Candidate payloads still cannot change this field. Joint Children
that include Numerical also use the existing numerical release-reference
validation based on their recomputed changed scopes.

No other Kernel, promotion, budget, path, or material-write behavior changes.

## Persistence and Resume

The run writes a deliberately small artifact set:

```text
run_manifest.json
budget_plan.json
checkpoint.json                 existing Kernel checkpoint
cooperative_checkpoint.json
objects/<sha256>.json           module/candidate/scheduler/aggregate objects
evaluations/<candidate>/<split>.json
progress.jsonl                  one row per closed scheduled step
archive/...                     existing Kernel Bundle archive
accepted_bundle.json
evaluation_complete.json
```

Artifact files use existing atomic canonical writes. Forecast vectors stay in
the in-process legacy pipeline and are not independently persisted. A resume
on an already complete run is a read-only no-op. Input or checkpoint mismatch
fails before new evaluation work.

## Unified CLI

The cooperative form is:

```text
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/cooperative/smoke.json \
  --seed-supply tests/fixtures/evolution_v2_cooperative/seed_supply.json \
  --task-manifest tests/fixtures/evolution_v2_cooperative/tasks_4_1.json \
  --retrieval-release tests/fixtures/evolution_v2_cooperative/retrieval_release.json \
  --decision-policy tests/fixtures/evolution_v2_cooperative/decision_policy.json \
  --output-dir /tmp/evolution-v2-cooperative-smoke
```

`evolve` keeps its existing two-argument fake form. The CLI detects a
cooperative configuration by its exact `CooperativeConfigV2` schema and then
requires the four seed inputs. `--resume` is inferred only from the exact
Project 3 run manifest in the output directory, matching the existing V2
commands.

`CooperativeConfigV2` embeds the existing `EvolutionV2Config` payload under
`control` and adds only:

```text
schema_version = 1
control
max_steps                 smoke=4, pilot=4
children_per_step         1
discount                   0.9
task_cost_weight           0.05
metric_cap                 5.0
acceptance_tolerance       1e-12
resource_ceilings
```

The embedded control must use `runner="production"`, must not use the Public
profile, and lists the enabled scheduler arms. A formal 80/20 configuration
may be created later without changing code.

## Fast Test Strategy

Project 3 completion uses only the following blocking tests:

1. contract and scheduler unit tests;
2. adapter tests with real legacy types and deterministic in-process agents;
3. runner tests on 4/1 fixtures covering every arm, accept, reject, and exact
   resume;
4. one CLI smoke for UCB and one for Thompson;
5. the existing focused Project 1 Bundle/Kernel tests and one Project 2 frozen
   Supply compatibility test; and
6. representative legacy package pipeline tests.

The Project 3 suite should complete within three minutes on the development
machine. It must not collect the full `test_evolution_v2_*` wildcard, the full
Numerical runner suite, 80/20, Public-99, real network clients, or a four-hour
epoch as a blocking gate.

Optional experiments, documented but not claimed by CI:

- real 8/2 pilot with injected Retrieval/Decision model clients;
- 80/20 comparison;
- repeated four-hour epochs; and
- frozen Public-99 evaluation after Projects 4 and 5 complete their trust
  boundaries.

## Acceptance Criteria

Project 3 is complete when:

- UCB and Thompson choose every enabled untried arm and resume deterministically;
- Numerical-, Retrieval-, Decision-, and joint-Child paths each produce an
  ownership-valid `EvolutionBundleV2` Child;
- every scored Child runs the complete Numerical -> Retrieval -> Decision
  pipeline on the same Parent/Child task tuple;
- Train-only feedback drives proposals and scheduler state, while Dev values
  remain only in Kernel acceptance evidence;
- one accepted and one rejected transition have correct Parent semantics;
- an interrupted 4/1 run resumes to the same accepted Bundle, scheduler state,
  counts, and terminal summary bytes as an uninterrupted run;
- `python -m evolving_loop.v2 evolve ...` completes UCB and Thompson smoke runs;
- existing fake `evolve`, `numerical-evolve`, and representative legacy package
  commands remain compatible; and
- the fast blocking gate finishes without Public access or a production-scale
  data run.
