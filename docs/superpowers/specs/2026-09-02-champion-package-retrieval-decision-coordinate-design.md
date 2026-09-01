# Champion Package Retrieval–Decision Coordinate Evolution Design

Date: 2026-09-02
Status: implemented and verified
Base: `c80e1db` (`merge: integrate champion numerical evolution`)

Implemented modules:

- `evolving_loop.package_registry.FrozenNumericalPackageRegistry`
- `evolving_loop.package_retrieval_evolution.PackageRetrievalEvaluator`
- `evolving_loop.package_decision_evolution.PackageDecisionEvaluator`
- `evolving_loop.package_decision_evolution.PackageDecisionEvolutionEngine`
- `evolving_loop.package_coordinate_evolution.PackageCoordinateBundle`
- `evolving_loop.package_coordinate_evolution.PackageCoordinateController`

## 1. Goal

Complete the missing package-native evolution path after Champion Numerical
evolution. Retrieval and Decision must evolve against the exact frozen
`NumericalForecastPackage`, and the accepted three-module bundle must preserve
single-coordinate ownership:

1. Numerical supplies only materialized forecast alternatives and provenance.
2. Retrieval supplies only evidence over the task's documents.
3. Decision selects only one already materialized Numerical alternative.

The implementation must make this path usable with deterministic Train/Dev
tests. It must not run a real LLM evolution, access Public-99, or reinterpret
the old `EvolvingForecastHarness` result as a package-native result.

## 2. Existing contracts that remain authoritative

The implementation extends rather than replaces these boundaries:

- `numerical_agent.evolution.numerical_package.NumericalForecastPackage` is
  the frozen Numerical handoff. Champion packages bind the accepted release,
  recipe, and assumptions by SHA-256 fingerprints.
- `evolving_loop.numerical_two_stage.run_numerical_two_stage` is the only
  package-native Retrieval/Decision runtime. It does not rerun Numerical and
  already constrains Decision to materialized alternatives.
- `evolving_loop.retrieval_agent.evolution.RetrievalEvolutionEngine` owns the
  exact 80 Train / 20 Dev schedule, mutation isolation, checkpointing, and
  accepted Retrieval release publication.
- `evolving_loop.coordinate_evolution` owns single-coordinate acceptance and
  exact-parent preservation on rejection.
- Dr-CiK capped sMAE and sRMSE remain the authoritative point metrics. Tail
  safety, invalid outcomes, and catastrophic outcomes remain non-regression
  gates.

Legacy harness-based Retrieval/Decision evolution remains supported. New
defaults must not change its scientific contract.

## 3. Non-goals

- Do not evolve or rerun the Champion Numerical Agent in this change.
- Do not add a second Numerical package or forecast representation.
- Do not allow Retrieval or Decision to create, edit, blend, extrapolate, or
  otherwise synthesize numeric forecasts.
- Do not promote new Retrieval skills in the first package-native path. It may
  load only already accepted, read-only Retrieval skills.
- Do not fabricate legacy `HarnessResult`, candidate-pool replay, or
  leave-one-out skill snapshots for package results.
- Do not run Public-99 or any paid/live proposal call as part of implementation
  or verification.
- Do not implement simultaneous multi-module mutation. Co-evolution is block
  coordinate evolution with one mutable module per generation.

## 4. Package-native trusted evaluation

### 4.1 Frozen package provider

A trusted provider supplies one `NumericalForecastPackage` per `ContextTask`.
Before evaluation it binds:

- the ordered task identities;
- the Champion release, recipe, and assumption fingerprints;
- the exact package and alternative identities for every task;
- the metric-policy fingerprint.

The provider must fail closed when a task is absent, duplicated, reordered
against the registered split, or returns a package whose task/provenance does
not match the registered manifest. Retrieval receives neither hidden labels
nor candidate scores.

### 4.2 Retrieval evaluator

Add a package-native evaluator implementing the existing `RetrievalEvaluator`
protocol. For each candidate `RetrievalGenome`, it:

1. constructs `TwoStageRetrievalAgent` from the candidate genome and the
   read-only accepted skill library;
2. freezes the exact Decision implementation and policy for the full
   evaluation;
3. obtains the registered package for each task;
4. calls `run_numerical_two_stage` once;
5. after labels resolve, scores the final selected forecast and every
   materialized Numerical alternative with canonical Dr-CiK point metrics;
6. computes Retrieval recall, distractor avoidance, target/temporal match, and
   exact-quote validity from the final retrieval card and trusted task
   documents;
7. returns a typed `RetrievalEvaluation` without exposing task-level outcomes
   to mutation.

The numerical oracle and contextual oracle are the same best member of the
frozen candidate pool in this path. Retrieval cannot improve that oracle; its
benefit is enabling Decision to choose a better existing member.

### 4.3 Retrieval acceptance target

Extend `RetrievalEvolutionConfig` with an explicit strict-gain target:

```python
strict_gain_target: Literal["contextual", "final"] = "contextual"
```

The legacy default remains `"contextual"`. Package-native evolution uses
`"final"`. The value is validated and included in the science signature and
checkpoint identity.

For `"final"`, a child passes only when:

- mean final sMAE and sRMSE are both no worse than Parent within tolerance;
- at least one mean final metric is strictly better;
- p90/p95 final tails are no worse;
- the frozen-pool contextual oracle is no worse;
- supporting recall and distractor avoidance satisfy existing tolerances;
- exact-quote validity is exactly 1.0;
- invalid and catastrophic counts are no worse.

No contextual-oracle strict improvement is required because it is
structurally impossible with a frozen Numerical pool.

### 4.4 Skill promotion

Package-native evaluation disables candidate skill promotion and replay in
this first implementation. The evaluator may consume an accepted library
read-only, but it does not emit fake candidate-pool snapshots. If the engine is
configured to promote new skills with this evaluator, it fails before
evaluation with an actionable configuration error.

## 5. Package-native Decision coordinate

Add a dedicated Decision evaluator/phase rather than routing package results
through `CoEvolutionEngine`'s old harness factory. During one Decision phase:

- Numerical packages and their complete provenance are frozen;
- the accepted non-`v000` Retrieval release, skill library, and implementation
  fingerprints are frozen;
- mutation may change only existing Decision policy fields;
- each candidate is evaluated by `run_numerical_two_stage` on fixed Train and
  Dev splits;
- acceptance uses existing Pareto/non-regression principles over final
  sMAE/sRMSE, p90/p95 tails, invalid outcomes, and selection regret relative to
  the best materialized alternative.

The phase rejects Decision output that names no alternative, names an unknown
alternative, returns a non-materialized vector, changes Numerical/Retrieval
fingerprints, accesses Public, or produces nonfinite metrics. Runtime fallback
remains valid behavior, but increased invalid/fallback counts block promotion.

The package-native phase may reuse `HarnessPolicy`'s Decision fields for
backward-compatible policy serialization. It must not claim that the old
Numerical/Morphology fields identify the Champion package.

## 6. Three-module accepted bundle

Introduce a package-native accepted-bundle identity containing:

- exact Champion Numerical release/package-manifest fingerprints;
- accepted Retrieval release payload and SHA-256;
- accepted Decision policy payload and SHA-256;
- exact bridge, Retrieval runtime, Decision runtime, and metric-policy
  fingerprints;
- parent bundle identity and generation.

The coordinate controller continues to mutate one module at a time. This
change supplies two executable coordinates:

1. Retrieval: Numerical and Decision frozen;
2. Decision: Numerical and accepted Retrieval frozen.

The Numerical coordinate is initially a read-only accepted release input. A
later outer adapter may consume an already gated `ChampionEvolutionOutcome`,
but this change neither launches nor duplicates Numerical evolution.

Every coordinate step records Parent, Child, and accepted bundle bytes plus
principal-module fingerprints. Acceptance is legal only when the target
module alone changes. Rejection must preserve the exact Parent bytes and
hashes. Decision cannot run before a non-`v000` Retrieval release is embedded.

## 7. Error handling and reproducibility

All trusted boundaries fail closed on:

- missing or mismatched tasks/packages/releases;
- changed callable or runtime fingerprints;
- malformed, nonfinite, or length-mismatched forecasts;
- unknown Decision selection identities;
- mutated non-target modules;
- cache/checkpoint science mismatch;
- any Public Regression access flag.

Evaluation order, task membership, package identities, metric cap, strict-gain
target, frozen dependencies, and implementation fingerprints are included in
cache/checkpoint or bundle identities. User-facing exceptions contain fixed
categories rather than hidden task data or model output.

## 8. Implementation order

1. Add failing unit tests for `strict_gain_target` validation, gate behavior,
   science identity, and legacy compatibility; implement the smallest config
   and gate change.
2. Add failing tests for frozen provider registration and package-native
   Retrieval scoring; implement the evaluator without skill promotion.
3. Add failing tests for package-native Decision isolation, selection regret,
   and invalid selection rejection; implement the Decision phase.
4. Add failing tests for the accepted triad bundle and coordinate transitions;
   implement package-native fingerprinting and orchestration.
5. Add a deterministic 80/20 end-to-end test using fake packages, Retrieval,
   and Decision implementations. Verify no Numerical rerun, no new forecast,
   exact-parent rejection, and zero Public access.

Each step follows red–green–refactor. No live LLM, external model download, or
Public-99 command belongs to the verification suite.

## 9. Acceptance criteria

The change is complete when:

- legacy Retrieval evolution retains contextual strict-gain behavior;
- package-native Retrieval can accept a final-metric improvement over a frozen
  Numerical pool while rejecting regressions and invalid evidence;
- package-native Decision evolves only after a non-`v000` Retrieval release
  and can select only materialized alternatives;
- the accepted bundle transitively binds Champion Numerical, Retrieval, and
  Decision authority;
- every accepted generation changes exactly one principal module;
- rejected children preserve exact Parent bytes;
- deterministic package-native 80/20 tests pass without Public access;
- the relevant focused suites and the repository regression suite pass, apart
  from documented pre-existing failures.
