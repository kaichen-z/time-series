# Three-Coordinate Package Co-Evolution Design

**Date:** 2026-09-03

**Status:** Approved by user

**Scope:** Numerical, Retrieval, and Decision co-evolution over the Dr-CiK 80/20 split, followed by one frozen Public-99 regression evaluation

## 1. Goal

Build and run one reproducible co-evolution lifecycle in which Numerical,
Retrieval, and Decision improve a shared final forecasting objective. Each
coordinate may evolve, but only one coordinate may change in an accepted step.
Every Child is ultimately judged by the final forecast produced by the complete
three-agent system.

The first formal run uses two ordered coordinate cycles:

```text
Numerical 1 -> Retrieval 1 -> Decision 1
Numerical 2 -> Retrieval 2 -> Decision 2
```

The lifecycle stops early after a complete cycle accepts no coordinate. It
freezes one transitive three-module bundle before a separate Public-99 command
can run.

## 2. Experimental authority

The registered data authority remains the existing 80 Train / 20 Dev / 99
Public split.

- The 80 Train tasks are divided by the registered group-aware authority into
  64 Build and 16 Calibration tasks.
- Build supports proposal screening, cross-fitting, bounded host-side parameter
  search, and finalist selection.
- Calibration is an internal Train holdout. It gates access to Dev but does not
  alter a Child after evaluation.
- Dev participates in co-evolution. One frozen Train finalist per coordinate
  generation may be evaluated on Dev, and an accepted Dev Child becomes the
  Parent of the next coordinate.
- Public-99 is inaccessible to proposal, fitting, selection, and acceptance.
  It is loaded only by a separate frozen evaluator after the complete bundle is
  sealed.

Public-99 has been consumed by historical experiments in this repository. Its
result is therefore final regression evidence for this co-evolution run, not a
claim about a previously unseen benchmark. The co-evolution run must not feed
its Public result back into another Child or rerun a modified bundle against
Public.

Agents never receive future values, ground-truth evidence labels, scorer output,
or task-level Dev/Public residuals. Trusted host evaluators may use resolved
labels only after an inference artifact is frozen.

## 3. Existing implementation baseline

The design composes, rather than replaces, the current package-native work:

- `NumericalForecastPackage` is the immutable per-task Numerical handoff.
- `FrozenNumericalPackageRegistry` binds packages to exact task inputs and one
  manifest fingerprint.
- `PackageRetrievalEvaluator` evaluates Retrieval against a frozen Numerical
  registry and a frozen Decision implementation.
- `PackageDecisionEvolutionEngine` evolves Decision against frozen Numerical
  and accepted Retrieval releases.
- `PackageCoordinateBundle` and `PackageCoordinateController` enforce
  Retrieval-then-Decision coordinate isolation.
- Specialist Atlas can mine complementary Numerical candidates, but its latest
  experimental fixed 70% Toto / 30% specialist route passed 64-task Build and
  failed the 16-task Calibration tail check.

The package-native Retrieval/Decision branch has deterministic integration
coverage but no real 80/20 runner. The current controller treats Numerical as a
read-only manifest, and no Numerical phase can publish a replacement registry.
Those two gaps must be completed before a real three-coordinate run.

## 4. Chosen approach

Use coordinate ascent with global final-forecast acceptance. Do not mutate all
three agents simultaneously.

This approach was chosen over:

1. simultaneous triad mutation, which obscures causal attribution and permits
   cross-module compensation; and
2. a single Numerical-Retrieval-Decision pass, which cannot revisit Numerical
   supply after Retrieval and Decision improve.

Every step compares exactly one Child bundle with the current accepted Parent.
Rejection preserves the Parent byte-for-byte. Acceptance advances the bundle
lineage and changes exactly one principal module fingerprint.

## 5. End-to-end inference contract

### 5.1 Numerical supply

Numerical produces one immutable `NumericalForecastPackage` per task. The
package contains:

- one safe anchor from the current accepted Numerical Champion, initially Toto;
- no more than four additional, already executed alternatives;
- each alternative's finite full-horizon forecast vector;
- history-only hindcast diagnostics and fold stability;
- a task morphology profile;
- typed assumptions and failure conditions that distinguish alternatives; and
- transitive source, runtime, model, and input provenance.

The additional alternatives must preserve useful diversity rather than simply
the four best global means. Subject to availability, the host selects at most
one Statistical, one TSFM, one Combined, and one Atlas-derived or bounded-overlay
alternative. Duplicate forecast vectors are removed deterministically.

The experimental 70% Toto / 30% Atlas specialist forecast is eligible only as
an already materialized alternative. It is not the initial anchor and cannot be
applied unconditionally. The failure on Calibration task `task_185` remains
negative evidence for its tail gate.

### 5.2 Retrieval

Retrieval receives task history, timestamps, target metadata, task documents,
and the typed assumptions required to distinguish the frozen Numerical
alternatives. It does not receive task labels or realized alternative quality.

Retrieval retains the verified two-stage protocol:

1. Round 1 identifies relevant evidence and unresolved gaps.
2. Decision may request Round 2 for a typed high-priority gap.
3. Round 2 searches and verifies only that gap.
4. A malformed or failed Round 2 preserves verified Round 1.

The first formal run may mutate existing `RetrievalGenome` fields, prompts,
query strategy, evidence budgets, and accepted-skill selection. Its Retrieval
Skill library is read-only. New skill promotion is deferred until the triad
runner itself has produced a reproducible accepted release.

Retrieval quality, coverage, citation accuracy, temporal match, and distractor
avoidance are proposal diagnostics. They cannot accept a Child without an
improvement in final forecast metrics.

### 5.3 Decision

Decision receives the frozen Numerical package and the verified Retrieval card.
It may select exactly one materialized Numerical alternative and may issue the
existing typed Round-2 request. It cannot:

- create, edit, extrapolate, or blend a forecast vector;
- select an unknown or unavailable candidate;
- change Numerical or Retrieval fingerprints; or
- convert unverified text into a numerically actionable impact.

If a useful blend is desired, Numerical must materialize and validate it as an
explicit alternative before Decision runs.

### 5.4 Final scoring

The trusted host scores only the final Decision selection. Capped sMAE and
sRMSE are the primary metrics; their equal-weight mean is the joint diagnostic.
Numerical oracle gap, Retrieval quality, and Decision selection regret are
secondary feedback signals and never substitute for final-metric acceptance.

## 6. Coordinate evolution

### 6.1 Numerical coordinate

Retrieval and Decision remain frozen. The Numerical proposer may change:

- reviewed Statistical methods;
- TSFM membership;
- Combined recipes;
- typed morphology and horizon routing structure;
- Specialist Atlas pool construction; and
- bounded, pre-materialized overlay structure.

The LLM proposes only reviewed structural recipes: operator, parents, typed
assumptions, failure condition, and region. The trusted host searches numeric
thresholds and weights on Build. A Numerical Child materializes a complete new
package registry for Train and Dev before evaluation. Accepted Numerical steps
replace the registry manifest while preserving Retrieval and Decision bytes.

### 6.2 Retrieval coordinate

Numerical and Decision remain frozen. Three Retrieval Children are generated
from the accepted Retrieval Parent. The phase uses the existing package-native
evaluator with `strict_gain_target="final"`; contextual-only gain cannot accept
a Child. An accepted Retrieval step publishes one immutable non-`v000` release.

### 6.3 Decision coordinate

Numerical and the accepted non-`v000` Retrieval release remain frozen. Three
Decision Children are generated by the typed Decision mutation path. An
accepted Decision step changes only Decision-owned policy fields.

Decision cannot run before a non-`v000` Retrieval release is embedded. If a
Retrieval phase rejects all Children and the current bundle has only `v000`, the
corresponding Decision phase records a deterministic skip.

## 7. Screening schedule

Each coordinate generation begins with three Children and uses the same
successive-halving schedule:

1. 8-task Build screen: retain at most two Children.
2. 32-task Build screen: retain at most one Child.
3. 64-task group-aware Build cross-fit: establish the sole finalist.
4. 16-task Calibration: gate the unchanged finalist.
5. 20-task Dev: compare the unchanged finalist with the current complete Parent.

Task subsets, group folds, seeds, and their ordering are registered before the
run. Ties are resolved by canonical Child fingerprint. Only one Child per
coordinate generation can access Calibration and Dev. A rejected Child cannot
be edited and reclassified as the same finalist.

The proposer may receive sanitized, anonymous Train aggregates and structural
recipe identities. It does not receive raw future values, task identities, or
task-level Dev outcomes. Later generations may receive Dev acceptance status
and aggregate gate names, but not per-task Dev residuals or documents.

## 8. Acceptance gates

All metrics compare the Child's complete final pipeline against the current
complete Parent on the same tasks. Numerical-only or Retrieval-only proxy gains
cannot satisfy these gates.

### 8.1 Screen gates

The 8- and 32-task screens require:

- finite final forecasts for every task, including safe Parent fallback;
- no cross-coordinate mutation or Public access;
- no increase in invalid or catastrophic outcomes; and
- Pareto non-regression in mean capped sMAE and sRMSE.

Eligible Children are ranked by joint capped error, then sRMSE, sMAE, and
canonical fingerprint. Screen ranking is not acceptance.

### 8.2 Build, Calibration, and Dev gates

At each of the three full gates, a Child must satisfy all of the following:

- mean capped sMAE does not exceed the Parent within `1e-12`;
- mean capped sRMSE does not exceed the Parent within `1e-12`;
- relative joint improvement is at least 0.5%;
- P90 and P95 capped sMAE and sRMSE do not regress;
- invalid, catastrophic, clipped, and final-fallback counts do not increase;
- final task coverage is exactly 100%; and
- no accepted artifact accessed Public.

The 64-task Build cross-fit additionally requires at least four of five folds
to be non-regressing, at least three folds to improve strictly, and maximum
single-task joint regret no greater than 0.25. Calibration and Dev reject any
new single-task joint regret greater than 0.25, even when their aggregate means
improve.

An accepted coordinate must change exactly its declared principal fingerprint,
have direct lineage from the current bundle, and reproduce the same metrics on
immediate cache-backed replay.

The final bundle must also remain Pareto-non-regressing relative to the initial
Toto bundle. Coordinate acceptance is always relative to the current Parent;
the Toto comparison prevents cumulative drift.

## 9. Caching and resumability

The runner uses three immutable cache layers:

- Numerical forecast cache key: candidate implementation, task history,
  frequency, horizon, model runtime, and source fingerprints.
- Retrieval inference cache key: task/document fingerprint, Numerical package
  assumptions, Retrieval genome, Skill snapshot, verifier, and LLM runtime.
- Decision inference cache key: Numerical package, verified Retrieval card,
  Decision policy, and Decision runtime.

A key or dependency mismatch is a cache miss, never a partial reuse. Cache-only
stages fail closed before Dev if required entries are absent. Writers use atomic
creation and never overwrite accepted artifacts.

The checkpoint records the current accepted bundle, completed coordinate steps,
candidate fingerprints, split/group authority, source and runtime identities,
cache identities, and consumed-stage markers. Resume verifies every recorded
byte and continues from the first incomplete stage. It does not reevaluate an
already consumed Dev finalist.

## 10. Failure handling

- Malformed, non-finite, or unauthorized LLM output rejects that Child.
- A failed Numerical alternative becomes unavailable; the task retains the
  Parent anchor and records the method failure.
- A Retrieval Round-1 fatal failure skips Round 2 and preserves the Numerical
  Parent selection.
- A malformed or transiently failed Round 2 preserves verified Round 1.
- An unknown Decision selection falls back to the Parent selection and records
  one invalid/fallback event.
- Final fallback prevents missing predictions but never hides failure burden;
  the acceptance gates compare failure and fallback counts explicitly.
- Transient infrastructure errors may use the existing bounded retry policy.
  Deterministic contract violations are not retried.
- A cross-coordinate fingerprint change, detached lineage, mutable accepted
  artifact, or Public-access marker invalidates the Child before scoring.

## 11. Bundle and artifact model

Extend `PackageCoordinateTarget` to include `"numerical"`. An accepted bundle
binds:

- Numerical registry manifest and accepted Numerical release;
- accepted Retrieval release payload and SHA-256;
- accepted Decision policy payload and SHA-256;
- bridge, Numerical, Retrieval, Decision, verifier, metric, model, and LLM
  runtime fingerprints;
- parent bundle fingerprint and coordinate generation; and
- exact Train/Calibration/Dev evidence and gate decisions.

Each coordinate step records Parent, Child, and accepted bundle bytes; the
three principal fingerprints; changed modules; aggregate metrics; tail and
failure diagnostics; and whether each stage was opened.

The run output contains at least:

```text
run_manifest.json
group_folds.json
checkpoint.json
coordinate_trace.jsonl
candidate_evidence/
accepted_bundles/
final_bundle.json
evaluation_complete.json
```

The frozen Public evaluator writes to a new output directory and never modifies
the evolution directory. It produces per-task frozen forecasts, a final report,
and a completion marker.

## 12. Real runner

Add one package-native runner under `evolving_loop` and a thin shell wrapper
under `scripts`. The runner consumes explicit paths for tasks, split authority,
Numerical release, ForecastStore, Retrieval seed release, Retrieval Skill
snapshot, output directory, and runtime/model configuration. It defaults to:

- two coordinate cycles;
- three Children per coordinate generation;
- the 8/32/64/16/20 schedule;
- fixed seed `20260903`;
- package-native final-gain gates; and
- zero Public access.

The runner must not call the legacy prompt-only `scripts/run_co_evolution.py`.
That script operates on a different three-agent representation and does not
bind Champion Numerical packages.

Add a separate frozen Public evaluator that accepts only a completed final
bundle and rejects a bundle whose evolution trace is incomplete, non-canonical,
Public-accessed, or inconsistent with runtime authority.

## 13. Implementation sequence

1. Integrate the completed package-native Retrieval/Decision branch without
   changing its accepted contracts.
2. Stabilize the Specialist Atlas implementation and convert Atlas outputs into
   optional Numerical package alternatives; do not promote the failed fixed
   70/30 route to the anchor;
3. add a Numerical coordinate phase and registry publication contract;
4. extend bundle fingerprints, coordinate traces, and controller order to
   `Numerical -> Retrieval -> Decision` for repeated cycles;
5. add the real runner, cache/checkpoint binding, and immutable artifacts;
6. add the separate frozen Public evaluator;
7. run deterministic unit and integration verification;
8. run one real 8/2 smoke with one Child per coordinate;
9. run the registered two-cycle 80/20 co-evolution; and
10. freeze the final bundle and invoke Public-99 once.

## 14. Verification

Required focused tests cover:

- Numerical package diversity and the five-alternative bound;
- Numerical registry replacement with Retrieval/Decision byte preservation;
- Retrieval and Decision coordinate isolation;
- two complete `N-R-D` cycles and no-acceptance early stopping;
- 8/32/64/16/20 stage counts and one-finalist Dev access;
- group-aware cross-fit and absence of future-label input to every agent;
- strict final-gain acceptance for all coordinates;
- mean, fold, tail, regret, clipping, catastrophic, fallback, and coverage gates;
- `task_185`-shaped catastrophic overlay rejection;
- cache identity mismatch and checkpoint replay rejection;
- direct lineage and single-principal-fingerprint transitions;
- deterministic fake-LLM 80/20 replay; and
- complete Public isolation during evolution.

After focused tests pass, run the repository's full test suite and
`git diff --check`. The real 8/2 smoke must complete with no model cache miss,
unauthorized write, cross-coordinate change, or Public access before the full
80/20 run begins.

## 15. Reporting

The final report compares four nested systems on identical tasks:

1. initial Toto bundle;
2. final Numerical-only package selection under the seed contextual modules;
3. final Numerical plus accepted Retrieval under the seed Decision; and
4. the final Numerical-Retrieval-Decision bundle.

For each system, report capped and raw sMAE/sRMSE, joint error, W/T/L against
its direct Parent and Toto, P90/P95/max tail metrics, clipping, catastrophic,
invalid, fallback, and coverage counts. Report selected alternatives and
supporting evidence per task, plus the incremental change attributable to every
accepted coordinate step.

## 16. Deferred work

The first run does not include:

- simultaneous multi-coordinate mutation;
- neural-network or TSFM weight training;
- Retrieval Skill creation or promotion;
- Decision-generated forecast vectors or free-form numeric arithmetic;
- optimization against Public-99; or
- claims that the historically consumed Public-99 is an unseen benchmark.

These capabilities require separate designs after the package-native triad has
completed one reproducible real run.

## 17. Acceptance criteria

The implementation is ready for the formal experiment when:

- a real package-native runner executes two ordered `N-R-D` cycles;
- exactly one principal coordinate can change in each accepted step;
- every Child is accepted or rejected using complete final forecast metrics;
- the staged 80/20 lifecycle, immutable lineage, cache identity, and resume
  behavior are enforced by tests;
- Atlas is a candidate supplier rather than an unconditional failed anchor;
- a deterministic and a real 8/2 smoke both pass without Public access;
- the full 80/20 run produces one canonical final bundle; and
- only the separate frozen evaluator can consume Public-99.
