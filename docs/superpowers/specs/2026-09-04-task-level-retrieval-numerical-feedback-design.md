# Task-Level Retrieval-to-Numerical Feedback Design

**Date:** 2026-09-04

**Status:** Draft for user review

**Scope:** Feed verified per-task Retrieval evidence from one accepted package
cycle into the next Numerical proposal generation, then run a small no-Public
A/B interaction smoke before changing the registered formal experiment.

## 1. Goal

Turn the existing ordered package loop into a directional feedback loop:

```text
Numerical 1 -> Retrieval 1 -> Decision 1
      ^                              |
      | verified task evidence       |
      +------------------------------+
Numerical 2 -> Retrieval 2 -> Decision 2
```

Retrieval does not veto or edit the current forecast. It reports which typed
Numerical assumptions are supported, falsified, or unresolved on each Train or
Dev task. The trusted host validates and sanitizes those records before the next
Numerical proposer sees them.

The experiment authority remains 80 Train / 20 Dev for evolution and 99 Public
for the final frozen test. Public tasks, labels, and artifacts never enter this
feedback loop.

## 2. Chosen approach

Use a host-validated, task-level evidence bridge.

This is preferred over:

1. aggregate-only feedback, which hides whether the same assumption fails on
   different morphology regimes; and
2. free-form agent dialogue or a shared blackboard, which permits task identity,
   unverified prose, scores, and cross-coordinate state to leak into Numerical.

The bridge keeps the useful per-task distinction while preventing the Numerical
proposer from memorizing repository task IDs.

## 3. Feedback contract

The trusted artifact is one immutable `TaskEvidenceFeedbackSnapshot` containing
one record per evaluated task and Numerical assumption. A record contains only:

- an opaque run-local `case_id`, never the benchmark task ID;
- a closed, label-free morphology projection: frequency, history/horizon bucket,
  trend, periodicity, intermittency, and recent-regime class;
- the existing Numerical `assumption_id` and its typed claim/failure condition;
- Retrieval stance: `supported`, `falsified`, or `uncertain`;
- target match: `matched` or `unmatched`;
- forecast-window relation: `overlaps`, `precedes`, `after`, or `unknown`;
- magnitude status: `present`, `missing`, `conflicting`, or `not_applicable`;
- a closed mechanism class: `event_shock`, `promotion`, `policy_change`,
  `capacity_change`, `measurement_error`, `regime_change`, `calendar_effect`,
  `macroeconomic`, or `unknown`;
- Decision action: `selected`, `rejected`, or `unresolved`; and
- an evidence-chain digest for audit and cache identity.

The model-facing projection omits benchmark IDs, document IDs, quotes, forecast
vectors, labels, residuals, task scores, gate results, and repository paths. A
fresh deterministic case namespace is generated for each proposal request, so a
case identity cannot become a reusable routing feature.

The Numerical proposal schema continues to permit only reviewed structural
recipes. Case IDs and evidence-chain digests are prompt-local correlation
handles only: they cannot appear anywhere in accepted model output, including
recipes, assumption IDs, candidate names, rationale, source code, or routing
predicates.

## 4. Trusted construction

The host constructs the snapshot only from a frozen inference trace whose:

- Numerical package and assumption fingerprints match the current accepted
  bundle;
- Retrieval card passed the existing citation, target, temporal, and provenance
  verifier;
- frozen Decision inference selected an alternative that exists in that
  Numerical package, regardless of whether a new Decision-coordinate mutation
  was accepted in the cycle;
- task belongs to the registered Train or Dev partition; and
- Public-access marker is false.

Unknown assumption IDs, unverifiable evidence, incomplete provenance, duplicate
task/assumption pairs, non-canonical enums, or mixed bundle fingerprints reject
the complete snapshot. No partial snapshot is supplied to Numerical.

The persisted audit artifact may retain trusted task/document identities needed
for reproducibility. Only the sanitized projection crosses into the proposal
model.

## 5. Lifecycle

### Cycle 1

`Numerical 1` receives no cross-coordinate evidence. Retrieval evaluates the
typed assumptions of the resulting accepted Numerical package. Decision selects
among already materialized alternatives as it does today.

After the cycle, the host freezes the task-level feedback snapshot from the
current accepted bundle and its verified inference traces.

### Cycle 2

`Numerical 2` receives its existing aggregate score feedback plus the sanitized
task-level evidence cases. It may respond by changing only Numerical-owned
structure, for example:

- replacing a trend parent when continuation assumptions are repeatedly
  falsified;
- adding a changepoint or level alternative for resolved event shocks;
- routing by morphology when the same assumption has different evidence stance
  across regimes; or
- retaining the Parent when evidence is missing or contradictory.

Retrieval and Decision fingerprints remain frozen during the Numerical step.
The Child still must improve final forecast metrics under the existing package
gates; evidence never grants acceptance by itself.

If there is no verified non-seed Retrieval trace, Cycle 2 receives an explicitly
empty snapshot. Transport failure, invalid schema, or a rejected snapshot cannot
be disguised as “no evidence.”

## 6. Interaction smoke and A/B comparison

Add a non-formal two-cycle interaction smoke using the registered 8/2/2 subset
and one Child per coordinate. It is separate from both the existing one-cycle
plumbing smoke and the formal 80/20 run.

To ensure the interaction itself is exercised, this smoke may publish a
Retrieval Child into an isolated, non-formal smoke lineage after it passes
schema, scope, provenance, and runtime-integrity checks. The smoke-only
publication does not require forecast gain, cannot update the formal registry,
and cannot be reused as formal acceptance evidence. Formal release gates remain
unchanged.

Run two lineages with the same initial bundle, tasks, seed, model, and runtime:

1. control: Cycle-2 Numerical receives no cross-coordinate cases;
2. treatment: Cycle-2 Numerical receives the verified task-level snapshot.

The interaction smoke answers only whether the bridge works and changes the
second Numerical proposal coherently. It does not promote a formal release or
access Public-99.

Report:

- whether both N-R-D cycles were actually invoked;
- schema-valid and materializable Child counts;
- feedback record count and assumption/morphology coverage;
- whether Numerical 2 changed a structure tied to a falsified assumption;
- accepted/rejected coordinate steps and exact reasons; and
- capped sMAE/sRMSE diagnostics for control and treatment.

The smoke completion marker requires `full_chain_exercised=true`. A skipped
Decision or missing feedback construction produces `incomplete_chain`, not a
successful completion marker.

## 7. Caching, checkpointing, and attribution

The feedback snapshot and sanitized projection receive separate SHA-256
fingerprints. Proposal cache identity, run manifest, checkpoint, and coordinate
trace bind both fingerprints plus the source bundle and evaluated task set.

Resume fails before model execution if feedback mode, snapshot bytes, task
membership, verifier identity, model/runtime, or proposal projection changes.

Every accepted step still changes exactly one principal coordinate fingerprint.
The feedback snapshot is host evidence, not a fourth evolvable coordinate.

## 8. Error handling

- Numerical candidate materialization tries every schema-valid proposed recipe
  in deterministic order and records typed failure reasons; one unavailable
  specialist does not discard the complete batch.
- Retrieval receives exact enum values and immutable-field requirements. One
  schema/scope correction retry is allowed; transport retries remain separate.
- Broad exception swallowing is removed from real smoke paths. Transport and
  integrity failures stop the run and preserve the last checkpoint.
- Formal acceptance gates and the Decision dependency on a non-seed accepted
  Retrieval release remain unchanged.

## 9. Verification

Required deterministic tests prove:

- Cycle 1 Numerical receives no task evidence and Cycle 2 receives the exact
  sanitized snapshot;
- Train and Dev records are accepted while Public records are rejected;
- benchmark IDs, document text/IDs, forecasts, labels, residuals, and scores
  cannot reach the Numerical proposer;
- unsupported assumption IDs and cross-bundle evidence fail closed;
- the Numerical output cannot encode a case identity;
- feedback changes only the Numerical coordinate;
- control and treatment have identical non-feedback authority;
- checkpoint/resume binds feedback bytes; and
- `full_chain_exercised` is false whenever any N-R-D phase is skipped.

After focused tests pass, run the two-cycle Luna interaction smoke. Do not start
the formal 80/20 evolution or Public-99 evaluation until the smoke artifacts are
audited and the user explicitly approves that next step.

## 10. Non-goals

- Retrieval does not create, edit, blend, or directly veto forecast vectors.
- Numerical does not read raw documents or unverified model prose.
- The first implementation does not train model weights or promote Retrieval
  skills.
- No Public-99 result is used for proposal, debugging, reranking, or reruns.
