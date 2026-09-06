# Train-80 Numerical Runner Design

Date: 2026-09-06

## Objective

Make the active Numerical Champion entry point use the package-native protocol that is already
implemented for co-evolution:

```text
80 Train
  -> nested 8-task screen
  -> nested 32-task screen
  -> five-fold entity-grouped OOF on all 80 Train tasks
  -> one read-only 20-task Dev gate
```

The active path must not create or consume a permanent 64 Build / 16 Calibration partition. The
existing 64/16 implementation remains readable only for historical artifacts and compatibility;
it is not invoked by the active wrapper and its checkpoints cannot resume into the new protocol.

## Why the old controller is not retuned

The standalone Champion controller embeds Calibration in its manifest, authority store,
checkpoint, reports, and release lifecycle. Changing only its size constants would not produce
true out-of-fold evaluation: a recipe could still be fitted and scored on overlapping rows.

The package-native implementation already has the required semantics. `PackageStageSchedule`
registers nested 8/32/80/20 membership, and `fit_numerical_recipe` fits one policy on all Train rows
plus five policies that each exclude one held-out entity group. Reusing these components avoids a
second, divergent implementation.

## Active entry point

Add `numerical_agent.run_train80_numerical_evolution` and make
`scripts/run_champion_evolution.sh` invoke it.

The new command consumes:

- a clean reviewed Python portfolio: `methods.py`, `policies.py`, `dictionary.py`, and optional
  `skills.py`;
- the exact registered 80/20/99 split and task source;
- an explicit immutable Parent Champion seed;
- the configured Statistical and TSFM runtimes;
- the proposer model/configuration, forecast store, output directory, and authority directory.

It constructs the same label-free task views, Train rows, entity-grouped fold manifest,
`NumericalPackageProposer`, materializer, and package stage gates used by formal package
co-evolution. Only the Numerical coordinate runs. Retrieval and Decision are neither constructed
nor called.

## Outputs and downstream handoff

The accepted result publishes a frozen Numerical supply bundle containing:

- the immutable Parent Champion anchor;
- fitted full-Train and per-fold recipe policies;
- candidate-bound structural assumptions;
- source, runtime, Dictionary, split, fold, metric, and proposer fingerprints;
- an exact Parent fallback when Train or Dev rejects the Child.

The package co-evolution runner gains an explicit `--numerical-supply-release` input mode for this
frozen Numerical supply bundle. In that mode it verifies and materializes the supplied bundle
rather than rebuilding it from the historical standalone Champion path. Existing callers that
provide the old Parent Champion seed remain a report/replay compatibility boundary, but the active
`scripts/run_package_coevolution.sh` path uses the new supply input. This change does not claim that
a single Dictionary-to-Package orchestration script already exists.

No lossy conversion from a package-native Numerical supply release back into the older standalone
Champion schema is allowed.

## Protocol and acceptance

- The 8-task and 32-task screens select whole entity groups and are nested inside Train-80.
- Numeric thresholds and weights are fitted by Host code, never supplied as free-form LLM values.
- Every Train task is held out exactly once by the five-fold manifest.
- The Train finalist must pass the existing joint capped/raw sMAE and sRMSE, coverage, fold,
  clipping, tail, catastrophe, and Parent-preservation gates.
- Only one frozen Train finalist may open Dev.
- Dev is read-only accept/reject evidence and cannot cause another proposal.
- Public-99 is not loaded by this runner.
- Rejection preserves the exact Parent bytes.

## Compatibility and migration

- New manifests and checkpoints use a new schema/protocol fingerprint naming the Train-80 OOF
  schedule.
- A 64/16 checkpoint, authority record, cache entry, or release cannot be interpreted as a
  Train-80 artifact.
- The old Python implementation remains available only to inspect or reproduce named historical
  runs. It is not the default CLI or shell-wrapper target.
- Documentation labels 64/16 only as historical and never as the current formal flow.

## Failure behavior

The new command fails closed before model or Dev access when the portfolio is dirty, the split or
task universe is wrong, the Parent seed or frozen supply is malformed, a required runtime is
unavailable, fingerprints drift, or an old checkpoint is presented. A malformed or invalid Child
is rejected without aborting sibling candidates. Runtime or gate failure never silently switches
back to the legacy controller.

## Testing

Tests must establish the following behavior before production changes are accepted:

1. The active wrapper invokes the Train-80 runner and never the legacy standalone module.
2. Formal scheduling is exactly 8/32/80/20 and rejects `build64` or `calibration16` stages.
3. All 80 Train tasks occur in the fold manifest and each appears in exactly one held-out fold.
4. Fold-specific fitting excludes the held-out entity group.
5. Only one Train finalist reaches the read-only Dev gate.
6. Train or Dev rejection preserves the exact Parent release.
7. Retrieval, Decision, and Public providers are never invoked.
8. Old 64/16 checkpoints and manifests fail before runtime access.
9. The frozen Numerical supply output can be consumed by package co-evolution without a lossy
   schema conversion.
10. Focused Numerical/package tests, static checks, shell syntax checks, and the relevant broader
    suite pass before completion.

## Files in scope

- `numerical_agent/run_train80_numerical_evolution.py` (new)
- `scripts/run_champion_evolution.sh`
- `evolving_loop/run_package_coevolution.py`
- existing package Numerical/schedule modules only when a small reusable public boundary is needed
- focused runner, package handoff, scheduling, and wrapper tests
- README/HTML protocol wording required to keep the active command accurate

The legacy standalone controller is not rewritten or deleted in this change.
