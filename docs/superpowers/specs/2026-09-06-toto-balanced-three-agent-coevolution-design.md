# Toto-Balanced Three-Agent Co-Evolution Design

**Date:** 2026-09-06

**Status:** Approved by user in chat

**Scope:** Run the package-native Numerical, Retrieval, and Decision co-evolution
on the frozen Toto-balanced v3 split, repair only evidenced blockers, and evaluate
the sealed bundle once on the registered Public-99 regression partition.

## 1. Objective

Execute and verify one complete two-cycle coordinate evolution:

```text
Numerical 1 -> Retrieval 1 -> Decision 1
Numerical 2 -> Retrieval 2 -> Decision 2
```

Each coordinate proposes three Children. Only one coordinate changes at a time,
and every Child is judged by the final forecast of the complete three-agent
pipeline. Task-level, host-validated Retrieval evidence from an accepted cycle is
available to the next Numerical proposal stage.

## 2. Frozen experiment authority

- Split: `splits/drcik_public_80_20_99_v3.json`
- File SHA-256:
  `e2cc50dc95b3bdcd0dcc4b0df8d7eac3ff59ad9ce18268aa3d663e182733307d`
- Evolution: all 80 Train tasks, with nested 8/32 screens and entity-grouped
  five-fold cross-fit, plus one read-only 20-task Dev acceptance gate.
- Final regression: 99 Public tasks, loaded only after a final bundle is sealed.
- Initial Numerical release: the frozen Toto safe-anchor release produced by the
  completed v3 Numerical run.
- Initial Retrieval release and Skill snapshot: the existing package-native
  `v000` seed.
- Model: `gpt-5.6-luna`, reasoning effort `medium`.
- Seed: `20260903`.

The v3 split was deliberately stratified using history values, future values, and
Toto error. This balances baseline difficulty across partitions, but it means the
Public-99 partition is a frozen, label-informed regression set rather than a
previously untouched test set. No result from it may be used to modify or rerun a
bundle. Claims of unseen generalization require the official Dr-CiK hidden test.

## 3. Execution protocol

1. Create an isolated implementation worktree from the current commit.
2. Run focused baseline tests for the package co-evolution path.
3. Run a minimal real-model smoke using the v3 authority.
4. If the smoke fails, identify the root cause and add a failing regression test
   before changing production code. Do not relax gates or artifact validation.
5. Run the formal two-cycle experiment with three Children per coordinate and
   `feedback-mode=task`.
6. Verify the completion marker, six coordinate trace records, fingerprints,
   stage access, replay evidence, and zero Public access during evolution.
7. Seal the selected final bundle and invoke the separate Public evaluator once.
8. Report nested-system metrics, coordinate decisions, fallback/failure counts,
   and Public-99 regression metrics.

Run and authority directories are fresh and immutable. Forecast, Retrieval, and
Decision caches may be reused only through their content-addressed identities.

## 4. Acceptance and stopping

The existing package-native gates remain authoritative:

- 8-task and 32-task screens precede the full 80-task Train cross-fit gate.
- Train cross-fit requires full coverage, bounded regret, tail non-regression, at least
  0.5% joint gain, and the configured fold-stability gate.
- The unchanged Train finalist must then pass Dev-20, the only independent
  acceptance gate. Dev task-level outcomes never enter proposal feedback.
- Acceptance changes exactly one principal module fingerprint and preserves
  direct lineage.
- A full Numerical-Retrieval-Decision cycle with no accepted coordinate stops
  evolution early.

No proxy Retrieval score, Numerical oracle score, or Decision regret can replace
an improvement in the complete final forecast.

## 5. Paper-alignment audit

The accompanying review will use primary papers and official code to assess:

- whether context is retrieved with point-in-time and target/entity controls;
- whether evidence changes forecasts through an explicit, falsifiable mechanism;
- whether the candidate-selection boundary prevents free-form numerical edits;
- whether coordinate ascent gives useful credit assignment without hiding
  cross-coordinate interactions;
- whether repeated Dev decisions create adaptive overfitting risk;
- whether baseline-relative split stratification changes the interpretation of
  the final result; and
- which conclusions require an untouched hidden test or a nested resampling
  ablation.

The audit may recommend a future Train-only paired-coordinate ablation, but it
will not change this registered run after Public-99 is opened.

## 6. Deliverables

- focused regression tests for any required repair;
- committed minimal code fixes, if any;
- complete v3 co-evolution and authority artifacts;
- one sealed Public-99 evaluation directory;
- an evidence-backed design review with links to primary sources; and
- a concise result summary distinguishing development, frozen regression, and
  genuinely hidden-test claims.
