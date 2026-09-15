# Task-local Numerical Shortlist Design (legacy lower-half contract)

The active architecture is now the schema-v2 Dictionary-only path described in
`docs/superpowers/specs/2026-09-15-p3-dictionary-only-design.md`: P2 seals the
complete executable Dictionary and P3 performs the task-local filter. This
document remains useful for the shortlist and Anchor constraints; statements
that assign shortlist authority to P2 are historical and superseded.

## Goal

Replace the globally bounded Numerical supply path with a complete global
Dictionary followed by a bounded, history-only shortlist for each task:

```text
complete global Dictionary
        -> per-task history-only shortlist (at most 8, including Anchor)
        -> task-local hindcast
        -> Anchor + zero, one, or two specialists
```

The protected Anchor remains the output whenever local evidence is missing,
invalid, unstable, or insufficiently better.

## Current mismatch

The repository already has most of the desired lower half:

- `FilterDictionary` indexes the executable statistical, TSFM, and combined
  candidates.
- task-local diagnostics run hindcast folds.
- `TaskLocalTournamentPolicy` already limits the final blend to at most two
  specialists and gives the Anchor at least half of the weight.

The upper half is incorrect for this design:

- the legacy `NumericalSupplyRelease` permits at most four alternatives and at most one
  per family.
- `fit_group_candidate_supply` constructs a global or morphology-group list
  of at most eight candidates.
- a task therefore inherits a group shortlist rather than independently
  selecting candidates from the complete Dictionary.

## Chosen architecture

### 1. Complete global Dictionary

The global Dictionary is the catalog of every executable candidate. It has no
cardinality limit and preserves, for every entry:

- candidate identity and family;
- executable/source identity;
- safety status (`keep`, `specialized`, `repair`, `quarantine`, `discard`);
- history-only applicability constraints;
- provenance and runtime commitments.

The Dictionary does not choose the final forecast. Quarantined, repair, and
discarded entries remain recorded but are not eligible for a task shortlist.

`NumericalSupplyRelease` must stop acting as a four-slot global Dictionary.
The release will bind the complete candidate catalog plus the protected Anchor;
family diversity becomes a ranking preference, not a one-per-family schema
restriction.

### 2. Per-task history-only shortlist

For each task, the Host constructs a `TaskCandidateShortlist` from the complete
Dictionary using only information available before the future:

- task frequency, history length, horizon ratio, trend, periodicity,
  intermittency, zeros, sign, and recent-regime features;
- Dictionary safety status and applicability;
- Train-fitted candidate priors and failure/coverage statistics;
- deterministic family-diversity tie breaking.

The active P3 shortlist contains at most eight candidates including the Anchor.
Its target size is eight. It may be underfilled when fewer safe candidates are
eligible and executable; this is recorded as `shortlist_underfilled`, never
filled with unsafe candidates. The Anchor is always first. Candidate
identities are unique, and no future values, task labels, Dev outcomes, or
Public evidence may influence selection.

The shortlist is task-specific. Morphology groups may provide Train-fitted
priors, but they may not replace the final per-task selection step.

### 3. Task-local hindcast

Only shortlist members are materialized and diagnosed for that task. Each
candidate must produce enough successful paired hindcast folds against the same
truth windows as the Anchor. Failed, non-finite, dominated, excessively
regretful, or incomplete candidates are excluded from specialist selection.

Shortlist construction and hindcast evaluation are distinct artifacts so the
system can determine whether a method was never eligible, shortlisted but
failed, or evaluated and rejected.

### 4. Anchor-heavy final selection

Reuse the existing task-local tournament:

- rank hindcast-safe specialists by median and worst paired error;
- prefer family diversity when choosing specialists;
- enumerate Anchor-heavy weight combinations;
- require Pareto-safe improvement and bounded worst-fold regret;
- select at most two specialists;
- require Anchor weight of at least 0.5;
- fall back exactly to the Anchor on any uncertainty.

The final result is therefore either Anchor alone, Anchor plus one specialist,
or Anchor plus two specialists. A specialist-only forecast is forbidden.

## Artifact and interface changes

Add a canonical per-task shortlist artifact containing:

- task-input fingerprint;
- complete Dictionary fingerprint;
- shortlist-policy fingerprint;
- ordered candidate identities;
- exclusion reason codes and underfilled status;
- `public_test_accessed=false`.

`TaskLocalEnsembleRelease` will bind the shortlist policy and Train-fitted
ranking priors rather than storing the old global/group candidate supplies as
the runtime selection authority. Existing releases remain readable through an
explicit legacy parser path; new releases use a new schema version and cannot
silently downgrade to group selection.

The P2 frozen numerical registry stores each task's shortlist identity,
hindcast diagnostics, selected specialists, and final weights. P3 performs the
sole active task-local filter from the complete Dictionary; legacy frozen
registries remain readable for inspection only.

## Evolution boundary

P3 may evolve over the complete P2 Dictionary:

- the history-only shortlist scoring policy;
- Train-fitted ranking priors;
- task-local tournament thresholds.

P2 still evolves Dictionary membership, status, applicability, and reusable
program provenance, but it does not emit a task-local shortlist as runtime
selection authority.

P2 may not evolve away:

- the protected Anchor;
- the at-most-eight shortlist ceiling and maximum two specialists;
- history-only selection;
- paired-hindcast comparability;
- finite-output and regret gates;
- Public isolation.

## Validation

Tests will establish:

1. a Dictionary with more than four alternatives round-trips canonically;
2. two tasks with different histories can receive different shortlists;
3. every shortlist contains the Anchor, contains at most ten candidates, and
   targets at least six without admitting unsafe candidates;
4. shortlist construction is unchanged when future labels are mutated;
5. hindcast executes only shortlisted candidates;
6. final selection is Anchor plus at most two specialists with Anchor weight at
   least 0.5;
7. missing/failed diagnostics fall back exactly to the Anchor;
8. legacy releases remain explicitly readable while new releases use the
   task-local shortlist path;
9. P2 frozen-registry and P3 handoff identities close over shortlist artifacts;
10. Public membership remains empty throughout.

## Non-goals

- Running every global method through hindcast for every task.
- Using future values or Dev/Public labels to form a shortlist.
- Removing the Anchor or allowing specialist-only output.
- Reworking P3, P4, or P5 beyond adapting to the new frozen P2 artifact schema.
