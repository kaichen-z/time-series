# P3 Dictionary-Only Numerical Selection Design

## Status

Approved on 2026-09-15. This design replaces the active P2-frozen-selection
handoff. Old frozen artifacts remain readable for inspection, but every new
real Evolution V2 run uses the Dictionary-only P3 path.

## Goal

Separate creation from use:

```text
P2 Dictionary self-evolution
  -> complete frozen Dictionary / QD archive
P3 cooperative evolution
  -> Numerical Selector + Retrieval + Decision
  -> complete-pipeline evaluation
```

P2 creates diverse executable Numerical capabilities. P3 learns, per task,
which of those capabilities to use with Retrieval and Decision.

## P2 Output

P2 preserves every feasible, safe QD elite needed to represent occupied
history-only morphology cells. It does not project those elites into final
task-local packages for P3 and does not reject a niche specialist merely
because it is weaker as a global standalone forecast.

The canonical P2 handoff binds:

- the complete eligible Dictionary and executable source identities;
- the protected Anchor;
- the QD archive snapshot and member provenance;
- safety, applicability, runtime, and protocol identities; and
- `public_test_accessed=false`.

## P3 Numerical Coordinate

The P3 numerical coordinate is a canonical Selector Genome, not a choice
among P2-preselected frozen registries. Its evolvable fields cover:

- shortlist target size in the closed interval `[6, 8]`;
- history-only ranking-feature weights and morphology routing;
- family-diversity preference;
- specialist improvement and worst-regret thresholds within Host bounds;
- Anchor-heavy ensemble preferences; and
- a bounded Selector proposal prompt or typed mutation policy.

The Selector receives the same frozen complete Dictionary for the entire P3
run. A changed Selector Genome is materialized into a new content-addressed
task-local Numerical registry for evaluation; the Dictionary itself does not
change inside P3.

## Per-Task Data Flow

For each task:

1. Compute descriptors from history, frequency, and horizon only.
2. Exclude ineligible or unsafe Dictionary members.
3. Apply the Selector Genome's cheap ranking and select 6--8 unique candidates,
   including the Anchor. If fewer than six safe candidates exist, preserve the
   available set and record `shortlist_underfilled`.
4. Run paired history-only hindcast only for shortlisted candidates.
5. Apply finite-output, paired-coverage, improvement, and regret gates.
6. Materialize Anchor alone or Anchor plus at most two specialists, with
   Anchor weight at least `0.5`.
7. Run Numerical -> Retrieval -> Decision and score the complete output.

Forecast and diagnostic results are cached by task-history, Dictionary member,
fold manifest, and runtime identities so Selector mutations can reuse unchanged
work.

## Cooperative Evolution

P3 evaluates Numerical Selector, Retrieval, and Decision artifacts as one
bundle. Single-coordinate children change exactly one of those artifacts;
joint children change at least two. Acceptance uses complete-pipeline Train
results and the existing Dev promotion boundary.

Only sanitized aggregate Train feedback reaches proposers. Retrieval and
Decision cannot inspect future values, task labels, raw residuals, or use
semantic context to alter the history-only Numerical shortlist.

## Fixed Host Invariants

The following are not evolvable:

- Anchor inclusion;
- shortlist maximum of eight;
- no more than two active specialists;
- Anchor weight of at least `0.5`;
- history-only descriptors, selection, and paired hindcast;
- common fold manifests, canonical metrics, finite-output checks, and verifier;
- Train/Dev/Public isolation; and
- content-addressed artifact validation and promotion.

## Compatibility and Migration

New runs have one active mode: `p3_dictionary`. The prior P2-frozen bundle
schema remains accepted only by explicit legacy readers and reporting tools.
It is not admitted as a P3 proposal, seed, fallback, or silent downgrade.

Resume requires the same mode, Dictionary identity, Selector identity, task
split, runtime identities, and protocol fingerprint. An old run cannot resume
as a Dictionary-only run; it must start in a new output directory.

## Artifacts

P3 persists, directly or by content digest:

- the Selector Genome and parent lineage;
- the complete Dictionary identity;
- each task's ordered shortlist and exclusion reasons;
- hindcast diagnostics and failure reasons;
- selected Anchor/specialists and weights;
- the materialized Numerical registry;
- Retrieval and Decision identities; and
- Train/Dev complete-pipeline evaluation evidence.

## Verification

Automated tests must prove:

1. new P2 completion exposes the complete safe QD Dictionary without a
   task-local final-bundle authority;
2. P3 rejects a new-run configuration that attempts the old frozen mode;
3. different Selector Genomes can produce different task-local shortlists from
   the same Dictionary;
4. changing future values cannot change a shortlist or hindcast inputs;
5. no task evaluates more than eight shortlisted candidates;
6. final Numerical packages satisfy the Anchor and specialist constraints;
7. Selector-only and joint children have correct ownership and lineage;
8. cache reuse does not change identities or results;
9. interrupted and resumed runs are identical; and
10. the real smoke path completes P2 -> P3 -> P4 -> P5 without Public access.

## Non-Goals

- Letting the Decision Agent freely choose Dictionary methods.
- Evolving forecast backbones, metrics, verifier, or promotion authority.
- Running every Dictionary member through hindcast for every task.
- Converting old run directories in place.
