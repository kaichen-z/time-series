# Reproducible Three-Agent Co-Evolution Bundle Design

## Objective

Evolve Coding, Retrieval, and Decision as one reproducible contextual time-series harness. Every
accepted generation must preserve the exact prompts, executable budgets, topology, evidence policy,
and validated reusable skills that produced its development score.

## Scientific boundaries

- Coding remains `llm_only`: it sees historical numbers, timestamps/frequency, horizon, its numeric
  skill library, and historical hindcast diagnostics. It never sees documents or resolved labels.
- Retrieval sees the document corpus and Coding hypotheses. Its existing typed impact ledger remains
  the context contract: mechanism layer, temporal relationship, direction, permanence, magnitude,
  time window, and verified citations.
- Decision sees executed candidates, their hindcast diagnostics, typed verified context, and its
  validated skills. It cannot invent an unexecuted trajectory.
- Resolved future values and document annotations are exposed only to the trusted public-task scorer,
  post-outcome skill curator, and Meta-Harness failure summary.
- Train, development, and holdout partitions remain entity-disjoint. Holdout is never used for
  mutation or acceptance.

## Policy bundle

`HarnessPolicy` is the complete inheritable artifact. In addition to its current prompts, budgets,
workflow, evidence adjustment settings, and aggregation rule, it stores immutable snapshots of:

- Coding executable skills;
- Retrieval strategies;
- Decision selection rules.

The Meta-Harness may mutate prompts, budgets, and topology, but it may not directly fabricate skill
records. Skills enter a bundle only through existing post-outcome validation gates. The mutation
prompt receives compact skill inventories rather than full Python source to control context size.

For every policy evaluation, the parent and each child begin from their own policy snapshot, learn
from the same resolved training partition, and freeze the resulting learned snapshot before
development evaluation. The development score and saved artifact therefore refer to the same state.
If a child is rejected, the trained parent snapshot remains the generation incumbent so the emitted
artifact still reproduces the evaluated parent.

## Candidate diagnostics

Forecast reward remains the inheritance objective, with Retrieval quality as the existing secondary
term. Evaluation additionally reports diagnostics that localize failure without replacing the
end-to-end gate:

- mean final sMAPE;
- mean Best-of-K/oracle candidate sMAPE;
- mean Decision selection regret (selected minus Best-of-K);
- mean number of executed candidates;
- mean Spearman correlation between historical hindcast ranking and resolved-future ranking.

Per-task failure traces retain every candidate's assumption, hindcast sMAPE, resolved sMAPE, oracle
candidate, and selected candidate. These diagnostics distinguish generation failure from selection
failure.

## Co-evolution generation

The primary experiment uses `genome` with `target=auto`, which may redesign mutually dependent
fields across all three roles. Targeted modes remain diagnostic ablations only.

Each child request receives a stable child index and an explicit diversity instruction so multiple
children do not collapse to one cached proposal. A generation proceeds as follows:

1. Evaluate and train the parent on the training split, then snapshot its learned skills.
2. Generate structurally distinct complete child genomes from parent failure traces.
3. Evaluate every child on the same training tasks and snapshot each child's learned skills.
4. Prune children that do not strictly improve training system reward.
5. Evaluate only the best improving child on the disjoint development split.
6. Accept it only on strict development improvement; otherwise retain the trained parent bundle.
7. Keep holdout sealed for one frozen confirmation run.

## Pilot protocol

The first complete run uses a frozen 30-task public subset with an entity-disjoint split of roughly
18 train, 6 development, and 6 holdout tasks. It runs one Genome generation with at least two diverse
children before increasing the generation budget. The pilot report must include policy version,
bundle skill counts, train/development rewards, candidate diagnostics, and untouched holdout status.

Current sMAPE is a development metric, not the final Dr-CiK paper metric. Official probabilistic
sCRPS requires sampled forecast trajectories and remains a separate subsequent deliverable.
