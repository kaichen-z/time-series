# Joint-stage submission curve (Dr-CiK)

This export reconstructs the archived joint-stage optimization trajectory used by the current Dr-CiK reconstruction-v2 ablation.

## Phase boundaries

| Scheduled phase | Submission IDs | Candidates | Accepted | Best accepted visible fitness at phase end |
|---|---:|---:|---:|---:|
| Seed | 0 | 1 | 1 | 0.459272 |
| Numerical a | 1--10 | 10 | 3 | 0.468354 |
| Decision a | 11--13 | 3 | 2 | 0.469528 |
| Numerical b | 14--20 | 7 | 5 | 0.483166 |
| Decision b | none | 0 | 0 | 0.483166 |

The orchestrator scheduled four role phases (numerical a, decision a, numerical b, decision b), i.e. two alternating numerical-to-decision cycles. The last decision-b phase completed but produced no candidate submission. The run therefore contains 20 candidate submissions plus the seed, and only three phases have points on a submission-indexed curve.

## Plotting recommendation

Plot `candidate_visible_fitness` as light points and `best_accepted_visible_fitness` as a step line. Mark accepted candidates separately and draw phase boundaries after submissions 10 and 13. Annotate that decision-b produced zero submissions rather than silently omitting the scheduled phase.

The archived per-submission records expose the joint optimizer's scalar visible-selection fitness, not candidate-level sMAE/sRMSE/sCRPS. Do not relabel this curve as an error-metric curve. The seed-vs-final sMAE/sRMSE/sCRPS comparison remains Table 3 and comes from the separate same-protocol replay.

This is a descriptive trace of one completed run, not a mean with uncertainty bars; it does not require or imply three random seeds.

## Source provenance

- Orchestrator: `experiments/self_evolving/final_method/unified/coevo_drcik/run_phases.sh` (in this repo)
- Phase log: original run directory `coevo_x/runs/drcik/phases.log` (not included)
- State history: original run directory `coevo_x/runs/drcik/private/state.json` (private, not included)
- Per-submission records: original run directory `coevo_x/runs/drcik/shared/attempts/*.json` (not included)
