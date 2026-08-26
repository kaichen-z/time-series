# Assumption-Guided Top-k Numerical Selector

## Goal

Replace direct 103-candidate ranking with a history-only reasoning layer:

`TaskProfile -> falsifiable assumptions -> diverse Top-k -> Verifier -> Toto protection`.

The first implementation is deterministic and typed. It does not call an LLM per task and never
reads documents, retrieved evidence, future labels, or GT evidence. The existing Meta-Harness may
evolve only approved policy parameters from aggregate Train results.

## Assumption contract

Each assumption contains:

- a stable identifier and kind;
- a falsifiable claim;
- supporting history-only signals;
- an explicit failure condition;
- compatible active candidate names;
- a confidence in `[0, 1]`.

The initial kinds are foundation-shape, periodic persistence, trend persistence, recent-regime,
intermittent-demand, stationary-local-dynamics, and robust fallback. A kind is emitted only when
its TaskProfile preconditions are met, except the foundation and robust fallbacks.

## Top-k contract

Every emitted assumption is evaluated using existing historical hindcast diagnostics. Its leading
candidate is selected by worst-fold MASE, median MASE, recent MASE, instability, and name. Greedy
Top-k selection rejects duplicate assumption kinds and duplicate leading candidates. The stable
Toto/TimesFM anchors are appended to the candidate pool even when no assumption ranks them first.

Top-k controls reasoning breadth, not final ensemble size. The final Verifier may still emit one
forecast or an existing guarded two-method combination.

## Verifier and safety

The Verifier is the existing numerical selector restricted to candidates justified by Top-k
assumptions. Existing eligibility, catastrophic-tail, fold-win, worst-fold-regret, and baseline
protection checks remain authoritative. The decision trace records the retained assumptions.

## Evolution boundary

DecisionPolicy gains typed parameters for enabling assumption guidance, Top-k size, candidates per
assumption, and minimum assumption confidence. The Meta-Harness may evolve those parameters on
Train. Entity-grouped Train cross-fold checks and one read-only Dev gate still decide survival.

## Evaluation

- Develop on the frozen 80 Train / 20 Dev partition.
- Primary metrics: clipped Dr-CiK sMAE and sRMSE.
- Safety: coverage, clipped counts, P90/P95 sMAE, and active-oracle regret.
- Public 99 is not accessed in this iteration.

