# Target-Wise Forecasting Method Evolution Design

## Goal

Produce independently testable forecasting-method children without rerunning every unchanged
method, and accept a child only when its executable behavior improves held-out validation under
the intended method identity.

## Current Failure Mode

The current two-stage loop evaluates all 93 methods on all 16 training tasks, asks one selector
for up to ten targets, and asks one mutator for a single batch of operations. The complete batch
is one atomic child. One invalid operation rejects every otherwise useful operation. Repeating a
generation also recomputes 1,488 unchanged method-task outcomes.

The first observed child exploited these properties by replacing six named algorithms with
simpler forecasts and deleting four specialists. Although several individual method scores
became finite, the four-task mini-dev oracle MASE regressed from 1.199123 to 1.252889. Identity
and deletion gates now prevent this behavior, but the atomic mutator consequently returned an
empty operation set.

## Scope

This phase changes the evolution loop, not the 93 method implementations:

- cache Parent outcomes by method source hash, task fingerprint, and execution configuration;
- select targets once, then mutate and evaluate one target at a time;
- generate one independent child operation per target;
- screen only the changed or added method before full validation;
- combine cached Parent outcomes with changed-method outcomes to compute child portfolio metrics;
- keep the existing monolithic `methods.py` format for compatibility;
- expose a repository adapter so a later phase can move methods to `methods/<name>.py` without
  changing the evolution engine.

One-method-per-file migration and a learned history-only router are deliberately deferred. The
engine will record both oracle portfolio metrics and per-method metrics so selector-aware reward
can be added without changing stored outcome records.

## Architecture

### Outcome Cache

`OutcomeCache` stores one JSON record per deterministic evaluation key:

```text
sha256(method source + task payload + isolated flag + evaluator schema version)
```

The cached value is the complete `Outcome` payload required to reconstruct reports and portfolio
MASE. Cache writes are atomic. Invalid JSON, a schema mismatch, or a hash mismatch is a cache miss,
never a run failure.

### Target-Wise Mutation

The selector returns at most three unique targets. Each target includes an explicit allowed action
set. For a verified repairable method the set may be `repair` and `fork`; a fork-only method may
only be forked; delete remains an exact action and retains its evidence gate. Multi-method merge
is deliberately batch-only in v1 because it is not an independent one-target mutation.

Each target is sent to the mutator separately. The mutator returns zero or one operation. The
controller validates and applies that operation to an isolated in-memory module, so one invalid
target cannot reject another target.

### Successive Screening

For each valid child operation:

1. Execute only changed or newly added methods on four deterministic screen tasks.
2. Reject crashes, invalid outputs, or an identity/coverage regression.
3. Execute those methods on the remaining training tasks.
4. Reconstruct the complete child outcome matrix from cached Parent outcomes plus changed outcomes.
5. Evaluate the child on the four mini-dev tasks.
6. Accept only if mean and median oracle MASE do not regress and at least one improves, or the
   operation is a repair whose own applicable-task MASE improves without reducing coverage.

Deletion is evaluated by removing the method's cached outcomes and running the same portfolio
ablation. A deletion cannot use `NotApplicable` or coverage below 0.5 as evidence.

### Promotion

Accepted operations are ranked by mini-dev improvement and applied one at a time to the current
Parent. After each promotion, affected cache entries are recomputed. Each accepted target becomes
its own Git commit. Rejected candidates leave the Parent and Git history unchanged.

## Interfaces

```python
class OutcomeCache:
    def evaluate_method(
        self,
        method: MethodDefinition,
        tasks: Sequence[Task],
        *,
        isolated: bool,
    ) -> tuple[Outcome, ...]: ...

@dataclass(frozen=True)
class TargetProposal:
    name: str
    allowed_actions: tuple[str, ...]
    reason: str

@dataclass(frozen=True)
class CandidateResult:
    target: TargetProposal
    operation: Mapping[str, object] | None
    accepted: bool
    reason: str
    train_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float]
```

The existing `evolve_once` remains available. A new `evolve_targets_once` drives the incremental
path and is selected by a CLI flag until the new path has passed the real 20-task experiment.

## Safety and Reproducibility

- Future values are used only by the trusted evaluator, never included in LLM prompts.
- Parent source, target selection, mutator response, cache key, and validation result are traced.
- Cache entries are content-addressed and cannot be reused after source or task changes.
- Same-name repairs retain the verified AST identity skeleton.
- A fork must use a new honest name and preserve the Parent.
- Duplicate selector targets and action escalation are rejected.
- A rejected target cannot modify the repository or suppress another candidate.

## Acceptance Criteria

- Repeating an unchanged Parent evaluation yields cache hits for every method-task pair.
- Two targets produce two isolated candidate results; one rejection does not block the other.
- Mutator output contains at most one operation and stays within the target's allowed actions.
- The 16-train/4-mini-dev split remains unchanged.
- No accepted child regresses both mean and median mini-dev MASE.
- Focused evolution tests, compile checks, shell syntax checks, and diff checks pass.
- A real 20-task run records elapsed time, cache-hit rate, candidate decisions, and final Parent.
