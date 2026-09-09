# Meta-Harness V2 — real 8/2 evolve report

Date: 2026-09-09 (Asia/Shanghai)

## Experiment

- Backend: `gpt-5.6-sol`, reasoning effort `medium`
- Data: 10 labeled Dr-CiK public tasks, entity-disjoint 8 Train / 2 Dev
- Evolution: one generation, four required Child shapes (`coding`, `retrieval`, `decision`, `joint`)
- Successive halving: 6-task Train screen, promote at most 2, complete 8 Train, then read-only 2 Dev gate
- Holdout/Public: not opened in this smoke (`holdout_fraction=0`)
- Measured generation time: 1:55:00 (from first to last progress event)

## Final result

The accepted policy is `v001`, a Coding-only Child. Retrieval, Decision, workflow,
aggregation, and evidence-adjustment behavior remain equal to Parent `v000`.

| Split | Policy | mean sMAE | mean sRMSE |
|---|---:|---:|---:|
| Train-8 | Parent v000 | 0.492059 | 0.745705 |
| Train-8 | Child v001 | 0.403046 | 0.611000 |
| Train improvement | | 18.09% | 18.06% |
| Dev-2 | Parent v000 | 1.576038 | 2.171386 |
| Dev-2 | Child v001 | 0.321387 | 0.689135 |
| Dev improvement | | 79.61% | 68.26% |

Train per-task win/tie/loss was 4/0/4 by sMAE and 5/0/3 by sRMSE. Dev was
1/0/1 on each metric: the large aggregate improvement is mainly from `task_51`,
while `task_128` regressed slightly. This is promising feasibility evidence, not a
Public-99 or hidden-test result.

## What the accepted Child changed

The Meta-Harness proposed the following falsifiable interaction hypothesis:

> Generating three structurally distinct numerical hypotheses and applying one
> fold-diagnostic-driven mutation will improve the chance that the unchanged
> retrieval and decision agents receive an executable candidate whose assumptions
> match the series, compared with the current single unmutated hypothesis.

The host accepted only the `coding` scope change. The new Coding prompt requests
three genuinely different inductive biases (robust local/damped trend,
frequency-aware seasonal/multi-lag, and adaptive selector/conservative ensemble),
then permits one causal fold-diagnostic-driven descendant. Every generated program
must carry a falsifiable assumption and an observable failure condition.

## All four Children

| Child | Scope | Screen/full outcome |
|---|---|---|
| v001 | Coding only | Screen improved; full Train improved; Dev improved; accepted |
| v002 | Retrieval only | Equal to Parent; promoted as control, then rejected before Dev |
| v003 | Decision only | Equal to Parent screen; not top-k |
| v004 | Joint | Malformed/out-of-contract proposal; atomically rejected, Parent preserved |

## Memory decision

No free-form Memory Agent or vector store was added. The checkpoint stores a
bounded, typed, Train-only memory of the last three generations. Each record may
contain only generation, Child kind, changed scopes, interaction hypothesis,
status, and Parent/Child Train sMAE+sRMSE. This run produced four records (about
2 KB). Audit confirmed there are no task IDs, future values, documents, Dev,
holdout, or ground-truth fields. A one-generation run records this memory; a later
generation can consume it.

## Reproducibility and artifacts

- `best_policy.json`: accepted full Harness Genome (`v001`)
- `evolution_trace.json`: Parent/Children, diagnostics, rewards, acceptance, and sanitized Train memory
- `checkpoint.json`: resumable schema-v2 incumbent and history
- `split_manifest.json`: exact entity-disjoint task membership
- `progress.jsonl`: append-only task/stage events
- `codex-cache/`: exact model outputs, including all four Meta-Harness proposals and task-level calls
- `run_command.txt`: exact command

The currently connected real runtime is the three-Agent EvolvingForecastHarness
(Coding + Retrieval + Decision). The newer 103-method Dictionary/Champion package
is deliberately not claimed as connected to this V2 run yet.
