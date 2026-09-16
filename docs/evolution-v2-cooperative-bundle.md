# Evolution V2 Cooperative Bundle Prototype

New real runs use [Decision-owned Dictionary execution](decision-owned-dictionary.md):
P2 supplies the full frozen catalog, Retrieval supplies evidence, and Decision
requests cached numerical evaluations and chooses methods and weights. The
bounded Selector description below documents the older standalone path.

Project 3 evaluates the immutable complete Project 2 Numerical Dictionary
together with mutable Retrieval and Decision modules as an Evolution V2 Bundle.
Its Numerical Selector filters that Dictionary to at most eight history-only
candidates per task, then materializes Anchor plus zero, one, or two
specialists (Anchor weight at least 0.5). Each scheduled Child uses the complete Numerical -> Retrieval -> Decision pipeline;
the Host evaluates Parent and Child on the same Train tasks, opens Dev only
after the Train gate, and promotes only a Train-eligible Child with a strictly
better Dev joint mean. Public tasks are rejected before a proposal is made.

## Smoke commands

Build (or confirm) the canonical 4 Train / 1 Dev fixture, then use a new empty
output directory for each command:

```bash
python tests/build_evolution_v2_cooperative_fixture.py

python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/cooperative/smoke-ucb.json \
  --seed-supply tests/fixtures/evolution_v2_cooperative/seed_supply.json \
  --task-manifest tests/fixtures/evolution_v2_cooperative/tasks_4_1.json \
  --retrieval-release tests/fixtures/evolution_v2_cooperative/retrieval_release.json \
  --decision-policy tests/fixtures/evolution_v2_cooperative/decision_policy.json \
  --output-dir /tmp/evolution-v2-cooperative-ucb

python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/cooperative/smoke-thompson.json \
  --seed-supply tests/fixtures/evolution_v2_cooperative/seed_supply.json \
  --task-manifest tests/fixtures/evolution_v2_cooperative/tasks_4_1.json \
  --retrieval-release tests/fixtures/evolution_v2_cooperative/retrieval_release.json \
  --decision-policy tests/fixtures/evolution_v2_cooperative/decision_policy.json \
  --output-dir /tmp/evolution-v2-cooperative-thompson
```

Re-run either exact command to resume; a completed output is a read-only
no-op. `cooperative_checkpoint.json` binds the active Bundle and scheduler
state, while `objects/` stores canonical module, scheduler, feedback, and
aggregate-evaluation objects. Progress advances only at closed candidate
boundaries.

### Real cooperative task progress

`run_real_cooperative` enables optional per-task persistence at
`<output_dir>/task_evaluations/<identity_sha256>.json`. Each record has
`identity`, `started_at`, `updated_at`, and `status` (`started`, `completed`,
or `error`). Completed records contain a typed score, numeric diagnostics,
original provisional/final Decision rejections, and Dictionary traces with
phase responses and failed tool requests/errors. Error records retain the
exception type and message. Rejected nonfinite tool inputs are stored as text;
score validation remains unchanged.

Only completed records are reused. The key binds the full task (including
Host-only labels), Numerical package, Bundle, stage, Retrieval/Decision
identities, Host runtime and model, metric policy/cap, and explicit execution
contract version. Bump `TASK_EXECUTION_CONTRACT` in `package_task_store.py`
when changing execution/scoring semantics not covered by another key field.
Interrupted tasks rerun; invalid cached scores fail closed. This single-writer
store is local diagnostic/evaluation state, not sealed promotion evidence.
It does **not** reopen a root run terminalized by a transient failure.

For a standalone, already-authorized canary using a configured adapter:

```python
from evolving_loop.package_task_store import PackageTaskStore

pipeline.task_store = PackageTaskStore(output / "task_evaluations",
                                      runtime_identity=runtime_identity)
evaluation = pipeline.evaluate(bundle, (resolved_train_task,), stage="train")
```

Use an identity covering the actual runtime/model. Generic adapters leave
this feature disabled; full-result `trace_sink` callbacks cannot be combined
with score-only reuse. No records or evaluation labels are passed to agents.

## Scope and interpretation

The active handoff is `P2 full executable Dictionary -> P3 Dictionary Selector
and cooperative Bundle`. Older frozen-supply artifacts remain readable only as
legacy inputs/reporting evidence. P2/P3 use schema-v2, history-only evidence:
future values, labels, raw forecasts, and Public data stay behind the Host.
The AlphaEvolve-style program context, Voyager-style curriculum/reuse, and
PromptBreeder-style task/mutation prompt lineages are mechanism-level
integrations, not faithful paper reproductions. Metrics, verifier, split
manifests, runtime identities, budgets, artifact validation, and promotion
authority remain fixed Host boundaries.

UCB and seeded Thompson schedule the four mutation scopes: `numerical`,
`retrieval`, `decision`, and `joint`. A joint Child atomically changes at least
two principal scopes. Scheduler and proposal feedback contain Train aggregates
only; Dev metrics are acceptance evidence, not optimization input.

The deterministic 4/1 smoke is CI evidence. An injected-host 8/2 pilot is an
optional experiment. 80/20 and four-hour runs are optional experiments, not
completion gates.

This prototype excludes DGM and L1 evolution, Public scoring, per-forecast
persistence, and production filesystem-attack testing.
