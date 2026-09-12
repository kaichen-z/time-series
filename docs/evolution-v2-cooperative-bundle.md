# Evolution V2 Cooperative Bundle Prototype

Project 3 evaluates one immutable Project 2 Numerical Supply together with
mutable Retrieval and Decision modules as an Evolution V2 Bundle.  Each
scheduled Child uses the complete Numerical -> Retrieval -> Decision pipeline;
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

## Scope and interpretation

UCB and seeded Thompson schedule the four mutation scopes: `numerical`,
`retrieval`, `decision`, and `joint`. A joint Child atomically changes at least
two principal scopes. Scheduler and proposal feedback contain Train aggregates
only; Dev metrics are acceptance evidence, not optimization input.

The deterministic 4/1 smoke is CI evidence. An injected-host 8/2 pilot is an
optional experiment. 80/20 and four-hour runs are optional experiments, not
completion gates.

This prototype excludes DGM and L1 evolution, Public scoring, per-forecast
persistence, and production filesystem-attack testing.
