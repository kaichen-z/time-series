# Evolution V2 Numerical QD operator guide

## Status and boundary

Project 2 implements resumable self-evolution of the Numerical Supply behind the
Evolution V2 Kernel. It evolves executable Numerical members, their typed
structure, the Train-credit mutation policy, the bounded proposer prompt, and a
MAP-Elites archive. A successful generation freezes an existing
`NumericalSupplyRelease` together with an exact
`FrozenNumericalPackageRegistry`; the Kernel promotes those two identities
atomically or retains the exact Parent bytes.

This is an intermediate Numerical-only command. It does **not** implement
Retrieval evolution, Decision evolution, joint Bundle evolution, the full
production `evolve` path, Public scoring, DGM mutation of Evolver/Harness Python
source, or the L1 protocol migration. Those remain Project 3, Project 4, and
Project 5 work as described under [Project handoff](#project-handoff).

## Prepare canonical inputs

The command requires three separate regular, non-symlink files:

1. A strict `NumericalQDConfigV2` profile.
2. A canonical legacy `NumericalSupplyRelease` seed.
3. A canonical task envelope containing exactly 80 Train and 20 Dev tasks plus
   the exact five-fold, entity-grouped Train manifest.

JSON must be canonical V2 JSON: UTF-8, sorted keys, compact separators, one
trailing newline, no duplicate keys, and no NaN or Infinity. The CLI hashes the
raw bytes of all three inputs before it creates output. Inputs must have distinct
filesystem identities and may not overlap the output in either direction.

For a deterministic local fixture:

```bash
fixture_dir=$(mktemp -d)
python tests/build_evolution_v2_numerical_fixture.py \
  --output-dir "$fixture_dir/inputs"
```

The builder creates `seed_supply.json` and `task_manifest.json`. It is a smoke
fixture, not an experimental result. Real inputs must preserve the same exact
schema and Train80/Dev20 authority boundary.

## Run and resume

Run deterministic smoke evolution with:

```bash
python -m evolving_loop.v2 numerical-evolve \
  --config configs/evolution_v2/numerical_qd/smoke.json \
  --seed-supply "$fixture_dir/inputs/seed_supply.json" \
  --task-manifest "$fixture_dir/inputs/task_manifest.json" \
  --output-dir "$fixture_dir/run"
```

Pass the same four paths again to resume. The CLI treats a nonempty output
directory as an exact resume request. A completed run is verified and returned
without changing bytes or nanosecond mtimes. A new run directory must be empty;
inside the repository, output is allowed only below
`runs/evolution_v2/<run-name>`.

The shipped profiles are:

| Profile | hard limit | reserve | proposer | purpose |
|---|---:|---:|---|---|
| `smoke` | 600 s | 20% | deterministic | offline reproducible wiring and algorithm exercise |
| `pilot` | 7,200 s | 20% | hybrid | Python Host integration profile; requires a forecast runtime |
| `formal` | 14,400 s | exactly 20% | hybrid | Python Host integration profile; requires a forecast runtime |

The config is closed. It binds the Kernel protocol, runtime fingerprints,
resource ceilings, fixed Retrieval/Decision/Harness/archive/scheduler
identities, descriptor thresholds, mutation limits, QD capacity, exact parent
sampling weights, exact Hyperband brackets, proposer limits, and adapter limits.
Changing any committed input or identity is not a resume.

## What evolves

The typed mutation grammar contains `add`, `repair`, `fork`, `combine`, `route`,
`specialize`, `crossover`, `remove`, `quarantine`, and `policy_tune`. Each tag
has a closed payload schema, bounded lineage and inventory growth, and explicit
field ownership. Candidate source is stored by SHA-256 and must contain one
bounded forecasting function with the exact `(history, horizon, frequency)`
signature. The shared parser, source checker, capability checker, and isolated
forecast runtime reject forbidden imports, dynamic execution, filesystem writes,
escape-hatch dunders, invalid signatures, missing documentation, nonfinite
forecasts, and wrong horizons.

The deterministic proposer emits legal structural repair/fork/combine/route,
specialization, removal/quarantine, and Host-generated prompt-policy variants
from the current inventory. It chooses from a canonical list using the persisted
counter draw. An LLM may additionally propose source-bearing mutations, but its
response passes the same typed parser and source ownership checks.

Mutation credit is integer Host state. For an operator, the Host increments
attempts, feasible outcomes, Hyperband promotions, archive insertions, and
credit; `credit += 1` only for a feasible Train insertion. A successful Train
insertion may produce a bounded Host-written prompt variant. Dev acceptance and
all Public information are excluded from policy and prompt memory.

## Information boundary

The proposer receives only canonical primitives:

```text
parent_genome
parent_state
selected_cells
sanitized Train feedback
remaining resource budget
allowed mutation operators
counter draw
proposal and response-byte limits
```

It never receives task futures, raw forecasts, Dev values, Public identifiers,
evaluator labels, filesystem paths, environment secrets, source paths, live
Store/Kernel/archive handles, callbacks, or clients. Futures are opened only by
the trusted task evaluator. Raw Dev values are allowed only in sealed Kernel
acceptance evidence. Public data is never opened by `numerical-evolve`, and the
completion must report `public_test_accessed: false`.

### LLM configuration boundary

The Python construction seam accepts a Host runtime whose `forecast_store`
implements `forecast(...)`; it may also provide a resource reporter, its bound
identity and resource kinds, and an existing `llm_client`. An explicit
`llm_client` argument takes precedence. The LLM provider sends one strict
JSON-schema request, applies call/token/byte limits, and records a closed
attempt. Unconfigured, unavailable, timed-out, malformed, empty, or over-budget
LLM attempts do not relax validation. If budget remains, `hybrid` continues
with the deterministic provider. Secrets configure the external client and
never belong in config or run artifacts.

The current `python -m evolving_loop.v2 numerical-evolve` CLI intentionally has
no model, API-key, provider-client, or forecast-runtime flags. It is therefore
an operational entrypoint for the shipped deterministic `smoke` profile only.
Supplying `pilot.json` or `formal.json` directly to that CLI fails closed with a
missing Host forecast-runtime error. Those profiles are integration contracts
for a Python Host that calls the Numerical construction seam with the required
runtime (and, optionally, the identified LLM client); Project 2 does not ship a
standalone real-adapter CLI launcher.

## Descriptors, ranking, and scheduling

### History-only cells

One cell is
`(trend, seasonality, intermittency, regime, horizon, method_family)`.
For history values `x`, `s = max(pstdev(x), variance_floor)`:

- trend score is
  `abs(mean(last quarter) - mean(first quarter)) / s`;
- intermittency is the fraction of history values exactly equal to zero;
- regime score is
  `abs(mean(second half) - mean(first half)) / s`;
- horizon score is `forecast_horizon / history_length`; and
- seasonality uses the strongest configured, scale-invariant lag
  autocorrelation with at least two complete repeats.

The versioned descriptor policy turns those scores into closed categorical
bins. Future values, labels, and document roles are not arguments to the
descriptor function. One candidate may occupy multiple cells; each cell entry
aggregates only its committed Train tasks.

### Objectives and constrained NSGA-II

For each task, the scale is the mean absolute future value. `sMAE` is MAE/scale
and `sRMSE` is RMSE/scale; each is capped independently at 5 before task
aggregation. A zero scale yields zero only for zero error, otherwise infinity
and therefore an invalid finite objective. The five minimized objectives, in
contract order, are:

1. mean capped sMAE;
2. mean capped sRMSE;
3. linearly interpolated P95 capped sRMSE;
4. mean raw joint error, `(raw_sMAE + raw_sRMSE) / 2`; and
5. normalized execution cost.

Constraint comparison happens before Pareto dominance: every feasible entry
beats every infeasible entry. Infeasible entries order by violation count,
sorted violation names, then artifact SHA. Feasible entries use ordinary
non-dominated fronts, normalized crowding distance, and final ascending artifact
SHA. No weighted scalar score or descriptor coordinate participates in
survival.

Each cell retains at most the configured capacity, which is constrained to
1–4. Parent categories have master mass 40/30/20/10 for underexplored cells,
rank-zero elites, failure-matched specialists, and non-elite stepping stones.
Empty categories are removed and their integer mass is not sampled; the
remaining weights are normalized by their exact sum. Both category and member
draws use the checkpointed SHA-256 counter stream.

### Hyperband and cache

Resource is the number of committed unique Train task evaluations. Registered
brackets are `explore: 8 -> 32 -> 80`, `confirm: 32 -> 80`, and
`replay: 80`. Three explore Children promote `3 -> 2 -> 1`; other nonfinal
rungs retain `max(1, count // reduction_factor)`. Entity groups remain whole,
and every cumulative task manifest is persisted before dispatch.

A single candidate or complete verified cache coverage prefers replay; at least
40% coverage prefers confirm; otherwise the scheduler uses explore, subject to
remaining search time. Cache identity binds candidate, exact task bytes, split,
metric, descriptor, runtime, protocol, and execution-adapter identities.
Missing, malformed, or mismatched rows are charged misses. Failed work is never
cached as success. Dev20 is not Hyperband resource: it is opened once for the
active Parent and the sealed final Train winner.

## Run layout and inspection

The root is the Project 1 Kernel authority. Important files are:

```text
RUN/
  run_manifest.json
  budget_plan.json
  checkpoint.json                 # mutable, atomic Kernel checkpoint
  accepted_bundle.json            # mutable, atomic active pointer
  promotion_history.jsonl
  acceptance/                     # sealed decisions; only raw Dev location
  archive/objects/                # immutable Kernel objects
  evaluation_complete.json        # sole completion authority
  numerical_qd/
    manifest.json
    checkpoint.json               # mutable, cross-bound runner checkpoint
    objects/<sha256>.json          # typed immutable artifacts
    sources/<sha256>.py            # verified source bytes
    proposals/<sha256>.json        # provider attempt and request binding
    results/<candidate>/<task>.json
    rungs/<sha256>.json
    archive/entries.jsonl
```

Useful read-only inspection commands:

```bash
python -m json.tool "$fixture_dir/run/evaluation_complete.json"
python -m json.tool "$fixture_dir/run/accepted_bundle.json"
python -m json.tool "$fixture_dir/run/numerical_qd/checkpoint.json"
tail -n 5 "$fixture_dir/run/numerical_qd/archive/entries.jsonl"
find "$fixture_dir/run/numerical_qd" -type f | sort
```

`evaluation_complete.json` reports frozen Supply, registry, Bundle, QD,
mutation-policy and prompt identities; occupied cells; accepted/rejected and
provider/LLM counts; complete budget state; Dev access; and
`public_test_accessed`. It does not report Retrieval, Decision, joint, Public,
or real-model scores.

## Resume and failure rules

All material is persisted and reread before publication. The order is source
and proposal, candidate, fixed rung reservation/manifest, task results and
charges, immutable rung, QD entries/snapshot, Train policy/prompt state, then a
cross-bound runner checkpoint. Only `checkpoint.json` and
`accepted_bundle.json` are atomically replaced; all other JSON artifacts are
write-once or append-only.

Resume verifies canonical bytes, SHA paths, the immutable catalog, source and
policy dependencies, rung/task/cache bindings, budget closures, Kernel
checkpoint, active Bundle, counter position, and operator input hashes. It does
not repair authority. A changed input, conflicting retry, symlink, traversal,
missing source, corrupt JSONL prefix, altered cache row, open reservation,
unreferenced completed work, or partial authoritative operation fails closed.

Operational outcomes are:

- malformed/out-of-scope proposal: charge and persist the closed rejection;
- LLM unavailable/invalid: close the attempt and fall back deterministically if
  the proposal budget remains;
- forecast/runtime failure: closed invalid result, never a success cache row;
- no feasible Child: preserve Parent and publish a closed no-improvement step;
- no strict Dev improvement: preserve the exact Parent Bundle;
- budget overrun: charge actual work, close the stage, stop opening new work;
- deadline during a rung: retain accounted completed rows but publish no partial
  rung or QD snapshot; and
- artifact/checkpoint corruption: reject resume without overwriting evidence.

Kernel acceptance requires the complete Train winner and materialized artifact
bindings, exact scope/protocol/runtime/accounting evidence, and Dev improvement:
candidate mean sMAE must be strictly lower than Parent and candidate mean sRMSE
must be no worse. Rejection returns the exact Parent object. Accepted and useful
rejected/nonwinning Children may remain immutable QD stepping stones, but only
the accepted atomic Supply/registry pair becomes the next active Parent.

## Project handoff

Project 3 consumes the exact frozen Numerical Supply/registry pair and active V2
Bundle produced here. It adds cooperative and joint Numerical/Retrieval/Decision
evolution and is the first project that may connect the full production
`evolve` path. It must not reopen Project 2 Train/Dev evidence or infer a new
Numerical registry from a mutable pointer.

Project 4 may add DGM evolution of proposer, scheduler, mutation-operator,
Evolver, or Harness source only under a separate source archive and fixed L0
evaluator. Project 5 owns L1 protocol migration. Public scoring is still
unavailable in Evolution V2; when implemented, it remains a separate, one-shot
frozen-artifact operation and cannot feed the same evolution run.
