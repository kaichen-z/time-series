# Task 6 report — bounded real CLI and production stage ports

## Outcome

Implemented the `python -m evolving_loop.v2 real-evolve` entry point, canonical
Toto-balanced-v3 manifests for both approved profiles, and the root-owned
production P2→P5 ports.  The command verifies every declared input and runtime
identity before constructing the output, keeps data inputs confined to the
shared authority checkout, permits only resolved executable/source runtimes to
cross the lexical checkout boundary, closes the shared Host on success and
failure, and emits the canonical root result.

The production ports now:

- derive a schema-2 `NumericalSupplyRelease` from the admitted historical
  ChampionRelease;
- build the full current-epoch Dictionary, Train80 fold-complement priors, and
  Train80/Dev20 local-hindcast evidence using only the shared cache-only Host;
- persist and authenticate the prepared P2 inputs separately from the pristine
  Numerical QD store, then load the active frozen pair at the P2 seal;
- run P3 through `run_real_cooperative` and authenticate its sealed Bundle
  closure;
- reconstruct the checked Source seed against P3's protocol and the Host
  runtime, run Source evolution, and bind the active Source and archive;
- run generic Protocol evolution when P3 supplies a real second Bundle, or
  produce a stable root-level incomplete P5 seal when it does not;
- use authenticated root stage wrappers so incomplete native stages need not
  fabricate `evaluation_complete.json`.

The Host exposes an explicit SHA-addressed method-source map, authenticates the
reviewed method/policy/skill/dictionary sources, and reproduces the historical
ForecastStore identity
`90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2`.
The authenticated `outputs/model-cache/hub` tree is used to recover the
historical `outputs/model-cache` HF_HOME, and the historical Toto worker
descriptor is bound at `tmp/toto2_worker_smoke.json`.

## TDD evidence

The initial production-port test failed with the expected unavailable-stub
error before the P2 adapter was implemented.  The root production-port E2E test
then exercised real port assembly, stage wrappers, root sealing, and
byte-identical completed resume while replacing only the expensive child
runners.

Additional red/green checks caught and fixed:

- manifest runtime drift that produced ForecastStore identity `9be8dc…`
  instead of the admitted `90a281…`;
- corrupt cache rows being downgraded to ordinary prior failures rather than
  aborting closed;
- the Source seed carrying the Numerical protocol fingerprint instead of P3's
  deterministic cooperative protocol fingerprint;
- historical Champion JSON being raw-SHA admitted but noncanonical in byte
  formatting;
- the P2 prepared seal using the release-domain fingerprint where the persisted
  input-byte digest was required.

## Verification

Fresh focused real gate after the final fixes:

```text
python -m pytest -q \
  tests/test_evolution_v2_real_contracts.py \
  tests/test_evolution_v2_real_runner.py \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cooperative.py \
  tests/test_evolution_v2_real_source_bridge.py \
  tests/test_evolution_v2_real_protocol_bridge.py \
  tests/test_evolution_v2_real_cli.py
61 passed in 29.35s
```

The broad six-project gate reached `1807 passed` before one legacy
budget-sensitive Numerical runner assertion completed only 877 of its expected
1200 dispatches under sustained suite load.  The command was interrupted at
the requested five-minute silence cap after 1979.35 seconds.  The exact failed
case immediately passed in isolation:

```text
python -m pytest -q \
  tests/test_evolution_v2_numerical_runner.py::test_interrupted_bootstrap_resumes_closed_forecast_cache_without_recharging
1 passed in 40.10s
```

The bounded remainder (Numerical safety, all Cooperative, Source, Protocol,
Kernel, and V2 CLI files) completed independently:

```text
357 passed in 157.43s
```

A non-billable production preparation check against the real cache-only Host
completed and resumed with identical identities: schema 2, Train80/Dev20, 100
task evidence bundles, and exactly 8 candidates in every shortlist.  It
observed 9167 cache hits and 889 cache misses; misses were recorded as closed
failure evidence, no forecast was synthesized, and no LLM was called.

Fresh `py_compile` for every modified Python file and `git diff --check` both
exited zero.  Both checked manifests were also accepted by a real Host build,
which loaded exactly 100 tasks and 93 SHA-addressed method sources.

## Boundary notes

- No billable canary or live evolution run was launched.
- The pre-existing legacy fake CLI dependency remains only for the existing
  fake/smoke command; the new real runner and bridges do not import tests,
  smoke helpers, or fake LLMs.
- The incomplete P5 result is intentional when the authenticated P3 closure
  has no distinct second Bundle.  No Bundle is fabricated to force completion.

Commit message: `feat(real): expose bounded evolution CLI`.

## Independent-review follow-up

Three bounded-production findings were addressed with focused red/green tests:

- each fresh root stage now receives an absolute monotonic deadline; P2
  preparation is included in that interval, derives the Numerical child config
  from only the remaining seconds, and seals an honest incomplete result if
  preparation consumes its grant;
- the shared Codex client caps every subprocess attempt at the remaining stage
  time and exposes cumulative call, UTF-8 input/output byte-budget, and
  subprocess counters;
- P3 now derives from the independent canonical
  `cooperative/real-luna-medium.json` config, whose LLM, input, output, and
  subprocess ceilings are all nonzero and bounded, and the Cooperative runner
  charges the Host counter delta for every evaluated candidate;
- completed and sealed-prefix resume re-invokes each stage's read-only native
  seal and compares the reconstructed closure exactly with the root handoff,
  so tampering with a native completion, frozen pair, Bundle closure, Source
  authority/archive, or Protocol handoff is rejected without rerunning a
  child.

The Host reporter identity changed to bind these accounting semantics, and the
checked Source seed and both real manifests were updated transitively.  Their
canonical bytes and SHA bindings were reverified.

Focused follow-up gate:

```text
python -m pytest -q \
  tests/test_evolution_v2_real_contracts.py \
  tests/test_evolution_v2_real_runner.py \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cooperative.py \
  tests/test_evolution_v2_real_source_bridge.py \
  tests/test_evolution_v2_real_protocol_bridge.py \
  tests/test_evolution_v2_real_cli.py \
  tests/test_codex_cli_client.py \
  tests/test_evolution_v2_cooperative_runner.py
87 passed in 32.88s
```

No billable live run was launched during this follow-up.

### Read-only/deadline re-review

The second review round added deadline checks before every initial-prior cache
read, every Train/Dev shortlist task and candidate, every OOF fold, and every
hindcast/long-horizon cache call.  Expiration propagates as the existing
bounded P2 incomplete result instead of being downgraded to diagnostic failure.

Production resume contexts are now explicitly read-only.  Before any native
seal constructor runs, validation requires each root wrapper and native
completion plus P4's active Source/archive directories or P5's frozen handoff.
The wrapper verifier only rereads exact bytes in read-only mode, and a missing
stage directory is never created.  Focused tests deleted the P3 wrapper, P4
active Source, and P5 native completion and verified each stayed absent after
the rejected resume; P5 content tampering remained rejected.

```text
python -m pytest -q \
  tests/test_evolution_v2_real_runner.py \
  tests/test_evolution_v2_real_cli.py \
  tests/test_task_local_ensemble_cli.py \
  tests/test_evolution_numerical_selector.py \
  tests/test_numerical_selector_script.py
175 passed in 2.36s
```
