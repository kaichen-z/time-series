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

### Live P2 parser-seam fix

The first authorized live attempt exposed a pre-LLM container mismatch: the
authenticated prepared loader returned read-only mapping proxies while the
Numerical payload parsers intentionally require exact dictionaries.  The
loader now returns its already detached canonical JSON dictionaries for the
config, seed Supply, and task manifest while retaining a read-only identity
map.  A focused test loads a fully sealed prepared value and passes all three
payloads through the real bridge's strict parsing seam.

```text
python -m pytest -q \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cli.py \
  tests/test_evolution_v2_real_runner.py
40 passed in 14.12s
```

No run artifacts were modified and no live or billable call was launched by
this fix.

### Live P2 operator-identity seam fix

The next pre-LLM live failure showed that P2 forwarded all six prepared
provenance identities to `LegacyNumericalAdapter`, whose operator contract
intentionally accepts exactly config, seed Supply, and task manifest.  The
production P2 port now projects precisely those three keys at the Numerical
bridge while the prepared closure retains Champion, Dictionary, and local
evidence identities for `seal_p2` provenance.

The focused regression crosses the real `numerical_evolve_payload` adapter
constructor rather than mocking its schema check.

```text
python -m pytest -q \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cli.py \
  tests/test_evolution_v2_real_runner.py
41 passed in 14.12s
```

### Live P2 executable-source seam fix

The subsequent pre-LLM live failure showed that `host.sources` incorrectly
treated all 93 historical Dictionary methods as candidate mutation code.  The
strict Numerical adapter correctly rejected every historical source; those
methods are instead the authenticated cache-only ForecastStore catalog used by
the schema-2 Supply and per-task evidence.

The real manifests now declare one versioned, validator-safe `naive_last`
mutation seed.  It preserves the historical function's valid-domain forecast
formula while replacing only the dynamic exception formatting that violates
the closed language.  Host admission validates the source, requires it to
define only `naive_last`, and binds both the new source SHA
`668bb3f09880adeffcc342b18039ebf53310ed4963924edd6af7bc17887127cc`
and historical origin-method SHA
`6852de9f1108173780aaedabfa96faa171c0e1814df2f15a976f32a2dd130129`.
Those identities are committed by both manifest L0 fingerprints, the derived
schema-2 Supply, and the Host reporter.  The reporter identity is now
`36fdec41981b25e148d6bad23cbf2cef4926f9a49d98035830d24a32f55c9bcc`;
the P4 Source seed and both canonical manifest hashes were updated
transitively.

Because the real Dictionary classifies `naive_last` as a generic fallback,
schema-2 bootstrap now excludes exactly the single protected Anchor sealed by
Task-4 evidence (`toto_2_0`) rather than excluding every generic fallback.
Legacy/no-evidence bootstrap retains its original all-fallback exclusion.  The
strict validator was not weakened, and the complete Dictionary/Supply and
100-task local-evidence paths remain cache-backed and untruncated.

Focused verification:

```text
python -m pytest -q \
  tests/test_evolution_v2_real_contracts.py \
  tests/test_evolution_v2_real_numerical.py \
  tests/test_evolution_v2_real_cli.py \
  tests/test_evolution_v2_numerical_runner.py::test_schema_two_host_run_requires_evidence_or_explicit_bootstrap \
  tests/test_evolution_v2_numerical_runner.py::test_schema_two_bootstrap_excludes_only_the_sealed_protected_anchor
40 passed in 14.74s
```

Canonical-manifest/source provenance verification, focused `compileall`, and
`git diff --check` also exited zero.  No live run or billable LLM call was
launched, and the existing live run artifacts were left untouched.

### Live P2 task-local fallback eligibility fix

The next live pre-LLM failure exposed divergent producer/consumer eligibility
semantics.  `build_task_candidate_shortlist` admits candidates from
`materialize_active_dictionary`, including reviewed fallbacks used to restore
minimum/family coverage, while `materialize_local_package` rechecked raw
applicability and rejected those same sealed candidates.

Task-local materialization now reconstructs the exact active Dictionary for
the task profile and validates shortlist membership against that result.  It
retains the explicit `keep`/`specialized` status check and all existing catalog
and protected-Anchor checks.  A focused regression verifies both sides: a
specialized candidate admitted only as a reviewed fallback materializes, and a
truly `quarantine` candidate remains rejected.

```text
python -m pytest -q \
  tests/test_evolution_v2_shortlist_runtime.py \
  tests/test_evolution_v2_numerical_adapters.py
545 passed in 209.50s
```

No live run or billable call was launched, and live artifacts were not
modified.
