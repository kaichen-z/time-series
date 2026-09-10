# Evolution V2: Project 1 kernel

Evolution V2 is a parallel, experimental control plane under `evolving_loop.v2`.
Project 1 implements strict artifacts, Bundle ownership, persistence, budget
accounting, Host acceptance/promotion/rollback, and deterministic fake execution.
It does not run production forecasting or Public scoring. Existing commands and
frozen legacy artifacts retain their original behavior and formats.

The [master design](superpowers/specs/2026-09-10-unified-evolution-v2-design.md)
describes the five-project destination. The
[Project 1 plan](superpowers/plans/2026-09-10-evolution-v2-kernel-contracts.md)
describes this delivery; this guide records the implemented boundary.

## Authority and implemented scope

| Layer | Project 1 behavior | Later work |
|---|---|---|
| L0: immutable kernel authority | Strict schemas and canonical SHA-256 identities; protocol/runtime commitments; scope and stage checks; Train-only feedback/archive schemas; Host-issued evaluations; acceptance sealing; atomic activation and rollback; resumable resource accounting | Real task materialization, split/data firewall adapters, primary scoring, runtime enforcement, and a separate-process read-only source sandbox |
| L1: infrastructure | Loader, metric, firewall, sandbox, and promotion policy identities are committed; module proposals cannot replace them | Evaluated infrastructure and protocol-migration epochs |
| L2: modules and Bundles | Complete frozen Bundle schema; Numerical, Retrieval, Decision, and joint scope validation; fake Numerical acceptance followed by fake Retrieval rejection | Production Numerical Supply and complete three-Agent evaluation/evolution |
| L3: evolvers | Detached primitive proposer request; no live kernel/store/Host capability in that request; source/path/callback fields rejected by the Bundle parser | DGM source variants, branching source search, automated meta-validation, and source canaries |

This is a data/API authority boundary. The fake proposer is in-process; Python
privacy and primitive requests are not hostile-source OS isolation. The separate
process with a read-only kernel root is Project 4 work, explicitly skipped in the
Project 1 safety suite. Project 1 does not execute untrusted source artifacts.
The kernel accepts trusted Host evaluation aggregates; it does not independently
compute forecasting scores or install the implementations named by commitments.

The schema-2 `EvolutionBundleV2` binds generation, Parent, Numerical release and
registry, Retrieval release, Decision policy, harness policy, archive snapshot,
scheduler state, protocol, runtimes, and acceptance evidence. Numerical mutation
replaces both release and registry digests atomically; they are distinct artifact
identities. A single-coordinate Child changes exactly its declared principal
scope, and a joint Child changes at least two. A proposer cannot change harness,
archive, scheduler, protocol/runtime, lineage, or evidence authority fields.

## Strict identities

`canonical_v2_bytes` emits sorted, compact UTF-8 JSON with one trailing newline.
SHA-256 hashes these exact bytes. Authoritative readers reject duplicate keys,
unsupported schema fields and types, noncanonical bytes, and invalid digests as
applicable to each contract. Numeric values must be finite: NaN and either
Infinity are rejected, including nested values. Legacy Infinity-sentinel
serialization is deliberately not used or changed. Paths, callbacks, clients,
file handles, secrets, and timestamps do not belong in identity payloads.

The shipped config component digests represent **configuration intent**, not
installed executable adapters. Their reproducible preimage is:

```python
canonical_v2_bytes({
    "binding_kind": "config_intent",
    "component": component,
    "configuration": base,
})
```

Here `base` is the complete config without `kernel_protocol` and
`runtime_fingerprints`; `component` is each key in those two mappings. Hash this
preimage with SHA-256. Config scheduler, capacity, and Hyperband settings do not
make those algorithms executable in Project 1.

## Run and resume

Run from the repository root with its Python environment and dependencies:

```bash
v2_scratch=$(mktemp -d)
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/smoke.json \
  --output-dir "$v2_scratch/smoke"

# The same command verifies/resumes the same run; there is no --resume flag.
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/smoke.json \
  --output-dir "$v2_scratch/smoke"
```

Expected status is `deterministic_fake_complete`, with one accepted Numerical
Child, one rejected Retrieval Child, and `public_test_accessed: false`. Fake
objectives are fixed synthetic values, not measured forecasting quality. No
dataset, LLM, or forecast backbone is loaded. The rejected Child returns the
exact Parent object/canonical bytes, leaving the accepted Numerical Bundle active.

New runs require a nonexistent or empty output directory. Existing nonempty
directories enter strict resume validation; a legacy directory is not adopted.
The same config and seed must match the recorded commitments. A completed-run
resume revalidates referenced artifacts and is a no-op: no duplicate objects,
index lines, evaluations, or changed file timestamps.

The fake runner can resume a clean stage boundary (its Python API also exposes
`stop_after_stage`). It cross-checks both checkpoints, config, seed, budget,
archive, proposals, evaluations, evidence, progress, active Bundle, and completion.
It rejects partial stages, open kernel reservations, pending publication, terminal
recovery, unknown files, missing bytes, and inconsistent identities. Only an
unreferenced hidden atomic-write `.tmp` remnant may be ignored. Do not edit or
repair authoritative JSON by hand.

Both `pilot.json` (7,200 seconds, intended real 8/2) and `formal.json` (14,400
seconds, intended real 80/20) use `runner: production`. This command deliberately
returns exit code 2 before creating its output:

```bash
python -m evolving_loop.v2 evolve \
  --config configs/evolution_v2/formal.json \
  --output-dir "$v2_scratch/formal"
# Evolution V2: production evolution requires Project 2+ adapters
```

## Persisted run layout

```text
smoke/
  run_manifest.json                 protocol/runtime, seed, budget-plan bindings
  budget_plan.json                  immutable ceilings and time reserve
  fake_run.json                     fake runner config/seed commitments
  checkpoint.json                   exact runner-stage checkpoint
  kernel/checkpoint.json            kernel authority and publication checkpoint
  progress.jsonl                    Train-only closed evaluation summaries
  promotion_history.jsonl           append-only activation/rollback authority
  archive/index.jsonl               canonical records with record SHA-256
  archive/objects/<sha256>.json      immutable Bundle objects
  candidates/<sha256>/proposal.json  provisional candidate proposal
  evaluations/<sha256>/train.json   sanitized Train evaluation
  evaluations/<sha256>/closed.json  schema-2 closed record with Dev digest only
  evaluations/<sha256>/budget_closure.json
  acceptance/<sha256>.json          sealed Host decision and Dev comparison
  accepted_bundle.json              atomic active Bundle payload
  canary/                          reserved store directory; fake run leaves empty
  evaluation_complete.json          final fake result and resource use
```

A completed smoke has two candidate directories and two closed evaluations;
each evaluation also has Train and budget-closure files (six evaluation JSON
files total). It has four archive objects/records: seed, provisional Numerical
Child, sealed accepted Numerical Child, and provisional rejected Retrieval Child.
There are two acceptance artifacts, two progress lines, and two promotion lines
(seed activation and accepted Child activation). Rejection does not publish.

Objects, manifests, plans, proposals, evaluations, evidence, and completion use
write-once canonical bytes: identical retries are allowed, different bytes fail.
Checkpoints and the active Bundle use atomic replacement; indexes/history append
with fsync durability. Before replacing its checkpoint, a live kernel rereads the
canonical body and requires its recomputed digest to match both the stored claim
and its last written digest. Corruption fails without overwriting those bytes.
Archive loading verifies index/record hashes, object bytes,
protocol/runtime bindings, and parent lineage. The sealed Bundle's archive
snapshot precedes its own seal record, avoiding a circular hash. The current
archive records leave `accepted_release_sha256s` empty; sealed-release authority
is instead recorded in promotion history and acceptance evidence.

## Promotion, rollback, and recovery

Only the Host may issue usable evaluation permits, bind closed evaluation work,
seal acceptance, and publish. It charges actual work even when transition
validation fails. Passing closed status, Dev comparison, and budget closure are
required for acceptance. Raw Dev values are persisted only in `acceptance/`
decision evidence, for both accepted and rejected decisions. The schema-2
`closed.json` records evaluation identities, terminal status, Train evaluation
reference, resource use, and `dev_comparison_sha256`. Its canonical digest is
the `evaluation_sha256` in both evidence and the immutable budget closure.
Evidence loading verifies the closed-record digest and its Dev digest before
reconstructing the evaluator-only aggregate in memory. Proposer requests,
archive objectives, progress, and budget records contain no raw Dev values.

An early transition failure still closes and charges issuer-owned work. If no
decision evidence is written, no raw Dev is persisted; the closed record retains
only its digest. Earlier experimental schema-1 closed files containing raw Dev
are not accepted as schema-2 records or silently rewritten. Preserve such runs
for audit and use a fresh run with this version.

Evidence binds the active Parent, provisional Child, target, protocol, runtimes,
Train evaluation, budget decision, and Dev comparison. The sealed Child contains
the evidence SHA; promotion history binds that sealed Child back to evidence.
Promotion needs no human-approval flag. Durable, reread pending-publication
intent precedes history/pointer writes; the kernel can reconcile an interrupted
publication on restart. A changed committed pointer is corruption, not a pending
publication to repair.

`kernel.promotion_host.rollback(reason="canary_failed")` restores a known prior
active Bundle and appends rollback history. `integrity_failed` and `safety_failed`
are also supported closed reasons. These are Host APIs, not CLI commands or an
implemented source-canary evaluator. With a fully valid archive, rollback can
continue and survives restart; repeating the same rollback is idempotent.

If the current artifact is corrupt, integrity/safety rollback can verify the
entire canonical index/record chain and only the destination's object ancestry.
It never skips corruption in that destination lineage or index. Such recovery
restores the verified Parent and seals `recovered_requires_new_epoch`; further
mutation and ordinary mutable resume in that run are forbidden. Preserve the
run for audit and start a new epoch from the verified Parent through a future
appropriate runner. The fake CLI does not offer a recovery/new-epoch importer.

## Resource accounting

Every `ResourceUse` includes `wall_seconds`, `task_executions`, `llm_calls`,
`input_tokens`, `output_tokens`, `gpu_seconds`, `subprocesses`, and `artifact_bytes`.
The budget plan contains all ceilings, `hard_limit_seconds`,
`finalization_reserve_fraction`, and computed `search_deadline_seconds`. Formal
values are exactly 14,400 seconds, 0.2, and 11,520 seconds, leaving 2,880 seconds
for finalization. A stage may end exactly at the search deadline but cannot open
at or after it; estimates must fit time and every remaining resource ceiling.

Reservations count toward pending use. Closing charges finite nonnegative actual
use and releases the reservation; an overrun is charged in full, closes with
`budget_overrun`, and exhausts subsequent search. Finalization permanently closes
search. Ledger resume preserves charged use, prior elapsed time, stage and
reservation identities, exhaustion, and finalization; a new monotonic origin
does not reset prior use. Closure verification requires elapsed time to be
nondecreasing from `budget_before` to `budget_after`. The checkpoint's saved
elapsed time must cover every verified closure's `budget_after`, even if its
checksum is recomputed. Downtime between processes is excluded; elapsed time
after resume is added to the saved total. These are accounting gates, not an
OS watchdog.

The fake runner declares 1.0 synthetic wall second and 20 task executions per
stage: total 2.0 seconds and 40 executions, with all other resource fields zero.
Those declared zeros are not measurements of filesystem bytes or real runtime
consumption. Smoke's 600-second plan reserves 120 seconds (search deadline 480),
with tighter fake ceilings of 2.0 seconds and 40 task executions.

## Public validation and fingerprint inspection

```bash
python -m evolving_loop.v2 public-evaluate \
  --bundle "$v2_scratch/smoke/accepted_bundle.json" \
  --output-dir "$v2_scratch/public-validation"

python -m json.tool "$v2_scratch/smoke/checkpoint.json"
python -m json.tool "$v2_scratch/smoke/run_manifest.json"
python -m json.tool "$v2_scratch/smoke/accepted_bundle.json"
shasum -a 256 "$v2_scratch/smoke/accepted_bundle.json"
```

The last SHA equals `active_bundle_sha256` in both checkpoints,
`final_bundle_sha256` in smoke completion, and the corresponding archive object
filename. `fake_run.json` records the canonical config SHA; `run_manifest.json`
records the seed and budget-plan SHA plus the full protocol commitment. JSON
pretty-printing is for inspection only: do not replace canonical source files
with formatted output.

Public requires the source run's sealed `accepted_bundle.json`, verifies its
canonical bytes against its archived object and manifest protocol/runtime
bindings, and checks the acceptance reference format. It does **not** load or
replay the acceptance evidence, fully audit source-run history, load Public-99,
or compute a score. Status is `validated_only_no_public_evaluator`, with
`public_test_accessed: false`; this is not certification of Public performance.

Public output must be new/empty and cannot equal, contain, or be contained by
the source run, including resolved symlink and filesystem-identity aliases.
It writes exactly `run_manifest.json`, `objects/<bundle-sha256>.json`, and
`evaluation_complete.json`. It creates no archive, checkpoint, budget, candidate,
policy, or learning state, and changes no source-run bytes or modification times.

## Project 2+ handoff

The following remain unimplemented by Project 1:

- Project 2: production Numerical Dictionary/Supply evolution; history morphology
  descriptors and bins; MAP-Elites cells and parent sampling; typed mutation and
  crossover grammar; constrained NSGA-II; real hindcasting and frozen executable
  Supply export; Hyperband brackets, pruning, and cache/replay integration.
- Project 3: cooperative and joint three-Agent search with real frozen Supply;
  collaborator replay and counterfactual rewards; cost-aware discounted UCB and
  Thompson scheduling; production resumable pilot/formal epochs and real resource
  planning. Scope contracts alone do not implement these searches.
- Project 4: DGM source mutation, branching source archive and stepping stones,
  editable-source audits, separate-process read-only OS sandbox, source
  meta-validation, replay, and one-epoch source-canary scoring/activation.
- Project 5: L1 loader/backbone/verifier/diagnostic/schema evolution, compatibility
  corpus evaluation, protocol migration, and deterministic artifact migration.
- Integration: legacy frozen-artifact import adapters, real task/label/scoring
  adapters, frozen Public-99 scoring, and real pilot/formal performance evidence.

Project 2 may start only after fresh Project 1 contract, hostile authority,
archive/lineage, budget/resume, deterministic transition, promotion/rollback,
Public non-interference, and named legacy regression gates pass, with a reviewed
diff that preserves legacy production modules and frozen artifacts. Passing
Project 1 does not satisfy the master design's full-system acceptance criteria.
