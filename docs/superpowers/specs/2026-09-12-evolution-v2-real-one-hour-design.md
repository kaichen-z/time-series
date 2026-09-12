# Evolution V2 Real Bounded Orchestrator Design

**Date:** 2026-09-12

**Status:** Proposed for implementation

## 1. Decision

Add one resumable, Host-owned command that runs the four existing Evolution V2
projects in a fixed causal order under one shared profile deadline:

```text
P2 numerical self-evolution
    -> P3 three-agent cooperative evolution
    -> P4 source-policy evolution
    -> P5 infrastructure-protocol evolution
    -> root validation and completion
```

The command is a thin orchestration and bridge layer. It does not replace the
four existing runners, merge their stores, or create a second evolution kernel.
It pins real inputs, derives bounded child configurations, passes verified frozen
artifacts between projects, owns the aggregate deadline, and publishes one root
result only after validating every sealed child result.

The production command is:

```bash
python -m evolving_loop.v2 real-evolve \
  --manifest configs/evolution_v2/real/real-30m-toto-balanced-v3.json \
  --output-dir runs/evolution_v2/<fresh-run-name>
```

Re-running the same command against the same output directory is a resume. A
changed input manifest, model binding, or stage configuration requires a fresh
output directory.

## 2. Goals

The implementation must provide:

1. Real, authenticated 30-minute and one-hour evolution profiles rather than a
   smoke fixture. The 30-minute profile is the first live run.
2. One aggregate wall-time authority covering P2 through root finalization.
3. Numerical dictionary/supply self-evolution followed by three-agent bundle
   co-evolution, then source-policy and protocol evolution.
4. Exact, content-addressed handoffs; downstream stages consume frozen upstream
   artifacts rather than recomputing equivalent-looking objects.
5. Train/Dev-only search and promotion. Public data remains inaccessible.
6. Closed-boundary resume with conservative accounting after interruption.
7. A concise root result that says what evolved, what was accepted, what stayed
   at seed, how much budget each stage used, and why a run was incomplete.

## 3. Non-goals

This remains a research prototype. The work does not add an OS sandbox,
distributed scheduling, multi-host leases, exhaustive tamper resistance,
long-running CI, automatic deployment, or production observability.

The work also does not:

- change forecast backbones, data loading, metrics, verifiers, artifact
  validators, or the promotion Host;
- let candidate code mutate L0 evaluation rules or Public-test access;
- combine P2, P3, P4, and P5 persistence schemas;
- claim that every stage uses an LLM when its existing proposer is deterministic;
- treat an unchanged seed as a promoted evolved result;
- write into historical run or authority directories.

These fixed Host components are the measuring apparatus. Prompts, candidate
formats, tool dictionaries, agent modules, source policies, protocol proposals,
and handoff representations may evolve only through their existing ownership
and validation boundaries.

## 4. Real Input Authority

Both initial real profiles use the frozen Toto-balanced v3 authority:

- split: `splits/drcik_public_80_20_99_v3.json`;
- task corpus: `external/Dr-CiK/full-download/Dr-CiK_public/tasks/`;
- numerical seed: the sealed champion release from
  `runs/champion_evolution/gpt56sol_high_toto_balanced_v3_20260906_g3_r2/`;
- forecast cache: the Host-owned records under
  `runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906/`;
- retrieval seed: `runs/retrieval_releases/package_nrd_20260903/v000/`;
- source seed: the versioned source files under
  `runs/method_evolution/v001/`;
- local runtime and model caches already configured by the repository.

The manifest records every admitted file by role, repository-relative path, and
SHA-256. Absolute paths, `..`, duplicate roles, non-canonical JSON, missing
records, SHA mismatches, cross-split overlap, and mismatched task/runtime/L0
bindings are rejected before any stage starts.

The v3 split's label-informed selection provenance must remain explicit through
the existing opt-in. It does not permit Public task materialization or Public
metric access during evolution.

The historical Luna-medium package run is read-only baseline evidence. Because
it accepted zero children and ended on its seed bundle, it is neither adopted as
the output of this run nor described as an evolved promotion.

## 5. Model and Runtime Binding

The root manifest binds:

```json
{
  "name": "gpt-5.6-luna",
  "reasoning_effort": "medium"
}
```

The launcher constructs the existing Codex CLI client from that binding and
injects it into the numerical hybrid proposer. Any later stage that already has
a model-client seam receives the same binding. Stages whose checked-in proposer
is deterministic remain deterministic; the result reports this per stage rather
than falsely labelling all proposals as model-generated.

The real path forbids `FakeLLM`, test fixture builders, smoke Host cases, and
imports from `tests/`. A failed or malformed model response may use the existing
deterministic numerical fallback, but the fallback count and reason are recorded
in the stage and root summaries.

Credentials remain process-owned. The manifest, checkpoints, caches, logs, and
completion artifacts must not serialize tokens, credential files, environment
values, or authentication command output. A run may use an already authenticated
Codex CLI session without inspecting or copying its credentials.

## 6. Shared Profile Budget

The root owns one `BudgetPlan` and `BudgetLedger`. Two admitted profiles use the
same 80% search / 20% finalization split:

| Allocation | `real-30m` | `real-1h` |
|---|---:|---:|
| P2 numerical | 840 | 1680 |
| P3 cooperative | 360 | 720 |
| P4 source | 120 | 240 |
| P5 protocol | 120 | 240 |
| Root finalization reserve | 360 | 720 |
| **Total** | **1800** | **3600** |

The first 80% of either profile is the search envelope. The final 20% is reserved
exclusively for readback, link validation, checkpointing, terminal summary
creation, and immutable publication. It cannot be loaned to a new candidate
evaluation.

Unused search time rolls forward only. Before stage `i`, let `elapsed` be the
root ledger's charged wall time, `carry` be unused allocation from earlier
sealed stages, and `later_base` be the sum of later stages' base allocations:

```text
search_remaining = profile_limit - finalization_reserve - elapsed
grant_i = min(base_i + carry, search_remaining - later_base)
```

The grant must be positive. P2 time may flow to P3, then P4, then P5; it never
flows backward and never causes an earlier stage to rerun. Profile name, total
limit, all five allocations, and their exact sum are schema-validated constants;
arbitrary user-supplied schedules are rejected. Each child receives a
canonical derived config whose legal wall-time fields equal its grant. Child
ledgers remain useful for their local resource accounting, but the root ledger
is the aggregate authority.

The parent reserves a stage before launch and closes it only after the child
returns at a sealed boundary. Parent monotonic elapsed time is authoritative. A
child overrun is terminal and no later stage starts.

## 7. Stage Integration

### 7.1 P2 numerical self-evolution

P2 is invoked through the existing Python seam with a real forecast-store Host
adapter and the bound Codex client. It uses a derived `pilot` configuration with
the granted hard limit; the fixed four-hour `formal` semantics remain unchanged.

The numerical search evolves the dictionary/supply using the implemented
quality-diversity and multi-objective machinery. Completion may legitimately
retain the seed if no candidate passes Train/Dev promotion.

A small public persistence API loads and verifies the active frozen numerical
pair from P2's content-addressed catalog. The orchestrator must not inspect
private catalog paths or rebuild the registry.

### 7.2 P2 to P3 handoff

`handoffs/p2_to_p3.json` commits:

- P2 completion and checkpoint SHA-256;
- frozen-pair object SHA-256;
- numerical release and registry SHA-256;
- exact task-manifest SHA-256.

P3 receives the restored `FrozenNumericalArtifactsV2` object through its existing
Host runtime seam. The P3 seed release and the supplied frozen release must be
identical. A semantically similar registry with a different content identity is
rejected.

### 7.3 P3 cooperative evolution

P3 runs the existing three-coordinate cooperative runner with real retrieval and
decision factories, real Train/Dev tasks, the exact P2 frozen numerical output,
and the repository's fixed evaluator. The three evolvable coordinates are the
numerical supply, retrieval module, and decision module; their accepted bundle is
sealed as one cooperative artifact.

The root accepts P3 only after validating its cooperative checkpoint, archived
bundle object, acceptance evidence, and `accepted_bundle.json`. A seed-only P3
completion is valid but is reported explicitly.

### 7.4 P3 to P4 handoff

A Host-owned bridge constructs a source-evolution case from the sealed P3 bundle
and a deterministic, manifest-bound small subset of the real frozen Train/Dev
tasks. The subset size follows the P4 runner's bounded research profile, but its
members come from the real split and task loader—not from a test fixture.

The source seed's protocol and runtime fingerprints must match P3. The bridge
owns the evaluator and immutable runtime bindings; candidate source cannot change
them. `handoffs/p3_to_p4.json` commits the P3 bundle, evidence, checkpoint, task
projection, and resulting case identities.

### 7.5 P4 source-policy evolution

P4 runs the existing DGM-lite/source-policy runner directly with the real Host
case. Its CLI smoke builder is not used. The active source, source archive,
evaluation evidence, evaluator-cache checkpoint, and whether any non-seed source
was promoted are included in its sealed result.

P4's existing resume helper resets its local wall-time window. The root must not
use that behavior to extend the aggregate epoch; interrupted in-flight P4 work is
handled by the conservative root recovery rule in section 9.

### 7.6 P4 to P5 handoff

P4 evolves source policy, not an `EvolutionBundleV2`. Therefore the exact sealed
P3 bundle remains P5's bundle under test. P4 contributes active-source, archive,
and evidence identities to provenance, but cannot rewrite the P3 bundle, L0
commitment, runtime registry, or P5 proposal sequence.

A Host-owned production bridge builds the compatibility corpus, runtime registry,
and typed P5 Host inputs from the frozen P3 bundle and P2/P3/P4 provenance. It
must not call the smoke builder. `handoffs/p4_to_p5.json` commits the complete
provenance closure and P5 case identity.

### 7.7 P5 infrastructure-protocol evolution

P5 runs its generic programmatic runner with a canonical configuration bounded
by the root grant. The existing smoke CLI's exact-profile guard remains intact.
P5 may evolve L1 protocol representations and adapters, but L0 evaluation and the
P3 bundle under test remain frozen.

The root accepts P5 only when compatibility evidence, release, protocol, bundle,
runtime, and L0 identities form a verified closed handoff.

## 8. Root Store and State Machine

The output layout is:

```text
RUN_DIR/
  run_manifest.json
  budget_plan.json
  checkpoint.json
  progress.jsonl
  handoffs/
  p2/
  p3/
  p4/
  p5/
  result_summary.json
  evaluation_complete.json
```

Each child retains its existing isolated store. At the root, immutable files use
canonical bytes and content identities; `checkpoint.json` is the only mutable
state. `progress.jsonl` receives one canonical row for each closed transition.

```text
NEW -> P2_RUNNING -> P2_SEALED -> P3_RUNNING -> P3_SEALED
    -> P4_RUNNING -> P4_SEALED -> P5_RUNNING -> P5_SEALED
    -> FINALIZING -> COMPLETE
                      \-> INCOMPLETE or FAILED
```

The root checkpoint commits the input manifest, root plan, model binding, phase,
stage records, active permit, forward carry, root ledger checkpoint, handoff
identities, and optional completion identity. Each stage record commits its
derived configuration, bounded inputs, child directory role, grant, charged
elapsed time, status, preceding progress identity, and—only after sealing—the
child completion identity.

The root checkpoint is written immediately after reserving a stage and after
every sealed child transition. Semantic artifacts do not contain timestamps or
machine-specific absolute paths.

## 9. Resume and Failure Semantics

Resume is permitted only with the exact root manifest and existing root store.
No child output directory may be adopted without that root authority.

- A fully sealed child is revalidated and may be adopted without reevaluation.
- A root crash after child completion but before root closure may adopt the child
  only after its own complete-resume validation. The interrupted stage is charged
  its full grant, so unused carry is lost in that crash window.
- An unsealed or unverifiable in-flight child is also charged its full grant and
  the root returns `INCOMPLETE`. It is not blindly replayed within the same
  current profile epoch.
- An in-flight P4 authority canary remains non-resumable, matching its existing
  source authority contract.
- Invalid input, changed identity, corrupted sealed artifacts, a terminal child
  integrity error, or budget overrun produces `FAILED`; no later stage starts.
- A documented child pause or exhaustion produces `INCOMPLETE`; sealed earlier
  artifacts and the exact reason remain available.
- Retrying incomplete unsealed work requires a fresh, user-started epoch and a
  new output directory. There are no per-candidate human approval gates.

A completed resume performs read-only linkage validation and returns byte-identical
semantic completion bytes.

## 10. Public Firewall and Promotion

Train tasks drive proposal feedback. Dev tasks drive acceptance and promotion.
Public membership may be checked structurally at admission, but Public task
contents, labels, forecasts, and metrics are never loaded by P2–P5.

Every child completion and the root completion must state
`public_test_accessed=false`. Any contrary or missing evidence blocks promotion
and root completion.

Promotion remains Host-owned. A candidate may change evolvable prompts, formats,
dictionaries, modules, source policies, or L1 protocol data only inside the
appropriate project. It cannot change task loading, forecasting truth, metrics,
verifiers, artifact validation, or promotion decisions.

## 11. Root Result

`result_summary.json` and `evaluation_complete.json` report:

- root status: `complete`, `incomplete`, or `failed`;
- input-manifest, model-binding, root-plan, and completion identities;
- per-stage grant, charged wall time, resource use, completion identity, and
  proposal mechanism (`gpt-5.6-luna`, deterministic, or fallback);
- P2 active supply/registry and whether it differs from seed;
- P3 active bundle coordinates and whether any coordinate differs from seed;
- P4 active source/archive and whether a non-seed source was promoted;
- P5 active protocol release and whether a non-seed protocol was promoted;
- every handoff identity and the final provenance closure;
- explicit Public-access evidence;
- exact exhaustion, rejection, fallback, pause, or failure reasons.

“Complete” means all four stages reached and validated a sealed semantic result;
it does not imply that every stage promoted a new candidate.

## 12. Implementation Surface

Add:

```text
evolving_loop/v2/real/__init__.py
evolving_loop/v2/real/contracts.py
evolving_loop/v2/real/runner.py
evolving_loop/v2/real/bridges.py
configs/evolution_v2/real/real-30m.json
configs/evolution_v2/real/real-1h.json
configs/evolution_v2/real/real-30m-toto-balanced-v3.json
configs/evolution_v2/real/real-1h-toto-balanced-v3.json
tests/test_evolution_v2_real.py
```

Modify only the necessary public seams:

- register `real-evolve` in `evolving_loop/v2/cli.py`;
- expose a verified active-frozen-pair loader from numerical persistence;
- add production P3-bundle case construction under `evolving_loop/v2/source/`;
- add production P3-bundle Host-case construction under
  `evolving_loop/v2/protocol/`;
- allow the programmatic P5 bridge to use its bounded canonical config while
  preserving the existing smoke CLI guard.

Canonical serialization, the core kernel, existing child artifact schemas,
fixed Host evaluation, and Public evaluation remain unchanged.

## 13. Verification

Automated tests use deterministic clocks and injected test adapters; CI never
waits an hour or calls a real model.

Coverage includes:

1. Manifest admission, SHA binding, path safety, split separation, runtime/L0
   matching, and fixed Luna-medium binding.
2. All base grants, forward rollover, final-reserve isolation, insufficient-time
   denial, and overrun behavior.
3. Exact P2 frozen-pair restoration and rejection of identity-changing rebuilds.
4. Real bridge ownership: P4/P5 production code cannot import test fixtures, P5
   retains the exact P3 bundle, and P4 cannot mutate it.
5. All state transitions, handoff commitments, root finalization, Public firewall,
   and byte-identical complete resume.
6. Failure at every stage, closed-child adoption, unknown in-flight charging,
   malformed child state, P4 canary interruption, and finalization write failure.
7. One short deterministic end-to-end CLI smoke through P2→P3→P4→P5.

After automated tests pass, run a fresh authenticated real canary in a new output
directory. The canary uses the real split, task loader, forecast store, model
client, retrieval/decision factories, and P4/P5 bridges, but deliberately tiny
candidate/generation limits. It must prove at least one real model request or
explicitly report the existing fallback, produce all four sealed child results,
show `public_test_accessed=false`, and pass resume validation.

Only after the canary passes should the operator start the approved 1800-second
live run requested for this implementation. The 3600-second profile remains
available for a later run. Starting an epoch is one operator action; evolution
and promotion within that epoch require no human approval.

## 14. Acceptance Criteria

The feature is complete when:

- the canonical `real-30m` manifest starts one fresh 1800-second epoch and the
  canonical `real-1h` manifest can start one fresh 3600-second epoch;
- a single root ledger enforces the four stage grants and protected reserve;
- P3 consumes the exact active frozen pair sealed by P2;
- P4 consumes a real Host case derived from the exact P3 bundle;
- P5 evaluates the exact P3 bundle with P4 provenance through a real Host case;
- no real path imports smoke/test fixture builders;
- all stages and the root prove `public_test_accessed=false`;
- seed-only outcomes, deterministic proposals, and model fallbacks are reported
  honestly;
- closed-boundary interruption and completed resume pass automated tests;
- the real authenticated canary completes and resumes successfully;
- the 30-minute launcher is exercised once with real inputs and the one-hour
  launcher remains runnable from the documented module command without modifying
  code or historical outputs.
