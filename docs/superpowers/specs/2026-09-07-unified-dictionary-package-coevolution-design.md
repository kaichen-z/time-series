# Unified Dictionary and Package Co-evolution Design

**Date:** 2026-09-07

**Status:** Approved by user

**Scope:** Unify Dictionary supply evolution with the existing package-native
Numerical, Retrieval, and Decision coordinate lifecycle without rewriting the
legacy generic evolution engine.

## 1. Objective

The repository currently has two real but separately invoked layers:

1. upstream Numerical supply construction and curation, including Git-backed
   Statistical methods, immutable TSFM runtimes, Combined policies, and
   task-conditioned screening; and
2. package-native `Numerical -> Retrieval -> Decision` coordinate evolution.

The new canonical workflow must bind these layers into one auditable run:

```text
Dictionary Supply Epoch
  -> Numerical -> Retrieval -> Decision
  -> Numerical -> Retrieval -> Decision
  -> immutable final bundle
  -> separate one-shot Public-99 evaluation
```

This is co-evolution because the accepted output of one coordinate becomes the
Parent observed by the next coordinate, and Train-only Retrieval/Decision
evidence may inform the next Numerical proposal. It is not simultaneous
mutation: one accepted transition changes exactly one principal coordinate.

## 2. Existing implementation to preserve

The implementation composes the tested production modules rather than replacing
them:

- `evolving_loop/co_evolution.py` remains the compatibility-level generic
  Parent/Child engine and `HarnessPolicy` contract.
- `evolving_loop/package_coordinate_evolution.py` remains the package state and
  one-coordinate-at-a-time controller.
- `evolving_loop/package_stage_runner.py` remains the registered
  `8 -> 32 -> Train-80 -> Dev-20` phase runner.
- `evolving_loop/package_numerical_evolution.py` remains the Numerical recipe,
  fitting, diagnostics, materialization, and release adapter.
- `evolving_loop/package_retrieval_evolution.py` remains the Retrieval Genome
  evolution adapter.
- `evolving_loop/package_decision_evolution.py` remains the Decision evolution
  adapter.
- `evolving_loop/numerical_two_stage.py` remains the frozen per-task
  Numerical-to-Retrieval-to-Decision inference bridge.
- `evolving_loop/evaluate_frozen_package_bundle.py` remains the only path that
  opens Public-99.

The old generic engine is not converted into a universal four-agent controller.
It has many compatibility callers and does not carry the package registry,
Numerical release, or one-coordinate lineage needed by the current system.

## 3. Chosen architecture

### 3.1 Outer Supply Epoch, inner coordinate loop

Dictionary evolution is an outer supply epoch, not a fourth mutation inserted
between every package coordinate. A Dictionary change may alter executable
source, candidate identity, screening, forecast caches, and every downstream
Numerical package. It therefore must finish and freeze before the package loop
starts.

```text
methods.py + skills.py + policies.py + TSFM manifests
  -> method/Combined evolution
  -> task-conditioned screening evolution
  -> frozen DictionarySupplyRelease
  -> build initial NumericalSupplyRelease and task registry
  -> [N -> R -> D] x 2
  -> sealed UnifiedCoEvolutionBundle
```

Changing the Dictionary after an inner cycle begins requires a new supply epoch
and a new downstream lineage. It cannot silently reuse forecasts, registry
entries, proposals, or acceptance evidence from the preceding epoch.

### 3.2 Why not mutate all four coordinates together

Simultaneously changing Dictionary, Numerical, Retrieval, and Decision would
make improvements impossible to attribute and would allow one broken coordinate
to be hidden by another. It would also make cache identity ambiguous. The chosen
design provides one direct Parent and one owned Child transition at every step.

### 3.3 Why not rewrite `CoEvolutionEngine`

The old engine remains useful to legacy prompt/genome/source experiments and as
a lower-level component of Decision evolution. Replacing it would produce a
large compatibility migration without solving the missing supply binding. A new
top-level orchestrator is smaller and makes the current package controller the
authoritative co-evolution path.

## 4. Dictionary Supply Epoch

### 4.1 Inputs

The epoch receives reviewed, explicit inputs:

- Git-tracked `methods.py` containing the executable Statistical functions;
- reviewed `skills.py` containing history-only analysis primitives;
- `policies.py` containing immutable TSFM bindings and typed Combined policies;
- task-conditioned screening source;
- exact model/runtime/license manifests;
- Train-80 and read-only Dev-20 split identities;
- proposal model and reasoning configuration; and
- Dr-CiK scaled metric policy.

TSFM parameters are not trained or merged. The epoch may change only reviewed
invocation/preprocessing policy and Combined structure. Checkpoint, adapter,
license, and model identities remain host-owned.

### 4.2 Subphases

The supply epoch runs three bounded subphases:

1. **Method repository evolution** proposes `add`, `repair`, `fork`, `merge`, or
   evidence-backed removal/quarantine operations over Python forecasting
   functions. The Host parses and executes the resulting module.
2. **Combined policy evolution** proposes typed Statistical--TSFM and
   TSFM--TSFM structures. Legal families include select, route, horizon route,
   weighted, median, and bounded overlay. The Host fits all thresholds and
   weights; the LLM never emits numerical fit authority.
3. **Screening evolution** assigns status and history-only applicability to
   Statistical, TSFM, and Combined candidates. It uses deterministic task
   morphology and must preserve the task's measured Train oracle while reducing
   crash/invalid exposure.

Each subphase rejects an invalid Child atomically and preserves its exact Parent.
Only the final accepted source and policy set becomes the supply release.

### 4.3 Supply release

Add a canonical `DictionarySupplyRelease` that binds:

- schema version and monotonically increasing supply version;
- direct Parent release fingerprint;
- `methods.py`, `skills.py`, `policies.py`, and screening fingerprints;
- exact Statistical, TSFM, and Combined inventory records;
- method status and applicability fingerprints;
- TSFM runtime and license-manifest fingerprints;
- Train/Dev split, metric-policy, proposer, and evaluator fingerprints;
- accepted stage evidence; and
- a release fingerprint over the complete canonical payload.

The release contains references and canonical records, not arbitrary Python code
inside JSON. Python remains Git-tracked; JSON binds the exact Git/source bytes and
the executable inventory derived from them.

## 5. Numerical coordinate

The Numerical coordinate consumes the frozen `DictionarySupplyRelease`. It does
not mutate method source in the inner loop.

For each task it:

1. builds a deterministic history-only `TaskProfile`;
2. instantiates an Active Dictionary from the frozen screening policy;
3. materializes each eligible Statistical or TSFM leaf once;
4. materializes legal Combined candidates from successful leaves;
5. computes rolling hindcast diagnostics;
6. proposes three structural Champion Children;
7. lets the Host fit thresholds and weights with Train-only cross-fitting;
8. associates every candidate that may reach Decision with one or more typed,
   falsifiable assumptions and failure conditions; and
9. emits a bounded `NumericalSupplyRelease` and per-task
   `NumericalForecastPackage` registry.

Legal proposal structures include:

- `select`;
- `route`;
- `horizon_route`;
- `weighted`;
- `median`; and
- `bounded_overlay`.

The initial Toto release is a safe anchor, not a permanent mandatory winner.
Non-Toto TSFM, Statistical, or Combined Children may replace it when they pass
the same Train/Dev safety gates. Rejection preserves the current anchor and all
accepted alternatives exactly.

Task-local adaptive weighting and morphology-group cross-fit belong inside this
coordinate. They are Host-fitted policies over materialized forecasts, not new
LLM-generated forecasts.

## 6. Retrieval coordinate

Retrieval receives only the sanitized task, frozen Numerical candidates, and
the typed assumptions needed to distinguish them. It never receives future
labels, per-task forecast scores, benchmark role labels, or raw Dev/Public
residuals.

The two-stage protocol remains:

1. Round 1 is assumption-blind and searches the bounded task corpus.
2. A provisional Decision may emit one sanitized evidence gap.
3. Round 2 receives only that gap and relevant assumption identifiers.
4. Host verification checks exact document, quote, target, window, stance, and
   assumption bindings.
5. A malformed or failed Round 2 retains valid Round-1 evidence.

Retrieval self-evolution may change its Genome, prompt, budgets, topology, and
accepted Retrieval Skills within the existing closed contracts. It cannot alter
the Dictionary, Numerical candidates, forecasts, or Decision policy.

## 7. Decision coordinate

Decision receives:

- only already materialized Numerical candidates;
- history-only diagnostics safe for the Decision boundary;
- verified Retrieval evidence cards;
- typed assumptions and failure conditions; and
- the current safe fallback.

Decision self-evolution may change its prompt, Skill set, evidence-use policy,
and aggregation policy. It may select one materialized candidate or a permitted
pre-materialized ensemble. It cannot create a new forecast vector, cite
unverified evidence, or override the safe anchor without exact host-certified
support for the selected candidate.

## 8. Cross-coordinate feedback

Feedback is asymmetric and Train-only:

- Numerical results inform Retrieval through candidate assumptions.
- Verified Retrieval evidence informs Decision.
- Accepted/rejected Retrieval and Decision task traces are projected into a
  sanitized task-evidence ledger for the next Numerical cycle.

The projection may expose closed structural fields such as candidate family,
recipe kind, parent families, assumption condition, evidence stance, and action
outcome. It must not expose task IDs, truth values, raw forecast values, numeric
thresholds/weights, Dev metrics, Public records, or free-form model rationale.

If no accepted non-seed Retrieval trace exists, the next Numerical cycle receives
an explicit empty/unavailable treatment projection rather than fabricated
evidence.

## 9. Data and evaluation protocol

### 9.1 Train-80

All proposal fitting and detailed feedback comes from Train-80.

- `screen8` quickly removes structurally invalid or clearly inferior Children.
- `screen32` expands the evidence while retaining at most the configured
  finalists.
- `train80` evaluates the unchanged finalist using entity-grouped five-fold OOF.
- Every task belongs to one fold, and tasks sharing the protected group/entity
  stay in the same fold.

For each held-out Train fold, parameters are fitted on the other four folds and
evaluated on the held-out fold. This provides all 80 tasks with out-of-fold
evidence without using the current task's label to fit its own route or weight.

### 9.2 Dev-20

Only the fixed Train finalist reaches Dev-20. Dev is read-only and may decide
accept/reject but may not modify a proposal, threshold, weight, Skill, prompt,
or later-cycle feedback payload.

### 9.3 Public-99

Public-99 is absent from evolution configuration, prompts, caches, checkpoints,
selection, and acceptance. A separate evaluator opens it once after the entire
bundle is sealed. Its result is report-only and cannot be used to evolve another
Child in the same experimental lineage.

## 10. Metrics and acceptance

All active selection and evolution gates use the canonical Dr-CiK point-metric
pair:

- sMAE; and
- sRMSE.

A joint score may provide deterministic ordering, but acceptance is pairwise:
neither metric may hide a regression in the other. Train and Dev gates also bind:

- task/fold coverage;
- invalid and crash counts;
- capped and raw P90/P95 tails;
- clipped/catastrophic counts;
- oracle regret;
- baseline/safe-anchor regret; and
- exact Parent/Child win, tie, loss, missing, and unavailable counts.

sMAPE, MAE, MSE, MASE, phase error, bias, and slope error may remain diagnostic
fields. They cannot independently authorize an active Child.

## 11. Unified state and lineage

Add a `UnifiedCoEvolutionBundle` or extend the package bundle schema so the
frozen state binds four independently identifiable coordinates:

1. `dictionary_supply`;
2. `numerical`;
3. `retrieval`; and
4. `decision`.

The bundle also binds runtime identities, split identity, metric policy, stage
schedule, acceptance evidence, and direct Parent fingerprint.

Supply publication initializes a fresh inner package lineage. During the inner
loop, exactly one of Numerical, Retrieval, or Decision may change. A rejected or
invalid transition returns the exact Parent object and bytes. A new Dictionary
release starts a new supply epoch rather than masquerading as an inner Numerical
transition.

## 12. Canonical top-level runner

Add one operator-facing entry point, conceptually:

```bash
python -m evolving_loop.run_unified_coevolution \
  --repo /absolute/path/to/method-repo \
  --split-file /absolute/path/to/split.json \
  --tasks-file /absolute/path/to/tasks \
  --retrieval-seed-release /absolute/path/to/v000 \
  --retrieval-skills /absolute/path/to/skills.jsonl \
  --forecast-store /absolute/path/to/forecast-cache \
  --authority-dir /absolute/path/to/operator-authority \
  --output-dir /absolute/path/to/run \
  --model gpt-5.6-sol \
  --reasoning-effort medium
```

The runner supports explicit bounded modes:

- `--supply-mode existing`: verify and freeze an already accepted Dictionary
  supply, then run the inner package loop;
- `--supply-mode evolve`: run the bounded supply epoch first;
- `--smoke`: deterministic/fake small wiring verification; and
- `--resume`: resume only when every source, split, runtime, schedule, and
  checkpoint identity matches.

Formal mode runs one supply epoch followed by two ordered N-R-D cycles. It does
not automatically start another supply epoch from the same Dev outcome.

## 13. Artifacts

The run directory publishes immutable, attributable artifacts:

```text
run_manifest.json
supply/
  dictionary_supply_release.json
  method_evolution/
  combined_evolution/
  screening_evolution/
package/
  stage_schedule.json
  checkpoint.json
  coordinate_trace.jsonl
  task_feedback/
  acceptance_evidence/
  final_bundle.json
completion.json
```

The final bundle refers to the exact supply release and package releases by
fingerprint. It does not copy mutable source trees into an unverified payload.
The Public evaluator writes to a separate directory.

## 14. Failure behavior

- Malformed LLM output rejects only that Child unless runtime integrity is lost.
- Failed Statistical/TSFM execution becomes a typed unavailable/crash/invalid
  outcome and cannot masquerade as success.
- An invalid Combined parent or non-finite forecast rejects the Combined Child.
- A Dictionary Child that breaks namespace, source, applicability, or identity
  validation leaves the Parent supply unchanged.
- A Numerical, Retrieval, or Decision ownership violation leaves the Parent
  package unchanged.
- Missing or mismatched source/runtime/split/checkpoint fingerprints stop the run
  before another external call.
- Dev rejection preserves the exact pre-Dev Parent.
- Public access during evolution is a hard failure.

## 15. Compatibility and migration

- Keep `evolving_loop/co_evolution.py` API-compatible.
- Keep the existing package co-evolution runner available for runs starting from
  an already frozen supply.
- The new runner initially composes existing CLIs/modules through typed adapters;
  it does not duplicate their evaluation logic.
- Existing Dictionary, Champion, task-local, Retrieval, and Decision artifacts
  remain historical evidence. They are not relabeled as one unified run unless
  their hashes prove a real connected lineage.
- The current HTML flow may document this new contract after the implementation
  exists, but must continue distinguishing current contract from recorded run.

## 16. Implementation slices

The implementation should proceed in bounded slices:

1. canonical `DictionarySupplyRelease` and parser;
2. supply-epoch adapter over existing method, Combined, and screening evolution;
3. unified bundle identity and lineage rules;
4. top-level controller and runner in `existing` supply mode;
5. `evolve` supply mode and checkpoint/resume binding;
6. frozen Public evaluator compatibility;
7. deterministic smoke and focused end-to-end tests;
8. documentation and wrapper updates.

No slice changes TSFM weights, the reviewed analysis-skill implementations, or
Public evaluation policy.

## 17. Verification

Tests must prove:

- the legacy generic engine remains byte/API compatible;
- a frozen existing supply can initialize and complete two N-R-D cycles;
- the evolved supply binds exact Git source and screening/runtime identities;
- a supply change invalidates stale package registries and caches;
- only one inner coordinate changes per accepted package step;
- rejected Children preserve exact Parent bytes;
- Numerical assumptions survive the N-to-R projection and only verified evidence
  reaches Decision;
- Train-only task feedback can reach only the next Numerical cycle;
- Dev cannot change proposals or feedback;
- Public IDs and labels cannot enter evolution artifacts;
- one frozen bundle can be evaluated once on the exact Public-99 registry; and
- resume fails closed on source, split, runtime, schedule, or checkpoint drift.

A real run is reported separately from implementation tests. Deterministic smoke
success does not claim an accepted 80/20 release or a Public-99 improvement.
