# Evolution V2 DGM-Lite Source Research Prototype Design

## Status and research question

Project 4 addendum to `2026-09-10-unified-evolution-v2-design.md`. The user
authorized a fast research prototype, automatic decisions without human
approval, and multi-agent implementation. Planning can overlap Project 3;
implementation starts after its focused deterministic compatibility gate passes.

Can a branching archive of small source policies discover a better proposal
selection strategy through real cooperative pipeline evaluations, validate it
without leaking held-out results, and automatically activate or roll it back?

The implemented slice evolves one pure `choose_arm(request)` function in
`policy.py`. This exercises executable scheduler/proposer-selection source
evolution. General Harness rewrites, arbitrary tool programs, L1 infrastructure,
and migration of the existing cooperative runner are outside this prototype.
An active source policy controls subsequent Project 4 epochs. Existing Project
1–3 commands retain their accepted semantics.

## Scope and alternatives

Use an isolated policy artifact plus a fixed Host bridge to Project 3. It is
small enough to test in roughly two hours and changes actual source behavior.
A repository-wide patch executor would add dependency and recovery work before
the research question can be tested. A table of parameter choices alone would
not exercise executable source evolution. The selected design permits source
expressions and branches, while keeping authority outside the editable root.

Binding prototype constraints:

- This is a research prototype; target implementation plus verification is 120 minutes.
- Editable candidate source consists of exactly `policy.py` exporting `choose_arm(request) -> str`.
- Kernel, evaluator, validation, source archive, publication, rollback, task materialization, and scoring code are never editable candidate files.
- Candidate execution uses a separate Python subprocess with JSON input/output; no candidate import occurs in the Host.
- Isolation is an editable-source and data/API boundary, not an OS security sandbox.
- Train meta-CV uses exactly two folds over four Train tasks; sealed meta-validation uses one disjoint Dev task; Public input is rejected before proposals.
- Source requests and archive sampling receive only Train aggregates and source lineage; no Dev values, labels, task IDs, forecasts, or held-out acceptance credit.
- One epoch proposes at most two source Children; one selected finalist may receive meta-validation and one canary epoch.
- Source safety/integrity and canary failures are terminal; static/unit-passing nonwinning source variants remain Train-ranked stepping stones.
- Promotion and rollback are automatic Host decisions; there is no human approval flag or pause.
- Smoke hard limit is 120 seconds, subprocess timeout is 2 seconds, source limit is 8192 UTF-8 bytes, and request/response limits are 16384 bytes each.
- Persist at closed source-candidate, finalist-validation, and canary boundaries; resume never repeats a completed stage.
- Do not add OS sandboxing, TOCTOU/race defenses, adversarial escape research, per-forecast writes, long formal/80-20 runs, or repeated full regression gates.

## Fixed integration boundary

Reuse Project 1 `canonical_v2_bytes`, `fingerprint_payload`, `require_sha256`,
`BudgetPlan`/`BudgetLedger`/`ResourceUse`, and canonical atomic/write-once store
helpers. Do not change `EvolutionBundleV2` or its mutation ownership.

The Host bridge consumes these fixed Project 3 interfaces:

```python
propose_bundle_candidate(parent, arm, catalog, adapters, feedback, step)
# -> BundleCandidateV2 | None; candidate.to_child(parent) validates scope
CooperativePipelineAdapter.evaluate(bundle, tasks, stage)
# -> PackageEvaluation; evaluates Numerical -> Retrieval -> Decision
sanitize_train_feedback(parent_eval, child_eval, normalized_cost)
# -> SanitizedEvolutionFeedback
```

Use Project 2 frozen Numerical artifacts and Project 3 catalog/adapter objects
unchanged. The source runtime returns only an enabled arm. The Host constructs
the legal Bundle proposal and computes all scores. Source code cannot return a
score, evaluation receipt, active pointer, callback, path, or Bundle identity.
No source API assumes a new Project 3 runner injection point.

## Source contracts and execution

New package: `evolving_loop/v2/source/`, distinct from legacy
`evolving_loop/source_evolution/`. Host-owned files are `contracts.py`,
`runtime.py`, `archive.py`, `meta.py`, `authority.py`, `runner.py`, and `cli.py`.
`__init__.py` exposes the public API; `__main__.py` runs the CLI.

`SourceVariantV2` canonical payload fields are exactly `schema_version=1`,
`parent_source_sha256` (null for seed), `operator`, `files`,
`protocol_fingerprint`, and `runtime_fingerprint`. `files` is exactly
`{"policy.py": <source text>}`. Its fingerprint hashes canonical bytes,
including parent and complete source. Novelty uses a separate source-text SHA
so changing lineage alone cannot inflate novelty. Parent must exist; branches
may descend from any eligible archived source, not only the active source.

`SourceRequestV2` has exactly `schema_version=1`, `enabled_arms`, `step`,
`seed`, and `train_reward_by_arm`. Arms are a nonempty canonical subset of
`numerical,retrieval,decision,joint`; rewards are finite aggregate Train scalars
for precisely those arms. No unrestricted nested feedback object is accepted.

Audit before execution: exact filename, size, syntax, and one undecorated
`choose_arm(request)` function with no defaults or annotations. Use a narrow
AST allowlist covering constants, local assignment, names, indexing,
arithmetic/comparison/boolean expressions, if/conditional expressions, and
return. No imports, attribute access, calls, comprehensions, loops, nested
functions, globals, classes, or dunder names. This deliberately bounded source
grammar is sufficient for seeded selection policies. The worker runs audited
source with empty builtins and only the request argument; the Host rechecks
that the returned JSON string is one of `enabled_arms`.

Start a fresh `python -I` worker with a temporary working directory, minimal
environment, fixed worker program, stdin JSON, captured stdout, and a two-second
timeout. Host filesystem/Kernel/store paths and credentials are never in the
request. The worker program is Host-owned, not supplied by the candidate.
These measures reduce accidental authority coupling; they do not claim hostile
Python containment or replace an OS sandbox. Exact manifest and AST tests are
the required isolation tests, not attacks against the interpreter.

## Branching archive and proposals

`SourceArchiveV2(root)` stores canonical source objects under `objects/<sha>.json`
and append-only lifecycle events under `events.jsonl`. Closed states are
`eligible`, `terminal`, and `validated`; promotion history is Host-private.
The sampler sees only eligibility, source-text novelty, Train gain, task cost,
depth, and child count. It never sees validation status or active/canary wins as
selection rewards. A meta-validation failure retains the fixed source as an
eligible stepping stone; its sealed held-out result is never passed back.

For deterministic minimal sampling, sort eligible entries by
`(-train_gain, task_cost, child_count, depth, source_sha256)` and use
`draw_counter % len(eligible)` to choose. This cycles through nonchampions;
prefer a not-yet-seen source-text digest when gains tie. Persist the draw
counter. The archive is branching but never prunes/deletes historical bytes.
Source integrity/static/unit failure closes a terminal lineage before any
scoring. Canary failure closes it after rollback. Source proposals are drawn
from frozen seed templates in smoke; an optional Python-injected provider may
return a complete `SourceVariantV2`, which passes identical audit/gates.

## Automatic meta-evaluation

Commit four Train task identities, two entity-disjoint folds, one disjoint Dev
task, frozen seed Bundle/catalog, deterministic agents, metric/runtime protocol,
and source evaluator identity before running. Reject cross-split duplicate
tasks/entity groups and all Public membership. Fixture construction supplies
the required entity separation; no 80/20 corpus is required.

For each source and fold, reset to the same seed Bundle. Derive the arm request
from the complementary Train fold only, construct at most one legal Child
through Project 3, then evaluate Parent and Child on the held-out Train fold.
Average the two fold gains. The complementary-fold rewards can be a Host-built
per-arm baseline scan, cached across source candidates by exact commitments.
Use `0.0` where an arm has no proposal; such an arm selected for evaluation is
a closed no-improvement result. All fitness comes from real complete-pipeline
outputs, never source length or synthetic scores.

Host joint error is `(mean_capped_smae + mean_capped_srmse) / 2`; gain is
`(parent_joint - child_joint) / max(abs(parent_joint), 1e-12)`.
Require complete coverage, finite values, no added invalid/catastrophic tasks,
and neither capped mean regressing by more than `1e-12`. Among feasible source
Children choose one with strictly positive mean Train gain above `1e-12`,
also improving on the active source's mean Train gain by more than `1e-12`,
breaking ties by task count then fingerprint. Retain other eligible variants.

Seal the finalist and its proposed Bundle using all four Train tasks before
opening Dev. Compare the active source's and finalist source's resulting
Bundles on the same one-task Dev set. Meta-validation passes on complete,
finite coverage, no extra invalid/catastrophic outcome, both capped means
nonregressing within `1e-12`, and joint improvement greater than `1e-12`.
Dev values live only in `sealed/<evidence-sha>.json`; no proposer or sampler
reads that directory. Only one finalist sees Dev per epoch. Repeat the exact
finalist Train execution once to verify output/Bundle fingerprints; matching
cache evidence may avoid rerunning forecasts. Failure is terminal integrity
failure. SHA-sealed Host publication is used; no cryptographic signer or
external key service is claimed.

## Canary, activation, rollback, and resume

`SourceAuthorityV2` is fixed Host code outside the candidate root. It publishes
only a source with verified sealed evidence. It is an L0-side source authority
adapter, not candidate source and not a replacement Bundle kernel.

The Host stores `active_source.json` and append-only `promotion_history.jsonl`.
Before canary, write a checkpoint containing the exact previous pointer bytes,
candidate/evidence identities, and `phase="canary_pending"`. Canary execution
uses the candidate through a separate `effective_source` field; the stable
active pointer remains the old source until canary succeeds. One canary epoch
is one further frozen proposal/evaluation episode with the candidate and
previous source on the same committed 4/1 fixture and fixed next epoch seed.
Reuse the same finite/coverage/nonregression requirements; strict improvement
is not required again. All Dev canary evidence remains sealed.

Success atomically publishes the candidate pointer and appends activation.
Exception, invalid output, incomplete coverage, or regression restores the
previous pointer bytes exactly, appends rollback even if stable pointer never
changed, and terminalizes the candidate. A normal budget boundary before canary
does not count as failure: save `canary_pending` and resume later with a fresh
bounded invocation. No partial candidate evaluation can activate anything.

The checkpoint binds config/input/runtime/protocol digests, archive event count
and prefix hash, source identities, draw counter, completed stage identifiers,
phase, evidence references, budget state, and previous pointer bytes. Verify
these bindings on resume. Support deliberate interruption only at closed stage
boundaries; interrupted in-flight work is reported incomplete and cannot be
promoted. Do not build generic crash recovery. Completed resume is a read-only
no-op returning the same semantic completion bytes. Wall-clock diagnostics are
not included in deterministic semantic result identities.

## Delivery and quick evidence

Five tasks cover contracts/runtime, archive/proposals, meta-evaluation,
authority/resume, and CLI/integration. The blocking gate consists of focused
Project 4 tests, the specific Project 3 proposal/pipeline compatibility tests,
and one offline CLI smoke. Unit fixtures force validation rejection and canary
failure separately; smoke exercises actual source behavior and real pipeline
scoring. Evidence must include one retained nonactive branch, one validation
decision, one automatic activation, exact rollback, and byte-identical closed
boundary resume. If the real pipeline fixture cannot produce an improvement,
fix the deterministic fixture; do not substitute fake fitness for smoke.

The command is initially `python -m evolving_loop.v2.source evolve --config
configs/evolution_v2/source/smoke.json --input-manifest INPUT --output-dir RUN`.
The input manifest binds the frozen Project 3 seed files and fold memberships.
Repeating the command verifies/resumes RUN. The same module supports
`make-smoke-inputs --output-dir INPUTS`; the builder reuses Project 3 fixture
construction. A 120-second smoke ceiling is a limit, not a sleep target.

Record the implemented boundary in `docs/evolution-v2-dgm-lite.md`. Longer
pilot/formal comparisons, production hostile-code isolation, broader editable
hooks, and L1 migration remain future projects rather than completion gates.
