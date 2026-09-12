# Evolution V2 Infrastructure Protocol Research Prototype Design

## Status and objective

Project 5 addendum to `2026-09-10-unified-evolution-v2-design.md`, following
Project 4 `2026-09-12-evolution-v2-dgm-lite-design.md`. The user authorized a
small research prototype, automatic decisions, and multi-agent acceleration.
Planning may overlap Projects 3/4; execution follows Project 4's focused gate.

Research question: can five kinds of versioned L1 infrastructure proposals be
executed, compared against the old protocol on committed data, automatically
accepted or rejected, and resumed without changing the immutable L0 contract?

Choose a data-only protocol manifest backed by a fixed Host implementation
registry. It makes actual loader/backbone/verifier/diagnostic/migration behavior
replaceable in a small offline experiment. Arbitrary infrastructure source
execution adds a sandbox project; a manifest-only validator never exercises
the proposed runtime. Neither is needed for this prototype.

## Global constraints

- Implementation plus focused verification targets 120 minutes across five tasks.
- Candidate kinds are exactly `backbone`, `loader`, `verifier_strategy`, `diagnostic_metric`, and `schema_migration`.
- Candidates contain typed/versioned data only; Host code resolves allowlisted implementation IDs and candidates cannot supply imports, source, callbacks, paths, or L0 policy overrides.
- `KernelProtocolCommitment`, `EvolutionBundleV2`, canonical task materialization, primary scoring, split commitments, label firewall, promotion rules, and existing mutation ownership remain unchanged.
- The committed compatibility corpus is four Train tasks and one disjoint Dev task, plus two frozen archive Bundles; all Public membership is rejected before runtime or proposals.
- The proposal sequence is committed before evaluation; proposal inputs contain no Dev values, future labels, task IDs, or Public results.
- Acceptance requires compatibility and finite complete-pipeline Train/Dev nonregression within `1e-12`; strict improvement is not required for infrastructure-only changes.
- Persist at closed proposal boundaries; complete resume is read-only and returns byte-identical semantic completion bytes.
- Smoke is offline, deterministic, and limited to 120 seconds; no new dependencies or model downloads.
- Exclude real large-model runs, 80/20 or four-hour runs, OS sandboxing, extreme migration matrices, per-forecast persistence, and whole-repository regression gates.

## Versioned contracts and authority

New package: `evolving_loop/v2/protocol/`. `contracts.py` defines frozen,
strict canonical dataclasses with `from_payload`, `to_payload`,
`canonical_bytes`, and `fingerprint`. Reject unknown keys, bool-as-int,
nonfinite numbers, unsupported versions, and malformed hashes.

`ProtocolComponentV2` has `kind`, `implementation_id`, `implementation_version`
(positive integer), and `artifact_schema_version` (positive integer).
`InfrastructureProtocolV2` has exactly `schema_version=1`, `protocol_version`,
`parent_protocol_sha256` (null for seed), `l0_commitment_sha256`, and `components`.
Components contain exactly one typed entry per kind in the order listed above.
The seed version is 1; an accepted child increments its parent's version by 1.
Component versions describe the selected adapter; they are not the protocol
version. Identical version numbers on different branches are disambiguated by
content SHA. Runtime identity commits registry implementation versions and
Host code digests in the input manifest.

`ProtocolProposalV2` has exactly `schema_version=1`, `parent_protocol_sha256`,
`kind`, `replacement` (one component of matching kind), and
`compatibility_corpus_sha256`. `to_child(parent)` changes exactly that component,
requires a changed component identity, keeps L0 identity, and sets parent/version.
Five named proposal subclasses are unnecessary; the closed discriminated kind
and typed replacement form the union. `from_payload` enforces both tags agree.

L1 protocol SHA is separate from `EvolutionBundleV2.protocol_fingerprint`, which
already identifies the immutable `KernelProtocolCommitment`. Do not overwrite
that Bundle field or broaden L2 mutation ownership to permit protocol edits.
`ProtocolReleaseV2` binds `schema_version=1`, `protocol_sha256`,
`l0_commitment_sha256`, `evidence_sha256`, `frozen_bundle_sha256`, and
`runtime_fingerprint`. Host-private publication verifies stored evidence and
writes `active_protocol.json` atomically. Proposal objects cannot publish.

## Executable adapters

`runtime.py` holds a fixed `ProtocolRuntimeRegistry` resolving typed components.
It builds a private evaluation runtime for each protocol; old and proposed
protocols never share mutable factories or overwrite registered implementations.
Registry versions remain addressable after promotion. No plugin installer or
arbitrary module loader is added.

Each kind must change observable executable behavior in focused tests:

| Kind | Minimal versions | Immutable boundary |
|---|---|---|
| backbone | Last-value forecast and history-mean forecast | History-only input, finite horizon output; canonical Host scoring |
| loader | Canonical fixture JSON and alternate history-key fixture JSON | Both map to the same canonical `ContextTask` and task hashes; Host alone materializes future labels |
| verifier_strategy | Baseline exact-evidence check and an additional duplicate-evidence filter | Fixed Host checks always run; no strategy can bypass support/document-role validation |
| diagnostic_metric | Forecast spread and absolute forecast movement | Finite diagnostic output only; excluded from primary scores and acceptance rules |
| schema_migration | Identity envelope v1 and deterministic envelope v1-to-v2 conversion | Preserve embedded artifact bytes/SHA; original objects remain available |

Backbone output is injected through a Host-owned numerical registry built for
each protocol. Reuse Project 3 artifact/factory resolution; call its unchanged
underlying `PackagePipelineEvaluator._evaluate_components` boundary with the
protocol-specific registry and a composite protocol/Bundle evaluation SHA.
The existing cooperative adapter has strict artifact identity checks and is
not an injection seam: do not register altered forecasts under an old registry
SHA. The private runtime constructs new, correctly hashed derived registry
objects; the two archived Bundles remain the frozen recipes being compared.
The real Retrieval and Decision pipeline consumes those forecasts; smoke must
not return synthetic fitness. Loader and verifier strategies run inside the fixed
Host pipeline bridge, with canonical materialization and verification retained.
Diagnostics run after inference and cannot rewrite `PackageEvaluation` fields.

Migration covers a P5 transport envelope, not a rewrite of P1/P2/P3 schemas:
v1 is `{"schema_version":1,"artifact":PAYLOAD,"artifact_sha256":SHA}`;
v2 is `{"schema_version":2,"content":PAYLOAD,"content_sha256":SHA}`.
Unwrapping both must yield byte-identical canonical PAYLOAD. Conversion is
idempotent for v2, rejects other versions and hash mismatch, writes new objects,
and retains the explicit old-to-new envelope SHA mapping. Both frozen archive
Bundles and their referenced artifacts are included in the manifest closure.
No accepted Bundle/acceptance-evidence identity is regenerated by migration.

## Compatibility and automatic decision

Commit an input manifest before proposals: L0 and runtime identity, exact
Train/Dev memberships and task hashes, seed protocol, two accepted archive
Bundles with referenced artifacts, and ordered replacement templates. Paths
are Host configuration only. Bundle L0 identities must match the manifest.
One proposal is evaluated at a time against the then-active protocol, with
its parent SHA constructed by the Host from the committed replacement template.

`check_compatibility(old, proposed, corpus, registry)` checks:

1. L0 SHA unchanged; exact supported kinds, implementations, and versions.
2. Both loaders reproduce committed canonical hashes and inference views lack
   future/evaluator-only labels. A data-version break is explicitly rejected
   in this prototype; supporting one is a new experiment epoch.
3. Every supplied valid artifact in the manifest closure retains its original
   identity after migration/unwrapping; conversion repeats deterministically.
4. Four verifier fixtures: valid support accepted; nonexistent support,
   evaluator-only evidence, and fabricated document identity rejected. Proposed
   strategy runs in addition to the unchanged baseline Host verifier.
5. Re-evaluate both frozen archive Bundles under both protocols on the same
   four Train tasks. Cache identity includes protocol/runtime/corpus/Bundle/stage;
   do not reuse an old protocol's score or diagnostic ranking as new evidence.
6. Require complete finite coverage, no added invalid/catastrophic cases, and
   neither primary capped mean regressing by more than `1e-12` for either
   archive Bundle. Rank the new-protocol archive by Train joint error then SHA.
7. For a compatible Train-passing proposal, evaluate the fixed incumbent Bundle
   under both protocols on the one Dev task with the same nonregression rules.
   Dev results are written only to sealed Host evidence.

Diagnostic changes never redefine primary metrics. Attempted primary-metric
policy replacement is an unsupported L0 change and is rejected. The prototype
does not inherit champions across different primary metrics or data versions.
Archive re-ranking is descriptive Train evidence; the frozen incumbent Bundle
does not change during a protocol migration epoch.

`CompatibilityEvidenceV2` stores commitments, explicit check outcomes,
per-protocol Train aggregates, migrated-envelope mapping, and sealed Dev
evidence SHA. `decide_protocol(evidence) -> ProtocolDecisionV2` is a fixed
Host function yielding `accept|reject` plus stable reason codes. Compatibility
failure skips Dev. Acceptance publishes a new release only after evidence
readback. Rejection leaves previous active-pointer bytes exactly unchanged.
No human confirmation or acceptance flag is introduced.

## Persistence, resume, and frozen handoff

Write content-addressed objects under `protocols/`, `proposals/`, `evidence/`,
`migrations/`, and `releases/`, with Dev details under `sealed/`. Use existing
canonical/write-once and atomic JSON helpers. `progress.jsonl` records proposal
SHA, before/after protocol SHA, decision/reason, and closed step index only.

`protocol_checkpoint.json` commits input/config/runtime/L0 digests, next index,
active release SHA, completed proposal/evidence/release SHAs, progress prefix
hash, and budget counters. Persist one checkpoint per fully decided proposal.
Resume validates these identities and skips closed proposals. A test-only
`stop_after` stops at that boundary. Arbitrary process crashes between writes
are detected as incomplete; generic crash recovery is outside scope. Completed
runs validate inputs and existing completion and perform no writes. Measured
elapsed seconds live only in diagnostics, not deterministic result identity.

`freeze_protocol_handoff(release, bundle) -> dict` creates a canonical
`frozen_protocol_handoff.json` referencing the exact accepted Bundle, L0
commitment, accepted L1 release/protocol, sealed acceptance evidence, and runtime
identity. Resolve all references, require Bundle/release L0 agreement, and mark
`public_test_accessed=false`. A rejected proposal cannot be exported as accepted.
Project 4 currently exposes no protocol/Public handoff API, so this is a new
P5 export; it is not a claim that a Public evaluator exists. Existing unified
`public-evaluate` remains validation-only and is not changed to score Public.
The frozen export contains no learning entry point; a future evaluator must
consume its pinned protocol and write to a separate output directory.

## CLI and completion

Add `python -m evolving_loop.v2 protocol-evolve --config
configs/evolution_v2/protocol/smoke.json --input-manifest INPUT --output-dir RUN`
and `python -m evolving_loop.v2 protocol-make-smoke-inputs --output-dir INPUTS`.
Existing commands remain compatible. Repeating `protocol-evolve` resumes.

The deterministic sequence covers one accepted proposal of each kind plus a
loader compatibility rejection (changed task hash); construct a history-mean
fixture with strictly improved backbone forecasts and equivalent forecasts for
the other four changes. Use unique logical IDs for the rejected loader registry
entry so it is executable and fails compatibility rather than parsing.
Semantic completion contains `schema_version=1`,
`status="protocol_evolution_complete"`, `active_protocol_sha256`,
`active_release_sha256`, `accepted`, `rejected`, `completed_proposal_sha256s`,
`frozen_handoff_sha256`, and `public_test_accessed=false`.

Delivery evidence: five executable adapter kinds, canonical compatibility
accept/reject, exact rejection preservation, old-protocol replay after
acceptance, archive re-evaluation under separate cache keys, closed-boundary
resume equality, frozen handoff verification, and one unified offline CLI smoke.
Focused tests plus one CLI smoke are the completion gate; they do not establish
large-model quality, Public performance, or hostile-code security.
