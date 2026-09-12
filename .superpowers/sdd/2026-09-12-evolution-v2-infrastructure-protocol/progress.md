# SDD ledger — plan: docs/superpowers/plans/2026-09-12-evolution-v2-infrastructure-protocol.md

Status: design/plan preparation only; no P5 implementation task dispatched.
Spec: docs/superpowers/specs/2026-09-12-evolution-v2-infrastructure-protocol-design.md
Precondition: Project 4 focused deterministic gate passes before Task 1.
Budget: five tasks, 15 + 30 + 30 + 25 + 20 = 120 minutes.
Current source context: Project 3/4 workers own their source; P5 planning edits
only its new docs and this plan-scoped SDD workspace.

## Preflight table

| Pair/task | Producer / consumer or self-check | Finding |
|---|---|---|
| 1 / 2 | Typed protocol and component -> runtime registry | Matching kinds/order/version fields |
| 1 / 3 | Protocol SHA/L0 identity -> evidence comparisons | L1 SHA separate from immutable Bundle field |
| 1 / 4 | Proposal/release constructors -> runner/publication | Exact fields and version increment agree |
| 1 / 5 | Strict artifacts -> manifest/CLI | Seed and replacement fields agree |
| 2 / 3 | ProtocolHostInputs/resolve/evaluate -> compatibility | Explicit Host-only input type and pipeline signature |
| 2 / 4 | Registry + Host inputs -> runner | Candidate inputs never contain runtime factories |
| 2 / 5 | Adapter implementation IDs -> smoke templates | All five accepted implementations + changed-history negative fixture |
| 3 / 4 | CompatibilityEvidenceV2/decide_protocol -> publication | Sealed Dev SHA, fixed reason codes, complete check booleans |
| 3 / 5 | 4/1 corpus + two Bundles -> fixture manifest | All referenced artifacts included; no Public |
| 4 / 5 | run_protocol_evolution/freeze_protocol_handoff -> CLI | Input/output signatures and completion fields agree |
| Task 1 | Test/code/files | Strict tagged component replacement and version 1-to-2 |
| Task 2 | Test/code/files | Real mean/last-value pipeline, equivalent loader, envelope conversion |
| Task 3 | Test/code/files | Uses existing capped mean_smae/mean_srmse fields; early rejection skips Dev |
| Task 4 | Test/code/files | Closed-boundary checkpoint, immutable originals, exact rejection preservation |
| Task 5 | Test/code/files | Two unified commands, six proposals, one smoke, no full regression |

## Rulings

Ruling: keep L1 protocol identity in a separate release instead of changing EvolutionBundleV2.protocol_fingerprint — that field currently commits L0 and ownership is immutable — cost if wrong: future public/runtime consumers need an explicit release resolver.
Ruling: support only a deterministic P5 envelope v1-to-v2 migration and reject data-version/primary-metric breaks — enough to test migration identity without an extreme schema matrix — cost if wrong: broader migrations require a new experiment/addendum.
Ruling: export a frozen protocol handoff but do not implement Public scoring — existing P4 plan has no Public handoff API and unified Public currently validates only — cost if wrong: Public experiments need a later evaluator integration.
Ruling: use the existing PackagePipelineEvaluator._evaluate_components scoring seam with correctly hashed derived numerical registries — CooperativePipelineAdapter intentionally binds original registry identities and cannot inject alternate backbones — cost if wrong: factory-resolution code may need a small P5-local adapter adjustment; no existing ownership checks may be weakened.

## Task state

Task 1: complete — contracts (15 minutes); implementer BASE `020a3f7`.
Task 2: pending — runtime adapters (30 minutes).
Task 3: pending — compatibility gates (30 minutes).
Task 4: pending — authority/resume/handoff (25 minutes).
Task 5: pending — unified CLI/fixture/docs (20 minutes).

## Planning validation

Specification coverage: typed/versioned five-kind proposals (1), actual adapters
and migration (2), canonical tasks/labels/verifiers/archive scoring (3), automatic
decisions and exact resume/frozen export (4), unified offline CLI (5).
No runtime test result is claimed during planning. Controller should record BASE
and agent identity on actual dispatch; reports use task-N-report.md.

## Execution record

Task 1 RED: `pytest -q tests/test_evolution_v2_protocol_contracts.py` failed at
collection with `ModuleNotFoundError: No module named 'evolving_loop.v2.protocol'`.
Task 1 GREEN: the same focused command passed `9 passed in 0.03s` after the
new strict protocol contracts were added. Scope is limited to the Task 1
protocol package, its focused test, and this SDD record.
