# Task 6 Report: Migration audit and one-task smoke

## Status

Implementation and local verification are complete on `feature/morphology-single-smoke`.
The scoped commit is created after this report is finalized. Nothing was pushed or merged.

## Changes

- Added a semantic AST audit over the active Numerical evolution modules. Legacy
  MASE/MAE/sMAPE/RMSSE operations are permitted only as exact, counted AST-operation
  nodes for diagnostics, serialization, or opt-in legacy readers.
- Migrated Morphology assumption ranking and package ranking from legacy MASE summaries
  to capped Dr-CiK sMAE/sRMSE joint summaries and per-metric stability fields.
- Removed the final active MASE seam from residual correction and long-horizon handling:
  residual clipping scales now come from aligned historical fold truth, never from
  `mase_scale`, MAE, or MASE diagnostics.
- Made both explicit legacy Decision readers reject `median_smape`, which has no valid
  lossless migration into the scaled pair policy.
- Made active Markdown renderers consume the already-bound canonical metric-policy
  fingerprint instead of synthesizing a missing value.
- Extended the deterministic morphology smoke artifact with a canonical `selection`
  contract, read-only evolution boundary, and separate Statistical, TSFM, and Combined
  execution/failure summaries.
- Updated the English/Chinese Numerical documentation to state the schema-v2 scaled-pair
  authority while retaining the prior 99-task numbers unchanged and explicitly historical.
- Migrated one stale mixed-runtime integration fixture to the exact schema-v2 dictionary and
  config envelope required by the active loader.

## TDD evidence

- Initial Task-6 contract RED: the audit found legacy ranking fields in active Morphology
  paths, the renderer test found fallback fingerprints, and the fake smoke lacked its
  selection/evolution/family contract.
- Explicit legacy Decision RED: `median_smape` was advertised by migration maps but could
  not be consumed by the active policy.
- Deep active-path RED after the first pass:
  `2 failed, 1 passed`; the semantic audit and behavior test proved that changing only the
  legacy `mase_scale` changed residual-correction behavior.
- After the fix, the exact audit/behavior slice passed: `3 passed, 122 deselected`.
- Selector and smoke compatibility passed: `125 passed`.
- Independent review found that two scaled-only consumers still used a different
  tie-break sequence than `DecisionPolicy`. Two regression tests failed before the fix;
  assumption leaders and ranked alternatives now share the canonical
  median-joint/recent-joint/worst-joint/median-sMAE/median-sRMSE/name order.

## Deterministic task_42 fake smoke

Command:

```bash
python -m numerical_agent.run_morphology_smoke \
  --task-file /Users/yyoraa/time-series/external/Dr-CiK/sample/tasks/task_42.json \
  --results-path /tmp/numerical-task42-fake-smoke-task6-final.json \
  --llm-backend fake \
  --hindcast-folds 2
```

Observed artifact:

- Metric-policy fingerprint:
  `fe5cf0fd10839f93a3dea81cb90a63641ad520a589bdce875bd11095a2a8bf8d`.
- Decision metrics: `smae`, `srmse`; Dev read-only; Public/hidden mutation disabled.
- Statistical: 6 attempted, 6 successful.
- TSFM: 5 attempted, 0 successful, 5 explicitly unavailable.
- Combined: 5 attempted, 0 successful, 5 explicitly unavailable because their required
  leaves did not materialize.
- Selected candidate: `naive_last` with weight 1.0.
- Post-freeze trusted execution diagnostics: sMAE `0.2248109293847605`, sRMSE
  `0.2750891381127843`, MAE `104.47841999999999`, MASE `1.3769179524948298`.

This is an execution smoke only, not a benchmark result. In particular, unavailable TSFM
and Combined families remain visible and are not relabelled as successful.

## Real task_42 smoke decision

No real-checkpoint smoke was run. Local Python environments and a worker-config file exist,
but no checked-in attestation proves that the exact immutable checkpoints were loaded and
successfully smoked. `docs/tsfm-runtime-matrix.md` explicitly records that no such attestation
exists, and the repository search found no attestation artifact. Per the Task-6 safety boundary,
an environment directory is not sufficient evidence and no model was downloaded, substituted,
or fabricated.

The exact prerequisite for the real smoke is an operator-reviewed artifact bundle
(`methods.py`, `skills.py`, `policies.py`, Screening and Decision sources), a local deployment
binding, and a persisted attestation for the exact checkpoint/adapter/runtime identity (including
the immutable checkpoint or model digest) showing a successful real-checkpoint load.

## Verification

- Required focused suite before review: `505 passed in 6.07s`.
- Final expanded focused suite after the review fix: `514 passed in 6.11s`.
- The first full-suite attempt exposed one stale schema-v1 integration fixture:
  `1 failed, 2275 passed, 1 skipped in 55.83s`. The fixture was migrated through the explicit
  legacy dictionary converter into the strict schema-v2 envelope.
- Pre-review full suite: `2277 passed, 1 skipped in 54.77s`.
- Final post-review full suite: `2279 passed, 1 skipped in 56.61s`.
- `python -m compileall -q common numerical_agent`: passed.
- `git diff --check`: passed.
- Both HTML documents passed duplicate-ID and missing-fragment-link checks.
- `runs/numerical_morphology/` remained 10 files with the same aggregate SHA-256 before and
  after Task 6:
  `703fbe89573c7e86a6e96a50f79629c08d8a7bddc90113663ddd340a52501b25`.

## Self-review

- Confirmed active ordering reads capped joint sMAE/sRMSE summaries and active acceptance
  retains independent per-metric guards; diagnostic legacy fields are not decision inputs.
- Confirmed Statistical, TSFM, and Combined candidates use the same scaled scoring kernel and
  smoke failures remain family-specific.
- Confirmed active schema/fingerprint consumers fail closed; report renderers cannot invent a
  missing fingerprint.
- Confirmed Dev remains read-only and no 80/20, Public-99, official hidden, or probabilistic
  evaluation was run.
- Confirmed the old 99-task tables and their numeric values were not rewritten.
- Independent review reported no other Critical or Important findings after identifying
  the two ordering inconsistencies above; both were fixed and reverified.

## Concerns

The repository is not currently eligible for the requested real-checkpoint smoke because it lacks
the required immutable load attestation. This is an honest prerequisite gap, not a failed
forecasting result.

## Fix round 1: dual-metric contract and exact-node audit

Independent review of the initial Task-6 commit found three active-contract gaps. They were fixed
with new failing tests before production changes:

1. A `DecisionPolicy` could previously accept a nonempty scaled subset such as
   `("median_smae",)`. Active and mutated policies now require the complete canonical ranking
   core: all three median/recent/worst joint scaled fields must precede the paired median sMAE and
   median sRMSE tie-breakers; only non-error safety fields may follow. Thus a complete-looking
   policy cannot restore single-metric authority by putting `recent_smae` first. Explicit legacy
   readers normalize unpaired old rankings to the complete pair. The smoke's `decision_metrics`
   and `decision_ranking_order` are derived from that validated policy and checked against the
   bound metric-policy metadata.
2. Selector active-oracle regret previously chose its oracle by sMAE alone, and Train/Dev tail
   gates checked only capped sMAE tails. Oracle identity is now ordered by the joint scaled pair;
   sMAE and sRMSE regrets are retained separately. Train, read-only Dev, cross-fold, and
   activation-aware gates now check both capped and raw P90/P95 tails, both clipped counts, and
   both oracle regrets. The Markdown selector report also renders the regret pair rather than the
   old scalar summary.
3. The semantic audit previously missed direct metric calls and dynamic access patterns and
   allowlisted entire functions. It now detects `ast.Name` calls, mapping `.get`, subscripts,
   `getattr`, and metric strings, then binds every allowed operation to the SHA-256 prefix of its
   normalized enclosing AST statement. A same-function diagnostic-to-authority repurpose therefore
   changes the node identity and fails the audit.

### Fix-round TDD evidence

- RED: incomplete direct and mutated ranking policies were accepted; the smoke omitted the
  validated ranking order; a sMAE-only oracle controlled regret; capped/raw sRMSE-tail
  adversaries passed; and the audit returned no operations for dynamic legacy access.
- GREEN: the final selector and smoke focused regression completed with `353 passed in 3.56s`.
- Deterministic task_42 fake smoke revalidated the same metric-policy fingerprint
  `fe5cf0fd10839f93a3dea81cb90a63641ad520a589bdce875bd11095a2a8bf8d`, the complete
  canonical ranking order, sMAE `0.2248109293847605`, and sRMSE `0.2750891381127843`.
- `python -m compileall -q common numerical_agent` and `git diff --check` passed.
- Final independent re-review reported no remaining Critical or Important findings after the
  canonical-order and statement-hash fixes.
- Per the review-fix instruction, the already-passing full suite was not run a second time.
- `runs/numerical_morphology/` remains untouched with the same aggregate SHA-256:
  `703fbe89573c7e86a6e96a50f79629c08d8a7bddc90113663ddd340a52501b25`.

## Fix round 2: indirect authority audit

A later review found that statement hashing alone remained fail-open under one indirect pattern:
an allowed diagnostic statement such as `observed = score.mean_mase` could remain byte-for-byte
unchanged while a later statement in the same function used `observed` to accept a policy.

Strict TDD captured that exact attack. Before the fix, the diagnostic and authority variants both
produced the same audit identity
`active:attribute:mean_mase@7c38aada28e06802`, so the adversarial test failed. Each allowance is
now bound to two normalized AST identities:

- the exact enclosing statement SHA-256 prefix; and
- the complete enclosing function-body SHA-256 prefix.

Consequently, changing either the metric operation itself or any downstream logic in its function
changes the audit identity and fails closed. The exact existing diagnostic, report-only, and
explicit legacy-reader bodies remain individually allowlisted.

Verification for this bounded fix:

- Exact indirect adversary: RED before the function-body binding, GREEN afterward.
- Semantic audit slice: `4 passed in 0.40s`.
- Audit plus smoke/report focused tests: `89 passed in 2.99s`.
- No full suite was run, as explicitly required for this bounded round.

## Final-review bridge fix: typed Decision authority and frozen execution scope

The final whole-branch review identified two bridge-specific authority gaps after the
Numerical migration itself was complete.

1. Final Decision overrides previously needed only a document ID present in the flattened
   legacy Retrieval projection. The host now keeps the verified `FinalRetrievalCard` through
   Decision validation, maps each opaque Retrieval assumption ID back to the exact accepted
   Numerical `AssumptionGrounding.candidate_names`, and accepts an override only when one fully
   cited verified chain either supports an assumption naming the selected candidate or challenges
   an assumption naming the protected host default. Round-1-only, neutral, unresolved,
   wrong-polarity, unrelated, or partially cited chains cannot authorize a final override.
2. The caller's `DecisionAgent` was previously reused across both Decision calls. The bridge now
   freezes the prompt and a deeply detached `persist=False` Decision Skill snapshot before Round 1,
   constructs a base `DecisionAgent` with the original LLM, and uses only that executor for both
   calls, host validation, and fingerprints. Caller prompt/library/row drift and subclassed
   `run()` implementations therefore have no execution authority, and no Decision Skill write is
   possible on this inference path.

Strict TDD reproduced both defects before production changes. The new adversarial/positive slice
first failed 8 of 10 cases; the separate provisional-gap contract also failed before its phase
boundary was added. After implementation, all 44 bridge tests and 148 affected
bridge/Retrieval/Decision tests passed. The final full-suite and independent-review evidence are
recorded with the scoped fix commit.
