# Task 3 report — task evaluation recovery and Decision diagnostics

Base: `fd16f18`. Scope: the supplied Task 3 brief. No model calls, pushes,
P2 freeze fixes, root terminal-policy changes, or Task 2 counting changes.

## Result

- Final Decision prompts explicitly make the supplied `host_default_id`
  authoritative while preserving citations and all host validators.
- Dictionary planning/selection remove the seed's final-decision JSON schema,
  retain evolved strategy text, and supply a single phase output contract.
- `decision_contract_rejected` remains the aggregate fallback category;
  provisional/final Decision artifacts keep the original rejection text.
- Dictionary traces retain successes, failed requests, exception types/messages,
  and planning/selection raw responses. Provider errors no longer discard earlier
  successful evaluations. No guessed historical failure reason is asserted:
  r3 did not preserve the original rejection, so fixtures reproduce the concrete
  schema/default defects rather than claim an exact historical reconstruction.
- Optional `PackageTaskStore` stores atomic start/completed/error records.
  Completion stores validated `PackageTaskScore`, diagnostics and error artifacts;
  interruption never stores a reusable completed score. Nonfinite rejected tool
  inputs become diagnostic strings, never score values.
- `run_real_cooperative` wires the store into the real adapter at
  `<output_dir>/task_evaluations/<identity_sha256>.json`.

## Reuse contract and limits

The key includes full resolved task fingerprint (Host-only labels included),
Numerical package fingerprint, Bundle, train/dev stage, expected Retrieval and
Decision identities, Host runtime plus model binding, metric policy/cap, and
`TASK_EXECUTION_CONTRACT = cooperative-task-execution-v1`. This explicit contract
must be bumped for execution/scoring changes not bound elsewhere. The Host's
resource reporter hash alone is not treated as Python source identity.

Only completed matching records are loaded, with typed score validation, task/
package binding checks and a result-content digest. Started/error tasks rerun.
Malformed completed records fail closed. The helper assumes a single writer and
trusted local output ownership; it is not a database or concurrent lock service.
Files use flush/fsync plus atomic replacement, not an append-only attempt history.
Latest task attempts replace older start/error state under the same exact key.

This is task-evaluator reuse, not root-run resume authorization. Task 1 deliberately
terminalizes transient root failures; that root still refuses resume. Generic
adapters default to no task persistence. A full-result `trace_sink` cannot be
combined with score-only reuse because a cached row cannot replay its callback.
Task records are diagnostics/evaluation cache, not sealed promotion authority.
All metrics, finite-call/work limits, split gates, and acceptance rules remain in
their original paths. Full historical r3 rejection reconstruction is unavailable.

## TDD evidence

Python: `/Users/yyoraa/time-series/.venv/bin/python`.

RED:

```sh
python -m pytest tests/test_task_evaluation_recovery.py -q
```

13 failing cases initially identified missing task-store API, overwritten rejection,
missing Dictionary traces and conflicting prompts. The first rejection fixture
was corrected to include a valid handoff, then rerun alone: 1 failed specifically
because `decision_contract_rejected` replaced the original rejection. Exact original
reason also includes the existing `invalid_retrieval_gaps:gaps must be a list` suffix.

Further focused RED commands:

```sh
python -m pytest tests/test_task_evaluation_recovery.py::test_dictionary_tool_errors_remain_in_result tests/test_task_evaluation_recovery.py::test_completed_fallback_keeps_original_error_and_rejection -q
python -m pytest tests/test_task_evaluation_recovery.py::test_nonfinite_rejected_tool_input_does_not_abort_score_persistence -q
```

Respectively: 2 failed for missing failed-request details/start timestamps;
1 failed for NaN diagnostic serialization aborting a completed score.

GREEN:

```sh
python -m pytest tests/test_task_evaluation_recovery.py tests/test_decision_dictionary_tools.py tests/test_numerical_retrieval_handoff.py tests/test_package_decision_evolution.py tests/test_evolution_v2_cooperative_pipeline.py tests/test_evolution_v2_real_cooperative.py -q
```

148 passed in 33.80s before the final prompt-invalidation and NaN diagnostic cases.
The preceding regression run had 3 expected old-contract assertion failures:
generic rejection wording and two exact system-prompt assertions. Updates retain
the protected forecast, aggregate fallback, and frozen caller-scope checks.

Final integrated command:

```sh
python -m pytest tests/test_task_evaluation_recovery.py tests/test_decision_dictionary_tools.py tests/test_numerical_retrieval_handoff.py tests/test_package_decision_evolution.py tests/test_package_coordinate_e2e.py tests/test_package_task_feedback.py tests/test_evolution_v2_cooperative_pipeline.py tests/test_evolution_v2_cooperative_runner.py tests/test_evolution_v2_cooperative_safety.py tests/test_evolution_v2_real_cooperative.py tests/test_evolution_v2_real_runner.py tests/test_evolution_v2_effective_trials.py tests/test_evolution_v2_budget.py -q
```

Result: **227 passed in 61.55s**. `git diff --check` passed.

## Files and self-review

Production: `decision_agent/agent.py`, `numerical_two_stage.py`,
`package_pipeline_evaluator.py`, new `package_task_store.py`,
`v2/cooperative/adapters.py`, `v2/real/bridges.py` (all under `evolving_loop/`).
Tests: new `test_task_evaluation_recovery.py`, existing
`test_decision_dictionary_tools.py`, `test_numerical_retrieval_handoff.py`,
`test_evolution_v2_real_cooperative.py` (all under `tests/`).
Usage note: `docs/evolution-v2-cooperative-bundle.md`, including shortest adapter
canary invocation and exact artifact path/reuse limitations.

Self-review checked unchanged fallback scoring, retrieval/prompt binding checks,
aggregate rejection category, finite Dictionary work bound, and root policy.
New task tests cover transient/keyboard interruption, process-style reconstruction,
stage/bundle/prompt/runtime/metric/contract/task/package invalidation, incomplete
records, invalid typed scores, exact rejection, retained contract errors, and NaN
in rejected tool diagnostics. Parent controller owns the bounded live canary.
