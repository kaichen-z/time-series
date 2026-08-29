# Task 1 report — Single-task Numerical morphology smoke

## Scope

Added `python -m numerical_agent.run_morphology_smoke`: a one-task, history-only
Numerical morphology smoke path. It accepts explicit optional paths for reviewed statistical
methods, skill source, policy portfolio, frozen screening, and frozen Decision artifacts; it
also exposes the repository TSFM runtime/deployment options. The deterministic fake backend is
self-contained for tests. No Retrieval or Decision source was modified, and no formal Task-6
runner was used.

## TDD evidence

RED was captured before production code existed:

```text
$ pytest -q tests/test_run_morphology_smoke.py
ModuleNotFoundError: No module named 'numerical_agent.run_morphology_smoke'
1 error during collection
```

The first implementation run exposed a missing serializer on `SelectionArithmetic`, then the
JSON boundary exposed `inf` hindcast diagnostics for unavailable TSFMs. Both were traced to the
new result adapter and corrected by serializing the immutable recipe with `asdict` and emitting
non-finite unavailable diagnostics as JSON `null`.

GREEN and compatibility verification:

```text
$ pytest -q tests/test_run_morphology_smoke.py
6 passed

$ pytest -q tests/test_run_morphology_smoke.py \
    tests/test_numerical_morphology_loop.py \
    tests/test_evolution_morphology.py \
    tests/test_evolution_morphology_consistency.py \
    tests/test_evolution_policy_targetwise.py
98 passed

$ python -m py_compile numerical_agent/run_morphology_smoke.py
$ git diff --check
```

## Files

- `numerical_agent/run_morphology_smoke.py`
- `tests/test_run_morphology_smoke.py`

## Behavioral coverage

- Exact single-task selection; multi-task inputs require `--task-id` before runtime/model work.
- Fake `python -m` end-to-end output fields, exact-horizon forecast, safe assumption audit
  projection, freeze marker, and post-freeze trusted metrics.
- The Numerical loop receives a task with an empty `future`; labels are scored only afterward.
- Missing TSFM runtime is reported as unavailable while the statistical anchor completes.
- Parent-directory creation, explicit overwrite, and malformed task/result paths.

## Commit

Implementation commit: `dfb86477a09d19d74eef030478506d7759ef116a`

## Concerns

The no-path default deliberately uses deterministic smoke statistical leaves so fake tests do
not depend on mutable local repositories. A real run should pass the explicit reviewed
`--methods-path`, `--policies-path`, `--screening-path`, and `--decision-path` artifacts.

## Fix round 1 — independent review findings

Implemented the five requested corrections in `24983c0d792b0ebf0bf73a5151725a945aa2bc7e`:

- JSONL `--task-id` selection scans only `benchmark_id` tokens, decodes exactly the selected
  record, constructs a history-only task, and extracts future labels only after the immutable
  package is returned.
- Real (`codex`) mode now requires explicit reviewed methods, skills, policies, screening, and
  Decision artifacts. Synthetic statistics/screening/Decision remain fake-only.
- Every supplied artifact is SHA-256 fingerprinted; methods use a content digest rather than a
  path fingerprint; the Decision's literal `SCREENING_POLICY_HASH` must exactly equal the
  supplied screening artifact digest.
- Non-overwrite output creation uses exclusive `x` mode, while `--overwrite` writes and fsyncs a
  same-directory temporary before atomic replacement.
- Post-freeze metric arithmetic catches overflow/non-finite results and fails before JSON
  serialization.

Additional RED evidence before these production changes:

```text
$ pytest -q tests/test_run_morphology_smoke.py
FAILED test_task_id_scans_jsonl_ids_without_decoding_an_unselected_task_body
AssertionError: unselected task body was decoded
FAILED test_real_mode_requires_an_explicit_reviewed_artifact_bundle
Failed: DID NOT RAISE SmokeError
2 failed, 6 passed

$ pytest -q tests/test_run_morphology_smoke.py
FAILED test_non_overwrite_result_creation_has_one_winner_under_a_race
AttributeError: module ... has no attribute '_write_result'
FAILED test_nonfinite_post_freeze_metrics_are_rejected_before_json_encoding
OverflowError: intermediate overflow in fsum
2 failed, 9 passed
```

GREEN evidence:

```text
$ pytest -q tests/test_run_morphology_smoke.py
11 passed

$ pytest -q tests/test_run_morphology_smoke.py \
    tests/test_numerical_morphology_loop.py \
    tests/test_evolution_morphology.py \
    tests/test_evolution_morphology_consistency.py \
    tests/test_evolution_policy_targetwise.py
103 passed

$ python -m py_compile numerical_agent/run_morphology_smoke.py
$ git diff --check
```

Review follow-up concern: the previously suggested v13 screening artifact hashes differently
from the supplied frozen Decision artifact's declared binding; the CLI now rejects that mismatch
instead of silently combining incompatible reviewed artifacts.

## Fix round 2 — re-review boundary hardening

Implemented `7e28894944aa8c8aa1527280ee22dcd7878329a1` to address the second re-review:

- The selected task remains raw bytes until post-freeze scoring. Pre-freeze JSON decoding uses a
  structural scanner that replaces every `future_values` JSON value with `null`, so label numeric
  tokens are never decoded. The history projection is a normal `common.data.Task` with an empty
  `future_values` tuple before it is converted to the Numerical loop task.
- `--task-id` is rejected unless it is one exact basename component; directory candidates cannot
  escape their source, and the decoded top-level `benchmark_id` must equal the selected ID for
  directory, JSON, and JSONL inputs.
- Reviewed artifacts are read exactly once into a private temporary snapshot. SHA-256 values are
  calculated from those bytes and every methods/skills/policy/screening/Decision consumer reads
  the snapshot rather than the caller path.

RED evidence before these changes:

```text
$ pytest -q tests/test_run_morphology_smoke.py
FAILED test_selected_future_json_is_not_decoded_until_after_package_freeze
AssertionError: future labels were decoded before package freeze
FAILED test_selected_task_uses_the_common_history_only_task_model
AttributeError: '_LoadedTask' object has no attribute 'task'
FAILED test_task_id_rejects_traversal_and_decoded_id_mismatches
Failed: DID NOT RAISE SmokeError
FAILED test_artifact_snapshot_binds_the_hashed_bytes_to_execution_input
AttributeError: module ... has no attribute '_ArtifactSnapshots'
4 failed, 11 passed
```

GREEN evidence:

```text
$ pytest -q tests/test_run_morphology_smoke.py
15 passed

$ pytest -q tests/test_run_morphology_smoke.py \
    tests/test_numerical_morphology_loop.py \
    tests/test_evolution_morphology.py \
    tests/test_evolution_morphology_consistency.py \
    tests/test_evolution_policy_targetwise.py
107 passed

$ python -m py_compile numerical_agent/run_morphology_smoke.py
$ git diff --check
```

## Fix round 3 — escaped future-key masking

Implemented `1ca99d83871f9032f95b90b47174eb17e2517bf7` for the remaining review
finding. The pre-freeze history projection now scans JSON objects and arrays
structurally. It decodes only isolated quoted property-name tokens, so escaped
spellings such as `"future\\u005fvalues"` are recognized, while the associated
JSON value is kept as an unparsed raw span and replaced with `null`. All future
labels remain decoded only by the post-freeze extraction step.

The new regression parametrizes two real JSON layouts: the ordinary key order
and a reordered series/task layout. Both use an escaped key and whitespace
before the colon; each raises if a pre-freeze `json.loads` sees its label array.

RED evidence before the structural scanner change:

```text
$ pytest -q tests/test_run_morphology_smoke.py
.........FF......                                                        [100%]
FAILED test_escaped_future_key_is_structurally_masked_before_freeze[True-False]
FAILED test_escaped_future_key_is_structurally_masked_before_freeze[True-True]
2 failed, 15 passed in 1.08s

AssertionError: escaped future labels were decoded before package freeze
```

GREEN and compatibility evidence:

```text
$ pytest -q tests/test_run_morphology_smoke.py
17 passed in 2.33s

$ pytest -q tests/test_run_morphology_smoke.py \
    tests/test_numerical_morphology_loop.py \
    tests/test_evolution_morphology.py \
    tests/test_evolution_morphology_consistency.py \
    tests/test_evolution_policy_targetwise.py
109 passed in 5.33s

$ python -m py_compile numerical_agent/run_morphology_smoke.py
$ git diff --check
```

## Final review wave — output identity, worker availability, and tool-window contract

Implemented `f8f359b3bdea1598bf04e93ba0cc154f4cf80060` for the final review wave:

- Before artifact loading, worker setup, or model work, `--results-path` is compared by resolved
  file identity and inode identity against the selected task source, each reviewed artifact, and
  `--tsfm-workers-config`. `--overwrite` now rejects direct, symlink, and hardlink aliases
  without replacing caller-owned inputs.
- The smoke command snapshots a worker config before registry creation. If the shared validator
  identifies an absent interpreter path, smoke-only filtering removes that environment and retries
  the unchanged shared registry. Its affected TSFMs become per-candidate unavailable. Invalid JSON,
  schema/security validation, non-executable interpreters, and virtual-environment/integrity
  failures remain fatal; the global registry was not modified.
- Both morphology prompts now include the exact tool-window template
  `"window":{"start":0,"end":N}` and say that `start` is inclusive and `end` is exclusive.
  The strict parser remains unchanged: observed Codex shapes with
  `start_inclusive`/`end_exclusive` keys or a `[0,156]` window are still rejected.

RED evidence before these production changes:

```text
$ pytest -q tests/test_run_morphology_smoke.py
...............FFFFFFFFFFFFFFFFFFFFFF...                                 [100%]
FAILED test_overwrite_rejects_task_and_configuration_identity_aliases_before_model_work
  (21 direct/hardlink/symlink cases across task, five artifacts, and worker config)
FAILED test_absent_worker_interpreter_leaves_worker_tsfms_unavailable
22 failed, 18 passed in 2.21s

ValueError: worker environment 'uni2ts' interpreter does not exist

$ pytest -q tests/test_evolution_morphology.py
...........FF....                                                        [100%]
FAILED test_window_prompt_gives_the_canonical_contract_without_loosening_real_response_parsing
  (real_luna start_inclusive/end_exclusive and real_terra [0,156] fixtures)
2 failed, 15 passed in 0.10s
```

GREEN and compatibility evidence:

```text
$ pytest -q tests/test_run_morphology_smoke.py tests/test_evolution_morphology.py
57 passed in 1.56s

$ pytest -q tests/test_run_morphology_smoke.py \
    tests/test_numerical_morphology_loop.py \
    tests/test_evolution_morphology.py \
    tests/test_evolution_morphology_consistency.py \
    tests/test_evolution_policy_targetwise.py \
    tests/test_numerical_tsfm_deployment.py \
    tests/test_numerical_dictionary_contracts.py
170 passed in 3.95s

$ python -m py_compile numerical_agent/run_morphology_smoke.py numerical_agent/evolution/morphology.py
$ git diff --check
```
