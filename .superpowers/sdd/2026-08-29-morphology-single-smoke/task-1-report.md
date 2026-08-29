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
