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
