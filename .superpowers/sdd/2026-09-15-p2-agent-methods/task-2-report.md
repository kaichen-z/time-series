# Task 2 Report: AlphaEvolve/Voyager proposer boundary

## Status

Implemented and verified. The exact new proposer request accepts bounded,
canonical `curriculum_targets` and SHA-bound `reusable_programs`; the canonical
LLM wire contains exact reusable source text. Existing response normalization
and the Host parser/sandbox gate were not changed.

Legacy requests with neither new context field remain readable. A new-context
request must supply both fields explicitly, including empty lists.

## RED evidence

Command:

`python -m pytest -q tests/test_evolution_v2_numerical_proposers.py tests/test_evolution_v2_numerical_agent_methods.py`

Observed before implementation: 58 failures, 76 passes. The proposer suite
failed because `curriculum_targets` and `reusable_programs` were unexpected in
the exact request schema. The focused Task 1 regression also failed because
`apply_prompt_train_credit` accepted `repair` feedback instead of requiring
`policy_tune`.

## GREEN evidence

Focused first GREEN:

`python -m pytest -q tests/test_evolution_v2_numerical_proposers.py tests/test_evolution_v2_numerical_agent_methods.py`

Result: 134 passed in 2.69s.

Fresh final verification (bytecode and pytest caches disabled):

`PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp/codex-task2-pytest python -m pytest -q -p no:cacheprovider tests/test_evolution_v2_numerical_proposers.py tests/test_evolution_v2_numerical_map_elites.py tests/test_evolution_v2_numerical_agent_methods.py tests/test_evolution_v2_numerical_persistence.py::test_contextual_identical_batches_remain_distinct_completion_attempts`

Result: 282 passed in 1.12s.

`git diff --check` also passed.

## Files

- `evolving_loop/v2/numerical_qd/proposers.py`
- `evolving_loop/v2/numerical_qd/agent_methods.py`
- `tests/test_evolution_v2_numerical_proposers.py`
- `tests/test_evolution_v2_numerical_agent_methods.py`
- `.superpowers/sdd/2026-09-15-p2-agent-methods/task-2-report.md`

`evolving_loop/v2/numerical_qd/map_elites.py` was intentionally unchanged:
Task 1's pure `select_reusable_programs` function already filters against the
Host-supplied feasible genome set and deterministically ranks target matches.

## Self-review and concerns

- Both context collections are capped at 32 records.
- Curriculum targets must name declared cells and use semantic canonical order.
- Reusable programs are contract-reparsed, SHA/source-text checked, target
  matched, source-unique, and ordered exactly like Task 1 selection.
- Records claiming the current parent genome must match the inventory member's
  exact source/applicability identity and cannot be quarantined.
- Source text is exempted from the metadata path-string check because valid
  Python source can contain division; all other metadata remains checked.
- No Dev/Public data, forecasts, raw objectives, paths, callbacks, or live
  objects were added to the boundary.
- Feasible-history eligibility for non-parent reusable genomes remains a Host
  construction responsibility through `select_reusable_programs`; runner/store
  population is intentionally deferred because Task 2 forbids runner
  integration.
- The worktree briefly ran out of disk space during verification. Only
  regenerable `.pytest_cache` and `__pycache__` directories in this worktree
  were removed; no `runs/` artifact was touched.
