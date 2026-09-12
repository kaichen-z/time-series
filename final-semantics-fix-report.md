# Final semantics fix report

Implemented I2, I3, and I4 in the numerical task-local shortlist path.

- I2: `task_morphology_key` now has one canonical implementation in
  `numerical_agent.evolution.task_shortlist`; OOF prior fitting and shortlist
  selection use that same key. The ranking regression covers two history
  buckets with reversed candidate advantages and otherwise tied global priors.
- I3: a diagnostic-only exception now leaves a valid full forecast intact and
  records no diagnostic. The tournament consequently excludes the specialist
  and records its exact-anchor fallback.
- I4: shortlist validation now requires one through ten selected names,
  selected/excluded disjointness, and an underfilled flag exactly matching the
  fixed minimum of six. `validate_task_candidate_shortlist` is reusable by
  artifact readers.

Verification:

- `pytest -q tests/test_task_shortlist.py tests/test_task_local_evolution.py tests/test_task_local_ensemble_cli.py` — 45 passed

An earlier adjacent frozen-package run passed (42 tests). A final rerun after
concurrent protected `evolving_loop` edits found six unrelated failures:
`task_local_result_sha256` calls `dataclasses.asdict` on a `mappingproxy`.
That failure is outside this change's ownership and was not modified here.
