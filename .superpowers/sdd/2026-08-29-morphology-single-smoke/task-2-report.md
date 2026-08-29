# Task 2 report — Exact Morphology window prompt

## Scope

Changed only the Morphology prompt contract and its regression test. The strict parser remains
unchanged: tool windows must be objects whose keys are exactly `start` and `end`.

## Root cause and fix

Live Codex responses used alternate window field names (Luna and Sol) or a two-item array
(Terra). The old prompt gave only a full-window shape and its initial context exposed the
alternate names, which was contradictory for a recent window.

The revised system prompt now:

- shows exact full and recent tool-action examples with the canonical window keys;
- describes `start` as inclusive and `end` as exclusive without printing the alternate
  identifiers;
- requires an object rather than an array and forbids alternative window key names; and
- removes the conflicting window-contract fields from the initial JSON context.

## Strict TDD evidence

- **RED:** `pytest -q tests/test_evolution_morphology.py -k window_prompt -x` failed before the
  prompt change because the required full/recent examples were absent and the initial context
  still contained alternate field names.
- **GREEN:** the same command passed all three parameterized regressions after the prompt-only
  change.

The regression covers the two observed alternate-key responses (including a nonzero recent start)
and the observed array response. It also asserts both exact canonical examples, inclusive/exclusive
language, explicit object-only/alternative-key rejection, and absence of the misleading
identifiers throughout the outbound prompt.

## Verification

- `pytest -q tests/test_evolution_morphology.py tests/test_run_morphology_smoke.py` — 58 passed.
- Reconstructed the combined system and initial prompt in Python; exact canonical examples were
  present and the misleading identifiers were absent.
- `git diff --check` — no whitespace errors.

## Self-review

Reviewed the final diff against the task brief. Only `numerical_agent/evolution/morphology.py`
and `tests/test_evolution_morphology.py` are implementation/test changes; no parser behavior was
relaxed. Per the task instruction, no subagent or external reviewer was used.
