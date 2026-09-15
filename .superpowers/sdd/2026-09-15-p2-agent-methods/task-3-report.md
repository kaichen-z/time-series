# Task 3 report

## TDD evidence

- RED: `test_runner_reusable_context_requires_feasible_store_verified_programs` failed with missing `_trusted_reusable_program_context`.
- GREEN: the test passes after adding Host-side archive/Store verification and filtering.
- Focused verification: proposer context/malformed tests 24 passed; persistence proposal/generation-status tests 6 passed; artifact legacy taxonomy tests 2 passed; runner authority test 1 passed.
- A broader runner invocation was stopped after the environment exhausted temporary disk; pytest-only temporary artifacts were removed.

## Changes

- Runner derives curriculum targets from Host-declared morphology and archive state, and derives reusable program context only from feasible archive entries whose genomes, inventories, statuses, and source bytes pass Store verification.
- Sampled genome keeps its own proposer prompt; mutation prompt population is seeded, selected deterministically, persisted as an immutable control object, restored from generation status on resume, and its selected lineage receives only Host Train `policy_tune` credit.
- Proposal attempts and generation records commit prompt-population, curriculum, and reusable-program identities.
- Persistence accepts the new contextual proposal commitments while preserving legacy proposal contexts; generation-status validation accepts legacy and new records.

## Review-round fix TDD evidence

- RED: runner Host-child test failed because `_host_mutation_prompt_child` was absent; the initial focused run also caught an unreachable population return.
- GREEN: Host derives a bounded `policy_tune` child with parent lineage; runner inserts it before selected-lineage Train credit, and persistence requires selected-prompt membership in the committed population.
- Focused verification: `PYTHONDONTWRITEBYTECODE=1 pytest -q ...` selected runner/proposer checks: 2 passed, 237 deselected. An attempted mixed focused invocation reached unrelated legacy runner tests and exposed only the temporary helper placement regression, which was corrected.

## Review-round 2 TDD evidence

- RED: true pre-context request shape (without all three reusable/curriculum commitments) was rejected; crash-window resume had no durable pending-generation decision.
- GREEN: oldest request schema is accepted; a `proposal_pending` generation marker records the exact request SHA and is checkpointed before provider dispatch; resume fails closed unless a completed attempt resolves it.
- Focused verification: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_evolution_v2_numerical_proposers.py -k 'legacy or mutation_prompt_context'` — 2 passed, 129 deselected; runner Host/context checks — 2 passed, 107 deselected.

## Review-round 3 TDD evidence

- RED: a proposal attempt alone incorrectly resolved `proposal_pending`, allowing resume to abandon a generation before terminal accounting.
- GREEN: pending state now resolves only with a same-generation terminal status carrying both exact request and attempt commitments; otherwise resume fails closed.
- Focused verification: pending/Host/legacy checks — 2 passed, 239 deselected.
