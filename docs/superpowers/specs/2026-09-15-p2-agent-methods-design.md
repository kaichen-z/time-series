# P2 Agent-Method Integration Design

## Goal

Add the useful mechanisms of AlphaEvolve, Voyager, and PromptBreeder to P2
without attempting a paper-code reproduction or changing the trusted
forecast evaluation boundary. P2 continues to evolve the complete Numerical
Dictionary. P3 remains the only task-local selection authority.

## Existing foundation

P2 already has content-addressed program/genome storage, LLM code proposals,
typed mutations and crossover, Host source validation, Train-only feedback,
Hyperband, constrained NSGA-II, MAP-Elites, and archive-driven parent draws.
The new work must reuse these components.

## Added mechanisms

### AlphaEvolve-style program context

The Host supplies a bounded set of verified parent/reusable program records to
the proposer. Each record binds member identity, applicability cells, source
SHA, and exact source text. The proposer may return edited source only through
the existing local-source response channel; the Host derives policy identity,
validates, executes, and scores it exactly as before.

### Voyager-style curriculum and skill reuse

The Host derives a deterministic curriculum from declared morphology cells,
archive visit counts, and sanitized Train failure categories. Unoccupied cells
come first, then least-visited and failure-matched cells. A reusable-method
view contains only Host-validated methods backed by feasible Train archive
entries. This view is derived from immutable archive/store artifacts and is
not a second mutable source of truth.

### PromptBreeder-style prompt evolution

The archive is also the bounded prompt population: every retained genome owns
its proposer-prompt lineage. Sampling a genome restores its own prompt instead
of overwriting it with one global prompt. `policy_tune` creates prompt children;
Host Train feedback updates only the sampled lineage. The existing prompt
template is the task prompt. Add one bounded mutation-prompt artifact that
governs how task-prompt variants are proposed; its lineage and Host-only Train
credit are committed in proposal context and durable state. No proposer may
grant itself credit or expand operator/response authority.

## One P2 generation

1. MAP-Elites/NSGA-II archive samples a parent genome.
2. Host derives curriculum targets and retrieves verified reusable programs.
3. The sampled task prompt and mutation prompt form the proposer context.
4. The LLM edits program code or proposes a bounded prompt child.
5. The fixed Host validates, hindcasts, scores, and inserts feasible entries.
6. Host-only Train outcomes update operator and prompt-lineage credit.
7. P2 seals the complete executable Dictionary for P3.

## Fixed boundaries

- Metrics, verifier, Train/Dev split, runtime fingerprints, artifact validator,
  budget authority, and promotion Host remain fixed.
- Curriculum and prompt credit use Train-only sanitized artifacts. Dev/Public
  values, forecasts, labels, raw objective vectors, paths, and callbacks never
  cross the proposer boundary.
- Program bytes are accepted only when their SHA matches and the existing Host
  parser/sandbox accepts them.
- All context collections are bounded, canonical, deterministic, and replayable.
- P2 never creates per-task shortlists or final ensembles.

## Compatibility and scope

Legacy sealed runs remain readable. New runs commit the new proposer-context
mode. This is a mechanism-level integration: it does not reproduce Minecraft,
Gemini model ensembles, or the exact paper training schedules.
