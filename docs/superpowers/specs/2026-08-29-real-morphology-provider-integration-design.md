# Real Morphology Provider Integration Design

**Date:** 2026-08-29

**Status:** Approved in chat; written design pending final user review

**Scope:** Opt-in integration of the existing history-only `MorphologyReasoner` with the
two-stage Retrieval runtime

## Objective

Replace the two-stage Retrieval CLI's conservative empty Morphology placeholder with an optional
real provider that converts grounded, history-only numerical analysis into sanitized Retrieval
assumptions. Existing commands and frozen artifacts must retain their current behavior unless the
operator explicitly selects the real provider.

The completed path is:

```text
executed Coding candidates
  -> safe candidate projection (name + family only)
  -> history-only MorphologyReasoner
  -> grounded ForecastAssumptions
  -> four-field RetrievalAssumptions
  -> provisional Decision named gaps
  -> optional Retrieval Round 2
  -> final Decision over already-executed candidates
```

## Non-Goals

- Do not reproduce KairosAgent's T-STAR corpus, SFT, GRPO, or gated TSFM cross-modal fusion.
- Do not train or change LLM, Statistical, or TSFM weights.
- Do not let Morphology generate forecast arrays, executable code, Retrieval evidence, or final
  candidate selections.
- Do not change the default single-pass or two-stage behavior of existing commands.
- Do not add Morphology content to Retrieval Round 1.
- Do not treat a successful fake-LLM integration test as an empirical forecasting result.

## Current State

The repository already contains the required components but does not connect them:

- `numerical_agent/evolution/morphology.py` implements a bounded history-only tool loop. Its
  `MorphologyReasoner.reason()` method consumes historical values, frequency, horizon, active
  candidate names, and candidate families, then returns a grounded `MorphologyCard`.
- `evolving_loop/morphology_adapter.py` defines `MorphologyProvider` and a preliminary adapter,
  but the existing provider accepts only a `ContextTask` and expects a wrapped `.run(task)` method.
  That contract cannot supply the active candidates required by `MorphologyReasoner.reason()`.
- `EvolvingForecastHarness._run_two_stage()` already computes Morphology assumptions, sends them
  to provisional Decision, and exposes only named gaps plus sanitized assumptions to Retrieval
  Round 2.
- `evolving_loop/cli.py` always constructs `_ConservativeMorphologyProvider`, whose
  `assumptions()` method returns an empty tuple. The `and assumptions` host gate therefore skips
  Round 2 in the formal CLI path.

## Approaches Considered

### Selected: Independent Provider Adapter

Extend the provider boundary to accept a safe candidate projection and adapt the existing
`MorphologyReasoner` behind that boundary. This preserves Morphology as an independently
replaceable component, keeps Retrieval Round 1 blind, and lets the host enforce the exact data
shape before any LLM call.

### Rejected: Put Morphology Inside Coding Agent

Adding `MorphologyCard` to `CodingEvolutionResult` would give the reasoner direct candidate access,
but it would couple Numerical hypothesis generation to the Retrieval topology and make independent
coordinate evolution and failure attribution harder.

### Rejected: Let Retrieval Invoke Numerical Reasoning

Calling the reasoner from `TwoStageRetrievalAgent` would be mechanically simple, but it would mix
document retrieval with numerical analysis and weaken the assumption-blind Round-1 boundary.

## Public Interfaces

### Safe Candidate Projection

Add a frozen host-owned record in `evolving_loop/morphology_adapter.py`:

```python
@dataclass(frozen=True)
class MorphologyCandidate:
    name: str
    family: str
```

Each record rejects empty values and non-identifier names. The host collection builder
de-duplicates equal name/family pairs and rejects conflicting family assignments for the same
name. The record deliberately excludes forecasts, hindcast metrics, source code, prompts,
documents, labels, and candidate assumptions.

### Provider Protocol

Change the runtime-checkable protocol to:

```python
class MorphologyProvider(Protocol):
    def assumptions(
        self,
        task: ContextTask,
        candidates: tuple[MorphologyCandidate, ...],
    ) -> tuple[RetrievalAssumption, ...]: ...
```

Both the empty provider and the real adapter implement this exact signature. The empty provider
ignores both arguments and returns `()`.

### Real Adapter

`MorphologyAdapter` owns an injected object with a callable `.reason(...)` method. It calls that
method with only:

- `task.numeric_view().history_values`;
- `task.numeric_view().frequency`;
- `task.numeric_view().prediction_length`;
- candidate `name` values as `active_names`;
- the candidate `name -> family` mapping.

The adapter accepts the returned card only when it exposes a sequence of assumptions. Each item is
projected through `RetrievalAssumption.from_payload()` using exactly:

```text
assumption_id
kind
claim
failure_condition
```

Candidate names, confidence, tool observations, supporting signals, tool-call IDs, descriptions,
and all other card fields stop at the adapter. Duplicate assumption IDs or an invalid four-field
contract fail closed with `RetrievalContractError`.

## Harness Integration

After Coding has executed and validated candidates, `_run_two_stage()` constructs a deterministic
`tuple[MorphologyCandidate, ...]` from each candidate's `program.name` and `program.source`.
Candidates are de-duplicated by name while preserving host order; conflicting duplicate families
are rejected.

The harness then calls:

```python
assumptions = self.morphology.assumptions(task, safe_candidates)
```

No candidate forecast, hindcast score, generated source, Coding prompt, or document is passed to
the provider. The remainder of the existing two-stage flow stays unchanged:

1. Retrieval Round 1 sees target metadata and documents only.
2. Provisional Decision sees executed candidates, verified Round-1 evidence, and the sanitized
   assumptions.
3. Round 2 runs only when the configured trigger, non-empty assumptions, and valid named gaps all
   permit it.
4. Round 2 receives the existing four-field assumption payload and never candidate details.
5. Final Decision selects only an already-executed host candidate.

## CLI And Compatibility

Add this explicit option to every unified CLI path that constructs the two-stage harness:

```text
--morphology-provider empty|reasoner
```

The default is `empty`. A centralized builder returns:

- `_ConservativeMorphologyProvider()` for `empty`;
- `MorphologyAdapter(MorphologyReasoner(llm))` for `reasoner`.

The real provider reuses the command's configured `LLMClient`, model, reasoning effort, timeout,
and subprocess environment. It does not create a second provider stack or read credentials.

Old commands, stored policies, accepted Retrieval releases, and frozen inference remain empty by
default. Enabling `reasoner` is an operator decision for a new experiment and cannot occur merely
because the new implementation is installed.

## Reproducibility And Fingerprints

Run summaries and harness fingerprints distinguish `conservative_empty_v1` from
`history_only_reasoner_v1`. The reasoner fingerprint covers these sources:

- `evolving_loop/morphology_adapter.py`;
- `numerical_agent/evolution/morphology.py`;
- `numerical_agent/evolution/analysis_skills_template.py`;
- `numerical_agent/evolution/assumptions.py`.

The existing LLM configuration fingerprint continues to bind backend, model, effort, and client
source. Retrieval releases remain Retrieval-owned artifacts and do not absorb Morphology source;
the outer run/harness manifest records the independently selected Morphology component.

## Error Handling

- `TransientLLMError` retains the harness's current retry/abort semantics and is not silently
  converted into a contract failure.
- Invalid JSON, unknown tools, invalid windows, ungrounded assumptions, duplicate IDs, unknown
  candidates, and malformed cards fail closed at the reasoner or adapter.
- Non-transient Morphology failures produce no assumptions, add the existing typed
  `morphology_provider_failed:<ExceptionType>` rejection marker, preserve verified Round-1
  evidence, skip Round 2, and continue to the safe final Decision.
- A Morphology failure never removes the Coding host candidate or changes its numeric forecast.

## Files And Ownership

- `evolving_loop/morphology_adapter.py`: safe candidate schema, provider protocol, real projection.
- `evolving_loop/harness.py`: safe candidate construction and provider invocation.
- `evolving_loop/cli.py`: opt-in flag, centralized provider builder, descriptors, and fingerprints.
- `tests/test_two_stage_retrieval.py`: provider boundary, Round-2 trigger, sanitization, and fallback.
- `tests/test_retrieval_runner.py` and/or the existing focused CLI suite: default/opt-in CLI
  construction and manifest fingerprinting.
- `README.md` or `docs/SELF_EVOLUTION_FRAMEWORK.md`: command and compatibility documentation.

No production behavior is added to `TwoStageRetrievalAgent`, `DecisionAgent`, the Retrieval
verifier, or the evaluator.

## TDD And Verification

Implementation follows one failing test per behavior:

1. The adapter receives only numeric task fields and safe candidate descriptors, then emits only
   the four Retrieval assumption fields.
2. The default empty provider preserves current CLI and Harness behavior.
3. The opt-in reasoner provider can cause a valid provisional Decision gap to trigger Round 2.
4. Invalid or ungrounded morphology fails closed and skips Round 2 without changing the numeric
   host forecast.
5. Round 1 remains assumption-blind and Round 2 contains no candidate names, forecasts, scores,
   code, labels, or tool observations.
6. Run summaries and fingerprints differ between empty and reasoner providers.

Focused verification runs the Morphology, two-stage Retrieval, Harness, runner, frozen-inference,
and verifier suites. Final verification runs the repository's complete pytest suite and
`git diff --check` without modifying existing user-owned artifacts.

## Acceptance Criteria

- `--morphology-provider reasoner` constructs and executes the existing history-only reasoner.
- `--morphology-provider empty` remains the default and is behavior-compatible with the current
  checkout.
- The provider receives no documents, future labels, forecasts, hindcast metrics, or generated
  source.
- Retrieval Round 1 receives no Morphology assumptions.
- Retrieval Round 2 receives only the four approved assumption fields and valid named gaps.
- Non-transient Morphology failures preserve the safe numerical fallback and verified Round 1.
- Experiment artifacts identify the selected provider and bind its implementation sources.
- Focused and full verification pass before the implementation is declared complete.
