# Real Co-evolution Flow Documentation Design

Date: 2026-09-06

## Objective

Replace the illustrative data in `docs/three-agent-coevolution-flow.html` with a provenance-aware account of the actual Method, Filter, Screening, Champion, Numerical, Retrieval, Decision, and Gate flows.

The page must never present invented values, reconstructed historical requests, or artifacts from unrelated runs as one connected execution. It must make a visible distinction between what the current source code guarantees and what a named run actually persisted.

## Truth model

Every prompt, input, output, metric, method name, mutation, and handoff shown in the interactive stepper must carry one of these truth classes:

1. `CURRENT CONTRACT`: exact content or schema derived from the checked-out source code. This includes complete system prompts, request builders, response schemas, parser rules, validation rules, Host transformations, and acceptance gates.
2. `RECORDED RUN`: unmodified content read from a named artifact that exists under `runs/`. The UI must show the run ID, artifact path, protocol date or schema version when available, and a content hash.
3. `NOT RECORDED`: an explicit absence marker. This is required when a run retained only a response, metric summary, hash, or derived release and did not retain the corresponding request or raw model exchange.

Illustrative payloads are prohibited inside the truth-aware stepper. Explanatory prose may describe a type of value, but it may not resemble a concrete run payload unless it is labeled outside the stepper as an illustration.

## Corrected Method Evolution account

The existing four-item list (`toto`, `seasonal_naive`, `linear_trend`, `croston`) must be removed. It was a manually selected teaching subset and has no upstream artifact or selection event.

The recorded bootstrap at `runs/method_evolution/v001/bootstrap_summary.json` reports 93 generated and 93 successful methods. The accompanying `methods.py` contains 93 method definitions. The recorded `gpt56_two_stage_20_20260822` generation contains 93 method reports and its post-mutation `methods.py` contains 89 method definitions. These counts and the complete method names may be displayed only with the corresponding run provenance.

The fabricated `train_method_reports` wrapper must be removed. The current source flow is:

```text
methods.py
  -> run_module(methods.py, frozen Train tasks)
  -> Outcome per method and task
  -> _report(...)
  -> MethodReport
  -> report_payload(...)
  -> generation_NNN_metrics.json["reports"]
  -> render_evolve_user(...)
  -> LLM proposal
  -> parse, validate, apply, execute, and commit by the Host
```

The current `MethodReport` contract must be shown using the exact source-defined fields. A recorded old-schema report must retain its historical fields rather than being silently upgraded to current sMAE/sRMSE fields.

The current batch-evolution user message must be shown in its real Markdown form: a `# Measured results` JSON section followed by the complete current `methods.py` source under `# Current module`. Selector/mutator strategies must appear as separate variants because their input contracts differ.

## Interactive step structure

The stepper retains nine logical steps:

1. Method Evolution
2. Dictionary Filter
3. Task-conditioned Screening
4. Champion Proposal
5. Numerical Execution
6. Retrieval Round 1
7. Retrieval Round 2
8. Decision
9. Acceptance Gate

Each selected step renders these disclosures in order:

- Provenance and truth class
- Complete system prompt, when the step invokes an LLM
- Complete user request or current request-construction contract
- Complete raw recorded response, when persisted
- Parsed response object and schema
- Host-owned validation and transformation
- Evolution actions or mutations
- Accepted/rejected/no-update result
- Persisted artifacts
- Exact downstream projection

Long content is collapsed by default, but the DOM contains the complete value and provides expand and copy controls. Labels such as “excerpt”, ellipses that imply omitted fields, and silent truncation are not allowed in full-content panels.

## Contract and recorded-run modes

The stepper has two explicit modes.

### Current contract

This mode documents the checked-out implementation. Content comes from source constants and request builders, including:

- Method: `EVOLVE_SYSTEM`, selector/mutator prompts, `render_evolve_user`, `render_select_user`, and `render_mutate_user`.
- Filter: filter system prompt, request fields assembled by `evolve_filter_once`, evidence aggregation, parser, and publication rules.
- Screening: screening system prompt, request fields assembled by `evolve_screening_once`, policy schema, Train/Dev gate, and publication rules.
- Champion: `CHAMPION_PROPOSAL_SYSTEM`, `_proposal_payload`, recipe schema, retry and validation logic, release construction, and lifecycle gates.
- Numerical: bundle binding, candidate execution, hindcast metrics, fallback handling, and sanitized downstream candidate view.
- Retrieval: complete round-one and round-two prompts, `retrieval_view`, assumption and gap inputs, verifier schema, citation checks, and evidence-card merge.
- Decision: complete decision prompt, candidate/evidence request, response schema, override checks, fallback behavior, and more-retrieval request contract.
- Gate: exact Train/Dev evaluation fields, failure conditions, no-update semantics, stage evidence, checkpoint, and final bundle publication.

Static HTML must not duplicate mutable source strings by hand without a verification mechanism. A test helper will extract or import source-backed constants/builders and compare them to the embedded documentation payload. If safe direct import is impractical, the test compares canonical hashes and required complete text sentinels.

### Recorded run

Recorded mode uses only existing files and names the source artifact beside every value. It may combine different runs for coverage only when each panel remains independently labeled; it must state that the collection is not one end-to-end execution.

Initial recorded sources are:

- Method bootstrap and generation: `runs/method_evolution/v001/` and `runs/method_evolution/gpt56_two_stage_20_20260822/`.
- Screening result: `runs/task_conditioned_screening/conditional_v13_failure_burden_train_only_80_20_20260825/generation_001_screening_result.json`.
- Champion response and release: the available proposer cache under `runs/champion_evolution/gpt56sol_medium_dictionary_80_20_20260902/` and the release/lifecycle artifacts under `gpt56sol_high_toto_balanced_v3_20260906_g3_r2/`.
- Package evidence and final bundle: `runs/package_coevolution/gpt56luna_medium_toto_balanced_v3_g2_20260906_r3/`.
- Package LLM response cache and inference records: the matching `runs/package_coevolution_authority/gpt56luna_medium_toto_balanced_v3_g2_20260906_r3/` directory.
- Retrieval release prompts and manifests: the named release artifacts under `runs/retrieval_releases/` used by the package manifest.

An artifact may not be called the input to another artifact unless a recorded hash, manifest binding, stage claim, or deterministic source transformation establishes that edge.

## Handoff semantics

“Output becomes next input” is represented as a typed handoff, not unconditional byte-for-byte copying.

For every edge, the page displays:

- source artifact and source field;
- Host projection, sanitization, verification, or serialization function;
- downstream field and destination step;
- fields deliberately withheld;
- fingerprints or hashes that bind the handoff.

This matters especially at the Numerical-to-Retrieval and Retrieval-to-Decision boundaries. Candidate identities, scores, forecasts, future labels, and task identity are not universally forwarded. Retrieval Round 1 is deliberately assumption-blind in the strict package path, and only verified evidence may influence Decision. The page must preserve these safety boundaries rather than visually suggesting that every preceding output is copied wholesale.

## Missing evidence behavior

When a historical prompt or request was not persisted, the page displays `NOT RECORDED` and explains exactly what remains provable. For example, the Codex cache stores a normalized JSON response under a hash derived from model, reasoning effort, and prompt, but does not store the prompt text itself. The page may show the current prompt constructor in contract mode and the historical response in recorded mode, but it must not claim they formed the same call unless the historical source revision can reproduce and verify the cache key.

Old and current schemas remain visibly separate. No migration, inferred field, or reconstructed object may be displayed as raw historical content.

## Data packaging

The HTML remains a self-contained file. A deterministic documentation-data generator or test fixture will produce canonical JSON payloads from approved source files and selected artifacts. Embedded payloads include provenance metadata and SHA-256 hashes.

Large files such as `methods.py`, multi-megabyte Champion build evidence, task rows, and full prompts are represented as complete collapsible text only when their size remains practical for the page. For very large artifacts, the page lists the complete top-level structure and supplies a local clickable artifact path plus its byte size and hash; it must say that the bytes live in the artifact rather than pretending a partial rendering is complete. The nine-step input/output panels themselves must contain every field actually passed across that interface.

## Error and acceptance semantics

The page distinguishes:

- malformed or forbidden LLM output;
- invalid schema or runtime contract;
- execution crash, invalid forecast, and not-applicable outcome;
- unverifiable citation or illegal Decision override;
- rejected child with exact Parent retained;
- legal `no_update` continuation;
- missing artifact or fingerprint mismatch that fails closed;
- public-test access violations;
- accepted release publication.

Reject is not synonymous with pipeline failure. A valid rejected child leaves the frozen Parent unchanged and may continue to the next scheduled coordinate. Integrity, data-boundary, or runtime-authority failures stop before downstream publication.

## Testing and verification

Tests will verify at minimum:

- the four invented methods and `train_method_reports` no longer occur;
- all nine steps remain navigable;
- every step exposes provenance, truth class, prompt/input/output, Host processing, evolve action, result, artifact, and handoff sections as applicable;
- current prompt text and response fields match source-controlled contracts;
- recorded values match the selected artifact files and hashes;
- absent transcripts render `NOT RECORDED` instead of reconstructed content;
- historical schemas are not relabeled as current schemas;
- no edge claims direct handoff without an explicit projection or binding;
- desktop and repository copies are byte-identical;
- the page works without network access and remains usable at desktop and narrow viewport widths.

Verification includes the focused HTML test suite, a scan for prohibited illustrative fields, embedded-data hash checks, and browser visual inspection when the browser runtime is available. If visual tooling is unavailable, that limitation is reported rather than treated as completed visual QA.

## Files in scope

- `docs/three-agent-coevolution-flow.html`
- `tests/test_coevolution_flow_html.py`
- a small deterministic extraction helper or checked fixture only if needed to keep source/artifact synchronization testable
- `/Users/yyoraa/Desktop/Time-Series-CoEvolution-Flow.html` as the final byte-identical delivery copy

No production runner, evaluation logic, model code, or historical artifact is modified by this documentation correction.
