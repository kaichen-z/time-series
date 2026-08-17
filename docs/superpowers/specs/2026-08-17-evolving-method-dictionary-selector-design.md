# Evolving Forecast-Method Dictionary and Selector Design

## Status

This specification extends the current numerical dictionary-curation harness. The current
implementation starts from 41 fixed statistical method definitions and may generate or revise an
implementation for each definition. It does not systematically collect forecasting methods,
change method definitions, add or remove methods, discover new methods, learn a deployable
dictionary selector, or evolve the Numerical Agent's final selection policy.

This design makes both the method dictionary and its selector first-class evolving artifacts.

## Goal

Build a provenance-grounded Numerical Agent that:

1. systematically collects statistical, time-series foundation-model (TSFM), and combined
   forecasting methods from authoritative sources;
2. implements and validates those methods on numerical time-series tasks;
3. evolves method definitions and the method set using Train failures;
4. discovers new executable methods when the existing dictionary has a coverage gap;
5. evolves a label-free selector that chooses and combines methods at inference time; and
6. accepts a new dictionary-selector pair only when frozen end-to-end Dev forecasting improves.

The 41 existing methods become a seed catalog, not the final dictionary.

## Approaches Considered

### Fixed manually curated dictionary

A human-maintained list is simple and auditable, but method coverage remains arbitrary and the
system cannot respond to observed forecasting failures. It is retained as a baseline only.

### Unrestricted LLM web collection and mutation

An LLM can search broadly and invent methods quickly, but unrestricted collection creates
duplicate names, unsupported claims, unverifiable definitions, and potential benchmark-specific
overfitting. It is not acceptable as the authoritative dictionary-building process.

### Provenance-first collection plus coupled dictionary-selector evolution

This is the selected design. A systematic source registry creates a versioned raw catalog. Trusted
validation separates sourced method definitions from executable implementations. Method and
selector evolution then operate under explicit mutation, label, and acceptance boundaries.

## System Overview

```text
Authoritative papers, textbooks, official documentation, model cards, repositories
                                  |
                                  v
                    Phase 0: Systematic Collection
                                  |
                    Raw Method Registry + Sources
                                  |
                     Verify, normalize, deduplicate
                                  |
                     Verified Seed Dictionary D0
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
   Method Implementation/Evolution          Selector Policy Evolution
              |                                       |
              +-------------------+-------------------+
                                  v
                    Numerical Policy (Dg, Sg)
                                  |
                      Frozen label-free inference
                                  |
                         Trusted Train scoring
                                  |
       coverage gaps / selection gaps / redundancy and cost gaps
                                  |
                         Read-only Dev gate
                                  |
                     Accepted Numerical Policy
```

The runtime Numerical Agent is the accepted pair `(method_dictionary, selector_policy)`. Method
evolution and selector evolution are separate mutation surfaces, but they share one end-to-end
acceptance criterion.

## Phase 0: Systematic Method Collection

### Source coverage

Collection covers four source tiers:

1. textbooks and monographs for established statistical definitions and assumptions;
2. primary papers for statistical, machine-learning, TSFM, selection, and combination methods;
3. official documentation, model cards, and official repositories for executable interfaces,
   checkpoints, dependencies, licenses, and availability; and
4. surveys and forecasting benchmarks for taxonomy coverage and omission audits.

Survey and benchmark sources identify candidates but do not replace a primary definition source.
Unofficial websites may provide discovery leads, but they cannot independently verify a method.

### Coverage taxonomy

The collection query matrix spans:

- baselines, local level, trend, damped trend, and exponential smoothing;
- autoregressive, moving-average, ARIMA-family, structural, and state-space methods;
- seasonal, spectral, decomposition, intermittent-demand, and count methods;
- robust forecasting, anomaly handling, change-point, regime-switching, and analogue methods;
- regression, tree, kernel, neural, ensemble, reconciliation, calibration, and probabilistic methods;
- zero-shot and fine-tuned TSFMs, including their released checkpoint variants when inference
  behavior or availability differs materially; and
- statistical-plus-TSFM selection, residual correction, ensembling, and fallback policies.

Collection is organized by taxonomy cell and source tier rather than one unrestricted search
query. Every collection run records its query, retrieval date, and reviewed sources.

### Source requirements

A method enters the verified seed dictionary only when:

- it has at least one authoritative definition source;
- its name, definition, assumptions, and cited source agree;
- its implementation or model availability is independently recorded;
- its aliases have been checked against existing canonical methods; and
- its family and input/output requirements are known.

Entries failing these checks remain in the raw registry with `verification_status = "unverified"`
and cannot be selected by the Numerical Agent.

### Deduplication and identity

Each conceptual method receives an immutable `method_uid`. Names and definitions are versioned
fields and may evolve without breaking lineage. Deduplication uses algorithmic identity rather
than string similarity alone:

- aliases map to one UID;
- automatic model-selection wrappers remain distinct from the underlying model;
- materially different TSFM checkpoints or inference contracts may be separate executable
  variants under one conceptual method; and
- composed or specialized descendants retain their parent UIDs.

### Collection stopping rule

The project does not claim literal coverage of every publication on the internet. A collection
version is considered saturated only when:

1. all taxonomy cells have been audited across the required source tiers;
2. three consecutive collection batches each add fewer than two percent new non-duplicate
   canonical methods;
3. unresolved candidates and rejected duplicates are reported; and
4. a coverage report is generated with the dictionary snapshot.

Later source refreshes create a new raw-registry version; they never silently change a frozen
experiment dictionary.

Method count is not a stopping condition and the dataset has no configured upper limit. A release
may contain 100, 200, 300, or more verified records when the systematic search finds distinct,
relevant methods. Collection continues past round-number milestones until the coverage and
saturation rules pass; provenance and deduplication requirements never relax to increase count.

## Data Contracts

### Source record

```json
{
  "source_id": "source_000123",
  "title": "Authoritative source title",
  "authors": ["Author"],
  "year": 2025,
  "source_type": "paper|textbook|official_docs|model_card|official_repo|survey|benchmark",
  "url": "https://example.org/authoritative-source",
  "retrieved_at": "2026-08-17",
  "primary": true,
  "review_status": "verified"
}
```

### Evolvable method definition

```json
{
  "method_uid": "method_000123",
  "definition_version": 2,
  "canonical_name": "Damped Holt Trend",
  "aliases": ["Damped Trend Method"],
  "family": "statistical",
  "category": "exponential_smoothing",
  "description": "A trend method whose extrapolated trend decays over the horizon.",
  "assumptions": ["A trend is present", "Long-range linear extrapolation is implausible"],
  "failure_conditions": ["Untreated strong seasonality", "Abrupt regime shift"],
  "applicability": {
    "minimum_history": 20,
    "frequencies": ["any"],
    "supports_univariate": true,
    "supports_covariates": false,
    "supports_probabilistic_output": false
  },
  "hyperparameters": ["damping_factor"],
  "source_ids": ["source_000123"],
  "lineage": {
    "operation": "rewrite_definition",
    "parent_method_uids": ["method_000123"],
    "reason": "Clarified the observed failure boundary from Train evidence"
  },
  "implementation_spec": {},
  "verification_status": "verified",
  "implementation_status": "unimplemented",
  "cost": 1.0
}
```

TSFM records additionally contain checkpoint identifier, release version, context-length and
horizon limits, zero-shot/fine-tuning mode, covariate support, sampling interface, device and
memory requirements, license, and code/weight availability.

### Selector policy

```json
{
  "selector_id": "selector_v003",
  "parent_selector_id": "selector_v002",
  "feature_schema_version": 1,
  "enabled_features": [
    "trend_strength",
    "seasonality_strength",
    "volatility",
    "intermittency",
    "outlier_rate",
    "change_point_score",
    "history_length",
    "frequency",
    "horizon",
    "hindcast_scores"
  ],
  "top_k": 3,
  "selection_program": {},
  "aggregation": "weighted_mean",
  "fallback_method_uid": "method_naive_last",
  "cost_budget": 10.0,
  "changelog": "Improved unstable-regime fallback and reduced redundant seasonal methods"
}
```

At inference time the selector can see only history-derived features, frequency, horizon, method
metadata, availability, cost, and history-only hindcast diagnostics. It cannot see future labels,
documents, GT evidence, or Retrieval/Decision Agent outputs.

## Method Dictionary Evolution

The Method Evolver receives the current dictionary and sanitized Train diagnostics. It can propose
the following structured operations:

- `rewrite_definition`: improve name, definition, assumptions, applicability, or failure
  conditions while preserving the UID;
- `repair_implementation`: revise executable code or adapter configuration;
- `parameterize`: expose or revise a method's parameter-search space;
- `specialize`: create a child for a coherent numerical regime;
- `generalize`: combine compatible specialized descendants;
- `split`: replace an over-broad definition with multiple regime-specific descendants;
- `merge`: consolidate duplicate or empirically redundant methods;
- `compose`: create an executable ensemble or residual-correction child;
- `discover`: propose a genuinely new falsifiable method for an uncovered failure cluster; and
- `archive`: remove an unsafe, invalid, unavailable, or consistently dominated method from the
  active set without deleting its lineage.

Each mutation must include a hypothesis, expected applicable regime, explicit failure condition,
source provenance when it changes scientific claims, lineage, and an executable implementation or
registered runtime. LLM text alone cannot create an active method.

### New-method discovery

Discovery is triggered by a coverage gap: a coherent Train cluster for which all valid active
methods exceed the configured error threshold. The Gap Miner gives the Method Evolver sanitized
numerical summaries, residual-shape descriptors, and method-level aggregate failures. It does not
provide raw future trajectories or Dev/Test feedback.

A discovered method must pass:

1. schema and provenance validation;
2. source-code or registered-runtime safety validation;
3. sandbox execution and shape/finite-value checks;
4. history-only hindcasting;
5. out-of-fold Train evaluation; and
6. the frozen end-to-end Dev acceptance gate as part of a numerical-policy child.

Unsupported scientific claims are prohibited. A novel composition derived from existing verified
methods may cite its parent methods and identify itself as a system-generated composition rather
than claiming a new published algorithm.

## Selector Evolution

The Selector Evolver changes how the Numerical Agent chooses and combines methods. Allowed
mutations include:

- add or remove history-derived features;
- change applicability gates and ranking rules;
- change Top-K, cost budget, or deterministic fallback;
- change ensemble weights or aggregation;
- add uncertainty or abstention thresholds; and
- add exploration challengers during Train while keeping frozen inference deterministic.

The selector outputs selected method UIDs, weights, confidence, and a fallback. Host Python
validates IDs, availability, cost, weight normalization, and budget. Invalid output uses the frozen
fallback and is counted as a selector failure.

## Coupled Evolution Loop

The loop alternates mutations so gains can be attributed before permitting joint mutations:

```text
Parent Numerical Policy Pg = (Dictionary Dg, Selector Sg)
  -> freeze Sg; propose dictionary children
  -> freeze each dictionary child; execute and screen on Train
  -> freeze best dictionary; propose selector children
  -> execute and screen on Train
  -> optionally propose one joint child after the separate ablations
  -> fully evaluate eligible children on Train
  -> freeze predictions
  -> evaluate only eligible finalists on read-only Dev
  -> accept the best child only if end-to-end deployable Dev forecasting improves
  -> persist Pg+1 or retain Pg
```

Trusted scoring produces three distinct diagnoses:

- `coverage_gap`: no active method performs adequately;
- `selection_gap`: an adequate method exists but the selector did not choose it; and
- `redundancy_or_cost_gap`: extra methods add cost without improving coverage or selected output.

Coverage gaps train the Method Evolver. Selection gaps train the Selector Evolver. Redundancy and
cost gaps can mutate either artifact but cannot override the end-to-end Dev gate.

## Reward and Acceptance

The primary acceptance metric is the mean forecasting error of the selector's actually deployed
prediction. The first implementation uses mean final sMAPE to remain compatible with the current
Numerical Agent harness; official Dr-CiK metrics are computed later by the frozen benchmark
evaluation path.

Additional diagnostics are:

- MAE and median sMAPE;
- per-task win/tie/loss against the parent;
- selection regret: selected error minus the best available active-method error;
- Top-K oracle coverage, used only as a diagnostic;
- invalid, unsafe, and unavailable execution rates;
- active method count, redundancy, inference cost, and worst-case regression.

The existing best-of-all-methods dictionary score is renamed `oracle_coverage_error`. It cannot
accept a child because it uses resolved labels to choose a different best method for each task and
is not deployable.

Acceptance is lexicographic:

1. all label, safety, schema, and reproducibility gates pass;
2. the child's deployable mean Dev sMAPE improves by the configured margin;
3. invalid/unsafe rates do not regress;
4. catastrophic-regression and inference-cost limits pass; and
5. ties prefer the smaller and cheaper active dictionary.

## Data and Leakage Boundary

The canonical manifest remains `splits/drcik_public_80_20_99_v1.json`:

- 80 public Train tasks for mutation feedback;
- 20 public Dev tasks for read-only Parent/Child acceptance;
- 99 public Test tasks for one-time frozen evaluation; and
- 80 hidden tasks excluded from local training and scoring.

Within the 80 Train tasks, four-fold cross-fitting generates out-of-fold method and selector
diagnostics. This reduces co-adaptation without permanently shrinking the already small Train set.
Dev returns only frozen aggregate acceptance metrics; it never produces method definitions,
selector changes, Skills, prompts, or detailed failure feedback. Public Test is accessed only after
the full numerical policy is frozen.

Phase 0 collection is independent of Dr-CiK task labels and documents. General forecasting methods
may be collected from current sources, but collection prompts cannot contain Dr-CiK future values,
GT evidence, document roles, or task-specific future context.

## Integration with the Three-Agent Harness

The accepted Numerical Policy replaces the current hard-coded statistical skill list only in the
new dictionary-backed condition. Existing baselines remain unchanged.

```text
History numbers
  -> Numerical Agent loads (working_dictionary, best_selector)
  -> selects Top-K executable methods
  -> returns forecasts + assumptions + failure conditions + hindcast diagnostics
  -> Retrieval Agent receives numerical hypotheses and documents
  -> retrieves evidence that supports or distinguishes those hypotheses
  -> Decision Agent chooses or combines frozen numerical candidates using verified evidence
  -> final forecast
```

The Numerical Agent never receives documents. The Retrieval Agent should receive the Numerical
Agent's assumptions because they identify what evidence can distinguish competing numerical
futures. Whole-harness co-evolution remains a later layer and cannot rewrite the trusted scorer,
splits, or label firewall.

## Package Changes

```text
numerical_agent/
  collection/
    contracts.py          Source, query, candidate, and verification schemas
    registry.py           Raw source and candidate persistence
    normalization.py      Alias and canonical-identity handling
    verification.py       Provenance and source-tier gates
    saturation.py         Taxonomy coverage and stopping report
  selectors/
    contracts.py          SelectorPolicy and selection result schemas
    features.py           Label-free numerical feature extraction
    runtime.py            Deterministic policy execution and fallback
    mutation.py           Selector mutation provider interface
  adapters/
    dictionary_curation.py  Retained implementation-curation baseline
    numerical_policy.py     Coupled dictionary-selector adapter
  gap_mining.py           Coverage, selection, redundancy, and cost diagnoses
  method_mutation.py      Structured method-set mutation operations
  numerical_policy.py     Versioned (dictionary, selector) artifact
  main.py                 collect, verify, curate, evolve, and frozen-evaluate commands
```

The generic `common/evolution_core` remains responsible for Parent/Child execution, scoring,
acceptance, checkpointing, and resume. The new numerical-policy adapter supplies the domain-specific
artifact, mutation, execution, evaluation, and acceptance logic.

## CLI Surface

```bash
# Import and normalize externally collected source/method records.
python -m numerical_agent collect \
  --source-manifest inputs/forecast_method_sources.jsonl \
  --candidate-manifest inputs/forecast_method_candidates.jsonl \
  --output-dir runs/method_collection/v000

# Verify provenance and publish the immutable experiment seed.
python -m numerical_agent verify-dictionary \
  --raw-registry runs/method_collection/v000/raw_method_registry.json \
  --output numerical_agent/dictionaries/method_dictionary_verified_v000.json

# Preserve the current fixed-definition implementation curation as a baseline.
python -m numerical_agent curate \
  --experiment-config runs/dictionary_curation/experiment.json \
  --base-methods numerical_agent/dictionaries/method_dictionary_verified_v000.json \
  --provider llm \
  --output-dir runs/dictionary_curation/v000

# Jointly evolve method definitions/set and the deployable selector.
python -m numerical_agent evolve-policy \
  --experiment-config runs/numerical_policy/experiment.json \
  --base-dictionary runs/dictionary_curation/v000/working_dictionary.json \
  --base-selector numerical_agent/selectors/selector_v000.json \
  --evolution-mode alternating \
  --output-dir runs/numerical_policy/v001
```

Automated internet search is not required for deterministic tests. The production collector may
use an injected research provider, while tests use frozen source and candidate manifests.

## Artifacts

Collection writes:

- `source_registry.jsonl`;
- `raw_method_registry.json`;
- `method_dictionary_verified_vNNN.json`;
- `duplicate_and_rejection_report.json`; and
- `collection_coverage_report.md`.

Evolution writes:

- `dictionary_genome_vNNN.json`;
- `selector_policy_vNNN.json`;
- `best_numerical_policy.json` containing hashes of both artifacts;
- `method_lineage.jsonl`;
- `trusted_failure_matrix.jsonl` with restricted access to mutation summaries;
- `train_evaluation.json`, `dev_evaluation.json`, and `evolution_trace.json`; and
- frozen Public-Test predictions only through an explicit evaluation command.

## Failure Handling

- Missing or inaccessible sources keep candidates unverified.
- Unavailable TSFM dependencies create structured `unavailable` results.
- Generated code is subject to the existing sandbox and static safety checks.
- Duplicate or cyclic method lineage is rejected.
- An invalid selector output invokes the deterministic fallback and records a failure.
- Transient LLM or network errors are retried and recorded; they are not converted into
  forecasting failures or unchanged competitive children.
- A failed collection refresh cannot modify the latest verified dictionary.
- Checkpoints allow collection and evolution to resume without repeating completed evaluations.

## Testing

Tests cover:

- source and method-card schema validation;
- primary-source and verification requirements;
- alias mapping, duplicate detection, and immutable UIDs;
- TSFM metadata requirements and unavailable-provider handling;
- every structured method-set mutation and lineage validation;
- new-method activation only after executable validation;
- selector feature extraction without labels or documents;
- selector ID, weight, budget, and fallback validation;
- explicit separation of oracle coverage and deployed selection score;
- coverage-gap versus selection-gap attribution;
- alternating and joint mutation boundaries;
- four-fold Train cross-fitting and read-only Dev behavior;
- rejection of Test, future-value, GT-evidence, role, and subtype leakage;
- checkpoint/resume and transient network handling;
- CLI smoke tests with deterministic fixtures; and
- the complete existing test suite.

## Delivery Sequence

1. Add collection contracts, registry import, verification, deduplication, and coverage reporting.
2. Migrate the 41 current methods into the richer schema as seed candidates without inventing
   missing sources.
3. Extend dictionary mutations to definitions and the active method set.
4. Add selector contracts, label-free features, deterministic runtime, and selector evaluation.
5. Add gap mining and the alternating dictionary-selector evolution adapter.
6. Replace oracle-only dictionary acceptance with deployable numerical-policy acceptance.
7. Add frozen integration into the three-agent harness while retaining existing baselines.
8. Run deterministic tests, a small Dr-CiK smoke experiment, then the frozen 80/20/99 protocol.

This sequence makes Phase 0 auditable before any self-evolution result is interpreted and keeps
method quality, method coverage, and selection quality separately measurable.
