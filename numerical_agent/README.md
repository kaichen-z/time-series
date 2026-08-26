# Parameterized Numerical Self-Evolution Harness

This package composes a generic Parent/Child Self-Evolution Core with a dictionary-curation
adapter. It ships an auditable forecasting-method dataset and reviewed runtime adapters for a
subset of foundation models. Catalog definitions and executable providers remain deliberately
separate: a method is counted as executable only when its implementation/runtime is enabled.

## Forecast Method Dataset v001

The publishable release is
[`datasets/forecast_method_dataset_v001.json`](datasets/forecast_method_dataset_v001.json). It
contains **166 canonical methods** grounded in **115 reviewed sources** through the collection
cutoff of **2026-08-17**. Of these sources, 105 are marked as primary definition sources and the
remaining 10 are official implementation repositories:

| Family | Methods | Scope |
|---|---:|---|
| Statistical | 111 | Classical, machine-learning, neural, multivariate, probabilistic, robust, calibration, reconciliation, and validation-oriented forecasting mechanisms. |
| Foundation | 31 | Distinct TSFM architectures or materially different releases with checkpoint/API metadata. |
| Combined | 24 | Ensembles, selectors, residual corrections, and fallbacks with explicit parent-method lineage. |

The dataset has three layers:

1. `collection/catalog_v001.py` is the deterministic, reviewed catalog source.
2. `datasets/source_registry_v001.jsonl` and `datasets/method_candidates_v001.jsonl` are generated
   review manifests.
3. `datasets/forecast_method_dataset_v001.json` is the release artifact produced only after
   provenance, taxonomy, duplicate-resolution, and scoped-saturation gates pass.

Every method card records assumptions, failure conditions, applicability, hyperparameters,
source IDs, verification status, and implementation availability. Foundation cards additionally
record checkpoint/API, release, context and prediction limits, inference mode, uncertainty and
covariate support, device requirements, license, and weight/code availability. Combined cards
identify at least two parent method UIDs.

The catalog is broad, not timelessly exhaustive. “Saturated” means that the final three
independent review batches each added fewer than 2% of the 166-method base under the documented
canonicalization rule. The evidence is in
[`datasets/collection_journal_v001.json`](datasets/collection_journal_v001.json), and the resulting
machine-readable audit is
[`datasets/collection_audit_v001.json`](datasets/collection_audit_v001.json).

Rebuild and validate the release with one command:

```bash
scripts/build_method_dataset.sh
```

The command deterministically regenerates the manifests, runs the publication gates, writes the
release and SHA-256 sidecar, and executes the collection tests. It does not download models or run
forecasting experiments.

The Dictionary Curation Adapter converts the release into the executable `ToolDictionary`
contract. Phase 1 imports the 111 statistical cards by default. Foundation and combined cards
remain in the research catalog until a caller explicitly enables their families and registers
honest model/dependency runtimes; they are never approximated by a similarly named NumPy method.
A collaborator must still provide an approved `MethodImplementer`, one or more `MethodRuntime`
providers, Train/Dev tasks, trusted labels, and a metric. The LLM implements or repairs code;
trusted Python evaluation assigns `accepted`, `specialized`, `quarantined`, `unavailable`, or
`discarded` status. Held-out Dev performance decides whether a Child dictionary replaces its
Parent.

## Git-based 103-candidate evolution portfolio

The method-repository loop now supports a separate, explicitly enabled portfolio:

| Executable family | Count | Evolved artifact |
|---|---:|---|
| Python statistical forecasters | 93 | `methods.py` |
| Flagship TSFMs | 5 | invocation settings in `policies.py` |
| TSFM/statistical Combined policies | 5 | blend/route settings in `policies.py` |
| **Total** | **103** | small auditable Git repository |

The five TSFMs are TimesFM 2.5, Moirai 2.0, Toto 2.0, Chronos-Bolt, and Granite TTM R2. Their
checkpoint, adapter, license, and model identity are immutable. Evolution may change only
history-only applicability, context window, reversible preprocessing, and bounded output
shrinkage. The five Combined policies bind one TSFM parent and one statistical parent; evolution
may change their weight or history-only routing rule but cannot substitute either parent.

Every Python forecaster can call the reviewed history-only functions in `skills.py`, including
periodicity, outlier, trend, change-point, intermittency, noise, stationarity, and recent-regime
detection. A skill never sees future labels and its source is part of the outcome-cache key.

Enable the portfolio through `scripts/run_method_evolution.sh` with
`ME_FOUNDATION_PORTFOLIO=flagship5`, `ME_TSFM_RUNTIMES=chronos,timesfm`, a worker deployment file,
and the exact required license acknowledgement. The run fails before evaluation if any of the
five reviewed runtime bindings cannot resolve. Install `requirements.txt` in the Python process
that launches evolution; otherwise dependency-backed statistical methods would be measured as
environment failures rather than as forecasting methods.

## Architecture

```text
External parameters
  - base method definitions
  - method implementer
  - runtime registry
  - Train/Dev tasks and trusted labels
  - metric and budgets
          ↓
Dictionary Curation Adapter
          ↓
History-only hindcasting selector
          ↓
Generic Self-Evolution Core
Parent → Children → Train → frozen Dev → accept/reject → checkpoint
```

The generic core is in `common/evolution_core/`, a shared package owned by no single agent. It
does not import this package or any time-series dependency. It receives six components:

| Component | Responsibility |
|---|---|
| `ArtifactAdapter` | Validate, identify, serialize, restore, and Train-annotate an artifact. |
| `Mutator` | Propose bounded Child artifacts from sanitized Parent feedback. |
| `Executor` | Execute a frozen artifact on label-free items. |
| `Evaluator` | Resolve trusted labels and calculate reports after execution. |
| `AcceptanceGate` | Accept only configured held-out Dev improvement. |
| `ArtifactStore` | Persist artifacts, checkpoints, and traces. |

## Parameter groups

### Generic evolution parameters

These do not depend on dictionary curation:

- generations and children per generation;
- primary metric and minimize/maximize direction;
- strict acceptance margin;
- checkpoint/resume behavior.

### Dictionary-curation parameters

These specialize the framework for the current task:

- allowed actions: keep, revise, quarantine, discard;
- allowed method families and statuses;
- per-method revision budget;
- transient implementation retry budget;
- minimum successful-task coverage;
- history-only selector folds and horizon;
- status-classification thresholds;
- method and dictionary metrics;
- the permanent prohibition on Dev learning.

### Externally injected parameters

Collaborators provide:

- the base-method JSON file (normally `datasets/forecast_method_dataset_v001.json`);
- a `MethodImplementer` implementation;
- one or more `MethodRuntime` implementations;
- Train and Dev inputs and trusted labels;
- metric functions.

The Harness does not invent base method IDs. Unknown runtime providers become structured
`unavailable` results instead of crashing the run. CLI provider names are resolved through an
application-owned registry; arbitrary import strings are rejected.

The primary dictionary score is deployable: for each task, the Executor chooses one eligible
method using only rolling hindcasts inside the observed history, freezes that choice, and only
then lets the trusted Evaluator compare its future forecast with the label. The hindsight best
method across the whole dictionary is recorded separately as `oracle_score` for coverage
diagnosis and is never used as the acceptance metric. Quarantined and unavailable methods can be
executed for repair diagnostics but cannot be selected for the final forecast.

Phase B additionally supports a guarded dynamic TSFM-plus-statistical forecast combination. It
searches a small TSFM-heavy weight grid and clipped residual corrections, accepts a combination
only after majority-fold and worst-fold-regret checks, and otherwise falls back to a stable
Toto/TimesFM forecast. See [`docs/GUARDED_COMBINED_PHASE_B.md`](../docs/GUARDED_COMBINED_PHASE_B.md)
for the exact policy contract and the cached 80/20 exploratory result.

Selector evolution now uses the Dr-CiK point-forecast definition for its trusted Train/Dev gate:
per-task MAE and RMSE are divided by the mean absolute true future value, independently capped at
`5.0`, and then averaged across tasks. A Child must improve clipped mean sMAE while preserving
100% coverage, mean sRMSE, clipped-task counts, active-oracle regret, and the P90/P95 sMAE tail.
MASE, RMSSE, MAE, and sMAPE remain diagnostic metrics. This phase intentionally does **not**
compute sCRPS or generate probabilistic trajectories.

Existing frozen point forecasts can be rescored without any LLM or TSFM calls:

```bash
python -m numerical_agent.rescore_point_forecasts \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --per-task-results runs/frozen_two_stage/public_test_99_20260823/per_task_results.jsonl \
  --output-dir runs/frozen_two_stage/public_test_99_20260823/point_rescore \
  --baseline-row E_toto_reference
```

The output is a public-label development/regression report, not a verified official Hidden Test
score. The tool writes `point_rescore_results.json` and `POINT_RESCORE_REPORT.md` and records that
sCRPS was not computed.

## Python integration

```python
task = DictionaryCurationTask(
    base_dictionary=provided_dictionary,
    config=curation_config,
    implementer=provided_method_implementer,
    runtimes=provided_runtime_registry,
    labels=trusted_train_dev_labels,
    metric=provided_metric,
    store=JsonArtifactStore(output_dir),
)

engine = SelfEvolutionEngine(evolution_config, task.components())
outcome = engine.evolve(provided_dictionary, train_items, dev_items)
```

Train reports may create sanitized revision feedback and update method status. Dev evaluates
frozen Parent and Child dictionaries only; the controller never applies a Dev report to an
artifact.

## Frozen Public Test evaluation

After evolution and all method choices are frozen, evaluate the accepted dictionary once on the
entity-disjoint 99-task Public Test partition:

```bash
scripts/run_dictionary_frozen_test.sh
```

The script reads the default artifacts from `runs/dictionary_curation/full/`. Paths can be
overridden with `NA_TASKS_FILE`, `NA_EXPERIMENT_CONFIG`, `NA_DICTIONARY`, and
`NA_FROZEN_OUTPUT_DIR`. The underlying command is also available directly:

```bash
python -m numerical_agent evaluate-frozen \
  --tasks-file /path/to/Dr-CiK/data/tasks/train.jsonl \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --experiment-config runs/dictionary_curation/full/experiment.json \
  --dictionary runs/dictionary_curation/full/working_dictionary.json \
  --output-dir runs/dictionary_curation/frozen_public_test
```

This path registers only the existing sandbox runtime. It has no LLM/provider option, does not
implement or revise methods, does not update statuses or checkpoints, and does not participate in
Parent/Child acceptance. It writes `frozen_test_report.json` and
`frozen_test_forecasts.jsonl`; these results must never be fed back into evolution. A completed
report is never overwritten, and it records both the split-manifest and frozen-dictionary SHA-256
identifiers.

## Offline smoke test

The repository includes one deterministic fake provider and fixture dictionary solely to verify
framework wiring:

```bash
python -m numerical_agent curate \
  --experiment-config tests/fixtures/numerical_agent/experiment.json \
  --base-methods tests/fixtures/numerical_agent/base_methods.json \
  --provider fake \
  --output-dir runs/dictionary_curation_smoke
```

The fake method reads an opaque constant from the fixture JSON. It is not a forecasting baseline
and must not be reported as an experimental result.

## Outputs

The command writes:

- `best_artifact.json`: accepted generic artifact;
- `working_dictionary.json`: dictionary-specific alias of the accepted artifact;
- `method_evaluations.jsonl`: trusted per-generation Train diagnostics;
- `quarantine.json`: quarantined, unavailable, and discarded method records;
- `evolution_trace.jsonl`: Parent/Child decisions;
- `checkpoint.json`: resumable accepted state;
- generation-specific Parent and Child JSON snapshots.

The separate frozen evaluation command writes only:

- `frozen_test_report.json`: the aggregate Public Test metric and diagnostics;
- `frozen_test_forecasts.jsonl`: auditable method forecasts and history-only selection scores.

## Adding real methods later

Adding real Statistical, Foundation, or Combined methods should require only:

1. supplying method definitions in the base JSON schema;
2. registering an approved `MethodImplementer`;
3. registering compatible `MethodRuntime` providers;
4. supplying task loading and trusted metric functions.

The generic controller and acceptance logic should not change.
