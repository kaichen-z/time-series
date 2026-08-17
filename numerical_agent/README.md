# Parameterized Numerical Self-Evolution Harness

This package composes a generic Parent/Child Self-Evolution Core with a dictionary-curation
adapter. It now ships an auditable forecasting-method dataset, but it still does not implement
ARIMA, ETS, Chronos, TimesFM, or the other listed forecasting runtimes. Method definitions and
executable providers remain deliberately separate.

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

## Adding real methods later

Adding real Statistical, Foundation, or Combined methods should require only:

1. supplying method definitions in the base JSON schema;
2. registering an approved `MethodImplementer`;
3. registering compatible `MethodRuntime` providers;
4. supplying task loading and trusted metric functions.

The generic controller and acceptance logic should not change.
