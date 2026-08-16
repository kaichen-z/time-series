# Parameterized Numerical Self-Evolution Harness

This package composes a generic Parent/Child Self-Evolution Core with a dictionary-curation
adapter. The current implementation builds the framework only. It does not implement ARIMA, ETS,
Chronos, TimesFM, or any real Statistical, Foundation, or Combined forecasting method.

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
- successive-halving prefixes, tolerance, and promotion budget;
- deterministic seed;
- checkpoint/resume behavior.

### Dictionary-curation parameters

These specialize the framework for the current task:

- allowed actions: keep, revise, quarantine, discard;
- allowed method families and statuses;
- per-method revision budget;
- status-classification thresholds;
- method and dictionary metrics;
- the permanent prohibition on Dev learning.

### Externally injected parameters

Collaborators provide:

- the base-method JSON file;
- a `MethodImplementer` implementation;
- one or more `MethodRuntime` implementations;
- Train and Dev inputs and trusted labels;
- metric functions.

The Harness does not invent base method IDs. Unknown runtime providers become structured
`unavailable` results instead of crashing the run. CLI provider names are resolved through an
application-owned registry; arbitrary import strings are rejected.

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
