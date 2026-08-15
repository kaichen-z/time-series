# Parameterized Self-Evolution Harness Design

## Goal

Build a reusable Self-Evolution Harness whose lifecycle is configured through parameters and
pluggable adapters. The first adapter targets numerical tool-dictionary curation, but the core
must not contain implementations of statistical models, time-series foundation models, or
combined forecasting methods.

The current work delivers the framework. Collaborators supply the base method definitions,
method implementation strategy, and runtime backends later.

## Current Phase

The target experiment is configured as:

1. Load one externally supplied dictionary of base methods.
2. Ask one Numerical Agent, through an injected implementer, to materialize or revise methods.
3. Test methods through an injected executor and trusted evaluator.
4. Keep, revise, quarantine, or discard methods according to configured performance rules.
5. Accept a Child dictionary only through the shared Train/Dev acceptance lifecycle.

The Harness provides this lifecycle and its contracts. It does not author the real method
dictionary or implement ARIMA, ETS, Chronos, TimesFM, ensembles, or any other forecasting method.

## Non-Goals

This phase does not implement:

- concrete statistical forecasting methods;
- concrete TSFM wrappers;
- concrete statistical/TSFM combined methods;
- context retrieval or document processing;
- Retrieval or Decision Agents;
- multi-agent co-evolution;
- LLM weight training.

Small deterministic fake methods are permitted only as test fixtures for verifying the framework.

## Architecture

The implementation has a parameterized core and task-specific adapters:

```text
External Experiment Inputs
  - Base methods
  - Tasks and split
  - LLM / method implementer
  - Method runtime providers
  - Metrics and budgets
            ↓
Dictionary Curation Adapter
  - Artifact schema
  - Mutation actions
  - Execution mapping
  - Failure attribution
            ↓
Generic Self-Evolution Core
  Parent → Children → Execute → Evaluate → Accept/Reject → Persist
```

The core is unaware of time-series forecasting. It operates on generic artifacts, candidates,
execution results, and scores.

## Generic Core Contracts

The core receives an `EvolutionComponents` bundle:

```python
ArtifactT = TypeVar("ArtifactT")
CandidateT = TypeVar("CandidateT")
ResultT = TypeVar("ResultT")

@dataclass(frozen=True)
class EvolutionComponents(Generic[ArtifactT, CandidateT, ResultT]):
    artifact_adapter: ArtifactAdapter[ArtifactT]
    mutator: Mutator[ArtifactT, CandidateT]
    executor: Executor[CandidateT, ResultT]
    evaluator: Evaluator[ResultT]
    acceptance_gate: AcceptanceGate
    store: ArtifactStore[ArtifactT]
```

The protocols are:

- `ArtifactAdapter`: load, validate, clone, fingerprint, and serialize an artifact;
- `Mutator`: propose bounded Children from a Parent and sanitized failure feedback;
- `Executor`: execute one Child on an evaluation item without seeing its label;
- `Evaluator`: score frozen execution outputs with trusted labels;
- `AcceptanceGate`: compare Parent and Child aggregate reports;
- `ArtifactStore`: persist generations, checkpoints, traces, and accepted artifacts.

The controller implements only:

```text
Load Parent
→ Evaluate Parent on Train
→ Propose Children
→ Validate mutation boundaries
→ Optional successive-halving screen
→ Evaluate eligible Children on Train
→ Evaluate the best eligible Child on read-only Dev
→ Accept or reject
→ Checkpoint and continue
```

## Generic Evolution Parameters

Every experiment supplies an `EvolutionConfig`:

```json
{
  "generations": 1,
  "children_per_generation": 2,
  "seed": 20260816,
  "primary_metric": "smape",
  "objective": "minimize",
  "acceptance_margin": 0.0,
  "successive_halving": {
    "enabled": true,
    "screen_train_items": 6,
    "screen_dev_items": 2,
    "max_promoted_children": 1,
    "tolerance": 0.01
  },
  "resume": true,
  "output_dir": "runs/dictionary_curation"
}
```

These parameters are shared by future Program, Dictionary, Prompt, Genome, or other adapters.

## Dictionary-Curation Task Parameters

The current task is described by a separate adapter configuration:

```json
{
  "artifact_type": "tool_dictionary",
  "adapter": "dictionary_curation",
  "allowed_actions": ["keep", "revise", "quarantine", "discard"],
  "allowed_families": ["statistical", "foundation", "combined"],
  "max_revisions_per_method": 1,
  "method_statuses": [
    "unimplemented",
    "accepted",
    "specialized",
    "quarantined",
    "unavailable",
    "discarded"
  ],
  "method_metric": "smape",
  "dictionary_metric": "smape",
  "discard_requires_dominance_evidence": true,
  "allow_dev_learning": false
}
```

These fields specialize the generic lifecycle for dictionary curation. They are data, not
hard-coded branches in the controller.

## Externally Supplied Base Methods

The real method definitions are an input parameter:

```json
{
  "schema_version": 1,
  "dictionary_id": "forecast_tools_raw_v000",
  "methods": [
    {
      "method_id": "provided_by_collaborator",
      "family": "statistical",
      "description": "Externally supplied method description.",
      "assumptions": [],
      "failure_conditions": [],
      "implementation_spec": {},
      "status": "unimplemented"
    }
  ]
}
```

The Harness validates only the common schema. It does not define the real methods.

The Python API receives the base methods and their capabilities through dependency injection:

```python
task = DictionaryCurationTask(
    base_methods=method_catalog,
    implementer=provided_method_implementer,
    runtimes=provided_runtime_registry,
    task_source=provided_train_dev_source,
    metric=provided_metric,
)

engine = SelfEvolutionEngine(
    config=evolution_config,
    components=task.components(),
)
```

For CLI usage, importable objects are not loaded from arbitrary user strings. The application
constructs a provider registry and the experiment config refers to approved provider names.

## Pluggable Method Implementer

The framework defines but does not implement the domain-specific `MethodImplementer` protocol:

```python
class MethodImplementer(Protocol):
    def implement(self, method: MethodDefinition, context: ImplementationContext) -> MethodCandidate:
        ...

    def revise(
        self,
        parent: MethodCandidate,
        feedback: SanitizedMethodFeedback,
    ) -> MethodCandidate:
        ...
```

A later experiment can inject:

- an LLM coding implementer for statistical methods;
- a registry-backed wrapper implementer for TSFMs;
- a composition implementer for combined methods;
- a single implementer that dispatches among all three families.

The framework invokes the interface identically and records which provider produced each
candidate.

## Pluggable Method Runtime

The framework also defines a runtime contract:

```python
class MethodRuntime(Protocol):
    def supports(self, candidate: MethodCandidate) -> bool:
        ...

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        ...
```

The runtime registry selects a supplied runtime by implementation kind or provider name. Missing
providers produce a structured `unavailable` result rather than crashing the whole run.

## Dictionary-Curation Adapter

The adapter maps the generic contracts to the current task:

| Generic contract | Dictionary-curation behavior |
|---|---|
| Artifact | Versioned method dictionary |
| Candidate | Child dictionary or revised method candidate |
| Mutator | Single-agent keep/revise/quarantine/discard proposal |
| Executor | Supplied method runtime on historical cutoffs |
| Evaluator | Supplied metric over frozen forecasts |
| Acceptance | Parent/Child dictionary comparison on Dev |
| Store | JSON artifacts, implementations, checkpoints, and traces |

For every supplied method, the adapter orchestrates:

```text
Method definition
→ injected implementer
→ common validation
→ injected runtime
→ historical evaluation
→ sanitized feedback
→ optional injected revision
→ trusted status classification
```

The framework owns orchestration and generic validation. Actual forecasting logic remains outside
the framework.

## Status Rules

The adapter supports the following results:

- `accepted`: valid and competitive on a broad applicable set;
- `specialized`: useful on a coherent subset despite weak global averages;
- `quarantined`: repairable or insufficiently validated;
- `unavailable`: required injected provider is absent;
- `discarded`: unsafe, irreparably invalid, or dominated with sufficient evidence.

The task configuration controls thresholds. The LLM may propose a status, but trusted Python rules
make the final transition.

## Train/Dev Boundary

The task source is injected but must expose separate Train and Dev iterables. The controller
enforces:

- Train may generate method feedback and revisions;
- Dev may score frozen Parent and Child artifacts only;
- Dev cannot update implementations, metadata, memory, or prompts;
- Test is not part of the evolution API.

The existing `splits/drcik_public_v1.json` can be passed by the Dr-CiK experiment, but the generic
core does not depend on Dr-CiK.

## Persistence

The generic store writes:

- `best_artifact.json`;
- `evolution_trace.json`;
- `checkpoint.json`;
- `train_evaluation.json`;
- `dev_evaluation.json`.

The dictionary adapter additionally writes:

- `working_dictionary.json`;
- `method_evaluations.jsonl`;
- `quarantine.json`;
- candidate implementation artifacts returned by the supplied implementer.

## Package Structure

```text
evolving_agent/
  evolution_core/
    contracts.py           Generic protocols and records.
    controller.py          Parent/Child generation lifecycle.
    acceptance.py          Metric direction and acceptance gates.
    halving.py             Optional successive-halving screening.
    persistence.py         Checkpoints, artifacts, and traces.

numerical_agent/
  adapters/
    dictionary_curation.py Dictionary-specific contract mapping.
  dictionary.py            Common method/dictionary schemas.
  providers.py             Approved implementer/runtime registry.
  config.py                Task-specific parameter validation.
  main.py                  CLI composition only.
```

Existing LLM, retry, metric, sandbox, and task-loading utilities may be adapted behind these
protocols. The generic core must not import `numerical_agent`.

## CLI

```bash
python -m numerical_agent curate \
  --experiment-config configs/dictionary_curation.json \
  --base-methods path/to/collaborator_methods.json \
  --provider-config path/to/provider_registry.json \
  --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --split-manifest splits/drcik_public_v1.json \
  --output-dir runs/dictionary_curation
```

The framework can be tested before real methods exist by injecting deterministic fake
implementers, runtimes, tasks, and metrics.

## Framework Acceptance Criteria

The current implementation is complete when it can:

1. Run a generic Parent/Child lifecycle without importing time-series modules.
2. Validate task and evolution parameters independently.
3. Load externally supplied method definitions without embedding real methods.
4. Invoke injected implementer and runtime protocols.
5. Isolate labels until execution outputs are frozen.
6. Produce keep/revise/quarantine/discard transitions through trusted rules.
7. Accept an improving Child and reject a non-improving Child on read-only Dev.
8. Resume from a checkpoint without repeating completed evaluations.
9. Persist generic and dictionary-specific artifacts.
10. Pass offline tests using fake providers and preserve the existing regression suite.

## Deferred Integration

After collaborators provide the real base dictionary and providers, the same framework can run
the full statistical, foundation, and combined-method experiment without changing the generic
controller. Program Self-Harness and later multi-agent evolution can be integrated as additional
adapters rather than separate orchestration implementations.
