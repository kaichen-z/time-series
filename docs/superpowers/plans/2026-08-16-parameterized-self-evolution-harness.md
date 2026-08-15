# Parameterized Self-Evolution Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Parent/Child Self-Evolution core and a parameterized dictionary-curation adapter without implementing any real statistical, TSFM, or combined forecasting methods.

**Architecture:** A time-series-independent controller consumes injected Artifact, Mutator, Executor, Evaluator, AcceptanceGate, and ArtifactStore components. The numerical dictionary adapter supplies task-specific schemas and keep/revise/quarantine/discard semantics, while all base methods, implementers, runtimes, task sources, and metrics remain external parameters. Offline fake providers prove the complete lifecycle.

**Tech Stack:** Python 3.10+, dataclasses, typing Protocol/Generic, JSON/JSONL persistence, pathlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-16-tool-dictionary-curation-self-harness-design.md`

## Global Constraints

- The generic core must not import `numerical_agent`, Dr-CiK, forecasting models, or LLM clients.
- No concrete statistical method, TSFM wrapper, or combined forecasting implementation is added.
- Train may produce mutation feedback; Dev is read-only and cannot update artifacts or memories.
- The trusted controller, not the LLM/provider, owns scores, statuses, acceptance, persistence, and rollback.
- Test fixtures may use deterministic fake methods only.
- Existing untracked run outputs, split files, and user changes remain untouched.

---

### Task 1: Generic Contracts and Configuration

**Files:**
- Create: `evolving_agent/evolution_core/__init__.py`
- Create: `evolving_agent/evolution_core/contracts.py`
- Test: `tests/test_evolution_core_contracts.py`

**Interfaces:**
- Consumes: Python standard library only.
- Produces: `MetricSpec`, `EvolutionConfig`, `EvaluationReport`, `MutationContext`, `EvolutionComponents`, `ArtifactAdapter`, `Mutator`, `Executor`, `Evaluator`, `AcceptanceGate`, and `ArtifactStore`.

- [ ] **Step 1: Write failing configuration and protocol tests**

```python
from dataclasses import dataclass

import pytest

from evolving_agent.evolution_core.contracts import EvolutionConfig, MetricSpec


def test_evolution_config_rejects_invalid_budgets():
    with pytest.raises(ValueError, match="generations"):
        EvolutionConfig(generations=0)
    with pytest.raises(ValueError, match="children_per_generation"):
        EvolutionConfig(children_per_generation=0)


def test_metric_spec_orders_minimized_scores():
    metric = MetricSpec(name="smape", objective="minimize")
    assert metric.better(10.0, 12.0)
    assert not metric.better(12.0, 10.0)
```

- [ ] **Step 2: Run the focused test and verify import failure**

Run: `pytest -q tests/test_evolution_core_contracts.py`

Expected: FAIL because `evolving_agent.evolution_core` does not exist.

- [ ] **Step 3: Implement immutable contracts**

Use Python-3.10-compatible `TypeVar` and `Generic`, not PEP 695 syntax. Define:

```python
@dataclass(frozen=True)
class MetricSpec:
    name: str
    objective: Literal["minimize", "maximize"] = "minimize"

    def better(self, candidate: float, parent: float, margin: float = 0.0) -> bool:
        return candidate < parent - margin if self.objective == "minimize" else candidate > parent + margin


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 1
    children_per_generation: int = 2
    seed: int = 20260816
    metric: MetricSpec = field(default_factory=lambda: MetricSpec("smape"))
    acceptance_margin: float = 0.0
    successive_halving: bool = False
    screen_train_items: int = 6
    screen_dev_items: int = 2
    max_promoted_children: int = 1
    screening_tolerance: float = 0.01
    resume: bool = True
```

Add positive-budget and non-negative-margin validation. Define protocol signatures around:

```python
ArtifactAdapter.validate(artifact) -> None
ArtifactAdapter.artifact_id(artifact) -> str
ArtifactAdapter.to_payload(artifact) -> dict[str, object]
ArtifactAdapter.from_payload(payload) -> artifact
ArtifactAdapter.apply_train_report(artifact, report) -> artifact
Mutator.propose(parent, context, count) -> Sequence[artifact]
Executor.execute(artifact, items, split) -> Sequence[result]
Evaluator.evaluate(artifact_id, results, split) -> EvaluationReport
AcceptanceGate.accept(parent_report, child_report) -> bool
ArtifactStore.save_artifact(name, payload) -> Path
ArtifactStore.save_checkpoint(payload) -> Path
ArtifactStore.load_checkpoint() -> dict[str, object] | None
ArtifactStore.append_trace(payload) -> None
```

- [ ] **Step 4: Export public contracts and run tests**

Run: `pytest -q tests/test_evolution_core_contracts.py`

Expected: PASS.

- [ ] **Step 5: Commit the contracts**

```bash
git add evolving_agent/evolution_core tests/test_evolution_core_contracts.py
git commit -m "feat(evolution): add generic contracts"
```

### Task 2: Acceptance Gate and Atomic JSON Store

**Files:**
- Create: `evolving_agent/evolution_core/acceptance.py`
- Create: `evolving_agent/evolution_core/persistence.py`
- Test: `tests/test_evolution_core_persistence.py`

**Interfaces:**
- Consumes: `MetricSpec`, `EvaluationReport`, and ArtifactStore protocol from Task 1.
- Produces: `MetricAcceptanceGate` and `JsonArtifactStore`.

- [ ] **Step 1: Write failing acceptance and round-trip tests**

```python
from evolving_agent.evolution_core.acceptance import MetricAcceptanceGate
from evolving_agent.evolution_core.contracts import EvaluationReport, MetricSpec
from evolving_agent.evolution_core.persistence import JsonArtifactStore


def report(score: float) -> EvaluationReport:
    return EvaluationReport("v", "dev", {"smape": score}, 2, {})


def test_acceptance_requires_strict_improvement():
    gate = MetricAcceptanceGate(MetricSpec("smape", "minimize"), margin=0.0)
    assert gate.accept(report(10.0), report(9.9))
    assert not gate.accept(report(10.0), report(10.0))


def test_json_store_round_trips_checkpoint(tmp_path):
    store = JsonArtifactStore(tmp_path)
    store.save_checkpoint({"generation": 2, "accepted_artifact": {"id": "v002"}})
    assert store.load_checkpoint()["generation"] == 2
```

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_evolution_core_persistence.py`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement strict metric acceptance**

Validate that both reports contain a finite primary metric. Delegate direction and margin handling
to `MetricSpec.better`.

- [ ] **Step 4: Implement atomic JSON persistence**

Write UTF-8 JSON to a sibling temporary file and replace the destination atomically. Implement
artifact snapshots, `checkpoint.json`, and append-only `evolution_trace.jsonl`. Reject trace
payloads that are not JSON objects.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_evolution_core_persistence.py`

Expected: PASS.

```bash
git add evolving_agent/evolution_core tests/test_evolution_core_persistence.py
git commit -m "feat(evolution): add acceptance and store"
```

### Task 3: Generic Parent/Child Controller

**Files:**
- Create: `evolving_agent/evolution_core/controller.py`
- Test: `tests/test_evolution_core_controller.py`

**Interfaces:**
- Consumes: all Task 1 protocols, `MetricAcceptanceGate`, and `JsonArtifactStore`.
- Produces: `SelfEvolutionEngine.evolve(parent, train_items, dev_items) -> EvolutionOutcome`.

- [ ] **Step 1: Write a fake lifecycle test**

Create in-test fake components where artifacts are `{"id": "v000", "quality": 0.0}` and the
mutator proposes qualities `1.0` and `-1.0`. The evaluator returns `loss = 10 - quality`.

```python
def test_engine_accepts_only_the_improving_child(tmp_path):
    engine = make_fake_engine(tmp_path, proposed_qualities=(1.0, -1.0))
    outcome = engine.evolve(
        parent={"id": "v000", "quality": 0.0},
        train_items=(1, 2),
        dev_items=(3, 4),
    )
    assert outcome.accepted_artifact["quality"] == 1.0
    assert outcome.steps[0].accepted
```

Also test that equal/worse Dev performance retains the Parent and that an empty Train or Dev split
raises `ValueError`.

- [ ] **Step 2: Verify the focused test fails**

Run: `pytest -q tests/test_evolution_core_controller.py`

Expected: FAIL because `SelfEvolutionEngine` is missing.

- [ ] **Step 3: Implement one-generation evaluation**

Add private helpers that execute label-free outputs before passing them to the evaluator. For each
Child: validate it, evaluate on Train, call `apply_train_report`, and revalidate the annotated
artifact. Select the Train-best Child by the configured metric, then evaluate frozen Parent and
Child artifacts on Dev. Never call `apply_train_report` for Dev.

- [ ] **Step 4: Add multiple generations and trace records**

Persist Parent, Children, Train reports, Dev reports, accepted version, and mutation context per
generation. The next generation starts from the accepted artifact only.

- [ ] **Step 5: Add successive-halving screening tests and implementation**

Test that obvious losers are evaluated only on deterministic prefixes and that at most
`max_promoted_children` receive full Train evaluation. Screening must not accept a Child; full Dev
evaluation remains mandatory.

- [ ] **Step 6: Add checkpoint/resume tests and implementation**

Interrupt after generation one using a test hook, load `checkpoint.json`, resume, and assert that
generation one is not re-executed. Validate checkpoint artifact payloads through the adapter before
use.

- [ ] **Step 7: Run controller tests and commit**

Run: `pytest -q tests/test_evolution_core_controller.py`

Expected: PASS.

```bash
git add evolving_agent/evolution_core/controller.py tests/test_evolution_core_controller.py
git commit -m "feat(evolution): add generic controller"
```

### Task 4: Dictionary Schemas and Provider Protocols

**Files:**
- Create: `numerical_agent/__init__.py`
- Create: `numerical_agent/config.py`
- Create: `numerical_agent/dictionary.py`
- Create: `numerical_agent/providers.py`
- Test: `tests/test_numerical_dictionary_contracts.py`

**Interfaces:**
- Consumes: standard library and `EvaluationReport`.
- Produces: `MethodDefinition`, `MethodCandidate`, `MethodRecord`, `ToolDictionary`,
  `DictionaryCurationConfig`, `MethodImplementer`, `MethodRuntime`, and `RuntimeRegistry`.

- [ ] **Step 1: Write failing schema tests**

```python
import pytest

from numerical_agent.dictionary import MethodDefinition, ToolDictionary


def test_dictionary_rejects_duplicate_method_ids():
    method = MethodDefinition("m1", "statistical", "external method")
    with pytest.raises(ValueError, match="duplicate"):
        ToolDictionary("d0", None, 0, (method, method))


def test_dictionary_accepts_external_method_without_implementation():
    method = MethodDefinition("m1", "foundation", "provided later")
    dictionary = ToolDictionary("d0", None, 0, (method,))
    assert dictionary.methods[0].status == "unimplemented"
```

Also test allowed families/statuses, dependency references, JSON round trips, provider lookup, and
missing runtime behavior.

- [ ] **Step 2: Verify import failure**

Run: `pytest -q tests/test_numerical_dictionary_contracts.py`

Expected: FAIL because the package contracts do not exist.

- [ ] **Step 3: Implement immutable schemas**

Use dataclasses with explicit `to_payload`/`from_payload`; do not embed callable objects in JSON.
`MethodCandidate` stores a provider name and opaque JSON-compatible implementation payload.
`ToolDictionary` stores version, Parent ID, generation, and method records.

- [ ] **Step 4: Implement injected provider protocols and registry**

Define `implement`, `revise`, `supports`, and `forecast` signatures from the approved spec.
`RuntimeRegistry.resolve(candidate)` returns a registered runtime or a structured unavailable
result; it never imports arbitrary module paths from experiment JSON.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_numerical_dictionary_contracts.py`

Expected: PASS.

```bash
git add numerical_agent tests/test_numerical_dictionary_contracts.py
git commit -m "feat(numerical): add dictionary contracts"
```

### Task 5: Dictionary-Curation Adapter

**Files:**
- Create: `numerical_agent/adapters/__init__.py`
- Create: `numerical_agent/adapters/dictionary_curation.py`
- Test: `tests/test_dictionary_curation_adapter.py`

**Interfaces:**
- Consumes: generic evolution contracts; dictionary schemas; injected implementer, runtimes,
  metric, and task items.
- Produces: `DictionaryCurationTask.components() -> EvolutionComponents`.

- [ ] **Step 1: Write deterministic fake-provider tests**

Define fake methods `good`, `repairable`, `unavailable`, and `unsafe` only inside the test. The fake
implementer returns opaque payloads; the fake runtime returns deterministic forecasts.

Assert:

```python
def test_adapter_implements_tests_revises_and_classifies_methods():
    task = make_fake_dictionary_task()
    child = task.mutator.propose(task.parent, empty_context(), count=1)[0]
    results = task.executor.execute(child, fake_train_items(), "train")
    report = task.evaluator.evaluate(child.dictionary_id, results, "train")
    annotated = task.artifact_adapter.apply_train_report(child, report)
    statuses = {method.method_id: method.status for method in annotated.methods}
    assert statuses == {
        "good": "accepted",
        "repairable": "accepted",
        "unavailable": "unavailable",
        "unsafe": "discarded",
    }
```

Add tests that Dev evaluation cannot call `implement`, `revise`, or `apply_train_report`, and that
a globally weak but subset-winning method becomes `specialized` rather than discarded.

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_dictionary_curation_adapter.py`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement artifact adapter and bounded mutator**

The mutator operates only on externally supplied method records. It calls the injected implementer
for `unimplemented` methods and `revise` only when sanitized Train feedback and remaining revision
budget exist. It does not invent base method IDs.

- [ ] **Step 4: Implement execution and trusted evaluation**

The executor resolves an injected runtime, freezes forecasts, and records structured success,
unavailable, invalid, timeout, or unsafe results. The evaluator invokes the injected metric and
aggregates per-method/per-task diagnostics without exposing labels to providers.

- [ ] **Step 5: Implement trusted status classification**

Map configured thresholds and evidence to accepted/specialized/quarantined/unavailable/discarded.
Require explicit invalidity or dominance evidence for discard; do not discard solely by global
mean error.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_dictionary_curation_adapter.py`

Expected: PASS.

```bash
git add numerical_agent/adapters tests/test_dictionary_curation_adapter.py
git commit -m "feat(numerical): add curation adapter"
```

### Task 6: CLI Composition and Offline Smoke Run

**Files:**
- Create: `numerical_agent/__main__.py`
- Modify: `numerical_agent/main.py`
- Modify: `pyproject.toml`
- Create: `tests/fixtures/numerical_agent/base_methods.json`
- Create: `tests/test_numerical_agent_cli.py`

**Interfaces:**
- Consumes: `SelfEvolutionEngine`, `DictionaryCurationTask`, approved provider registry, JSON
  experiment config, base-method path, task source, and output directory.
- Produces: `python -m numerical_agent curate ...` and persisted run artifacts.

- [ ] **Step 1: Write a failing CLI smoke test**

```python
def test_curate_cli_runs_with_fake_provider(tmp_path):
    completed = subprocess.run(
        [
            sys.executable, "-m", "numerical_agent", "curate",
            "--experiment-config", str(FIXTURE_CONFIG),
            "--base-methods", str(FIXTURE_METHODS),
            "--provider", "fake",
            "--output-dir", str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "best_artifact.json").exists()
    assert (tmp_path / "working_dictionary.json").exists()
```

- [ ] **Step 2: Verify the CLI test fails**

Run: `pytest -q tests/test_numerical_agent_cli.py`

Expected: FAIL because CLI composition is missing.

- [ ] **Step 3: Implement CLI parsing and approved provider registration**

Support only the `curate` command, required config/base-method/output paths, and provider names
registered by application code. Add `numerical_agent*` to setuptools package discovery. Real
providers are absent; `fake` is available only for test/smoke execution.

- [ ] **Step 4: Implement artifact mapping**

After the generic store writes `best_artifact.json`, materialize dictionary-specific aliases:
`working_dictionary.json`, `method_evaluations.jsonl`, and `quarantine.json` through the adapter.

- [ ] **Step 5: Run CLI test and commit**

Run: `pytest -q tests/test_numerical_agent_cli.py`

Expected: PASS.

```bash
git add numerical_agent pyproject.toml tests/fixtures/numerical_agent tests/test_numerical_agent_cli.py
git commit -m "feat(numerical): add curation CLI"
```

### Task 7: Regression Verification and Usage Documentation

**Files:**
- Modify: `README.md`
- Create: `numerical_agent/README.md`
- Test: all new tests and existing suite.

**Interfaces:**
- Consumes: completed framework and CLI.
- Produces: user-facing explanation of injected parameters and explicit non-implementation of real
  methods.

- [ ] **Step 1: Document framework-only scope**

Document the Generic Core parameters, Dictionary Adapter parameters, external base-method/provider
inputs, fake smoke command, output artifacts, Train/Dev firewall, and the fact that no real
Statistical/TSFM/Combined methods ship in this phase.

- [ ] **Step 2: Run focused framework tests**

Run:

```bash
pytest -q \
  tests/test_evolution_core_contracts.py \
  tests/test_evolution_core_persistence.py \
  tests/test_evolution_core_controller.py \
  tests/test_numerical_dictionary_contracts.py \
  tests/test_dictionary_curation_adapter.py \
  tests/test_numerical_agent_cli.py
```

Expected: all PASS.

- [ ] **Step 3: Run the entire existing regression suite**

Run: `pytest -q`

Expected: all PASS with no tests accessing Hidden-Test labels or external networks.

- [ ] **Step 4: Run the offline fake-provider command manually**

Run the command documented in `numerical_agent/README.md`, inspect `best_artifact.json`,
`working_dictionary.json`, `evolution_trace.jsonl`, and `checkpoint.json`, and verify no real method
implementation appears in source or artifacts.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md numerical_agent/README.md
git commit -m "docs: explain curation harness"
```
