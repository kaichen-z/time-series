# Champion Package Retrieval–Decision Coordinate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete package-native Retrieval and Decision evolution over a frozen Champion Numerical package, then bind both accepted coordinates into one reproducible triad bundle.

**Architecture:** Preserve `RetrievalEvolutionEngine` as the Retrieval scheduler and add a typed evaluator that calls `run_numerical_two_stage` over a frozen package registry. Add a package-native Decision evaluator and a `CoEvolutionEngine` subclass that reuses only its Decision mutation logic while replacing the old harness evaluation path. Wrap the accepted Numerical manifest, Retrieval release, and Decision policy in a package-coordinate bundle whose controller permits one changed module per step.

**Tech Stack:** Python 3.11+, frozen dataclasses, SHA-256 canonical JSON identities, pytest, existing Dr-CiK `drcik_point_metrics`, existing `NumericalForecastPackage`, `TwoStageRetrievalAgent`, `DecisionAgent`, `RetrievalEvolutionEngine`, and `HarnessPolicy`.

**Spec:** `docs/superpowers/specs/2026-09-02-champion-package-retrieval-decision-coordinate-design.md`

## Global Constraints

- Do not run or mutate Champion Numerical evolution in this change.
- Retrieval and Decision may select only forecasts already materialized in `NumericalForecastPackage.ranked_alternatives`.
- Retrieval evolution uses exactly 80 Train / 20 Dev tasks and package mode requires strict final sMAE/sRMSE gain.
- Legacy Retrieval evolution retains strict contextual-oracle gain by default.
- Decision requires an accepted non-`v000` Retrieval release.
- Every coordinate generation changes exactly one principal module; rejection preserves exact Parent bytes.
- Do not promote candidate Retrieval skills in package mode.
- Do not access Public-99, invoke a live LLM, download models, or run paid evolution during implementation and verification.

## File map

- `evolving_loop/retrieval_agent/evolution.py`: add the explicit strict-gain target and bind it into checkpoints/science replay.
- `evolving_loop/retrieval_agent/quality.py`: score a typed `FinalRetrievalCard` without depending on legacy `HarnessResult`.
- `evolving_loop/retrieval_agent/credit.py`: delegate legacy quality scoring to the new typed helper without changing legacy outputs.
- `evolving_loop/package_registry.py`: own immutable task-to-Champion-package registration and manifest identity.
- `evolving_loop/package_retrieval_evolution.py`: implement the trusted package-native `RetrievalEvaluator`.
- `evolving_loop/package_decision_evolution.py`: score package-native Decision policies and run Train/Dev Decision evolution.
- `evolving_loop/package_coordinate_evolution.py`: define the triad bundle and one-coordinate controller.
- `tests/test_retrieval_evolution.py`: protect old/new strict-gain and checkpoint behavior.
- `tests/test_package_retrieval_evolution.py`: protect the registry, typed quality scorer, and Retrieval evaluator.
- `tests/test_package_decision_evolution.py`: protect Decision-only mutation, selection regret, and Dev gates.
- `tests/test_package_coordinate_evolution.py`: protect triad identity, coordinate isolation, and exact-parent rejection.
- `tests/test_package_coordinate_e2e.py`: deterministic 80/20 package-native integration test with no Public access.

---

### Task 1: Make Retrieval strict gain explicit

**Files:**
- Modify: `evolving_loop/retrieval_agent/evolution.py`
- Modify: `tests/test_retrieval_evolution.py`

**Interfaces:**
- Produces: `RetrievalStrictGainTarget = Literal["contextual", "final"]`
- Produces: `RetrievalEvolutionConfig.strict_gain_target: RetrievalStrictGainTarget = "contextual"`
- Extends: `RetrievalEvaluation.gate_failures(..., require_strict_final_gain: bool = False)`
- Preserves: existing `require_strict_contextual_gain` callers and default behavior

- [ ] **Step 1: Write failing gate and config tests**

```python
def test_package_gate_accepts_strict_final_gain_with_frozen_oracle():
    parent = _evaluation(mean_final_smae=1.0, mean_final_srmse=1.0)
    child = _evaluation(
        mean_final_smae=0.9,
        mean_final_srmse=1.0,
        mean_contextual_oracle_smae=parent.mean_contextual_oracle_smae,
        mean_contextual_oracle_srmse=parent.mean_contextual_oracle_srmse,
    )
    assert child.gate_failures(
        parent,
        1e-12,
        require_strict_contextual_gain=False,
        require_strict_final_gain=True,
    ) == ()


def test_retrieval_config_rejects_unknown_strict_gain_target():
    with pytest.raises(RetrievalEvolutionError, match="strict_gain_target"):
        RetrievalEvolutionConfig(strict_gain_target="oracle")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_retrieval_evolution.py -k 'strict_final_gain or strict_gain_target'`

Expected: FAIL because the config field and final strict-gain argument do not exist.

- [ ] **Step 3: Implement the minimal gate/config behavior**

```python
RetrievalStrictGainTarget = Literal["contextual", "final"]


@dataclass(frozen=True)
class RetrievalEvolutionConfig:
    strict_gain_target: RetrievalStrictGainTarget = "contextual"


def gate_failures(
    self,
    parent: "RetrievalEvaluation",
    tolerance: float,
    *,
    require_strict_contextual_gain: bool = True,
    require_strict_final_gain: bool = False,
) -> tuple[str, ...]:
    if require_strict_contextual_gain and require_strict_final_gain:
        raise RetrievalEvolutionError("only one strict Retrieval gain target is allowed")
    # Keep all existing non-regression gates.
    if require_strict_final_gain and not (
        self.mean_final_smae < parent.mean_final_smae - tolerance
        or self.mean_final_srmse < parent.mean_final_srmse - tolerance
    ):
        failures.append("strict_final_gain")
```

- [ ] **Step 4: Route every Train/Dev and checkpoint-replay strict gate through the config**

```python
def _strict_gate_kwargs(self) -> dict[str, bool]:
    return {
        "require_strict_contextual_gain": self.config.strict_gain_target == "contextual",
        "require_strict_final_gain": self.config.strict_gain_target == "final",
    }
```

Use `**self._strict_gate_kwargs()` in live Train/Dev acceptance and completed-checkpoint attestation. Keep screen/fold Pareto calls non-strict. Add `strict_gain_target` to `_science_signature` and bump `RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION` from 2 to 3 so an old checkpoint cannot resume under different science.

- [ ] **Step 5: Run the complete Retrieval evolution suite**

Run: `pytest -q tests/test_retrieval_evolution.py tests/test_retrieval_e2e.py`

Expected: PASS, with old tests continuing to use contextual strict gain.

- [ ] **Step 6: Commit**

```bash
git add evolving_loop/retrieval_agent/evolution.py tests/test_retrieval_evolution.py tests/test_retrieval_e2e.py
git commit -m "feat(retrieval): select strict gain target"
```

---

### Task 2: Add the frozen package registry and Retrieval evaluator

**Files:**
- Create: `evolving_loop/package_registry.py`
- Create: `evolving_loop/retrieval_agent/quality.py`
- Create: `evolving_loop/package_retrieval_evolution.py`
- Modify: `evolving_loop/retrieval_agent/credit.py`
- Create: `tests/test_package_retrieval_evolution.py`
- Modify: `tests/test_retrieval_credit.py`

**Interfaces:**
- Produces: `FrozenNumericalPackageRegistry(entries: Sequence[tuple[ContextTask, NumericalForecastPackage]])`
- Produces: `FrozenNumericalPackageRegistry.package_for(task: ContextTask) -> NumericalForecastPackage`
- Produces: `FrozenNumericalPackageRegistry.fingerprint: str`
- Produces: `RetrievalCardQuality`
- Produces: `score_retrieval_card_quality(task: ContextTask, card: FinalRetrievalCard) -> RetrievalCardQuality`
- Produces: `PackageRetrievalEvaluator.evaluate(...) -> RetrievalEvaluation`
- Consumes: `run_numerical_two_stage`, `drcik_point_metrics`, and read-only accepted Retrieval skills

- [ ] **Step 1: Write failing registry tests**

```python
def test_registry_rejects_rebound_task_or_package():
    registry = FrozenNumericalPackageRegistry(((task, package),))
    assert registry.package_for(task) is package
    with pytest.raises(PackageRegistryError, match="task binding"):
        registry.package_for(replace(task, target_name="different"))


def test_registry_identity_changes_when_a_materialized_forecast_changes():
    left = FrozenNumericalPackageRegistry(((task, package),))
    right = FrozenNumericalPackageRegistry(((task, changed_package),))
    assert left.fingerprint != right.fingerprint
```

- [ ] **Step 2: Run registry tests and verify RED**

Run: `pytest -q tests/test_package_retrieval_evolution.py -k registry`

Expected: collection FAIL because `evolving_loop.package_registry` does not exist.

- [ ] **Step 3: Implement canonical immutable registration**

```python
class FrozenNumericalPackageRegistry:
    def __init__(
        self,
        entries: Sequence[tuple[ContextTask, NumericalForecastPackage]],
    ) -> None:
        # Reject empty, duplicate task IDs, unlabeled/malformed task bindings,
        # and inconsistent Champion component fingerprints.
        self._entries = MappingProxyType(validated_entries)
        self._task_sha256 = MappingProxyType(task_hashes)
        self._package_sha256 = MappingProxyType(package_hashes)
        self.fingerprint = _digest(manifest_payload)

    def package_for(self, task: ContextTask) -> NumericalForecastPackage:
        if _task_fingerprint(task) != self._task_sha256.get(task.numeric.task_id):
            raise PackageRegistryError("Numerical package task binding mismatch")
        return self._entries[task.numeric.task_id]
```

Expose a public `numerical_package_fingerprint()` wrapper in `evolving_loop.numerical_two_stage` if needed; do not duplicate its canonical package projection.

- [ ] **Step 4: Write failing typed Retrieval-quality tests**

```python
def test_typed_card_quality_counts_support_distractors_and_invalid_quotes():
    quality = score_retrieval_card_quality(task, final_card)
    assert quality.supporting_recall == 1.0
    assert quality.distractor_avoidance == 1.0
    assert quality.exact_quote_validity == 1.0
    assert quality.rejection_count == 0
```

Name the break: this fails if package scoring falls back to a fake `HarnessResult` or ignores verified quote audits.

- [ ] **Step 5: Implement `RetrievalCardQuality` and preserve legacy credit output**

```python
@dataclass(frozen=True)
class RetrievalCardQuality:
    supporting_recall: float
    gt_evidence_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    rejection_count: int


def score_retrieval_card_quality(
    task: ContextTask,
    card: FinalRetrievalCard,
) -> RetrievalCardQuality:
    retrieved = set(card.selected_document_ids)
    supporting = {
        item.document_id for item in task.documents if item.role == "supporting"
    }
    distractors = {
        item.document_id for item in task.documents if item.role == "distractor"
    }
    supporting_recall = (
        len(retrieved & supporting) / len(supporting) if supporting else 1.0
    )
    distractor_avoidance = (
        1.0 - len(retrieved & distractors) / len(distractors)
        if distractors
        else 1.0
    )
    citations = tuple(
        citation for chain in card.chains for citation in chain.citations
    )
    attempts = sum(
        item.quote_attempt_count
        for item in (card.round1, card.round2)
        if item is not None
    )
    valid = sum(
        item.valid_quote_count
        for item in (card.round1, card.round2)
        if item is not None
    )
    return RetrievalCardQuality(
        supporting_recall=supporting_recall,
        gt_evidence_recall=_ground_truth_recall(task, card.chains),
        distractor_avoidance=distractor_avoidance,
        exact_quote_validity=(valid / attempts if attempts else 1.0),
        complete_chain_rate=(
            sum(chain.numeric_eligible for chain in card.chains) / len(card.chains)
            if card.chains
            else 0.0
        ),
        rejection_count=len(card.rejected),
    )
```

Refactor `credit._retrieval_quality` to construct or obtain the final card and return the six values from this helper. Run `tests/test_retrieval_credit.py` to prove identical legacy behavior.

- [ ] **Step 6: Write failing package evaluator tests**

```python
def test_package_retrieval_evaluator_scores_final_gain_over_frozen_pool(tmp_path):
    evaluation = evaluator.evaluate(
        genome,
        (task,),
        stage="g0_parent_screen_train",
        skill_library=library,
        harness_factory=None,
        persist=False,
        writers_enabled=False,
        evolver_enabled=False,
        cache_keys=(cache_key,),
        metric_cap=5.0,
    )
    assert evaluation.mean_contextual_oracle_smae == 0.0
    assert evaluation.mean_final_smae == 0.0
    assert evaluation.promotion_evidence == ()
    assert evaluation.promotion_replays == ()
```

Also test task/package mismatch, unlabeled tasks, writable flags, nonempty `harness_factory`, changed dependency fingerprint, fallback/invalid counting, and materialized-pool oracle selection.

- [ ] **Step 7: Run evaluator tests and verify RED**

Run: `pytest -q tests/test_package_retrieval_evolution.py -k evaluator`

Expected: FAIL because `PackageRetrievalEvaluator` does not exist.

- [ ] **Step 8: Implement the trusted evaluator**

```python
class PackageRetrievalEvaluator:
    def __init__(
        self,
        registry: FrozenNumericalPackageRegistry,
        retrieval_factory: Callable[
            [RetrievalGenome, RetrievalSkillLibrary], TwoStageRetrievalAgent
        ],
        decision_factory: Callable[[], DecisionAgent],
        *,
        dependency_fingerprints: Mapping[str, str],
    ) -> None:
        if set(dependency_fingerprints) != {
            "retrieval_factory",
            "decision_factory",
            "bridge_runtime",
        }:
            raise RetrievalEvolutionError("package dependency fingerprints are incomplete")
        self.registry = registry
        self.retrieval_factory = retrieval_factory
        self.decision_factory = decision_factory
        self.dependency_fingerprints = MappingProxyType(
            dict(sorted(dependency_fingerprints.items()))
        )
        self.evaluator_hash = _digest(
            {
                "registry": registry.fingerprint,
                "dependencies": dict(self.dependency_fingerprints),
            }
        )

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        skill_library: RetrievalSkillLibrary | None,
        harness_factory: Callable[..., object] | None,
        persist: bool,
        writers_enabled: bool,
        evolver_enabled: bool,
        cache_keys: tuple[RetrievalInferenceCacheKey, ...],
        metric_cap: float,
    ) -> RetrievalEvaluation:
        _validate_evaluator_call(
            tasks,
            harness_factory=harness_factory,
            persist=persist,
            writers_enabled=writers_enabled,
            evolver_enabled=evolver_enabled,
            cache_keys=cache_keys,
        )
        library = _require_read_only_library(skill_library)
        rows = tuple(
            _score_package_task(
                task,
                self.registry.package_for(task),
                self.retrieval_factory(genome, library),
                self.decision_factory(),
                metric_cap=metric_cap,
            )
            for task in tasks
        )
        return _aggregate_retrieval_rows(genome.version, rows)
```

For each task, call `run_numerical_two_stage`, score its final forecast and every ranked alternative, select one oracle by `(srmse, smae, candidate_name)`, score `FinalRetrievalCard`, and aggregate only after every task resolves. Return empty promotion artifacts and never mutate the supplied library.

- [ ] **Step 9: Run package and legacy Retrieval suites**

Run: `pytest -q tests/test_package_retrieval_evolution.py tests/test_retrieval_credit.py tests/test_numerical_retrieval_handoff.py`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add evolving_loop/package_registry.py evolving_loop/package_retrieval_evolution.py evolving_loop/retrieval_agent/quality.py evolving_loop/retrieval_agent/credit.py evolving_loop/numerical_two_stage.py tests/test_package_retrieval_evolution.py tests/test_retrieval_credit.py
git commit -m "feat(retrieval): evaluate frozen packages"
```

---

### Task 3: Add package-native Decision evolution

**Files:**
- Create: `evolving_loop/package_decision_evolution.py`
- Create: `tests/test_package_decision_evolution.py`
- Modify: `evolving_loop/coordinate_evolution.py`
- Modify: `tests/test_coordinate_evolution.py`

**Interfaces:**
- Produces: `PackageDecisionEvaluator.evaluate(policy: HarnessPolicy, tasks: Sequence[ContextTask]) -> PolicyEvaluation`
- Produces: `PackageDecisionEvolutionEngine(CoEvolutionEngine)`
- Preserves: `DecisionEvolutionPhaseAdapter` and its accepted-release authority checks
- Consumes: frozen registry, frozen accepted Retrieval factory, `Callable[[HarnessPolicy], DecisionAgent]`

- [ ] **Step 1: Write failing Decision evaluator tests**

```python
def test_decision_evaluator_reports_materialized_selection_regret():
    evaluation = evaluator.evaluate(policy, (task,))
    assert evaluation.diagnostics["mean_selection_smae_regret"] == 0.5
    assert evaluation.diagnostics["mean_selection_srmse_regret"] == 0.5
    assert evaluation.diagnostics["public_test_accessed"] == 0.0


def test_decision_evaluator_rejects_policy_without_accepted_retrieval():
    with pytest.raises(PackageDecisionEvolutionError, match="non-v000"):
        evaluator.evaluate(HarnessPolicy(), (task,))
```

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `pytest -q tests/test_package_decision_evolution.py -k evaluator`

Expected: collection FAIL because the package Decision module does not exist.

- [ ] **Step 3: Implement package scoring as existing typed outcomes**

```python
class PackageDecisionEvaluator:
    def evaluate(
        self,
        policy: HarnessPolicy,
        tasks: Sequence[ContextTask],
    ) -> PolicyEvaluation:
        task_rows = tuple(self._score_task(policy, task) for task in tasks)
        outcomes = tuple(row.outcome for row in task_rows)
        diagnostics = evaluation_diagnostics(outcomes)
        diagnostics.update(
            p90_smae=linear_quantile([float(row.final_smae) for row in outcomes], 0.90),
            p95_smae=linear_quantile([float(row.final_smae) for row in outcomes], 0.95),
            invalid_count=float(sum(row.invalid_count for row in task_rows)),
            catastrophic_count=float(
                sum(row.catastrophic_count for row in task_rows)
            ),
            public_test_accessed=0.0,
        )
        return PolicyEvaluation(
            version=policy.version,
            system_reward=-diagnostics["mean_srmse"],
            module_rewards={"coding": 0.0, "retrieval": 0.0, "decision": -diagnostics["mean_srmse"]},
            outcomes=outcomes,
            failure_traces=failure_traces,
            diagnostics=diagnostics,
        )
```

The `ResolvedOutcome` coding/contextual oracle fields both point to the best frozen Numerical alternative. The final fields score only `NumericalTwoStageResult.forecast`.

- [ ] **Step 4: Write failing Decision-engine isolation and Dev-gate tests**

```python
def test_package_decision_engine_accepts_only_decision_coordinate(monkeypatch):
    accepted, trace = engine.evolve(parent, train, dev)
    assert accepted.parent == parent.version
    assert principal_module_fingerprints(accepted)["retrieval"] == principal_module_fingerprints(parent)["retrieval"]


def test_package_decision_engine_preserves_exact_parent_on_dev_rejection():
    selected, trace = engine.evolve(parent, train, dev)
    assert selected is parent
    assert selected.canonical_bytes() == parent.canonical_bytes()
    assert trace[-1].accepted_version == parent.version
```

Also test tail regression, invalid/fallback increase, cross-coordinate mutation, and Public flag rejection.

- [ ] **Step 5: Run engine tests and verify RED**

Run: `pytest -q tests/test_package_decision_evolution.py -k engine`

Expected: FAIL because the custom engine is absent.

- [ ] **Step 6: Implement a package-native `evolve` method while reusing Decision mutation**

```python
class PackageDecisionEvolutionEngine(CoEvolutionEngine):
    def __init__(self, llm, evaluator, config=None):
        super().__init__(llm, _unreachable_legacy_harness_factory, config)
        if self.config.mode != "genome" or self.config.target != "decision":
            raise PackageDecisionEvolutionError("package Decision target must be decision")
        self.package_evaluator = evaluator

    def evolve(self, seed, train_tasks, dev_tasks):
        incumbent = seed
        history = []
        for generation in range(self.config.generations):
            parent_train = self.package_evaluator.evaluate(incumbent, train_tasks)
            children = tuple(
                self.mutate(incumbent, parent_train, child_index=index)
                for index in range(self.config.children_per_generation)
            )
            eligible = tuple(
                child
                for child in children
                if _decision_coordinate_only(incumbent, child)
                and not _package_decision_gate_failures(
                    self.package_evaluator.evaluate(child, train_tasks),
                    parent_train,
                    self.config.screening_tolerance,
                    require_strict=True,
                )
            )
            train_winner = min(
                eligible,
                key=lambda child: self.package_evaluator.evaluate(
                    child, train_tasks
                ).rank_key,
                default=incumbent,
            )
            parent_dev = self.package_evaluator.evaluate(incumbent, dev_tasks)
            child_dev = self.package_evaluator.evaluate(train_winner, dev_tasks)
            failures = _package_decision_gate_failures(
                child_dev,
                parent_dev,
                self.config.screening_tolerance,
                require_strict=True,
            )
            accepted = train_winner is not incumbent and not failures
            selected = train_winner if accepted else incumbent
            history.append(
                _decision_evolution_step(
                    generation, incumbent, children, parent_train, parent_dev,
                    child_dev, selected
                )
            )
            incumbent = selected
        return incumbent, tuple(history)
```

Implement `_package_decision_gate_failures(child, parent, tolerance, require_strict)` to enforce both means, one strict gain, p90/p95, invalid/catastrophic/fallback non-regression, and `public_test_accessed == 0.0`. Validate `principal_module_fingerprints` before evaluating a child.

- [ ] **Step 7: Prove compatibility with the existing coordinate adapter**

Add a test that passes `PackageDecisionEvolutionEngine` through `DecisionEvolutionPhaseAdapter`, loads the exact accepted Retrieval release, and accepts only a Decision fingerprint change. Make only type/error-message adjustments in `coordinate_evolution.py` if the subclass exposes a stricter protocol.

- [ ] **Step 8: Run Decision and coordinate suites**

Run: `pytest -q tests/test_package_decision_evolution.py tests/test_coordinate_evolution.py tests/test_co_evolution.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add evolving_loop/package_decision_evolution.py evolving_loop/coordinate_evolution.py tests/test_package_decision_evolution.py tests/test_coordinate_evolution.py
git commit -m "feat(decision): evolve frozen package choices"
```

---

### Task 4: Bind the accepted Numerical/Retrieval/Decision triad

**Files:**
- Create: `evolving_loop/package_coordinate_evolution.py`
- Create: `tests/test_package_coordinate_evolution.py`

**Interfaces:**
- Produces: `PackageCoordinateBundle`
- Produces: `package_principal_fingerprints(bundle) -> Mapping[str, str]`
- Produces: `PackageCoordinateOutcome`
- Produces: `PackageCoordinateController.run(...) -> tuple[PackageCoordinateBundle, tuple[PackageCoordinateStep, ...]]`
- Consumes: already gated Retrieval/Decision phase runners; the Numerical coordinate is read-only in this change

- [ ] **Step 1: Write failing bundle identity tests**

```python
def test_bundle_identity_binds_champion_release_retrieval_and_decision():
    fingerprints = package_principal_fingerprints(bundle)
    assert set(fingerprints) == {"numerical", "retrieval", "decision"}
    assert fingerprints["numerical"] != package_principal_fingerprints(changed_numerical)["numerical"]
    assert fingerprints["retrieval"] != package_principal_fingerprints(changed_retrieval)["retrieval"]
    assert fingerprints["decision"] != package_principal_fingerprints(changed_decision)["decision"]
```

- [ ] **Step 2: Run bundle tests and verify RED**

Run: `pytest -q tests/test_package_coordinate_evolution.py -k bundle`

Expected: collection FAIL because the package-coordinate module does not exist.

- [ ] **Step 3: Implement canonical bundle bytes and transitive identities**

```python
@dataclass(frozen=True)
class PackageCoordinateBundle:
    generation: int
    parent_sha256: str | None
    numerical_manifest_sha256: str
    policy: HarnessPolicy
    runtime_fingerprints: Mapping[str, str]

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
```

Require canonical SHA-256 runtime values for the bridge, Retrieval runtime, Decision runtime, verifier, and metric policy. Numerical identity comes from the frozen registry manifest, Retrieval identity from the embedded accepted release payload/SHA, and Decision identity from only Decision-owned policy fields.

- [ ] **Step 4: Write failing controller isolation tests**

```python
def test_controller_accepts_one_target_module_only():
    selected, trace = controller.run(parent, train, dev)
    assert trace[0].changed_modules == ("retrieval",)
    assert trace[1].changed_modules == ("decision",)


def test_controller_rejection_preserves_parent_bytes():
    selected, trace = rejecting_controller.run(parent, train, dev)
    assert selected.canonical_bytes() == parent.canonical_bytes()
    assert trace[0].accepted_bytes_sha256 == trace[0].parent_bytes_sha256
```

Also test Decision-before-non-v000 rejection, Public access rejection, changed Numerical manifest during Retrieval/Decision, detached parent lineage, and accepted Child with zero changed modules.

- [ ] **Step 5: Run controller tests and verify RED**

Run: `pytest -q tests/test_package_coordinate_evolution.py -k controller`

Expected: FAIL because `PackageCoordinateController` does not exist.

- [ ] **Step 6: Implement one-coordinate transitions**

```python
class PackageCoordinateController:
    def run(self, parent, train_tasks, dev_tasks):
        current = parent
        steps = []
        for target, runner in self._scheduled_runners():
            outcome = runner.run(current.policy, train_tasks, dev_tasks)
            child = current.with_policy(outcome.bundle) if outcome.accepted else current
            changed = _changed_principal_modules(current, child)
            accepted = outcome.accepted and changed == (target,)
            current = child if accepted else current
            steps.append(
                PackageCoordinateStep.from_transition(
                    generation=current.generation,
                    target=target,
                    parent=current,
                    child=child,
                    accepted=accepted,
                    reason=outcome.reason,
                    public_test_accessed=outcome.public_test_accessed,
                )
            )
        return current, tuple(steps)
```

The controller never publishes a Numerical Child; it only binds the exact frozen manifest supplied at construction.

- [ ] **Step 7: Run package-coordinate and legacy-coordinate suites**

Run: `pytest -q tests/test_package_coordinate_evolution.py tests/test_coordinate_evolution.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evolving_loop/package_coordinate_evolution.py tests/test_package_coordinate_evolution.py
git commit -m "feat(evolution): bind champion triad"
```

---

### Task 5: Prove the deterministic 80/20 closed loop

**Files:**
- Create: `tests/test_package_coordinate_e2e.py`
- Modify: `evolving_loop/__init__.py` only if the package uses explicit public exports
- Modify: `docs/superpowers/specs/2026-09-02-champion-package-retrieval-decision-coordinate-design.md` only to change status to implemented and record the final module names

**Interfaces:**
- Consumes: all Task 1–4 public interfaces
- Produces: deterministic end-to-end proof with no external model, live LLM, filesystem release fabrication, or Public access

- [ ] **Step 1: Write the failing 80/20 integration test**

```python
def test_package_coordinate_evolution_is_deterministic_and_never_accesses_public(tmp_path):
    train = tuple(_context_task(index, split="train") for index in range(80))
    dev = tuple(_context_task(index, split="dev") for index in range(20))
    registry = FrozenNumericalPackageRegistry(
        tuple((task, _champion_package(task)) for task in (*train, *dev))
    )

    retrieval_result = retrieval_engine.evolve(seed_genome, train, dev)
    assert retrieval_result.accepted
    assert retrieval_result.selected_genome.version != "v000"

    final_bundle, trace = coordinate_controller.run(seed_bundle, train, dev)
    assert tuple(step.target for step in trace) == ("retrieval", "decision")
    assert all(not step.public_test_accessed for step in trace)
    assert final_bundle.numerical_manifest_sha256 == registry.fingerprint
    assert numerical_call_count == 0
    assert all(selected_forecast in materialized_forecasts for selected_forecast in decisions)
```

The fake LLM responses are fixed JSON and all expectations are hand-derived literals. The registry receives exactly 100 labeled Train/Dev tasks and no Public task object exists in the fixture.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `pytest -q tests/test_package_coordinate_e2e.py`

Expected: FAIL at the first missing integration or identity contract.

- [ ] **Step 3: Add only the minimal exports/wiring required by the integration test**

Do not add a CLI that launches live evolution. Keep construction explicit so callers must supply frozen packages, exact accepted release paths, factories, and dependency fingerprints.

- [ ] **Step 4: Run all focused tests**

Run:

```bash
pytest -q \
  tests/test_package_retrieval_evolution.py \
  tests/test_package_decision_evolution.py \
  tests/test_package_coordinate_evolution.py \
  tests/test_package_coordinate_e2e.py \
  tests/test_retrieval_evolution.py \
  tests/test_retrieval_credit.py \
  tests/test_coordinate_evolution.py \
  tests/test_numerical_retrieval_handoff.py \
  tests/test_co_evolution.py
```

Expected: PASS.

- [ ] **Step 5: Run the complete repository suite**

Run: `pytest -q`

Expected: PASS apart from an explicitly documented pre-existing failure reproduced on `c80e1db`.

- [ ] **Step 6: Verify diff hygiene and Public isolation**

Run:

```bash
git diff --check
git status --short
rg -n "Public-99|public_test_accessed" evolving_loop/package_* tests/test_package_*
```

Inspect every match: package modules may reject/record Public access but must not load a Public dataset or scorer.

- [ ] **Step 7: Update the spec status and commit**

```bash
git add evolving_loop tests docs/superpowers/specs/2026-09-02-champion-package-retrieval-decision-coordinate-design.md
git commit -m "test(evolution): prove package triad loop"
```
