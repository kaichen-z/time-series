"""Build-only Champion evolution controller regressions."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import common.llm as llm_module
import numerical_agent.evolution.champion_controller as controller_module
import numerical_agent.evolution.champion_proposal as proposal_module
from common.data import Task
from common.payload import canonical_json_bytes
from numerical_agent.dictionary import MethodDefinition, ToolDictionary
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    champion_fingerprint,
)
from numerical_agent.evolution.champion_controller import (
    ChampionArtifactStore,
    ChampionAuthorityStore,
    ChampionCallableBinding,
    ChampionCheckpoint,
    ChampionControllerError,
    ChampionEvolutionController,
    ChampionEvolutionConfig,
    ChampionLifecycleError,
    ChampionProposerAdapter,
    ChampionRowProviderAdapter,
    ChampionRunAttestations,
    ChampionRuntimeBindings,
    ChampionRunManifest,
    canonical_release_bytes,
    partition_train_tasks,
    run_build_evolution,
)
from numerical_agent.evolution.champion_evidence import (
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionTaskRow,
    ProposerEvidence,
)
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.screening import TaskProfile


def _profile(task_id: str, index: int) -> TaskProfile:
    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=100 + index,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=False,
        trend_direction="up",
        trend_strength=0.5,
        periodicity_periods=(7,),
        periodicity_strength=float(index % 8) / 8.0,
        periodicity_confidence=0.75,
        outlier_fraction=0.0,
        noise_relative_scale=0.2,
        likely_stationary=False,
        stationarity_score=0.25,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _parent() -> ChampionRelease:
    recipe = ChampionRecipe(
        name="active_parent",
        kind="select",
        parents=("baseline_leaf",),
        fallback_parent="baseline_leaf",
        assumptions=(
            EvolutionAssumption(
                assumption_id="parent_history",
                candidate_name="baseline_leaf",
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator="select",
                rationale="History length supports the baseline forecast.",
                failure_condition="History length no longer supports the baseline forecast.",
            ),
        ),
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=recipe,
            thresholds=(("parent_history", 0.0),),
        ),
        source_hashes=(
            (
                "baseline_leaf",
                hashlib.sha256(b"baseline dictionary\n").hexdigest(),
            ),
        ),
        metric_policy_fingerprint="1" * 64,
        lineage=("active_parent",),
    )


def _history_diagnostic(name: str, *, eligible: bool = True) -> ChampionHistoryDiagnostic:
    return ChampionHistoryDiagnostic.from_candidate(
        CandidateDiagnostics.synthetic(
            name=name,
            family="statistical",
            median_mase=0.5,
            eligible=eligible,
        )
    )


def _rows(count: int = 64, *, enriched: bool = False) -> tuple[ChampionTaskRow, ...]:
    errors = {
        "baseline_leaf": 2.0,
        "better_a": 0.8,
        "better_b": 0.9,
        "better_c": 1.0,
        "better_d": 1.1,
        "better_e": 1.2,
        "worse_a": 2.8,
        "worse_b": 2.9,
        "worse_c": 3.0,
        "worse_d": 3.1,
        "worse_e": 3.2,
    }
    rows: list[ChampionTaskRow] = []
    for index in range(count):
        task_id = f"build_case_{index:03d}"
        profile = _profile(task_id, index)
        truth = (10.0 + index, 12.0 + index)
        history = (10.0,) * profile.history_length if enriched else None
        for candidate_name, error in errors.items():
            rows.append(
                ChampionTaskRow(
                    task_id=task_id,
                    candidate_name=candidate_name,
                    profile=profile,
                    truth=truth,
                    forecast=tuple(value + error for value in truth),
                    fold=index % 5,
                    split="build",
                    history=history,
                    diagnostic=(_history_diagnostic(candidate_name) if enriched else None),
                )
            )
    return tuple(rows)


def _config(*, generations: int = 1) -> ChampionEvolutionConfig:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    return ChampionEvolutionConfig(
        generations=generations,
        screen_task_ids=(task_ids[:8], task_ids[:32], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )


def _proposal_batch(prefix: str, generation: int = 0) -> tuple[ChampionRecipe, ...]:
    return tuple(
        ChampionRecipe(
            name=f"proposal_{generation}_{index}",
            kind="select",
            parents=(f"{prefix}_{letter}",),
            fallback_parent=f"{prefix}_{letter}",
            assumptions=(
                EvolutionAssumption(
                    assumption_id=f"history_{generation}_{index}",
                    candidate_name=f"{prefix}_{letter}",
                    feature="history_length",
                    direction="above",
                    horizon_region="full",
                    operator="select",
                    rationale="History length supports the candidate forecast.",
                    failure_condition="History length stops supporting the candidate forecast.",
                ),
            ),
        )
        for index, letter in enumerate("abcde")
    )


@dataclass
class RecordingProposer:
    prefix: str
    calls: int = 0

    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT
        assert evidence.label == "adaptive_train_build_diagnostic"
        generation = self.calls
        self.calls += 1
        return _proposal_batch(self.prefix, generation)


@dataclass
class DiverseProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        assert parent is PARENT

        def assumption(index: int, kind: str, candidate: str) -> EvolutionAssumption:
            return EvolutionAssumption(
                assumption_id=f"diverse_history_{index}",
                candidate_name=candidate,
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator=kind,  # type: ignore[arg-type]
                rationale="History length supports the candidate forecast.",
                failure_condition="History length stops supporting the candidate forecast.",
            )

        recipes: list[ChampionRecipe] = []
        for index, letter in enumerate("abc"):
            candidate = f"better_{letter}"
            recipes.append(
                ChampionRecipe(
                    name=f"select_proposal_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(assumption(index, "select", candidate),),
                )
            )
        for index, kind in enumerate(("weighted", "median"), start=3):
            recipes.append(
                ChampionRecipe(
                    name=f"{kind}_proposal",
                    kind=kind,  # type: ignore[arg-type]
                    parents=("better_d", "better_e"),
                    fallback_parent="better_d",
                    assumptions=(assumption(index, kind, "better_d"),),
                )
            )
        return tuple(recipes)


@dataclass
class AllOperatorProposer:
    def __call__(
        self,
        parent: ChampionRelease,
        evidence: ProposerEvidence,
    ) -> tuple[ChampionRecipe, ...]:
        kinds = (
            "select",
            "route",
            "horizon_route",
            "weighted",
            "median",
            "bounded_overlay",
        )
        recipes: list[ChampionRecipe] = []
        for index, kind in enumerate(kinds):
            parents = (
                ("better_a",)
                if kind == "select"
                else (
                    "baseline_leaf",
                    "better_a",
                )
            )
            recipes.append(
                ChampionRecipe(
                    name=f"all_operator_{kind}",
                    kind=kind,  # type: ignore[arg-type]
                    parents=parents,
                    fallback_parent=parents[0],
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"operator_history_{index}",
                            candidate_name=parents[-1],
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator=kind,  # type: ignore[arg-type]
                            rationale="History diagnostics support this operator.",
                            failure_condition="History diagnostics stop supporting this operator.",
                        ),
                    ),
                )
            )
        return tuple(recipes)


PARENT = _parent()
ROWS_64 = _rows()


def _task(task_id: str, entity_name: str) -> Task:
    return Task(
        task_id=task_id,
        history_values=(1.0, 2.0, 3.0),
        future_values=(4.0,),
        prediction_length=1,
        frequency="D",
        seasonal_period=None,
        entity_name=entity_name,
    )


TRAIN_80 = tuple(_task(f"train_{index:03d}", f"entity_{index // 4:03d}") for index in range(80))
DEV_20 = tuple(_task(f"dev_{index:03d}", f"dev_entity_{index:03d}") for index in range(20))
_CALIBRATION_TASK_IDS = tuple(
    task.task_id
    for task in partition_train_tasks(
        TRAIN_80, build_size=64, calibration_size=16, seed=20260901
    ).calibration
)


def _lifecycle_config(parts) -> ChampionEvolutionConfig:
    build_ids = tuple(task.task_id for task in parts.build)
    return ChampionEvolutionConfig(
        build_task_ids=build_ids,
        screen_task_ids=(build_ids[:8], build_ids[:32], build_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )


def _manifest(
    parts,
    config: ChampionEvolutionConfig,
    attestations: ChampionRunAttestations | None = None,
) -> ChampionRunManifest:
    return ChampionRunManifest(
        schema_version=1,
        partition_seed=20260901,
        source_hashes=(
            attestations.source_hashes
            if attestations is not None
            else (("dev_tasks", "b" * 64), ("train_tasks", "a" * 64))
        ),
        train_tasks=tuple((task.task_id, task.entity_name) for task in TRAIN_80),
        train_task_hashes=tuple(
            (task.task_id, controller_module.task_content_fingerprint(task)) for task in TRAIN_80
        ),
        dev_tasks=tuple((task.task_id, task.entity_name) for task in DEV_20),
        dev_task_hashes=tuple(
            (task.task_id, controller_module.task_content_fingerprint(task)) for task in DEV_20
        ),
        split_manifest_fingerprint=(
            attestations.split_manifest_fingerprint if attestations is not None else "c" * 64
        ),
        build_tasks=tuple((task.task_id, task.entity_name) for task in parts.build),
        calibration_tasks=tuple((task.task_id, task.entity_name) for task in parts.calibration),
        dictionary_hashes=(
            attestations.dictionary_hashes if attestations is not None else PARENT.source_hashes
        ),
        forecast_store_fingerprint=(
            attestations.forecast_store_fingerprint if attestations is not None else "d" * 64
        ),
        metric_policy_fingerprint=PARENT.metric_policy_fingerprint,
        proposal_model=(
            attestations.proposal_model
            if attestations is not None
            else "deterministic-test-proposer"
        ),
        proposal_config_fingerprint=(
            attestations.proposal_config_fingerprint if attestations is not None else "e" * 64
        ),
        proposal_implementation_fingerprint=(
            attestations.proposal_implementation_fingerprint
            if attestations is not None
            else "7" * 64
        ),
        row_provider_fingerprint=(
            attestations.row_provider_fingerprint if attestations is not None else "8" * 64
        ),
        schedule_fingerprint=config.fingerprint,
        numeric_grid_fingerprint=(
            attestations.numeric_grid_fingerprint if attestations is not None else "f" * 64
        ),
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        runtime_fingerprint=(
            attestations.runtime_fingerprint if attestations is not None else "9" * 64
        ),
        runtime_implementation_fingerprint=(
            attestations.runtime_implementation_fingerprint
            if attestations is not None
            else "6" * 64
        ),
        forecast_runtime_identity_fingerprint=(
            attestations.forecast_runtime_identity_fingerprint
            if attestations is not None
            else "5" * 64
        ),
    )


@dataclass(frozen=True)
class LifecycleRows:
    calibration_error: float = 0.5
    dev_error: float = 0.5

    def __call__(self, tasks: tuple[Task, ...], split: str) -> tuple[ChampionTaskRow, ...]:
        error = {
            "build": 0.5,
            "calibration": self.calibration_error,
            "dev": self.dev_error,
        }[split]
        rows: list[ChampionTaskRow] = []
        for index, task in enumerate(tasks):
            base_profile = _profile(task.task_id, index)
            profile = TaskProfile(
                task_id=base_profile.task_id,
                frequency=base_profile.frequency,
                history_length=len(task.history_values),
                horizon=task.prediction_length,
                zero_fraction=base_profile.zero_fraction,
                signed=base_profile.signed,
                integer_valued=base_profile.integer_valued,
                trend_direction=base_profile.trend_direction,
                trend_strength=base_profile.trend_strength,
                periodicity_periods=base_profile.periodicity_periods,
                periodicity_strength=base_profile.periodicity_strength,
                periodicity_confidence=base_profile.periodicity_confidence,
                outlier_fraction=base_profile.outlier_fraction,
                noise_relative_scale=base_profile.noise_relative_scale,
                likely_stationary=base_profile.likely_stationary,
                stationarity_score=base_profile.stationarity_score,
                recent_regime_start=base_profile.recent_regime_start,
                recent_regime_confidence=base_profile.recent_regime_confidence,
                intermittency_adi=base_profile.intermittency_adi,
                intermittency_cv2=base_profile.intermittency_cv2,
            )
            for candidate_name, candidate_error in {
                "baseline_leaf": 2.0,
                "better_a": error,
                "better_b": error + 0.01,
                "better_c": error + 0.02,
                "better_d": error + 0.03,
                "better_e": error + 0.04,
            }.items():
                rows.append(
                    ChampionTaskRow(
                        task_id=task.task_id,
                        candidate_name=candidate_name,
                        profile=profile,
                        truth=task.future_values,
                        forecast=tuple(value + candidate_error for value in task.future_values),
                        fold=index % 5,
                        split=split,  # type: ignore[arg-type]
                        history=task.history_values,
                        diagnostic=_history_diagnostic(candidate_name),
                    )
                )
        return tuple(rows)


@dataclass(frozen=True)
class SerializedRecordingProposer:
    def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
        serialized = repr((parent, evidence))
        assert not any(task_id in serialized for task_id in _CALIBRATION_TASK_IDS)
        assert "calibration" not in serialized.lower()
        return _proposal_batch("better")


@dataclass(frozen=True)
class ForbiddenProposer:
    def __call__(self, parent, evidence):
        raise AssertionError("proposer callback must remain unopened")


@dataclass(frozen=True)
class ForbiddenRows:
    def __call__(self, tasks, split):
        raise AssertionError("row-provider callback must remain unopened")


@dataclass(frozen=True)
class InterruptAtCalibration:
    delegate: LifecycleRows = field(default_factory=LifecycleRows)

    def __call__(self, tasks, split):
        if split == "calibration":
            raise RuntimeError("simulated interruption")
        return self.delegate(tasks, split)


def test_formal_adapter_rejects_noncallable_state_with_behavior_methods() -> None:
    @dataclass(frozen=True)
    class BehaviorState:
        def materialize(self, tasks, split):
            return ("unregistered-behavior",)

    @dataclass(frozen=True)
    class DelegatingRows:
        behavior: BehaviorState = field(default_factory=BehaviorState)

        def __call__(self, tasks, split):
            return self.behavior.materialize(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        ChampionRowProviderAdapter.bind(
            identity="noncallable-state-row-provider",
            callback=DelegatingRows(),
            config={},
        )


def test_formal_adapter_rejects_staticmethod_dispatch() -> None:
    @dataclass(frozen=True)
    class StaticRows:
        @staticmethod
        def materialize(tasks, split):
            return ("unregistered-behavior",)

        def __call__(self, tasks, split):
            return self.materialize(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        ChampionRowProviderAdapter.bind(
            identity="staticmethod-row-provider",
            callback=StaticRows(),
            config={},
        )


def test_formal_adapter_rejects_inherited_classmethod_dispatch() -> None:
    class InheritedRows:
        @classmethod
        def materialize(cls, tasks, split):
            return ("unregistered-behavior",)

    @dataclass(frozen=True)
    class DelegatingRows(InheritedRows):
        def __call__(self, tasks, split):
            return self.materialize(tasks, split)

    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        ChampionRowProviderAdapter.bind(
            identity="inherited-classmethod-row-provider",
            callback=DelegatingRows(),
            config={},
        )


def test_formal_adapter_rejects_external_transitive_callable_before_execution(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class StatisticsRows:
        def __call__(self, tasks, split):
            return statistics.mean((1.0, 3.0))

    monkeypatch.setattr(statistics, "_exact_ratio", lambda value: (int(value) * 2, 1))

    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        binding = ChampionRowProviderAdapter.bind(
            identity="statistics-row-provider",
            callback=StatisticsRows(),
            config={},
        )
        binding((), "build")


def test_scripted_proposer_fingerprint_binds_canonical_batches() -> None:
    better = ChampionProposerAdapter.scripted(
        identity="stateful-test-proposer",
        proposal_batches=(_proposal_batch("better"),),
        config={"temperature": 0.0},
    )
    worse = ChampionProposerAdapter.scripted(
        identity="stateful-test-proposer",
        proposal_batches=(_proposal_batch("worse"),),
        config={"temperature": 0.0},
    )

    assert better.fingerprint != worse.fingerprint


def test_formal_adapter_rejects_mutable_or_generic_custom_callable() -> None:
    class MutableProposer:
        def __init__(self) -> None:
            self.mode = ["better"]

        def __call__(self, parent, evidence):
            return _proposal_batch(self.mode[0])

    with pytest.raises(ChampionLifecycleError, match="immutable|mutable|closed"):
        ChampionProposerAdapter.bind(
            identity="mutable-test-proposer",
            callback=MutableProposer(),
            config={"temperature": 0.0},
        )
    with pytest.raises(ChampionLifecycleError, match="closed|adapter|runtime"):
        ChampionCallableBinding.bind(
            kind="proposer",
            identity="generic-test-proposer",
            callback=lambda parent, evidence: (),
            config={},
        )


def test_formal_runtime_bindings_detect_live_expander_mutation(monkeypatch) -> None:
    runtime = ChampionRuntimeBindings.formal()

    def replaced(recipe, rows):
        return ()

    monkeypatch.setattr(controller_module, "expand_recipe", replaced)

    with pytest.raises(ChampionLifecycleError, match="runtime|expander|executable"):
        runtime.verify_live_globals()


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("WEIGHT_GRID", ((0.9, 0.1),)),
        ("OVERLAY_ALPHA_GRID", (0.125,)),
        ("CORRECTION_CAP_GRID", (0.125,)),
        ("HORIZON_SPLIT_GRID", (0.125,)),
        ("_QUANTILES", (0.1, 0.9)),
    ),
)
def test_formal_runtime_bindings_detect_transitive_grid_mutation(
    monkeypatch, name, replacement
) -> None:
    runtime = ChampionRuntimeBindings.formal()
    monkeypatch.setattr(proposal_module, name, replacement)

    with pytest.raises(ChampionLifecycleError, match="runtime|behavior|global|implementation"):
        runtime.verify_live_globals()


def test_formal_adapter_rejects_unlisted_behavior_dependency() -> None:
    unlisted_dependency = object()

    @dataclass(frozen=True)
    class UnknownDependencyProposer:
        def __call__(self, parent, evidence):
            if unlisted_dependency is None:
                raise AssertionError("unreachable")
            return ()

    with pytest.raises(ChampionLifecycleError, match="unsupported|unlisted|closed"):
        ChampionProposerAdapter.bind(
            identity="unknown-dependency-proposer",
            callback=UnknownDependencyProposer(),
            config={},
        )


def test_sealed_scripted_proposer_and_materialized_rows_execute() -> None:
    proposals = _proposal_batch("better")
    proposer = ChampionProposerAdapter.scripted(
        identity="deterministic-test-proposer",
        proposal_batches=(proposals,),
        config={"temperature": 0.0},
    )
    rows = _rows(1)
    provider = ChampionRowProviderAdapter.materialized(
        identity="deterministic-test-row-provider",
        rows=rows,
        config={"contract": "materialized_rows_v1"},
    )
    evidence = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    task = _task("build_case_000", "entity_000")

    assert proposer.propose(PARENT, evidence, generation=1) == proposals
    assert provider.provide((task,), "build") == rows
    assert proposer.kind == "scripted"
    assert provider.kind == "materialized"


def test_sealed_codex_proposer_config_is_lazy_and_manifest_distinct(
    monkeypatch,
) -> None:
    class ExplodingClient:
        def __init__(self, config) -> None:
            raise AssertionError("Codex client must be constructed only on invocation")

    monkeypatch.setattr(llm_module, "CodexCLIClient", ExplodingClient)
    inventory = ToolDictionary(
        dictionary_id="sealed_task9_inventory",
        parent_dictionary_id=None,
        generation=0,
        methods=(
            MethodDefinition(
                method_id="baseline_leaf",
                family="statistical",
                description="Sealed adapter construction fixture.",
                status="accepted",
            ),
        ),
    )


def test_sealed_codex_proposer_allocates_generation_scoped_host_ids(monkeypatch) -> None:
    class Response:
        text = json.dumps(
            {
                "recipes": [
                    {
                        "kind": "select",
                        "parents": [f"better_{letter}"],
                        "fallback_parent": f"better_{letter}",
                        "assumptions": [
                            {
                                "candidate_name": f"better_{letter}",
                                "feature": feature,
                                "direction": "above",
                                "horizon_region": "full",
                                "operator": "select",
                            }
                        ],
                    }
                    for letter, feature in zip(
                        "abcde",
                        (
                            "history_length",
                            "horizon",
                            "horizon_ratio",
                            "zero_fraction",
                            "trend_strength",
                        ),
                        strict=True,
                    )
                ]
            }
        )

    class DeterministicClient:
        def __init__(self, config) -> None:
            self.config = config

        def complete(self, **kwargs):
            return Response()

    monkeypatch.setattr(llm_module, "CodexCLIClient", DeterministicClient)
    inventory = ToolDictionary(
        "generation_scoped_inventory",
        None,
        0,
        tuple(
            MethodDefinition(name, "statistical", "fixture", status="accepted")
            for name in ("baseline_leaf", "better_a", "better_b", "better_c", "better_d", "better_e")
        ),
    )
    adapter = ChampionProposerAdapter.codex_cli(
        identity="gpt-5.6-sol",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        inventory=inventory,
    )
    evidence = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )

    first = adapter.propose(PARENT, evidence, generation=1)
    second = adapter.propose(PARENT, evidence, generation=2)

    assert first[0].name == "generation_1_recipe_0"
    assert second[0].name == "generation_2_recipe_0"
    assert first[0].assumptions[0].assumption_id == "generation_1_assumption_0_0"
    assert second[0].assumptions[0].assumption_id == "generation_2_assumption_0_0"
    adapter = ChampionProposerAdapter.codex_cli(
        identity="gpt-5.6-sol",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        inventory=inventory,
        cache_dir="runs/champion-agent-cache",
    )

    assert adapter.kind == "codex_cli"
    assert adapter.config["model"] == "gpt-5.6-sol"
    assert adapter.config["reasoning_effort"] == "high"
    assert (
        adapter.fingerprint
        != ChampionProposerAdapter.scripted(
            identity="gpt-5.6-sol",
            proposal_batches=(_proposal_batch("better"),),
            config=adapter.config,
        ).fingerprint
    )


@pytest.mark.parametrize("raw_boundary", ("proposer", "row_provider"))
def test_formal_controller_rejects_raw_unbound_callables(tmp_path, raw_boundary) -> None:
    bound = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows())
    arguments = {
        "manifest": bound.manifest,
        "config": bound.config,
        "attestations": bound.attestations,
        "proposer": bound.proposer,
        "row_provider": bound.row_provider,
        "runtime_bindings": bound.runtime_bindings,
        "artifact_store": bound.artifact_store,
        "authority_store": bound.authority_store,
    }
    arguments[raw_boundary] = (
        SerializedRecordingProposer() if raw_boundary == "proposer" else LifecycleRows()
    )

    with pytest.raises(ChampionLifecycleError, match="typed|binding"):
        ChampionEvolutionController(**arguments)  # type: ignore[arg-type]


def _actual_attestations(
    run_root,
    proposer_binding: ChampionProposerAdapter,
    row_provider_binding: ChampionRowProviderAdapter,
    runtime_bindings: ChampionRuntimeBindings,
) -> ChampionRunAttestations:
    inputs = run_root.parent / f"{run_root.name}.inputs"
    inputs.mkdir(exist_ok=True)
    contents = {
        "dev_tasks.src": b"dev task source\n",
        "train_tasks.src": b"train task source\n",
        "split.json": b'{"split":"registered"}\n',
        "baseline.py": b"baseline dictionary\n",
        "forecast.store": b"forecast store identity\n",
        "runtime.py": b"runtime identity\n",
    }
    for name, content in contents.items():
        path = inputs / name
        if not path.exists():
            path.write_bytes(content)
    return ChampionRunAttestations(
        source_files=(
            ("dev_tasks", inputs / "dev_tasks.src"),
            ("train_tasks", inputs / "train_tasks.src"),
        ),
        split_manifest_file=inputs / "split.json",
        dictionary_files=(("baseline_leaf", inputs / "baseline.py"),),
        forecast_store=inputs / "forecast.store",
        proposal_model=proposer_binding.identity,
        proposal_config=proposer_binding.config,
        proposer_binding=proposer_binding,
        row_provider_binding=row_provider_binding,
        runtime_bindings=runtime_bindings,
        numeric_grid={"thresholds": (0.0, 1.0)},
        runtime_files=(("python_runtime", inputs / "runtime.py"),),
        forecast_runtime_identity={"provider": "deterministic-test-runtime"},
    )


def _lifecycle_controller(
    tmp_path, proposer, rows, *, authority_root=None
) -> ChampionEvolutionController:
    parts = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    config = _lifecycle_config(parts)
    if type(proposer) is ChampionProposerAdapter:
        proposer_binding = proposer
    else:
        raw_proposals = getattr(proposer, "proposals", _proposal_batch("better"))
        assert type(raw_proposals) is tuple
        proposer_binding = ChampionProposerAdapter.scripted(
            identity="deterministic-test-proposer",
            proposal_batches=(raw_proposals,),
            config={"temperature": 0.0},
        )
    unavailable_splits: tuple[str, ...] = ()
    if type(rows) is InterruptAtCalibration:
        row_source = rows.delegate
        unavailable_splits = ("calibration",)
    elif type(rows) in {LifecycleRows, ForbiddenRows}:
        row_source = rows if type(rows) is LifecycleRows else LifecycleRows()
    else:
        row_source = getattr(rows, "delegate", LifecycleRows())
    assert type(row_source) is LifecycleRows
    materialized_rows = (
        row_source(parts.build, "build")
        + row_source(parts.calibration, "calibration")
        + row_source(DEV_20, "dev")
    )
    row_provider_binding = ChampionRowProviderAdapter.materialized(
        identity="deterministic-test-row-provider",
        rows=materialized_rows,
        config={"contract": "materialized_rows_v1"},
        unavailable_splits=unavailable_splits,
    )
    runtime_bindings = ChampionRuntimeBindings.formal()
    attestations = _actual_attestations(
        tmp_path,
        proposer_binding,
        row_provider_binding,
        runtime_bindings,
    )
    resolved_authority_root = (
        authority_root
        if authority_root is not None
        else tmp_path.parent / f"{tmp_path.name}.authority"
    )
    authority_anchor = resolved_authority_root.parent / (
        f"{resolved_authority_root.name}.operator-identity"
    )
    if authority_anchor.exists():
        expected_authority_identity = authority_anchor.read_text(encoding="ascii")
        authority_store = ChampionAuthorityStore(
            resolved_authority_root,
            expected_authority_identity=expected_authority_identity,
        )
    else:
        expected_authority_identity = champion_fingerprint(
            {"operator_authority": str(resolved_authority_root.absolute())}
        )
        authority_anchor.write_text(expected_authority_identity, encoding="ascii")
        authority_store = ChampionAuthorityStore.provision(
            resolved_authority_root,
            authority_identity=expected_authority_identity,
        )
    return ChampionEvolutionController(
        manifest=_manifest(parts, config, attestations),
        config=config,
        attestations=attestations,
        proposer=proposer_binding,
        row_provider=row_provider_binding,
        runtime_bindings=runtime_bindings,
        artifact_store=ChampionArtifactStore(tmp_path),
        authority_store=authority_store,
    )


def test_internal_partition_is_exact_deterministic_and_entity_disjoint() -> None:
    first = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    second = partition_train_tasks(
        tuple(reversed(TRAIN_80)),
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )

    assert len(first.build) == 64
    assert len(first.calibration) == 16
    assert {task.entity_name for task in first.build}.isdisjoint(
        {task.entity_name for task in first.calibration}
    )
    assert tuple(task.task_id for task in first.build) == tuple(
        task.task_id for task in second.build
    )
    assert tuple(task.task_id for task in first.calibration) == tuple(
        task.task_id for task in second.calibration
    )


def test_partition_fails_when_entity_groups_cannot_make_exact_sizes() -> None:
    tasks = (
        _task("large_1", "large"),
        _task("large_2", "large"),
        _task("small_1", "small"),
        _task("small_2", "small"),
    )

    with pytest.raises(ChampionLifecycleError, match="entity-disjoint"):
        partition_train_tasks(tasks, build_size=3, calibration_size=1, seed=1)


@pytest.mark.parametrize(
    "tasks, message",
    (
        (
            (
                _task("duplicate", "first"),
                _task("duplicate", "second"),
            ),
            "duplicate task IDs",
        ),
        (
            (
                _task("first", "known"),
                _task("second", "unknown"),
            ),
            "unknown entity",
        ),
        (
            (
                _task("first", "Model_K"),
                _task("second", "Model_K"),
            ),
            "duplicate entity",
        ),
    ),
)
def test_partition_rejects_duplicate_and_unknown_identities(tasks, message) -> None:
    with pytest.raises(ChampionLifecycleError, match=message):
        partition_train_tasks(tasks, build_size=1, calibration_size=1, seed=1)


def test_manifest_fingerprint_binds_every_registered_input(tmp_path) -> None:
    parts = partition_train_tasks(
        TRAIN_80,
        build_size=64,
        calibration_size=16,
        seed=20260901,
    )
    config = _lifecycle_config(parts)
    manifest = _manifest(parts, config)
    variants = (
        replace(
            manifest,
            source_hashes=(
                ("dev_tasks", "b" * 64),
                ("train_tasks", "0" * 64),
            ),
        ),
        replace(
            manifest,
            train_tasks=tuple(reversed(manifest.train_tasks)),
            train_task_hashes=tuple(reversed(manifest.train_task_hashes)),
        ),
        replace(
            manifest,
            dev_tasks=tuple(reversed(manifest.dev_tasks)),
            dev_task_hashes=tuple(reversed(manifest.dev_task_hashes)),
        ),
        replace(manifest, split_manifest_fingerprint="0" * 64),
        replace(manifest, build_tasks=tuple(reversed(manifest.build_tasks))),
        replace(
            manifest,
            calibration_tasks=tuple(reversed(manifest.calibration_tasks)),
        ),
        replace(manifest, dictionary_hashes=(("baseline_leaf", "2" * 64),)),
        replace(manifest, forecast_store_fingerprint="0" * 64),
        replace(manifest, metric_policy_fingerprint="0" * 64),
        replace(manifest, proposal_model="different-model"),
        replace(manifest, proposal_config_fingerprint="0" * 64),
        replace(manifest, schedule_fingerprint="0" * 64),
        replace(manifest, numeric_grid_fingerprint="0" * 64),
        replace(manifest, candidate_minimum_gain=0.006),
        replace(manifest, research_target_gain=0.051),
        replace(manifest, runtime_fingerprint="0" * 64),
        replace(manifest, partition_seed=20260902),
    )

    assert all(variant.input_fingerprint != manifest.input_fingerprint for variant in variants)
    for index, variant in enumerate(variants):
        store = ChampionArtifactStore(tmp_path / f"variant_{index:02d}")
        store.bind_manifest(manifest)
        with pytest.raises(ChampionLifecycleError, match="manifest|drift"):
            store.bind_manifest(variant)


def test_checkpoint_keeps_exact_frozen_sanitized_feedback() -> None:
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    checkpoint = ChampionCheckpoint(
        schema_version=1,
        input_fingerprint="1" * 64,
        completed_stage="build_generation_1",
        active_parent=PARENT,
        proposal_archive=(),
        sanitized_feedback=feedback,
        build_evidence_fingerprint="2" * 64,
    )

    assert checkpoint.sanitized_feedback is feedback


def test_calibration_is_never_returned_to_the_proposer(tmp_path) -> None:
    proposer = SerializedRecordingProposer()

    outcome = _lifecycle_controller(tmp_path, proposer, LifecycleRows()).evolve(
        PARENT, TRAIN_80, DEV_20
    )

    assert outcome.release is not PARENT


def test_calibration_rejection_keeps_parent_and_never_reads_dev(tmp_path) -> None:
    class ExplodingSequence:
        def __iter__(self):
            raise AssertionError("Dev must remain unopened")

        def __len__(self):
            raise AssertionError("Dev must remain unopened")

    rows = LifecycleRows(calibration_error=3.0)
    outcome = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), rows).evolve(
        PARENT, TRAIN_80, ExplodingSequence()
    )

    assert outcome.release is PARENT
    assert outcome.dev_report is None


def test_dev_rejection_preserves_prior_release_bytes(tmp_path) -> None:
    rows = LifecycleRows(dev_error=3.0)
    outcome = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), rows).evolve(
        PARENT, TRAIN_80, DEV_20
    )

    assert outcome.release is PARENT
    assert canonical_release_bytes(outcome.release) == canonical_release_bytes(PARENT)
    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(PARENT)
    assert outcome.dev_report is not None


def test_manifest_drift_fails_before_task_or_model_execution(tmp_path) -> None:
    rows = ForbiddenRows()
    proposer = ForbiddenProposer()
    controller = _lifecycle_controller(tmp_path, proposer, rows)
    controller.artifact_store.bind_manifest(controller.manifest)
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    payload["runtime_fingerprint"] = "8" * 64
    (tmp_path / "run_manifest.json").write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ChampionLifecycleError, match="manifest|drift"):
        controller.evolve(PARENT, TRAIN_80, DEV_20)


def test_manifest_binds_attested_forecast_runtime_identity(tmp_path) -> None:
    controller = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows())

    assert (
        controller.manifest.forecast_runtime_identity_fingerprint
        == controller.attestations.forecast_runtime_identity_fingerprint
    )


def test_same_task_ids_with_changed_train_values_fail_before_callbacks(
    tmp_path,
) -> None:
    rows = ForbiddenRows()
    proposer = ForbiddenProposer()
    controller = _lifecycle_controller(tmp_path, proposer, rows)
    changed = list(TRAIN_80)
    changed[0] = replace(changed[0], history_values=(91.0, 92.0, 93.0))

    with pytest.raises(ChampionLifecycleError, match="task content|Train"):
        controller.evolve(PARENT, tuple(changed), DEV_20)


def test_actual_source_content_drift_fails_before_callbacks(tmp_path) -> None:
    rows = ForbiddenRows()
    proposer = ForbiddenProposer()
    controller = _lifecycle_controller(tmp_path, proposer, rows)
    source_path = controller.attestations.source_files[0][1]
    source_path.write_bytes(b"changed source bytes\n")

    with pytest.raises(ChampionLifecycleError, match="source|attestation|drift"):
        controller.evolve(PARENT, TRAIN_80, DEV_20)


def test_provider_failure_burns_calibration_across_fresh_controller(tmp_path) -> None:
    first_proposer = SerializedRecordingProposer()
    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(tmp_path, first_proposer, InterruptAtCalibration()).evolve(
            PARENT, TRAIN_80, DEV_20
        )

    resumed_proposer = SerializedRecordingProposer()
    resumed_provider = InterruptAtCalibration()
    with pytest.raises(ChampionLifecycleError, match="consumed"):
        _lifecycle_controller(tmp_path, resumed_proposer, resumed_provider).evolve(
            PARENT, TRAIN_80, DEV_20
        )


def test_new_run_root_cannot_reuse_complete_holdout_bundle(monkeypatch, tmp_path) -> None:
    first_root = tmp_path / "first_run"
    second_root = tmp_path / "second_run"
    authority_root = tmp_path / "operator_authority"
    evaluate = controller_module._evaluate_lifecycle_stage

    def interrupt_after_provider(*args, **kwargs):
        if kwargs.get("split") == "calibration":
            raise RuntimeError("crash after committed provider bundle")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(controller_module, "_evaluate_lifecycle_stage", interrupt_after_provider)
    with pytest.raises(RuntimeError, match="committed provider bundle"):
        _lifecycle_controller(
            first_root,
            SerializedRecordingProposer(),
            LifecycleRows(),
            authority_root=authority_root,
        ).evolve(PARENT, TRAIN_80, DEV_20)
    monkeypatch.setattr(controller_module, "_evaluate_lifecycle_stage", evaluate)

    second_rows = LifecycleRows()
    with pytest.raises(ChampionLifecycleError, match="consumed"):
        _lifecycle_controller(
            second_root,
            SerializedRecordingProposer(),
            second_rows,
            authority_root=authority_root,
        ).evolve(PARENT, TRAIN_80, DEV_20)


def test_foreign_checkpoint_fails_before_task_or_model_execution(tmp_path) -> None:
    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["input_fingerprint"] = "7" * 64
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    resumed_provider = InterruptAtCalibration()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="stale|foreign"):
        _lifecycle_controller(tmp_path, proposer, resumed_provider).evolve(
            PARENT, TRAIN_80, DEV_20
        )


def test_checkpoint_shortlist_must_match_immutable_build_attempt_evidence(
    monkeypatch, tmp_path
) -> None:
    original_claim = ChampionAuthorityStore.claim_or_resume_split

    def interrupt_before_calibration(*args, **kwargs):
        raise RuntimeError("stop after Build checkpoint")

    monkeypatch.setattr(
        ChampionAuthorityStore, "claim_or_resume_split", interrupt_before_calibration
    )
    with pytest.raises(RuntimeError, match="stop after Build"):
        _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
            PARENT, TRAIN_80, DEV_20
        )
    monkeypatch.setattr(ChampionAuthorityStore, "claim_or_resume_split", original_claim)

    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["proposal_archive"][0]["thresholds"][0][1] += 0.125
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="Build evidence|shortlist"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(PARENT, TRAIN_80, DEV_20)


def test_checkpoint_feedback_must_exactly_match_immutable_build_evidence(
    monkeypatch, tmp_path
) -> None:
    original_claim = ChampionAuthorityStore.claim_or_resume_split

    def interrupt_before_calibration(*args, **kwargs):
        raise RuntimeError("stop after Build checkpoint")

    monkeypatch.setattr(
        ChampionAuthorityStore, "claim_or_resume_split", interrupt_before_calibration
    )
    with pytest.raises(RuntimeError, match="stop after Build"):
        _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
            PARENT, TRAIN_80, DEV_20
        )
    monkeypatch.setattr(ChampionAuthorityStore, "claim_or_resume_split", original_claim)

    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    comparisons = checkpoint["sanitized_feedback"]["comparisons"]
    assert comparisons
    comparisons[0]["joint_improvement"] += 0.125
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="Build.*feedback|feedback.*Build"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(PARENT, TRAIN_80, DEV_20)


def test_duplicate_key_checkpoint_json_fails_before_callbacks(tmp_path) -> None:
    controller = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows())
    controller.artifact_store.bind_manifest(controller.manifest)
    controller.artifact_store.ensure_release(PARENT)
    (tmp_path / "checkpoint.json").write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="malformed JSON"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(PARENT, TRAIN_80, DEV_20)


def test_checkpoint_rejects_nested_cached_pass_authority_before_callbacks(
    tmp_path,
) -> None:
    with pytest.raises(ChampionLifecycleError, match="calibration row provider failed"):
        _lifecycle_controller(
            tmp_path, SerializedRecordingProposer(), InterruptAtCalibration()
        ).evolve(PARENT, TRAIN_80, DEV_20)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    comparisons = checkpoint["sanitized_feedback"]["comparisons"]
    assert comparisons
    comparisons[0]["passed"] = True
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint))
    resumed_provider = InterruptAtCalibration()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="feedback|malformed|forbidden"):
        _lifecycle_controller(tmp_path, proposer, resumed_provider).evolve(
            PARENT, TRAIN_80, DEV_20
        )


def test_stored_calibration_pass_cannot_open_dev(monkeypatch, tmp_path) -> None:
    evaluate = controller_module._evaluate_lifecycle_stage

    def interrupt_after_provider(*args, **kwargs):
        if kwargs.get("split") == "calibration":
            raise RuntimeError("simulated post-provider interruption")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(controller_module, "_evaluate_lifecycle_stage", interrupt_after_provider)
    with pytest.raises(RuntimeError, match="post-provider interruption"):
        _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
            PARENT, TRAIN_80, DEV_20
        )
    monkeypatch.setattr(controller_module, "_evaluate_lifecycle_stage", evaluate)
    monkeypatch.setattr(
        controller_module,
        "_fresh_comparisons",
        lambda states, gate, comparator=None: (),
    )
    rows = LifecycleRows()
    outcome = _lifecycle_controller(tmp_path, SerializedRecordingProposer(), rows).evolve(
        PARENT, TRAIN_80, DEV_20
    )

    assert outcome.calibration_report is not None
    assert all(
        candidate.comparison is not None and candidate.comparison.accepted
        for candidate in outcome.calibration_report.candidates
    )
    assert outcome.release is PARENT
    assert outcome.dev_report is None


def test_committed_calibration_evidence_resumes_without_provider_reread(
    monkeypatch, tmp_path
) -> None:
    fresh = controller_module._fresh_comparisons
    calls = 0

    def interrupt_before_boundary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("crash after committed evaluation")
        return fresh(*args, **kwargs)

    monkeypatch.setattr(controller_module, "_fresh_comparisons", interrupt_before_boundary)
    with pytest.raises(RuntimeError, match="committed evaluation"):
        _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
            PARENT, TRAIN_80, DEV_20
        )
    monkeypatch.setattr(controller_module, "_fresh_comparisons", fresh)

    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()
    outcome = _lifecycle_controller(tmp_path, proposer, rows).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.release is not PARENT


def test_stored_dev_pass_cannot_publish_release(monkeypatch, tmp_path) -> None:
    fresh = controller_module._fresh_comparisons
    boundary_calls = 0

    def reject_only_dev(states, gate, comparator=None):
        nonlocal boundary_calls
        boundary_calls += 1
        result = fresh(states, gate, comparator)
        return () if boundary_calls == 2 else result

    monkeypatch.setattr(controller_module, "_fresh_comparisons", reject_only_dev)
    outcome = _lifecycle_controller(
        tmp_path, SerializedRecordingProposer(), LifecycleRows()
    ).evolve(PARENT, TRAIN_80, DEV_20)

    assert outcome.dev_report is not None
    comparison = outcome.dev_report.candidates[0].comparison
    assert comparison is not None and comparison.accepted
    assert outcome.release is PARENT
    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(PARENT)


def test_release_rejects_normalized_lineage_replay(tmp_path) -> None:
    parent = replace(PARENT, lineage=("active_parent", "Model_K"))
    collision_proposals = list(_proposal_batch("better"))
    collision_proposals[0] = replace(collision_proposals[0], name="Model_K")

    @dataclass(frozen=True)
    class LineageCollisionProposer:
        proposals: tuple[ChampionRecipe, ...]

        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            return self.proposals

    controller = _lifecycle_controller(
        tmp_path,
        LineageCollisionProposer(tuple(collision_proposals)),
        LifecycleRows(),
    )

    with pytest.raises(ChampionLifecycleError, match="lineage|replayed"):
        controller.evolve(parent, TRAIN_80, DEV_20)

    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(parent)


def test_existing_parent_lineage_rejects_normalized_duplicate_identities(
    tmp_path,
) -> None:
    parent = replace(PARENT, lineage=("Model_K", "Model_K"))
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="lineage|duplicate"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(parent, TRAIN_80, DEV_20)


def test_completed_lifecycle_reports_cannot_be_replayed(tmp_path) -> None:
    _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
        PARENT, TRAIN_80, DEV_20
    )
    rows = LifecycleRows()
    proposer = SerializedRecordingProposer()

    with pytest.raises(ChampionLifecycleError, match="overwritten|replayed"):
        _lifecycle_controller(tmp_path, proposer, rows).evolve(PARENT, TRAIN_80, DEV_20)


def test_hostile_calibration_callback_is_rejected_before_release_mutation(
    tmp_path,
) -> None:
    tampered = replace(PARENT, lineage=("active_parent", "tampered"))
    tampered_bytes = canonical_release_bytes(tampered)

    @dataclass(frozen=True)
    class ReleaseTamperingRows:
        delegate: LifecycleRows
        release_path: Path
        replacement_bytes: bytes

        def __call__(self, tasks, split):
            result = self.delegate(tasks, split)
            if split == "calibration":
                self.release_path.write_bytes(self.replacement_bytes)
            return result

    rows = ReleaseTamperingRows(
        LifecycleRows(),
        tmp_path / "champion_release.json",
        tampered_bytes,
    )
    rows.release_path.write_bytes(canonical_release_bytes(PARENT))

    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        ChampionRowProviderAdapter.bind(
            identity="hostile-row-provider",
            callback=rows,
            config={},
        )

    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(PARENT)


def test_hostile_proposer_callback_is_rejected_before_release_mutation(
    tmp_path,
) -> None:
    tampered = replace(PARENT, lineage=("active_parent", "tampered"))
    tampered_bytes = canonical_release_bytes(tampered)

    @dataclass(frozen=True)
    class ReleaseTamperingProposer:
        delegate: SerializedRecordingProposer
        release_path: Path
        replacement_bytes: bytes

        def __call__(self, parent, evidence):
            result = self.delegate(parent, evidence)
            self.release_path.write_bytes(self.replacement_bytes)
            return result

    proposer = ReleaseTamperingProposer(
        SerializedRecordingProposer(),
        tmp_path / "champion_release.json",
        tampered_bytes,
    )
    proposer.release_path.write_bytes(canonical_release_bytes(PARENT))
    with pytest.raises(ChampionLifecycleError, match="closed|adapter|registered"):
        ChampionProposerAdapter.bind(
            identity="hostile-proposer",
            callback=proposer,
            config={},
        )

    assert (tmp_path / "champion_release.json").read_bytes() == canonical_release_bytes(PARENT)


def test_artifact_store_rejects_symlinked_ancestor(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="symlink|alias"):
        ChampionArtifactStore(alias / "run")


def test_artifact_store_rejects_broken_symlinked_ancestor(tmp_path) -> None:
    alias = tmp_path / "broken_alias"
    alias.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="symlink|alias"):
        ChampionArtifactStore(alias / "run")


def test_artifact_store_root_swap_fails_without_writing_outside_pinned_root(
    tmp_path,
) -> None:
    root = tmp_path / "run"
    store = ChampionArtifactStore(root)
    store.ensure_release(PARENT)
    held = tmp_path / "held_run"
    root.rename(held)
    outside = tmp_path / "outside"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="pinned|drift|alias"):
        store.ensure_release(PARENT)

    assert tuple(outside.iterdir()) == ()


def test_artifact_store_rejects_ancestor_swap_even_when_final_inode_is_same(
    tmp_path,
) -> None:
    ancestor = tmp_path / "operator"
    root = ancestor / "run"
    store = ChampionArtifactStore(root)
    store.ensure_release(PARENT)
    prior_bytes = canonical_release_bytes(PARENT)
    held_ancestor = tmp_path / "held_operator"
    ancestor.rename(held_ancestor)
    ancestor.symlink_to(held_ancestor, target_is_directory=True)
    next_release = replace(PARENT, lineage=("active_parent", "next_release"))

    with pytest.raises(ChampionLifecycleError, match="pinned|drift|alias"):
        store.publish_release(next_release)

    assert (held_ancestor / "run" / "champion_release.json").read_bytes() == (prior_bytes)


def test_artifact_store_rejects_hardlinked_manifest(tmp_path) -> None:
    root = tmp_path / "run"
    controller = _lifecycle_controller(root, SerializedRecordingProposer(), LifecycleRows())
    path = controller.artifact_store.bind_manifest(controller.manifest)
    os.link(path, tmp_path / "foreign_manifest.json")

    with pytest.raises(ChampionLifecycleError, match="alias|immutable"):
        controller.artifact_store.bind_manifest(controller.manifest)


def test_artifact_store_rejects_hardlinked_release_archive(tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path / "run")
    store.ensure_release(PARENT)
    archive = store.root / "releases" / f"{champion_fingerprint(PARENT)}.json"
    os.link(archive, tmp_path / "foreign_release.json")

    with pytest.raises(ChampionLifecycleError, match="alias|immutable"):
        store.publish_release(PARENT)


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_release_alias_is_unlinked_before_exact_parent_restoration(tmp_path, alias_kind) -> None:
    root = tmp_path / "run"
    store = ChampionArtifactStore(root)
    store.ensure_release(PARENT)
    current = root / "champion_release.json"
    external = tmp_path / "external.json"
    external_bytes = b'{"hostile":"outside"}\n'
    external.write_bytes(external_bytes)
    current.unlink()
    if alias_kind == "symlink":
        current.symlink_to(external)
    else:
        os.link(external, current)

    with pytest.raises(ChampionLifecycleError, match="drift|alias|Parent"):
        store.ensure_release(PARENT)

    assert not current.is_symlink()
    assert current.read_bytes() == canonical_release_bytes(PARENT)
    assert external.read_bytes() == external_bytes


def test_release_restoration_stages_replacement_before_touching_alias(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "run"
    store = ChampionArtifactStore(root)
    store.ensure_release(PARENT)
    current = root / "champion_release.json"
    external = tmp_path / "external.json"
    external_bytes = b'{"hostile":"outside"}\n'
    external.write_bytes(external_bytes)
    current.unlink()
    current.symlink_to(external)

    def fail_temporary_write(name, content):
        raise OSError("injected restoration temp-write failure")

    monkeypatch.setattr(store, "_write_temporary", fail_temporary_write)
    with pytest.raises(OSError, match="temp-write failure"):
        store.ensure_release(PARENT)

    assert os.path.lexists(current)
    assert current.is_symlink()
    assert external.read_bytes() == external_bytes


def test_artifact_store_rejects_nonfinite_json(tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path)
    (tmp_path / "checkpoint.json").write_text(
        '{"schema_version":1,"score":1e999}\n', encoding="utf-8"
    )

    with pytest.raises(ChampionLifecycleError, match="nonfinite"):
        store.load_checkpoint()


def test_pinned_stores_are_idempotent_context_managers(tmp_path) -> None:
    with ChampionArtifactStore(tmp_path / "run") as artifact_store:
        artifact_descriptors = tuple(
            entry[1]
            for entry in (artifact_store._root_chain + artifact_store._release_archive._root_chain)
        )
        for descriptor in artifact_descriptors:
            os.fstat(descriptor)

    artifact_store.close()
    for descriptor in artifact_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)

    authority_root = tmp_path / "authority"
    identity = "9" * 64
    provisioned = ChampionAuthorityStore.provision(authority_root, authority_identity=identity)
    provisioned.close()
    with ChampionAuthorityStore(
        authority_root, expected_authority_identity=identity
    ) as authority_store:
        authority_descriptors = tuple(entry[1] for entry in authority_store._root_chain)
    authority_store.close()
    for descriptor in authority_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_failed_artifact_store_construction_closes_all_pinned_fds(tmp_path) -> None:
    root = tmp_path / "malformed_run"
    root.mkdir()
    (root / "release_prepare.json").write_text("{}\n", encoding="utf-8")
    gc.collect()
    before = len(os.listdir("/dev/fd"))

    for _ in range(12):
        with pytest.raises(ChampionLifecycleError, match="publication record"):
            ChampionArtifactStore(root)

    gc.collect()
    assert len(os.listdir("/dev/fd")) == before


def test_failed_authority_store_construction_closes_all_pinned_fds(tmp_path) -> None:
    root = tmp_path / "malformed_authority"
    root.mkdir()
    gc.collect()
    before = len(os.listdir("/dev/fd"))

    for _ in range(12):
        with pytest.raises(ChampionLifecycleError, match="provisioned"):
            ChampionAuthorityStore(root, expected_authority_identity="8" * 64)

    gc.collect()
    assert len(os.listdir("/dev/fd")) == before


def test_partial_ancestry_pin_failure_closes_opened_fds(tmp_path) -> None:
    real_root = tmp_path / "real_run"
    real_root.mkdir()
    alias_root = tmp_path / "aliased_run"
    alias_root.symlink_to(real_root, target_is_directory=True)
    gc.collect()
    before = len(os.listdir("/dev/fd"))

    for _ in range(12):
        with pytest.raises(ChampionLifecycleError, match="symlink|alias"):
            ChampionArtifactStore(alias_root)

    gc.collect()
    assert len(os.listdir("/dev/fd")) == before


def test_split_consumption_lease_is_operator_owned_and_burns_on_failure(
    tmp_path,
) -> None:
    authority_root = tmp_path / "operator_authority"
    expected_identity = "6" * 64
    first = ChampionAuthorityStore.provision(authority_root, authority_identity=expected_identity)
    lease = first.acquire_split(
        role="calibration",
        split_manifest_fingerprint="a" * 64,
        task_content_fingerprint="b" * 64,
        run_input_fingerprint="c" * 64,
    )

    assert lease.role == "calibration"
    assert lease.split_key == champion_fingerprint(
        {
            "role": "calibration",
            "task_content_fingerprint": "b" * 64,
        }
    )
    assert tuple(authority_root.glob("*.lease.json"))

    # Deleting/replacing a run directory cannot reopen this operator-owned split.
    (tmp_path / "fresh_run").mkdir()
    reopened = ChampionAuthorityStore(
        authority_root, expected_authority_identity=expected_identity
    )
    with pytest.raises(ChampionLifecycleError, match="consumed"):
        reopened.acquire_split(
            role="calibration",
            split_manifest_fingerprint="a" * 64,
            task_content_fingerprint="b" * 64,
            run_input_fingerprint="d" * 64,
        )
    with pytest.raises(ChampionLifecycleError, match="consumed"):
        reopened.acquire_split(
            role="calibration",
            split_manifest_fingerprint="a" * 64,
            task_content_fingerprint="b" * 64,
            run_input_fingerprint="c" * 64,
        )


def test_deleted_authority_root_cannot_be_silently_reprovisioned(tmp_path) -> None:
    root = tmp_path / "operator_authority"
    expected_identity = "7" * 64
    ChampionAuthorityStore.provision(root, authority_identity=expected_identity)
    root.rename(tmp_path / "deleted_authority_snapshot")

    with pytest.raises(ChampionLifecycleError, match="missing|provision|authority"):
        ChampionAuthorityStore(root, expected_authority_identity=expected_identity)

    assert not root.exists()


def test_split_manifest_declaration_cannot_namespace_identical_holdout(
    tmp_path,
) -> None:
    root = tmp_path / "operator_authority"
    identity = "4" * 64
    authority = ChampionAuthorityStore.provision(root, authority_identity=identity)
    authority.acquire_split(
        role="calibration",
        split_manifest_fingerprint="a" * 64,
        task_content_fingerprint="b" * 64,
        run_input_fingerprint="c" * 64,
    )
    reopened = ChampionAuthorityStore(root, expected_authority_identity=identity)

    with pytest.raises(ChampionLifecycleError, match="consumed"):
        reopened.acquire_split(
            role="calibration",
            split_manifest_fingerprint="d" * 64,
            task_content_fingerprint="b" * 64,
            run_input_fingerprint="e" * 64,
        )


def test_authority_root_swap_fails_without_writing_to_replacement(tmp_path) -> None:
    root = tmp_path / "authority"
    authority = ChampionAuthorityStore.provision(root, authority_identity="5" * 64)
    held = tmp_path / "held_authority"
    root.rename(held)
    outside = tmp_path / "outside_authority"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ChampionLifecycleError, match="pinned|drift"):
        authority.acquire_split(
            role="dev",
            split_manifest_fingerprint="a" * 64,
            task_content_fingerprint="b" * 64,
            run_input_fingerprint="c" * 64,
        )

    assert tuple(outside.iterdir()) == ()


def test_authority_root_must_be_outside_run_and_attested_inputs(tmp_path) -> None:
    with pytest.raises(ChampionLifecycleError, match="outside|non-aliasing"):
        _lifecycle_controller(
            tmp_path,
            SerializedRecordingProposer(),
            LifecycleRows(),
            authority_root=tmp_path / "authority",
        )


def test_lifecycle_artifacts_are_exact_canonical_json(tmp_path) -> None:
    _lifecycle_controller(tmp_path, SerializedRecordingProposer(), LifecycleRows()).evolve(
        PARENT, TRAIN_80, DEV_20
    )

    artifacts = tuple(sorted(tmp_path.rglob("*.json")))
    assert artifacts
    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert artifact.read_bytes() == canonical_json_bytes(payload)


@pytest.mark.parametrize(
    "failure_phase",
    ("before_prepare", "after_prepare", "after_current_replace", "after_commit"),
)
def test_interrupted_publication_preserves_last_release(
    monkeypatch, tmp_path, failure_phase
) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    prior_bytes = (tmp_path / "champion_release.json").read_bytes()
    next_release = replace(
        PARENT,
        lineage=("active_parent", "next_release"),
    )

    def interrupted(phase):
        if phase == failure_phase:
            raise OSError("simulated publication interruption")

    monkeypatch.setattr(store, "_publication_hook", interrupted)

    with pytest.raises(OSError, match="publication interruption"):
        store.publish_release(next_release)

    reopened = ChampionArtifactStore(tmp_path)
    assert (tmp_path / "champion_release.json").read_bytes() == prior_bytes
    reopened.ensure_release(PARENT)


def test_fresh_process_recovery_rolls_back_uncommitted_release_pointer(
    tmp_path,
) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    prior_bytes = canonical_release_bytes(PARENT)
    next_release = replace(PARENT, lineage=("active_parent", "next_release"))
    next_id = champion_fingerprint(next_release)
    prior_id = champion_fingerprint(PARENT)
    archive = store.root / "releases" / f"{next_id}.json"
    store._atomic_create(archive, next_release.to_payload())
    transaction_id = champion_fingerprint(
        {
            "prior_release_id": prior_id,
            "next_release_id": next_id,
            "accepted": False,
        }
    )
    store._atomic_create_name(
        "release_prepare.json",
        {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "prior_release_id": prior_id,
            "next_release_id": next_id,
            "accepted": False,
        },
    )
    store._atomic_replace(
        store.root / "champion_release.json",
        canonical_release_bytes(next_release),
    )

    reopened = ChampionArtifactStore(tmp_path)

    assert (tmp_path / "champion_release.json").read_bytes() == prior_bytes
    reopened.ensure_release(PARENT)


@pytest.mark.parametrize("surviving_markers", ("prepare", "commit", "both"))
def test_accepted_publication_authority_survives_partial_marker_cleanup(
    tmp_path, surviving_markers
) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    next_release = replace(PARENT, lineage=("active_parent", "next_release"))
    store.publish_release(next_release)
    prior_id = champion_fingerprint(PARENT)
    next_id = champion_fingerprint(next_release)
    transaction_id = champion_fingerprint(
        {
            "prior_release_id": prior_id,
            "next_release_id": next_id,
            "accepted": True,
        }
    )
    record = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "prior_release_id": prior_id,
        "next_release_id": next_id,
        "accepted": True,
    }
    if surviving_markers in {"prepare", "both"}:
        store._atomic_create_name("release_prepare.json", record)
    if surviving_markers in {"commit", "both"}:
        store._atomic_create_name("release_commit.json", record)
    store._atomic_replace(store.root / "champion_release.json", canonical_release_bytes(PARENT))

    reopened = ChampionArtifactStore(tmp_path)

    assert (tmp_path / "champion_release.json").read_bytes() == (
        canonical_release_bytes(next_release)
    )
    assert reopened.has_accepted_release()
    assert not (tmp_path / "release_prepare.json").exists()
    assert not (tmp_path / "release_commit.json").exists()


def test_accepted_marker_cleanup_is_durable_before_success(monkeypatch, tmp_path) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    release_id = champion_fingerprint(PARENT)
    transaction_id = champion_fingerprint(
        {
            "prior_release_id": None,
            "next_release_id": release_id,
            "accepted": True,
        }
    )
    sync_root = store._sync_root
    calls = 0

    def interrupt_cleanup_sync():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated accepted-marker cleanup interruption")
        sync_root()

    monkeypatch.setattr(store, "_sync_root", interrupt_cleanup_sync)
    with pytest.raises(OSError, match="cleanup interruption"):
        store._atomic_create_name(
            "last_accepted_release.json",
            {
                "schema_version": 1,
                "transaction_id": transaction_id,
                "release_id": release_id,
            },
        )

    marker = tmp_path / "last_accepted_release.json"
    assert marker.stat().st_nlink == 1
    assert not tuple(tmp_path.glob(".last_accepted_release.json.*.tmp"))
    monkeypatch.setattr(store, "_sync_root", sync_root)
    reopened = ChampionArtifactStore(tmp_path)
    assert reopened.has_accepted_release()


@pytest.mark.parametrize("failing_fsync", (1, 2, 3))
def test_publication_directory_fsync_failure_recovers_exact_prior_release(
    monkeypatch, tmp_path, failing_fsync
) -> None:
    store = ChampionArtifactStore(tmp_path)
    store.ensure_release(PARENT)
    prior_bytes = canonical_release_bytes(PARENT)
    next_release = replace(PARENT, lineage=("active_parent", "next_release"))
    sync_root = store._sync_root
    calls = 0

    def fail_one_sync():
        nonlocal calls
        calls += 1
        if calls == failing_fsync:
            raise OSError("simulated directory fsync failure")
        sync_root()

    monkeypatch.setattr(store, "_sync_root", fail_one_sync)
    with pytest.raises(OSError, match="directory fsync failure"):
        store.publish_release(next_release)

    reopened = ChampionArtifactStore(tmp_path)
    assert (tmp_path / "champion_release.json").read_bytes() == prior_bytes
    assert not reopened.has_accepted_release()


def test_successive_halving_runs_fixed_8_32_64_schedule() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.generations[0].stage_counts == (8, 32, 64)
    assert result.generations[0].full_build_children <= 3


def test_materialized_child_uses_safe_fallback_when_specialist_is_unavailable() -> None:
    rows = tuple(
        replace(row, forecast=None, failure_reason="NotApplicable")
        if row.candidate_name == "better_a"
        else row
        for row in _rows(1, enriched=True)
    )
    recipe = ChampionRecipe(
        name="safe_specialist_route",
        kind="route",
        parents=("baseline_leaf", "better_a"),
        fallback_parent="baseline_leaf",
        assumptions=(
            EvolutionAssumption(
                assumption_id="specialist_history",
                candidate_name="better_a",
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator="route",
                rationale="History length supports the specialist.",
                failure_condition="History length stops supporting the specialist.",
            ),
        ),
    )
    policy = FittedChampionPolicy(
        recipe=recipe,
        thresholds=(("specialist_history", 0.0),),
    )

    materialized = controller_module._materialize_policy(
        policy,
        "safe_specialist_child",
        rows,
        ("build_case_000",),
    )
    fallback = next(row for row in rows if row.candidate_name == "baseline_leaf")

    assert materialized[0].forecast == fallback.forecast
    assert materialized[0].failure_reason is None


def test_stage_uses_active_parent_when_child_fallback_is_unavailable() -> None:
    rows = tuple(
        replace(row, forecast=None, failure_reason="NotApplicable")
        if row.candidate_name == "better_a"
        else row
        for row in _rows(8, enriched=True)
    )
    recipe = ChampionRecipe(
        name="specialist_pair_without_anchor",
        kind="route",
        parents=("better_a", "better_b"),
        fallback_parent="better_a",
        assumptions=(
            EvolutionAssumption(
                assumption_id="route_to_b",
                candidate_name="better_b",
                feature="history_length",
                direction="above",
                horizon_region="full",
                operator="route",
                rationale="History length supports the specialist.",
                failure_condition="History length stops supporting the specialist.",
            ),
        ),
    )
    policy = FittedChampionPolicy(
        recipe=recipe,
        thresholds=(("route_to_b", 0.0),),
    )
    state = controller_module._AttemptState(
        fitted_id=champion_fingerprint(policy),
        score_name="outer_anchor_child",
        policy=policy,
    )
    task_ids = tuple(f"build_case_{index:03d}" for index in range(8))

    evaluated = controller_module._evaluate_stage(
        state,
        parent_recipe=PARENT.policy.recipe,
        parent_policy=PARENT.policy,
        rows=rows,
        task_ids=task_ids,
        gate=ChampionGateConfig(minimum_improved_folds=0),
    )
    child_rows = tuple(
        row for row in evaluated.evidence_rows if row.candidate_name == state.score_name
    )
    parent_by_task = {
        row.task_id: row.forecast
        for row in rows
        if row.candidate_name == "baseline_leaf"
    }

    assert len(child_rows) == 8
    assert all(row.forecast == parent_by_task[row.task_id] for row in child_rows)
    assert all(row.failure_reason is None for row in child_rows)
    assert evaluated.comparison is not None
    assert evaluated.comparison.child_coverage == 1.0
    assert evaluated.comparison.child_failure_rate == 0.0


def test_outer_parent_fallback_does_not_hide_invalid_child_execution() -> None:
    child = replace(
        next(row for row in _rows(1) if row.candidate_name == "better_a"),
        candidate_name="invalid_child",
        forecast=None,
        failure_reason="invalid_policy_materialization",
    )
    parent = next(row for row in _rows(1) if row.candidate_name == "baseline_leaf")

    result = controller_module._apply_outer_parent_fallback((child,), (parent,))

    assert result == (child,)


def test_provisional_halving_allows_local_gain_but_full_build_keeps_fold_gate() -> None:
    def localized_rows(improved_indices: set[int]) -> tuple[ChampionTaskRow, ...]:
        baseline = {
            row.task_id: row.forecast
            for row in ROWS_64
            if row.candidate_name == "baseline_leaf"
        }
        localized: list[ChampionTaskRow] = []
        for row in ROWS_64:
            index = int(row.task_id.rsplit("_", 1)[1])
            if row.candidate_name.startswith("better_") and index not in improved_indices:
                localized.append(replace(row, forecast=baseline[row.task_id]))
            else:
                localized.append(row)
        return tuple(localized)

    strict_config = replace(
        _config(),
        gate_config=ChampionGateConfig(minimum_improved_folds=4),
    )
    enough_full_folds = run_build_evolution(
        PARENT,
        localized_rows({0, 11, 12, 13}),
        RecordingProposer("better"),
        strict_config,
    )
    only_one_full_fold = run_build_evolution(
        PARENT,
        localized_rows({0}),
        RecordingProposer("better"),
        strict_config,
    )

    assert enough_full_folds.shortlist
    assert any(
        attempt.stage_task_counts == (8, 32, 64)
        for attempt in enough_full_folds.generations[0].attempts
    )
    assert only_one_full_fold.shortlist == ()


def test_rejected_child_feedback_does_not_change_the_mutation_parent() -> None:
    parent = _parent()
    proposer = RecordingProposer("worse")
    global PARENT
    previous = PARENT
    PARENT = parent
    try:
        result = run_build_evolution(parent, ROWS_64, proposer, _config(generations=2))
    finally:
        PARENT = previous

    assert all(
        generation.mutation_parent_sha256 == champion_fingerprint(parent)
        for generation in result.generations
    )
    assert proposer.calls == 2
    assert result.active_parent is parent


def test_build_feedback_is_labeled_non_independent() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())

    assert result.score_label == "adaptive_train_build_diagnostic"
    assert result.independent_generalization_claim is False
    assert all(
        attempt.score_label == "adaptive_train_build_diagnostic"
        and attempt.independent_generalization_claim is False
        for attempt in result.generations[0].attempts
    )
    recipe_count = len(
        {
            champion_fingerprint(attempt.policy.recipe)
            for attempt in result.generations[0].attempts
        }
    )
    assert len(result.generations[0].feedback.comparisons) == recipe_count


def test_build_feedback_is_bounded_by_proposed_recipe_not_numeric_expansions() -> None:
    class FeedbackRecorder:
        def __init__(self) -> None:
            self.feedback: list[ProposerEvidence] = []

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            self.feedback.append(evidence)
            return _proposal_batch("better", len(self.feedback) - 1)

    proposer = FeedbackRecorder()
    result = run_build_evolution(PARENT, ROWS_64, proposer, _config(generations=2))

    first = result.generations[0]
    assert len(first.attempts) > 5
    assert len(first.feedback.comparisons) == 5
    assert len(first.feedback.structures) == 5
    assert {item.candidate_name for item in first.feedback.structures} == {
        item.candidate_name for item in first.feedback.comparisons
    }
    assert all(item.stage_support in {8, 32, 64} for item in first.feedback.structures)
    attempted_recipes = {
        champion_fingerprint(attempt.policy.recipe): attempt.policy.recipe
        for attempt in first.attempts
    }
    for item in first.feedback.structures:
        assert item.recipe_sha256 in attempted_recipes
        recipe = attempted_recipes[item.recipe_sha256]
        assert item.kind == recipe.kind
        assert item.parents == recipe.parents
        assert item.fallback_parent == recipe.fallback_parent
        assert tuple(
            (
                assumption.candidate_name,
                assumption.feature,
                assumption.direction,
                assumption.horizon_region,
                assumption.operator,
            )
            for assumption in recipe.assumptions
        ) == tuple(
            (
                assumption.candidate_name,
                assumption.feature,
                assumption.direction,
                assumption.horizon_region,
                assumption.operator,
            )
            for assumption in item.assumptions
        )
    assert len(proposer.feedback[1].comparisons) == 5
    assert proposer.feedback[1].structures == first.feedback.structures


def test_first_screen_keeps_low_middle_high_tied_threshold_representatives() -> None:
    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), _config())
    attempts_by_recipe: dict[str, list[object]] = {}
    for attempt in result.generations[0].attempts:
        attempts_by_recipe.setdefault(
            champion_fingerprint(attempt.policy.recipe), []
        ).append(attempt)

    preserved = False
    for attempts in attempts_by_recipe.values():
        all_thresholds = sorted(
            attempt.policy.thresholds[0][1] for attempt in attempts
        )
        next_screen = sorted(
            attempt.policy.thresholds[0][1]
            for attempt in attempts
            if attempt.stage_task_counts[:2] == (8, 32)
        )
        if len(next_screen) != 3:
            continue
        assert next_screen == [
            all_thresholds[0],
            all_thresholds[len(all_thresholds) // 2],
            all_thresholds[-1],
        ]
        preserved = True
        break

    assert preserved


def test_threshold_defaults_and_membership_are_fingerprinted() -> None:
    default = ChampionEvolutionConfig()
    candidate_changed = replace(default, candidate_minimum_gain=0.006)
    target_changed = replace(default, research_target_gain=0.051)

    assert default.candidate_minimum_gain == 0.005
    assert default.research_target_gain == 0.05
    assert default.fingerprint != candidate_changed.fingerprint
    assert default.fingerprint != target_changed.fingerprint
    assert _config().fingerprint != _config().screen_membership_sha256


def test_candidate_threshold_gates_but_research_target_does_not() -> None:
    high_target = replace(_config(), research_target_gain=0.99)
    high_candidate = replace(_config(), candidate_minimum_gain=0.99, research_target_gain=1.0)

    target_result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), high_target)
    candidate_result = run_build_evolution(
        PARENT, ROWS_64, RecordingProposer("better"), high_candidate
    )

    assert target_result.shortlist
    assert candidate_result.shortlist == ()


def test_halving_preserves_distinct_recipe_kinds_before_a_second_kind() -> None:
    result = run_build_evolution(PARENT, ROWS_64, DiverseProposer(), _config())

    kinds = tuple(policy.recipe.kind for policy in result.shortlist)
    assert len(kinds) == 3
    assert set(kinds) == {"select", "weighted", "median"}


def test_controller_rejects_normalized_duplicate_policy_ids() -> None:
    class DuplicatePolicyProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            names = ("Model_K", "Model_K", "unique_two", "unique_three", "unique_four")
            return tuple(
                ChampionRecipe(
                    name=name,
                    kind="select",
                    parents=("better_a",),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"unique_assumption_{index}",
                            candidate_name="better_a",
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, name in enumerate(names)
            )

    with pytest.raises(ChampionControllerError, match="duplicate policy"):
        run_build_evolution(PARENT, ROWS_64, DuplicatePolicyProposer(), _config())


def test_hostile_callback_cannot_leave_the_exact_parent_mutated() -> None:
    parent = _parent()
    original_sha256 = champion_fingerprint(parent)

    class MutatingProposer:
        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            assert received is parent
            object.__setattr__(received.policy.recipe, "name", "tampered_parent")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="mutate the active Parent"):
        run_build_evolution(parent, ROWS_64, MutatingProposer(), _config())

    assert parent.policy.recipe.name == "active_parent"
    assert champion_fingerprint(parent) == original_sha256


def test_win_tie_loss_is_diagnostic_and_never_a_standalone_gate() -> None:
    mixed_rows = list(ROWS_64)
    for row in ROWS_64:
        if row.candidate_name != "baseline_leaf":
            continue
        assert row.truth is not None
        index = int(row.task_id.rsplit("_", 1)[1])
        error = 0.0 if index < 4 else 2.05
        mixed_rows.append(
            replace(
                row,
                candidate_name="mixed_leaf",
                forecast=tuple(value + error for value in row.truth),
            )
        )

    class MixedProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            candidates = ("mixed_leaf", "worse_b", "worse_c", "worse_d", "worse_e")
            return tuple(
                ChampionRecipe(
                    name=f"mixed_batch_{index}",
                    kind="select",
                    parents=(candidate,),
                    fallback_parent=candidate,
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"mixed_history_{index}",
                            candidate_name=candidate,
                            feature="history_length",
                            direction="above",
                            horizon_region="full",
                            operator="select",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index, candidate in enumerate(candidates)
            )

    gate = ChampionGateConfig(
        tail_regression_tolerance=1.0,
        maximum_task_regret_smae=1.0,
        maximum_task_regret_srmse=1.0,
        minimum_improved_folds=0,
    )
    result = run_build_evolution(
        PARENT,
        tuple(mixed_rows),
        MixedProposer(),
        replace(_config(), gate_config=gate),
    )

    full = next(
        attempt
        for attempt in result.generations[0].attempts
        if attempt.policy.recipe.parents == ("mixed_leaf",) and attempt.stage_task_counts[-1] == 64
    )
    assert full.comparison is not None
    assert full.comparison.wtl.losses > full.comparison.wtl.wins
    assert result.shortlist


def test_malformed_proposer_output_fails_closed() -> None:
    class MalformedProposer:
        def __call__(self, parent, evidence) -> object:
            return {"recipes": []}

    with pytest.raises(ChampionControllerError, match="exact tuple or list"):
        run_build_evolution(PARENT, ROWS_64, MalformedProposer(), _config())


def test_row_universe_and_split_drift_fail_before_proposer_execution() -> None:
    rows = list(ROWS_64)
    rows[0] = replace(rows[0], split="calibration")
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="Build rows only"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_hostile_callback_cannot_mutate_sanitized_feedback() -> None:
    class FeedbackMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(evidence, "label", "tampered")
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="sanitized Build feedback"):
        run_build_evolution(PARENT, ROWS_64, FeedbackMutator(), _config())


def test_unscorable_child_is_typed_pruned_without_aborting_valid_siblings() -> None:
    class MixedValidityProposer:
        calls = 0
        feedback: list[ProposerEvidence] = []

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            self.feedback.append(evidence)
            generation = self.calls
            self.calls += 1
            valid = _proposal_batch("better", generation)[:1]
            invalid = tuple(
                ChampionRecipe(
                    name=f"overlay_{generation}_{index}",
                    kind="bounded_overlay",
                    parents=("better_a", "better_b"),
                    fallback_parent="better_a",
                    assumptions=(
                        EvolutionAssumption(
                            assumption_id=f"overlay_history_{generation}_{index}",
                            candidate_name="better_b",
                            feature="periodicity_confidence",
                            direction="above",
                            horizon_region="full",
                            operator="bounded_overlay",
                            rationale="History length supports the candidate forecast.",
                            failure_condition=(
                                "History length stops supporting the candidate forecast."
                            ),
                        ),
                    ),
                )
                for index in range(4)
            )
            return valid + invalid

    proposer = MixedValidityProposer()
    result = run_build_evolution(PARENT, ROWS_64, proposer, _config(generations=2))

    invalid = tuple(
        attempt for attempt in result.generations[0].attempts if attempt.status == "invalid"
    )
    assert invalid
    assert all(attempt.comparison is None for attempt in invalid)
    assert all(attempt.invalid_reason == "unscorable_child" for attempt in invalid)
    generation_feedback = result.generations[0].feedback
    valid_count = len(
        {
            champion_fingerprint(attempt.policy.recipe)
            for attempt in result.generations[0].attempts
            if attempt.status != "invalid"
        }
    )
    invalid_count = len(
        {champion_fingerprint(attempt.policy.recipe) for attempt in invalid}
    )
    assert len(generation_feedback.comparisons) == valid_count
    assert len(generation_feedback.invalid_attempts) == invalid_count
    assert len(generation_feedback.comparisons) + len(generation_feedback.invalid_attempts) == len(
        {champion_fingerprint(attempt.policy.recipe) for attempt in result.generations[0].attempts}
    )
    invalid_fingerprints = {
        champion_fingerprint(attempt.policy.recipe) for attempt in invalid
    }
    assert {
        item.structure_sha256 for item in generation_feedback.invalid_attempts
    } == invalid_fingerprints
    assert all(
        item.kind == "bounded_overlay"
        and item.stage_support == 8
        and item.reason_code == "unscorable_child"
        for item in generation_feedback.invalid_attempts
    )
    invalid_payloads = generation_feedback.to_payload()["invalid_attempts"]
    assert all(
        set(item) == {"structure_sha256", "kind", "stage_support", "reason_code"}
        for item in invalid_payloads
    )
    assert not any(
        forbidden in repr(invalid_payloads).casefold()
        for forbidden in (
            "overlay_0_",
            "better_a",
            "accepted",
            "rejected",
            "passed",
            "smae",
            "srmse",
            "truth",
            "forecast",
            "task_id",
            "split",
            "gate",
        )
    )
    assert proposer.feedback[1].invalid_attempts == generation_feedback.invalid_attempts
    assert not invalid_fingerprints & {champion_fingerprint(policy) for policy in result.shortlist}
    assert result.shortlist
    assert result.active_parent is PARENT


def test_deterministic_smoke_uses_only_the_4_8_schedule() -> None:
    task_ids = tuple(f"build_case_{index:03d}" for index in range(8))
    config = ChampionEvolutionConfig(
        build_size=8,
        calibration_size=2,
        screen_sizes=(4, 8),
        screen_task_ids=(task_ids[:4], task_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=0),
    )

    result = run_build_evolution(PARENT, _rows(8), RecordingProposer("better"), config)

    assert result.generations[0].stage_counts == (4, 8)
    assert result.config.build_task_ids == task_ids
    assert result.config_fingerprint == result.config.fingerprint


def test_history_only_materializer_scores_all_six_task3_operators() -> None:
    result = run_build_evolution(
        PARENT,
        _rows(enriched=True),
        AllOperatorProposer(),
        _config(),
    )

    valid_kinds = {
        attempt.policy.recipe.kind
        for attempt in result.generations[0].attempts
        if attempt.comparison is not None
    }
    assert valid_kinds == {
        "select",
        "route",
        "horizon_route",
        "weighted",
        "median",
        "bounded_overlay",
    }


def test_overlay_forecast_materialization_never_reads_future_truth(monkeypatch) -> None:
    rows = [
        row
        for row in _rows(1, enriched=True)
        if row.candidate_name
        in {
            "baseline_leaf",
            "better_a",
        }
    ]

    class PoisonTruth:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("future truth reached history-only materialization")

    for row in rows:
        object.__setattr__(row, "truth", PoisonTruth())
    assumption = EvolutionAssumption(
        assumption_id="poison_free_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="bounded_overlay",
        rationale="History diagnostics support the overlay.",
        failure_condition="History diagnostics stop supporting the overlay.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="poison_free_overlay",
            kind="bounded_overlay",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("poison_free_history", 0.75),),
        overlay_alpha=0.5,
        correction_cap=0.1,
    )
    runtime_calls = []
    execute = controller_module.execute_champion

    def history_only_execute(policy, forecasts, diagnostics, profile, history, horizon):
        runtime_calls.append((forecasts, diagnostics, profile, history, horizon))
        assert all(diagnostic.folds == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_forecasts == () for diagnostic in diagnostics.values())
        assert all(diagnostic.fold_truths == () for diagnostic in diagnostics.values())
        assert all(diagnostic.long_horizon_fold is None for diagnostic in diagnostics.values())
        return execute(policy, forecasts, diagnostics, profile, history, horizon)

    monkeypatch.setattr(controller_module, "execute_champion", history_only_execute)

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert len(runtime_calls) == 1
    assert failure is None
    assert forecast == pytest.approx((11.4, 13.4))


def test_valid_unsatisfied_history_condition_materializes_the_declared_fallback() -> None:
    rows = [
        row
        for row in _rows(1, enriched=True)
        if row.candidate_name in {"baseline_leaf", "better_a"}
    ]
    specialist_index = next(
        index for index, row in enumerate(rows) if row.candidate_name == "better_a"
    )
    specialist = rows[specialist_index]
    rows[specialist_index] = replace(
        specialist,
        diagnostic=_history_diagnostic("better_a", eligible=False),
    )
    assumption = EvolutionAssumption(
        assumption_id="unsatisfied_history",
        candidate_name="better_a",
        feature="periodicity_confidence",
        direction="above",
        horizon_region="full",
        operator="weighted",
        rationale="History diagnostics support the specialist.",
        failure_condition="History diagnostics do not support the specialist.",
    )
    policy = FittedChampionPolicy(
        recipe=ChampionRecipe(
            name="fallback_on_unsatisfied_history",
            kind="weighted",
            parents=("baseline_leaf", "better_a"),
            fallback_parent="baseline_leaf",
            assumptions=(assumption,),
        ),
        thresholds=(("unsatisfied_history", 0.0),),
        weights=(0.5, 0.5),
    )

    forecast, failure = controller_module._policy_forecast(
        policy, {row.candidate_name: row for row in rows}
    )

    assert failure is None
    assert forecast == rows[0].forecast


def test_proposal_policy_id_cannot_collide_with_the_active_parent() -> None:
    class ParentCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            return (parent.policy.recipe, *_proposal_batch("better")[:4])

    with pytest.raises(ChampionControllerError, match="duplicate policy ID"):
        run_build_evolution(PARENT, ROWS_64, ParentCollisionProposer(), _config())


def test_invalid_nested_parent_is_rejected_before_the_proposer() -> None:
    parent = _parent()
    object.__setattr__(parent.policy.recipe, "name", "")

    class CountingProposer:
        calls = 0

        def __call__(self, received, evidence) -> tuple[ChampionRecipe, ...]:
            type(self).calls += 1
            return _proposal_batch("better")

    with pytest.raises(ChampionControllerError, match="active Parent"):
        run_build_evolution(parent, ROWS_64, CountingProposer(), _config())

    assert CountingProposer.calls == 0


def test_callback_recipe_alias_cannot_corrupt_prior_generation_fitted_ids() -> None:
    class AliasingProposer:
        calls = 0
        first_batch: tuple[ChampionRecipe, ...] | None = None

        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            if self.first_batch is not None:
                object.__setattr__(self.first_batch[0], "name", "mutated_external_alias")
            batch = _proposal_batch("better", self.calls)
            if self.first_batch is None:
                self.first_batch = batch
            self.calls += 1
            return batch

    result = run_build_evolution(PARENT, ROWS_64, AliasingProposer(), _config(generations=2))

    assert all(
        attempt.fitted_id == champion_fingerprint(attempt.policy)
        for generation in result.generations
        for attempt in generation.attempts
    )
    assert all(
        attempt.policy.recipe.name != "mutated_external_alias"
        for generation in result.generations
        for attempt in generation.attempts
    )


def test_callback_cannot_loosen_or_leave_the_registered_gate_mutated() -> None:
    gate = ChampionGateConfig(minimum_improved_folds=0)
    config = replace(_config(), gate_config=gate)

    class GateMutator:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            object.__setattr__(gate, "minimum_coverage", 0.0)
            return _proposal_batch("better")

    original_fingerprint = config.fingerprint
    with pytest.raises(ChampionControllerError, match="config"):
        run_build_evolution(PARENT, ROWS_64, GateMutator(), config)

    assert gate.minimum_coverage == 1.0
    assert config.fingerprint == original_fingerprint


def test_provisional_screen_relaxes_fold_gate_before_full_build() -> None:
    all_ids = tuple(f"build_case_{index:03d}" for index in range(64))
    first_stage = tuple(f"build_case_{index:03d}" for index in range(0, 40, 5))
    second_stage = (
        first_stage + tuple(task_id for task_id in all_ids if task_id not in first_stage)[:24]
    )
    config = replace(
        _config(),
        screen_task_ids=(first_stage, second_stage, all_ids),
        gate_config=ChampionGateConfig(minimum_improved_folds=2),
    )

    result = run_build_evolution(PARENT, ROWS_64, RecordingProposer("better"), config)

    assert any(
        attempt.stage_task_counts == (8, 32, 64)
        for attempt in result.generations[0].attempts
    )
    assert result.generations[0].full_build_children
    assert result.shortlist


def test_normalized_row_inventory_collision_fails_before_proposer() -> None:
    rows = list(ROWS_64)
    for index in range(64):
        task_id = f"build_case_{index:03d}"
        source = next(
            row
            for row in ROWS_64
            if row.task_id == task_id and row.candidate_name == "baseline_leaf"
        )
        rows.extend(
            (
                replace(source, candidate_name="Model_K"),
                replace(source, candidate_name="Model_K"),
            )
        )
    proposer = RecordingProposer("better")

    with pytest.raises(ChampionControllerError, match="normalized row inventory"):
        run_build_evolution(PARENT, tuple(rows), proposer, _config())

    assert proposer.calls == 0


def test_proposal_name_cannot_collide_with_materialized_row_inventory() -> None:
    class InventoryCollisionProposer:
        def __call__(self, parent, evidence) -> tuple[ChampionRecipe, ...]:
            first, *remaining = _proposal_batch("better")
            return (replace(first, name="better_a"), *remaining)

    with pytest.raises(ChampionControllerError, match="materialized row inventory"):
        run_build_evolution(PARENT, ROWS_64, InventoryCollisionProposer(), _config())
