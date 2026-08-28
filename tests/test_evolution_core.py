"""Tests for common/evolution_core: typed contracts, the acceptance gate, and JSON persistence."""
from __future__ import annotations

import pytest
import json
from common.evolution_core.contracts import (
    EvaluationReport,
    EvolutionComponents,
    EvolutionConfig,
    MetricSpec,
    MutationContext,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
from common.evolution_core.acceptance import MetricAcceptanceGate
from common.evolution_core.controller import SelfEvolutionEngine
from common.evolution_core.persistence import JsonArtifactStore


def test_evolution_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="generations"):
        EvolutionConfig(generations=0)
    with pytest.raises(ValueError, match="children_per_generation"):
        EvolutionConfig(children_per_generation=0)
    with pytest.raises(ValueError, match="acceptance_margin"):
        EvolutionConfig(acceptance_margin=-0.1)


def test_metric_spec_orders_minimized_scores() -> None:
    metric = MetricSpec(name="smae", objective="minimize")

    assert metric.better(10.0, 12.0)
    assert not metric.better(12.0, 10.0)
    assert not metric.better(10.0, 10.0)


def test_metric_spec_orders_maximized_scores_with_margin() -> None:
    metric = MetricSpec(name="reward", objective="maximize")

    assert metric.better(0.7, 0.5, margin=0.1)
    assert not metric.better(0.6, 0.5, margin=0.1)


def test_metric_spec_rejects_invalid_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        MetricSpec(name="smae", objective="sideways")  # type: ignore[arg-type]

@dataclass(frozen=True)
class FakeArtifact:
    artifact_id: str
    quality: float
    train_annotations: int = 0


@dataclass(frozen=True)
class FakeResult:
    artifact_id: str
    quality: float
    item: int


class FakeAdapter:
    def __init__(self) -> None:
        self.applied_splits: list[str] = []

    def validate(self, artifact: FakeArtifact) -> None:
        if not artifact.artifact_id:
            raise ValueError("artifact_id")

    def artifact_id(self, artifact: FakeArtifact) -> str:
        return artifact.artifact_id

    def to_payload(self, artifact: FakeArtifact) -> dict[str, object]:
        return {
            "artifact_id": artifact.artifact_id,
            "quality": artifact.quality,
            "train_annotations": artifact.train_annotations,
        }

    def from_payload(self, payload: Mapping[str, object]) -> FakeArtifact:
        return FakeArtifact(
            artifact_id=str(payload["artifact_id"]),
            quality=float(payload["quality"]),
            train_annotations=int(payload.get("train_annotations", 0)),
        )

    def apply_train_report(
        self, artifact: FakeArtifact, report: EvaluationReport
    ) -> FakeArtifact:
        self.applied_splits.append(report.split)
        assert report.split == "train"
        return replace(artifact, train_annotations=artifact.train_annotations + 1)


class FakeMutator:
    def __init__(self, qualities: Sequence[float]) -> None:
        self.qualities = tuple(qualities)
        self.calls = 0

    def propose(
        self, parent: FakeArtifact, context: MutationContext, count: int
    ) -> Sequence[FakeArtifact]:
        self.calls += 1
        return tuple(
            FakeArtifact(f"v{context.generation:03d}_{index}", quality)
            for index, quality in enumerate(self.qualities[:count], start=1)
        )


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def execute(
        self, artifact: FakeArtifact, items: Sequence[int], split: str
    ) -> Sequence[FakeResult]:
        self.calls.append((artifact.artifact_id, split, len(items)))
        return tuple(FakeResult(artifact.artifact_id, artifact.quality, item) for item in items)


class FakeEvaluator:
    def evaluate(
        self, artifact_id: str, results: Sequence[FakeResult], split: str
    ) -> EvaluationReport:
        quality = results[0].quality
        return EvaluationReport(
            artifact_id=artifact_id,
            split=split,
            metrics={"loss": 10.0 - quality},
            item_count=len(results),
            diagnostics={},
        )


def make_engine(
    tmp_path: Path,
    qualities: Sequence[float],
    **config_overrides: object,
) -> tuple[SelfEvolutionEngine[FakeArtifact, int, FakeResult], FakeAdapter, FakeMutator, FakeExecutor]:
    adapter = FakeAdapter()
    mutator = FakeMutator(qualities)
    executor = FakeExecutor()
    config = EvolutionConfig(
        generations=int(config_overrides.pop("generations", 1)),
        children_per_generation=int(config_overrides.pop("children_per_generation", len(qualities))),
        metric=MetricSpec("loss", "minimize"),
        **config_overrides,
    )
    components = EvolutionComponents(
        artifact_adapter=adapter,
        mutator=mutator,
        executor=executor,
        evaluator=FakeEvaluator(),
        acceptance_gate=MetricAcceptanceGate(config.metric),
        store=JsonArtifactStore(tmp_path),
    )
    return SelfEvolutionEngine(config, components), adapter, mutator, executor


def test_engine_accepts_only_the_improving_child(tmp_path: Path) -> None:
    engine, adapter, _, _ = make_engine(tmp_path, qualities=(1.0, -1.0))

    outcome = engine.evolve(
        parent=FakeArtifact("v000", 0.0),
        train_items=(1, 2),
        val_items=(3, 4),
    )

    assert outcome.accepted_artifact.quality == 1.0
    assert outcome.steps[0].accepted
    assert adapter.applied_splits and set(adapter.applied_splits) == {"train"}


def test_engine_retains_parent_when_dev_does_not_improve(tmp_path: Path) -> None:
    engine, _, _, _ = make_engine(tmp_path, qualities=(0.0, -1.0))

    outcome = engine.evolve(FakeArtifact("v000", 0.0), (1, 2), (3, 4))

    assert outcome.accepted_artifact.artifact_id == "v000"
    assert not outcome.steps[0].accepted


def test_engine_requires_nonempty_train_and_dev(tmp_path: Path) -> None:
    engine, _, _, _ = make_engine(tmp_path, qualities=(1.0,))

    with pytest.raises(ValueError, match="train_items"):
        engine.evolve(FakeArtifact("v000", 0.0), (), (1,))
    with pytest.raises(ValueError, match="val_items"):
        engine.evolve(FakeArtifact("v000", 0.0), (1,), ())


def test_engine_resumes_from_persisted_generation(tmp_path: Path) -> None:
    first_engine, _, _, _ = make_engine(tmp_path, qualities=(1.0,), generations=1)
    first = first_engine.evolve(FakeArtifact("v000", 0.0), (1, 2), (3, 4))
    assert first.accepted_artifact.quality == 1.0

    resumed_engine, _, resumed_mutator, _ = make_engine(
        tmp_path, qualities=(2.0,), generations=2
    )
    resumed = resumed_engine.evolve(FakeArtifact("ignored", -5.0), (1, 2), (3, 4))

    assert resumed.resumed_from_generation == 1
    assert resumed_mutator.calls == 1
    assert resumed.accepted_artifact.quality == 2.0

def report(score: float) -> EvaluationReport:
    return EvaluationReport(
        artifact_id="v",
        split="val",
        metrics={"smae": score},
        item_count=2,
        diagnostics={},
    )


def test_acceptance_requires_strict_improvement() -> None:
    gate = MetricAcceptanceGate(MetricSpec("smae", "minimize"), margin=0.0)

    assert gate.accept(report(10.0), report(9.9))
    assert not gate.accept(report(10.0), report(10.0))
    assert not gate.accept(report(10.0), report(10.1))


def test_acceptance_rejects_missing_metric() -> None:
    gate = MetricAcceptanceGate(MetricSpec("smae", "minimize"))
    missing = EvaluationReport("v", "val", {"mae": 1.0}, 2, {})

    with pytest.raises(ValueError, match="smae"):
        gate.accept(report(10.0), missing)


def test_json_store_round_trips_checkpoint_and_artifact(tmp_path) -> None:
    store = JsonArtifactStore(tmp_path)

    artifact_path = store.save_artifact("parent", {"id": "v000", "quality": 1})
    checkpoint_path = store.save_checkpoint(
        {"generation": 2, "accepted_artifact": {"id": "v002"}}
    )

    assert json.loads(artifact_path.read_text())["id"] == "v000"
    assert checkpoint_path.name == "checkpoint.json"
    assert store.load_checkpoint() == {
        "generation": 2,
        "accepted_artifact": {"id": "v002"},
    }


def test_json_store_appends_one_trace_object_per_line(tmp_path) -> None:
    store = JsonArtifactStore(tmp_path)

    store.append_trace({"generation": 0, "accepted": False})
    store.append_trace({"generation": 1, "accepted": True})

    trace_path = tmp_path / "evolution_trace.jsonl"
    assert [json.loads(line) for line in trace_path.read_text().splitlines()] == [
        {"generation": 0, "accepted": False},
        {"generation": 1, "accepted": True},
    ]
