"""Domain-independent Parent/Child self-evolution controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Generic, Mapping, Sequence, TypeVar

from .contracts import (
    EvaluationReport,
    EvolutionComponents,
    EvolutionConfig,
    MutationContext,
)


ArtifactT = TypeVar("ArtifactT")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class EvolutionStep:
    generation: int
    parent_id: str
    child_ids: tuple[str, ...]
    promoted_ids: tuple[str, ...]
    selected_child_id: str | None
    accepted: bool
    accepted_artifact_id: str
    parent_train_report: EvaluationReport
    child_train_reports: tuple[EvaluationReport, ...]
    parent_dev_report: EvaluationReport | None
    child_dev_report: EvaluationReport | None


@dataclass(frozen=True)
class EvolutionOutcome(Generic[ArtifactT]):
    accepted_artifact: ArtifactT
    steps: tuple[EvolutionStep, ...]
    resumed_from_generation: int = 0


class SelfEvolutionEngine(Generic[ArtifactT, ItemT, ResultT]):
    """Execute a parameterized evolution lifecycle without domain knowledge."""

    def __init__(
        self,
        config: EvolutionConfig,
        components: EvolutionComponents[ArtifactT, ItemT, ResultT],
    ) -> None:
        self.config = config
        self.components = components

    def evolve(
        self,
        parent: ArtifactT,
        train_items: Sequence[ItemT],
        val_items: Sequence[ItemT],
    ) -> EvolutionOutcome[ArtifactT]:
        if not train_items:
            raise ValueError("train_items must not be empty")
        if not val_items:
            raise ValueError("val_items must not be empty")

        adapter = self.components.artifact_adapter
        current = parent
        start_generation = 0
        if self.config.resume:
            checkpoint = self.components.store.load_checkpoint()
            if checkpoint is not None:
                current, start_generation = self._restore_checkpoint(checkpoint)

        adapter.validate(current)

        steps: list[EvolutionStep] = []
        for generation in range(start_generation + 1, self.config.generations + 1):
            
            parent_train = self._evaluate(current, train_items, "train")
            current = adapter.apply_train_report(current, parent_train)
            adapter.validate(current)
            parent_id = adapter.artifact_id(current)
            self.components.store.save_artifact(
                f"generation_{generation:03d}_parent", adapter.to_payload(current)
            )

            context = MutationContext(
                generation=generation,
                parent_train_report=parent_train,
                failure_traces=self._failure_traces(parent_train),
            )
            proposed = tuple(
                self.components.mutator.propose(
                    current, context, self.config.children_per_generation
                )
            )
            for child in proposed:
                adapter.validate(child)

            promoted = proposed
            train_pairs: list[tuple[ArtifactT, EvaluationReport]] = []
            for child in promoted:
                report = self._evaluate(child, train_items, "train")
                annotated = adapter.apply_train_report(child, report)
                adapter.validate(annotated)
                train_pairs.append((annotated, report))
                self.components.store.save_artifact(
                    f"generation_{generation:03d}_child_{adapter.artifact_id(annotated)}",
                    adapter.to_payload(annotated),
                )

            selected_pair = self._best_train_pair(train_pairs)
            parent_val: EvaluationReport | None = None
            child_val: EvaluationReport | None = None
            accepted = False
            selected_id: str | None = None
            if selected_pair is not None:
                selected, _ = selected_pair
                selected_id = adapter.artifact_id(selected)
                parent_val = self._evaluate(current, val_items, "val")
                child_val = self._evaluate(selected, val_items, "val")
                accepted = self.components.acceptance_gate.accept(parent_val, child_val)
                if accepted:
                    current = selected

            step = EvolutionStep(
                generation=generation,
                parent_id=parent_id,
                child_ids=tuple(adapter.artifact_id(child) for child in proposed),
                promoted_ids=tuple(adapter.artifact_id(child) for child in promoted),
                selected_child_id=selected_id,
                accepted=accepted,
                accepted_artifact_id=adapter.artifact_id(current),
                parent_train_report=parent_train,
                child_train_reports=tuple(report for _, report in train_pairs),
                parent_dev_report=parent_val,
                child_dev_report=child_val,
            )
            steps.append(step)
            self.components.store.append_trace(self._step_payload(step))
            self.components.store.save_artifact(
                "best_artifact", adapter.to_payload(current)
            )
            self.components.store.save_checkpoint(
                {
                    "generation": generation,
                    "accepted_artifact": adapter.to_payload(current),
                }
            )

        return EvolutionOutcome(
            accepted_artifact=current,
            steps=tuple(steps),
            resumed_from_generation=start_generation,
        )

    def _evaluate(
        self, artifact: ArtifactT, items: Sequence[ItemT], split: str
    ) -> EvaluationReport:
        artifact_id = self.components.artifact_adapter.artifact_id(artifact)
        results = tuple(self.components.executor.execute(artifact, items, split))
        if not results:
            raise ValueError(f"executor returned no results for {artifact_id!r} on {split}")
        report = self.components.evaluator.evaluate(artifact_id, results, split)
        if report.artifact_id != artifact_id:
            raise ValueError("evaluator returned a report for the wrong artifact")
        if report.split != split:
            raise ValueError("evaluator returned a report for the wrong split")
        return report

    def _best_train_pair(
        self, pairs: Sequence[tuple[ArtifactT, EvaluationReport]]
    ) -> tuple[ArtifactT, EvaluationReport] | None:
        if not pairs:
            return None
        metric_name = self.config.metric.name
        reverse = self.config.metric.objective == "maximize"
        return sorted(
            pairs,
            key=lambda pair: pair[1].metrics[metric_name],
            reverse=reverse,
        )[0]

    def _restore_checkpoint(
        self, checkpoint: Mapping[str, object]
    ) -> tuple[ArtifactT, int]:
        generation = checkpoint.get("generation")
        artifact_payload = checkpoint.get("accepted_artifact")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("checkpoint generation must be a non-negative integer")
        if not isinstance(artifact_payload, Mapping):
            raise ValueError("checkpoint accepted_artifact must be an object")
        artifact = self.components.artifact_adapter.from_payload(artifact_payload)
        self.components.artifact_adapter.validate(artifact)
        return artifact, generation

    @staticmethod
    def _failure_traces(
        report: EvaluationReport,
    ) -> tuple[Mapping[str, object], ...]:
        traces = report.diagnostics.get("failure_traces", ())
        if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
            return ()
        return tuple(item for item in traces if isinstance(item, Mapping))

    @staticmethod
    def _step_payload(step: EvolutionStep) -> dict[str, object]:
        return asdict(step)
