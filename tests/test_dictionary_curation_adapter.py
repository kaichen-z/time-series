from __future__ import annotations

from pathlib import Path
from typing import Sequence

from common.evolution_core.contracts import EvaluationReport, MutationContext
from common.evolution_core.persistence import JsonArtifactStore
from numerical_agent.curation import (
    DictionaryArtifactAdapter,
    DictionaryCurationTask,
    DictionaryEvaluator,
    DictionaryExecutor,
    DictionaryMutator,
    MethodExecutionResult,
    NumericalTaskItem,
)
from numerical_agent.config import DictionaryCurationConfig
from numerical_agent.dictionary import (
    MethodCandidate,
    MethodDefinition,
    MethodRecord,
    ToolDictionary,
)
from numerical_agent.providers import (
    ImplementationContext,
    MethodImplementer,
    RuntimeRegistry,
    SanitizedMethodFeedback,
)


class FakeImplementer(MethodImplementer):
    def __init__(self) -> None:
        self.implemented: list[str] = []
        self.revised: list[str] = []

    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        self.implemented.append(method.method_id)
        if method.method_id == "good":
            return MethodCandidate("good", "fake", "opaque", {"prediction": 10.0})
        if method.method_id == "repairable":
            return MethodCandidate("repairable", "fake", "opaque", {"prediction": 80.0})
        if method.method_id == "unavailable":
            return MethodCandidate("unavailable", "missing", "opaque", {})
        if method.method_id == "unsafe":
            return MethodCandidate("unsafe", "fake", "opaque", {"unsafe": True})
        if method.method_id == "specialized":
            return MethodCandidate("specialized", "fake", "opaque", {"prediction": 40.0})
        raise AssertionError(method.method_id)

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate:
        self.revised.append(parent.method_id)
        return MethodCandidate(
            method_id=parent.method_id,
            provider="fake",
            implementation_kind="opaque",
            implementation={"prediction": 10.0},
            version=parent.version + 1,
            parent_version=parent.version,
        )


class FakeRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "fake"

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        return [float(candidate.implementation["prediction"])] * horizon


def absolute_error(prediction: Sequence[float], truth: Sequence[float]) -> float:
    return sum(abs(left - right) for left, right in zip(prediction, truth)) / len(truth)


def make_task(tmp_path: Path) -> tuple[DictionaryCurationTask, FakeImplementer]:
    methods = tuple(
        MethodDefinition(method_id, "statistical", f"external {method_id}")
        for method_id in ("good", "repairable", "unavailable", "unsafe")
    )
    implementer = FakeImplementer()
    task = DictionaryCurationTask(
        base_dictionary=ToolDictionary("d0", None, 0, methods),
        config=DictionaryCurationConfig(
            accepted_max_error=20.0,
            specialized_max_error=60.0,
        ),
        implementer=implementer,
        runtimes=RuntimeRegistry({"fake": FakeRuntime()}),
        labels={
            "train": {"t1": (10.0,), "t2": (10.0,)},
            "dev": {"d1": (10.0,)},
        },
        metric=absolute_error,
        store=JsonArtifactStore(tmp_path),
    )
    return task, implementer


def context(parent: ToolDictionary, report: EvaluationReport, generation: int) -> MutationContext:
    return MutationContext(generation=generation, parent_train_report=report)


def test_adapter_implements_tests_and_classifies_external_methods(tmp_path: Path) -> None:
    task, implementer = make_task(tmp_path)
    components = task.components()
    empty_report = EvaluationReport("d0", "train", {"smape": 999.0}, 2, {})

    child = components.mutator.propose(
        task.base_dictionary, context(task.base_dictionary, empty_report, 1), count=1
    )[0]
    results = components.executor.execute(
        child,
        (
            NumericalTaskItem("t1", (1.0, 2.0), 1, "D"),
            NumericalTaskItem("t2", (2.0, 3.0), 1, "D"),
        ),
        "train",
    )
    report = components.evaluator.evaluate(child.dictionary_id, results, "train")
    annotated = components.artifact_adapter.apply_train_report(child, report)

    statuses = {method.definition.method_id: method.status for method in annotated.methods}
    assert statuses == {
        "good": "accepted",
        "repairable": "discarded",
        "unavailable": "unavailable",
        "unsafe": "discarded",
    }
    assert set(implementer.implemented) == set(statuses)


def test_adapter_revises_quarantined_method_on_next_generation(tmp_path: Path) -> None:
    task, implementer = make_task(tmp_path)
    components = task.components()
    initial = components.mutator.propose(
        task.base_dictionary,
        context(task.base_dictionary, EvaluationReport("d0", "train", {"smape": 9.0}, 1, {}), 1),
        count=1,
    )[0]
    train_items = (NumericalTaskItem("t1", (1.0,), 1, "D"),)
    first_report = components.evaluator.evaluate(
        initial.dictionary_id,
        components.executor.execute(initial, train_items, "train"),
        "train",
    )
    first = components.artifact_adapter.apply_train_report(initial, first_report)

    # Make the weak but repairable method eligible for one revision rather than permanent discard.
    records = tuple(
        method
        if method.definition.method_id != "repairable"
        else method.__class__(
            method.definition,
            method.candidate,
            "quarantined",
            method.revision_count,
            method.train_summary,
        )
        for method in first.methods
    )
    repair_parent = ToolDictionary("d1", "d0", 1, records)
    repaired = components.mutator.propose(
        repair_parent, context(repair_parent, first_report, 2), count=1
    )[0]
    repaired_report = components.evaluator.evaluate(
        repaired.dictionary_id,
        components.executor.execute(repaired, train_items, "train"),
        "train",
    )
    annotated = components.artifact_adapter.apply_train_report(repaired, repaired_report)

    repairable = next(
        method for method in annotated.methods if method.definition.method_id == "repairable"
    )
    assert repairable.status == "accepted"
    assert repairable.candidate.version == 2
    assert implementer.revised == ["repairable"]


def test_adapter_marks_subset_winner_specialized_instead_of_discarded(tmp_path: Path) -> None:
    method = MethodDefinition("specialized", "statistical", "external specialized")
    implementer = FakeImplementer()
    task = DictionaryCurationTask(
        base_dictionary=ToolDictionary("d0", None, 0, (method,)),
        config=DictionaryCurationConfig(
            accepted_max_error=20.0,
            specialized_max_error=60.0,
        ),
        implementer=implementer,
        runtimes=RuntimeRegistry({"fake": FakeRuntime()}),
        labels={"train": {"near": (40.0,), "far": (100.0,)}, "dev": {"d": (40.0,)}},
        metric=absolute_error,
        store=JsonArtifactStore(tmp_path),
    )
    components = task.components()
    child = components.mutator.propose(
        task.base_dictionary,
        context(task.base_dictionary, EvaluationReport("d0", "train", {"smape": 1.0}, 2, {}), 1),
        1,
    )[0]
    items = (
        NumericalTaskItem("near", (1.0,), 1, "D"),
        NumericalTaskItem("far", (1.0,), 1, "D"),
    )
    report = components.evaluator.evaluate(
        child.dictionary_id, components.executor.execute(child, items, "train"), "train"
    )
    annotated = components.artifact_adapter.apply_train_report(child, report)

    assert annotated.methods[0].status == "specialized"


def test_dev_evaluation_never_updates_or_revises_methods(tmp_path: Path) -> None:
    task, implementer = make_task(tmp_path)
    components = task.components()
    child = components.mutator.propose(
        task.base_dictionary,
        context(task.base_dictionary, EvaluationReport("d0", "train", {"smape": 1.0}, 1, {}), 1),
        1,
    )[0]
    calls_before = (len(implementer.implemented), len(implementer.revised))
    dev_results = components.executor.execute(
        child, (NumericalTaskItem("d1", (1.0,), 1, "D"),), "dev"
    )
    report = components.evaluator.evaluate(child.dictionary_id, dev_results, "dev")

    assert report.split == "dev"
    assert (len(implementer.implemented), len(implementer.revised)) == calls_before


class RaisingRuntime:
    """A runtime whose failures carry a distinctive message, not just an exception type."""

    def __init__(self, message: str) -> None:
        self.message = message

    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "fake"

    def forecast(
        self, candidate: MethodCandidate, history: Sequence[float], horizon: int, frequency: str
    ) -> Sequence[float]:
        raise IndexError(self.message)


class OneShotImplementer:
    """Always returns the same opaque candidate; revise() is unused in this test."""

    def implement(self, method: MethodDefinition, context: ImplementationContext) -> MethodCandidate:
        return MethodCandidate(method.method_id, "fake", "opaque", {})

    def revise(self, parent: MethodCandidate, feedback: SanitizedMethodFeedback) -> MethodCandidate:
        raise AssertionError("not expected in this test")


def test_execute_one_preserves_the_real_exception_message(tmp_path: Path) -> None:
    method = MethodDefinition("m", "statistical", "external m")
    task = DictionaryCurationTask(
        base_dictionary=ToolDictionary("d0", None, 0, (method,)),
        config=DictionaryCurationConfig(),
        implementer=OneShotImplementer(),
        runtimes=RuntimeRegistry({"fake": RaisingRuntime("list index out of range")}),
        labels={"train": {"t1": (1.0,)}, "dev": {}},
        metric=absolute_error,
        store=JsonArtifactStore(tmp_path),
    )
    components = task.components()
    child = components.mutator.propose(
        task.base_dictionary,
        context(task.base_dictionary, EvaluationReport("d0", "train", {"smape": 1.0}, 1, {}), 1),
        1,
    )[0]

    results = components.executor.execute(child, (NumericalTaskItem("t1", (1.0,), 1, "D"),), "train")

    # The real message must survive, not just "runtime error: IndexError".
    assert results[0].error == "list index out of range"


def test_method_summary_deduplicates_and_caps_sample_errors() -> None:
    evaluator = DictionaryEvaluator(DictionaryCurationConfig(), {"train": {}}, absolute_error)
    results = (
        [MethodExecutionResult("d0", "m", f"a{i}", "invalid", error="IndexError: bad") for i in range(50)]
        + [MethodExecutionResult("d0", "m", f"b{i}", "invalid", error="TypeError: worse") for i in range(10)]
        + [MethodExecutionResult("d0", "m", f"c{i}", "invalid", error="ValueError: also bad") for i in range(3)]
        + [MethodExecutionResult("d0", "m", f"d{i}", "invalid", error="KeyError: overflow") for i in range(2)]
        + [MethodExecutionResult("d0", "m", "ok", "success", forecast=(1.0,))]
    )

    summary = evaluator._method_summary(results, {"ok": (1.0,)})

    # Exactly the cap, deduplicated, first-seen order, and never from a successful item.
    assert summary["sample_errors"] == (
        "IndexError: bad",
        "TypeError: worse",
        "ValueError: also bad",
    )


def test_feedback_carries_sample_errors_past_the_metrics_float_filter() -> None:
    report = EvaluationReport(
        "d0",
        "train",
        {"smape": 1.0},
        1,
        diagnostics={
            "per_method": {
                "m": {
                    "invalid_count": 1,
                    "sample_errors": ["IndexError: list index out of range"],
                }
            }
        },
    )
    mutator = DictionaryMutator(DictionaryCurationConfig(), implementer=None)  # type: ignore[arg-type]

    feedback = mutator._feedback("m", MutationContext(generation=2, parent_train_report=report))

    assert feedback.sample_errors == ("IndexError: list index out of range",)
    # A plain string field must not leak into the numeric metrics mapping.
    assert "sample_errors" not in feedback.metrics


def test_dictionary_score_uses_the_history_selected_method_not_oracle_minimum() -> None:
    evaluator = DictionaryEvaluator(
        DictionaryCurationConfig(),
        {"dev": {"t1": (0.0,), "t2": (0.0,)}},
        absolute_error,
    )
    results = (
        MethodExecutionResult("d", "A", "t1", "success", (0.0,), selected=True),
        MethodExecutionResult("d", "A", "t2", "success", (100.0,), selected=True),
        MethodExecutionResult("d", "B", "t1", "success", (100.0,)),
        MethodExecutionResult("d", "B", "t2", "success", (0.0,)),
    )

    report = evaluator.evaluate("d", results, "dev")

    assert report.metrics["smape"] == 50.0
    assert report.diagnostics["oracle_score"] == 0.0


class PatternRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "pattern"

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        if candidate.implementation["kind"] == "last":
            return [float(history[-1])] * horizon
        step = float(history[-1] - history[-2]) if len(history) > 1 else 0.0
        return [float(history[-1] + step * offset) for offset in range(1, horizon + 1)]


def test_executor_selects_a_method_using_history_only_hindcasting() -> None:
    methods = (
        MethodRecord(
            MethodDefinition("last", "statistical", "last value"),
            MethodCandidate("last", "pattern", "opaque", {"kind": "last"}),
        ),
        MethodRecord(
            MethodDefinition("trend", "statistical", "linear continuation"),
            MethodCandidate("trend", "pattern", "opaque", {"kind": "trend"}),
        ),
    )
    dictionary = ToolDictionary("d", None, 0, methods)
    executor = DictionaryExecutor(
        RuntimeRegistry({"pattern": PatternRuntime()}),
        selection_metric=absolute_error,
        selection_folds=2,
        selection_horizon=1,
    )

    results = executor.execute(
        dictionary,
        (NumericalTaskItem("t", (1.0, 2.0, 3.0, 4.0, 5.0), 1, "D"),),
        "dev",
    )

    selected = [result.method_id for result in results if result.selected]
    assert selected == ["trend"]
    assert all(result.selection_score is not None for result in results)


def test_executor_never_selects_a_quarantined_method() -> None:
    methods = (
        MethodRecord(
            MethodDefinition("accepted", "statistical", "deployable"),
            MethodCandidate("accepted", "fake", "opaque", {"prediction": 4.0}),
            status="accepted",
        ),
        MethodRecord(
            MethodDefinition("quarantined", "statistical", "not deployable"),
            MethodCandidate("quarantined", "fake", "opaque", {"prediction": 5.0}),
            status="quarantined",
        ),
    )
    dictionary = ToolDictionary("d", None, 1, methods)
    executor = DictionaryExecutor(
        RuntimeRegistry({"fake": FakeRuntime()}),
        selection_metric=absolute_error,
        selection_folds=2,
        selection_horizon=1,
    )

    results = executor.execute(
        dictionary,
        (NumericalTaskItem("t", (5.0, 5.0, 5.0, 5.0, 5.0), 1, "D"),),
        "dev",
    )

    assert [result.method_id for result in results if result.selected] == ["accepted"]


class FragileHindcastRuntime:
    def supports(self, candidate: MethodCandidate) -> bool:
        return candidate.provider == "fragile"

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> Sequence[float]:
        if candidate.implementation["kind"] == "fragile":
            if len(history) == 3:
                raise RuntimeError("one historical fold fails")
            return [5.0] * horizon
        return [4.0] * horizon


def test_selector_requires_hindcast_fold_coverage() -> None:
    methods = (
        MethodRecord(
            MethodDefinition("fragile", "statistical", "partially runnable"),
            MethodCandidate("fragile", "fragile", "opaque", {"kind": "fragile"}),
        ),
        MethodRecord(
            MethodDefinition("stable", "statistical", "fully runnable"),
            MethodCandidate("stable", "fragile", "opaque", {"kind": "stable"}),
        ),
    )
    executor = DictionaryExecutor(
        RuntimeRegistry({"fragile": FragileHindcastRuntime()}),
        selection_metric=absolute_error,
        selection_folds=2,
        selection_horizon=1,
        min_selection_success_rate=0.8,
    )

    results = executor.execute(
        ToolDictionary("d", None, 0, methods),
        (NumericalTaskItem("t", (5.0, 5.0, 5.0, 5.0, 5.0), 1, "D"),),
        "dev",
    )

    assert [result.method_id for result in results if result.selected] == ["stable"]
    fragile = next(result for result in results if result.method_id == "fragile")
    assert fragile.selection_score is None


def test_low_coverage_method_is_not_accepted() -> None:
    adapter = DictionaryArtifactAdapter(
        DictionaryCurationConfig(min_success_rate=0.8)
    )

    status = adapter._classify(
        {
            "total_count": 100,
            "success_count": 1,
            "success_rate": 0.01,
            "invalid_count": 99,
            "mean_error": 1.0,
            "subset_win_rate": 1.0,
        }
    )

    assert status == "quarantined"


def test_discarded_method_remains_terminal_without_a_new_execution_report() -> None:
    record = MethodRecord(
        MethodDefinition("unsafe", "statistical", "unsafe method"),
        MethodCandidate("unsafe", "fake", "opaque", {"unsafe": True}),
        status="discarded",
    )
    dictionary = ToolDictionary("d", None, 1, (record,))
    report = EvaluationReport(
        "d",
        "train",
        {"smape": 1000.0},
        1,
        diagnostics={"per_method": {}},
    )

    updated = DictionaryArtifactAdapter(DictionaryCurationConfig()).apply_train_report(
        dictionary, report
    )

    assert updated.methods[0].status == "discarded"


class ContextAwareImplementer:
    def __init__(self) -> None:
        self.contexts: list[ImplementationContext] = []

    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        self.contexts.append(context)
        return MethodCandidate(
            method.method_id,
            "fake",
            "opaque",
            {"child_index": context.child_index},
        )

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate:
        return parent


def test_mutator_gives_each_child_a_distinct_mutation_context() -> None:
    implementer = ContextAwareImplementer()
    parent = ToolDictionary(
        "d0", None, 0, (MethodDefinition("m", "statistical", "method"),)
    )
    mutator = DictionaryMutator(DictionaryCurationConfig(), implementer)
    report = EvaluationReport("d0", "train", {"smape": 1.0}, 1)

    children = mutator.propose(parent, MutationContext(1, report), 2)

    assert len(children) == 2
    assert [item.child_index for item in implementer.contexts] == [1, 2]
    assert len({item.diversity_instruction for item in implementer.contexts}) == 2


class ConstantImplementer(ContextAwareImplementer):
    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        self.contexts.append(context)
        return MethodCandidate(method.method_id, "fake", "opaque", {"same": True})


def test_mutator_deduplicates_identical_children() -> None:
    parent = ToolDictionary(
        "d0", None, 0, (MethodDefinition("m", "statistical", "method"),)
    )
    mutator = DictionaryMutator(DictionaryCurationConfig(), ConstantImplementer())
    report = EvaluationReport("d0", "train", {"smape": 1.0}, 1)

    children = mutator.propose(parent, MutationContext(1, report), 2)

    assert len(children) == 1


class FlakyImplementer:
    def __init__(self) -> None:
        self.calls = 0

    def implement(
        self, method: MethodDefinition, context: ImplementationContext
    ) -> MethodCandidate:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary model failure")
        return MethodCandidate(method.method_id, "fake", "opaque", {"prediction": 1.0})

    def revise(
        self, parent: MethodCandidate, feedback: SanitizedMethodFeedback
    ) -> MethodCandidate:
        return parent


def test_transient_implementation_failure_is_retried_next_generation() -> None:
    implementer = FlakyImplementer()
    parent = ToolDictionary(
        "d0", None, 0, (MethodDefinition("m", "statistical", "method"),)
    )
    mutator = DictionaryMutator(
        DictionaryCurationConfig(max_implementation_attempts=2), implementer
    )
    report = EvaluationReport("d0", "train", {"smape": 1.0}, 1)

    first = mutator.propose(parent, MutationContext(1, report), 1)[0]
    second = mutator.propose(first, MutationContext(2, report), 1)[0]

    assert first.methods[0].status == "unimplemented"
    assert first.methods[0].implementation_attempts == 1
    assert second.methods[0].candidate is not None
    assert second.methods[0].implementation_attempts == 2
