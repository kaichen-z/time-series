from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from common.data import Task
from common.llm import FakeLLMClient, LLMResponse, TransientLLMError
from evolving_loop.data import ContextTask, Document
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCheckpointError,
    RetrievalEvaluation,
    RetrievalEvolutionConfig,
    RetrievalEvolutionEngine,
    RetrievalEvolutionResult,
    RetrievalGenerationTrace,
    build_inference_cache_key,
    combine_retrieval_evaluations,
    parse_scoped_child,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillError,
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
    _migrate_legacy_for_operator,
)


def _tasks(prefix: str, count: int, *, entity_offset: int = 0) -> tuple[ContextTask, ...]:
    result = []
    for index in range(count):
        entity = f"entity_{entity_offset + index // 8:03d}"
        result.append(
            ContextTask(
                numeric=Task(
                    task_id=f"{prefix}_{index:03d}",
                    history_values=(1.0, 2.0, 3.0),
                    future_values=(4.0, 5.0),
                    prediction_length=2,
                    frequency="D",
                    seasonal_period=None,
                    entity_name=entity,
                ),
                target_name="volume",
                target_description="private resolved target",
                history_timestamps=("2026-01-01", "2026-01-02", "2026-01-03"),
                future_timestamps=("2026-01-04", "2026-01-05"),
                documents=(
                    Document(
                        f"{prefix}_doc_{index:03d}",
                        "A public document.",
                        role="PRIVATE_RELEVANCE_LABEL",
                    ),
                ),
                gt_evidence=("PRIVATE_GT_EVIDENCE",),
                labels_public=True,
            )
        )
    return tuple(result)


def _evaluation(
    version: str,
    tasks: tuple[ContextTask, ...],
    *,
    error: float = 1.0,
    **overrides: object,
) -> RetrievalEvaluation:
    values: dict[str, object] = {
        "version": version,
        "task_count": len(tasks),
        "mean_final_smae": error,
        "mean_final_srmse": error,
        "mean_contextual_oracle_smae": error,
        "mean_contextual_oracle_srmse": error,
        "p90_smae": error,
        "p95_smae": error,
        "supporting_recall": 0.90,
        "distractor_avoidance": 0.90,
        "exact_quote_validity": 1.0,
        "complete_chain_rate": 0.80,
        "invalid_count": 0,
        "catastrophic_count": 0,
        "task_traces": tuple(
            {
                "task_id": task.numeric.task_id,
                "entity_name": task.numeric.entity_name,
                "final_smae": error,
                "final_srmse": error,
                "contextual_oracle_smae": error,
                "contextual_oracle_srmse": error,
                "failure_example": "TRAIN_OR_DEV_EVALUATOR_SECRET",
            }
            for task in tasks
        ),
    }
    values.update(overrides)
    return RetrievalEvaluation(**values)


def _proposal(parent: RetrievalGenome, version: str, scope: str) -> dict[str, object]:
    payload = parent.to_payload()
    payload.update({"version": version, "parent": parent.version})
    if scope == "A":
        payload["round1_prompt"] = f"{parent.round1_prompt}\nScoped change {version}."
    elif scope == "B":
        payload["max_citations_per_chain"] = (
            parent.max_citations_per_chain % 8
        ) + 1
    elif scope == "C":
        payload["round2_strategy"] = (
            "gap_first"
            if parent.round2_strategy != "gap_first"
            else "causal_chain_first"
        )
    else:
        raise AssertionError(scope)
    return payload


def _responses(generations: int) -> list[str]:
    parent = RetrievalGenome.seed()
    responses: list[str] = []
    for generation in range(generations):
        children = []
        for index, scope in enumerate(("A", "B", "C"), start=1):
            version = f"v{generation * 3 + index:03d}"
            payload = _proposal(parent, version, scope)
            responses.append(json.dumps(payload))
            children.append(RetrievalGenome.from_payload(payload))
        parent = children[0]
    return responses


@dataclass(frozen=True)
class _EvaluationCall:
    version: str
    task_ids: tuple[str, ...]
    entities: tuple[str, ...]
    stage: str
    persist: bool
    writers_enabled: bool
    evolver_enabled: bool
    cache_keys: tuple[object, ...]


class _FakeEvaluator:
    evaluator_hash = "fake-evaluator-v1"
    verifier_hash = "fake-verifier-v1"

    def __init__(
        self,
        *,
        errors: dict[str, float] | None = None,
        dev_overrides: dict[str, object] | None = None,
        transient_stage: str | None = None,
        failing_version: str | None = None,
    ) -> None:
        self.errors = errors or {}
        self.dev_overrides = dev_overrides or {}
        self.transient_stage = transient_stage
        self.failing_version = failing_version
        self.calls: list[_EvaluationCall] = []
        self._transient_raised = False

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        persist: bool,
        writers_enabled: bool,
        evolver_enabled: bool,
        cache_keys: tuple[object, ...],
        **_unused: object,
    ) -> RetrievalEvaluation:
        self.calls.append(
            _EvaluationCall(
                genome.version,
                tuple(task.numeric.task_id for task in tasks),
                tuple(sorted({task.numeric.entity_name for task in tasks})),
                stage,
                persist,
                writers_enabled,
                evolver_enabled,
                cache_keys,
            )
        )
        if self.transient_stage == stage and not self._transient_raised:
            self._transient_raised = True
            raise TransientLLMError("temporary evaluator outage")
        if genome.version == self.failing_version and "screen_train" in stage:
            raise ValueError("forecasting candidate failed")
        error = self.errors.get(genome.version, 1.0)
        overrides = (
            self.dev_overrides
            if stage == "child_dev" and genome.version != "v000"
            else {}
        )
        return _evaluation(genome.version, tasks, error=error, **overrides)


def _engine(
    evaluator: _FakeEvaluator,
    responses: list[str],
    *,
    checkpoint_path: Path | None = None,
    generations: int = 1,
    transient_retries: int = 1,
    dataset_split_hash: str = "split-v1",
) -> tuple[RetrievalEvolutionEngine, FakeLLMClient]:
    llm = FakeLLMClient(responses)
    config = RetrievalEvolutionConfig(
        generations=generations,
        screen_tasks=8,
        promote=2,
        train_folds=4,
        random_seed=17,
        transient_retries=transient_retries,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=dataset_split_hash,
        verifier_hash="verifier-v1",
        evaluator_hash="evaluator-v1",
        metric_hash="metric-v1",
        metric_cap=5.0,
    )
    return RetrievalEvolutionEngine(llm, evaluator, config), llm


def test_dev_acceptance_enforces_every_pareto_tail_recall_and_safety_gate() -> None:
    tasks = _tasks("dev", 20, entity_offset=100)
    parent = _evaluation("v000", tasks, error=1.0)
    accepted = _evaluation(
        "v001",
        tasks,
        error=0.99,
        mean_contextual_oracle_smae=0.98,
        mean_contextual_oracle_srmse=1.0,
    )
    assert accepted.dev_accepts(parent, tolerance=1e-12)

    failures = (
        {"task_count": 19},
        {"mean_contextual_oracle_smae": 1.01},
        {"mean_contextual_oracle_smae": 1.0, "mean_contextual_oracle_srmse": 1.0},
        {"mean_final_smae": 1.01},
        {"mean_final_srmse": 1.01},
        {"p90_smae": 1.01},
        {"p95_smae": 1.01},
        {"supporting_recall": 0.879},
        {"distractor_avoidance": 0.879},
        {"exact_quote_validity": 0.999},
        {"catastrophic_count": 1},
        {"invalid_count": 1},
    )
    for changes in failures:
        assert not replace(accepted, **changes).dev_accepts(parent, tolerance=1e-12)


def test_combined_tail_metrics_require_complete_finite_task_traces_and_linear_quantiles() -> None:
    tasks = _tasks("train", 4)
    left = _evaluation(
        "v001",
        tasks[:2],
        p90_smae=99.0,
        p95_smae=99.0,
        task_traces=(
            {**_evaluation("v001", tasks[:1]).task_traces[0], "final_smae": 0.0},
            {**_evaluation("v001", tasks[1:2]).task_traces[0], "final_smae": 1.0},
        ),
    )
    right = _evaluation(
        "v001",
        tasks[2:],
        p90_smae=77.0,
        p95_smae=77.0,
        task_traces=(
            {**_evaluation("v001", tasks[2:3]).task_traces[0], "final_smae": 2.0},
            {**_evaluation("v001", tasks[3:]).task_traces[0], "final_smae": 10.0},
        ),
    )

    combined = combine_retrieval_evaluations("v001", (left, right))

    assert combined.p90_smae == pytest.approx(7.6)
    assert combined.p95_smae == pytest.approx(8.8)
    with pytest.raises(Exception, match="trace|final_smae|finite|task_count"):
        combine_retrieval_evaluations(
            "v001",
            (replace(left, task_traces=left.task_traces[:1]), right),
        )
    with pytest.raises(Exception, match="trace|final_smae|finite"):
        combine_retrieval_evaluations(
            "v001",
            (
                replace(
                    left,
                    task_traces=(
                        {**left.task_traces[0], "final_smae": float("nan")},
                        left.task_traces[1],
                    ),
                ),
                right,
            ),
        )
    with pytest.raises(Exception, match="trace|task|coverage|duplicate"):
        combine_retrieval_evaluations(
            "v001",
            (
                left,
                replace(
                    right,
                    task_traces=tuple(
                        {**trace, "task_id": left.task_traces[0]["task_id"]}
                        for trace in right.task_traces
                    ),
                ),
            ),
        )


def test_complete_candidate_scope_validation_owns_only_a_b_or_c_fields(tmp_path) -> None:
    parent = RetrievalGenome.seed()
    library = RetrievalSkillLibrary(
        tmp_path / "skills.json",
        (
            RetrievalSkill(
                skill_id="round1_skill",
                version=1,
                parent_version=None,
                stage="round1",
                status="candidate",
                name="round1_skill",
                description="Round one.",
                applicability=RetrievalApplicability(),
                query_steps=("Search.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
            RetrievalSkill(
                skill_id="both_skill",
                version=1,
                parent_version=None,
                stage="both",
                status="candidate",
                name="both_skill",
                description="Both rounds.",
                applicability=RetrievalApplicability(),
                query_steps=("Compose.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
            RetrievalSkill(
                skill_id="round2_skill",
                version=1,
                parent_version=None,
                stage="round2",
                status="candidate",
                name="round2_skill",
                description="Round two.",
                applicability=RetrievalApplicability(),
                query_steps=("Fill the gap.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
        ),
        persist=False,
    )

    assert parse_scoped_child(parent, _proposal(parent, "v001", "A"), scope="A")
    assert parse_scoped_child(parent, _proposal(parent, "v002", "B"), scope="B")
    assert parse_scoped_child(parent, _proposal(parent, "v003", "C"), scope="C")

    cross_scope = _proposal(parent, "v001", "A")
    cross_scope["round2_strategy"] = "gap_first"
    assert parse_scoped_child(parent, cross_scope, scope="A") is None
    incomplete = _proposal(parent, "v001", "A")
    incomplete.pop("round2_prompt")
    assert parse_scoped_child(parent, incomplete, scope="A") is None
    stale_parent = replace(parent, active_skill_ids=("round1_skill",))
    retains_untrusted_skill = _proposal(stale_parent, "v001", "A")
    assert (
        parse_scoped_child(
            stale_parent,
            retains_untrusted_skill,
            scope="A",
            skill_library=library,
        )
        is None
    )

    for scope, skill_id in (
        ("A", "round1_skill"),
        ("B", "both_skill"),
        ("C", "round2_skill"),
    ):
        payload = _proposal(parent, f"v{10 + len(skill_id) + ord(scope):03d}", scope)
        payload["active_skill_ids"] = [skill_id]
        child = parse_scoped_child(
            parent, payload, scope=scope, skill_library=library
        )
        assert child is None

    def active_payload(skill_id: str, stage: str, *, constrained: bool = False):
        return {
            "skill_id": skill_id,
            "version": 1,
            "parent_version": None,
            "stage": stage,
            "status": "accepted",
            "name": skill_id,
            "description": "Trusted prompt-active Skill.",
            "applicability": {
                "assumption_kinds": ["future_event"] if constrained else [],
                "gap_types": [],
                "temporal_relations": [],
            },
            "query_steps": ["Search."],
            "required_chain_fields": ["entity"],
            "counterevidence_rule": "Check the opposite.",
            "failure_conditions": ["Wrong entity."],
            "validated_task_ids": ["t1", "t2", "t3"],
            "validated_entities": ["e1", "e2"],
            "validation_smae_gain": 0.1,
            "validation_srmse_gain": 0.1,
            "merged_from_skill_ids": [],
            "quarantine_reason": None,
        }

    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        replace(
            parent,
            version="v900",
            parent=parent.version,
            active_skill_ids=(
                "both_skill",
                "constrained_round1",
                "round1_skill",
                "round2_skill",
            ),
        ),
        skills=(
            active_payload("round1_skill", "round1"),
            active_payload("both_skill", "both"),
            active_payload("round2_skill", "round2"),
            active_payload("constrained_round1", "round1", constrained=True),
        ),
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )
    active_library = RetrievalSkillLibrary.from_release(release.path)
    for scope, skill_id, expected in (
        ("A", "round1_skill", True),
        ("A", "round2_skill", False),
        ("B", "both_skill", True),
        ("B", "round1_skill", False),
        ("C", "round2_skill", True),
        ("C", "both_skill", False),
        ("A", "constrained_round1", False),
    ):
        payload = _proposal(parent, f"v{100 + len(skill_id) + ord(scope):03d}", scope)
        payload["active_skill_ids"] = [skill_id]
        child = parse_scoped_child(
            parent, payload, scope=scope, skill_library=active_library
        )
        assert (child is not None) is expected

    scoped_a = _proposal(parent, "v001", "A")
    scoped_a["active_skill_ids"] = ["round1_skill"]
    engine, _llm = _engine(
        _FakeEvaluator(),
        [
            json.dumps(scoped_a),
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )
    engine.skill_library = active_library
    engine._original_parent = parent

    _children, rejections = engine._children_for_generation(
        0,
        parent,
        parent_library=library,
    )

    assert rejections["v001"] == "invalid_scoped_candidate"


def test_invalid_scope_proposal_still_leaves_exactly_three_a_b_c_child_slots() -> None:
    parent = RetrievalGenome.seed()
    invalid_a = _proposal(parent, "v001", "A")
    invalid_a["round2_strategy"] = "gap_first"
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v002": 0.90, "v003": 0.95}
    )
    engine, llm = _engine(
        evaluator,
        [
            json.dumps(invalid_a),
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    assert len(llm.calls) == 3
    assert generation.child_versions == ("v001", "v002", "v003")
    assert generation.child_scopes == ("A", "B", "C")
    assert len(generation.child_fingerprints) == 3
    assert generation.rejection_reasons["v001"] == "invalid_scoped_candidate"


def test_malformed_mutation_response_is_a_rejected_slot_not_an_aborted_run() -> None:
    parent = RetrievalGenome.seed()
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v002": 0.9, "v003": 0.95})
    engine, llm = _engine(
        evaluator,
        [
            "not json",
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert len(llm.calls) == 3
    assert result.generations[0].child_versions == ("v001", "v002", "v003")
    assert result.generations[0].rejection_reasons["v001"] == (
        "invalid_scoped_candidate"
    )


def test_three_malformed_responses_keep_distinct_scoped_fingerprints() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0})
    engine, _llm = _engine(evaluator, ["not json", "not json", "not json"])

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    assert generation.child_scopes == ("A", "B", "C")
    assert len(set(generation.child_fingerprints)) == 3
    assert set(generation.rejection_reasons) == {"v001", "v002", "v003"}


def test_child_versions_continue_after_nonseed_parent_version() -> None:
    parent = replace(RetrievalGenome.seed(), version="v010", parent="v009")
    evaluator = _FakeEvaluator(
        errors={"v010": 1.0, "v011": 0.9, "v012": 0.95, "v013": 1.2}
    )
    responses = [
        json.dumps(_proposal(parent, "v011", scope))
        for scope in ("A", "B", "C")
    ]
    # Give each fixed scope its own consecutive host-owned version.
    responses[1] = json.dumps(_proposal(parent, "v012", "B"))
    responses[2] = json.dumps(_proposal(parent, "v013", "C"))
    engine, _llm = _engine(evaluator, responses)

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.generations[0].child_versions == ("v011", "v012", "v013")
    assert result.train_winner.version == "v011"


def test_train_only_successive_halving_and_final_read_only_dev_schedule() -> None:
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(
        errors={
            "v000": 1.0,
            "v001": 0.90,
            "v002": 0.95,
            "v003": 1.20,
            "v004": 0.80,
            "v005": 0.85,
            "v006": 1.10,
        }
    )
    engine, _llm = _engine(evaluator, _responses(2), generations=2)

    result = engine.evolve(RetrievalGenome.seed(), train, dev)

    stages = [call.stage for call in evaluator.calls]
    assert stages.count("parent_dev") == 1
    assert stages.count("child_dev") == 1
    assert stages.index("parent_dev") > max(
        index for index, stage in enumerate(stages) if "train" in stage
    )
    screen_calls = [call for call in evaluator.calls if "screen_train" in call.stage]
    assert screen_calls
    assert {call.task_ids for call in screen_calls} == {
        result.generations[0].screen_task_ids
    }
    for generation in result.generations:
        assert len(generation.child_fingerprints) == 3
        assert len(generation.promoted_fingerprints) <= 2
        version_by_fingerprint = dict(
            zip(
                generation.child_fingerprints,
                generation.child_versions,
                strict=True,
            )
        )
        completion = next(
            event
            for event in result.trace
            if event["kind"] == "generation_completed"
            and event["generation"] == generation.generation
        )
        assert completion["promoted"] == [
            version_by_fingerprint[fingerprint]
            for fingerprint in generation.promoted_fingerprints
        ]

    for generation in range(2):
        prefix = f"g{generation}_parent_train_fold_"
        fold_calls = [call for call in evaluator.calls if call.stage.startswith(prefix)]
        seen_entities: set[str] = set()
        for call in fold_calls:
            assert seen_entities.isdisjoint(call.entities)
            seen_entities.update(call.entities)
        assert {
            task_id for call in fold_calls for task_id in call.task_ids
        } == {
            task.numeric.task_id
            for task in train
            if task.numeric.task_id not in result.generations[generation].screen_task_ids
        }

    promoted_full_versions = {
        call.version
        for call in evaluator.calls
        if "child_train_fold" in call.stage
    }
    assert len(promoted_full_versions.intersection({"v001", "v002", "v003"})) <= 2
    assert len(promoted_full_versions.intersection({"v004", "v005", "v006"})) <= 2
    dev_calls = [call for call in evaluator.calls if call.stage in {"parent_dev", "child_dev"}]
    assert [call.version for call in dev_calls] == ["v000", "v004"]
    assert all(
        not call.persist and not call.writers_enabled and not call.evolver_enabled
        for call in dev_calls
    )
    assert result.train_winner.version == "v004"


def test_screen_and_remaining_folds_partition_all_train_entities_before_selection() -> None:
    grouped = _tasks("train", 80)
    train = tuple(
        grouped[entity_index * 8 + within_entity]
        for within_entity in range(8)
        for entity_index in range(10)
    )
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        train,
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    screen_ids = set(generation.screen_task_ids)
    screen_entities = {
        task.numeric.entity_name for task in train if task.numeric.task_id in screen_ids
    }
    fold_entities = [set(entities) for entities in generation.fold_entities]
    assert len(screen_ids) == 8
    assert len(screen_entities) == 1
    assert len(fold_entities) == 4
    assert all(fold_entities)
    assert all(screen_entities.isdisjoint(entities) for entities in fold_entities)
    assert len(set.union(screen_entities, *fold_entities)) == 10


def test_train_partition_never_reduces_configured_entity_fold_count() -> None:
    one_entity = tuple(
        replace(task, numeric=replace(task.numeric, entity_name="single"))
        for task in _tasks("train", 80)
    )
    engine, _llm = _engine(_FakeEvaluator(), _responses(1))

    with pytest.raises(Exception, match="entit|fold|divers"):
        engine.evolve(
            RetrievalGenome.seed(),
            one_entity,
            _tasks("dev", 20, entity_offset=100),
        )


def test_every_evaluator_call_receives_the_complete_write_disabled_contract() -> None:
    calls: list[tuple[bool, bool, bool]] = []

    class ExactEvaluator:
        evaluator_hash = "exact-evaluator-v1"
        verifier_hash = "exact-verifier-v1"

        def evaluate(
            self,
            genome: RetrievalGenome,
            tasks: tuple[ContextTask, ...],
            *,
            stage: str,
            skill_library: RetrievalSkillLibrary | None,
            harness_factory: object,
            persist: bool,
            writers_enabled: bool,
            evolver_enabled: bool,
            cache_keys: tuple[object, ...],
            metric_cap: float,
        ) -> RetrievalEvaluation:
            del stage, skill_library, harness_factory, cache_keys, metric_cap
            calls.append((persist, writers_enabled, evolver_enabled))
            return _evaluation(genome.version, tasks, error=0.9)

    engine, _llm = _engine(ExactEvaluator(), _responses(1))  # type: ignore[arg-type]
    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert calls
    assert set(calls) == {(False, False, False)}


def test_evaluator_that_cannot_accept_all_control_flags_fails_closed() -> None:
    class FlagDroppingEvaluator:
        def evaluate(
            self,
            genome: RetrievalGenome,
            tasks: tuple[ContextTask, ...],
            *,
            stage: str,
            persist: bool,
            evolver_enabled: bool,
            cache_keys: tuple[object, ...],
        ) -> RetrievalEvaluation:
            del stage, persist, evolver_enabled, cache_keys
            return _evaluation(genome.version, tasks)

    engine, _llm = _engine(FlagDroppingEvaluator(), _responses(1))  # type: ignore[arg-type]

    with pytest.raises(Exception, match="writers_enabled|contract|evaluator"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )


def test_engine_supplies_immutable_candidate_library_snapshots(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False)
    attempted = RetrievalSkill(
        skill_id="evaluator_write",
        version=1,
        parent_version=None,
        stage="round1",
        status="candidate",
        name="evaluator_write",
        description="Must never be installed by evaluation.",
        applicability=RetrievalApplicability(),
        query_steps=("Write.",),
        required_chain_fields=("entity",),
        counterevidence_rule="Check conflicts.",
        failure_conditions=("Any write.",),
    )

    class MutatingEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, *, skill_library, **kwargs):
            assert skill_library is not None
            skill_library.add(attempted)
            return super().evaluate(
                genome,
                tasks,
                skill_library=skill_library,
                **kwargs,
            )

    evaluator = MutatingEvaluator()
    base, llm = _engine(evaluator, _responses(1))
    engine = RetrievalEvolutionEngine(
        llm,
        evaluator,
        base.config,
        skill_library=library,
    )

    with pytest.raises(RetrievalSkillError, match="read.only|immutable|mutation"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )
    assert library.get_by_id("evaluator_write") is None


def test_mutation_prompt_never_contains_tasks_labels_metrics_or_failure_examples() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, llm = _engine(evaluator, _responses(1))

    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("PRIVATE_TRAIN_TASK", 80),
        _tasks("PRIVATE_DEV_TASK", 20, entity_offset=100),
    )

    encoded = json.dumps(llm.calls, sort_keys=True)
    for forbidden in (
        "PRIVATE_TRAIN_TASK",
        "PRIVATE_DEV_TASK",
        "PRIVATE_GT_EVIDENCE",
        "PRIVATE_RELEVANCE_LABEL",
        "TRAIN_OR_DEV_EVALUATOR_SECRET",
        "mean_final_smae",
        "task_traces",
        "failure_example",
    ):
        assert forbidden not in encoded


def test_cache_key_binds_full_trusted_task_and_every_scientific_input(tmp_path) -> None:
    task = _tasks("train", 1)[0]
    genome = RetrievalGenome.seed()
    first_library = RetrievalSkillLibrary(tmp_path / "first.json", persist=False)
    second_library = RetrievalSkillLibrary(
        tmp_path / "second.json",
        (
            RetrievalSkill(
                skill_id="candidate",
                version=1,
                parent_version=None,
                stage="round1",
                status="candidate",
                name="candidate",
                description="Candidate.",
                applicability=RetrievalApplicability(),
                query_steps=("Search.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Search for conflicts.",
                failure_conditions=("Wrong entity.",),
            ),
        ),
        persist=False,
    )
    base = build_inference_cache_key(
        task,
        genome,
        first_library,
        verifier_hash="verifier-a",
        evaluator_hash="evaluator-a",
        metric_hash="metric-a",
        metric_cap=5.0,
        harness_hash="harness-a",
        scientific_inputs_hash="science-a",
    )
    variants = (
        build_inference_cache_key(
            replace(task, numeric=replace(task.numeric, task_id="other")),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, numeric=replace(task.numeric, history_values=(9.0,))),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(
                task,
                documents=(replace(task.documents[0], content="changed document"),),
            ),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, future_timestamps=("2099-01-01", "2099-01-02")),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, gt_evidence=("changed trusted label",)),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            replace(genome, round1_prompt="changed"),
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            second_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-b",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-b",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-b",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=6.0,
            harness_hash="harness-b",
            scientific_inputs_hash="science-b",
        ),
    )
    assert len({base.digest(), *(item.digest() for item in variants)}) == 12


def test_cache_key_revalidates_the_exact_current_skill_authority_epoch(tmp_path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "historical_skill",
                    "name": "historical_skill",
                    "description": "Operator-approved historical strategy.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require an exact quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    library = _migrate_legacy_for_operator(path)
    stale_snapshot = library.clone(persist=False)
    task = _tasks("train", 1)[0]

    def key(candidate: RetrievalSkillLibrary):
        return build_inference_cache_key(
            task,
            RetrievalGenome.seed(),
            candidate,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        )

    accepted_key = key(stale_snapshot)
    library.apply_operations(
        (RetrievalSkillOperation.quarantine("historical_skill", "unsafe"),)
    )

    with pytest.raises(RetrievalSkillError, match="current|authority|epoch"):
        key(stale_snapshot)
    assert key(library).digest() != accepted_key.digest()


def test_checkpoint_resume_binds_science_completion_and_child_fingerprints(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)

    first = engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["scientific_inputs"]["dataset_split_hash"] == "split-v1"
    assert payload["scientific_inputs"]["verifier_hash"] == "verifier-v1"
    assert payload["scientific_inputs"]["evaluator_hash"] == "evaluator-v1"
    assert payload["scientific_inputs"]["metric_cap"] == 5.0
    assert payload["scientific_inputs"]["random_seed"] == 17
    assert payload["original_parent_fingerprint"] == RetrievalGenome.seed().fingerprint()
    assert len(payload["child_fingerprints"]) == 3
    assert payload["task_completion"]

    resumed_evaluator = _FakeEvaluator()
    resumed, resumed_llm = _engine(
        resumed_evaluator, [], checkpoint_path=checkpoint
    )
    second = resumed.evolve(RetrievalGenome.seed(), train, dev)
    assert second == first
    assert resumed_evaluator.calls == []
    assert resumed_llm.calls == []

    mismatched, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        dataset_split_hash="different-split",
    )
    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        mismatched.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_task_completion_binds_cached_evaluation_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    first_cache_key = next(iter(payload["evaluation_cache"]))
    payload["evaluation_cache"][first_cache_key]["evaluation"][
        "mean_final_smae"
    ] = 4.5
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="evaluation|completion|digest|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_scientific_binding_rejects_json_type_drift(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["scientific_inputs"]["random_seed"] = 17.0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="scientific inputs|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_binds_the_retrieval_harness_factory(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)

    def factory_a(*_args, **_kwargs):
        return None

    def factory_b(*_args, **_kwargs):
        return None

    first = RetrievalEvolutionEngine(
        llm,
        evaluator,
        engine.config,
        harness_factory=factory_a,
    )
    first.evolve(RetrievalGenome.seed(), train, dev)
    resumed = RetrievalEvolutionEngine(
        FakeLLMClient([]),
        _FakeEvaluator(),
        engine.config,
        harness_factory=factory_b,
    )

    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_binds_transient_retry_science_policy(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    first, _ = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    first.evolve(RetrievalGenome.seed(), train, dev)
    resumed, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        transient_retries=1,
    )

    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_writer_ignores_attacker_fixed_temp_symlink(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    fixed_temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    fixed_temporary.symlink_to(victim)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )

    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert victim.read_text(encoding="utf-8") == "do-not-touch"
    assert fixed_temporary.is_symlink()


@pytest.mark.parametrize("symlink_kind", ("path", "parent"))
def test_checkpoint_rejects_symlink_paths_and_parents(tmp_path, symlink_kind) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    if symlink_kind == "path":
        victim = real_directory / "victim.json"
        victim.write_text("do-not-touch", encoding="utf-8")
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.symlink_to(victim)
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(real_directory, target_is_directory=True)
        checkpoint = linked_parent / "checkpoint.json"
    config = RetrievalEvolutionConfig(
        generations=1,
        train_folds=4,
        checkpoint_path=checkpoint,
        resume=False,
    )
    engine = RetrievalEvolutionEngine(
        FakeLLMClient(_responses(1)),
        _FakeEvaluator(),
        config,
    )

    with pytest.raises(RetrievalCheckpointError, match="symlink|unsafe|path"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )
    if symlink_kind == "path":
        assert victim.read_text(encoding="utf-8") == "do-not-touch"


def test_completed_checkpoint_rejects_boolean_type_drift(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["result"]["accepted"] = "true"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="result|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_authority_rejects_a_fully_rewritten_release_decision(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        dev_overrides={"p95_smae": 1.01},
    )
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    assert result.accepted is False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["result"].update(
        {
            "accepted": True,
            "acceptance_reasons": ["all_dev_gates_passed"],
            "rejection_reasons": [],
            "selected_genome": payload["result"]["train_winner"],
            "release_genome": payload["result"]["train_winner"],
        }
    )
    checkpoint.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="authority|authenticate|digest"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_authenticated_checkpoint_rederives_rejected_dev_acceptance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
            dev_overrides={"p95_smae": 1.01},
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    forged = replace(
        result,
        accepted=True,
        acceptance_reasons=("all_dev_gates_passed",),
        rejection_reasons=(),
        selected_genome=result.train_winner,
        release_genome=result.train_winner,
    )
    engine._save_checkpoint(status="complete", result=forged)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="accept|Dev|gate|result"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize("mutation", ("cache_key", "coverage"))
def test_authenticated_checkpoint_recomputes_cache_keys_and_coverage(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    cache_key, record = next(iter(engine._cache_records.items()))
    if mutation == "cache_key":
        record["task_cache_keys"][0] = "f" * 64
    else:
        record["task_ids"][0] = "invented-task"
    engine._cache_records[cache_key] = record
    engine._save_checkpoint(status="complete", result=result)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="cache|coverage|task"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize("mutation", ("generation_trace", "child_dev_trace"))
def test_authenticated_checkpoint_replays_selection_and_child_dev_trace(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    if mutation == "generation_trace":
        forged_generation = replace(
            result.generations[0],
            rejection_reasons={},
        )
        engine._generations[0] = forged_generation
        forged_result = replace(result, generations=(forged_generation,))
    else:
        assert result.child_dev is not None
        forged_child_dev = replace(
            result.child_dev,
            p95_smae=4.0,
            task_traces=tuple(
                {**trace, "final_smae": 4.0}
                for trace in result.child_dev.task_traces
            ),
        )
        forged_result = replace(result, child_dev=forged_child_dev)
    engine._save_checkpoint(status="complete", result=forged_result)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="generation|selection|trace|Dev|result|evaluation",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_rejects_a_rewritten_complete_child_vector(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    generation = payload["generations"][0]
    for field_name in ("child_versions", "child_fingerprints", "child_scopes"):
        generation[field_name].pop()
    payload["child_fingerprints"].pop()
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError, match="Child|generation|result|checkpoint"
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_generation_cursor_must_match_completed_generation_records(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["next_generation"] = 0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="generation|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("generation", True),
        ("parent_version", 7),
        ("child_versions", [1, "v002", "v003"]),
        ("screen_task_ids", [False]),
        ("promoted_fingerprints", "not-a-list"),
        ("rejection_reasons", {"v001": False}),
    ),
)
def test_generation_checkpoint_parser_rejects_exact_json_type_drift(
    field_name,
    replacement,
) -> None:
    trace = RetrievalGenerationTrace(
        generation=0,
        parent_version="v000",
        parent_fingerprint="0" * 64,
        child_versions=("v001", "v002", "v003"),
        child_fingerprints=("1" * 64, "2" * 64, "3" * 64),
        child_scopes=("A", "B", "C"),
        child_proposals=({}, {}, {}),
        screen_task_ids=("train_001",),
        fold_entities=(("entity_a",), ("entity_b",)),
        promoted_fingerprints=("1" * 64,),
        train_winner_version="v001",
        train_winner_fingerprint="1" * 64,
        rejection_reasons={"v002": "screen_rank:not_promoted"},
        screen_summaries={},
        train_summaries={},
    )
    payload = trace.to_payload()
    payload[field_name] = replacement

    with pytest.raises(RetrievalCheckpointError, match="generation|trace|checkpoint"):
        RetrievalGenerationTrace.from_payload(payload)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("acceptance_reasons", [False]),
        ("rejection_reasons", [7]),
        ("generations", {}),
        ("trace", ["not-an-event"]),
    ),
)
def test_completed_result_parser_never_coerces_json_types(
    field_name,
    replacement,
) -> None:
    parent = RetrievalGenome.seed()
    result = RetrievalEvolutionResult(
        original_parent=parent,
        train_winner=parent,
        selected_genome=parent,
        accepted=False,
        acceptance_reasons=(),
        rejection_reasons=("no_gain",),
        parent_dev=None,
        child_dev=None,
        generations=(),
        trace=(),
        release_genome=None,
        release_published=False,
    )
    payload = result.to_payload()
    payload[field_name] = replacement

    with pytest.raises(RetrievalCheckpointError, match="result|checkpoint"):
        RetrievalEvolutionResult.from_payload(payload)


@pytest.mark.parametrize("mutation", ("summary_integer", "trace_integer"))
def test_checkpoint_evaluation_parser_requires_exact_metric_json_types(
    mutation,
) -> None:
    payload = _evaluation("v001", _tasks("train", 1)).to_payload()
    if mutation == "summary_integer":
        payload["mean_final_smae"] = 1
    else:
        payload["task_traces"][0]["final_smae"] = 1

    with pytest.raises(RetrievalCheckpointError, match="evaluation|metric|trace"):
        RetrievalEvaluation.from_payload(payload)


def test_duplicate_task_trace_is_a_forecasting_failure_not_full_coverage() -> None:
    class DuplicateTraceEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            if genome.version == "v001" and "screen_train" in kwargs["stage"]:
                duplicate = dict(result.task_traces[0])
                return replace(
                    result,
                    task_traces=tuple(duplicate for _task in tasks),
                )
            return result

    evaluator = DuplicateTraceEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.generations[0].rejection_reasons["v001"].startswith(
        "forecasting_failure:"
    )


def test_transient_evaluator_retry_is_not_a_forecasting_failure() -> None:
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_child_A_screen_train",
        failing_version="v002",
    )
    engine, _llm = _engine(evaluator, _responses(1), transient_retries=1)

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    transient_calls = [
        call for call in evaluator.calls if call.stage == "g0_child_A_screen_train"
    ]
    forecasting_calls = [
        call
        for call in evaluator.calls
        if call.version == "v002" and call.stage == "g0_child_B_screen_train"
    ]
    assert len(transient_calls) == 2
    assert len(forecasting_calls) == 1
    assert any(
        event["kind"] == "transient_retry" for event in result.trace
    )
    assert result.generations[0].rejection_reasons["v002"].startswith(
        "forecasting_failure:"
    )


def test_resume_reuses_checkpointed_children_after_transient_exhaustion(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    interrupted_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_parent_screen_train",
    )
    interrupted, first_llm = _engine(
        interrupted_evaluator,
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(TransientLLMError):
        interrupted.evolve(RetrievalGenome.seed(), train, dev)
    assert len(first_llm.calls) == 3

    resumed_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    resumed, second_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )

    result = resumed.evolve(RetrievalGenome.seed(), train, dev)

    assert second_llm.calls == []
    assert result.generations[0].child_versions == ("v001", "v002", "v003")
    assert len(result.generations[0].child_fingerprints) == 3


@pytest.mark.parametrize("mutation", ("reorder", "invalid_fingerprint"))
def test_pending_checkpoint_requires_canonical_scoped_rows_and_recomputed_fingerprints(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    parent = RetrievalGenome.seed()
    interrupted_evaluator = _FakeEvaluator(
        transient_stage="g0_parent_screen_train"
    )
    interrupted, _llm = _engine(
        interrupted_evaluator,
        [
            "not json",
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    with pytest.raises(TransientLLMError):
        interrupted.evolve(parent, train, dev)
    assert interrupted._pending_children is not None
    rows = interrupted._pending_children["children"]
    assert isinstance(rows, list) and len(rows) == 3
    if mutation == "reorder":
        rows[:] = (rows[1], rows[0], rows[2])
        interrupted._all_child_fingerprints[:] = [
            row["fingerprint"] for row in rows
        ]
    else:
        rows[0]["fingerprint"] = "f" * 64
        interrupted._all_child_fingerprints[0] = "f" * 64
    interrupted._save_checkpoint(status="running", result=None)

    resumed, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(RetrievalCheckpointError, match="pending|scope|fingerprint"):
        resumed.evolve(parent, train, dev)


def test_resume_never_retries_a_checkpointed_terminal_forecasting_failure(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    interrupted_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_child_B_screen_train",
        failing_version="v001",
    )
    interrupted, _first_llm = _engine(
        interrupted_evaluator,
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(TransientLLMError):
        interrupted.evolve(RetrievalGenome.seed(), train, dev)
    assert len(
        [
            call
            for call in interrupted_evaluator.calls
            if call.version == "v001"
            and call.stage == "g0_child_A_screen_train"
        ]
    ) == 1

    resumed_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.1, "v002": 0.95, "v003": 1.2}
    )
    resumed, resumed_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )

    result = resumed.evolve(RetrievalGenome.seed(), train, dev)

    assert resumed_llm.calls == []
    assert not any(
        call.version == "v001" and call.stage == "g0_child_A_screen_train"
        for call in resumed_evaluator.calls
    )
    assert result.generations[0].rejection_reasons["v001"].startswith(
        "forecasting_failure:ValueError:"
    )


def test_rejected_dev_gate_has_complete_trace_and_never_publishes_release() -> None:
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        dev_overrides={"p95_smae": 1.01},
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.accepted is False
    assert result.selected_genome == RetrievalGenome.seed()
    assert result.release_genome is None
    assert result.release_published is False
    assert "p95_smae" in result.rejection_reasons
    assert result.parent_dev is not None
    assert result.child_dev is not None
    assert len(result.generations[0].child_fingerprints) == 3
    assert {event["kind"] for event in result.trace} >= {
        "generation_started",
        "child_proposed",
        "screen_completed",
        "generation_completed",
        "dev_completed",
        "release_rejected",
    }


class _TransientMutationClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.raised = False

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        if not self.raised:
            self.raised = True
            raise TransientLLMError("temporary mutation outage")
        return LLMResponse(self.responses.pop(0))


def test_transient_mutation_is_retried_without_creating_a_failed_child() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    llm = _TransientMutationClient(_responses(1))
    config = RetrievalEvolutionConfig(
        generations=1,
        screen_tasks=8,
        promote=2,
        train_folds=4,
        transient_retries=1,
        dataset_split_hash="split-v1",
        verifier_hash="verifier-v1",
        evaluator_hash="evaluator-v1",
        metric_hash="metric-v1",
    )

    result = RetrievalEvolutionEngine(llm, evaluator, config).evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert len(llm.calls) == 4
    assert len(result.generations[0].child_fingerprints) == 3
    assert "v001" not in result.generations[0].rejection_reasons
