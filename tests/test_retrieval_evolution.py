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
    build_inference_cache_key,
    parse_scoped_child,
)
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillLibrary,
)


def _tasks(prefix: str, count: int, *, entity_offset: int = 0) -> tuple[ContextTask, ...]:
    result = []
    for index in range(count):
        entity = f"entity_{entity_offset + index // 5:03d}"
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

    for scope, skill_id, expected in (
        ("A", "round1_skill", True),
        ("A", "round2_skill", False),
        ("B", "both_skill", True),
        ("B", "round1_skill", False),
        ("C", "round2_skill", True),
        ("C", "both_skill", False),
    ):
        payload = _proposal(parent, f"v{10 + len(skill_id) + ord(scope):03d}", scope)
        payload["active_skill_ids"] = [skill_id]
        child = parse_scoped_child(
            parent, payload, scope=scope, skill_library=library
        )
        assert (child is not None) is expected


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
        tuple(task.numeric.task_id for task in train[:8])
    }
    for generation in result.generations:
        assert len(generation.child_fingerprints) == 3
        assert len(generation.promoted_fingerprints) <= 2

    for generation in range(2):
        prefix = f"g{generation}_parent_train_fold_"
        fold_calls = [call for call in evaluator.calls if call.stage.startswith(prefix)]
        seen_entities: set[str] = set()
        for call in fold_calls:
            assert seen_entities.isdisjoint(call.entities)
            seen_entities.update(call.entities)
        assert {
            task_id for call in fold_calls for task_id in call.task_ids
        } == {task.numeric.task_id for task in train[8:]}

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


def test_cache_key_binds_task_genome_skill_verifier_and_evaluator_hashes(tmp_path) -> None:
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
    )
    variants = (
        build_inference_cache_key(
            replace(task, numeric=replace(task.numeric, task_id="other")),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
        ),
        build_inference_cache_key(
            task,
            replace(genome, round1_prompt="changed"),
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            second_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-b",
            evaluator_hash="evaluator-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-b",
        ),
    )
    assert len({base.digest(), *(item.digest() for item in variants)}) == 6


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
    with pytest.raises(RetrievalCheckpointError, match="evaluation|completion|digest"):
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
    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
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
    with pytest.raises(RetrievalCheckpointError, match="result"):
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
    with pytest.raises(RetrievalCheckpointError, match="generation"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


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
