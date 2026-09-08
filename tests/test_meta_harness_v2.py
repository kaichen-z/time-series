from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from common.llm import FakeLLMClient
from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    CoEvolutionEngine,
    HarnessPolicy,
    PolicyEvaluation,
)
from evolving_loop.meta_harness_v2 import (
    MetaHarnessProposal,
    MetaTrainMemoryRecord,
    child_kind_for_slot,
    policy_change_scope,
    project_train_memory,
    validate_child_scope,
)


def _legal_proposal(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "mutation_scope": ["coding"],
        "interaction_hypothesis": "A broader candidate search should improve coverage.",
        "coding_generation_prompt": "Generate diverse executable forecasters.",
        "coding_revision_prompt": "Repair invalid numerical programs.",
        "retrieval_prompt": "Retrieve exact evidence for unresolved assumptions.",
        "decision_prompt": "Select the safest supported forecast candidate.",
        "coding_initial_programs": 4,
        "coding_mutations": 2,
        "coding_mutation_children": 2,
        "coding_validation_folds": 4,
        "coding_validation_horizon": 12,
        "workflow": ["retrieve", "decide"],
        "enable_evidence_adjustments": True,
        "max_evidence_adjustments": 3,
        "decision_aggregation": "last",
        "changelog": "Increase Coding search diversity within the fixed Host budget.",
    }
    payload.update(overrides)
    return payload


def _parent_payload() -> dict[str, object]:
    payload = _legal_proposal()
    payload.pop("mutation_scope")
    payload.pop("interaction_hypothesis")
    payload.update(
        {
            "version": "v000",
            "parent": None,
            "coding_initial_programs": 3,
            "coding_mutations": 1,
            "coding_mutation_children": 1,
            "coding_validation_folds": 3,
            "coding_validation_horizon": 8,
            "coding_skills": [],
            "retrieval_skills": [],
            "decision_skills": [],
        }
    )
    return payload


def test_meta_proposal_requires_exact_closed_schema() -> None:
    payload = _legal_proposal()
    payload["future_values"] = [999.0]

    with pytest.raises(ValueError, match="exact schema"):
        MetaHarnessProposal.from_payload(payload)


def test_four_initial_slots_have_ablation_and_joint_shapes() -> None:
    assert tuple(child_kind_for_slot(index) for index in range(4)) == (
        "coding",
        "retrieval",
        "decision",
        "joint",
    )
    assert child_kind_for_slot(8) == "joint"
    for invalid in (-1, True):
        with pytest.raises(ValueError, match="slot"):
            child_kind_for_slot(invalid)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"changelog": ""}, "non-empty"),
        ({"coding_initial_programs": True}, "integer"),
        ({"coding_validation_folds": 0}, "range"),
        ({"workflow": ["retrieve", "invent", "decide"]}, "workflow"),
        ({"workflow": ["retrieve"]}, "workflow"),
        ({"decision_aggregation": "mean"}, "aggregation"),
        ({"mutation_scope": ["coding", "coding"]}, "scope"),
        ({"mutation_scope": ["weights"]}, "scope"),
    ],
)
def test_meta_proposal_rejects_malformed_closed_values(
    mutation: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MetaHarnessProposal.from_payload(_legal_proposal(**mutation))


def test_meta_proposal_round_trip_is_canonical_and_detached() -> None:
    source = _legal_proposal(mutation_scope=["coding", "coordination"])
    proposal = MetaHarnessProposal.from_payload(source)
    source["workflow"] = ["decide"]

    assert proposal.to_payload() == _legal_proposal(
        mutation_scope=["coding", "coordination"]
    )


def test_policy_change_scope_ignores_lineage_and_frozen_skill_records() -> None:
    parent = _parent_payload()
    child = deepcopy(parent)
    child.update(
        {
            "version": "v001",
            "parent": "v000",
            "coding_generation_prompt": "Generate seasonal and robust candidates.",
            "decision_prompt": "Prefer jointly stable candidates.",
            "workflow": ["retrieve", "retrieve", "decide"],
            "changelog": "Joint mutation.",
            "coding_skills": [{"name": "snapshot changed outside this adapter"}],
        }
    )

    assert policy_change_scope(parent, child) == (
        "coding",
        "decision",
        "coordination",
    )


def test_child_scope_requires_exact_ablation_or_two_role_joint_change() -> None:
    validate_child_scope("coding", ("coding",), ("coding",))
    validate_child_scope(
        "joint",
        ("coding", "retrieval", "coordination"),
        ("coding", "retrieval", "coordination"),
    )

    with pytest.raises(ValueError, match="declared"):
        validate_child_scope("coding", ("coding",), ("decision",))
    with pytest.raises(ValueError, match="isolated"):
        validate_child_scope(
            "retrieval",
            ("retrieval", "coordination"),
            ("retrieval", "coordination"),
        )
    with pytest.raises(ValueError, match="two role"):
        validate_child_scope(
            "joint",
            ("coding", "coordination"),
            ("coding", "coordination"),
        )


def test_train_memory_is_strict_finite_and_contains_no_task_level_data() -> None:
    record = MetaTrainMemoryRecord(
        generation=1,
        child_kind="joint",
        changed_scopes=("coding", "decision"),
        interaction_hypothesis="Broader candidates need stricter final selection.",
        status="train_improved",
        parent_train_smae=1.2,
        parent_train_srmse=1.1,
        child_train_smae=1.0,
        child_train_srmse=0.9,
    )

    payload = record.to_payload()
    assert MetaTrainMemoryRecord.from_payload(payload) == record
    assert set(payload) == {
        "generation",
        "child_kind",
        "changed_scopes",
        "interaction_hypothesis",
        "status",
        "parent_train_smae",
        "parent_train_srmse",
        "child_train_smae",
        "child_train_srmse",
    }
    assert not any(
        token in str(payload).casefold()
        for token in ("task_id", "future_values", "document", "public", "dev_")
    )

    bad = dict(payload)
    bad["child_train_smae"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        MetaTrainMemoryRecord.from_payload(bad)

    bad = dict(payload)
    bad["interaction_hypothesis"] = "Improves task_42 specifically."
    with pytest.raises(ValueError, match="task-level"):
        MetaTrainMemoryRecord.from_payload(bad)


def test_invalid_memory_requires_absent_child_metrics() -> None:
    invalid = MetaTrainMemoryRecord(
        generation=0,
        child_kind="retrieval",
        changed_scopes=(),
        interaction_hypothesis="The typed Retrieval release is unavailable.",
        status="invalid",
        parent_train_smae=1.0,
        parent_train_srmse=1.0,
        child_train_smae=None,
        child_train_srmse=None,
    )
    assert MetaTrainMemoryRecord.from_payload(invalid.to_payload()) == invalid

    with pytest.raises(ValueError, match="invalid"):
        MetaTrainMemoryRecord(
            generation=0,
            child_kind="retrieval",
            changed_scopes=(),
            interaction_hypothesis="The proposal failed validation.",
            status="invalid",
            parent_train_smae=1.0,
            parent_train_srmse=1.0,
            child_train_smae=0.9,
            child_train_srmse=0.9,
        )


def test_memory_projection_keeps_last_three_generations_in_order() -> None:
    records = tuple(
        MetaTrainMemoryRecord(
            generation=generation,
            child_kind="coding",
            changed_scopes=("coding",),
            interaction_hypothesis=f"Generation {generation} tests Coding diversity.",
            status="train_rejected",
            parent_train_smae=1.0,
            parent_train_srmse=1.0,
            child_train_smae=1.1,
            child_train_srmse=1.2,
        )
        for generation in range(5)
    )

    projected = project_train_memory(records, window=3)

    assert [item["generation"] for item in projected] == [2, 3, 4]
    with pytest.raises(ValueError, match="window"):
        project_train_memory(records, window=0)


def _proposal_for_policy(
    parent: HarnessPolicy, scopes: tuple[str, ...], suffix: str
) -> str:
    payload = {
        "mutation_scope": list(scopes),
        "interaction_hypothesis": f"The {suffix} change should improve aggregate Train error.",
        "coding_generation_prompt": parent.coding_generation_prompt,
        "coding_revision_prompt": parent.coding_revision_prompt,
        "retrieval_prompt": parent.retrieval_prompt,
        "decision_prompt": parent.decision_prompt,
        "coding_initial_programs": parent.coding_initial_programs,
        "coding_mutations": parent.coding_mutations,
        "coding_mutation_children": parent.coding_mutation_children,
        "coding_validation_folds": parent.coding_validation_folds,
        "coding_validation_horizon": parent.coding_validation_horizon,
        "workflow": list(parent.workflow),
        "enable_evidence_adjustments": parent.enable_evidence_adjustments,
        "max_evidence_adjustments": parent.max_evidence_adjustments,
        "decision_aggregation": parent.decision_aggregation,
        "changelog": f"Test the {suffix} mutation.",
    }
    if "coding" in scopes:
        payload["coding_generation_prompt"] = f"{parent.coding_generation_prompt} {suffix}"
    if "retrieval" in scopes:
        payload["retrieval_prompt"] = f"{parent.retrieval_prompt} {suffix}"
    if "decision" in scopes:
        payload["decision_prompt"] = f"{parent.decision_prompt} {suffix}"
    if "coordination" in scopes:
        payload["workflow"] = ["retrieve", "retrieve", "decide"]
    return json.dumps(payload)


def _evaluation(version: str, smae: float, srmse: float) -> PolicyEvaluation:
    return PolicyEvaluation(
        version=version,
        system_reward=-srmse,
        module_rewards={"coding": 0.0, "retrieval": 0.0, "decision": 0.0},
        outcomes=(),
        diagnostics={"mean_smae": smae, "mean_srmse": srmse},
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "prompt"},
        {"target": "coding"},
        {"children_per_generation": 3},
        {"meta_memory_window": 0},
    ],
)
def test_meta_v2_configuration_fails_before_model_calls(
    overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="Meta-Harness V2"):
        CoEvolutionConfig(meta_harness_v2=True, **overrides)


def test_legacy_configuration_does_not_apply_meta_memory_constraints() -> None:
    assert CoEvolutionConfig(meta_memory_window=0).meta_harness_v2 is False


def test_meta_v2_requests_three_role_ablations_and_one_joint_child() -> None:
    parent = HarnessPolicy()
    replies = [
        _proposal_for_policy(parent, ("coding",), "coding-only"),
        _proposal_for_policy(parent, ("retrieval",), "retrieval-only"),
        _proposal_for_policy(parent, ("decision",), "decision-only"),
        _proposal_for_policy(parent, ("coding", "retrieval"), "joint"),
    ]
    llm = FakeLLMClient(replies)
    engine = CoEvolutionEngine(
        llm,
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(
            generations=1,
            children_per_generation=4,
            meta_harness_v2=True,
        ),
    )

    children = [
        engine.mutate(parent, _evaluation(parent.version, 1.0, 1.0), child_index=index)
        for index in range(4)
    ]

    assert [policy_change_scope(parent.to_payload(), child.to_payload()) for child in children] == [
        ("coding",),
        ("retrieval",),
        ("decision",),
        ("coding", "retrieval"),
    ]
    assert [
        json.loads(call["messages"][0]["content"])["requested_child_kind"]
        for call in llm.calls
    ] == ["coding", "retrieval", "decision", "joint"]


def test_meta_v2_rejects_declared_scope_drift_without_weakening_legacy() -> None:
    parent = HarnessPolicy()
    drift = _proposal_for_policy(parent, ("coding",), "coding-only")
    raw = json.loads(drift)
    raw["decision_prompt"] = "This undeclared Decision mutation must not survive."
    strict = CoEvolutionEngine(
        FakeLLMClient([json.dumps(raw)]),
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(children_per_generation=4, meta_harness_v2=True),
    ).mutate(parent, _evaluation(parent.version, 1.0, 1.0), child_index=0)

    legacy = CoEvolutionEngine(
        FakeLLMClient([json.dumps({"decision_prompt": "Legacy open proposal."})]),
        harness_factory=lambda _policy: object(),
    ).mutate(parent, _evaluation(parent.version, 1.0, 1.0))

    assert policy_change_scope(parent.to_payload(), strict.to_payload()) == ()
    assert "Invalid Meta-Harness V2" in strict.changelog
    assert legacy.decision_prompt == "Legacy open proposal."


class _DeterministicEngine(CoEvolutionEngine):
    def _evaluate(
        self,
        policy: HarnessPolicy,
        _tasks: object,
        *,
        stage: str,
        generation: int,
        learn_skills: bool,
        harness: object,
    ) -> PolicyEvaluation:
        del generation, learn_skills, harness
        if stage == "parent_dev":
            return _evaluation(policy.version, 8765.0, 9876.0)
        if policy.version == "v000":
            return _evaluation(policy.version, 1.0, 1.0)
        return _evaluation(policy.version, 1.2, 1.1)


def test_meta_v2_next_generation_sees_train_only_memory(tmp_path: Path) -> None:
    parent = HarnessPolicy()
    scopes = (
        ("coding",),
        ("retrieval",),
        ("decision",),
        ("coding", "decision"),
    )
    replies = [
        _proposal_for_policy(parent, scope, f"generation-{generation}-{index}")
        for generation in range(2)
        for index, scope in enumerate(scopes)
    ]
    llm = FakeLLMClient(replies)
    checkpoint = tmp_path / "checkpoint.json"
    engine = _DeterministicEngine(
        llm,
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(
            generations=2,
            children_per_generation=4,
            meta_harness_v2=True,
            checkpoint_path=checkpoint,
        ),
    )

    accepted, _history = engine.evolve(parent, [object()], [object()])

    assert accepted == parent
    second_generation = json.loads(llm.calls[4]["messages"][0]["content"])
    memory = second_generation["train_evolution_memory"]
    assert len(memory) == 4
    assert {item["generation"] for item in memory} == {0}
    assert {item["status"] for item in memory} == {"train_rejected"}
    serialized = json.dumps(memory)
    assert "8765" not in serialized
    assert "9876" not in serialized
    assert "task_id" not in serialized
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 2
    assert stored["meta_harness_v2"] is True
    assert stored["meta_memory"] == list(engine.meta_train_memory_payload())


def test_meta_v2_checkpoint_revalidates_memory_before_next_llm_call(
    tmp_path: Path,
) -> None:
    parent = HarnessPolicy()
    checkpoint = tmp_path / "checkpoint.json"
    engine = _DeterministicEngine(
        FakeLLMClient(
            [
                _proposal_for_policy(parent, ("coding",), "coding"),
                _proposal_for_policy(parent, ("retrieval",), "retrieval"),
                _proposal_for_policy(parent, ("decision",), "decision"),
                _proposal_for_policy(parent, ("coding", "decision"), "joint"),
            ]
        ),
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(
            generations=1,
            children_per_generation=4,
            meta_harness_v2=True,
            checkpoint_path=checkpoint,
        ),
    )
    engine.evolve(parent, [object()], [object()])
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["meta_memory"][0]["future_values"] = [999.0]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    resumed_llm = FakeLLMClient([])

    with pytest.raises(ValueError, match="exact schema"):
        _DeterministicEngine(
            resumed_llm,
            harness_factory=lambda _policy: object(),
            config=CoEvolutionConfig(
                generations=2,
                children_per_generation=4,
                meta_harness_v2=True,
                checkpoint_path=checkpoint,
            ),
        ).evolve(parent, [object()], [object()])

    assert resumed_llm.calls == []


def test_invalid_task_specific_hypothesis_is_safely_recorded_as_invalid() -> None:
    parent = HarnessPolicy()
    first = json.loads(_proposal_for_policy(parent, ("coding",), "coding"))
    first["interaction_hypothesis"] = "Tune specifically for task_42."
    replies = [
        json.dumps(first),
        _proposal_for_policy(parent, ("retrieval",), "retrieval"),
        _proposal_for_policy(parent, ("decision",), "decision"),
        _proposal_for_policy(parent, ("coding", "decision"), "joint"),
    ]
    engine = _DeterministicEngine(
        FakeLLMClient(replies),
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(
            generations=1,
            children_per_generation=4,
            meta_harness_v2=True,
        ),
    )

    engine.evolve(parent, [object()], [object()])

    memory = engine.meta_train_memory_payload()
    assert memory[0]["status"] == "invalid"
    assert "task_42" not in json.dumps(memory)


class _HalvingEngine(CoEvolutionEngine):
    def _evaluate(
        self,
        policy: HarnessPolicy,
        _tasks: object,
        *,
        stage: str,
        generation: int,
        learn_skills: bool,
        harness: object,
    ) -> PolicyEvaluation:
        del generation, learn_skills, harness
        if stage == "parent_dev":
            return _evaluation(policy.version, 1.0, 1.0)
        if stage == "child_dev":
            return _evaluation(policy.version, 0.12345, 0.12345)
        if stage == "child_screen_train" and policy.version == "v001":
            return _evaluation(policy.version, 0.9, 0.9)
        if stage == "child_train_remaining":
            return _evaluation(policy.version, 0.7, 0.7)
        return _evaluation(policy.version, 1.0, 1.0)


def test_successive_halving_memory_uses_train_screen_or_full_train_only() -> None:
    parent = HarnessPolicy()
    replies = [
        _proposal_for_policy(parent, ("coding",), "coding"),
        _proposal_for_policy(parent, ("retrieval",), "retrieval"),
        _proposal_for_policy(parent, ("decision",), "decision"),
        _proposal_for_policy(parent, ("coding", "decision"), "joint"),
    ]
    engine = _HalvingEngine(
        FakeLLMClient(replies),
        harness_factory=lambda _policy: object(),
        config=CoEvolutionConfig(
            generations=1,
            children_per_generation=4,
            meta_harness_v2=True,
            successive_halving=True,
            screening_train_tasks=1,
            screening_promote=1,
        ),
    )

    accepted, history = engine.evolve(parent, [object(), object()], [object()])

    assert accepted.version == "v001"
    assert history[0].promoted_versions == ("v001",)
    memory = engine.meta_train_memory_payload()
    assert memory[0]["status"] == "train_improved"
    assert memory[0]["child_train_smae"] == pytest.approx(0.8)
    assert {record["status"] for record in memory[1:]} == {"train_rejected"}
    assert "0.12345" not in json.dumps(memory)
