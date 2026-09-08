from __future__ import annotations

from copy import deepcopy

import pytest

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
