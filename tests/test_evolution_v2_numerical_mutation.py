"""Mutation ownership, atomicity, and Train-only credit regressions."""
import copy
from dataclasses import replace

import pytest

from evolving_loop.v2.numerical_qd.mutation import (
    MutationProposalV2, apply_mutation, record_train_outcome,
)
from evolving_loop.v2.numerical_qd.contracts import (
    MUTATION_OPERATORS, MutationStateV2, NumericalInventoryV2,
    NumericalMutationPolicyV2, NumericalProposerPromptV2,
)

SHA = "a" * 64
CELL = "b" * 64
OTHER_CELL = "c" * 64


def member(name="a", **updates):
    return dict(member_id=name, family="statistical", source_sha256=SHA,
                policy_sha256=SHA, parent_ids=[], applicability_cells=[CELL, OTHER_CELL],
                status="active") | updates


def parent_state(**updates):
    stats = dict(attempts=0, feasible=0, promotions=0, insertions=0, credit=0)
    return MutationStateV2(**(dict(
        inventory=NumericalInventoryV2(1, [member(), member("b")]),
        mutation_policy=NumericalMutationPolicyV2(1, {op: stats for op in MUTATION_OPERATORS}),
        proposer_prompt=NumericalProposerPromptV2(1, "Propose a Train mutation.",
            "numerical_mutation_batch_v1", 16000, sorted(MUTATION_OPERATORS), None),
        declared_cells=(CELL, OTHER_CELL), max_parents_per_child=2, max_inventory_size=4,
    ) | updates))


def operations():
    prompt = parent_state().proposer_prompt
    tuned = replace(prompt, template="Use Train diagnostics.", parent_prompt_sha256=prompt.fingerprint())
    return [
        dict(operator="add", reason="new method", member=member("c")),
        dict(operator="repair", reason="repair", member_id="a", replacement=member("c", parent_ids=["a"])),
        dict(operator="fork", reason="fork", member_id="a", child=member("c", parent_ids=["a"])),
        *[dict(operator=op, reason=op, parent_ids=["a", "b"],
               child=member("c", family="combined", parent_ids=["a", "b"]))
          for op in ("combine", "route", "crossover")],
        dict(operator="specialize", reason="narrow", member_id="a", applicability_cells=[CELL]),
        dict(operator="remove", reason="redundant", member_id="a"),
        dict(operator="quarantine", reason="failed", member_id="a"),
        dict(operator="policy_tune", reason="Train prompt", prompt=tuned.to_payload(), credit_delta={}),
    ]


@pytest.mark.parametrize("operation", operations())
def test_each_mutation_round_trips_and_changes_only_owned_fields(operation):
    parent = parent_state()
    before = parent.to_payload()
    proposal = MutationProposalV2.from_payload(operation)
    assert proposal.to_payload() == operation
    result = apply_mutation(parent, proposal)
    assert result.changed_fields == (("proposer_prompt",) if proposal.operator == "policy_tune" else ("inventory",))
    assert parent.to_payload() == before
    assert result.state is not parent
    if proposal.operator != "policy_tune":
        assert result.state.proposer_prompt == parent.proposer_prompt
        assert result.state.mutation_policy == parent.mutation_policy
        assert result.state.inventory != parent.inventory
    else:
        assert result.state.inventory == parent.inventory
    if proposal.operator == "repair":
        assert [m.member_id for m in result.state.inventory.members] == ["c", "b"]
    if proposal.operator == "quarantine":
        assert result.state.inventory.members[0].status == "quarantined"
    if proposal.operator == "specialize":
        assert result.state.inventory.members[0].applicability_cells == (CELL,)
        assert result.state.inventory.members[0].status == "specialized"


@pytest.mark.parametrize("operation", operations())
def test_tag_schema_and_nested_immutability(operation):
    for key in operation:
        bad = copy.deepcopy(operation)
        del bad[key]
        with pytest.raises(ValueError):
            MutationProposalV2.from_payload(bad)
    for key in ("code", "dev_metrics", "inventory", "member_ids"):
        with pytest.raises(ValueError):
            MutationProposalV2.from_payload(operation | {key: "injected"})
    proposal = MutationProposalV2.from_payload(operation)
    before = proposal.canonical_bytes()
    operation["reason"] = "changed"
    proposal.to_payload()["reason"] = "also changed"
    assert proposal.canonical_bytes() == before
    with pytest.raises((AttributeError, TypeError)):
        proposal.operator = "add"


@pytest.mark.parametrize("change", [
    {"source_sha256": "import os"}, {"policy_sha256": "not-sha"},
    {"member_id": "exec('bad')"}, {"parent_ids": ["a", "a"]},
    {"applicability_cells": ["arbitrary code"]}, {"family": "import os"},
    {"status": "active; exec()"}, {"code": "def forecast(): pass"},
])
def test_member_metadata_cannot_carry_code_or_duplicate_ids(change):
    with pytest.raises(ValueError):
        MutationProposalV2.from_payload(dict(operator="add", reason="new", member=member("c", **change)))


@pytest.mark.parametrize("operation", [
    dict(operator="add", reason="duplicate", member=member()),
    dict(operator="add", reason="foreign cell", member=member("c", applicability_cells=["d" * 64])),
    dict(operator="add", reason="lineage", member=member("c", parent_ids=["a"])),
    dict(operator="fork", reason="wrong lineage", member_id="a", child=member("c", parent_ids=["b"])),
    dict(operator="repair", reason="wrong target", member_id="missing", replacement=member("c", parent_ids=["a"])),
    dict(operator="combine", reason="too many", parent_ids=["a", "b", "d"], child=member("c", parent_ids=["a", "b", "d"])),
    dict(operator="crossover", reason="not two", parent_ids=["a"], child=member("c", parent_ids=["a"])),
    dict(operator="specialize", reason="undeclared", member_id="a", applicability_cells=["d" * 64]),
    dict(operator="policy_tune", reason="unearned credit", prompt=None, credit_delta={"add": 1}),
])
def test_invalid_mutation_is_atomic(operation):
    parent = parent_state()
    before = parent.canonical_bytes()
    with pytest.raises(ValueError):
        apply_mutation(parent, MutationProposalV2.from_payload(operation))
    assert parent.canonical_bytes() == before


@pytest.mark.parametrize("operator", ["remove", "quarantine", "specialize"])
def test_last_active_member_is_preserved(operator):
    parent = parent_state(inventory=NumericalInventoryV2(1, [member()]))
    payload = dict(operator=operator, reason="test", member_id="a")
    if operator == "specialize":
        payload["applicability_cells"] = [CELL]
    with pytest.raises(ValueError):
        apply_mutation(parent, MutationProposalV2.from_payload(payload))


def test_capacity_and_configured_parent_limit_are_enforced():
    parent = parent_state(max_inventory_size=2)
    with pytest.raises(ValueError):
        apply_mutation(parent, MutationProposalV2.from_payload(operations()[0]))
    with pytest.raises(ValueError):
        apply_mutation(parent_state(max_parents_per_child=1), MutationProposalV2.from_payload(operations()[3]))


def feedback(**updates):
    return dict(split="train", operator="fork", feasible=True, promoted=False,
                inserted=True, diagnostic_categories=[]) | updates


def test_two_generations_credit_and_prompt_only_change_from_train():
    parent = parent_state()
    child = record_train_outcome(parent, feedback())
    stats = child.mutation_policy.operators["fork"]
    assert (stats.attempts, stats.feasible, stats.promotions, stats.insertions, stats.credit) == (1, 1, 0, 1, 1)
    assert child.mutation_policy.fingerprint() != parent.mutation_policy.fingerprint()
    assert child.proposer_prompt.fingerprint() != parent.proposer_prompt.fingerprint()
    assert child.proposer_prompt.parent_prompt_sha256 == parent.proposer_prompt.fingerprint()
    before = child.canonical_bytes()
    for bad in (feedback(split="dev"), feedback(dev_metrics="UNIQUE_DEV_SENTINEL"),
                feedback(diagnostic_categories=["UNIQUE_DEV_SENTINEL"])):
        with pytest.raises(ValueError):
            record_train_outcome(child, bad)
    assert child.canonical_bytes() == before
    next_state = record_train_outcome(child, feedback(inserted=False, feasible=False))
    assert next_state.mutation_policy.operators["fork"].credit == 1
    assert next_state.proposer_prompt == child.proposer_prompt


def test_direct_state_construction_cannot_bypass_frozen_validation():
    parent = parent_state()
    with pytest.raises(ValueError):
        replace(parent, max_inventory_size=True)
    with pytest.raises(ValueError):
        replace(parent, declared_cells=["bad"])
    with pytest.raises(ValueError):
        replace(parent, max_inventory_size=1)
    with pytest.raises(ValueError):
        record_train_outcome(parent, feedback(feasible=False, inserted=True))
