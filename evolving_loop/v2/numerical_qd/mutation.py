"""Closed tagged mutations and atomic, immutable Host transitions."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from ..contracts import (
    _freeze_json_value, _require_exact_schema, _strict_json_value,
    canonical_v2_bytes, fingerprint_payload,
)
from .contracts import (
    MutationStateV2, NumericalInventoryV2, NumericalMemberV2,
    NumericalMutationPolicyV2, NumericalProposerPromptV2, TrainMutationFeedbackV2,
    _sorted_strings,
)

OPERATION_KEYS = MappingProxyType({
    "add": frozenset({"operator", "reason", "member"}),
    "repair": frozenset({"operator", "reason", "member_id", "replacement"}),
    "fork": frozenset({"operator", "reason", "member_id", "child"}),
    "combine": frozenset({"operator", "reason", "parent_ids", "child"}),
    "route": frozenset({"operator", "reason", "parent_ids", "child"}),
    "specialize": frozenset({"operator", "reason", "member_id", "applicability_cells"}),
    "crossover": frozenset({"operator", "reason", "parent_ids", "child"}),
    "remove": frozenset({"operator", "reason", "member_id"}),
    "quarantine": frozenset({"operator", "reason", "member_id"}),
    "policy_tune": frozenset({"operator", "reason", "prompt", "credit_delta"}),
})


def _identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError("member/local ID must be a bounded metadata identifier")
    return value


def _validate_proposal(payload):
    values = _strict_json_value(payload)
    if not isinstance(values, dict) or type(values.get("operator")) is not str:
        raise ValueError("mutation must have an operator tag")
    operator = values["operator"]
    if operator not in OPERATION_KEYS:
        raise ValueError("unknown mutation operator")
    _require_exact_schema(values, OPERATION_KEYS[operator], field=operator)
    reason = values["reason"]
    if type(reason) is not str or not reason.strip() or len(reason.encode()) > 2048:
        raise ValueError("reason must be bounded nonempty text")
    if "\n" in reason or "`" in reason or re.search(r"\b(import\s+\w|def\s+\w|(?:exec|eval|open)\s*\()", reason):
        raise ValueError("reason must be metadata, not source code")
    if "member_id" in values:
        _identifier(values["member_id"])
    if "parent_ids" in values:
        parents = _sorted_strings(values["parent_ids"], "parent_ids", nonempty=True)
        for parent in parents:
            _identifier(parent)
        if len(parents) < 2 or (operator == "crossover" and len(parents) != 2):
            raise ValueError("multi-parent operation requires at least two (crossover exactly two)")
    if "applicability_cells" in values:
        _sorted_strings(values["applicability_cells"], "applicability_cells", sha=True, nonempty=True)
    for key in ("member", "child", "replacement"):
        if key in values:
            member = NumericalMemberV2.from_payload(values[key])
            _identifier(member.member_id)
            for parent in member.parent_ids:
                _identifier(parent)
    if operator == "policy_tune":
        if values["prompt"] is not None:
            NumericalProposerPromptV2.from_payload(values["prompt"])
        if type(values["credit_delta"]) is not dict or values["credit_delta"]:
            raise ValueError("proposal cannot grant credit; Host records Train outcomes")
        if values["prompt"] is None:
            raise ValueError("policy_tune requires a prompt variant")
    return values


@dataclass(frozen=True, slots=True)
class MutationProposalV2:
    """The tag owns an exact payload; construction cannot bypass validation."""

    payload: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "payload", _freeze_json_value(_validate_proposal(self.payload)))

    @classmethod
    def from_payload(cls, payload):
        return cls(payload)

    @property
    def operator(self):
        return self.payload["operator"]

    def to_payload(self):
        return _strict_json_value(self.payload)

    def canonical_bytes(self):
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self):
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class MutationResultV2:
    state: MutationStateV2
    changed_fields: tuple[str, ...]

    def __post_init__(self):
        if type(self.state) is not MutationStateV2:
            raise ValueError("state must be MutationStateV2")
        fields = _sorted_strings(self.changed_fields, "changed_fields", nonempty=True,
            choices=frozenset({"inventory", "mutation_policy", "proposer_prompt"}))
        object.__setattr__(self, "changed_fields", fields)


def apply_mutation(parent: MutationStateV2, proposal: MutationProposalV2) -> MutationResultV2:
    """Validate complete ownership and limits before constructing a new state."""
    if type(parent) is not MutationStateV2 or type(proposal) is not MutationProposalV2:
        raise ValueError("typed parent and proposal required")
    values = proposal.to_payload()
    op = proposal.operator
    if op not in parent.mutation_policy.operators or op not in parent.proposer_prompt.allowed_mutation_operators:
        raise ValueError("operator not allowed by Parent policy/prompt")
    if op == "policy_tune":
        prompt = NumericalProposerPromptV2.from_payload(values["prompt"])
        if prompt.parent_prompt_sha256 != parent.proposer_prompt.fingerprint():
            raise ValueError("prompt must bind its Parent")
        if (not set(prompt.allowed_mutation_operators) <= set(parent.proposer_prompt.allowed_mutation_operators)
                or prompt.max_response_bytes > parent.proposer_prompt.max_response_bytes):
            raise ValueError("prompt may not expand mutation authority")
        return MutationResultV2(replace(parent, proposer_prompt=prompt), ("proposer_prompt",))

    members = list(parent.inventory.members)
    by_id = {member.member_id: member for member in members}
    targets = values.get("parent_ids", [values["member_id"]] if "member_id" in values else [])
    if len(targets) > parent.max_parents_per_child:
        raise ValueError("too many parents")
    if any(target not in by_id for target in targets):
        raise ValueError("unknown target member")
    if op in {"add", "repair", "fork", "combine", "route", "crossover"}:
        key = "member" if op == "add" else "replacement" if op == "repair" else "child"
        child = NumericalMemberV2.from_payload(values[key])
        if child.member_id in by_id:
            raise ValueError("child must have a new member ID")
        if child.parent_ids != tuple(targets):
            raise ValueError("child lineage must exactly match owned parents")
        if op == "add" and child.family not in {"statistical", "program"}:
            raise ValueError("add supports Statistical/program members only")
        if op in {"combine", "route"} and child.family != "combined":
            raise ValueError("combine/route require combined family")
        if op == "crossover" and len({by_id[target].family for target in targets}) != 1:
            raise ValueError("crossover parents must have compatible families")
        if child.status != "active":
            raise ValueError("new child must be active")
        if op == "repair":
            members[members.index(by_id[values["member_id"]])] = child
        else:
            members.append(child)
    else:
        target = by_id[values["member_id"]]
        index = members.index(target)
        if op == "remove":
            members.pop(index)
        elif op == "quarantine":
            members[index] = replace(target, status="quarantined")
        elif op == "specialize":
            cells = tuple(values["applicability_cells"])
            if not set(cells) < set(target.applicability_cells):
                raise ValueError("specialize must strictly narrow applicability")
            members[index] = replace(target, status="specialized", applicability_cells=cells)
    inventory = NumericalInventoryV2(1, members)
    # Preserve every previously executable cell, not just the active-member count.
    covered_before = {cell for m in parent.inventory.members if m.status != "quarantined" for cell in m.applicability_cells}
    covered_after = {cell for m in inventory.members if m.status != "quarantined" for cell in m.applicability_cells}
    if not covered_before <= covered_after:
        raise ValueError("mutation removes required coverage")
    if inventory == parent.inventory:
        raise ValueError("mutation must change its owned state")
    return MutationResultV2(replace(parent, inventory=inventory), ("inventory",))


def record_train_outcome(parent: MutationStateV2, feedback) -> MutationStateV2:
    """Host-only credit: one integer per feasible Train archive insertion."""
    outcome = TrainMutationFeedbackV2.from_payload(feedback)
    if outcome.operator not in parent.mutation_policy.operators:
        raise ValueError("feedback operator not in mutation policy")
    stats = parent.mutation_policy.operators[outcome.operator]
    updated = replace(stats, attempts=stats.attempts + 1,
        feasible=stats.feasible + int(outcome.feasible), promotions=stats.promotions + int(outcome.promoted),
        insertions=stats.insertions + int(outcome.inserted), credit=stats.credit + int(outcome.inserted))
    operators = dict(parent.mutation_policy.operators) | {outcome.operator: updated}
    policy = NumericalMutationPolicyV2(1, operators)
    prompt = parent.proposer_prompt
    if outcome.inserted:
        # Host-generated bounded text prevents raw diagnostics or labels entering memory.
        prompt = replace(prompt, template=f"Propose a Train mutation. Prefer {outcome.operator}; Train insertion credit {updated.credit}.",
            parent_prompt_sha256=prompt.fingerprint())
    return replace(parent, mutation_policy=policy, proposer_prompt=prompt)
