"""Primitive-only proposal boundary, Host source normalization, charged fallback."""
from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass
from itertools import combinations

from common.llm import LLMClient
from common.payload import strict_json_loads
from common.sandbox import check_code
from numerical_agent.evolution.champion import ChampionRecipe, EvolutionAssumption
from numerical_agent.evolution.module import parse_method

from ..budget import BudgetLedger, ResourceUse
from ..contracts import _require_exact_schema, _strict_json_value, canonical_v2_bytes, fingerprint_payload
from .agent_methods import CurriculumTargetV2, VerifiedReusableProgramV2
from .contracts import (
    MEMBER_FAMILIES, MUTATION_OPERATORS, MutationStateV2, NumericalGenomeV2,
    NumericalProposerPromptV2, TrainMutationFeedbackV2,
    _sorted_strings,
)
from .mutation import OPERATION_KEYS, MutationProposalV2, _identifier, apply_mutation

REQUEST_KEYS = frozenset({
    "parent_genome", "parent_state", "selected_cells", "train_feedback",
    "curriculum_targets", "reusable_programs", "eligible_reusable_program_sha256s",
    "mutation_prompt", "mutation_prompt_population_sha256",
    "remaining_budget", "allowed_mutation_operators", "counter_draw",
    "max_proposals", "max_response_bytes",
})
LEGACY_CONTEXT_REQUEST_KEYS = REQUEST_KEYS - {
    "mutation_prompt", "mutation_prompt_population_sha256",
}
LEGACY_CONTEXT_UNCOMMITTED_REQUEST_KEYS = LEGACY_CONTEXT_REQUEST_KEYS - {
    "eligible_reusable_program_sha256s",
}
LEGACY_REQUEST_KEYS = LEGACY_CONTEXT_REQUEST_KEYS - {
    "curriculum_targets", "reusable_programs", "eligible_reusable_program_sha256s",
}
FAILURE_REASONS = frozenset({"unavailable", "timeout", "malformed", "empty", "budget_exhausted"})
_MAX_CONTEXT_RECORDS = 32
_CURRICULUM_REASON_ORDER = {
    "unoccupied": 0,
    "least_visited": 1,
    "failure_matched": 2,
}


class HostSourceValidationError(ValueError):
    """A syntactically valid proposal that the execution adapter cannot load."""


def _reject_provider_paths(value):
    """The accepted metadata language excludes POSIX/Windows path separators.

    Check keys as well as values: runtime fingerprint labels and prompt text
    otherwise admit arbitrary paths despite satisfying their artifact schemas.
    This applies only to provider input, never candidate source or Host schema.
    """
    if type(value) is str:
        if "/" in value or "\\" in value:
            raise ValueError("provider metadata must not contain filesystem paths")
    elif type(value) is dict:
        for key, item in value.items():
            _reject_provider_paths(key)
            _reject_provider_paths(item)
    elif type(value) is list:
        for item in value:
            _reject_provider_paths(item)


def primitive_proposer_request(**payload) -> dict:
    """Reject unknown data at every depth before any provider can receive it.

    Inputs are payloads, never live artifacts, paths, Store/Kernel handles or
    callbacks. Free-form evaluation text/forecasts are deliberately absent.
    """
    # Legacy request artifacts remain readable. New callers opt into the
    # committed context atomically by supplying all exact fields, including
    # empty lists.
    supplied = set(payload) - {"no_time_limit"}
    if "no_time_limit" in payload and type(payload["no_time_limit"]) is not bool:
        raise ValueError("no_time_limit must be a boolean")
    if supplied == LEGACY_REQUEST_KEYS:
        expected = LEGACY_REQUEST_KEYS
    elif supplied in (LEGACY_CONTEXT_REQUEST_KEYS, LEGACY_CONTEXT_UNCOMMITTED_REQUEST_KEYS):
        expected = supplied
    else:
        expected = REQUEST_KEYS
    if "no_time_limit" in payload:
        expected = set(expected) | {"no_time_limit"}
    values = _require_exact_schema(payload, expected, field="proposer request")
    values = _strict_json_value(values)
    metadata = dict(values)
    if type(metadata.get("reusable_programs")) is list:
        metadata["reusable_programs"] = [
            {key: value for key, value in record.items() if key != "source_text"}
            if type(record) is dict else record
            for record in metadata["reusable_programs"]
        ]
    _reject_provider_paths(metadata)
    state = MutationStateV2.from_payload(values["parent_state"])
    genome = NumericalGenomeV2.from_payload(values["parent_genome"])
    for label in genome.runtime_fingerprints:
        _identifier(label)
    for name, actual in (("inventory_sha256", state.inventory.fingerprint()),
                         ("mutation_policy_sha256", state.mutation_policy.fingerprint()),
                         ("proposer_prompt_sha256", state.proposer_prompt.fingerprint())):
        if getattr(genome, name) != actual:
            raise ValueError(f"Parent genome {name} mismatch")
    allowed = _sorted_strings(values["allowed_mutation_operators"], "allowed_mutation_operators",
        choices=MUTATION_OPERATORS, nonempty=True)
    if not set(allowed) <= (set(state.mutation_policy.operators) & set(state.proposer_prompt.allowed_mutation_operators)):
        raise ValueError("request expands Parent mutation authority")
    for name in ("counter_draw", "max_proposals", "max_response_bytes"):
        value = values[name]
        if type(value) is not int or value < (0 if name == "counter_draw" else 1):
            raise ValueError(f"{name} must be a bounded integer")
    if values["max_response_bytes"] > 1_048_576:
        raise ValueError("response cap exceeds artifact bound")
    ResourceUse.from_payload(values["remaining_budget"])
    members = {member.member_id for member in state.inventory.members}
    for member in state.inventory.members:
        _identifier(member.member_id)
        for parent in member.parent_ids:
            _identifier(parent)
    if type(values["selected_cells"]) is not list:
        raise ValueError("selected_cells must be a list")
    cells = []
    for entry in values["selected_cells"]:
        row = _require_exact_schema(entry, {"cell_sha256", "member_ids"}, field="selected cell")
        if row["cell_sha256"] not in state.declared_cells:
            raise ValueError("selected cell is undeclared")
        ids = _sorted_strings(row["member_ids"], "member_ids")
        if not set(ids) <= members:
            raise ValueError("cell summary contains unknown member")
        cells.append(row["cell_sha256"])
    if cells != sorted(set(cells)):
        raise ValueError("selected cells must be sorted and unique")
    if type(values["train_feedback"]) is not list:
        raise ValueError("train_feedback must be a list")
    for feedback in values["train_feedback"]:
        TrainMutationFeedbackV2.from_payload(feedback)
    if "curriculum_targets" in values:
        _validate_program_context(values, state, genome)
    if "mutation_prompt" in values:
        prompt = NumericalProposerPromptV2.from_payload(values["mutation_prompt"])
        if prompt.allowed_mutation_operators != ("policy_tune",):
            raise ValueError("mutation_prompt must authorize only policy_tune")
        from ..contracts import require_sha256
        require_sha256(values["mutation_prompt_population_sha256"], "mutation_prompt_population_sha256")
    return values


def _validate_program_context(values, state, genome):
    targets_payload = values["curriculum_targets"]
    programs_payload = values["reusable_programs"]
    if type(targets_payload) is not list or type(programs_payload) is not list:
        raise ValueError("proposer context fields must be lists")
    for name, records in (("curriculum_targets", targets_payload),
                          ("reusable_programs", programs_payload)):
        if len(records) > _MAX_CONTEXT_RECORDS:
            raise ValueError(f"{name} may contain at most {_MAX_CONTEXT_RECORDS} records")

    targets = tuple(CurriculumTargetV2.from_payload(item) for item in targets_payload)
    target_cells = tuple(target.cell.fingerprint() for target in targets)
    if not set(target_cells) <= set(state.declared_cells):
        raise ValueError("curriculum target cell is undeclared")
    target_order = tuple(
        (_CURRICULUM_REASON_ORDER[target.reason], target.cell.fingerprint())
        for target in targets
    )
    if target_order != tuple(sorted(target_order)) or len(target_cells) != len(set(target_cells)):
        raise ValueError("curriculum targets must be canonical and cell-unique")

    programs = tuple(
        VerifiedReusableProgramV2.from_payload(item) for item in programs_payload
    )
    if programs and not targets:
        raise ValueError("reusable programs require curriculum targets")
    target_set = set(target_cells)
    if any(not set(program.applicability_cells) <= set(state.declared_cells)
           or not target_set.intersection(program.applicability_cells)
           for program in programs):
        raise ValueError("reusable program applicability must match declared targets")
    program_order = tuple(
        (-len(target_set.intersection(program.applicability_cells)),
         program.source_sha256, program.fingerprint())
        for program in programs
    )
    source_shas = tuple(program.source_sha256 for program in programs)
    if program_order != tuple(sorted(program_order)) or len(source_shas) != len(set(source_shas)):
        raise ValueError("reusable programs must be canonical with unique source identities")

    parent_sha = genome.fingerprint()
    committed_payload = values.get("eligible_reusable_program_sha256s")
    if committed_payload is None:
        if any(program.genome_sha256 != parent_sha for program in programs):
            raise ValueError("non-parent reusable program lacks Host eligibility commitment")
    else:
        committed = _sorted_strings(
            committed_payload,
            "eligible_reusable_program_sha256s",
            sha=True,
        )
        if len(committed) > _MAX_CONTEXT_RECORDS:
            raise ValueError(
                f"eligible_reusable_program_sha256s may contain at most {_MAX_CONTEXT_RECORDS} records"
            )
        actual = tuple(sorted(program.fingerprint() for program in programs))
        if committed != actual:
            raise ValueError("reusable program lacks exact Host eligibility commitment")

    members = {member.member_id: member for member in state.inventory.members}
    for program in programs:
        if program.genome_sha256 != parent_sha:
            continue
        member = members.get(program.member_id)
        if member is None:
            raise ValueError("parent program names an unknown member")
        if member.status == "quarantined":
            raise ValueError("quarantined parent program cannot cross proposer boundary")
        if (program.source_sha256 != member.source_sha256
                or program.applicability_cells != member.applicability_cells):
            raise ValueError("parent program identity conflicts with inventory")


@dataclass(frozen=True, slots=True)
class ProviderAttemptV2:
    provider: str
    resource_use: ResourceUse
    failure_reason: str | None

    def __post_init__(self):
        if self.provider not in {"llm", "deterministic"}:
            raise ValueError("unknown attempt provider")
        if type(self.resource_use) is not ResourceUse:
            raise ValueError("resource_use must be ResourceUse")
        if self.failure_reason is not None and self.failure_reason not in FAILURE_REASONS:
            raise ValueError("unknown failure reason")


@dataclass(frozen=True, slots=True)
class NormalizedProposalBatchV2:
    proposals: tuple[MutationProposalV2, ...]
    source_artifacts: tuple[tuple[str, bytes], ...]
    resource_use: ResourceUse
    provider: str
    failure_reason: str | None
    attempts: tuple[ProviderAttemptV2, ...]

    def __post_init__(self):
        proposals = tuple(self.proposals)
        artifacts = tuple(tuple(item) for item in self.source_artifacts)
        attempts = tuple(self.attempts)
        if any(type(item) is not MutationProposalV2 for item in proposals):
            raise ValueError("batch requires typed proposals")
        if any(len(item) != 2 or type(item[1]) is not bytes
               or hashlib.sha256(item[1]).hexdigest() != item[0] for item in artifacts):
            raise ValueError("source artifacts must be immutable digest-bound bytes")
        if [item[0] for item in artifacts] != sorted({item[0] for item in artifacts}):
            raise ValueError("source artifacts must be sorted and unique")
        if self.provider not in {"llm", "deterministic", "hybrid"}:
            raise ValueError("unknown provider")
        if self.failure_reason is not None and self.failure_reason not in FAILURE_REASONS:
            raise ValueError("unknown failure reason")
        if self.failure_reason is not None and (proposals or artifacts):
            raise ValueError("failed batch must be empty")
        total = ResourceUse()
        for attempt in attempts:
            if type(attempt) is not ProviderAttemptV2:
                raise ValueError("attempts must be typed")
            total = total + attempt.resource_use
        if type(self.resource_use) is not ResourceUse or total != self.resource_use:
            raise ValueError("batch resources must equal closed attempts")
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "source_artifacts", artifacts)
        object.__setattr__(self, "attempts", attempts)


def _batch(provider, proposals=(), artifacts=(), use=ResourceUse(), reason=None):
    return NormalizedProposalBatchV2(tuple(proposals), tuple(artifacts), use, provider, reason,
        (ProviderAttemptV2(provider, use, reason),))


def host_source_recipe(source: str | bytes) -> ChampionRecipe:
    """Derive the only executable recipe a newly supplied source can claim."""
    if type(source) is bytes:
        source = source.decode("utf-8")
    if type(source) is not str:
        raise TypeError("source recipe requires UTF-8 source text")
    method = parse_method(source)
    assumption = EvolutionAssumption(
        method.name + "_history",
        method.name,
        "history_length",
        "above",
        "full",
        "select",
        "History supports the candidate.",
        "History is unavailable.",
    )
    return ChampionRecipe(
        "select_" + method.name,
        "select",
        (method.name,),
        method.name,
        (assumption,),
    )


def _normalize_sources(response, request):
    # Import locally so the proposal boundary can use the exact execution gate
    # without coupling the module import graph to the legacy adapter.
    from .adapters import LegacyNumericalAdapter

    raw = _require_exact_schema(response, {"source_candidates", "proposals"}, field="LLM response")
    if type(raw["source_candidates"]) is not list or type(raw["proposals"]) is not list:
        raise ValueError("response arrays required")
    if len(raw["proposals"]) > request["max_proposals"]:
        raise ValueError("too many proposals")
    if len(raw["source_candidates"]) > len(raw["proposals"]):
        raise ValueError("unowned source candidate")
    sources = {}
    for candidate in raw["source_candidates"]:
        row = _require_exact_schema(candidate, {"local_id", "code"}, field="source candidate")
        identity = _identifier(row["local_id"])
        if identity in sources or (len(identity) == 64 and all(c in "0123456789abcdef" for c in identity)):
            raise ValueError("source IDs must be unique request-local names, not digests")
        code = row["code"]
        if type(code) is not str or not code.strip():
            raise ValueError("source code required")
        code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"
        # parse_method validates the forecasting signature/docstring. It permits
        # unrelated module statements, so close that gap before the shared gate.
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("source must contain exactly one forecast method")
        recipe = host_source_recipe(code)
        check_code(code)
        try:
            LegacyNumericalAdapter.validate_source(code)
        except (TypeError, ValueError) as error:
            raise HostSourceValidationError(str(error)) from error
        source_bytes = code.encode("utf-8")
        sources[identity] = (
            hashlib.sha256(source_bytes).hexdigest(), source_bytes, recipe
        )
    state = MutationStateV2.from_payload(request["parent_state"])
    proposals, used, owned = [], set(), set()
    for payload in raw["proposals"]:
        value = _strict_json_value(payload)
        if not isinstance(value, dict):
            raise ValueError("proposal must be an object")
        for key in ("member", "child", "replacement"):
            if key in value:
                member = value[key]
                if not isinstance(member, dict):
                    raise ValueError("member must be an object")
                local_id = member.get("source_sha256")
                if type(local_id) is not str or local_id not in sources or local_id in used:
                    raise ValueError("source reference must have one request-local owner")
                if "policy_sha256" in member:
                    raise ValueError("source policy identity is Host-derived")
                used.add(local_id)
                member["source_sha256"] = sources[local_id][0]
                member["policy_sha256"] = fingerprint_payload(
                    sources[local_id][2].to_payload()
                )
        proposal = MutationProposalV2.from_payload(value)
        if proposal.operator not in request["allowed_mutation_operators"]:
            raise ValueError("operator not allowed in this request")
        apply_mutation(state, proposal)
        target = value.get("member_id")
        children = [value[key]["member_id"] for key in ("member", "child", "replacement") if key in value]
        targets = set(children + ([target] if target is not None else []))
        if proposal.operator == "policy_tune":
            targets.add("__prompt__")
        if targets & owned:
            raise ValueError("batch has duplicate mutation ownership")
        owned.update(targets)
        proposals.append(proposal)
    if used != set(sources):
        raise ValueError("unowned source candidate")
    artifacts = tuple(sorted({(value[0], value[1]) for value in sources.values()}))
    return tuple(proposals), artifacts


class DeterministicProposalProvider:
    """Select one legal action from a canonical sorted list with the supplied draw.

    Structural children reuse verified Parent source identities and commit
    their typed structure to a canonical policy identity. Source changes go
    through the LLM/Host parser gate.
    """

    def __init__(self, *, monotonic=time.monotonic):
        self.monotonic = monotonic

    def propose(self, request) -> NormalizedProposalBatchV2:
        request = primitive_proposer_request(**request)
        if not request.get("no_time_limit", False) and request["remaining_budget"]["wall_seconds"] <= 0:
            return _batch("deterministic", reason="budget_exhausted")
        started = self.monotonic()
        state = MutationStateV2.from_payload(request["parent_state"])
        allowed = set(request["allowed_mutation_operators"])
        candidates = []
        members = sorted(state.inventory.members, key=lambda item: item.member_id)
        for member in members:
            candidates.append(_structural_candidate("repair", [member], member.applicability_cells))
            for op in ("remove", "quarantine"):
                candidates.append(dict(operator=op, reason="Train inventory maintenance", member_id=member.member_id))
            for cell in member.applicability_cells:
                candidates.append(dict(operator="specialize", reason="Train cell specialization", member_id=member.member_id,
                    applicability_cells=[cell]))
                if len(member.applicability_cells) > 1:
                    candidates.append(_structural_candidate("fork", [member], [cell]))
        for parents in combinations(members, 2):
            if parents[0].family != parents[1].family or any(m.status == "quarantined" for m in parents):
                continue
            cells = sorted(set(parents[0].applicability_cells) | set(parents[1].applicability_cells))
            for op in ("combine", "route"):
                candidates.append(_structural_candidate(op, parents, cells))
        prompt = state.proposer_prompt.to_payload() | dict(
            parent_prompt_sha256=state.proposer_prompt.fingerprint(),
            template="Propose a bounded mutation using only Train feasibility and declared cells.")
        candidates.append(dict(operator="policy_tune", reason="Train prompt variant", prompt=prompt, credit_delta={}))
        preferred = min(allowed, key=lambda op: (-state.mutation_policy.operators[op].credit, op))
        credit_prompt = prompt | dict(template=f"Propose a bounded Train mutation. Prefer {preferred} using recorded Train insertion credit.")
        candidates.append(dict(operator="policy_tune", reason="Recorded Train credit preference", prompt=credit_prompt, credit_delta={}))
        feasible = []
        for candidate in candidates:
            if candidate["operator"] not in allowed:
                continue
            try:
                proposal = MutationProposalV2.from_payload(candidate)
                apply_mutation(state, proposal)
            except ValueError:
                continue
            feasible.append(proposal)
        feasible.sort(key=lambda proposal: (proposal.operator, proposal.canonical_bytes()))
        use = ResourceUse(wall_seconds=max(0.0, float(self.monotonic() - started)))
        if not request.get("no_time_limit", False) and use.wall_seconds >= request["remaining_budget"]["wall_seconds"]:
            return _batch("deterministic", use=use, reason="budget_exhausted")
        if not feasible:
            return _batch("deterministic", use=use, reason="empty")
        return _batch("deterministic", [feasible[request["counter_draw"] % len(feasible)]], use=use)


def _structural_candidate(operator, parents, cells):
    """Policy digest binds the reusable source, lineage, and typed structure.

    Task 8 can reconstruct this policy payload from the Parent and proposal;
    this provider performs no source or policy persistence.
    """
    cells = list(cells)
    policy = dict(schema_version=1, operator=operator,
        parents=[parent.to_payload() for parent in parents], applicability_cells=cells)
    digest = fingerprint_payload(policy)
    child = parents[0].to_payload() | dict(member_id=f"{operator}_{digest[:24]}",
        policy_sha256=digest, parent_ids=[parent.member_id for parent in parents],
        family="combined" if operator in {"combine", "route"} else parents[0].family,
        applicability_cells=cells, status="active")
    if operator in {"repair", "fork"}:
        return dict(operator=operator, reason="Train structural mutation", member_id=parents[0].member_id,
            **{"replacement" if operator == "repair" else "child": child})
    return dict(operator=operator, reason="Train structural mutation", parent_ids=child["parent_ids"], child=child)


def _closed_object_schema(properties):
    return dict(type="object", properties=properties,
        required=sorted(properties), additionalProperties=False)


def _response_schema(request, state):
    """Host-owned wire grammar; prompt evolution cannot change its authority.

    Exact operation keys come from the same tagged parser. Contextual lineage,
    source ownership, canonical ordering, byte limits, and atomic feasibility
    are still enforced by normalization, not delegated to the model.
    """
    identity = dict(type="string", minLength=1, maxLength=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    local_id = identity | {"not": {"pattern": "^[0-9a-f]{64}$"}}
    cells = dict(type="array", uniqueItems=True, items={"enum": list(state.declared_cells)},
        description="Sorted declared cell identities only.")
    member_ids = [member.member_id for member in state.inventory.members]
    parents = dict(type="array", uniqueItems=True, maxItems=state.max_parents_per_child,
        items={"enum": sorted(member_ids)}, description="Sorted exact owned parent IDs.")
    member = _closed_object_schema(dict(
        member_id=identity | {"not": {"enum": sorted(member_ids)}},
        family={"enum": sorted(MEMBER_FAMILIES)},
        source_sha256=local_id | {"description": "Reference one source_candidates.local_id from this response; never a source SHA or a path."},
        parent_ids=parents, applicability_cells=cells,
        status={"const": "active"}))
    prompt = _closed_object_schema(dict(
        schema_version={"const": 1},
        template=dict(type="string", minLength=1, maxLength=65536,
            description="Bounded prompt text, at most 65536 UTF-8 bytes; no filesystem paths."),
        response_schema={"const": state.proposer_prompt.response_schema},
        max_response_bytes=dict(type="integer", minimum=1, maximum=state.proposer_prompt.max_response_bytes),
        allowed_mutation_operators=dict(type="array", minItems=1, uniqueItems=True,
            items={"enum": list(state.proposer_prompt.allowed_mutation_operators)}),
        parent_prompt_sha256={"const": state.proposer_prompt.fingerprint()}))
    fields = dict(
        reason=dict(type="string", minLength=1, maxLength=2048,
            description="Nonempty metadata reason, at most 2048 UTF-8 bytes; no executable source."),
        member={"$ref": "#/$defs/member"}, child={"$ref": "#/$defs/member"},
        replacement={"$ref": "#/$defs/member"}, member_id={"enum": sorted(member_ids)},
        parent_ids=parents | {"minItems": 2}, applicability_cells=cells | {"minItems": 1},
        prompt={"$ref": "#/$defs/prompt"},
        credit_delta=dict(type="object", maxProperties=0, additionalProperties=False,
            description="Exactly {}. Only the Host records Train credit."))
    operations = []
    for operator in request["allowed_mutation_operators"]:
        properties = {name: fields[name] for name in sorted(OPERATION_KEYS[operator] - {"operator"})}
        properties["operator"] = {"const": operator}
        if operator == "add":
            properties["member"] = _closed_object_schema({
                **member["properties"], "parent_ids": {"const": []},
            })
        elif operator in {"fork", "repair"}:
            key = "replacement" if operator == "repair" else "child"
            properties[key] = _closed_object_schema({
                **member["properties"],
                "parent_ids": parents | {"minItems": 1, "maxItems": 1,
                    "description": "Exactly [member_id] from this operation; must match the owned parent."},
            })
        if operator == "crossover":
            properties["parent_ids"] = parents | {"minItems": 2, "maxItems": min(2, state.max_parents_per_child)}
        operations.append(_closed_object_schema(properties))
    source_candidate = _closed_object_schema(dict(local_id=local_id,
        code=dict(type="string", minLength=1, description=(
            "One top-level Python function taking exactly (history, horizon, frequency), "
            "with an applicability docstring. Imports belong inside the function and must "
            "pass the Host sandbox gate. Never use `%` (including numeric remainder); use "
            "`divmod` instead. Never mutate a collection through methods such as `.sort()`; "
            "use `sorted(...)`. Avoid f-strings and percent formatting. A valid pattern is: "
            "def recent_mean(history, horizon, frequency):\n"
            "    \"\"\"Use a recent local level for short noisy series.\"\"\"\n"
            "    if not history:\n"
            "        return [0.0] * horizon\n"
            "    window = history[-min(len(history), 8):]\n"
            "    level = sum(window) / len(window)\n"
            "    return [level] * horizon"))))
    envelope = _closed_object_schema(dict(
        source_candidates=dict(type="array", maxItems=request["max_proposals"],
            items={"$ref": "#/$defs/source_candidate"}),
        proposals=dict(type="array", maxItems=request["max_proposals"], items={"oneOf": operations})))
    return envelope | {"$defs": dict(source_candidate=source_candidate, member=member, prompt=prompt)}


def _llm_limits(request):
    state = MutationStateV2.from_payload(request["parent_state"])
    remaining = ResourceUse.from_payload(request["remaining_budget"])
    encoded = canonical_v2_bytes(dict(request=request,
        allowed_mutation_response_schema=_response_schema(request, state))).decode("utf-8")
    system = state.proposer_prompt.template
    if "mutation_prompt" in request:
        system += "\n\nMutation prompt:\n" + request["mutation_prompt"]["template"]
    cap = min(request["max_response_bytes"], state.proposer_prompt.max_response_bytes, remaining.output_tokens)
    # LLMClient exposes text only: UTF-8 byte counts conservatively bound tokens.
    input_bound = len(encoded.encode()) + len(system.encode())
    return remaining, encoded, system, cap, input_bound


class LLMProposalProvider:
    def __init__(self, client: LLMClient | None, *, monotonic=time.monotonic):
        self.client = client
        self.monotonic = monotonic

    def propose(self, request) -> NormalizedProposalBatchV2:
        request = primitive_proposer_request(**request)
        remaining, encoded, system, cap, input_bound = _llm_limits(request)
        unlimited = request.get("no_time_limit", False)
        if remaining.llm_calls < 1 or (not unlimited and remaining.wall_seconds <= 0) or remaining.input_tokens < input_bound or cap < 1:
            return _batch("llm", reason="budget_exhausted")
        started = self.monotonic()
        proposals, artifacts, reason = (), (), None
        output_bytes = input_bytes = call_count = 0
        try:
            if self.client is None:
                call_count = 1
                reason = "unavailable"
            else:
                messages = [{"role": "user", "content": encoded}]
                for attempt in range(2):
                    call_input = len(system.encode()) + sum(
                        len(message["content"].encode()) for message in messages
                    )
                    available_output = min(cap, remaining.output_tokens - output_bytes)
                    if (call_count >= remaining.llm_calls
                            or input_bytes + call_input > remaining.input_tokens
                            or available_output < 1):
                        reason = "budget_exhausted"
                        break
                    call_count += 1
                    input_bytes += call_input
                    response = self.client.complete(
                        system=system, messages=messages, temperature=0.0
                    )
                    if type(response.text) is not str:
                        raise ValueError("response text required")
                    response_bytes = len(response.text.encode("utf-8"))
                    output_bytes += response_bytes
                    if response_bytes > available_output:
                        raise ValueError("response byte limit exceeded")
                    try:
                        parsed = strict_json_loads(response.text, context="Numerical mutation batch")
                        proposals, artifacts = _normalize_sources(parsed, request)
                    except (ValueError, TypeError, SyntaxError) as error:
                        if attempt or remaining.llm_calls < 2:
                            raise
                        repair = canonical_v2_bytes({
                            "instruction": "Return one complete corrected response using the same schema.",
                            "host_source_validation_error": str(error)[:2048],
                            "proposal_validation_error": str(error)[:2048],
                        }).decode("utf-8")
                        messages = [
                            *messages,
                            {"role": "assistant", "content": response.text},
                            {"role": "user", "content": "Host source validation failed. " + repair},
                        ]
                        continue
                    if not proposals:
                        reason = "empty"
                    break
        except TimeoutError:
            reason = "timeout"
        except (ValueError, TypeError, SyntaxError, RecursionError):
            reason = "malformed"
        except Exception:
            reason = "unavailable"
        use = ResourceUse(wall_seconds=max(0.0, float(self.monotonic() - started)), llm_calls=call_count,
            input_tokens=input_bytes, output_tokens=output_bytes)
        if (not unlimited and use.wall_seconds >= remaining.wall_seconds) or any(
                getattr(use, name) > getattr(remaining, name) for name in ResourceUse.field_names()
                if name != "wall_seconds" or not unlimited):
            reason = "budget_exhausted"
        return _batch("llm", proposals if reason is None else (), artifacts if reason is None else (), use, reason)


class HybridProposalProvider:
    """Budget ledger stays with the Host; only primitives cross propose()."""

    def __init__(self, llm: LLMProposalProvider, deterministic: DeterministicProposalProvider, ledger: BudgetLedger):
        self.llm = llm
        self.deterministic = deterministic
        self.ledger = ledger

    def propose(self, request) -> NormalizedProposalBatchV2:
        request = primitive_proposer_request(**request)
        remaining, _, _, cap, input_bound = _llm_limits(request)
        stage = "numerical-proposal-" + fingerprint_payload(request)
        attempts = []
        estimate = ResourceUse(
            wall_seconds=remaining.wall_seconds,
            llm_calls=min(2, remaining.llm_calls),
            input_tokens=remaining.input_tokens if remaining.llm_calls > 1 else input_bound,
            output_tokens=remaining.output_tokens if remaining.llm_calls > 1 else cap,
        )
        permit = self.ledger.reserve_stage(stage + "-llm", estimate)
        if permit.allowed:
            result = self.llm.propose(request)
            closed = self.ledger.close_stage(permit, result.resource_use)
            attempts.extend(result.attempts)
            if not closed.allowed or not self.ledger.can_open_stage(ResourceUse()).allowed:
                return self._result(attempts, reason="budget_exhausted")
            if result.proposals:
                return self._result(attempts, result.proposals, result.source_artifacts)
            available = {
                name: max(0.0 if type(value) is float else 0, value - getattr(result.resource_use, name))
                for name, value in request["remaining_budget"].items()
            }
            request = primitive_proposer_request(**(request | {"remaining_budget": available}))
        if not request.get("no_time_limit", False) and request["remaining_budget"]["wall_seconds"] <= 0:
            return self._result(attempts, reason="budget_exhausted")
        fallback = self.ledger.reserve_stage(stage + "-deterministic",
            ResourceUse(wall_seconds=request["remaining_budget"]["wall_seconds"]))
        if not fallback.allowed:
            return self._result(attempts, reason="budget_exhausted")
        result = self.deterministic.propose(request)
        closed = self.ledger.close_stage(fallback, result.resource_use)
        attempts.extend(result.attempts)
        if not closed.allowed or not self.ledger.can_open_stage(ResourceUse()).allowed:
            return self._result(attempts, reason="budget_exhausted")
        return self._result(attempts, result.proposals, result.source_artifacts, result.failure_reason)

    @staticmethod
    def _result(attempts, proposals=(), artifacts=(), reason=None):
        use = ResourceUse()
        for attempt in attempts:
            use = use + attempt.resource_use
        return NormalizedProposalBatchV2(tuple(proposals), tuple(artifacts), use, "hybrid", reason, tuple(attempts))
