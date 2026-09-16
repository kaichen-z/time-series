"""Real proposer boundaries, response ownership, and charged fallback."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from common.llm import LLMResponse
from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.numerical_qd.agent_methods import (
    CurriculumTargetV2, VerifiedReusableProgramV2,
)
from evolving_loop.v2.numerical_qd.contracts import (
    MorphologyCellV2, MutationStateV2, NumericalGenomeV2,
    NumericalInventoryV2,
)
from evolving_loop.v2.numerical_qd.proposers import (
    DeterministicProposalProvider, HybridProposalProvider, LLMProposalProvider,
    _response_schema, primitive_proposer_request,
)
from evolving_loop.v2.numerical_qd.mutation import apply_mutation, record_train_outcome
from test_evolution_v2_numerical_mutation import CELL, SHA, feedback, member, parent_state

CODE = 'def forecast(history, horizon, frequency):\n    """Use a constant baseline."""\n    return [history[-1]] * horizon\n'
CONTEXT_CELL = MorphologyCellV2(
    "low", "none", "low", "stable", "short", "program",
)


def request_args(**updates):
    state = parent_state()
    genome = dict(schema_version=1, generation=0, parent_genome_sha256s=[],
        mutation_operator="add", inventory_sha256=state.inventory.fingerprint(),
        screening_policy_sha256=SHA, combined_policy_sha256=SHA,
        mutation_policy_sha256=state.mutation_policy.fingerprint(),
        proposer_prompt_sha256=state.proposer_prompt.fingerprint(),
        runtime_fingerprints={"python": SHA}, protocol_fingerprint=SHA)
    return dict(parent_genome=genome, parent_state=state.to_payload(),
        selected_cells=[dict(cell_sha256=CELL, member_ids=["a", "b"])],
        train_feedback=[feedback()], curriculum_targets=[], reusable_programs=[],
        eligible_reusable_program_sha256s=[],
        remaining_budget=ResourceUse(wall_seconds=10.0,
            llm_calls=1, input_tokens=100000, output_tokens=16000).to_payload(),
        allowed_mutation_operators=sorted(state.mutation_policy.operators),
        counter_draw=0, max_proposals=2, max_response_bytes=16000) | updates


def request(**updates):
    return primitive_proposer_request(**request_args(**updates))


class ScriptedClient:
    """The network-only seam: record bytes actually delivered to complete()."""
    def __init__(self, result):
        self.result = result
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return LLMResponse(self.result)


def raw_response(**updates):
    proposed = member("c") | {"source_sha256": "candidate_1"}
    proposed.pop("policy_sha256")
    return dict(source_candidates=[dict(local_id="candidate_1", code=CODE)],
        proposals=[dict(operator="add", reason="Train baseline",
            member=proposed)]) | updates


def assert_primitives(value):
    assert type(value) in (dict, list, str, int, float, bool, type(None))
    if type(value) is dict:
        assert all(type(k) is str for k in value)
        for item in value.values():
            assert_primitives(item)
    if type(value) is list:
        for item in value:
            assert_primitives(item)


def ledger(clock=lambda: 0.0):
    return BudgetLedger(BudgetPlan(100, 0.2, ResourceUse(wall_seconds=80.0,
        llm_calls=10, input_tokens=1000000, output_tokens=1000000)), monotonic=clock)


def contextual_request_args(*, status="active", source=CODE):
    cell_sha = CONTEXT_CELL.fingerprint()
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    original = parent_state()
    members = tuple(
        replace(member, applicability_cells=(cell_sha,),
                source_sha256=source_sha if member.member_id == "a" else member.source_sha256,
                status=status if member.member_id == "a" else member.status)
        for member in original.inventory.members
    )
    state = parent_state(
        inventory=NumericalInventoryV2(1, members), declared_cells=(cell_sha,),
    )
    args = request_args(
        parent_state=state.to_payload(),
        selected_cells=[dict(cell_sha256=cell_sha, member_ids=["a", "b"])],
        curriculum_targets=[CurriculumTargetV2(
            1, CONTEXT_CELL, "unoccupied", 0, (),
        ).to_payload()],
    )
    args["parent_genome"].update(
        inventory_sha256=state.inventory.fingerprint(),
        mutation_policy_sha256=state.mutation_policy.fingerprint(),
        proposer_prompt_sha256=state.proposer_prompt.fingerprint(),
    )
    genome_sha = NumericalGenomeV2.from_payload(args["parent_genome"]).fingerprint()
    program = VerifiedReusableProgramV2(
        1, "a", genome_sha, (cell_sha,), source_sha, source,
    )
    args["reusable_programs"] = [program.to_payload()]
    args["eligible_reusable_program_sha256s"] = [program.fingerprint()]
    return args


@pytest.mark.parametrize("provider_kind", ["deterministic", "llm", "hybrid"])
def test_actual_provider_boundary_is_closed_primitive_request(provider_kind):
    client = ScriptedClient(json.dumps(raw_response()))
    providers = dict(deterministic=DeterministicProposalProvider(), llm=LLMProposalProvider(client),
        hybrid=HybridProposalProvider(LLMProposalProvider(client), DeterministicProposalProvider(), ledger()))
    payload = request()
    provider = providers[provider_kind]
    actual_propose = provider.propose
    def inspected_propose(boundary_payload):
        assert set(boundary_payload) == {"parent_genome", "parent_state", "selected_cells", "train_feedback",
            "curriculum_targets", "reusable_programs", "remaining_budget",
            "eligible_reusable_program_sha256s",
            "allowed_mutation_operators", "counter_draw", "max_proposals", "max_response_bytes"}
        assert_primitives(boundary_payload)
        return actual_propose(boundary_payload)
    provider.propose = inspected_propose
    result = provider.propose(payload)
    assert result.proposals
    for proposal in result.proposals:
        apply_mutation(parent_state(), proposal)
    if client.calls:
        message = client.calls[0]["messages"][0]
        wire = json.loads(message["content"])
        assert message["role"] == "user"
        assert wire["request"] == payload
        assert message["content"] == canonical_v2_bytes(wire).decode()
        assert client.calls[0]["system"] == parent_state().proposer_prompt.template


def test_llm_wire_contains_exact_verified_program_source_and_curriculum_context():
    source = CODE.replace("history[-1]", "history[-1] / 1")
    args = contextual_request_args(source=source)
    payload = primitive_proposer_request(**args)
    client = ScriptedClient(json.dumps(raw_response(source_candidates=[], proposals=[])))

    result = LLMProposalProvider(client).propose(payload)

    assert result.failure_reason == "empty"
    wire = json.loads(client.calls[0]["messages"][0]["content"])
    assert wire["request"]["curriculum_targets"] == args["curriculum_targets"]
    assert wire["request"]["reusable_programs"] == args["reusable_programs"]
    assert wire["request"]["reusable_programs"][0]["source_text"] == args["reusable_programs"][0]["source_text"]
    assert client.calls[0]["messages"][0]["content"] == canonical_v2_bytes(wire).decode()


def test_mutation_prompt_context_is_required_and_reaches_llm_wire():
    args = request_args(
        mutation_prompt=parent_state().proposer_prompt.to_payload() | {
            "allowed_mutation_operators": ["policy_tune"],
        },
        mutation_prompt_population_sha256=SHA,
    )
    payload = primitive_proposer_request(**args)
    client = ScriptedClient(json.dumps(raw_response()))
    result = LLMProposalProvider(client, monotonic=lambda: 0.0).propose(payload)
    assert result.proposals
    wire = json.loads(client.calls[0]["messages"][0]["content"])
    assert wire["request"]["mutation_prompt"] == args["mutation_prompt"]
    assert wire["request"]["mutation_prompt_population_sha256"] == SHA


def test_context_rejects_source_text_that_does_not_match_sha():
    changed = contextual_request_args()
    changed["reusable_programs"][0]["source_text"] += "# changed\n"
    with pytest.raises(ValueError, match="source SHA"):
        primitive_proposer_request(**changed)


def test_context_rejects_noncanonical_reusable_program_order():
    unordered = contextual_request_args()
    first = unordered["reusable_programs"][0]
    other_source = CODE.replace("forecast", "other_forecast")
    other = VerifiedReusableProgramV2(
        1, "other", "d" * 64, (CONTEXT_CELL.fingerprint(),),
        hashlib.sha256(other_source.encode()).hexdigest(), other_source,
    ).to_payload()
    unordered["reusable_programs"] = sorted(
        (first, other), key=lambda row: row["source_sha256"], reverse=True,
    )
    with pytest.raises(ValueError, match="canonical"):
        primitive_proposer_request(**unordered)


def test_context_rejects_quarantined_parent_program():
    quarantined = contextual_request_args(status="quarantined")
    with pytest.raises(ValueError, match="quarantined"):
        primitive_proposer_request(**quarantined)


def test_context_rejects_forged_non_parent_program_without_matching_host_commitment():
    forged = contextual_request_args()
    source = CODE.replace("forecast", "forged_forecast")
    record = VerifiedReusableProgramV2(
        1, "forged", "d" * 64, (CONTEXT_CELL.fingerprint(),),
        hashlib.sha256(source.encode()).hexdigest(), source,
    )
    forged["reusable_programs"] = [record.to_payload()]

    with pytest.raises(ValueError, match="Host eligibility"):
        primitive_proposer_request(**forged)


def test_context_accepts_host_committed_non_parent_program():
    request = contextual_request_args()
    source = CODE.replace("forecast", "reusable_forecast")
    record = VerifiedReusableProgramV2(
        1, "reusable", "d" * 64, (CONTEXT_CELL.fingerprint(),),
        hashlib.sha256(source.encode()).hexdigest(), source,
    )
    request["reusable_programs"] = [record.to_payload()]
    request["eligible_reusable_program_sha256s"] = [record.fingerprint()]

    assert primitive_proposer_request(**request)["reusable_programs"] == [
        record.to_payload()
    ]


def test_context_fields_are_jointly_required():
    for missing in ("curriculum_targets", "reusable_programs"):
        args = request_args()
        del args[missing]
        with pytest.raises(ValueError, match="exact schema"):
            primitive_proposer_request(**args)


def test_legacy_parent_context_without_host_commitment_remains_readable():
    legacy = contextual_request_args()
    del legacy["eligible_reusable_program_sha256s"]

    assert primitive_proposer_request(**legacy)["reusable_programs"] == legacy[
        "reusable_programs"
    ]


def test_true_old_request_shape_remains_readable():
    legacy = request_args()
    for key in ("curriculum_targets", "reusable_programs", "eligible_reusable_program_sha256s"):
        del legacy[key]
    assert primitive_proposer_request(**legacy)["parent_genome"] == legacy["parent_genome"]


def test_context_fields_are_bounded():
    args = request_args(curriculum_targets=[
        CurriculumTargetV2(1, CONTEXT_CELL, "unoccupied", 0, ()).to_payload()
    ] * 33)
    with pytest.raises(ValueError, match="at most 32"):
        primitive_proposer_request(**args)


@pytest.mark.parametrize("key,value", [
    ("dev_metrics", "UNIQUE_DEV"), ("public_ids", "UNIQUE_PUBLIC"),
    ("future_values", [987654321.0]), ("path", Path("UNIQUE_PATH")),
    ("callback", lambda: None), ("kernel", object()), ("store", object()),
    ("raw_forecasts", [123456789.0]), ("document_labels", "UNIQUE_DOCUMENT"),
])
@pytest.mark.parametrize("location", ["root", "state", "cell", "feedback", "budget", "genome"])
def test_forbidden_sentinels_rejected_before_provider_execution(key, value, location):
    args = request_args()
    target = {"root": args, "state": args["parent_state"], "cell": args["selected_cells"][0],
              "feedback": args["train_feedback"][0], "budget": args["remaining_budget"],
              "genome": args["parent_genome"]}[location]
    target[key] = value
    client = ScriptedClient(json.dumps(raw_response()))
    with pytest.raises((ValueError, TypeError)):
        LLMProposalProvider(client).propose(primitive_proposer_request(**args))
    assert client.calls == []


def test_providers_revalidate_direct_requests_and_genome_binding():
    client = ScriptedClient(json.dumps(raw_response()))
    for payload in (request() | {"dev": "SENTINEL"},
                    request() | {"parent_genome": request()["parent_genome"] | {"inventory_sha256": SHA}}):
        for provider in (DeterministicProposalProvider(), LLMProposalProvider(client)):
            with pytest.raises(ValueError):
                provider.propose(payload)
    assert client.calls == []


def test_llm_normalizes_only_local_sources_and_returns_immutable_artifacts():
    client = ScriptedClient(json.dumps(raw_response()))
    result = LLMProposalProvider(client, monotonic=lambda: 0.0).propose(request())
    digest = hashlib.sha256(CODE.encode()).hexdigest()
    assert result.source_artifacts == ((digest, CODE.encode()),)
    assert result.proposals[0].to_payload()["member"]["source_sha256"] == digest
    assert result.resource_use.llm_calls == 1
    assert result.resource_use.output_tokens == len(client.result.encode())
    assert result.failure_reason is None
    assert len(client.calls) == 1
    with pytest.raises((AttributeError, TypeError)):
        result.source_artifacts += ((SHA, b"bad"),)


def test_llm_source_member_gets_host_derived_executable_recipe_identity():
    """Catches trusting a model-authored policy digest for new source code."""
    result = LLMProposalProvider(
        ScriptedClient(json.dumps(raw_response())), monotonic=lambda: 0.0
    ).propose(request())
    expected_recipe = {
        "name": "select_forecast",
        "kind": "select",
        "parents": ["forecast"],
        "fallback_parent": "forecast",
        "assumptions": [{
            "assumption_id": "forecast_history",
            "candidate_name": "forecast",
            "feature": "history_length",
            "direction": "above",
            "horizon_region": "full",
            "operator": "select",
            "rationale": "History supports the candidate.",
            "failure_condition": "History is unavailable.",
        }],
    }
    assert result.failure_reason is None
    assert result.proposals[0].to_payload()["member"]["policy_sha256"] == fingerprint_payload(expected_recipe)


def test_llm_rejects_source_that_the_execution_adapter_cannot_load():
    """A proposal admitted here must not fail the stricter Host gate later."""
    invalid = CODE.replace("return [history[-1]] * horizon", "return [history[-1] % 2] * horizon")
    payload = raw_response(source_candidates=[dict(local_id="candidate_1", code=invalid)])

    result = LLMProposalProvider(
        ScriptedClient(json.dumps(payload)), monotonic=lambda: 0.0
    ).propose(request())

    assert result.failure_reason == "malformed"
    assert result.proposals == ()
    assert result.source_artifacts == ()


def test_host_schema_teaches_the_model_the_executable_source_subset():
    description = _response_schema(request(), parent_state())["$defs"][
        "source_candidate"
    ]["properties"]["code"]["description"]

    assert "Never use `%`" in description
    assert "use `divmod`" in description
    assert "use `sorted(...)`" in description
    assert "def recent_mean(history, horizon, frequency):" in description


def test_llm_repairs_inconsistent_add_lineage_when_budget_allows():
    bad = raw_response()
    bad['proposals'][0]['member']['parent_ids'] = ['a']

    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(json.dumps(bad if len(self.calls) == 1 else raw_response()))

    client = Client()
    batch = LLMProposalProvider(client, monotonic=lambda: 0.0).propose(request(
        remaining_budget=ResourceUse(wall_seconds=10.0, llm_calls=2,
                                     input_tokens=100000, output_tokens=32000).to_payload()))
    assert batch.failure_reason is None
    assert len(batch.proposals) == 1
    assert batch.resource_use.llm_calls == 2
    assert 'child lineage must exactly match owned parents' in client.calls[1]['messages'][-1]['content']


def test_llm_repairs_one_host_rejected_source_when_budget_allows():
    invalid = CODE.replace("return [history[-1]] * horizon", "return [history[-1] % 2] * horizon")
    responses = iter((
        json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=invalid)])),
        json.dumps(raw_response()),
    ))

    class RepairingClient:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(next(responses))

    client = RepairingClient()
    result = LLMProposalProvider(client, monotonic=lambda: 0.0).propose(request(
        remaining_budget=ResourceUse(
            wall_seconds=10.0, llm_calls=2, input_tokens=100000,
            output_tokens=16000,
        ).to_payload()
    ))

    assert result.failure_reason is None
    assert result.proposals
    assert result.resource_use.llm_calls == 2
    assert len(client.calls) == 2
    assert "Host source validation failed" in client.calls[1]["messages"][-1]["content"]


def test_hybrid_reserves_and_keeps_a_successful_llm_repair():
    invalid = CODE.replace("return [history[-1]] * horizon", "return [history[-1] % 2] * horizon")
    responses = iter((
        json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=invalid)])),
        json.dumps(raw_response()),
    ))

    class RepairingClient:
        def complete(self, **kwargs):
            return LLMResponse(next(responses))

    budget = ledger()
    result = HybridProposalProvider(
        LLMProposalProvider(RepairingClient(), monotonic=lambda: 0.0),
        DeterministicProposalProvider(), budget,
    ).propose(request(remaining_budget=ResourceUse(
        wall_seconds=10.0, llm_calls=2, input_tokens=100000,
        output_tokens=16000,
    ).to_payload()))

    assert result.failure_reason is None
    assert result.proposals
    assert result.resource_use.llm_calls == 2
    assert budget.charged_use.llm_calls == 2


@pytest.mark.parametrize("bad", [
    "```json\n{}\n```", '{"source_candidates":[],"proposals":[],"proposals":[]}',
    '{"source_candidates":[],"proposals":[],"dev":"SENTINEL"}',
    json.dumps(raw_response(source_candidates=[])),
    json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=CODE)] * 2)),
    json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code="import os\n" + CODE)])),
    json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=CODE + "x = 1\n")])),
    json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=CODE.replace("history, horizon, frequency", "history"))])),
    json.dumps(raw_response(proposals=[dict(operator="add", reason="foreign SHA", member=member("c"))])),
    json.dumps(raw_response(source_candidates=[dict(local_id="candidate_1", code=CODE), dict(local_id="unused", code=CODE)])),
    json.dumps(raw_response(proposals=raw_response()["proposals"] * 3)),
])
def test_llm_malformed_and_unowned_sources_close_without_retry(bad):
    client = ScriptedClient(bad)
    result = LLMProposalProvider(client).propose(request())
    assert result.failure_reason == "malformed"
    assert result.proposals == result.source_artifacts == ()
    assert result.resource_use.llm_calls == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize("raw,reason", [(None, "unavailable"), (TimeoutError("secret"), "timeout"),
    (RuntimeError("secret"), "unavailable"), ("not json", "malformed"),
    ('{"source_candidates":[],"proposals":[]}', "empty")])
def test_hybrid_closes_and_charges_llm_before_deterministic_fallback(raw, reason):
    budget = ledger()
    client = None if raw is None else ScriptedClient(raw)
    provider = HybridProposalProvider(LLMProposalProvider(client, monotonic=lambda: 0.0),
        DeterministicProposalProvider(), budget)
    result = provider.propose(request())
    assert result.provider == "hybrid"
    assert result.proposals
    assert [attempt.provider for attempt in result.attempts] == ["llm", "deterministic"]
    assert result.attempts[0].failure_reason == reason
    assert budget.charged_use.llm_calls == 1
    checkpoint = budget.checkpoint()
    assert checkpoint["open_reservations"] == []
    assert len(checkpoint["closed_stage_ids"]) == 2


def test_hybrid_does_not_fallback_after_llm_uses_remaining_wall_budget():
    now = [0.0]
    class TimedOutClient:
        def complete(self, **kwargs):
            now[0] = 81.0
            raise TimeoutError("timeout")
    budget = ledger(lambda: now[0])
    result = HybridProposalProvider(LLMProposalProvider(TimedOutClient(), monotonic=lambda: now[0]),
        DeterministicProposalProvider(), budget).propose(request())
    assert result.proposals == ()
    assert result.failure_reason == "budget_exhausted"
    assert len(result.attempts) == 1
    assert budget.charged_use.llm_calls == 1
    assert budget.checkpoint()["open_reservations"] == []


def test_response_byte_cap_is_utf8_and_attempt_still_charged():
    client = ScriptedClient("é" * 100)
    result = LLMProposalProvider(client).propose(request(max_response_bytes=150))
    assert result.failure_reason == "malformed"
    assert result.resource_use.output_tokens == 200


def test_deterministic_selection_is_stable_feasible_and_uses_counter_draw():
    provider = DeterministicProposalProvider()
    first = provider.propose(request(counter_draw=0))
    second = provider.propose(request(counter_draw=1))
    assert first.proposals != second.proposals
    assert first.proposals == provider.propose(request(counter_draw=0)).proposals
    for proposal in first.proposals + second.proposals:
        apply_mutation(parent_state(), proposal)
    assert first.resource_use.llm_calls == 0


def test_train_memory_in_next_request_never_contains_dev():
    state = record_train_outcome(parent_state(), feedback())
    args = request_args(parent_state=state.to_payload())
    args["parent_genome"].update(mutation_policy_sha256=state.mutation_policy.fingerprint(),
        proposer_prompt_sha256=state.proposer_prompt.fingerprint(), generation=1)
    payload = primitive_proposer_request(**args)
    assert payload["parent_state"]["mutation_policy"]["operators"]["fork"]["credit"] == 1
    assert "UNIQUE_DEV_SENTINEL" not in canonical_v2_bytes(payload).decode()


def test_standalone_deterministic_provider_requires_remaining_wall_budget():
    result = DeterministicProposalProvider().propose(request(
        remaining_budget=ResourceUse().to_payload()))
    assert result.failure_reason == "budget_exhausted"
    assert result.proposals == ()


def test_hybrid_passes_remaining_budget_after_closing_llm_attempt():
    now = [0.0]
    budget = ledger(lambda: now[0])
    seen = []
    class FailingClient:
        def complete(self, **kwargs):
            now[0] = 2.0
            raise TimeoutError("closed")
    class InspectedDeterministic(DeterministicProposalProvider):
        def propose(self, payload):
            assert_primitives(payload)
            seen.append((payload, budget.checkpoint()))
            return super().propose(payload)
    result = HybridProposalProvider(LLMProposalProvider(FailingClient(), monotonic=lambda: now[0]),
        InspectedDeterministic(), budget).propose(request())
    assert result.proposals
    assert seen[0][0]["remaining_budget"]["wall_seconds"] == 8.0
    assert seen[0][0]["remaining_budget"]["llm_calls"] == 0
    assert seen[0][1]["charged_use"]["llm_calls"] == 1
    assert all("-llm" not in row["stage_id"] for row in seen[0][1]["open_reservations"])


def test_hybrid_does_not_fallback_when_request_budget_is_exactly_consumed():
    now = [0.0]
    class FailingClient:
        def complete(self, **kwargs):
            now[0] = 10.0
            raise TimeoutError("closed")
    budget = ledger(lambda: now[0])
    result = HybridProposalProvider(LLMProposalProvider(FailingClient(), monotonic=lambda: now[0]),
        DeterministicProposalProvider(), budget).propose(request())
    assert result.proposals == ()
    assert result.failure_reason == "budget_exhausted"
    assert len(result.attempts) == 1


def test_llm_batch_duplicate_target_is_rejected_atomically():
    raw = raw_response(source_candidates=[], proposals=[dict(operator="remove", reason="redundant", member_id="a")] * 2)
    result = LLMProposalProvider(ScriptedClient(json.dumps(raw))).propose(request())
    assert result.failure_reason == "malformed"
    assert result.proposals == ()


def test_request_allowed_operators_are_enforced_for_llm():
    result = LLMProposalProvider(ScriptedClient(json.dumps(raw_response()))).propose(
        request(allowed_mutation_operators=["remove"]))
    assert result.failure_reason == "malformed"


def test_response_source_newlines_are_canonicalized_before_hashing():
    raw = raw_response(source_candidates=[dict(local_id="candidate_1", code=CODE.replace("\n", "\r\n"))])
    result = LLMProposalProvider(ScriptedClient(json.dumps(raw))).propose(request())
    assert result.source_artifacts == ((hashlib.sha256(CODE.encode()).hexdigest(), CODE.encode()),)


@pytest.mark.parametrize("operator", ["repair", "fork", "combine", "route", "specialize", "remove", "policy_tune"])
def test_each_required_deterministic_action_is_reachable_and_legal(operator):
    parent = parent_state()
    result = DeterministicProposalProvider().propose(request(allowed_mutation_operators=[operator]))
    assert result.failure_reason is None
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.operator == operator
    child = apply_mutation(parent, proposal).state
    assert child != parent
    assert result.source_artifacts == ()
    if operator in {"repair", "fork", "combine", "route"}:
        payload = proposal.to_payload()
        member_payload = payload.get("child", payload.get("replacement"))
        assert member_payload["source_sha256"] == SHA
        assert member_payload["policy_sha256"] != SHA
        assert member_payload["parent_ids"] == (["a", "b"] if operator in {"combine", "route"} else [payload["member_id"]])
    assert child.mutation_policy == parent.mutation_policy


def test_deterministic_policy_sha_commits_distinct_typed_structures():
    policies = []
    for op in ("repair", "fork", "combine", "route"):
        proposal = DeterministicProposalProvider().propose(request(allowed_mutation_operators=[op])).proposals[0]
        payload = proposal.to_payload()
        policies.append(payload.get("child", payload.get("replacement"))["policy_sha256"])
    assert len(set(policies)) == 4


def test_deterministic_attempt_measures_wall_consumption():
    times = iter([0.0, 3.0])
    provider = DeterministicProposalProvider()
    provider.monotonic = lambda: next(times)
    result = provider.propose(request())
    assert result.proposals
    assert result.resource_use.wall_seconds == 3.0
    assert result.attempts[0].resource_use.wall_seconds == 3.0


@pytest.mark.parametrize("elapsed", [10.0, 11.0])
def test_deterministic_discards_work_when_request_wall_budget_exhausted(elapsed):
    times = iter([0.0, elapsed])
    provider = DeterministicProposalProvider()
    provider.monotonic = lambda: next(times)
    result = provider.propose(request())
    assert result.failure_reason == "budget_exhausted"
    assert result.proposals == result.source_artifacts == ()
    assert result.resource_use.wall_seconds == elapsed
    assert result.attempts[0].failure_reason == "budget_exhausted"


def test_hybrid_fallback_reserves_and_charges_bounded_wall_work():
    now = [0.0]
    budget = ledger(lambda: now[0])
    reservations = []
    def clock():
        if not reservations:
            reservations.extend(budget.checkpoint()["open_reservations"])
            return 0.0
        now[0] = 2.0
        return 2.0
    deterministic = DeterministicProposalProvider()
    deterministic.monotonic = clock
    result = HybridProposalProvider(LLMProposalProvider(None, monotonic=lambda: now[0]),
        deterministic, budget).propose(request())
    assert result.proposals
    assert len(reservations) == 1
    assert reservations[0]["stage_id"].endswith("-deterministic")
    assert reservations[0]["estimate"]["wall_seconds"] == 10.0
    assert result.resource_use.wall_seconds == budget.charged_use.wall_seconds == 2.0
    assert result.attempts[-1].resource_use.wall_seconds == 2.0
    assert budget.checkpoint()["open_reservations"] == []


@pytest.mark.parametrize("denial", ["elapsed_deadline", "close_overrun"])
def test_hybrid_discards_fallback_when_ledger_forbids_completion(denial):
    now = [0.0]
    budget = ledger(lambda: now[0])
    calls = []
    def clock():
        calls.append(True)
        if len(calls) == 1:
            return 0.0
        if denial == "elapsed_deadline":
            now[0] = 81.0
        else:
            budget.charge(ResourceUse(wall_seconds=79.0))
        return 2.0
    deterministic = DeterministicProposalProvider()
    deterministic.monotonic = clock
    result = HybridProposalProvider(LLMProposalProvider(None, monotonic=lambda: 0.0),
        deterministic, budget).propose(request())
    assert result.failure_reason == "budget_exhausted"
    assert result.proposals == result.source_artifacts == ()
    assert result.resource_use.wall_seconds == 2.0
    assert budget.charged_use.wall_seconds == (2.0 if denial == "elapsed_deadline" else 81.0)
    assert budget.checkpoint()["open_reservations"] == []


def test_standalone_llm_rejects_late_valid_response_but_keeps_charge():
    now = [0.0]
    class LateClient(ScriptedClient):
        def complete(self, **kwargs):
            now[0] = 11.0
            return super().complete(**kwargs)
    client = LateClient(json.dumps(raw_response()))
    result = LLMProposalProvider(client, monotonic=lambda: now[0]).propose(request())
    assert result.failure_reason == "budget_exhausted"
    assert result.proposals == result.source_artifacts == ()
    assert result.resource_use.wall_seconds == 11.0
    assert result.resource_use.llm_calls == 1
    assert result.resource_use.output_tokens == len(client.result.encode())
    assert result.attempts[0].failure_reason == "budget_exhausted"
    assert len(client.calls) == 1


@pytest.mark.parametrize("sentinel", ["/tmp/UNIQUE_PATH", "../UNIQUE_PATH", "~/UNIQUE_PATH",
    "models/UNIQUE_PATH", r"C:\tmp\UNIQUE_PATH", "file:///tmp/UNIQUE_PATH"])
@pytest.mark.parametrize("location", ["runtime_label", "prompt_template"])
def test_schema_valid_nested_paths_rejected_before_provider_execution(sentinel, location):
    args = request_args()
    if location == "runtime_label":
        args["parent_genome"]["runtime_fingerprints"] = {sentinel: SHA}
    else:
        args["parent_state"]["proposer_prompt"]["template"] = f"Use this reference: {sentinel}"
        state = MutationStateV2.from_payload(args["parent_state"])
        args["parent_genome"]["proposer_prompt_sha256"] = state.proposer_prompt.fingerprint()
    client = ScriptedClient(json.dumps(raw_response()))
    with pytest.raises(ValueError):
        LLMProposalProvider(client).propose(primitive_proposer_request(**args))
    assert client.calls == []


def test_schema_valid_nested_path_rejected_on_direct_provider_entry():
    payload = request()
    payload["parent_genome"]["runtime_fingerprints"] = {"/tmp/UNIQUE_PATH": SHA}
    client = ScriptedClient(json.dumps(raw_response()))
    for provider in (DeterministicProposalProvider(), LLMProposalProvider(client),
                     HybridProposalProvider(LLMProposalProvider(client), DeterministicProposalProvider(), ledger())):
        with pytest.raises(ValueError):
            provider.propose(payload)
    assert client.calls == []


def test_real_llm_transport_contains_exact_allowed_response_schema_across_prompt_evolution():
    expected_keys = {
        "add": {"operator", "reason", "member"},
        "repair": {"operator", "reason", "member_id", "replacement"},
        "fork": {"operator", "reason", "member_id", "child"},
        "combine": {"operator", "reason", "parent_ids", "child"},
        "route": {"operator", "reason", "parent_ids", "child"},
        "specialize": {"operator", "reason", "member_id", "applicability_cells"},
        "crossover": {"operator", "reason", "parent_ids", "child"},
        "remove": {"operator", "reason", "member_id"},
        "quarantine": {"operator", "reason", "member_id"},
        "policy_tune": {"operator", "reason", "prompt", "credit_delta"},
    }
    for state in (parent_state(), record_train_outcome(parent_state(), feedback())):
        args = request_args(parent_state=state.to_payload())
        args["parent_genome"].update(mutation_policy_sha256=state.mutation_policy.fingerprint(),
            proposer_prompt_sha256=state.proposer_prompt.fingerprint())
        payload = primitive_proposer_request(**args)
        client = ScriptedClient(json.dumps(raw_response()))
        result = LLMProposalProvider(client).propose(payload)
        assert result.proposals
        call = client.calls[0]
        wire = json.loads(call["messages"][0]["content"])
        assert set(wire) == {"request", "allowed_mutation_response_schema"}
        assert call["messages"][0]["content"] == canonical_v2_bytes(wire).decode()
        assert call["system"] == state.proposer_prompt.template
        assert wire["request"] == payload
        assert result.resource_use.input_tokens == len(call["system"].encode()) + len(call["messages"][0]["content"].encode())
        schema = wire["allowed_mutation_response_schema"]
        assert set(schema["required"]) == {"source_candidates", "proposals"}
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["prompt"]["properties"]["parent_prompt_sha256"]["const"] == state.proposer_prompt.fingerprint()
        assert set(schema["$defs"]["source_candidate"]["required"]) == {"local_id", "code"}
        member_schema = schema["$defs"]["member"]
        assert "local_id" in member_schema["properties"]["source_sha256"]["description"]
        assert "policy_sha256" not in member_schema["properties"]
        union = schema["properties"]["proposals"]["items"]["oneOf"]
        by_op = {item["properties"]["operator"]["const"]: item for item in union}
        assert set(by_op) == set(expected_keys)
        for op, keys in expected_keys.items():
            assert set(by_op[op]["required"]) == set(by_op[op]["properties"]) == keys
            assert by_op[op]["additionalProperties"] is False
        assert by_op["policy_tune"]["properties"]["credit_delta"]["maxProperties"] == 0


def test_real_llm_schema_limits_mutations_to_request_authority():
    payload = request(allowed_mutation_operators=["remove"])
    raw = raw_response(source_candidates=[], proposals=[dict(operator="remove", reason="redundant", member_id="a")])
    client = ScriptedClient(json.dumps(raw))
    result = LLMProposalProvider(client).propose(payload)
    assert result.proposals
    wire = json.loads(client.calls[0]["messages"][0]["content"])
    union = wire["allowed_mutation_response_schema"]["properties"]["proposals"]["items"]["oneOf"]
    assert [item["properties"]["operator"]["const"] for item in union] == ["remove"]
