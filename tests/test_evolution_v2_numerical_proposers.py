"""Real proposer boundaries, response ownership, and charged fallback."""
import hashlib
import json
from pathlib import Path

import pytest

from common.llm import LLMResponse
from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.numerical_qd.proposers import (
    DeterministicProposalProvider, HybridProposalProvider, LLMProposalProvider,
    primitive_proposer_request,
)
from evolving_loop.v2.numerical_qd.mutation import apply_mutation, record_train_outcome
from test_evolution_v2_numerical_mutation import CELL, SHA, feedback, member, parent_state

CODE = 'def forecast(history, horizon, frequency):\n    """Use a constant baseline."""\n    return [history[-1]] * horizon\n'


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
        train_feedback=[feedback()], remaining_budget=ResourceUse(wall_seconds=10.0,
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
    return dict(source_candidates=[dict(local_id="candidate_1", code=CODE)],
        proposals=[dict(operator="add", reason="Train baseline",
            member=(member("c") | {"source_sha256": "candidate_1"}))]) | updates


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
            "remaining_budget", "allowed_mutation_operators", "counter_draw", "max_proposals", "max_response_bytes"}
        assert_primitives(boundary_payload)
        return actual_propose(boundary_payload)
    provider.propose = inspected_propose
    result = provider.propose(payload)
    assert result.proposals
    for proposal in result.proposals:
        apply_mutation(parent_state(), proposal)
    if client.calls:
        assert client.calls[0]["messages"] == [{"role": "user", "content": canonical_v2_bytes(payload).decode()}]
        assert client.calls[0]["system"] == parent_state().proposer_prompt.template


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
    assert first == provider.propose(request(counter_draw=0))
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
