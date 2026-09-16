import pytest
import json
from types import SimpleNamespace

from common.llm import JsonExtractionError, LLMResponse, TransientLLMError
from evolving_loop.decision_agent.agent import DecisionAgent

from evolving_loop.decision_agent.dictionary_tools import DictionaryExecutionTool
from tests.test_package_numerical_supply import _ranked


def test_real_retrieval_request_supplies_parseable_response_contract(tmp_path):
    from evolving_loop.retrieval_agent.schemas import RetrievalRoundResult, EvidenceChain
    from tests.test_numerical_retrieval_handoff import _context_task, _retrieval

    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs['messages'][0]['content'])
            contract = payload['response_contract']
            EvidenceChain.from_payload(contract['chain_example'])
            result = RetrievalRoundResult.from_payload(contract['empty_response'])
            return LLMResponse(json.dumps(result.to_payload()))

    retrieval = _retrieval([], tmp_path)
    retrieval.llm = Client()
    result = retrieval.run_round1(_context_task())
    assert result.rejected == ()
    assert result.chains == ()


def test_retrieval_repairs_one_structurally_invalid_model_response(tmp_path):
    from tests.test_numerical_retrieval_handoff import _context_task, _retrieval, _round
    retrieval = _retrieval(['{"timeline": []}', _round()], tmp_path)
    result = retrieval.run_round1(_context_task())
    assert result.rejected == ()


def test_retrieval_stops_after_one_failed_format_repair(tmp_path):
    from tests.test_numerical_retrieval_handoff import _context_task, _retrieval, _round
    retrieval = _retrieval(['{"timeline": []}', '{"timeline": []}', _round()], tmp_path)
    result = retrieval.run_round1(_context_task())
    assert result.rejected


def test_tool_evaluates_more_than_eight_methods_and_arbitrary_ensemble():
    inputs = tuple(_ranked(f'm{i}', 'statistical', (float(i),) * 2) for i in range(10))
    tool = DictionaryExecutionTool(inputs)
    assert len(tool.catalog()) == 10
    result = tool.evaluate({'method_ids': ['m1', 'm2', 'm3'], 'weights': [0.2, 0.3, 0.5]})
    assert result.forecast == pytest.approx((2.3, 2.3))
    assert result.hindcast_srmse > 0


@pytest.mark.parametrize('tool_request', [
    {'method_ids': ['unknown'], 'weights': [1.0]},
    {'method_ids': ['m1'], 'weights': [-1.0]},
    {'method_ids': ['m1'], 'weights': [float('nan')]},
    {'method_ids': ['m1', 'm1'], 'weights': [0.5, 0.5]},
])
def test_tool_rejects_invalid_method_requests(tool_request):
    tool = DictionaryExecutionTool((_ranked('m1', 'statistical', (1., 1.)),))
    with pytest.raises(ValueError):
        tool.evaluate(tool_request)


def test_decision_calls_tool_then_selects_from_executed_results():
    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            payload = json.loads(kwargs['messages'][0]['content'])
            self.calls.append(payload)
            if len(self.calls) == 1:
                assert len(payload['dictionary']) == 10
                return LLMResponse(json.dumps({'evaluations': [
                    {'method_ids': ['m8', 'm9'], 'weights': [0.4, 0.6]},
                ]}))
            return LLMResponse(json.dumps({'selected_candidate_id': payload['evaluated'][0]['candidate_id']}))

    client = Client()
    tool = DictionaryExecutionTool(tuple(_ranked(f'm{i}', 'statistical', (float(i),) * 2) for i in range(10)))
    candidate, trace = DecisionAgent(client).select_dictionary(
        tool, SimpleNamespace(evidence=()),
    )
    assert candidate.forecast == pytest.approx((8.6, 8.6))
    assert len(trace['evaluations']) == 1
    assert len(client.calls) == 2


def test_two_stage_pipeline_uses_decision_dictionary_tools(tmp_path):
    from dataclasses import replace
    from tests.test_numerical_retrieval_handoff import _context_task, _package, _retrieval, _round, _decision
    from evolving_loop.numerical_two_stage import run_numerical_two_stage

    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs['messages'][0]['content'])
            assert 'future_values' not in payload and 'gt_evidence' not in payload
            if 'dictionary' in payload:
                assert len(payload['history']) == 84
                return LLMResponse(json.dumps({'evaluations': [
                    {'method_ids': ['seasonal_specialist'], 'weights': [1.0]},
                ]}))
            if 'evaluated' in payload:
                return LLMResponse(json.dumps({'selected_candidate_id': payload['evaluated'][0]['candidate_id']}))
            return LLMResponse(_decision(payload['host_default_id']))

    package = _package(with_assumption=False)
    package = replace(package, component_fingerprints={
        **dict(package.component_fingerprints), 'decision_dictionary': 'a' * 64,
    })
    result = run_numerical_two_stage(_context_task(), package, _retrieval([_round()], tmp_path), DecisionAgent(Client()))
    assert result.forecast == (8.0, 9.0)
    assert result.fallback_reason is None
    assert 'decision_dictionary_execution' in result.fingerprints


def test_second_retrieval_round_can_change_dictionary_selection(tmp_path):
    from dataclasses import replace
    from tests.test_numerical_retrieval_handoff import _context_task, _package, _retrieval, _round, _decision, _chain
    from evolving_loop.numerical_two_stage import run_numerical_two_stage
    task = _context_task()
    package = _package()
    package = replace(package, component_fingerprints={
        **dict(package.component_fingerprints), 'decision_dictionary': 'a' * 64,
    })

    class Client:
        plans = 0
        decisions = 0

        def complete(self, **kwargs):
            payload = json.loads(kwargs['messages'][0]['content'])
            if 'dictionary' in payload:
                self.plans += 1
                name = 'safe_anchor' if self.plans == 1 else 'seasonal_specialist'
                if self.plans == 2:
                    assert any(e['document_id'] == 'doc_round2' for e in payload['verified_evidence'])
                return LLMResponse(json.dumps({'evaluations': [{'method_ids': [name], 'weights': [1.0]}]}))
            if 'evaluated' in payload:
                return LLMResponse(json.dumps({'selected_candidate_id': payload['evaluated'][0]['candidate_id']}))
            self.decisions += 1
            return LLMResponse(_decision(payload['host_default_id'], request_more=self.decisions == 1))

    client = Client()
    retrieval = _retrieval([
        _round(_chain(task, chain_id='r1', document_id='doc_round1', direction='up', magnitude=5.0)),
        _round(_chain(task, chain_id='r2', document_id='doc_round2', direction='down', magnitude=2.0, addressed=('assumption_001',))),
    ], tmp_path)
    result = run_numerical_two_stage(task, package, retrieval, DecisionAgent(client))
    assert result.fallback_reason is None
    assert result.forecast == (8.0, 9.0)
    assert client.plans == 2
    assert 'decision_dictionary_execution_round2' in result.fingerprints


@pytest.mark.parametrize('failure_call', [1, 2])
@pytest.mark.parametrize('error_type', [RuntimeError, JsonExtractionError])
def test_dictionary_provider_failure_returns_baseline_control(failure_call, error_type):
    class Client:
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == failure_call:
                raise error_type('provider failed')
            return LLMResponse(json.dumps({'evaluations': [{'method_ids': ['m'], 'weights': [1.0]}]}))

    result, trace = DecisionAgent(Client()).select_dictionary(
        DictionaryExecutionTool((_ranked('m', 'statistical', (1., 1.)),)),
        SimpleNamespace(evidence=()),
    )
    assert result is None
    assert trace['errors'] == [f'dictionary_provider_failure:{error_type.__name__}:provider failed']
    assert len(trace['evaluations']) == failure_call - 1


def test_dictionary_transient_provider_failure_remains_retryable():
    class Client:
        def complete(self, **kwargs):
            raise TransientLLMError('retry later')

    with pytest.raises(TransientLLMError):
        DecisionAgent(Client()).select_dictionary(
            DictionaryExecutionTool((_ranked('m', 'statistical', (1., 1.)),)),
            SimpleNamespace(evidence=()),
        )
