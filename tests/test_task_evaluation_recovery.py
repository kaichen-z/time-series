"""Deterministic recovery of completed task scores, never partial inference."""
import json
from dataclasses import replace

import pytest

from common.llm import FakeLLMClient, LLMResponse, TransientLLMError
from evolving_loop.decision_agent.agent import DECISION_PROMPT, DecisionAgent
from evolving_loop.numerical_two_stage import run_numerical_two_stage
from tests.test_evolution_v2_cooperative_pipeline import pipeline_case
from tests.test_numerical_retrieval_handoff import (
    _context_task, _package, _retrieval, _round, _decision,
)


def test_original_decision_rejection_survives_full_pipeline(tmp_path):
    result = run_numerical_two_stage(
        _context_task(), _package(),
        _retrieval([_round()], tmp_path), DecisionAgent(FakeLLMClient(['{}', '{}'])),
    )
    assert result.fallback_reason == 'decision_contract_rejected'
    assert result.final_decision.rejection_reason == (
        'invalid_decision_response_schema:missing fields gaps,rationale,'
        'request_more_retrieval,selected_candidate_id,supporting_document_ids,used_skill_names'
        ';invalid_retrieval_gaps:gaps must be a list'
    )


def test_phase_contracts_keep_strategy_without_final_schema(tmp_path):
    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs['messages'][0]['content'])
            system = kwargs['system']
            assert 'Prefer complementary method families.' in system
            if 'dictionary' in payload or 'evaluated' in payload:
                assert '"request_more_retrieval": false' not in system
                assert '"supporting_document_ids": ["doc_1"]' not in system
            if 'dictionary' in payload:
                return LLMResponse('{"evaluations":[{"method_ids":["seasonal_specialist"],"weights":[1.0]}]}')
            if 'evaluated' in payload:
                return LLMResponse(json.dumps({'selected_candidate_id': payload['evaluated'][0]['candidate_id']}))
            assert 'host_default_id is authoritative' in system
            return LLMResponse(_decision(payload['host_default_id']))

    package = _package(with_assumption=False)
    package = replace(package, component_fingerprints={
        **package.component_fingerprints, 'decision_dictionary': 'a' * 64,
    })
    result = run_numerical_two_stage(
        _context_task(), package, _retrieval([_round()], tmp_path),
        DecisionAgent(Client(), prompt=DECISION_PROMPT + '\nPrefer complementary method families.'),
    )
    assert result.fallback_reason is None
    assert result.forecast == (8.0, 9.0)


def test_dictionary_tool_errors_remain_in_result(tmp_path):
    package = _package(with_assumption=False)
    package = replace(package, component_fingerprints={
        **package.component_fingerprints, 'decision_dictionary': 'a' * 64,
    })
    result = run_numerical_two_stage(
        _context_task(), package, _retrieval([_round()], tmp_path),
        DecisionAgent(FakeLLMClient([
            '{"evaluations":[{"method_ids":["does_not_exist"],"weights":[1.0]}]}',
            _decision('safe_anchor'), _decision('safe_anchor'),
        ])),
    )
    assert result.dictionary_traces[0]['errors']
    assert result.dictionary_traces[0]['errors'] == ['Invalid methods or convex weights']
    assert result.dictionary_traces[0]['failed_evaluations'][0] == {
        'request': {'method_ids': ['does_not_exist'], 'weights': [1.0]},
        'error_type': 'ValueError', 'message': 'Invalid methods or convex weights',
    }


@pytest.mark.parametrize('interrupt', [TransientLLMError, KeyboardInterrupt])
def test_interrupted_tasks_resume_only_completed_scores(pipeline_case, tmp_path, interrupt):
    from evolving_loop.package_task_store import PackageTaskStore

    case = pipeline_case
    pipeline = case['pipeline']
    pipeline.task_store = PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    original = pipeline.decision_factory
    calls = 0

    def fail_second(module):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt('interrupted evaluation details')
        return original(module)

    pipeline.decision_factory = fail_second
    with pytest.raises(interrupt):
        pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    records = [json.loads(path.read_text()) for path in (tmp_path / 'progress').glob('*.json')]
    assert sorted(row['status'] for row in records) == ['completed', 'error']
    assert next(row for row in records if row['status'] == 'error')['error']['message'] == 'interrupted evaluation details'

    pipeline.decision_factory = original
    resumed = pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    assert len(case['decision_clients']) == 4
    # New evaluator/store object simulates a process restart, without replaying inference.
    pipeline.task_store = PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    again = pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    assert again == resumed
    assert len(case['decision_clients']) == 4


@pytest.mark.parametrize('change', ['stage', 'bundle', 'prompt', 'runtime', 'metric', 'contract', 'task', 'package'])
def test_completed_task_cache_invalidates_on_identity_change(pipeline_case, tmp_path, monkeypatch, change):
    from evolving_loop import package_task_store as module
    case = pipeline_case
    pipeline = case['pipeline']
    pipeline.task_store = module.PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    stage = 'train'
    if change == 'stage':
        stage = 'dev'
    elif change == 'bundle':
        case['bundle'] = replace(case['bundle'], generation=1, parent_bundle_sha256=case['bundle'].fingerprint())
    elif change == 'prompt':
        decision = replace(case['decision'], prompt='Different evolved strategy')
        identity = case['catalog'].add_decision(decision)
        case['bundle'] = replace(case['bundle'], decision_policy_sha256=identity)
    elif change == 'runtime':
        pipeline.task_store = module.PackageTaskStore(tmp_path / 'progress', runtime_identity='r2')
    elif change == 'metric':
        pipeline.metric_cap = 4.0
    elif change == 'contract':
        monkeypatch.setattr(module, 'TASK_EXECUTION_CONTRACT', 'changed-contract')
    else:
        # Identity fixture exercises changed task/package fingerprints without bypassing registry validation.
        monkeypatch.setattr(module, 'task_registry_fingerprint' if change == 'task' else 'numerical_package_fingerprint', lambda _: 'f' * 64)
    pipeline.evaluate(case['bundle'], case['tasks'], stage)
    assert len(case['decision_clients']) == 8


def test_started_or_invalid_typed_score_is_never_reused(pipeline_case, tmp_path):
    from evolving_loop.package_task_store import PackageTaskStore
    case = pipeline_case
    pipeline = case['pipeline']
    pipeline.task_store = PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    paths = sorted((tmp_path / 'progress').glob('*.json'))
    row = json.loads(paths[0].read_text())
    row['status'] = 'started'
    paths[0].write_text(json.dumps(row))
    pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    assert len(case['decision_clients']) == 5
    row = json.loads(paths[1].read_text())
    row['score']['final_smae'] = -1
    paths[1].write_text(json.dumps(row))
    with pytest.raises(ValueError):
        pipeline.evaluate(case['bundle'], case['tasks'], 'train')


def test_completed_fallback_keeps_original_error_and_rejection(pipeline_case, tmp_path, monkeypatch):
    from evolving_loop.package_task_store import PackageTaskStore
    import evolving_loop.package_pipeline_evaluator as evaluator
    case = pipeline_case
    pipeline = case['pipeline']
    pipeline.task_store = PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    original = evaluator.run_numerical_two_stage
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('original tool contract detail')
        return original(*args, **kwargs)

    pipeline.decision_factory = lambda module: DecisionAgent(FakeLLMClient(['{}', '{}']), prompt=module.prompt)
    monkeypatch.setattr(evaluator, 'run_numerical_two_stage', fail_once)
    first = pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    records = [json.loads(path.read_text()) for path in (tmp_path / 'progress').glob('*.json')]
    assert all(row['status'] == 'completed' for row in records)
    assert all(row['started_at'] <= row['updated_at'] for row in records)
    artifacts = [row['artifacts'] for row in records]
    assert {'contract_error': {'type': 'ValueError', 'message': 'original tool contract detail'}} in artifacts
    assert any('missing fields' in str(row.get('final_decision_rejection')) for row in artifacts)
    assert pipeline.evaluate(case['bundle'], case['tasks'], 'train') == first
    assert calls == 4


def test_nonfinite_rejected_tool_input_does_not_abort_score_persistence(pipeline_case, tmp_path, monkeypatch):
    from evolving_loop.package_task_store import PackageTaskStore
    import evolving_loop.package_pipeline_evaluator as evaluator
    case = pipeline_case
    pipeline = case['pipeline']
    pipeline.task_store = PackageTaskStore(tmp_path / 'progress', runtime_identity='r1')
    original = evaluator.run_numerical_two_stage

    def rejected_tool(*args, **kwargs):
        return replace(original(*args, **kwargs), dictionary_traces=({
            'errors': ['Invalid methods or convex weights'],
            'failed_evaluations': [{'request': {'weights': [float('nan')]}}],
        },))

    monkeypatch.setattr(evaluator, 'run_numerical_two_stage', rejected_tool)
    result = pipeline.evaluate(case['bundle'], case['tasks'], 'train')
    assert len(result.task_rows) == 4
    record = json.loads(next((tmp_path / 'progress').glob('*.json')).read_text())
    assert record['status'] == 'completed'
    assert record['artifacts']['dictionary_traces'][0]['failed_evaluations'][0]['request']['weights'] == ['nan']
