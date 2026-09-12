from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
from evolving_loop.v2.real.bridges import load_sealed_bundle_closure, run_real_cooperative
from evolving_loop.v2.source import SourceVariantV2
from tests.test_evolution_v2_real_cooperative import (
    _host,
    _p2_pair,
    _real_config_payload,
    _tasks_100,
)
from tests.test_package_stage_runner import _evaluation


@pytest.fixture
def sealed_p3(tmp_path, monkeypatch):
    tasks = _tasks_100()
    host = _host(tasks, object())
    p2 = _p2_pair(tasks)

    def deterministic_evaluate(self, bundle, projected, stage):
        del self, stage
        return _evaluation(
            bundle.fingerprint(),
            tuple(task.numeric.task_id for task in projected),
            2.0 if bundle.generation == 0 else 3.0,
        )

    monkeypatch.setattr(CooperativePipelineAdapter, "evaluate", deterministic_evaluate)
    root = tmp_path / "p3"
    run_real_cooperative(
        p2=p2, host=host, config_payload=_real_config_payload(), output_dir=root
    )
    return load_sealed_bundle_closure(root, tasks=tasks, host=host), host


def test_source_bridge_reconstructs_p3_bundle_with_real_host_factories(
    sealed_p3, tmp_path
):
    from evolving_loop.v2.source.bridge import build_source_case_from_p3

    closure, host = sealed_p3
    source = SourceVariantV2.seed(
        "def choose_arm(request):\n    return request['enabled_arms'][0]\n",
        closure.active_bundle.protocol_fingerprint,
        host.resource_reporter_sha256,
    )
    before = closure.active_bundle.canonical_bytes()
    case = build_source_case_from_p3(
        closure,
        host,
        source_seed=source,
        input_digest="b" * 64,
        empty_skill_path=tmp_path / "empty-skills.json",
    )

    catalog, reconstructed = case.catalog_factory()
    assert reconstructed.canonical_bytes() == before
    assert catalog.resolve_numerical(
        reconstructed.numerical_release_sha256, reconstructed.numerical_registry_sha256
    ) == closure.numerical
    assert catalog.resolve_retrieval(reconstructed.retrieval_release_sha256) == closure.retrieval
    assert catalog.resolve_decision(reconstructed.decision_policy_sha256) == closure.decision
    pipeline = case.pipeline_factory(catalog)
    assert pipeline.retrieval_factory.__self__ is host
    assert pipeline.decision_factory.__self__ is host
    assert case.train_tasks == closure.train_tasks
    assert case.dev_tasks == closure.dev_tasks
    assert len({task.numeric.entity_name for task in (*case.train_tasks, *case.dev_tasks)}) == 5
    assert case.seed_source == source
    assert case.seed_source.protocol_fingerprint == closure.active_bundle.protocol_fingerprint
    assert case.seed_source.runtime_fingerprint == host.resource_reporter_sha256
    assert closure.active_bundle.canonical_bytes() == before
    assert case.evaluator.seed_bundle.canonical_bytes() == before


def test_source_bridge_production_module_has_no_test_or_fake_imports():
    import ast
    from pathlib import Path

    module = Path(__file__).parents[1] / "evolving_loop/v2/source/bridge.py"
    tree = ast.parse(module.read_text())
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(name.startswith("tests") or name == "FakeLLMClient" for name in imports)
