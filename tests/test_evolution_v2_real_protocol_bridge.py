from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
from evolving_loop.v2.protocol import ProtocolComponentV2
from evolving_loop.v2.real.bridges import load_sealed_bundle_closure, run_real_cooperative
from tests.test_evolution_v2_real_cooperative import (
    _host,
    _p2_pair,
    _real_config_payload,
    _tasks_100,
)
from tests.test_package_stage_runner import _evaluation


def _sealed_p3(tmp_path, monkeypatch, *, entity_conflicts, candidate_score):
    tasks = _tasks_100(entity_conflicts=entity_conflicts)
    host = _host(tasks, object())
    p2 = _p2_pair(tasks)

    def deterministic_evaluate(self, bundle, projected, stage):
        del self, stage
        return _evaluation(
            bundle.fingerprint(),
            tuple(task.numeric.task_id for task in projected),
            2.0 if bundle.generation == 0 else candidate_score,
        )

    monkeypatch.setattr(CooperativePipelineAdapter, "evaluate", deterministic_evaluate)
    root = tmp_path / "p3"
    run_real_cooperative(
        p2=p2, host=host, config_payload=_real_config_payload(), output_dir=root
    )
    return load_sealed_bundle_closure(root, tasks=tasks, host=host), host


@pytest.fixture(params=[False, True], ids=["unique-entities", "conflicting-entities"])
def sealed_p3(tmp_path, monkeypatch, request):
    return _sealed_p3(
        tmp_path,
        monkeypatch,
        entity_conflicts=request.param,
        candidate_score=1.0,
    )


@pytest.fixture
def rejected_p3(tmp_path, monkeypatch):
    return _sealed_p3(
        tmp_path,
        monkeypatch,
        entity_conflicts=False,
        candidate_score=3.0,
    )


def test_protocol_bridge_closes_exactly_two_p3_bundles_without_public_inputs(
    sealed_p3,
):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = sealed_p3
    before = closure.active_bundle.canonical_bytes()
    case = build_protocol_case_from_p3(closure, host, hard_limit_seconds=120)

    assert case.status == "protocol_case_ready"
    assert case.config == {
        "schema_version": 1,
        "profile": "real",
        "seed": 0,
        "max_proposals": 5,
        "hard_limit_seconds": 120,
    }
    corpus = case.inputs.split_manifest
    assert corpus["public_task_sha256s"] == ()
    assert len(case.corpus.archive_bundle_sha256s) == 2
    assert case.corpus.archive_bundle_sha256s[0] == closure.active_bundle.fingerprint()
    assert len(set(case.corpus.archive_bundle_sha256s)) == 2
    assert len(case.inputs.verifier_fixtures) == 4
    assert case.inputs.runtime_inputs.fixture_scope == "ordinary"
    assert set(case.inputs.inference_projections) == set(
        case.corpus.train_task_sha256s + case.corpus.dev_task_sha256s
    )
    for projection in case.inputs.inference_projections.values():
        assert not {"future_values", "labels", "ground_truth"} & set(projection)
    assert closure.active_bundle.canonical_bytes() == before
    changed_loader = replace(
        case.seed_protocol.components[1], implementation_id="changed_history_json"
    )
    changed = replace(
        case.seed_protocol,
        components=(case.seed_protocol.components[0], changed_loader, *case.seed_protocol.components[2:]),
    )
    with pytest.raises(ValueError, match="restricted"):
        case.registry.resolve(changed, case.inputs.runtime_inputs)


def test_protocol_bridge_uses_the_accepted_candidate_bound_to_active_authority(
    sealed_p3,
):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = sealed_p3
    case = build_protocol_case_from_p3(closure, host, hard_limit_seconds=120)
    second_sha = case.corpus.archive_bundle_sha256s[1]
    receipt = next(
        receipt
        for candidate, receipt in zip(
            closure.closed_candidate_bundles,
            closure.acceptance_evidence,
            strict=True,
        )
        if candidate.fingerprint() == second_sha
    )

    assert receipt.decision == "accept"
    assert receipt.candidate_bundle_sha256 == second_sha
    assert receipt.fingerprint() == closure.active_bundle.acceptance_evidence_sha256
    assert (
        case.input_manifest["frozen_bundle_sha256"]
        == closure.active_bundle.fingerprint()
    )


def test_protocol_bridge_does_not_run_with_only_rejected_p3_candidates(
    rejected_p3,
    tmp_path,
):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = rejected_p3
    assert closure.closed_candidate_bundles
    assert closure.p5_handoff_available
    assert {receipt.decision for receipt in closure.acceptance_evidence} == {"reject"}

    case = build_protocol_case_from_p3(closure, host, hard_limit_seconds=120)
    output_dir = tmp_path / "p5-must-not-run"

    assert case.status == "p5_handoff_unavailable"
    assert case.config is None
    assert case.input_manifest is None
    assert case.run(output_dir) == {
        "schema_version": 1,
        "status": "p5_handoff_unavailable",
    }
    assert not output_dir.exists()


def test_protocol_bridge_returns_stable_unavailable_handoff(sealed_p3):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = sealed_p3
    unavailable = replace(
        closure,
        closed_candidate_bundles=(),
        acceptance_evidence=(),
        p5_handoff_available=False,
        p5_handoff_reason="p5_handoff_unavailable",
    )
    case = build_protocol_case_from_p3(unavailable, host, hard_limit_seconds=120)

    assert case.status == "p5_handoff_unavailable"
    assert case.config is None
    assert case.input_manifest is None


def test_protocol_bridge_rejects_changed_projection(sealed_p3):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = sealed_p3
    changed = replace(closure, train_tasks=tuple(reversed(closure.train_tasks)))
    with pytest.raises(ValueError, match="projection"):
        build_protocol_case_from_p3(changed, host, hard_limit_seconds=120)


def test_protocol_bridge_case_uses_the_generic_runner(sealed_p3, tmp_path):
    from evolving_loop.v2.protocol.bridge import build_protocol_case_from_p3

    closure, host = sealed_p3
    case = build_protocol_case_from_p3(closure, host, hard_limit_seconds=120)

    result = case.run(tmp_path / "p5", stop_after=0)

    assert result == {
        "schema_version": 1,
        "status": "protocol_evolution_incomplete",
        "next_index": 0,
        "accepted": 0,
        "rejected": 0,
        "public_test_accessed": False,
    }


def test_protocol_bridge_production_module_never_uses_test_or_cli_dispatch():
    import ast

    module = Path(__file__).parents[1] / "evolving_loop/v2/protocol/bridge.py"
    tree = ast.parse(module.read_text())
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(
        name.startswith("tests") or name in {"FakeLLMClient", "dispatch_protocol"}
        for name in imports
    )
