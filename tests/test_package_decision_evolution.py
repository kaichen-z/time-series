from __future__ import annotations

import json
from dataclasses import replace

import pytest

from common.llm import FakeLLMClient
from evolving_loop.co_evolution import CoEvolutionConfig, HarnessPolicy
from evolving_loop.coordinate_evolution import (
    DecisionEvolutionPhaseAdapter,
    principal_module_fingerprints,
)
from evolving_loop.data import Document
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_decision_evolution import (
    PackageDecisionEvaluator,
    PackageDecisionEvolutionEngine,
    PackageDecisionEvolutionError,
    package_decision_gate_failures,
)
from evolving_loop.package_coordinate_evolution import PackageCoordinateBundle
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.schemas import EvidenceCitation
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from tests.test_coordinate_evolution import _accepted_release, _bound_policy
from tests.test_package_retrieval_evolution import (
    _decision_response,
    _frozen_registry,
    _package,
    _retrieval_card,
    _round_response,
    _task,
)


def _safe_default_package():
    package = _package()
    safe = next(
        item for item in package.ranked_alternatives if item.name == "safe_anchor"
    )
    decision = replace(
        package.selection_decision,
        mode="single",
        selected=(safe.name,),
        weights=(1.0,),
        forecast=safe.forecast,
        reason_codes=("frozen_champion", "test_safe_default"),
        assumption_ids=(),
        assumption_kinds=(),
        considered_candidates=(safe.name,),
    )
    return replace(package, selection_decision=decision, final_forecast=safe.forecast)


def _decision_task():
    task = _task()
    return replace(
        task,
        documents=(
            *task.documents,
            Document(
                "cycle_support",
                "A scheduled seasonal cycle will persist and increase Entity A sales by 1 unit from 2026-02-06 through 2026-02-07.",
                role="supporting",
            ),
        ),
    )


def _round2_response(task) -> str:
    chain = _retrieval_card().chains[0]
    document = next(
        item for item in task.documents if item.document_id == "cycle_support"
    )
    payload = replace(
        chain,
        chain_id="assumption_support",
        claim=document.content,
        citations=(EvidenceCitation(document.document_id, document.content),),
        mechanism="future_driver",
        direction="up",
        magnitude_kind="absolute",
        magnitude_value=1.0,
        addressed_assumption_ids=("assumption_001",),
        stance="supports",
    ).to_payload()
    return json.dumps(
        {
            "evidence_chains": [payload],
            "counterevidence": [],
            "missing_information": [],
            "sufficient": True,
        }
    )


def _request_specialist() -> str:
    payload = json.loads(_decision_response("specialist"))
    payload.update(
        {
            "request_more_retrieval": True,
            "gaps": [
                {
                    "assumption_id": "assumption_001",
                    "gap_type": "continuation_or_reversal",
                    "missing_information": "Evidence that the seasonal cycle persists",
                    "priority": "high",
                }
            ],
        }
    )
    return json.dumps(payload)


def _final_specialist() -> str:
    payload = json.loads(_decision_response("specialist"))
    payload["supporting_document_ids"] = ["cycle_support"]
    return json.dumps(payload)


def _decision_dependencies() -> dict[str, str]:
    return {
        "retrieval_factory": "1" * 64,
        "decision_factory": "2" * 64,
        "bridge_runtime": "3" * 64,
    }


def _evaluator(tmp_path, parent: HarnessPolicy, *, child_improves: bool = True):
    task = _decision_task()
    registry = _frozen_registry(((task, _safe_default_package()),))
    skills = RetrievalSkillLibrary(tmp_path / "retrieval.json", persist=False).clone(
        read_only=True
    )

    def retrieval_factory(policy: HarnessPolicy):
        responses = (
            [_round_response(), _round2_response(task)]
            if child_improves and "prefer specialist" in policy.decision_prompt
            else [_round_response()]
        )
        return TwoStageRetrievalAgent(
            FakeLLMClient(responses),
            policy.retrieval_genome,
            skills,
        )

    def decision_factory(policy: HarnessPolicy):
        if child_improves and "prefer specialist" in policy.decision_prompt:
            responses = [_request_specialist(), _final_specialist()]
        else:
            responses = [
                _decision_response("safe_anchor"),
                _decision_response("safe_anchor"),
            ]
        return DecisionAgent(FakeLLMClient(responses), prompt=policy.decision_prompt)

    return task, PackageDecisionEvaluator(
        registry,
        retrieval_factory,
        decision_factory,
        dependency_fingerprints=_decision_dependencies(),
    )


def _engine(evaluator, *, response: str):
    return PackageDecisionEvolutionEngine(
        FakeLLMClient([response]),
        evaluator,
        CoEvolutionConfig(
            generations=1,
            children_per_generation=1,
            mode="genome",
            target="decision",
            screening_tolerance=1e-12,
        ),
    )


def _mutation_response() -> str:
    return json.dumps(
        {
            "decision_prompt": "prefer specialist using verified evidence",
            "changelog": "Prefer the materialized specialist when evidence supports it.",
        }
    )


def test_decision_evaluator_reports_materialized_selection_regret(tmp_path) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent)

    evaluation = evaluator.evaluate(parent, (task,))

    assert evaluation.diagnostics["mean_selection_smae_regret"] == pytest.approx(
        11.0 / 3.0
    )
    assert evaluation.diagnostics["mean_selection_srmse_regret"] == pytest.approx(
        (30.5**0.5) / 1.5
    )
    assert evaluation.diagnostics["public_test_accessed"] == 0.0
    assert evaluation.diagnostics["fallback_count"] == 0.0


def test_package_decision_evaluator_exposes_shared_package_evaluation(tmp_path) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent)

    evaluation = evaluator.evaluate_package(parent, (task,), stage="build")

    assert isinstance(evaluation, PackageEvaluation)
    assert evaluation.coverage == 1.0
    assert evaluation.task_rows[0].selected_candidate_id == "safe_anchor"
    assert evaluation.secondary_diagnostics[
        "decision_selection_smae_regret"
    ] == pytest.approx(11.0 / 3.0)


def test_pipeline_preserves_round1_when_typed_round2_is_malformed(
    tmp_path,
    monkeypatch,
) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    policy = _bound_policy(release)
    round2_task = _decision_task()
    registry = _frozen_registry(((round2_task, _safe_default_package()),))
    skills = RetrievalSkillLibrary(tmp_path / "pipeline-skills.json", persist=False).clone(
        read_only=True
    )
    original_round2 = TwoStageRetrievalAgent.run_round2

    def malformed_round2(self, *args, **kwargs):
        verified = original_round2(self, *args, **kwargs)
        return replace(
            verified,
            rejected=(*verified.rejected, "invalid_round2_response"),
        )

    monkeypatch.setattr(TwoStageRetrievalAgent, "run_round2", malformed_round2)
    retrieval_llm = FakeLLMClient([_round_response(), _round2_response(round2_task)])
    retrieval = TwoStageRetrievalAgent(retrieval_llm, policy.retrieval_genome, skills)
    decision = DecisionAgent(
        FakeLLMClient([_request_specialist(), _decision_response("safe_anchor")]),
        prompt=policy.decision_prompt,
    )
    bundle = PackageCoordinateBundle(
        generation=0,
        parent_sha256=None,
        numerical_manifest_sha256=registry.fingerprint,
        policy=policy,
        runtime_fingerprints={
            "bridge_runtime": "1" * 64,
            "retrieval_runtime": "2" * 64,
            "decision_runtime": "3" * 64,
            "retrieval_verifier": "4" * 64,
            "metric_policy": "5" * 64,
        },
    )
    pipeline_evaluator = PackagePipelineEvaluator(
        lambda _policy: retrieval,
        lambda _policy: decision,
    )
    expected_round1_sha256 = (
        "d5665af50cc5250bbb423546e0e0ef037c257b3dbdda2a3c6cf533331d3d85e3"
    )

    evaluation = pipeline_evaluator.evaluate(
        bundle,
        registry,
        (round2_task,),
        stage="screen8",
    )

    assert evaluation.task_rows[0].fallback_count == 0
    assert evaluation.task_rows[0].final_retrieval_sha256 == expected_round1_sha256
    assert len(retrieval_llm.calls) == 2


def test_decision_evaluator_rejects_policy_without_accepted_retrieval(
    tmp_path,
) -> None:
    parent = HarnessPolicy()
    _task_value, evaluator = _evaluator(tmp_path, parent)

    with pytest.raises(PackageDecisionEvolutionError, match="non-v000"):
        evaluator.evaluate(parent, (_task(),))


def test_package_decision_engine_accepts_only_the_decision_coordinate(tmp_path) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent)
    engine = _engine(evaluator, response=_mutation_response())

    accepted, trace = engine.evolve(parent, (task,), (task,))

    parent_fingerprints = principal_module_fingerprints(parent)
    accepted_fingerprints = principal_module_fingerprints(accepted)
    assert accepted is not parent
    assert accepted.parent == parent.version
    assert accepted_fingerprints["numerical_morphology"] == parent_fingerprints[
        "numerical_morphology"
    ]
    assert accepted_fingerprints["retrieval"] == parent_fingerprints["retrieval"]
    assert accepted_fingerprints["decision"] != parent_fingerprints["decision"]
    assert trace[-1].accepted_version == accepted.version


def test_package_decision_engine_preserves_exact_parent_on_dev_rejection(
    tmp_path,
) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent, child_improves=False)
    engine = _engine(evaluator, response=_mutation_response())

    selected, trace = engine.evolve(parent, (task,), (task,))

    assert selected is parent
    assert selected.canonical_bytes() == parent.canonical_bytes()
    assert trace[-1].accepted_version == parent.version


def test_package_engine_runs_through_existing_decision_phase_adapter(tmp_path) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent)
    engine = _engine(evaluator, response=_mutation_response())
    adapter = DecisionEvolutionPhaseAdapter(
        engine,
        accepted_release_path=release.path,
    )

    outcome = adapter.run(parent, (task,), (task,))

    assert outcome.accepted is True
    assert outcome.bundle.parent == parent.version
    assert outcome.target == "decision"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("p95_smae", 4.0, "p95_smae"),
        ("invalid_count", 1.0, "invalid_count"),
        ("catastrophic_count", 1.0, "catastrophic_count"),
        ("fallback_count", 1.0, "fallback_count"),
        ("public_test_accessed", 1.0, "public_test_accessed"),
    ),
)
def test_package_decision_gate_rejects_safety_regressions(
    tmp_path,
    field,
    value,
    reason,
) -> None:
    release = _accepted_release(tmp_path / "releases", "v001", "v000")
    parent = _bound_policy(release)
    task, evaluator = _evaluator(tmp_path, parent)
    parent_evaluation = evaluator.evaluate(parent, (task,))
    child = replace(
        parent_evaluation,
        version="v002",
        diagnostics={**parent_evaluation.diagnostics, field: value},
    )

    failures = package_decision_gate_failures(
        child,
        parent_evaluation,
        1e-12,
        require_strict=False,
    )

    assert reason in failures
