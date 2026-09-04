"""Typed, redacted proposal adapters for package-coordinate evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from common.llm import FakeLLMClient
from evolving_loop.co_evolution import CoEvolutionConfig, HarnessPolicy
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    PackageProposalFeedback,
    embed_retrieval_candidate,
)
from evolving_loop.package_coordinate_evolution import package_principal_fingerprints
from evolving_loop.package_decision_evolution import (
    DecisionCandidateProposer,
    PackageDecisionEvaluator,
    PackageDecisionEvolutionEngine,
)
from evolving_loop.package_metrics import PackageEvaluation, PackageTaskScore
from evolving_loop.package_numerical_evolution import (
    NumericalCandidateProposer,
    NumericalCoordinateCandidate,
)
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCandidateProposer,
    RetrievalGenomeProposer,
    rebase_retrieval_candidate,
    retrieval_behavior_fingerprint,
)
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from numerical_agent.evolution.champion_evidence import ProposerEvidence
from tests.test_package_coordinate_evolution import _bundle
from tests.test_package_decision_evolution import _decision_task
from tests.test_package_retrieval_evolution import _frozen_registry, _package


def _evaluation(candidate: str, task_id: str, error: float) -> PackageEvaluation:
    row = PackageTaskScore(
        task_id=task_id,
        entity_name="private_entity",
        final_smae=error,
        final_srmse=error,
        final_smae_raw=error,
        final_srmse_raw=error,
        final_forecast=(10.0, 11.0),
        numerical_oracle_smae=error,
        numerical_oracle_srmse=error,
        numerical_candidate_count=2,
        smae_clipped=False,
        srmse_clipped=False,
        invalid_count=0,
        catastrophic_count=0,
        fallback_count=0,
        selected_candidate_id="private_candidate",
        numerical_package_sha256="1" * 64,
        final_retrieval_sha256="2" * 64,
        final_decision_sha256="3" * 64,
    )
    return PackageEvaluation.from_rows(candidate, (row,), (task_id,))


def _feedback() -> PackageProposalFeedback:
    return PackageProposalFeedback.from_evaluations(
        parent=_evaluation("a" * 64, "task_185", 1.0),
        rejected_children=(_evaluation("b" * 64, "task_186", 1.1),),
        gate_names=("p95_srmse", "maximum_task_joint_regret"),
        structures=({"kind": "route", "structure_sha256": "c" * 64},),
    )


def _retrieval_proposal(
    parent: RetrievalGenome,
    version: str,
    scope: str,
) -> dict[str, object]:
    payload = parent.to_payload()
    payload.update({"version": version, "parent": parent.version})
    if scope == "A":
        payload["round1_prompt"] = f"{parent.round1_prompt}\nCandidate {version}."
    elif scope == "B":
        payload["max_citations_per_chain"] = parent.max_citations_per_chain + 1
    elif scope == "C":
        payload["round2_strategy"] = "gap_first"
    else:  # pragma: no cover - test helper contract.
        raise AssertionError(scope)
    return payload


def _numerical_children(parent, *, generation: int) -> tuple[NumericalCoordinateCandidate, ...]:
    task = _decision_task()
    children = []
    parent_release = parse_numerical_supply_release(
        parent.bundle.to_payload()["numerical_release_payload"]
    )
    for slot in range(3):
        payload = parent_release.to_payload()
        payload.update(
            {
                "version": f"n{generation * 3 + slot + 1:03d}",
                "parent_sha256": parent_release.fingerprint,
                "source_fingerprints": {"dictionary": str(slot + 6) * 64},
            }
        )
        release = parse_numerical_supply_release(payload)
        package = _package()
        package = replace(
            package,
            component_fingerprints={
                **dict(package.component_fingerprints),
                "numerical_supply_release": release.fingerprint,
            },
        )
        registry = _frozen_registry(((task, package),), release=release)
        children.append(
            NumericalCoordinateCandidate(
                release,
                registry,
                hashlib.sha256(f"numerical:{slot}".encode()).hexdigest(),
            )
        )
    return tuple(children)


class _NumericalProposer:
    def __init__(self, parent) -> None:
        self.parent = parent
        self.feedback = None

    def propose(
        self,
        _release,
        _registry,
        feedback,
        *,
        generation: int,
        child_count: int,
    ):
        assert child_count == 3
        self.feedback = feedback
        return _numerical_children(self.parent, generation=generation)


def _decision_engine(responses: list[str]) -> PackageDecisionEvolutionEngine:
    evaluator = object.__new__(PackageDecisionEvaluator)
    return PackageDecisionEvolutionEngine(
        FakeLLMClient(responses),
        evaluator,
        CoEvolutionConfig(
            generations=1,
            children_per_generation=3,
            mode="genome",
            target="decision",
        ),
    )


def test_package_feedback_contains_no_task_ids_values_or_dev_residuals() -> None:
    feedback = _feedback()

    payload = json.dumps(feedback.to_payload(), sort_keys=True)

    assert "task_185" not in payload
    assert "task_186" not in payload
    assert "private_entity" not in payload
    assert "final_forecast" not in payload
    assert "task_rows" not in payload
    assert set(feedback.gate_names) == {
        "maximum_task_joint_regret",
        "p95_srmse",
    }
    assert set(feedback.parent_summary) == {
        "task_count",
        "mean_smae",
        "mean_srmse",
        "mean_joint",
        "p90_smae",
        "p95_smae",
        "p90_srmse",
        "p95_srmse",
        "invalid_count",
        "catastrophic_count",
        "clipped_count",
        "fallback_count",
        "coverage",
    }


@pytest.mark.parametrize(
    "forbidden",
    (
        {"task_id": "task_185"},
        {"entity_name": "secret"},
        {"forecast": [1.0]},
        {"truth": [1.0]},
        {"document": "secret"},
        {"quote": "secret"},
        {"residual": 0.1},
        {"per_task_metric": 0.1},
    ),
)
def test_package_feedback_rejects_nested_task_level_leakage(forbidden) -> None:
    with pytest.raises(ValueError, match="forbidden|allowlist|aggregate"):
        PackageProposalFeedback(
            parent_summary={"task_count": 1},
            rejected_summaries=(),
            gate_names=(),
            structures=({"kind": "route", "details": forbidden},),
        )


def test_package_feedback_rejects_task_identity_disguised_as_structure() -> None:
    for structure in (
        {"kind": "task_185", "structure_sha256": "a" * 64},
        {"candidate_name": "task_185", "structure_sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="closed|structure|forbidden"):
            PackageProposalFeedback(
                parent_summary={"task_count": 1},
                rejected_summaries=(),
                gate_names=(),
                structures=(structure,),
            )


@pytest.mark.parametrize(
    ("field", "leak"),
    (
        ("feature", "private entity observation"),
        ("direction", ["quoted document excerpt"]),
        ("horizon_region", {"note": "forecast truth residual"}),
        ("feature", [1.0, 2.0]),
    ),
)
def test_package_feedback_rejects_nonstructural_assumption_values(field, leak) -> None:
    assumption = {
        "candidate_name": "safe_anchor",
        "feature": "trend_strength",
        "direction": "above",
        "horizon_region": "full",
        "operator": "route",
    }
    assumption[field] = leak

    with pytest.raises(ValueError, match="assumption|closed|structural"):
        PackageProposalFeedback(
            parent_summary={"task_count": 1},
            rejected_summaries=(),
            gate_names=(),
            structures=(
                {
                    "kind": "route",
                    "parents": ("safe_anchor", "seasonal_anchor"),
                    "fallback_parent": "safe_anchor",
                    "assumptions": (assumption,),
                    "structure_sha256": "a" * 64,
                },
            ),
        )


def test_package_feedback_rejects_nested_task_identity_substring() -> None:
    with pytest.raises(ValueError, match="task identity|forbidden"):
        PackageProposalFeedback(
            parent_summary={"task_count": 1},
            rejected_summaries=(),
            gate_names=(),
            structures=(
                {
                    "kind": "route",
                    "parents": ("candidate_task_185", "seasonal_anchor"),
                    "fallback_parent": "candidate_task_185",
                    "assumptions": (
                        {
                            "candidate_name": "candidate_task_185",
                            "feature": "trend_strength",
                            "direction": "above",
                            "horizon_region": "full",
                            "operator": "route",
                        },
                    ),
                    "structure_sha256": "a" * 64,
                },
            ),
        )


def test_package_candidate_rejects_open_ended_invalid_reason(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)

    with pytest.raises(ValueError, match="reason"):
        PackageCandidate(
            slot=0,
            target="numerical",
            state=parent,
            proposal_sha256="a" * 64,
            invalid_reason="arbitrary free text",
        )


def test_package_candidate_rejects_preaccepted_child_state(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    policy = replace(
        parent.bundle.policy,
        version="v099",
        parent=parent.bundle.policy.version,
        decision_prompt="candidate decision prompt",
    )
    accepted = parent.with_policy(policy, target="decision").seal_acceptance("f" * 64)

    with pytest.raises(ValueError, match="acceptance"):
        PackageCandidate(
            slot=0,
            target="decision",
            state=accepted,
            proposal_sha256="a" * 64,
        )


def test_each_proposer_returns_exactly_three_canonical_slots(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    feedback = _feedback()
    numerical_engine = _NumericalProposer(parent)
    numerical = NumericalCandidateProposer(numerical_engine)
    parent_genome = parent.bundle.policy.retrieval_genome
    assert parent_genome is not None
    retrieval_versions = tuple(
        f"v{int(parent_genome.version[1:]) + 2 * 3 + index:03d}"
        for index in range(1, 4)
    )
    retrieval_llm = FakeLLMClient(
        [
            json.dumps(_retrieval_proposal(parent_genome, version, scope))
            for version, scope in zip(retrieval_versions, ("A", "B", "C"), strict=True)
        ]
    )
    library = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False)
    retrieval = RetrievalCandidateProposer(
        RetrievalGenomeProposer(
            retrieval_llm,
            transient_retries=2,
            version_origin=parent_genome.version,
        ),
        skill_library=library,
    )
    decision = DecisionCandidateProposer(
        _decision_engine(
            [
                json.dumps({"decision_prompt": f"decision child {index}"})
                for index in range(3)
            ]
        )
    )

    proposers = {
        "numerical": numerical,
        "retrieval": retrieval,
        "decision": decision,
    }
    for target, proposer in proposers.items():
        children = proposer.propose(
            parent,
            feedback,
            generation=2,
            child_count=3,
        )
        assert tuple(child.slot for child in children) == (0, 1, 2)
        assert all(child.target == target for child in children)
        assert len({child.proposal_sha256 for child in children}) == 3

    assert isinstance(numerical_engine.feedback, ProposerEvidence)


def test_invalid_proposal_slots_retain_exact_parent_and_unique_identity(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    parent_release = parse_numerical_supply_release(
        parent.bundle.to_payload()["numerical_release_payload"]
    )

    class InvalidNumericalProposer:
        def propose(self, *_args, **_kwargs):
            return tuple(
                NumericalCoordinateCandidate(
                    parent_release,
                    parent.registry,
                    hashlib.sha256(f"invalid:{slot}".encode()).hexdigest(),
                    invalid_reason="insufficient_valid_proposals",
                )
                for slot in range(3)
            )

    children = NumericalCandidateProposer(InvalidNumericalProposer()).propose(
        parent,
        _feedback(),
        generation=0,
        child_count=3,
    )

    assert all(child.state is parent for child in children)
    assert all(child.invalid_reason == "materialization_failed" for child in children)
    assert len({child.proposal_sha256 for child in children}) == 3


def test_retrieval_proposal_preserves_numerical_and_decision_bytes(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    parent_genome = parent.bundle.policy.retrieval_genome
    assert parent_genome is not None
    responses = [
        json.dumps(
            _retrieval_proposal(parent_genome, f"v{index:03d}", scope)
        )
        for index, scope in zip((2, 3, 4), ("A", "B", "C"), strict=True)
    ]
    proposer = RetrievalCandidateProposer(
        RetrievalGenomeProposer(
            FakeLLMClient(responses),
            transient_retries=2,
            version_origin=parent_genome.version,
        ),
        skill_library=RetrievalSkillLibrary(
            tmp_path / "retrieval-skills.json", persist=False
        ),
    )

    child = proposer.propose(
        parent,
        _feedback(),
        generation=0,
        child_count=3,
    )[0]

    before = package_principal_fingerprints(parent.bundle)
    after = package_principal_fingerprints(child.state.bundle)
    assert child.invalid_reason is None
    assert before["numerical"] == after["numerical"]
    assert before["decision"] == after["decision"]
    assert before["retrieval"] != after["retrieval"]
    assert child.state.registry is parent.registry
    manifest = child.state.bundle.policy.retrieval_release_payload["manifest"]
    assert manifest["state"] == "candidate"
    assert child.state.bundle.policy.has_accepted_retrieval_release is False
    assert getattr(child.state.bundle.policy.retrieval_skill_source, "_read_only") is True


def test_decision_proposal_cannot_change_materialized_forecasts(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)
    engine = _decision_engine(
        [
            json.dumps(
                {
                    "decision_prompt": f"decision child {index}",
                    "forecast": [1, 2],
                }
            )
            for index in range(3)
        ]
    )

    child = DecisionCandidateProposer(engine).propose(
        parent,
        _feedback(),
        generation=0,
        child_count=3,
    )[0]

    before = package_principal_fingerprints(parent.bundle)
    after = package_principal_fingerprints(child.state.bundle)
    assert child.invalid_reason is None
    assert child.state.registry is parent.registry
    assert child.state.bundle.numerical_manifest_sha256 == (
        parent.bundle.numerical_manifest_sha256
    )
    assert "forecast" not in child.state.bundle.policy.to_payload()
    assert before["numerical"] == after["numerical"]
    assert before["retrieval"] == after["retrieval"]
    assert before["decision"] != after["decision"]


def test_invalid_decision_slots_keep_exact_parent_and_invalid_schema(tmp_path) -> None:
    _task, parent = _bundle(tmp_path)

    children = DecisionCandidateProposer(
        _decision_engine(["not JSON", "not JSON", "not JSON"])
    ).propose(
        parent,
        _feedback(),
        generation=0,
        child_count=3,
    )

    assert all(child.state is parent for child in children)
    assert all(child.invalid_reason == "invalid_schema" for child in children)
    assert len({child.proposal_sha256 for child in children}) == 3


def test_retrieval_candidate_embedding_is_non_authoritative_and_rebase_is_behavior_only(
    tmp_path,
) -> None:
    _task, parent = _bundle(tmp_path)
    parent_genome = parent.bundle.policy.retrieval_genome
    assert parent_genome is not None
    candidate = RetrievalGenome.from_payload(
        _retrieval_proposal(parent_genome, "v009", "A")
    )
    library = RetrievalSkillLibrary(tmp_path / "candidate-skills.json", persist=False)

    policy = embed_retrieval_candidate(
        parent.bundle.policy,
        candidate,
        library.clone(read_only=True),
        changelog="provisional Retrieval candidate",
    )
    rebased = rebase_retrieval_candidate(candidate, parent_genome)

    assert policy.retrieval_release_payload["manifest"]["state"] == "candidate"
    assert policy.has_accepted_retrieval_release is False
    restored = HarnessPolicy(**policy.to_payload())
    assert restored.to_payload() == policy.to_payload()
    assert restored.has_accepted_retrieval_release is False
    assert rebased.version == "v002"
    assert rebased.parent == parent_genome.version
    assert retrieval_behavior_fingerprint(rebased) == retrieval_behavior_fingerprint(
        candidate
    )
