from __future__ import annotations

import copy

import pytest

from evolving_loop.package_numerical_supply import build_package_registry
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.bundle import EvolutionBundleV2, changed_scopes
from evolving_loop.v2.contracts import SanitizedEvolutionFeedback
from evolving_loop.v2.cooperative.adapters import (
    CooperativeArtifactCatalog,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from evolving_loop.v2.cooperative.proposals import propose_bundle_candidate
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2, import_numerical_seed
from tests.test_package_numerical_supply import (
    _package_for_task,
    _registry_tasks,
    _supply_release,
)


def _pair(*, alternate: bool) -> FrozenNumericalArtifactsV2:
    release = _supply_release() if alternate else _supply_release(alternatives=())
    tasks = _registry_tasks()
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _package_for_task(task, supplied),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


@pytest.fixture
def proposal_context():
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    seed_pair = _pair(alternate=False)
    alternate_pair = _pair(alternate=True)
    numerical_release, numerical_registry = catalog.add_numerical(seed_pair)
    retrieval = RetrievalModuleV2(
        1, "a" * 64, RetrievalGenome.seed().to_payload(), ()
    )
    decision = DecisionModuleV2(1, "seed prompt", (), True, 2, "last")
    retrieval_sha = catalog.add_retrieval(retrieval)
    decision_sha = catalog.add_decision(decision)
    parent = EvolutionBundleV2(
        schema_version=2,
        generation=0,
        parent_bundle_sha256=None,
        numerical_release_sha256=numerical_release,
        numerical_registry_sha256=numerical_registry,
        retrieval_release_sha256=retrieval_sha,
        decision_policy_sha256=decision_sha,
        harness_policy_sha256="5" * 64,
        archive_snapshot_sha256="6" * 64,
        scheduler_state_sha256="7" * 64,
        protocol_fingerprint="8" * 64,
        runtime_fingerprints={"python": "9" * 64},
        acceptance_evidence_sha256=None,
    )
    feedback = SanitizedEvolutionFeedback(
        parent_sha256=parent.fingerprint(),
        train_evaluation_sha256="b" * 64,
        train_objectives={"relative_joint_improvement": 0.0},
        train_behavior_descriptors={"invalid_count": 0},
        failure_categories=(),
        remaining_proposal_budget={"children": 1},
    )
    return {
        "parent": parent,
        "catalog": catalog,
        "adapters": {
            "numerical": NumericalCoordinateAdapter((alternate_pair,)),
            "retrieval": RetrievalCoordinateAdapter(),
            "decision": DecisionCoordinateAdapter(("changed prompt",)),
        },
        "feedback": feedback,
        "step": 0,
    }


@pytest.mark.parametrize("arm", ["numerical", "retrieval", "decision"])
def test_single_proposal_changes_only_its_arm(arm, proposal_context):
    candidate = propose_bundle_candidate(arm=arm, **proposal_context)

    assert candidate is not None
    child = candidate.to_child(proposal_context["parent"])
    assert changed_scopes(proposal_context["parent"], child) == (arm,)


def test_joint_proposal_combines_first_two_real_module_children(proposal_context):
    candidate = propose_bundle_candidate(arm="joint", **proposal_context)

    assert candidate is not None
    child = candidate.to_child(proposal_context["parent"])
    assert changed_scopes(proposal_context["parent"], child) == (
        "numerical",
        "retrieval",
    )


def test_joint_skips_exhausted_coordinate_and_uses_next_two(proposal_context):
    parent_pair = proposal_context["catalog"].resolve_numerical(
        proposal_context["parent"].numerical_release_sha256,
        proposal_context["parent"].numerical_registry_sha256,
    )
    proposal_context["adapters"]["numerical"] = NumericalCoordinateAdapter(
        (parent_pair,)
    )

    candidate = propose_bundle_candidate(arm="joint", **proposal_context)

    assert candidate is not None
    child = candidate.to_child(proposal_context["parent"])
    assert changed_scopes(proposal_context["parent"], child) == (
        "retrieval",
        "decision",
    )


def test_joint_returns_none_when_fewer_than_two_coordinates_can_change(
    proposal_context,
):
    parent_pair = proposal_context["catalog"].resolve_numerical(
        proposal_context["parent"].numerical_release_sha256,
        proposal_context["parent"].numerical_registry_sha256,
    )
    proposal_context["adapters"] = {
        "numerical": NumericalCoordinateAdapter((parent_pair,)),
        "retrieval": RetrievalCoordinateAdapter(),
        "decision": DecisionCoordinateAdapter(("seed prompt",)),
    }

    assert propose_bundle_candidate(arm="joint", **proposal_context) is None


def test_proposal_rejects_plain_or_tampered_feedback(proposal_context):
    with pytest.raises(TypeError, match="SanitizedEvolutionFeedback"):
        propose_bundle_candidate(
            **(proposal_context | {"arm": "retrieval", "feedback": {}})
        )

    tampered = copy.copy(proposal_context["feedback"])
    object.__setattr__(
        tampered, "train_objectives", {"nested": {"future_values": [1.0]}}
    )
    with pytest.raises(ValueError, match="reserved evaluator-only key"):
        propose_bundle_candidate(
            **(proposal_context | {"arm": "retrieval", "feedback": tampered})
        )


def test_proposal_feedback_must_be_bound_to_parent(proposal_context):
    feedback = SanitizedEvolutionFeedback(
        parent_sha256="c" * 64,
        train_evaluation_sha256="b" * 64,
        train_objectives={},
        train_behavior_descriptors={},
        failure_categories=(),
        remaining_proposal_budget={},
    )
    with pytest.raises(ValueError, match="Parent"):
        propose_bundle_candidate(
            **(proposal_context | {"arm": "retrieval", "feedback": feedback})
        )
