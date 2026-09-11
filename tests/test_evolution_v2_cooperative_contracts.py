from __future__ import annotations

import copy
import hashlib

import pytest

from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.bundle import EvolutionBundleV2, changed_scopes
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.cooperative.contracts import (
    BundleCandidateV2,
    CooperativeCheckpointV2,
    CooperativeRunResultV2,
    CooperativeSchedulerStateV2,
    DecisionModuleV2,
    RetrievalModuleV2,
    SchedulerArmStateV2,
)


SHA = "a" * 64
OTHER_SHA = "b" * 64


@pytest.fixture
def parent_bundle() -> EvolutionBundleV2:
    return EvolutionBundleV2(
        schema_version=2,
        generation=0,
        parent_bundle_sha256=None,
        numerical_release_sha256="1" * 64,
        numerical_registry_sha256="2" * 64,
        retrieval_release_sha256="3" * 64,
        decision_policy_sha256="4" * 64,
        harness_policy_sha256="5" * 64,
        archive_snapshot_sha256="6" * 64,
        scheduler_state_sha256="7" * 64,
        protocol_fingerprint="8" * 64,
        runtime_fingerprints={"python": "9" * 64},
        acceptance_evidence_sha256=None,
    )


def scheduler_state() -> CooperativeSchedulerStateV2:
    return CooperativeSchedulerStateV2(
        schema_version=1,
        mode="ucb",
        seed=17,
        draw_counter=0,
        completed_step=0,
        discount=0.9,
        arms={
            "numerical": SchedulerArmStateV2(0, 0, 0.0, 0.0),
            "retrieval": SchedulerArmStateV2(0, 0, 0.0, 0.0),
        },
    )


def payloads() -> dict[type, dict[str, object]]:
    state = scheduler_state()
    checkpoint_body = {
        "schema_version": 1,
        "config_sha256": SHA,
        "input_sha256s": {"seed_supply": OTHER_SHA},
        "active_bundle_sha256": SHA,
        "scheduler_state": state.to_payload(),
        "scheduler_state_sha256": state.fingerprint(),
        "next_step": 0,
        "accepted_steps": 0,
        "rejected_steps": 0,
        "completed_candidate_sha256s": [],
        "kernel_checkpoint_sha256": OTHER_SHA,
    }
    return {
        RetrievalModuleV2: {
            "schema_version": 1,
            "source_release_sha256": SHA,
            "genome_payload": RetrievalGenome.seed().to_payload(),
            "skills_payload": [],
        },
        DecisionModuleV2: {
            "schema_version": 1,
            "prompt": "Choose a safe forecast.",
            "skills": [],
            "enable_evidence_adjustments": True,
            "max_evidence_adjustments": 2,
            "aggregation": "last",
        },
        BundleCandidateV2: {
            "schema_version": 1,
            "target": "retrieval",
            "parent_bundle_sha256": SHA,
            "operator": "typed_mutation",
            "numerical_release_sha256": None,
            "numerical_registry_sha256": None,
            "retrieval_release_sha256": OTHER_SHA,
            "decision_policy_sha256": None,
        },
        SchedulerArmStateV2: {
            "attempts": 2,
            "acceptances": 1,
            "discounted_reward_sum": 0.5,
            "discounted_cost_sum": 0.25,
        },
        CooperativeSchedulerStateV2: state.to_payload(),
        CooperativeCheckpointV2: checkpoint_body
        | {"checkpoint_sha256": fingerprint_payload(checkpoint_body)},
        CooperativeRunResultV2: {
            "schema_version": 1,
            "status": "cooperative_complete",
            "active_bundle_sha256": SHA,
            "scheduler_state_sha256": state.fingerprint(),
            "attempted_arms": ["numerical", "retrieval"],
            "accepted_steps": 1,
            "rejected_steps": 1,
            "public_test_accessed": False,
        },
    }


@pytest.mark.parametrize("cls", tuple(payloads()))
def test_all_artifacts_round_trip_as_frozen_canonical_values(cls):
    payload = copy.deepcopy(payloads()[cls])
    artifact = cls.from_payload(payload)
    assert artifact.to_payload() == payload
    assert cls.from_payload(artifact.to_payload()) == artifact
    assert artifact.canonical_bytes() == canonical_v2_bytes(payload)
    assert artifact.fingerprint() == hashlib.sha256(
        artifact.canonical_bytes()
    ).hexdigest()
    assert not hasattr(artifact, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        setattr(artifact, next(iter(payload)), None)


@pytest.mark.parametrize("cls", tuple(payloads()))
def test_all_artifacts_require_their_exact_schema(cls):
    payload = copy.deepcopy(payloads()[cls])
    payload["dev_metrics"] = {"mean_smae": 0.1}
    with pytest.raises(ValueError, match="exact schema"):
        cls.from_payload(payload)


def test_joint_candidate_requires_two_changed_scopes(parent_bundle):
    candidate = BundleCandidateV2(
        schema_version=1,
        target="joint",
        parent_bundle_sha256=parent_bundle.fingerprint(),
        operator="paired_typed_mutation",
        numerical_release_sha256=None,
        numerical_registry_sha256=None,
        retrieval_release_sha256="a" * 64,
        decision_policy_sha256=None,
    )
    with pytest.raises(ValueError, match="at least two"):
        candidate.to_child(parent_bundle)


def test_single_candidate_constructs_an_owned_child(parent_bundle):
    candidate = BundleCandidateV2(
        1,
        "retrieval",
        parent_bundle.fingerprint(),
        "typed_mutation",
        None,
        None,
        "a" * 64,
        None,
    )
    child = candidate.to_child(parent_bundle)
    assert changed_scopes(parent_bundle, child) == ("retrieval",)


def test_scheduler_state_rejects_dev_or_task_feedback():
    payload = scheduler_state().to_payload()
    payload["dev_metrics"] = {"mean_smae": 0.1}
    with pytest.raises(ValueError, match="exact schema"):
        CooperativeSchedulerStateV2.from_payload(payload)


def test_scheduler_arms_must_use_canonical_scope_order():
    payload = scheduler_state().to_payload()
    payload["arms"] = {
        "retrieval": SchedulerArmStateV2(0, 0, 0.0, 0.0).to_payload(),
        "numerical": SchedulerArmStateV2(0, 0, 0.0, 0.0).to_payload(),
    }
    with pytest.raises(ValueError, match="canonical order"):
        CooperativeSchedulerStateV2.from_payload(payload)


def test_checkpoint_binds_scheduler_and_its_own_body():
    payload = payloads()[CooperativeCheckpointV2]
    payload["scheduler_state_sha256"] = OTHER_SHA
    body = dict(payload)
    body.pop("checkpoint_sha256")
    payload["checkpoint_sha256"] = fingerprint_payload(body)
    with pytest.raises(ValueError, match="scheduler state SHA mismatch"):
        CooperativeCheckpointV2.from_payload(payload)
