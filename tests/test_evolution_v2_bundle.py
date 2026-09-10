from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from evolving_loop.v2 import (
    BundleContractError,
    EvolutionBundleV2,
    changed_scopes,
    principal_fingerprints,
    validate_child_scope,
)


def sha256_for(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def seed_bundle() -> EvolutionBundleV2:
    return EvolutionBundleV2(
        schema_version=2,
        generation=0,
        parent_bundle_sha256=None,
        numerical_release_sha256=sha256_for("numerical-release"),
        numerical_registry_sha256=sha256_for("numerical-registry"),
        retrieval_release_sha256=sha256_for("retrieval-release"),
        decision_policy_sha256=sha256_for("decision-policy"),
        harness_policy_sha256=sha256_for("harness-policy"),
        archive_snapshot_sha256=sha256_for("archive-snapshot"),
        scheduler_state_sha256=sha256_for("scheduler-state"),
        protocol_fingerprint=sha256_for("protocol"),
        runtime_fingerprints={
            "python": sha256_for("python-runtime"),
            "worker": sha256_for("worker-runtime"),
        },
        acceptance_evidence_sha256=None,
    )


def reject_transition(
    parent: EvolutionBundleV2, child: EvolutionBundleV2
) -> EvolutionBundleV2:
    del child
    return parent


def test_seed_bundle_binds_every_replay_dependency():
    bundle = seed_bundle()
    assert set(bundle.to_payload()) == {
        "schema_version",
        "generation",
        "parent_bundle_sha256",
        "numerical_release_sha256",
        "numerical_registry_sha256",
        "retrieval_release_sha256",
        "decision_policy_sha256",
        "harness_policy_sha256",
        "archive_snapshot_sha256",
        "scheduler_state_sha256",
        "protocol_fingerprint",
        "runtime_fingerprints",
        "acceptance_evidence_sha256",
    }
    assert bundle.fingerprint() == hashlib.sha256(bundle.canonical_bytes()).hexdigest()


def test_single_coordinate_and_joint_children_enforce_actual_diff():
    parent = seed_bundle()
    numerical = parent.provisional_child("numerical", {"numerical": "1" * 64})
    assert numerical.numerical_release_sha256 == "1" * 64
    assert numerical.numerical_registry_sha256 == "1" * 64
    assert changed_scopes(parent, numerical) == ("numerical",)

    with pytest.raises(BundleContractError, match="exactly one"):
        parent.provisional_child(
            "retrieval", {"retrieval": "2" * 64, "decision": "3" * 64}
        )

    joint = parent.provisional_child(
        "joint", {"retrieval": "2" * 64, "decision": "3" * 64}
    )
    assert changed_scopes(parent, joint) == ("retrieval", "decision")


@pytest.mark.parametrize(
    "changes",
    [
        {"harness": "1" * 64},
        {"harness_policy_sha256": "1" * 64},
        {"protocol": "1" * 64},
        {"runtime_fingerprints": "1" * 64},
        {"archive": "1" * 64},
        {"scheduler": "1" * 64},
        {"acceptance_evidence_sha256": "1" * 64},
    ],
)
def test_candidate_updates_reject_host_owned_metadata(changes):
    with pytest.raises(BundleContractError, match="principal"):
        seed_bundle().provisional_child("numerical", changes)


def test_child_lineage_and_non_principal_authority_are_host_computed():
    parent = seed_bundle()
    child = parent.provisional_child("retrieval", {"retrieval": "2" * 64})

    assert child.generation == 1
    assert child.parent_bundle_sha256 == parent.fingerprint()
    assert child.acceptance_evidence_sha256 is None

    for field in (
        "harness_policy_sha256",
        "archive_snapshot_sha256",
        "scheduler_state_sha256",
        "protocol_fingerprint",
        "runtime_fingerprints",
    ):
        assert getattr(child, field) == getattr(parent, field)

    invalid_children = (
        replace(child, generation=2),
        replace(child, parent_bundle_sha256=sha256_for("other-parent")),
        replace(child, harness_policy_sha256=sha256_for("other-harness")),
        replace(child, protocol_fingerprint=sha256_for("other-protocol")),
        replace(
            child,
            runtime_fingerprints={"python": sha256_for("other-runtime")},
        ),
        replace(child, archive_snapshot_sha256=sha256_for("other-archive")),
        replace(child, scheduler_state_sha256=sha256_for("other-scheduler")),
        replace(child, acceptance_evidence_sha256=sha256_for("early-evidence")),
    )
    for invalid in invalid_children:
        with pytest.raises(BundleContractError):
            validate_child_scope(parent, invalid, "retrieval")


def test_scope_validation_requires_declared_change_and_atomic_numerical_pair():
    parent = seed_bundle()
    unchanged = replace(
        parent,
        generation=1,
        parent_bundle_sha256=parent.fingerprint(),
    )
    with pytest.raises(BundleContractError, match="exactly one"):
        validate_child_scope(parent, unchanged, "retrieval")

    wrong_scope = replace(
        unchanged,
        decision_policy_sha256=sha256_for("new-decision"),
    )
    with pytest.raises(BundleContractError, match="declared target"):
        validate_child_scope(parent, wrong_scope, "retrieval")

    release_only = replace(
        unchanged,
        numerical_release_sha256=sha256_for("new-numerical-release"),
    )
    with pytest.raises(BundleContractError, match="release and registry"):
        validate_child_scope(parent, release_only, "numerical")

    with pytest.raises(BundleContractError, match="at least two"):
        validate_child_scope(parent, wrong_scope, "joint")


def test_bundle_seed_and_child_lineage_constraints():
    seed = seed_bundle()
    with pytest.raises(BundleContractError, match="seed.*Parent"):
        replace(seed, parent_bundle_sha256=sha256_for("parent"))
    with pytest.raises(BundleContractError, match="seed.*evidence"):
        replace(seed, acceptance_evidence_sha256=sha256_for("evidence"))
    with pytest.raises(BundleContractError, match="positive-generation.*Parent"):
        replace(seed, generation=1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("generation", True),
        ("generation", -1),
        ("runtime_fingerprints", {}),
    ],
)
def test_bundle_rejects_schema_integer_aliases_and_empty_runtime_map(field, value):
    with pytest.raises(BundleContractError, match=field):
        replace(seed_bundle(), **{field: value})


def test_bundle_requires_exact_payload_schema_and_round_trips_exact_bytes():
    original = seed_bundle()
    payload = original.to_payload()
    restored = EvolutionBundleV2.from_payload(payload)
    assert restored.to_payload() == payload
    assert restored.canonical_bytes() == original.canonical_bytes()

    for malformed in (
        {key: value for key, value in payload.items() if key != "generation"},
        payload | {"coordinate": "seed"},
    ):
        with pytest.raises(BundleContractError, match="exact schema"):
            EvolutionBundleV2.from_payload(malformed)


def test_runtime_fingerprints_are_sorted_validated_and_immutable():
    bundle = seed_bundle()
    assert list(bundle.runtime_fingerprints) == ["python", "worker"]
    assert list(bundle.to_payload()["runtime_fingerprints"]) == ["python", "worker"]

    with pytest.raises(TypeError):
        bundle.runtime_fingerprints["python"] = sha256_for("replacement")
    with pytest.raises(BundleContractError, match="runtime_fingerprints.python"):
        replace(bundle, runtime_fingerprints={"python": "not-a-sha"})
    with pytest.raises(BundleContractError, match="runtime_fingerprints.*keys"):
        replace(bundle, runtime_fingerprints={1: sha256_for("runtime")})


def test_principal_fingerprints_ignore_host_owned_and_acceptance_metadata():
    bundle = seed_bundle()
    host_changed = replace(
        bundle,
        archive_snapshot_sha256=sha256_for("later-archive"),
        scheduler_state_sha256=sha256_for("later-scheduler"),
    )
    assert principal_fingerprints(host_changed) == principal_fingerprints(bundle)
    assert changed_scopes(bundle, host_changed) == ()
    assert tuple(principal_fingerprints(bundle)) == (
        "numerical",
        "retrieval",
        "decision",
    )


def test_acceptance_sealing_is_host_only_non_circular_and_one_time():
    parent = seed_bundle()
    provisional = parent.provisional_child("decision", {"decision": "4" * 64})
    evidence = hashlib.sha256(
        provisional.canonical_bytes() + b"evaluation-evidence"
    ).hexdigest()
    sealed = provisional.seal_acceptance(
        evidence,
        sha256_for("post-evaluation-archive"),
        sha256_for("post-evaluation-scheduler"),
    )

    assert sealed.acceptance_evidence_sha256 == evidence
    assert sealed.archive_snapshot_sha256 == sha256_for("post-evaluation-archive")
    assert sealed.scheduler_state_sha256 == sha256_for("post-evaluation-scheduler")
    assert sealed.fingerprint() != provisional.fingerprint()
    assert principal_fingerprints(sealed) == principal_fingerprints(provisional)
    assert sealed.parent_bundle_sha256 == provisional.parent_bundle_sha256

    with pytest.raises(BundleContractError, match="only once"):
        sealed.seal_acceptance(
            sha256_for("replacement-evidence"),
            sha256_for("replacement-archive"),
            sha256_for("replacement-scheduler"),
        )
    with pytest.raises(BundleContractError, match="provisional Child"):
        parent.seal_acceptance(
            evidence,
            sha256_for("post-evaluation-archive"),
            sha256_for("post-evaluation-scheduler"),
        )


def test_rejected_transition_returns_exact_parent_object_and_bytes():
    parent = seed_bundle()
    child = parent.provisional_child("retrieval", {"retrieval": "2" * 64})
    returned = reject_transition(parent, child)
    assert returned is parent
    assert returned.canonical_bytes() == parent.canonical_bytes()
