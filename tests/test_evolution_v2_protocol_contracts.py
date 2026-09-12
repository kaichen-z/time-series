from __future__ import annotations

import pytest

from evolving_loop.v2.protocol.contracts import (
    KINDS,
    InfrastructureProtocolV2,
    ProtocolComponentV2,
    ProtocolProposalV2,
    ProtocolReleaseV2,
)


def components() -> tuple[ProtocolComponentV2, ...]:
    return (
        ProtocolComponentV2("backbone", "last_value", 1, 1),
        ProtocolComponentV2("loader", "canonical_json", 1, 1),
        ProtocolComponentV2("verifier_strategy", "exact_support", 1, 1),
        ProtocolComponentV2("diagnostic_metric", "forecast_spread", 1, 1),
        ProtocolComponentV2("schema_migration", "identity_envelope", 1, 1),
    )


def seed_protocol() -> InfrastructureProtocolV2:
    return InfrastructureProtocolV2(1, 1, None, "a" * 64, components())


def proposal_payload(seed: InfrastructureProtocolV2, *, kind: str) -> dict[str, object]:
    component = next(item for item in components() if item.kind == kind)
    return {
        "schema_version": 1,
        "parent_protocol_sha256": seed.fingerprint(),
        "kind": kind,
        "replacement": component.to_payload(),
        "compatibility_corpus_sha256": "c" * 64,
    }


def test_proposal_versions_only_one_l1_component() -> None:
    """Changing a proposal kind must not alter the other four L1 bindings."""
    seed = seed_protocol()
    component = ProtocolComponentV2("backbone", "history_mean", 1, 1)
    proposal = ProtocolProposalV2(1, seed.fingerprint(), "backbone", component, "c" * 64)

    child = proposal.to_child(seed)

    assert child.protocol_version == 2
    assert child.parent_protocol_sha256 == seed.fingerprint()
    assert child.l0_commitment_sha256 == seed.l0_commitment_sha256
    assert child.components[1:] == seed.components[1:]
    assert InfrastructureProtocolV2.from_payload(child.to_payload()) == child


def test_mismatched_kind_and_l0_override_are_rejected() -> None:
    """A tagged replacement or an L0 override cannot enter the L1 manifest."""
    seed = seed_protocol()
    proposal = proposal_payload(seed, kind="loader")
    replacement = proposal["replacement"]
    assert isinstance(replacement, dict)
    replacement["kind"] = "backbone"
    with pytest.raises(ValueError):
        ProtocolProposalV2.from_payload(proposal)

    payload = seed.to_payload()
    payload["metric_policy"] = "0" * 64
    with pytest.raises(ValueError):
        InfrastructureProtocolV2.from_payload(payload)


@pytest.mark.parametrize("value", [0, -1, True])
def test_component_versions_must_be_positive_plain_integers(value: object) -> None:
    """Permitting zero, negatives, or bool would make adapter versions ambiguous."""
    with pytest.raises(ValueError):
        ProtocolComponentV2("backbone", "last_value", value, 1)  # type: ignore[arg-type]


def test_protocol_rejects_unknown_schema_and_duplicate_or_misordered_kinds() -> None:
    """A manifest must have exactly one component in the stable kind order."""
    seed = seed_protocol()
    unsupported = seed.to_payload()
    unsupported["schema_version"] = 2
    with pytest.raises(ValueError):
        InfrastructureProtocolV2.from_payload(unsupported)

    duplicated = seed.to_payload()
    entries = duplicated["components"]
    assert isinstance(entries, list)
    entries[1] = entries[0]
    with pytest.raises(ValueError):
        InfrastructureProtocolV2.from_payload(duplicated)

    misordered = seed.to_payload()
    ordered = misordered["components"]
    assert isinstance(ordered, list)
    ordered[0], ordered[1] = ordered[1], ordered[0]
    with pytest.raises(ValueError):
        InfrastructureProtocolV2.from_payload(misordered)


def test_contract_payloads_are_exact_canonical_and_cannot_mutate_instances() -> None:
    """Hash identities must remain stable after callers mutate returned payloads."""
    seed = seed_protocol()
    before = seed.fingerprint()
    payload = seed.to_payload()
    entries = payload["components"]
    assert isinstance(entries, list)
    entries[0]["implementation_id"] = "tampered"

    assert seed.fingerprint() == before
    assert seed.components[0].implementation_id == "last_value"
    assert seed.canonical_bytes().endswith(b"\n")
    assert tuple(component.kind for component in seed.components) == KINDS


def test_to_child_rejects_wrong_parent_or_no_component_change() -> None:
    """A proposal cannot be replayed on another parent or create a no-op child."""
    seed = seed_protocol()
    wrong_parent = ProtocolProposalV2(
        1,
        "b" * 64,
        "backbone",
        ProtocolComponentV2("backbone", "history_mean", 1, 1),
        "c" * 64,
    )
    with pytest.raises(ValueError):
        wrong_parent.to_child(seed)

    unchanged = ProtocolProposalV2(
        1,
        seed.fingerprint(),
        "backbone",
        seed.components[0],
        "c" * 64,
    )
    with pytest.raises(ValueError):
        unchanged.to_child(seed)


def test_release_is_strictly_round_trippable() -> None:
    """Publication references are fixed content hashes, not mutable configuration."""
    release = ProtocolReleaseV2(1, "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
    assert ProtocolReleaseV2.from_payload(release.to_payload()) == release

    payload = release.to_payload()
    payload["unexpected"] = "x"
    with pytest.raises(ValueError):
        ProtocolReleaseV2.from_payload(payload)
