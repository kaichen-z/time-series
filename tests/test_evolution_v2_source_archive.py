from __future__ import annotations

import pytest

from common.payload import strict_json_loads
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.source import (
    SourceArchiveV2,
    SourceRequestV2,
    SourceVariantV2,
    propose_sources,
    run_policy,
)


def test_nonactive_stepping_stone_remains_a_parent(tmp_path) -> None:
    """Deleting non-active eligible branches would prevent later useful descendants."""
    seed = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "numerical"\n', "a" * 64, "b" * 64
    )
    archive = SourceArchiveV2(tmp_path)
    archive.add(seed)
    archive.close(seed.fingerprint(), status="eligible")

    children = propose_sources(seed, draw_counter=0)
    for child in children:
        archive.add(child)
        archive.close(child.fingerprint(), status="eligible", train_gain=0.1)

    seen = {archive.sample(index).fingerprint() for index in range(3)}
    assert seen == {seed.fingerprint(), *(child.fingerprint() for child in children)}

    branch = propose_sources(children[0], draw_counter=1, limit=1)[0]
    archive.add(branch)
    assert branch.parent_source_sha256 == children[0].fingerprint()
    assert SourceArchiveV2(tmp_path).snapshot_sha256() == archive.snapshot_sha256()


def test_terminal_source_is_not_sampled_and_dev_metadata_is_rejected(tmp_path) -> None:
    """A closed failure must neither re-enter selection nor carry held-out credit."""
    source = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "numerical"\n', "a" * 64, "b" * 64
    )
    archive = SourceArchiveV2(tmp_path)
    archive.add(source)
    archive.close(source.fingerprint(), status="terminal")

    with pytest.raises(ValueError, match="no eligible"):
        archive.sample(0)

    event_path = tmp_path / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    event = strict_json_loads(lines[0], context="event")
    event["event"]["dev_reward"] = 0.5
    lines[0] = canonical_v2_bytes(event).decode("utf-8").strip()
    event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        SourceArchiveV2(tmp_path)


def test_invalid_source_is_retained_before_its_terminal_close(tmp_path) -> None:
    """Dropping a rejected source would erase the lineage/evidence needed for replay."""
    invalid = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "disabled-arm"\n', "a" * 64, "b" * 64
    )
    archive = SourceArchiveV2(tmp_path)
    assert archive.add(invalid) == invalid.fingerprint()

    with pytest.raises(ValueError, match="policy probe"):
        archive.close(invalid.fingerprint(), status="eligible")
    archive.close(invalid.fingerprint(), status="terminal")
    with pytest.raises(ValueError, match="no eligible"):
        SourceArchiveV2(tmp_path).sample(0)


def test_load_rejects_tampered_object_and_broken_event_prefix(tmp_path) -> None:
    """A replay must not accept either changed source bytes or a spliced lifecycle."""
    seed = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "numerical"\n', "a" * 64, "b" * 64
    )
    archive = SourceArchiveV2(tmp_path / "object")
    archive.add(seed)
    (tmp_path / "object" / "objects" / f"{seed.fingerprint()}.json").write_bytes(
        canonical_v2_bytes(SourceVariantV2.seed(
            'def choose_arm(request):\n    return "decision"\n', "a" * 64, "b" * 64
        ).to_payload())
    )
    with pytest.raises(ValueError, match="digest"):
        SourceArchiveV2(tmp_path / "object")

    archive = SourceArchiveV2(tmp_path / "prefix")
    archive.add(seed)
    archive.close(seed.fingerprint(), status="terminal")
    event_path = tmp_path / "prefix" / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    close_event = strict_json_loads(lines[1], context="close event")
    close_event["event"]["previous_event_sha256"] = None
    close_event["event_sha256"] = fingerprint_payload(close_event["event"])
    lines[1] = canonical_v2_bytes(close_event).decode("utf-8").strip()
    event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        SourceArchiveV2(tmp_path / "prefix")


def test_proposals_are_distinct_audited_policies_for_enabled_arms() -> None:
    """Template drift must not emit a duplicate or an arm unavailable to the request."""
    parent = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "numerical"\n', "a" * 64, "b" * 64
    )
    request = SourceRequestV2(
        1, ("numerical", "decision"), 3, 11, {"numerical": 0.2, "decision": 0.1}
    )
    children = propose_sources(parent, draw_counter=0)

    assert len(children) == 2
    assert len({child.source for child in children}) == 2
    assert all(child.source != parent.source for child in children)
    assert all(child.operator.startswith("seed-template:") for child in children)
    assert all(run_policy(child, request) in request.enabled_arms for child in children)


def test_proposal_limit_never_exceeds_the_two_child_epoch_cap() -> None:
    """A caller cannot bypass the fixed per-epoch source proposal budget."""
    parent = SourceVariantV2.seed(
        'def choose_arm(request):\n    return "numerical"\n', "a" * 64, "b" * 64
    )

    assert propose_sources(parent, draw_counter=0, limit=0) == ()
    with pytest.raises(ValueError, match="at most 2"):
        propose_sources(parent, draw_counter=0, limit=3)
