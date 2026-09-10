from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from evolving_loop.v2.archive import (
    ArchiveContractError,
    ArchiveRecord,
    EvolutionArchive,
)
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload


def sha256_for(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


PROTOCOL_SHA = sha256_for("protocol")
RUNTIMES = {"python": sha256_for("python-runtime")}


@dataclass(frozen=True)
class FakeArtifact:
    label: str
    protocol_fingerprint: str = PROTOCOL_SHA
    runtime_fingerprints: object = None

    def __post_init__(self) -> None:
        if self.runtime_fingerprints is None:
            object.__setattr__(self, "runtime_fingerprints", dict(RUNTIMES))

    @property
    def artifact_kind(self) -> str:
        return "fake"

    @property
    def schema_version(self) -> int:
        return 1

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "label": self.label,
            "protocol_fingerprint": self.protocol_fingerprint,
            "runtime_fingerprints": self.runtime_fingerprints,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class MinimalProtocolArtifact:
    label: str

    @property
    def artifact_kind(self) -> str:
        return "minimal"

    @property
    def schema_version(self) -> int:
        return 1

    def to_payload(self) -> dict[str, object]:
        return {"label": self.label}

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def train_record_payload(
    artifact: FakeArtifact | None = None,
    *,
    parents: tuple[str, ...] = (),
) -> dict[str, object]:
    artifact = artifact or FakeArtifact("record")
    return {
        "schema_version": 1,
        "artifact_sha256": artifact.fingerprint(),
        "artifact_kind": artifact.artifact_kind,
        "parent_sha256s": list(parents),
        "mutation_operator": "seed" if not parents else "mutate",
        "protocol_fingerprint": artifact.protocol_fingerprint,
        "runtime_fingerprints": dict(artifact.runtime_fingerprints),
        "train_behavior_descriptors": {"label_firewall_runtime": "sealed"},
        "train_objectives": {"smae": 0.5},
        "evaluation_status": "passed",
        "resource_use": {"wall_seconds": 1.0, "task_executions": 2},
        "accepted_release_sha256s": [],
        "source_lineage_sha256s": [],
    }


def record_for(
    artifact: FakeArtifact, *, parents: tuple[str, ...] = ()
) -> ArchiveRecord:
    return ArchiveRecord.from_payload(train_record_payload(artifact, parents=parents))


def test_archive_is_content_addressed_append_only_and_reconstructs_lineage(tmp_path):
    archive = EvolutionArchive(tmp_path / "archive")
    parent_artifact = FakeArtifact("parent")
    parent = archive.append(parent_artifact, record_for(parent_artifact))
    child_artifact = FakeArtifact("child")
    child = archive.append(
        child_artifact, record_for(child_artifact, parents=(parent,))
    )

    assert (tmp_path / "archive" / "objects" / f"{child}.json").is_file()
    assert archive.lineage(child) == (parent, child)
    with pytest.raises(ArchiveContractError, match="already indexed"):
        archive.append(
            child_artifact, record_for(child_artifact, parents=(parent,))
        )


def test_archive_accepts_protocol_artifact_without_duplicate_payload_metadata(tmp_path):
    artifact = MinimalProtocolArtifact("minimal")
    payload = train_record_payload()
    payload["artifact_sha256"] = artifact.fingerprint()
    payload["artifact_kind"] = artifact.artifact_kind
    record = ArchiveRecord.from_payload(payload)

    archive = EvolutionArchive(tmp_path / "archive")
    identity = archive.append(artifact, record)

    assert identity == artifact.fingerprint()
    assert EvolutionArchive(archive.root).lineage(identity) == (identity,)


def test_archive_record_has_exact_train_only_schema_and_no_reserved_key_segments():
    payload = train_record_payload()
    record = ArchiveRecord.from_payload(payload)
    assert record.to_payload() == payload

    with pytest.raises(ArchiveContractError, match="exact schema"):
        ArchiveRecord.from_payload(payload | {"dev_metrics": {"smae": 0.1}})

    for reserved in (
        "dev_metrics",
        "dev_comparison",
        "public_ids",
        "future_values",
        "evaluator_labels",
        "holdout",
    ):
        hostile = dict(payload)
        hostile["resource_use"] = {"nested": [{reserved: "secret"}]}
        with pytest.raises(ArchiveContractError, match=reserved):
            ArchiveRecord.from_payload(hostile)

    harmless = dict(payload)
    harmless["train_behavior_descriptors"] = {
        "label_firewall_runtime": "sealed"
    }
    assert ArchiveRecord.from_payload(harmless).to_payload() == harmless


def test_archive_record_rejects_non_finite_and_non_terminal_status():
    payload = train_record_payload()
    payload["train_objectives"] = {"smae": math.nan}
    with pytest.raises(ArchiveContractError, match="finite"):
        ArchiveRecord.from_payload(payload)

    payload = train_record_payload()
    payload["evaluation_status"] = "running"
    with pytest.raises(ArchiveContractError, match="terminal evaluation_status"):
        ArchiveRecord.from_payload(payload)


def test_archive_rejects_missing_parent_and_artifact_record_digest_mismatch(tmp_path):
    archive = EvolutionArchive(tmp_path / "archive")
    artifact = FakeArtifact("child")
    with pytest.raises(ArchiveContractError, match="parent.*not indexed"):
        archive.append(
            artifact, record_for(artifact, parents=(sha256_for("missing"),))
        )

    other = FakeArtifact("other")
    with pytest.raises(ArchiveContractError, match="artifact SHA"):
        archive.append(artifact, record_for(other))


@pytest.mark.parametrize("binding", ["protocol", "runtime"])
def test_archive_rejects_artifact_protocol_or_runtime_mismatch(tmp_path, binding):
    archive = EvolutionArchive(tmp_path / "archive")
    artifact = FakeArtifact("candidate")
    payload = train_record_payload(artifact)
    if binding == "protocol":
        payload["protocol_fingerprint"] = sha256_for("other-protocol")
    else:
        payload["runtime_fingerprints"] = {"python": sha256_for("other-runtime")}

    with pytest.raises(ArchiveContractError, match=binding):
        archive.append(artifact, ArchiveRecord.from_payload(payload))


def test_reload_rejects_truncated_index_line(tmp_path):
    root = tmp_path / "archive"
    archive = EvolutionArchive(root)
    artifact = FakeArtifact("parent")
    archive.append(artifact, record_for(artifact))
    index = root / "index.jsonl"
    index.write_bytes(index.read_bytes()[:-1])

    with pytest.raises(ArchiveContractError, match="truncated"):
        EvolutionArchive(root)


def test_reload_rejects_object_digest_mismatch(tmp_path):
    root = tmp_path / "archive"
    archive = EvolutionArchive(root)
    artifact = FakeArtifact("parent")
    identity = archive.append(artifact, record_for(artifact))
    (root / "objects" / f"{identity}.json").write_bytes(
        canonical_v2_bytes(FakeArtifact("tampered").to_payload())
    )

    with pytest.raises(ArchiveContractError, match="object digest"):
        EvolutionArchive(root)


def test_reload_rejects_duplicate_index_entry(tmp_path):
    root = tmp_path / "archive"
    archive = EvolutionArchive(root)
    artifact = FakeArtifact("parent")
    archive.append(artifact, record_for(artifact))
    index = root / "index.jsonl"
    index.write_bytes(index.read_bytes() + index.read_bytes())

    with pytest.raises(ArchiveContractError, match="already indexed"):
        EvolutionArchive(root)


def test_reload_verifies_record_digest_and_exact_canonical_envelope(tmp_path):
    root = tmp_path / "archive"
    archive = EvolutionArchive(root)
    artifact = FakeArtifact("parent")
    archive.append(artifact, record_for(artifact))
    index = root / "index.jsonl"
    envelope = {
        "record": train_record_payload(artifact),
        "record_sha256": sha256_for("fabricated-record"),
    }
    index.write_bytes(canonical_v2_bytes(envelope))

    with pytest.raises(ArchiveContractError, match="record digest"):
        EvolutionArchive(root)


def test_archive_snapshot_sha_is_stable_and_hashes_ordered_canonical_index(tmp_path):
    root = tmp_path / "archive"
    archive = EvolutionArchive(root)
    parent_artifact = FakeArtifact("parent")
    parent = archive.append(parent_artifact, record_for(parent_artifact))
    child_artifact = FakeArtifact("child")
    archive.append(child_artifact, record_for(child_artifact, parents=(parent,)))

    index_bytes = (root / "index.jsonl").read_bytes()
    expected = hashlib.sha256(index_bytes).hexdigest()
    assert archive.snapshot_sha256() == expected
    assert EvolutionArchive(root).snapshot_sha256() == expected
    assert archive.snapshot_sha256() != fingerprint_payload(
        {"records": list(reversed(index_bytes.decode("utf-8").splitlines()))}
    )
