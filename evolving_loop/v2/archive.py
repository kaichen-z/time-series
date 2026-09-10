"""Content-addressed append-only archive for Train-only Evolution V2 state."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from common.payload import strict_json_loads

from .contracts import (
    EvolutionArtifact,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)
from .store import (
    StoreContractError,
    _ensure_directory,
    append_jsonl,
    write_once_json,
)


_RECORD_FIELDS = (
    "schema_version",
    "artifact_sha256",
    "artifact_kind",
    "parent_sha256s",
    "mutation_operator",
    "protocol_fingerprint",
    "runtime_fingerprints",
    "train_behavior_descriptors",
    "train_objectives",
    "evaluation_status",
    "resource_use",
    "accepted_release_sha256s",
    "source_lineage_sha256s",
)
_ENVELOPE_FIELDS = frozenset({"record", "record_sha256"})
_RESERVED_EVALUATOR_KEYS = frozenset(
    {
        "dev_metrics",
        "dev_comparison",
        "public_ids",
        "future_values",
        "evaluator_labels",
        "holdout",
    }
)
_TERMINAL_EVALUATION_STATUSES = frozenset(
    {"seed", "passed", "failed", "invalid"}
)


class ArchiveContractError(ValueError):
    """Raised when archive content violates identity or Train-only contracts."""


def _archive_sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except ValueError as error:
        raise ArchiveContractError(str(error)) from error


def _plain_json(value: object, field: str) -> object:
    try:
        encoded = canonical_v2_bytes({"value": value})
        parsed = strict_json_loads(encoded.decode("utf-8"), context=field)
    except (TypeError, ValueError) as error:
        raise ArchiveContractError(str(error)) from error
    assert isinstance(parsed, dict)
    return parsed["value"]


def _mapping(value: object, field: str) -> dict[str, object]:
    plain = _plain_json(value, field)
    if not isinstance(plain, dict):
        raise ArchiveContractError(f"{field} must be an object")
    return plain


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _reject_evaluator_keys(value: object, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RESERVED_EVALUATOR_KEYS:
                raise ArchiveContractError(
                    f"{field} contains reserved evaluator-only key {key}"
                )
            _reject_evaluator_keys(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_evaluator_keys(item, f"{field}[]")


def _sha256_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArchiveContractError(f"{field} must be a list of SHA-256 strings")
    result = tuple(
        _archive_sha256(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ArchiveContractError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Closed Train-only metadata indexed for one immutable artifact."""

    schema_version: int
    artifact_sha256: str
    artifact_kind: str
    parent_sha256s: tuple[str, ...]
    mutation_operator: str
    protocol_fingerprint: str
    runtime_fingerprints: Mapping[str, str]
    train_behavior_descriptors: Mapping[str, object]
    train_objectives: Mapping[str, object]
    evaluation_status: str
    resource_use: Mapping[str, object]
    accepted_release_sha256s: tuple[str, ...]
    source_lineage_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ArchiveContractError("schema_version must be a positive integer")
        artifact_sha256 = _archive_sha256(
            self.artifact_sha256, "artifact_sha256"
        )
        if type(self.artifact_kind) is not str or not self.artifact_kind:
            raise ArchiveContractError("artifact_kind must be a non-empty string")
        if type(self.mutation_operator) is not str or not self.mutation_operator:
            raise ArchiveContractError("mutation_operator must be a non-empty string")
        parents = _sha256_sequence(self.parent_sha256s, "parent_sha256s")
        if artifact_sha256 in parents:
            raise ArchiveContractError("artifact cannot be its own parent")
        protocol = _archive_sha256(
            self.protocol_fingerprint, "protocol_fingerprint"
        )

        if not isinstance(self.runtime_fingerprints, Mapping):
            raise ArchiveContractError("runtime_fingerprints must be a non-empty object")
        runtime = dict(self.runtime_fingerprints)
        if not runtime:
            raise ArchiveContractError("runtime_fingerprints must be a non-empty object")
        if any(type(name) is not str or not name for name in runtime):
            raise ArchiveContractError(
                "runtime_fingerprints keys must be non-empty strings"
            )
        checked_runtime = {
            name: _archive_sha256(value, f"runtime_fingerprints.{name}")
            for name, value in runtime.items()
        }

        descriptors = _mapping(
            self.train_behavior_descriptors, "train_behavior_descriptors"
        )
        objectives = _mapping(self.train_objectives, "train_objectives")
        resource_use = _mapping(self.resource_use, "resource_use")
        accepted = _sha256_sequence(
            self.accepted_release_sha256s, "accepted_release_sha256s"
        )
        source_lineage = _sha256_sequence(
            self.source_lineage_sha256s, "source_lineage_sha256s"
        )
        if (
            type(self.evaluation_status) is not str
            or self.evaluation_status not in _TERMINAL_EVALUATION_STATUSES
        ):
            raise ArchiveContractError(
                "evaluation_status must be a terminal evaluation_status: "
                + ", ".join(sorted(_TERMINAL_EVALUATION_STATUSES))
            )

        for field, value in (
            ("runtime_fingerprints", checked_runtime),
            ("train_behavior_descriptors", descriptors),
            ("train_objectives", objectives),
            ("resource_use", resource_use),
        ):
            _reject_evaluator_keys(value, field)

        object.__setattr__(self, "artifact_sha256", artifact_sha256)
        object.__setattr__(self, "parent_sha256s", parents)
        object.__setattr__(self, "protocol_fingerprint", protocol)
        object.__setattr__(
            self,
            "runtime_fingerprints",
            MappingProxyType(dict(sorted(checked_runtime.items()))),
        )
        object.__setattr__(
            self, "train_behavior_descriptors", _freeze(descriptors)
        )
        object.__setattr__(self, "train_objectives", _freeze(objectives))
        object.__setattr__(self, "resource_use", _freeze(resource_use))
        object.__setattr__(self, "accepted_release_sha256s", accepted)
        object.__setattr__(self, "source_lineage_sha256s", source_lineage)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArchiveRecord":
        if not isinstance(payload, Mapping) or any(
            type(key) is not str for key in payload
        ):
            raise ArchiveContractError("ArchiveRecord must use the exact schema")
        actual = set(payload)
        expected = set(_RECORD_FIELDS)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            suffix = f" ({', '.join(details)})" if details else ""
            raise ArchiveContractError(
                f"ArchiveRecord must use the exact schema{suffix}"
            )
        return cls(
            **{field: payload[field] for field in _RECORD_FIELDS}  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "artifact_kind": self.artifact_kind,
            "parent_sha256s": list(self.parent_sha256s),
            "mutation_operator": self.mutation_operator,
            "protocol_fingerprint": self.protocol_fingerprint,
            "runtime_fingerprints": dict(self.runtime_fingerprints),
            "train_behavior_descriptors": _thaw(
                self.train_behavior_descriptors
            ),
            "train_objectives": _thaw(self.train_objectives),
            "evaluation_status": self.evaluation_status,
            "resource_use": _thaw(self.resource_use),
            "accepted_release_sha256s": list(self.accepted_release_sha256s),
            "source_lineage_sha256s": list(self.source_lineage_sha256s),
        }


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


class EvolutionArchive:
    """Immutable object store plus canonical append-only record index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise ArchiveContractError("archive root must be a directory")
        self.objects = self.root / "objects"
        self.index = self.root / "index.jsonl"
        _ensure_directory(self.objects)
        self._records: dict[str, ArchiveRecord] = {}
        self._load()

    def append(self, artifact: EvolutionArtifact, record: ArchiveRecord) -> str:
        if not isinstance(record, ArchiveRecord):
            raise ArchiveContractError("record must be an ArchiveRecord")
        self._load()
        payload, identity = self._validate_artifact(artifact, record)
        if identity in self._records:
            raise ArchiveContractError(f"artifact {identity} is already indexed")
        self._validate_prior_parents(record)
        self._validate_archive_bindings(record)

        object_path = self.objects / f"{identity}.json"
        try:
            write_once_json(object_path, payload)
        except StoreContractError as error:
            raise ArchiveContractError(str(error)) from error

        record_payload = record.to_payload()
        envelope = {
            "record": record_payload,
            "record_sha256": fingerprint_payload(record_payload),
        }
        try:
            append_jsonl(self.index, envelope)
        except StoreContractError as error:
            raise ArchiveContractError(str(error)) from error
        self._records[identity] = record
        return identity

    def lineage(self, artifact_sha256: str) -> tuple[str, ...]:
        identity = _archive_sha256(artifact_sha256, "artifact_sha256")
        self._load()
        return self._ordered_lineage(identity)

    def _ordered_lineage(self, identity: str) -> tuple[str, ...]:
        if identity not in self._records:
            raise ArchiveContractError(f"artifact {identity} is not indexed")
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(current: str) -> None:
            if current in seen:
                return
            for parent in self._records[current].parent_sha256s:
                visit(parent)
            seen.add(current)
            ordered.append(current)

        visit(identity)
        return tuple(ordered)

    def snapshot_sha256(self) -> str:
        self._load()
        try:
            data = self.index.read_bytes()
        except FileNotFoundError:
            data = b""
        return hashlib.sha256(data).hexdigest()

    def _load(self, *, verify_objects: bool = True) -> None:
        self._records = {}
        try:
            data = self.index.read_bytes()
        except FileNotFoundError:
            return
        except OSError as error:
            raise ArchiveContractError("cannot read archive index") from error
        if not data:
            return
        if not data.endswith(b"\n"):
            raise ArchiveContractError("archive index contains a truncated line")

        for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
            context = f"archive index line {line_number}"
            try:
                text = line.decode("utf-8")
                envelope = strict_json_loads(text, context=context)
            except (UnicodeError, ValueError) as error:
                raise ArchiveContractError(f"invalid {context}: {error}") from error
            if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_FIELDS:
                raise ArchiveContractError(
                    f"{context} envelope must use the exact schema"
                )
            try:
                canonical_envelope = canonical_v2_bytes(envelope)
            except (TypeError, ValueError) as error:
                raise ArchiveContractError(f"invalid {context}: {error}") from error
            if line != canonical_envelope:
                raise ArchiveContractError(f"{context} is not canonical JSON")

            record_payload = envelope["record"]
            if not isinstance(record_payload, Mapping):
                raise ArchiveContractError(f"{context} record must be an object")
            record = ArchiveRecord.from_payload(record_payload)
            claimed_record_sha256 = _archive_sha256(
                envelope["record_sha256"], f"{context}.record_sha256"
            )
            if claimed_record_sha256 != fingerprint_payload(record.to_payload()):
                raise ArchiveContractError(f"{context} record digest mismatch")
            identity = record.artifact_sha256
            if identity in self._records:
                raise ArchiveContractError(f"artifact {identity} is already indexed")
            self._validate_prior_parents(record)
            self._validate_archive_bindings(record)
            if verify_objects:
                self._verify_object(identity, record)
            self._records[identity] = record

    def _validate_artifact(
        self, artifact: EvolutionArtifact, record: ArchiveRecord
    ) -> tuple[dict[str, object], str]:
        try:
            payload_value = artifact.to_payload()
            artifact_bytes = artifact.canonical_bytes()
            claimed_identity = artifact.fingerprint()
            artifact_kind = artifact.artifact_kind
            schema_version = artifact.schema_version
        except (AttributeError, TypeError, ValueError) as error:
            raise ArchiveContractError(
                "artifact must implement the EvolutionArtifact protocol"
            ) from error
        if not isinstance(payload_value, Mapping):
            raise ArchiveContractError("artifact payload must be an object")
        payload = dict(payload_value)
        try:
            canonical = canonical_v2_bytes(payload)
        except (TypeError, ValueError) as error:
            raise ArchiveContractError(str(error)) from error
        if type(artifact_bytes) is not bytes or artifact_bytes != canonical:
            raise ArchiveContractError("artifact canonical bytes mismatch")
        identity = _archive_sha256(claimed_identity, "artifact SHA")
        if hashlib.sha256(canonical).hexdigest() != identity:
            raise ArchiveContractError("artifact SHA does not match canonical payload")
        if record.artifact_sha256 != identity:
            raise ArchiveContractError("record artifact SHA does not match artifact SHA")
        if record.artifact_kind != artifact_kind:
            raise ArchiveContractError("artifact kind mismatch")
        if record.schema_version != schema_version:
            raise ArchiveContractError("artifact schema_version mismatch")
        self._validate_payload_bindings(payload, record)
        return payload, identity

    def _validate_prior_parents(self, record: ArchiveRecord) -> None:
        for parent in record.parent_sha256s:
            if parent not in self._records:
                raise ArchiveContractError(
                    f"parent {parent} is not indexed before its child"
                )

    def _validate_archive_bindings(self, record: ArchiveRecord) -> None:
        if not self._records:
            return
        anchor = next(iter(self._records.values()))
        if record.protocol_fingerprint != anchor.protocol_fingerprint:
            raise ArchiveContractError("archive protocol fingerprint mismatch")
        if dict(record.runtime_fingerprints) != dict(anchor.runtime_fingerprints):
            raise ArchiveContractError("archive runtime fingerprints mismatch")

    def _verify_object(self, identity: str, record: ArchiveRecord) -> None:
        object_path = self.objects / f"{identity}.json"
        if not object_path.is_file():
            raise ArchiveContractError(f"archive object is missing: {identity}")
        try:
            data = object_path.read_bytes()
        except OSError as error:
            raise ArchiveContractError(f"cannot read archive object {identity}") from error
        if hashlib.sha256(data).hexdigest() != identity:
            raise ArchiveContractError(f"archive object digest mismatch: {identity}")
        try:
            parsed = strict_json_loads(
                data.decode("utf-8"), context=f"archive object {identity}"
            )
        except (UnicodeError, ValueError) as error:
            raise ArchiveContractError(f"invalid archive object {identity}: {error}") from error
        if not isinstance(parsed, dict):
            raise ArchiveContractError(f"archive object {identity} must be an object")
        try:
            canonical = canonical_v2_bytes(parsed)
        except (TypeError, ValueError) as error:
            raise ArchiveContractError(f"invalid archive object {identity}: {error}") from error
        if data != canonical:
            raise ArchiveContractError(f"archive object {identity} is not canonical JSON")
        if fingerprint_payload(parsed) != identity:
            raise ArchiveContractError(f"archive object digest mismatch: {identity}")
        self._validate_payload_bindings(parsed, record)

    @staticmethod
    def _validate_payload_bindings(
        payload: Mapping[str, object], record: ArchiveRecord
    ) -> None:
        if (
            "artifact_kind" in payload
            and payload["artifact_kind"] != record.artifact_kind
        ):
            raise ArchiveContractError("artifact/record artifact kind mismatch")
        if (
            "schema_version" in payload
            and payload["schema_version"] != record.schema_version
        ):
            raise ArchiveContractError("artifact/record schema_version mismatch")
        if (
            "protocol_fingerprint" in payload
            and payload["protocol_fingerprint"] != record.protocol_fingerprint
        ):
            raise ArchiveContractError("artifact/record protocol mismatch")
        if "runtime_fingerprints" in payload:
            runtime = payload["runtime_fingerprints"]
            if not isinstance(runtime, Mapping) or dict(runtime) != dict(
                record.runtime_fingerprints
            ):
                raise ArchiveContractError("artifact/record runtime mismatch")


def verify_selected_lineage(root: str | Path, artifact_sha256: str) -> Mapping[str, object]:
    """Read-only integrity recovery: verify the index and destination ancestry.

    A corrupt object outside the requested lineage does not prevent inspecting
    a verified prior state. No partially verified archive instance escapes this
    function; normal archive construction and mutation remain fully strict.
    """
    identity = _archive_sha256(artifact_sha256, "artifact_sha256")
    source = Path(root)
    if not (source / "index.jsonl").is_file():
        raise ArchiveContractError("selected lineage requires the archive index")
    reader = object.__new__(EvolutionArchive)
    reader.root = source
    reader.objects = source / "objects"
    reader.index = source / "index.jsonl"
    reader._load(verify_objects=False)
    lineage = reader._ordered_lineage(identity)
    payloads = {}
    for ancestor in lineage:
        reader._verify_object(ancestor, reader._records[ancestor])
        payloads[ancestor] = strict_json_loads(
            (reader.objects / f"{ancestor}.json").read_text(encoding="utf-8"),
            context=f"verified archive object {ancestor}",
        )
    return _freeze({
        "lineage": lineage,
        "payloads": payloads,
        "archive_snapshot_sha256": hashlib.sha256(reader.index.read_bytes()).hexdigest(),
    })


__all__ = ["ArchiveContractError", "ArchiveRecord", "EvolutionArchive", "verify_selected_lineage"]
