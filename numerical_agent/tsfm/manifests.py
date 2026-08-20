"""Validated, immutable bindings from foundation cards to reviewed runtimes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast


ManifestStatus = Literal["direct", "experimental_unverified", "unavailable"]
PointReduction = Literal["direct", "mean", "median", "none"]

_STATUSES = frozenset({"direct", "experimental_unverified", "unavailable"})
_REDUCTIONS = frozenset({"direct", "mean", "median", "none"})
_FIELDS = frozenset(
    {
        "method_id",
        "checkpoint",
        "worker_environment",
        "adapter",
        "license_id",
        "license_acknowledgement_required",
        "point_reduction",
        "status",
        "reason_code",
        "runtime_options",
        "official_source_ids",
    }
)


def _string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field_name} must be {qualifier}")
    return value


def _deep_freeze(value: object, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{field_name} keys must be strings")
        return MappingProxyType(
            {key: _deep_freeze(item, field_name) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item, field_name) for item in value)
    raise ValueError(f"{field_name} must contain only JSON-compatible values")


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _strict_json(payload: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"manifest JSON has duplicate key {key!r}")
            decoded[key] = value
        return decoded

    return json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


@dataclass(frozen=True)
class TSFMManifest:
    """One reviewed and immutable foundation-model runtime binding."""

    method_id: str
    checkpoint: str
    worker_environment: str
    adapter: str
    license_id: str
    license_acknowledgement_required: bool
    point_reduction: PointReduction
    status: ManifestStatus
    reason_code: str
    runtime_options: Mapping[str, object]
    official_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string(self.method_id, "method_id")
        _string(self.checkpoint, "checkpoint")
        _string(self.worker_environment, "worker_environment", allow_empty=True)
        _string(self.adapter, "adapter", allow_empty=True)
        _string(self.license_id, "license_id")
        _string(self.reason_code, "reason_code", allow_empty=True)
        if not isinstance(self.license_acknowledgement_required, bool):
            raise ValueError("license_acknowledgement_required must be a boolean")
        if self.point_reduction not in _REDUCTIONS:
            raise ValueError(f"unsupported point_reduction: {self.point_reduction!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported manifest status: {self.status!r}")
        if not isinstance(self.runtime_options, Mapping):
            raise ValueError("runtime_options must be an object")
        immutable_options = _deep_freeze(self.runtime_options, "runtime_options")
        assert isinstance(immutable_options, Mapping)
        object.__setattr__(self, "runtime_options", immutable_options)
        if (
            not isinstance(self.official_source_ids, tuple)
            or not self.official_source_ids
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in self.official_source_ids
            )
            or len(self.official_source_ids) != len(set(self.official_source_ids))
        ):
            raise ValueError("official_source_ids must contain unique non-empty strings")

        if self.status == "unavailable":
            if not self.reason_code:
                raise ValueError("unavailable manifest requires reason_code")
            if self.worker_environment:
                raise ValueError("unavailable manifest cannot select worker_environment")
            if self.adapter:
                raise ValueError("unavailable manifest cannot select adapter")
            if self.point_reduction != "none":
                raise ValueError("unavailable manifest point_reduction must be 'none'")
            if self.runtime_options:
                raise ValueError("unavailable manifest cannot have runtime_options")
            if self.license_acknowledgement_required:
                raise ValueError("unavailable manifest cannot require license acknowledgement")
            return

        if self.reason_code:
            raise ValueError("executable manifest cannot have reason_code")
        if not self.adapter:
            raise ValueError("executable manifest requires adapter")
        if self.point_reduction == "none":
            raise ValueError("executable manifest requires a point_reduction")
        if self.status == "experimental_unverified" and not self.worker_environment:
            raise ValueError("experimental manifest requires worker environment")
        if self.status == "direct" and self.worker_environment:
            raise ValueError("direct manifest cannot select worker_environment")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TSFMManifest":
        unknown = set(payload) - _FIELDS
        missing = _FIELDS - set(payload)
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing fields: {sorted(missing)!r}")
            if unknown:
                details.append(f"unknown fields: {sorted(unknown)!r}")
            raise ValueError("manifest " + "; ".join(details))
        options = payload["runtime_options"]
        if not isinstance(options, Mapping):
            raise ValueError("runtime_options must be an object")
        acknowledgement = payload["license_acknowledgement_required"]
        if not isinstance(acknowledgement, bool):
            raise ValueError("license_acknowledgement_required must be a boolean")
        source_ids = payload["official_source_ids"]
        if isinstance(source_ids, (str, bytes)) or not isinstance(source_ids, Sequence):
            raise ValueError("official_source_ids must be an array")
        return cls(
            method_id=_string(payload["method_id"], "method_id"),
            checkpoint=_string(payload["checkpoint"], "checkpoint"),
            worker_environment=_string(
                payload["worker_environment"], "worker_environment", allow_empty=True
            ),
            adapter=_string(payload["adapter"], "adapter", allow_empty=True),
            license_id=_string(payload["license_id"], "license_id"),
            license_acknowledgement_required=acknowledgement,
            point_reduction=cast(PointReduction, payload["point_reduction"]),
            status=cast(ManifestStatus, payload["status"]),
            reason_code=_string(payload["reason_code"], "reason_code", allow_empty=True),
            runtime_options=options,
            official_source_ids=tuple(source_ids),
        )

    def candidate_binding(self) -> dict[str, object]:
        """Return the complete, serializable runtime binding for this manifest."""

        return {
            "manifest_id": self.method_id,
            "checkpoint": self.checkpoint,
            "model_id": self.checkpoint,
            "worker_environment": self.worker_environment,
            "runtime_family": self.adapter,
            "runtime_options": _deep_thaw(self.runtime_options),
            "point_reduction": self.point_reduction,
            "license_id": self.license_id,
            "license_acknowledgement_required": (
                self.license_acknowledgement_required
            ),
        }

    def matches_candidate(self, candidate: object, *, provider: str) -> bool:
        """Reject candidates that substitute any reviewed runtime field."""

        implementation = getattr(candidate, "implementation", None)
        return (
            getattr(candidate, "method_id", None) == self.method_id
            and getattr(candidate, "provider", None) == provider
            and getattr(candidate, "implementation_kind", None) == "tsfm_checkpoint"
            and isinstance(implementation, Mapping)
            and all(
                implementation.get(key) == value
                for key, value in self.candidate_binding().items()
            )
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "checkpoint": self.checkpoint,
            "worker_environment": self.worker_environment,
            "adapter": self.adapter,
            "license_id": self.license_id,
            "license_acknowledgement_required": self.license_acknowledgement_required,
            "point_reduction": self.point_reduction,
            "status": self.status,
            "reason_code": self.reason_code,
            "runtime_options": _deep_thaw(self.runtime_options),
            "official_source_ids": list(self.official_source_ids),
        }


class ManifestRegistry(Mapping[str, TSFMManifest]):
    """Read-only index of reviewed manifests by exact catalog method ID."""

    def __init__(self, manifests: Mapping[str, TSFMManifest]) -> None:
        if any(key != manifest.method_id for key, manifest in manifests.items()):
            raise ValueError("manifest registry keys must match method IDs")
        self._manifests = MappingProxyType(dict(manifests))

    def __getitem__(self, method_id: str) -> TSFMManifest:
        return self._manifests[method_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._manifests)

    def __len__(self) -> int:
        return len(self._manifests)

    @classmethod
    def load(cls, path: str | Path) -> "ManifestRegistry":
        source = Path(path)
        try:
            payload = _strict_json(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load TSFM manifests from {source}: {error}") from error
        if not isinstance(payload, list):
            raise ValueError("TSFM manifest file must contain a list")
        manifests: dict[str, TSFMManifest] = {}
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest entry {index} must be an object")
            manifest = TSFMManifest.from_payload(item)
            if manifest.method_id in manifests:
                raise ValueError(f"duplicate TSFM manifest {manifest.method_id!r}")
            manifests[manifest.method_id] = manifest
        return cls(manifests)

    @classmethod
    def load_default(cls) -> "ManifestRegistry":
        registry = cls.load(Path(__file__).with_name("runtime_manifests.json"))
        registry._validate_default_provenance()
        return registry

    def _validate_default_provenance(self) -> None:
        from ..collection.registry import load_method_cards, load_source_records

        dataset_dir = Path(__file__).resolve().parents[1] / "datasets"
        sources = load_source_records(dataset_dir / "source_registry_v002.jsonl")
        cards = {
            card.method_uid: card
            for card in load_method_cards(dataset_dir / "method_candidates_v002.jsonl")
            if card.family == "foundation"
        }
        if set(self) != set(cards):
            raise ValueError("default manifests do not match the foundation catalog")
        known_source_ids = {source.source_id for source in sources}
        for method_id, manifest in self.items():
            card = cards[method_id]
            expected_sources = tuple(
                dict.fromkeys(card.definition_source_ids + card.implementation_source_ids)
            )
            if manifest.checkpoint != card.foundation_metadata["checkpoint_or_api"]:
                raise ValueError(f"manifest {method_id!r} checkpoint differs from catalog")
            if manifest.license_id != card.foundation_metadata["license"]:
                raise ValueError(f"manifest {method_id!r} license differs from catalog")
            if manifest.official_source_ids != expected_sources:
                raise ValueError(f"manifest {method_id!r} sources differ from catalog")
            if not set(manifest.official_source_ids) <= known_source_ids:
                raise ValueError(f"manifest {method_id!r} references an unknown source")

    def require(self, method_id: str) -> TSFMManifest:
        try:
            return self[method_id]
        except KeyError as error:
            raise ValueError(f"foundation method {method_id!r} has no reviewed manifest") from error
