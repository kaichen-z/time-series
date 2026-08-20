"""Strict JSON-lines protocol shared by TSFM brokers and isolated workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from types import MappingProxyType
from typing import Mapping


PROTOCOL_VERSION = 1
_RESPONSE_STATUSES = frozenset(
    {"success", "unavailable", "invalid_request", "runtime_error"}
)
_IMMUTABLE_CHECKPOINT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _json_object(payload: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded_object: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded_object:
                raise ValueError(f"protocol payload has duplicate field {key!r}")
            decoded_object[key] = value
        return decoded_object

    try:
        decoded = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("protocol payload must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("protocol payload must be a JSON object")
    return decoded


def _require_exact_fields(
    payload: Mapping[str, object], required: set[str], *, kind: str
) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing:
        raise ValueError(f"{kind} is missing fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{kind} has unexpected fields: {', '.join(unexpected)}")


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def is_immutable_checkpoint_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and _IMMUTABLE_CHECKPOINT_REVISION.fullmatch(value) is not None
    )


def _checkpoint_revision(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("checkpoint_revision must be a string")
    if value and not is_immutable_checkpoint_revision(value):
        raise ValueError(
            "checkpoint_revision must be empty or a 40-64 character lowercase "
            "hexadecimal revision"
        )
    return value


def _finite_values(value: object, name: str, *, non_empty: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    if non_empty and not value:
        raise ValueError(f"{name} must not be empty")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain only finite numbers")
        try:
            number = float(item)
        except OverflowError as error:
            raise ValueError(f"{name} must contain only finite numbers") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        values.append(number)
    return tuple(values)


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    copied = dict(value)
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only JSON-compatible values") from error
    return copied


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    provider: str
    checkpoint: str
    history: tuple[float, ...]
    horizon: int
    frequency: str
    runtime_options: Mapping[str, object] = field(default_factory=dict)
    checkpoint_revision: str = ""
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {self.protocol_version!r}")
        object.__setattr__(self, "request_id", _required_string(self.request_id, "request_id"))
        object.__setattr__(self, "provider", _required_string(self.provider, "provider"))
        object.__setattr__(self, "checkpoint", _required_string(self.checkpoint, "checkpoint"))
        object.__setattr__(
            self, "history", _finite_values(self.history, "history", non_empty=True)
        )
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon <= 0
        ):
            raise ValueError("horizon must be a positive integer")
        object.__setattr__(self, "frequency", _required_string(self.frequency, "frequency"))
        object.__setattr__(
            self,
            "checkpoint_revision",
            _checkpoint_revision(self.checkpoint_revision),
        )
        object.__setattr__(
            self,
            "runtime_options",
            MappingProxyType(_json_mapping(self.runtime_options, "runtime_options")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "provider": self.provider,
            "checkpoint": self.checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "history": list(self.history),
            "horizon": self.horizon,
            "frequency": self.frequency,
            "runtime_options": dict(self.runtime_options),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorkerRequest":
        required = {
            "protocol_version",
            "request_id",
            "provider",
            "checkpoint",
            "history",
            "horizon",
            "frequency",
            "runtime_options",
        }
        if "checkpoint_revision" in payload:
            required.add("checkpoint_revision")
        _require_exact_fields(
            payload,
            required,
            kind="worker request",
        )
        return cls(
            protocol_version=payload["protocol_version"],  # type: ignore[arg-type]
            request_id=payload["request_id"],  # type: ignore[arg-type]
            provider=payload["provider"],  # type: ignore[arg-type]
            checkpoint=payload["checkpoint"],  # type: ignore[arg-type]
            checkpoint_revision=payload.get("checkpoint_revision", ""),  # type: ignore[arg-type]
            history=payload["history"],  # type: ignore[arg-type]
            horizon=payload["horizon"],  # type: ignore[arg-type]
            frequency=payload["frequency"],  # type: ignore[arg-type]
            runtime_options=payload["runtime_options"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> "WorkerRequest":
        return cls.from_payload(_json_object(payload))


@dataclass(frozen=True)
class WorkerResponse:
    request_id: str
    status: str
    values: tuple[float, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    reason_code: str = ""
    message: str = ""
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {self.protocol_version!r}")
        object.__setattr__(self, "request_id", _required_string(self.request_id, "request_id"))
        if self.status not in _RESPONSE_STATUSES:
            raise ValueError(f"unsupported worker response status {self.status!r}")
        object.__setattr__(
            self,
            "values",
            _finite_values(self.values, "values", non_empty=self.status == "success"),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_json_mapping(self.metadata, "metadata"))
        )
        if self.status == "success":
            if self.reason_code or self.message:
                raise ValueError("successful response cannot contain failure details")
        else:
            if self.values or self.metadata:
                raise ValueError("failed response cannot contain success fields")
            object.__setattr__(
                self, "reason_code", _required_string(self.reason_code, "reason_code")
            )
            object.__setattr__(self, "message", _required_string(self.message, "message"))

    @classmethod
    def success(
        cls,
        request_id: str,
        values: tuple[float, ...] | list[float],
        metadata: Mapping[str, object] | None = None,
    ) -> "WorkerResponse":
        return cls(
            request_id=request_id,
            status="success",
            values=tuple(values),
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls, request_id: str, status: str, reason_code: str, message: str
    ) -> "WorkerResponse":
        return cls(
            request_id=request_id,
            status=status,
            reason_code=reason_code,
            message=message,
        )

    def to_payload(self) -> dict[str, object]:
        common: dict[str, object] = {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "status": self.status,
        }
        if self.status == "success":
            common.update(values=list(self.values), metadata=dict(self.metadata))
        else:
            common.update(reason_code=self.reason_code, message=self.message)
        return common

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorkerResponse":
        status = payload.get("status")
        if status == "success":
            _require_exact_fields(
                payload,
                {"protocol_version", "request_id", "status", "values", "metadata"},
                kind="worker response",
            )
            return cls(
                protocol_version=payload["protocol_version"],  # type: ignore[arg-type]
                request_id=payload["request_id"],  # type: ignore[arg-type]
                status=status,
                values=payload["values"],  # type: ignore[arg-type]
                metadata=payload["metadata"],  # type: ignore[arg-type]
            )
        _require_exact_fields(
            payload,
            {"protocol_version", "request_id", "status", "reason_code", "message"},
            kind="worker response",
        )
        return cls(
            protocol_version=payload["protocol_version"],  # type: ignore[arg-type]
            request_id=payload["request_id"],  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            reason_code=payload["reason_code"],  # type: ignore[arg-type]
            message=payload["message"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> "WorkerResponse":
        return cls.from_payload(_json_object(payload))
