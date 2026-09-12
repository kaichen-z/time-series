"""Immutable branching archive and deterministic source-policy proposals."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from common.payload import strict_json_loads

from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from ..store import StoreContractError, _ensure_directory, append_jsonl, write_once_json
from .contracts import ARM_ORDER, SourceRequestV2, SourceVariantV2
from .runtime import audit_source, run_policy


_ADD_FIELDS = ("schema_version", "kind", "source_sha256", "previous_event_sha256")
_CLOSE_FIELDS = _ADD_FIELDS + ("status", "train_gain", "task_cost")
_ENVELOPE_FIELDS = frozenset({"event", "event_sha256"})
_CLOSED_STATUSES = frozenset({"eligible", "terminal", "validated"})


class SourceArchiveError(ValueError):
    """Raised when source archive bytes or lifecycle transitions are invalid."""


@dataclass(frozen=True, slots=True)
class _ClosedSource:
    variant: SourceVariantV2
    status: str
    train_gain: float
    task_cost: int
    depth: int
    child_count: int


def _sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except ValueError as error:
        raise SourceArchiveError(str(error)) from error


def _exact_event(payload: object, fields: tuple[str, ...], context: str) -> dict[str, object]:
    if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
        raise SourceArchiveError(f"{context} event must be an object with the exact schema")
    if set(payload) != set(fields):
        raise SourceArchiveError(f"{context} event must use the exact schema")
    return dict(payload)


def _event_values(payload: object, context: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise SourceArchiveError(f"{context} event must be an object")
    kind = payload.get("kind")
    if kind == "add":
        values = _exact_event(payload, _ADD_FIELDS, context)
    elif kind == "close":
        values = _exact_event(payload, _CLOSE_FIELDS, context)
    else:
        raise SourceArchiveError(f"{context} event kind must be add or close")
    if type(values["schema_version"]) is not int or values["schema_version"] != 1:
        raise SourceArchiveError(f"{context} event schema_version must be exactly 1")
    _sha256(values["source_sha256"], f"{context}.source_sha256")
    previous = values["previous_event_sha256"]
    if previous is not None:
        _sha256(previous, f"{context}.previous_event_sha256")
    if kind == "close":
        if type(values["status"]) is not str or values["status"] not in _CLOSED_STATUSES:
            raise SourceArchiveError(f"{context} close status is invalid")
        gain = values["train_gain"]
        if type(gain) not in (int, float) or not math.isfinite(float(gain)):
            raise SourceArchiveError(f"{context} train_gain must be finite")
        cost = values["task_cost"]
        if type(cost) is not int or cost < 0:
            raise SourceArchiveError(f"{context} task_cost must be a non-negative integer")
    return values


def _probe_eligible(variant: SourceVariantV2) -> None:
    """Run the fixed no-credit probe required before a source becomes eligible."""
    audit_source(variant)
    request = SourceRequestV2(1, ARM_ORDER, 0, 0, {arm: 0.0 for arm in ARM_ORDER})
    try:
        run_policy(variant, request)
    except ValueError as error:
        raise SourceArchiveError("eligible source failed the policy probe") from error


class SourceArchiveV2:
    """Content-addressed source variants plus an append-only lifecycle log."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise SourceArchiveError("source archive root must be a directory")
        self.objects = self.root / "objects"
        self.events = self.root / "events.jsonl"
        _ensure_directory(self.objects)
        self._variants: dict[str, SourceVariantV2] = {}
        self._closed: dict[str, _ClosedSource] = {}
        self._last_event_sha256: str | None = None
        self._load()

    def add(self, variant: SourceVariantV2) -> str:
        """Persist a canonical source object before it receives any terminal state."""
        if not isinstance(variant, SourceVariantV2):
            raise TypeError("variant must be a SourceVariantV2")
        self._load()
        identity = variant.fingerprint()
        if identity in self._variants:
            raise SourceArchiveError(f"source {identity} is already archived")
        parent = variant.parent_source_sha256
        if parent is not None and parent not in self._variants:
            raise SourceArchiveError(f"parent source {parent} is not archived")
        try:
            write_once_json(self.objects / f"{identity}.json", variant.to_payload())
        except StoreContractError as error:
            raise SourceArchiveError(str(error)) from error
        self._append("add", identity)
        self._variants[identity] = variant
        return identity

    def close(
        self, source_sha256: str, *, status: str, train_gain: float = 0.0, task_cost: int = 0
    ) -> None:
        """Close a recorded source with Train-only ranking data or a terminal outcome."""
        identity = _sha256(source_sha256, "source_sha256")
        self._load()
        if identity not in self._variants:
            raise SourceArchiveError(f"source {identity} is not archived")
        if identity in self._closed:
            raise SourceArchiveError(f"source {identity} is already closed")
        event = {
            "status": status,
            "train_gain": train_gain,
            "task_cost": task_cost,
        }
        _event_values(
            {
                "schema_version": 1,
                "kind": "close",
                "source_sha256": identity,
                "previous_event_sha256": self._last_event_sha256,
                **event,
            },
            "close",
        )
        if status == "eligible":
            _probe_eligible(self._variants[identity])
        self._append("close", identity, **event)
        self._rebuild_closed()

    def sample(self, draw_counter: int) -> SourceVariantV2:
        """Cycle deterministically through eligible branches without Dev-derived rewards."""
        if type(draw_counter) is not int or draw_counter < 0:
            raise ValueError("draw_counter must be a non-negative integer")
        self._load()
        eligible = [entry for entry in self._closed.values() if entry.status == "eligible"]
        if not eligible:
            raise SourceArchiveError("no eligible source is available for sampling")
        ordered = _ordered_eligible(eligible)
        return ordered[draw_counter % len(ordered)].variant

    def snapshot_sha256(self) -> str:
        self._load()
        try:
            contents = self.events.read_bytes()
        except FileNotFoundError:
            contents = b""
        return hashlib.sha256(contents).hexdigest()

    def _append(self, kind: str, identity: str, **extra: object) -> None:
        event = {
            "schema_version": 1,
            "kind": kind,
            "source_sha256": identity,
            "previous_event_sha256": self._last_event_sha256,
            **extra,
        }
        values = _event_values(event, "new")
        envelope = {"event": values, "event_sha256": fingerprint_payload(values)}
        try:
            append_jsonl(self.events, envelope)
        except StoreContractError as error:
            raise SourceArchiveError(str(error)) from error
        self._last_event_sha256 = envelope["event_sha256"]

    def _load(self) -> None:
        self._variants = {}
        self._closed = {}
        self._last_event_sha256 = None
        try:
            contents = self.events.read_bytes()
        except FileNotFoundError:
            return
        except OSError as error:
            raise SourceArchiveError("cannot read source archive events") from error
        if not contents:
            return
        if not contents.endswith(b"\n"):
            raise SourceArchiveError("source archive events contain a truncated line")
        for number, line in enumerate(contents.splitlines(keepends=True), start=1):
            context = f"source archive event {number}"
            try:
                envelope = strict_json_loads(line.decode("utf-8"), context=context)
            except (UnicodeDecodeError, ValueError) as error:
                raise SourceArchiveError(f"invalid {context}: {error}") from error
            if not isinstance(envelope, Mapping) or set(envelope) != _ENVELOPE_FIELDS:
                raise SourceArchiveError(f"{context} envelope must use the exact schema")
            try:
                canonical = canonical_v2_bytes(envelope)
            except (TypeError, ValueError) as error:
                raise SourceArchiveError(f"invalid {context}: {error}") from error
            if line != canonical:
                raise SourceArchiveError(f"{context} is not canonical JSON")
            values = _event_values(envelope["event"], context)
            claimed = _sha256(envelope["event_sha256"], f"{context}.event_sha256")
            if claimed != fingerprint_payload(values):
                raise SourceArchiveError(f"{context} event digest mismatch")
            if values["previous_event_sha256"] != self._last_event_sha256:
                raise SourceArchiveError(f"{context} event prefix does not match")
            identity = values["source_sha256"]
            if values["kind"] == "add":
                if identity in self._variants:
                    raise SourceArchiveError(f"{context} adds a duplicate source")
                variant = self._read_object(identity)
                parent = variant.parent_source_sha256
                if parent is not None and parent not in self._variants:
                    raise SourceArchiveError(f"{context} parent source is not already archived")
                self._variants[identity] = variant
            else:
                if identity not in self._variants:
                    raise SourceArchiveError(f"{context} closes an unknown source")
                if identity in self._closed:
                    raise SourceArchiveError(f"{context} closes a source more than once")
                status = values["status"]
                assert isinstance(status, str)
                if status == "eligible":
                    _probe_eligible(self._variants[identity])
                self._closed[identity] = _ClosedSource(
                    self._variants[identity], status, float(values["train_gain"]), values["task_cost"], 0, 0
                )
            self._last_event_sha256 = claimed
        self._rebuild_closed()

    def _read_object(self, identity: str) -> SourceVariantV2:
        path = self.objects / f"{identity}.json"
        if not path.is_file():
            raise SourceArchiveError(f"source object is missing: {identity}")
        try:
            contents = path.read_bytes()
        except OSError as error:
            raise SourceArchiveError(f"cannot read source object {identity}") from error
        if hashlib.sha256(contents).hexdigest() != identity:
            raise SourceArchiveError(f"source object digest mismatch: {identity}")
        try:
            payload = strict_json_loads(contents.decode("utf-8"), context=f"source object {identity}")
        except (UnicodeDecodeError, ValueError) as error:
            raise SourceArchiveError(f"invalid source object {identity}: {error}") from error
        if not isinstance(payload, Mapping) or contents != canonical_v2_bytes(payload):
            raise SourceArchiveError(f"source object is not canonical JSON: {identity}")
        try:
            variant = SourceVariantV2.from_payload(payload)
        except ValueError as error:
            raise SourceArchiveError(f"invalid source object {identity}: {error}") from error
        if variant.fingerprint() != identity:
            raise SourceArchiveError(f"source object digest mismatch: {identity}")
        return variant

    def _rebuild_closed(self) -> None:
        children = {identity: 0 for identity in self._variants}
        depths: dict[str, int] = {}
        for identity, variant in self._variants.items():
            parent = variant.parent_source_sha256
            if parent is None:
                depths[identity] = 0
            else:
                children[parent] += 1
                depths[identity] = depths[parent] + 1
        self._closed = {
            identity: _ClosedSource(entry.variant, entry.status, entry.train_gain, entry.task_cost,
                                    depths[identity], children[identity])
            for identity, entry in self._closed.items()
        }


def _ordered_eligible(entries: list[_ClosedSource]) -> list[_ClosedSource]:
    """Sort exact Train key while keeping a new text ahead of duplicate text on ties."""
    ordered: list[_ClosedSource] = []
    quality_groups: dict[tuple[float, int], list[_ClosedSource]] = {}
    for entry in entries:
        quality_groups.setdefault((-entry.train_gain, entry.task_cost), []).append(entry)
    for quality in sorted(quality_groups):
        group = sorted(
            quality_groups[quality],
            key=lambda entry: (entry.child_count, entry.depth, entry.variant.fingerprint()),
        )
        seen_texts: set[str] = set()
        novel, repeated = [], []
        for entry in group:
            target = novel if entry.variant.source_text_sha256() not in seen_texts else repeated
            target.append(entry)
            seen_texts.add(entry.variant.source_text_sha256())
        ordered.extend(novel)
        ordered.extend(repeated)
    return ordered


_TEMPLATES = (
    (
        "enabled-index-zero",
        'def choose_arm(request):\n    return request["enabled_arms"][0]\n',
    ),
    (
        "enabled-last-index",
        'def choose_arm(request):\n    return request["enabled_arms"][-1]\n',
    ),
    (
        "compare-first-last-reward",
        'def choose_arm(request):\n'
        '    first = request["enabled_arms"][0]\n'
        '    last = request["enabled_arms"][-1]\n'
        '    if request["train_reward_by_arm"][first] >= request["train_reward_by_arm"][last]:\n'
        '        return first\n'
        '    return last\n',
    ),
)


def propose_sources(
    parent: SourceVariantV2, *, draw_counter: int, limit: int = 2
) -> tuple[SourceVariantV2, ...]:
    """Return deterministic, distinct, audited source-text descendants of ``parent``."""
    if not isinstance(parent, SourceVariantV2):
        raise TypeError("parent must be a SourceVariantV2")
    if type(draw_counter) is not int or draw_counter < 0:
        raise ValueError("draw_counter must be a non-negative integer")
    if type(limit) is not int or not 0 <= limit <= 2:
        raise ValueError("limit must be an integer from 0 through at most 2")
    if limit == 0:
        return ()
    result: list[SourceVariantV2] = []
    for offset in range(len(_TEMPLATES)):
        name, source = _TEMPLATES[(draw_counter + offset) % len(_TEMPLATES)]
        if source == parent.source:
            continue
        child = SourceVariantV2.child(parent, source, f"seed-template:{name}")
        audit_source(child)
        if all(child.source != prior.source for prior in result):
            result.append(child)
        if len(result) == limit:
            break
    return tuple(result)


__all__ = ["SourceArchiveError", "SourceArchiveV2", "propose_sources"]
