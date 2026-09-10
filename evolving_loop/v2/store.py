"""Crash-safe, version-isolated persistence for Evolution V2 runs."""
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from common.payload import strict_json_loads

from .contracts import canonical_v2_bytes, require_sha256


_LAYOUT = (
    "acceptance",
    "archive",
    "archive/objects",
    "candidates",
    "evaluations",
    "canary",
)
_MUTABLE_JSON_NAMES = frozenset({"checkpoint.json", "accepted_bundle.json"})
_SIMPLE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class StoreContractError(ValueError):
    """Raised when a V2 write would violate the persistence contract."""


def _fsync_directory(directory: Path) -> None:
    """Durably publish directory entries where the platform supports it."""
    if os.name == "nt":
        # Windows does not support opening directories for os.fsync. File data
        # is still flushed before os.replace; directory fsync is POSIX-only.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(directory: Path) -> None:
    """Create a directory chain and durably publish each new path entry."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    directory.mkdir(parents=True, exist_ok=True)
    fsynced: set[Path] = set()
    for created in missing:
        for target in (created, created.parent):
            if target not in fsynced:
                _fsync_directory(target)
                fsynced.add(target)
    for target in (directory, directory.parent):
        if target not in fsynced:
            _fsync_directory(target)
            fsynced.add(target)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return canonical_v2_bytes(payload)
    except (TypeError, ValueError) as error:
        raise StoreContractError(str(error)) from error


def _atomic_write(path: Path, data: bytes) -> None:
    _ensure_directory(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_once_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Persist canonical JSON once, permitting only byte-identical retries."""
    destination = Path(path)
    data = _canonical_bytes(payload)
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as error:
            raise StoreContractError(f"cannot read immutable file {destination}") from error
        if existing == data:
            _ensure_directory(destination.parent)
            return destination
        raise StoreContractError(f"immutable file already exists: {destination}")
    _atomic_write(destination, data)
    return destination


def write_atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Atomically replace one of the two explicitly mutable V2 JSON files."""
    destination = Path(path)
    if destination.name not in _MUTABLE_JSON_NAMES:
        raise StoreContractError(
            "atomic replacement supports only checkpoint.json and "
            "accepted_bundle.json"
        )
    _atomic_write(destination, _canonical_bytes(payload))
    return destination


def append_jsonl(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Append one canonical JSON object and make the append durable."""
    destination = Path(path)
    data = _canonical_bytes(payload)
    _ensure_directory(destination.parent)
    is_new = not destination.exists()
    with destination.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if is_new:
        _fsync_directory(destination.parent)
    return destination


def _sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except ValueError as error:
        raise StoreContractError(str(error)) from error


def _stage_name(stage: object) -> str:
    if (
        type(stage) is not str
        or stage in {".", ".."}
        or _SIMPLE_COMPONENT.fullmatch(stage) is None
    ):
        raise StoreContractError("stage must be a simple path component")
    return stage


class V2RunStore:
    """Exact directory and write-policy API for one isolated Evolution V2 run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, root: str | Path) -> "V2RunStore":
        destination = Path(root)
        if destination.exists() and not destination.is_dir():
            raise StoreContractError("V2 run root must be a directory")
        if destination.is_dir() and any(destination.iterdir()):
            cls._require_v2_manifest(destination)
        else:
            _ensure_directory(destination)
        for relative in _LAYOUT:
            _ensure_directory(destination / relative)
        return cls(destination)

    @staticmethod
    def _require_v2_manifest(root: Path) -> None:
        manifest_path = root / "run_manifest.json"
        try:
            raw = manifest_path.read_bytes()
            parsed = strict_json_loads(
                raw.decode("utf-8"),
                context=str(manifest_path),
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise StoreContractError(
                "refuse to adopt non-empty directory without an Evolution V2 manifest"
            ) from error
        if not isinstance(parsed, dict):
            raise StoreContractError(
                "refuse to adopt non-empty directory without an Evolution V2 manifest"
            )
        canonical = _canonical_bytes(parsed)
        if raw != canonical:
            raise StoreContractError("run manifest must use canonical Evolution V2 JSON")
        if parsed.get("system") != "evolution_v2":
            raise StoreContractError(
                "refuse to adopt non-empty directory without an Evolution V2 manifest"
            )

    def write_run_manifest(self, payload: Mapping[str, object]) -> Path:
        destination = self.root / "run_manifest.json"
        if destination.exists():
            return write_once_json(destination, payload)
        if not isinstance(payload, Mapping) or payload.get("system") != "evolution_v2":
            raise StoreContractError(
                "run manifest system must be exactly evolution_v2"
            )
        return write_once_json(destination, payload)

    def write_budget_plan(self, payload: Mapping[str, object]) -> Path:
        return write_once_json(self.root / "budget_plan.json", payload)

    def write_checkpoint(self, payload: Mapping[str, object]) -> Path:
        return write_atomic_json(self.root / "checkpoint.json", payload)

    def append_progress(self, payload: Mapping[str, object]) -> Path:
        return append_jsonl(self.root / "progress.jsonl", payload)

    def append_promotion(self, payload: Mapping[str, object]) -> Path:
        return append_jsonl(self.root / "promotion_history.jsonl", payload)

    def write_candidate(
        self, candidate_sha256: str, payload: Mapping[str, object]
    ) -> Path:
        identity = _sha256(candidate_sha256, "candidate_sha256")
        return write_once_json(self.root / "candidates" / f"{identity}.json", payload)

    def write_evaluation(
        self,
        candidate_sha256: str,
        stage: str,
        payload: Mapping[str, object],
    ) -> Path:
        identity = _sha256(candidate_sha256, "candidate_sha256")
        stage_name = _stage_name(stage)
        return write_once_json(
            self.root / "evaluations" / identity / f"{stage_name}.json",
            payload,
        )

    def write_acceptance(
        self, evidence_sha256: str, payload: Mapping[str, object]
    ) -> Path:
        identity = _sha256(evidence_sha256, "evidence_sha256")
        return write_once_json(self.root / "acceptance" / f"{identity}.json", payload)

    def publish_active_bundle(self, bundle_payload: Mapping[str, object]) -> Path:
        return write_atomic_json(self.root / "accepted_bundle.json", bundle_payload)

    def write_canary(
        self, source_sha256: str, payload: Mapping[str, object]
    ) -> Path:
        identity = _sha256(source_sha256, "source_sha256")
        return write_once_json(self.root / "canary" / f"{identity}.json", payload)

    def write_completion(self, payload: Mapping[str, object]) -> Path:
        return write_once_json(self.root / "completion.json", payload)


__all__ = [
    "StoreContractError",
    "V2RunStore",
    "append_jsonl",
    "write_atomic_json",
    "write_once_json",
]
