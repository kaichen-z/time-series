"""Versioned, declarative retrieval skills and their auditable lifecycle."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import stat
import weakref
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping


SKILL_LIBRARY_SCHEMA_VERSION = 1
SKILL_CHECKPOINT_WITNESS_SCHEMA_VERSION = 1
SkillStage = Literal["round1", "round2", "both"]
SkillStatus = Literal["candidate", "accepted", "specialized", "quarantined"]
SkillOperationKind = Literal["add", "repair", "specialize", "merge", "quarantine"]


class RetrievalSkillError(ValueError):
    """Raised when a skill row or lifecycle transition is unsafe or malformed."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not (cleaned := value.strip()):
        raise RetrievalSkillError(f"{field} must be a non-empty string")
    return cleaned


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RetrievalSkillError(f"{field} must be a list of strings")
    result = tuple(_text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise RetrievalSkillError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class RetrievalApplicability:
    """Typed constraints used to project a skill into a stage prompt."""

    assumption_kinds: tuple[str, ...] = ()
    gap_types: tuple[str, ...] = ()
    temporal_relations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("assumption_kinds", "gap_types", "temporal_relations"):
            object.__setattr__(self, field, _text_tuple(getattr(self, field), field))

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalApplicability":
        expected = {"assumption_kinds", "gap_types", "temporal_relations"}
        if set(raw) != expected:
            raise RetrievalSkillError("applicability must contain exactly its typed selectors")
        return cls(**{field: _text_tuple(raw[field], field) for field in expected})

    def to_payload(self) -> dict[str, list[str]]:
        return {
            "assumption_kinds": list(self.assumption_kinds),
            "gap_types": list(self.gap_types),
            "temporal_relations": list(self.temporal_relations),
        }

    def matches(
        self,
        *,
        assumption_kinds: Iterable[str] = (),
        gap_types: Iterable[str] = (),
        temporal_relations: Iterable[str] = (),
    ) -> bool:
        for required, available in (
            (self.assumption_kinds, tuple(assumption_kinds)),
            (self.gap_types, tuple(gap_types)),
            (self.temporal_relations, tuple(temporal_relations)),
        ):
            if required and (
                not available or not set(required).intersection(available)
            ):
                return False
        return True

    def is_narrower_than(self, parent: "RetrievalApplicability") -> bool:
        """Return whether every added constraint is no broader than ``parent``."""
        strict = False
        for child, previous in (
            (self.assumption_kinds, parent.assumption_kinds),
            (self.gap_types, parent.gap_types),
            (self.temporal_relations, parent.temporal_relations),
        ):
            if previous:
                if not child or not set(child).issubset(previous):
                    return False
                strict = strict or set(child) != set(previous)
            elif child:
                strict = True
        return strict


@dataclass(frozen=True, init=False)
class RetrievalSkill:
    """A versioned, host-interpreted retrieval strategy with no executable content.

    The custom initializer accepts the historical flat constructor as a migration
    boundary.  New callers must use the typed keyword fields documented below.
    """

    skill_id: str
    version: int
    parent_version: int | None
    stage: SkillStage
    status: SkillStatus
    name: str
    description: str
    applicability: RetrievalApplicability
    query_steps: tuple[str, ...]
    required_chain_fields: tuple[str, ...]
    counterevidence_rule: str
    failure_conditions: tuple[str, ...]
    validated_task_ids: tuple[str, ...]
    validated_entities: tuple[str, ...]
    validation_smae_gain: float | None
    validation_srmse_gain: float | None
    merged_from_skill_ids: tuple[str, ...]
    quarantine_reason: str | None

    _LEGACY_FIELDS = (
        "name",
        "description",
        "applicability",
        "query_strategy",
        "verification_rule",
        "created_from_task",
        "validation_score",
    )

    def __init__(self, skill_id: str, *args: object, **values: object) -> None:
        legacy_mode = bool(args and not isinstance(args[0], int)) or any(
            field in values
            for field in (
                "query_strategy",
                "verification_rule",
                "created_from_task",
                "validation_smae",
                "validation_srmse",
                "validation_score",
                "uses",
                "avg_smae",
                "avg_srmse",
                "avg_score",
            )
        ) or isinstance(values.get("applicability"), str)
        if legacy_mode:
            values = self._legacy_values(skill_id, args, values)
        elif args:
            field_names = (
                "version", "parent_version", "stage", "status", "name", "description",
                "applicability", "query_steps", "required_chain_fields",
                "counterevidence_rule", "failure_conditions", "validated_task_ids",
                "validated_entities", "validation_smae_gain", "validation_srmse_gain",
            )
            if len(args) > len(field_names):
                raise TypeError("too many positional RetrievalSkill arguments")
            values = {**dict(zip(field_names, args)), **values}
        self._set_typed(skill_id, values, allow_active=False)

    @classmethod
    def _legacy_values(
        cls,
        skill_id: str,
        args: tuple[object, ...],
        supplied: Mapping[str, object],
        *,
        historical_migration: bool = False,
    ) -> dict[str, object]:
        if len(args) > len(cls._LEGACY_FIELDS):
            raise TypeError("too many legacy RetrievalSkill arguments")
        raw = {**dict(zip(cls._LEGACY_FIELDS, args)), **dict(supplied)}
        applicability = raw.get("applicability", "")
        query_strategy = raw.get("query_strategy", "")
        verification_rule = raw.get("verification_rule", "")
        created_from_task = raw.get("created_from_task", "")
        smae = raw.get("validation_smae", raw.get("validation_smae_gain"))
        srmse = raw.get("validation_srmse", raw.get("validation_srmse_gain"))
        historically_validated = (
            historical_migration
            and raw.get("validation_smae") is not None
            and raw.get("validation_srmse") is not None
        )
        if not historically_validated:
            smae = None
            srmse = None
        return {
            "version": raw.get("version", 1),
            "parent_version": raw.get("parent_version"),
            "stage": raw.get("stage", "both"),
            "status": "accepted" if historically_validated else raw.get("status", "candidate"),
            "name": raw.get("name", skill_id),
            "description": raw.get("description", "legacy retrieval skill"),
            "applicability": RetrievalApplicability(),
            "query_steps": (query_strategy,) if isinstance(query_strategy, str) and query_strategy.strip() else (),
            "required_chain_fields": (),
            "counterevidence_rule": verification_rule if isinstance(verification_rule, str) and verification_rule.strip() else "Search for counterevidence.",
            "failure_conditions": (),
            "validated_task_ids": (created_from_task,) if isinstance(created_from_task, str) and created_from_task.strip() else (),
            "validated_entities": (),
            "validation_smae_gain": smae,
            "validation_srmse_gain": srmse,
            "merged_from_skill_ids": raw.get("merged_from_skill_ids", ()),
            "quarantine_reason": raw.get("quarantine_reason"),
        }

    def _set_typed(
        self,
        skill_id: object,
        values: Mapping[str, object],
        *,
        allow_active: bool,
    ) -> None:
        allowed = {
            "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions", "validated_task_ids", "validated_entities",
            "validation_smae_gain", "validation_srmse_gain", "merged_from_skill_ids",
            "quarantine_reason",
        }
        unknown = set(values).difference(allowed)
        missing = {
            "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions",
        }.difference(values)
        if unknown or missing:
            raise RetrievalSkillError(
                f"invalid typed RetrievalSkill fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        typed_id = _text(skill_id, "skill_id")
        if not all(character.isalnum() or character in "_-" for character in typed_id):
            raise RetrievalSkillError("skill_id must be a stable identifier")
        version = values["version"]
        parent_version = values["parent_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise RetrievalSkillError("version must be a positive integer")
        if parent_version is not None and (
            isinstance(parent_version, bool)
            or not isinstance(parent_version, int)
            or not 1 <= parent_version < version
        ):
            raise RetrievalSkillError("parent_version must name an earlier version")
        stage = values["stage"]
        status = values["status"]
        if stage not in {"round1", "round2", "both"}:
            raise RetrievalSkillError("invalid skill stage")
        if status not in {"candidate", "accepted", "specialized", "quarantined"}:
            raise RetrievalSkillError("invalid skill status")
        if status != "candidate" and not allow_active:
            raise RetrievalSkillError(
                "public RetrievalSkill construction is candidate-only; active and quarantine statuses require trusted transitions"
            )
        applicability = values["applicability"]
        if isinstance(applicability, Mapping):
            applicability = RetrievalApplicability.from_payload(applicability)
        if not isinstance(applicability, RetrievalApplicability):
            raise RetrievalSkillError("applicability must be a RetrievalApplicability")
        smae = self._metric(values.get("validation_smae_gain"), "validation_smae_gain")
        srmse = self._metric(values.get("validation_srmse_gain"), "validation_srmse_gain")
        if (smae is None) != (srmse is None):
            raise RetrievalSkillError("validation gains must be supplied together")
        if status in {"accepted", "specialized"} and (smae is None or srmse is None):
            raise RetrievalSkillError("active skills require both validation gains")
        reason = values.get("quarantine_reason")
        if reason is not None:
            reason = _text(reason, "quarantine_reason")
        if status == "quarantined" and not reason:
            raise RetrievalSkillError("quarantined skills require a reason")
        if status != "quarantined" and reason is not None:
            raise RetrievalSkillError("only quarantined skills may carry a reason")
        typed = {
            "skill_id": typed_id,
            "version": version,
            "parent_version": parent_version,
            "stage": stage,
            "status": status,
            "name": _text(values["name"], "name"),
            "description": _text(values["description"], "description"),
            "applicability": applicability,
            "query_steps": _text_tuple(values["query_steps"], "query_steps"),
            "required_chain_fields": _text_tuple(values["required_chain_fields"], "required_chain_fields"),
            "counterevidence_rule": _text(values["counterevidence_rule"], "counterevidence_rule"),
            "failure_conditions": _text_tuple(values["failure_conditions"], "failure_conditions"),
            "validated_task_ids": _text_tuple(values.get("validated_task_ids", ()), "validated_task_ids"),
            "validated_entities": _text_tuple(values.get("validated_entities", ()), "validated_entities"),
            "validation_smae_gain": smae,
            "validation_srmse_gain": srmse,
            "merged_from_skill_ids": _text_tuple(values.get("merged_from_skill_ids", ()), "merged_from_skill_ids"),
            "quarantine_reason": reason,
        }
        for field, value in typed.items():
            object.__setattr__(self, field, value)

    @staticmethod
    def _metric(value: object, field: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RetrievalSkillError(f"{field} must be a finite number or null")
        return float(value)

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalSkill":
        skill_id, values = cls._payload_values(raw)
        return cls(skill_id, **values)

    @classmethod
    def _from_storage_payload(cls, raw: Mapping[str, object]) -> "RetrievalSkill":
        """Hydrate a record only after its containing artifact was verified."""
        skill_id, values = cls._payload_values(raw)
        record = object.__new__(cls)
        record._set_typed(skill_id, values, allow_active=True)
        return record

    @classmethod
    def _payload_values(
        cls, raw: Mapping[str, object]
    ) -> tuple[object, dict[str, object]]:
        required = {
            "skill_id", "version", "parent_version", "stage", "status", "name", "description",
            "applicability", "query_steps", "required_chain_fields", "counterevidence_rule",
            "failure_conditions",
        }
        optional = {
            "validated_task_ids", "validated_entities", "validation_smae_gain",
            "validation_srmse_gain", "merged_from_skill_ids", "quarantine_reason",
        }
        if not required.issubset(raw) or set(raw).difference(required | optional):
            raise RetrievalSkillError("typed RetrievalSkill rows must contain only schema fields")
        values = {
            "validated_task_ids": (),
            "validated_entities": (),
            "validation_smae_gain": None,
            "validation_srmse_gain": None,
            "merged_from_skill_ids": (),
            "quarantine_reason": None,
            **dict(raw),
        }
        skill_id = values.pop("skill_id")
        return skill_id, values

    @classmethod
    def _from_legacy_payload(
        cls,
        raw: Mapping[str, object],
        *,
        authorize_historical_metrics: bool = False,
    ) -> "RetrievalSkill":
        allowed = {
            "skill_id",
            "name",
            "description",
            "applicability",
            "query_strategy",
            "verification_rule",
            "created_from_task",
            "validation_smae",
            "validation_srmse",
            "validation_score",
            "uses",
            "avg_smae",
            "avg_srmse",
            "avg_score",
        }
        if set(raw).difference(allowed) or not {
            "skill_id",
            "name",
            "description",
            "applicability",
            "query_strategy",
            "verification_rule",
            "created_from_task",
        }.issubset(raw):
            raise RetrievalSkillError(
                "legacy migration accepts only historical flat Retrieval Skill rows"
            )
        if not isinstance(raw.get("applicability"), str):
            raise RetrievalSkillError(
                "legacy migration cannot hydrate typed or current-schema rows"
            )
        skill_id = raw["skill_id"]
        if not isinstance(skill_id, str):
            raise RetrievalSkillError("legacy skill_id must be a string")
        values = cls._legacy_values(
            skill_id,
            (),
            {key: value for key, value in raw.items() if key != "skill_id"},
            historical_migration=authorize_historical_metrics,
        )
        record = object.__new__(cls)
        record._set_typed(
            skill_id,
            values,
            allow_active=authorize_historical_metrics,
        )
        return record

    def to_payload(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "stage": self.stage,
            "status": self.status,
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability.to_payload(),
            "query_steps": list(self.query_steps),
            "required_chain_fields": list(self.required_chain_fields),
            "counterevidence_rule": self.counterevidence_rule,
            "failure_conditions": list(self.failure_conditions),
            "validated_task_ids": list(self.validated_task_ids),
            "validated_entities": list(self.validated_entities),
            "validation_smae_gain": self.validation_smae_gain,
            "validation_srmse_gain": self.validation_srmse_gain,
            "merged_from_skill_ids": list(self.merged_from_skill_ids),
            "quarantine_reason": self.quarantine_reason,
        }

    @property
    def is_active(self) -> bool:
        return self.status in {"accepted", "specialized"}


@dataclass(frozen=True)
class RetrievalSkillOperation:
    kind: SkillOperationKind
    skill_id: str | None = None
    skill: RetrievalSkill | None = None
    reason: str | None = None
    source_skill_ids: tuple[str, ...] = ()

    @classmethod
    def add(cls, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("add", skill=skill)

    @classmethod
    def repair(cls, skill_id: str, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("repair", skill_id=skill_id, skill=skill)

    @classmethod
    def specialize(cls, skill_id: str, skill: RetrievalSkill) -> "RetrievalSkillOperation":
        return cls("specialize", skill_id=skill_id, skill=skill)

    @classmethod
    def merge(
        cls, source_skill_ids: Iterable[str], skill: RetrievalSkill
    ) -> "RetrievalSkillOperation":
        return cls("merge", skill=skill, source_skill_ids=tuple(source_skill_ids))

    @classmethod
    def quarantine(cls, skill_id: str, reason: str) -> "RetrievalSkillOperation":
        return cls("quarantine", skill_id=skill_id, reason=reason)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_digest(skill: RetrievalSkill) -> str:
    return _canonical_digest(skill.to_payload())


def _safe_artifact_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if len(absolute.parts) > 1:
        root_alias = Path(absolute.anchor) / absolute.parts[1]
        try:
            if stat.S_ISLNK(os.lstat(root_alias).st_mode):
                absolute = root_alias.resolve(strict=True).joinpath(
                    *absolute.parts[2:]
                )
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError) as error:
            raise RetrievalSkillError(
                f"cannot canonicalize retrieval Skill system path: {path}"
            ) from error
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RetrievalSkillError(
                f"cannot inspect retrieval Skill checkpoint path: {path}"
            ) from error
        if stat.S_ISLNK(mode):
            raise RetrievalSkillError(
                f"retrieval Skill checkpoint path contains a symlink: {path}"
            )
    return absolute


def _safe_read_bytes(path: Path) -> bytes:
    try:
        safe, parent_descriptor = _open_artifact_parent(path, create=False)
    except FileNotFoundError as error:
        raise RetrievalSkillError(
            f"retrieval Skill checkpoint does not exist: {path}"
        ) from error
    try:
        _revalidate_artifact_parent(safe, parent_descriptor)
        encoded = _read_artifact_entry(parent_descriptor, safe.name)
        _revalidate_artifact_parent(safe, parent_descriptor)
        return encoded
    except RetrievalSkillError:
        raise
    except OSError as error:
        raise RetrievalSkillError(
            f"retrieval Skill checkpoint path changed: {path}"
        ) from error
    finally:
        os.close(parent_descriptor)


def _open_artifact_parent(path: Path, *, create: bool) -> tuple[Path, int]:
    safe = _safe_artifact_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(safe.anchor, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise RetrievalSkillError(
            "cannot open retrieval Skill checkpoint root"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RetrievalSkillError(
                "retrieval Skill checkpoint root is not a directory"
            )
        for component in safe.parent.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise RetrievalSkillError(
                    "retrieval Skill checkpoint parent is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return safe, descriptor
    except FileNotFoundError:
        os.close(descriptor)
        raise
    except RetrievalSkillError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise RetrievalSkillError(
            "retrieval Skill checkpoint parent path changed while opening"
        ) from error
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_artifact_parent(path: Path, descriptor: int) -> None:
    try:
        _safe, current_descriptor = _open_artifact_parent(path, create=False)
    except FileNotFoundError as error:
        raise RetrievalSkillError(
            "retrieval Skill checkpoint parent path changed"
        ) from error
    try:
        expected = os.fstat(descriptor)
        current = os.fstat(current_descriptor)
        if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
            raise RetrievalSkillError(
                "retrieval Skill checkpoint parent directory changed"
            )
    finally:
        os.close(current_descriptor)


def _read_artifact_entry(parent_descriptor: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RetrievalSkillError(
                "retrieval Skill checkpoint is not a regular file"
            )
        return handle.read()


def _artifact_entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise RetrievalSkillError(
            "retrieval Skill checkpoint path contains a symlink"
        )
    return True


def _safe_artifact_exists(path: Path) -> bool:
    try:
        safe, parent_descriptor = _open_artifact_parent(path, create=False)
    except FileNotFoundError:
        return False
    try:
        _revalidate_artifact_parent(safe, parent_descriptor)
        exists = _artifact_entry_exists(parent_descriptor, safe.name)
        _revalidate_artifact_parent(safe, parent_descriptor)
        return exists
    finally:
        os.close(parent_descriptor)


def _unique_temporary(
    parent_descriptor: int, target_name: str, encoded: bytes
) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(128):
        temporary = f".{target_name}.{os.urandom(16).hex()}.tmp"
        try:
            descriptor = os.open(
                temporary,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            break
        except FileExistsError:
            continue
    else:
        raise RetrievalSkillError(
            "cannot allocate a unique retrieval Skill checkpoint temporary"
        )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(temporary, dir_fd=parent_descriptor)
        except OSError:
            pass
        raise
    return temporary


def _open_artifact_directory_entry(
    parent_descriptor: int,
    name: str,
    *,
    create: bool,
) -> tuple[int, bool, tuple[int, int]]:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise RetrievalSkillError("invalid retrieval Skill artifact directory name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def open_directory(entry_name: str) -> tuple[int, tuple[int, int]]:
        descriptor: int | None = None
        try:
            descriptor = os.open(entry_name, flags, dir_fd=parent_descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RetrievalSkillError(
                    "retrieval Skill artifact parent is not a directory"
                )
            identity = (metadata.st_dev, metadata.st_ino)
            visible = os.stat(
                entry_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(visible.st_mode)
                or (visible.st_dev, visible.st_ino) != identity
            ):
                raise RetrievalSkillError(
                    "retrieval Skill artifact directory changed while opening"
                )
            return descriptor, identity
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            raise

    try:
        descriptor, identity = open_directory(name)
    except FileNotFoundError:
        if not create:
            raise
    except RetrievalSkillError:
        raise
    except OSError as error:
        raise RetrievalSkillError(
            "cannot open retrieval Skill artifact directory"
        ) from error
    else:
        return descriptor, False, identity

    unpublished: str | None = None
    creation_error: OSError | None = None
    for _attempt in range(128):
        candidate = f".{name}.{os.urandom(16).hex()}.unpublished"
        try:
            os.mkdir(candidate, 0o755, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            # mkdir(2) may have committed before an injected or filesystem
            # error surfaced.  The random, unpublished name is retained and
            # inspected through a descriptor; it is never removed by name.
            creation_error = error
        unpublished = candidate
        break
    if unpublished is None:
        raise RetrievalSkillError(
            "cannot allocate a unique retrieval Skill artifact directory"
        )

    try:
        descriptor, identity = open_directory(unpublished)
    except Exception as error:
        raise RetrievalSkillError(
            "cannot inspect newly created retrieval Skill artifact directory; "
            f"retained as {unpublished}"
        ) from (creation_error or error)

    try:
        try:
            _rename_artifact_entry_noreplace(
                parent_descriptor,
                unpublished,
                name,
            )
        except FileExistsError:
            os.close(descriptor)
            descriptor = -1
            try:
                existing_descriptor, existing_identity = open_directory(name)
            except Exception as error:
                raise RetrievalSkillError(
                    "retrieval Skill artifact directory appeared during "
                    f"publication; owned directory retained as {unpublished}"
                ) from error
            return existing_descriptor, False, existing_identity
        except Exception as error:
            visible = _artifact_entry_metadata(parent_descriptor, name)
            if (
                visible is not None
                and stat.S_ISDIR(visible.st_mode)
                and (visible.st_dev, visible.st_ino) == identity
            ):
                return descriptor, True, identity
            raise RetrievalSkillError(
                "retrieval Skill artifact directory publication failed; "
                f"owned directory retained as {unpublished}"
            ) from error

        try:
            visible = _artifact_entry_metadata(parent_descriptor, name)
        except OSError as error:
            raise RetrievalSkillError(
                "cannot inspect published retrieval Skill artifact directory; "
                "owned inode retained"
            ) from error
        if (
            visible is None
            or not stat.S_ISDIR(visible.st_mode)
            or (visible.st_dev, visible.st_ino) != identity
        ):
            raise RetrievalSkillError(
                "retrieval Skill artifact directory changed during publication"
            )
        return descriptor, True, identity
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _remove_owned_empty_artifact_directory(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int] | None,
    *,
    operation_created: bool = False,
    _search_displaced: bool = True,
) -> None:
    quarantine = _move_artifact_entry_to_quarantine(parent_descriptor, name)
    if quarantine is None:
        if identity is not None and _search_displaced:
            displaced = _find_displaced_artifact_entry(
                parent_descriptor,
                identity,
                exclude=frozenset({name}),
            )
            if displaced is not None:
                _remove_owned_empty_artifact_directory(
                    parent_descriptor,
                    displaced,
                    identity,
                    _search_displaced=False,
                )
        return

    try:
        metadata = os.stat(
            quarantine,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except Exception as inspection_error:
        _restore_quarantined_artifact_entry(
            parent_descriptor,
            quarantine,
            name,
            expected_identity=None,
        )
        raise RetrievalSkillError(
            "cannot inspect quarantined retrieval Skill artifact directory"
        ) from inspection_error

    quarantined_identity = (metadata.st_dev, metadata.st_ino)
    owned = stat.S_ISDIR(metadata.st_mode) and (
        quarantined_identity == identity
        or (identity is None and operation_created)
    )
    if not owned:
        restore_error: RetrievalSkillError | None = None
        try:
            _restore_quarantined_artifact_entry(
                parent_descriptor,
                quarantine,
                name,
                expected_identity=quarantined_identity,
            )
        except RetrievalSkillError as error:
            restore_error = error
        if identity is not None and _search_displaced:
            displaced = _find_displaced_artifact_entry(
                parent_descriptor,
                identity,
                exclude=frozenset({name}),
            )
            if displaced is not None:
                _remove_owned_empty_artifact_directory(
                    parent_descriptor,
                    displaced,
                    identity,
                    _search_displaced=False,
                )
        if restore_error is not None:
            raise restore_error
        return
    # POSIX has no portable way to remove a directory by an already-held
    # descriptor.  Deleting this name after the identity check would allow a
    # concurrent replacement to be removed.  Retain the unique quarantine
    # entry instead; it remains auditable and cannot shadow a live artifact.
    return


def _read_optional_artifact_entry(
    parent_descriptor: int, name: str
) -> bytes | None:
    try:
        return _read_artifact_entry(parent_descriptor, name)
    except FileNotFoundError:
        return None


def _read_optional_artifact_entry_snapshot(
    parent_descriptor: int, name: str
) -> tuple[tuple[int, int], bytes] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RetrievalSkillError(
                "retrieval Skill checkpoint is not a regular file"
            )
        return (metadata.st_dev, metadata.st_ino), handle.read()


def _artifact_entry_metadata(
    parent_descriptor: int, name: str
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _find_displaced_artifact_entry(
    parent_descriptor: int,
    identity: tuple[int, int],
    *,
    exclude: frozenset[str],
) -> str | None:
    matches: list[str] = []
    for candidate in os.listdir(parent_descriptor):
        if candidate in exclude:
            continue
        metadata = _artifact_entry_metadata(parent_descriptor, candidate)
        if metadata is not None and (
            metadata.st_dev,
            metadata.st_ino,
        ) == identity:
            matches.append(candidate)
    if len(matches) > 1:
        raise RetrievalSkillError(
            "retrieval Skill owned artifact was displaced ambiguously"
        )
    return matches[0] if matches else None


def _rename_artifact_entry_noreplace(
    parent_descriptor: int,
    source: str,
    destination: str,
) -> None:
    for value in (source, destination):
        if Path(value).name != value or value in {"", ".", ".."}:
            raise RetrievalSkillError(
                "invalid retrieval Skill artifact quarantine name"
            )

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000001,
        )
    else:
        raise RetrievalSkillError(
            "atomic no-replace artifact quarantine is unavailable"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), source)


def _move_artifact_entry_to_quarantine(
    parent_descriptor: int,
    name: str,
) -> str | None:
    for _attempt in range(128):
        quarantine = f".retrieval-quarantine-{os.urandom(16).hex()}"
        try:
            _rename_artifact_entry_noreplace(
                parent_descriptor,
                name,
                quarantine,
            )
        except FileExistsError:
            continue
        except Exception:
            quarantined = _artifact_entry_metadata(
                parent_descriptor,
                quarantine,
            )
            if quarantined is not None:
                return quarantine
            if _artifact_entry_metadata(parent_descriptor, name) is None:
                return None
            raise
        return quarantine
    raise RetrievalSkillError(
        "cannot allocate a unique retrieval Skill artifact quarantine"
    )


def _restore_quarantined_artifact_entry(
    parent_descriptor: int,
    quarantine: str,
    name: str,
    *,
    expected_identity: tuple[int, int] | None,
) -> None:
    try:
        _rename_artifact_entry_noreplace(
            parent_descriptor,
            quarantine,
            name,
        )
    except FileExistsError as error:
        raise RetrievalSkillError(
            "retrieval Skill artifact restore name is occupied; "
            f"entry retained in quarantine {quarantine}"
        ) from error
    except Exception as error:
        quarantined = _artifact_entry_metadata(parent_descriptor, quarantine)
        restored = _artifact_entry_metadata(parent_descriptor, name)
        if quarantined is None and (
            expected_identity is None
            or (
                restored is not None
                and (restored.st_dev, restored.st_ino) == expected_identity
            )
        ):
            return
        raise RetrievalSkillError(
            "retrieval Skill artifact restore failed; "
            f"entry retained in quarantine {quarantine}"
        ) from error


def _unlink_owned_artifact_entry(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int],
    encoded: bytes,
    *,
    _search_displaced: bool = True,
) -> None:
    quarantine = _move_artifact_entry_to_quarantine(parent_descriptor, name)
    if quarantine is None:
        if _search_displaced:
            displaced = _find_displaced_artifact_entry(
                parent_descriptor,
                identity,
                exclude=frozenset({name}),
            )
            if displaced is not None:
                _unlink_owned_artifact_entry(
                    parent_descriptor,
                    displaced,
                    identity,
                    encoded,
                    _search_displaced=False,
                )
        return

    try:
        current = _read_optional_artifact_entry_snapshot(
            parent_descriptor,
            quarantine,
        )
    except Exception as inspection_error:
        metadata = _artifact_entry_metadata(parent_descriptor, quarantine)
        _restore_quarantined_artifact_entry(
            parent_descriptor,
            quarantine,
            name,
            expected_identity=(
                None
                if metadata is None
                else (metadata.st_dev, metadata.st_ino)
            ),
        )
        raise RetrievalSkillError(
            "cannot inspect quarantined retrieval Skill artifact"
        ) from inspection_error

    if current is None:
        return
    if current != (identity, encoded):
        restore_error: RetrievalSkillError | None = None
        try:
            _restore_quarantined_artifact_entry(
                parent_descriptor,
                quarantine,
                name,
                expected_identity=current[0],
            )
        except RetrievalSkillError as error:
            restore_error = error
        if _search_displaced:
            displaced = _find_displaced_artifact_entry(
                parent_descriptor,
                identity,
                exclude=frozenset({name}),
            )
            if displaced is not None:
                _unlink_owned_artifact_entry(
                    parent_descriptor,
                    displaced,
                    identity,
                    encoded,
                    _search_displaced=False,
                )
        if restore_error is not None:
            raise restore_error
        return
    # A file cannot be portably unlinked through its held descriptor.  Keep the
    # verified inode under its random quarantine name rather than risk deleting
    # a replacement installed after this ownership check.
    return


def _replace_artifact_entry_bytes(
    parent_descriptor: int, name: str, encoded: bytes
) -> None:
    temporary = _unique_temporary(parent_descriptor, name, encoded)
    try:
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary = ""
        if _read_artifact_entry(parent_descriptor, name) != encoded:
            raise RetrievalSkillError(
                "retrieval Skill artifact rollback verification failed"
            )
    finally:
        if temporary:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def _file_digest(path: Path) -> str:
    return hashlib.sha256(_safe_read_bytes(path)).hexdigest()


def _checkpoint_path_digest(path: Path) -> str:
    return _canonical_digest(str(_safe_artifact_path(path)))


def _checkpoint_witness_path(path: Path, checkpoint_sha256: str) -> Path:
    directory = path.parent / f".{path.name}.provenance"
    return directory / f"{checkpoint_sha256}.json"


def _checkpoint_witness_bytes(
    path: Path,
    checkpoint_sha256: str,
    skills_payload: list[dict[str, object]],
    active_record_origins: Mapping[str, str],
) -> bytes:
    payload = {
        "schema_version": SKILL_CHECKPOINT_WITNESS_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path_sha256": _checkpoint_path_digest(path),
        "skills_sha256": _canonical_digest(skills_payload),
        "active_records": [
            {"sha256": digest, "origin": active_record_origins[digest]}
            for digest in sorted(active_record_origins)
        ],
    }
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _checkpoint_authority_key(path: Path, checkpoint_sha256: str) -> tuple[str, str]:
    return _checkpoint_path_digest(path), checkpoint_sha256


def _build_skill_authority_boundary():
    """Keep active authority outside serializable Skill and artifact state."""
    current_checkpoints: dict[str, tuple[str, int]] = {}
    authorized_libraries: weakref.WeakKeyDictionary[
        object, tuple[object, ...]
    ] = weakref.WeakKeyDictionary()
    outstanding: dict[int, tuple[object, str, object]] = {}

    def consume(capability: object | None, operation: str, target: object) -> None:
        entry = outstanding.pop(id(capability), None)
        if (
            capability is None
            or entry is None
            or entry[0] is not capability
            or entry[1:] != (operation, target)
        ):
            raise RetrievalSkillError(
                "active Retrieval Skill operation requires evaluator/operator authority"
            )

    def run(operation: str, target: object, action):
        capability = object()
        outstanding[id(capability)] = (capability, operation, target)
        try:
            return action(capability)
        finally:
            outstanding.pop(id(capability), None)

    def register_checkpoint(library: object, checkpoint_sha256: str | None) -> None:
        if checkpoint_sha256 is None:
            authorized_libraries[library] = ("detached", object())
            return
        path_sha256 = _checkpoint_path_digest(library.path)
        previous = current_checkpoints.get(path_sha256)
        epoch = 1 if previous is None else previous[1] + 1
        current_checkpoints[path_sha256] = (checkpoint_sha256, epoch)
        authorized_libraries[library] = (
            "checkpoint",
            path_sha256,
            checkpoint_sha256,
            epoch,
        )

    def require_library(library: object) -> None:
        if not any(skill.is_active for skill in library.all()):
            return
        authority = authorized_libraries.get(library)
        if authority is None:
            raise RetrievalSkillError(
                "active Retrieval Skills lack evaluator/operator runtime authority"
            )
        if authority[0] == "checkpoint":
            _, path_sha256, checkpoint_sha256, epoch = authority
            if current_checkpoints.get(path_sha256) != (checkpoint_sha256, epoch):
                raise RetrievalSkillError(
                    "active Retrieval Skill library is not bound to the current checkpoint epoch"
                )

    def cache_identity(library: object) -> str:
        """Return an opaque identity after revalidating live Skill authority."""
        require_library(library)
        authority = authorized_libraries.get(library)
        if authority is None:
            return _canonical_digest({"kind": "unbound"})
        if authority[0] == "checkpoint":
            _, path_sha256, checkpoint_sha256, epoch = authority
            if current_checkpoints.get(path_sha256) != (
                checkpoint_sha256,
                epoch,
            ):
                raise RetrievalSkillError(
                    "Retrieval Skill cache identity is not bound to the current authority epoch"
                )
            payload: object = {
                "kind": "checkpoint",
                "path_sha256": path_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "epoch": epoch,
            }
        elif authority[0] == "release":
            payload = {
                "kind": "release",
                "path_sha256": authority[1],
                "checkpoint_sha256": authority[2],
            }
        else:
            payload = {"kind": authority[0], "grant": id(authority[1])}
        return _canonical_digest(payload)

    def commit_evaluator_records(
        library: object,
        proposed: Mapping[str, tuple[RetrievalSkill, ...]],
        active_record_origins: Mapping[str, str],
    ) -> None:
        target = (id(library), _checkpoint_path_digest(library.path))

        def commit(capability: object) -> None:
            file_sha256 = library._file_sha256
            if library.persist:
                file_sha256 = library._write(
                    proposed,
                    active_record_origins=active_record_origins,
                    _capability=capability,
                    _operation="evaluator_checkpoint_commit",
                    _target=target,
                )
            else:
                consume(capability, "evaluator_checkpoint_commit", target)
            library._skills = proposed
            library._active_record_origins = dict(active_record_origins)
            library._file_sha256 = file_sha256
            register_checkpoint(library, file_sha256 if library.persist else None)

        run("evaluator_checkpoint_commit", target, commit)

    def commit_authorized_update(
        library: object,
        proposed: Mapping[str, tuple[RetrievalSkill, ...]],
    ) -> None:
        require_library(library)
        target = (id(library), _checkpoint_path_digest(library.path))

        def commit(capability: object) -> None:
            file_sha256 = library._file_sha256
            if library.persist:
                file_sha256 = library._write(
                    proposed,
                    _capability=capability,
                    _operation="authorized_checkpoint_update",
                    _target=target,
                )
            else:
                consume(capability, "authorized_checkpoint_update", target)
            library._skills = dict(proposed)
            library._file_sha256 = file_sha256
            register_checkpoint(library, file_sha256 if library.persist else None)

        run("authorized_checkpoint_update", target, commit)

    def activate_checkpoint(
        library: object,
        checkpoint_sha256: str,
        *,
        operator: bool,
    ) -> object:
        key = _checkpoint_authority_key(library.path, checkpoint_sha256)
        path_sha256, requested_sha256 = key
        current = current_checkpoints.get(path_sha256)
        if not operator and (
            current is None or current[0] != requested_sha256
        ):
            raise RetrievalSkillError(
                "active Retrieval Skill checkpoint lacks authority for the current epoch"
            )
        operation = "operator_checkpoint_load" if operator else "authorized_checkpoint_load"

        def activate(capability: object) -> object:
            consume(capability, operation, key)
            if operator:
                previous = current_checkpoints.get(path_sha256)
                epoch = 1 if previous is None else previous[1] + 1
                current_checkpoints[path_sha256] = (requested_sha256, epoch)
            else:
                assert current is not None
                epoch = current[1]
            authorized_libraries[library] = (
                "checkpoint",
                path_sha256,
                requested_sha256,
                epoch,
            )
            return library

        return run(operation, key, activate)

    def load_checkpoint_for_operator(
        path: str | Path, *, persist: bool = True
    ) -> object:
        source = Path(path)
        if not _safe_artifact_exists(source):
            return RetrievalSkillLibrary(source, persist=persist)
        encoded, payloads = RetrievalSkillLibrary._read_current_checkpoint(source)
        if not any(
            record.get("status") in {"accepted", "specialized"}
            for record in payloads
        ):
            return RetrievalSkillLibrary.load(source, persist=persist)
        library, checkpoint_sha256 = RetrievalSkillLibrary._hydrate_verified_checkpoint(
            source,
            persist=persist,
            encoded=encoded,
            payloads=payloads,
        )
        return activate_checkpoint(library, checkpoint_sha256, operator=True)

    def migrate_legacy_for_operator(
        path: str | Path, *, persist: bool = True
    ) -> object:
        library = RetrievalSkillLibrary._hydrate_legacy(
            Path(path),
            persist=persist,
            authorize_historical_metrics=True,
        )
        if not any(skill.is_active for skill in library.all()):
            if persist:
                library.save()
            return library
        target = (id(library), _checkpoint_path_digest(library.path))

        def migrate(capability: object) -> object:
            file_sha256 = library._file_sha256
            if persist:
                file_sha256 = library._write(
                    library._skills,
                    _capability=capability,
                    _operation="legacy_operator_migration",
                    _target=target,
                )
            else:
                consume(capability, "legacy_operator_migration", target)
            library._file_sha256 = file_sha256
            register_checkpoint(library, file_sha256 if persist else None)
            return library

        return run("legacy_operator_migration", target, migrate)

    def activate_release_library(library: object) -> object:
        target = id(library)

        def activate(capability: object) -> object:
            consume(capability, "verified_release_load", target)
            authorized_libraries[library] = (
                "release",
                _checkpoint_path_digest(library.path),
                library._file_sha256 or _canonical_digest(library.all()),
            )
            return library

        return run("verified_release_load", target, activate)

    def inherit_library(source: object, target_library: object) -> None:
        if not any(skill.is_active for skill in target_library.all()):
            return
        try:
            require_library(source)
        except RetrievalSkillError as error:
            raise RetrievalSkillError(
                "copied Retrieval Skill libraries do not inherit active authority"
            ) from error
        authority = authorized_libraries.get(source)
        if authority is None:
            raise RetrievalSkillError(
                "copied Retrieval Skill libraries do not inherit active authority"
            )
        target = (id(source), id(target_library))

        def inherit(capability: object) -> None:
            consume(capability, "verified_library_replay", target)
            authorized_libraries[target_library] = authority

        run("verified_library_replay", target, inherit)

    return (
        consume,
        commit_evaluator_records,
        commit_authorized_update,
        activate_checkpoint,
        load_checkpoint_for_operator,
        migrate_legacy_for_operator,
        activate_release_library,
        require_library,
        cache_identity,
        inherit_library,
    )


(
    _consume_skill_capability,
    _commit_evaluator_records,
    _commit_authorized_library_update,
    _activate_checkpoint_library,
    _load_verified_checkpoint_for_operator,
    _migrate_legacy_for_operator,
    _activate_verified_release_library,
    _require_active_library_authority,
    _skill_library_cache_identity,
    _inherit_active_library_authority,
) = _build_skill_authority_boundary()
del _build_skill_authority_boundary


class RetrievalSkillLibrary:
    """An append-only skill history whose mutations commit atomically."""

    def __init__(
        self,
        path: str | Path,
        skills: Iterable[RetrievalSkill] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self.path = Path(path)
        self.persist = persist
        self._active_record_origins: dict[str, str] = {}
        self._file_sha256: str | None = None
        self._skills = self._validated_index(
            skills or (), active_record_hashes=self._active_record_origins
        )

    @classmethod
    def load(cls, path: str | Path, *, persist: bool = True) -> "RetrievalSkillLibrary":
        source = Path(path)
        if not _safe_artifact_exists(source):
            return cls(source, persist=persist)
        encoded, payloads = cls._read_current_checkpoint(source)
        if any(
            record.get("status") in {"accepted", "specialized"}
            for record in payloads
        ):
            raise RetrievalSkillError(
                "active Retrieval Skills require load_verified_checkpoint(), "
                "verified release, or explicit legacy migration provenance"
            )
        skills = tuple(
            RetrievalSkill._from_storage_payload(record) for record in payloads
        )
        library = object.__new__(cls)
        library.path = source
        library.persist = persist
        library._active_record_origins = {}
        library._file_sha256 = hashlib.sha256(encoded).hexdigest()
        library._skills = library._validated_index(skills)
        return library

    @staticmethod
    def _read_current_checkpoint(
        source: Path,
    ) -> tuple[bytes, tuple[Mapping[str, object], ...]]:
        try:
            encoded = _safe_read_bytes(source)
            raw = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalSkillError(f"invalid retrieval Skill checkpoint: {source}") from error
        if isinstance(raw, list):
            raise RetrievalSkillError(
                "legacy Retrieval Skill files require explicit migrate_legacy()"
            )
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "skills"}:
            raise RetrievalSkillError(
                "invalid retrieval Skill checkpoint schema or provenance"
            )
        if (
            raw["schema_version"] != SKILL_LIBRARY_SCHEMA_VERSION
            or not isinstance(raw["skills"], list)
        ):
            raise RetrievalSkillError("unsupported retrieval skill library schema")
        if any(not isinstance(record, dict) for record in raw["skills"]):
            raise RetrievalSkillError("skill rows must be objects")
        return encoded, tuple(raw["skills"])

    @classmethod
    def load_verified_checkpoint(
        cls, path: str | Path, *, persist: bool = True
    ) -> "RetrievalSkillLibrary":
        """Load a path-bound checkpoint written by migration or the evaluator.

        Ordinary current-schema files are data, not authority. Active records
        additionally require the immutable, content-addressed witness emitted
        during the atomic trusted commit.
        """
        source = Path(path)
        if not _safe_artifact_exists(source):
            return cls(source, persist=persist)
        encoded, payloads = cls._read_current_checkpoint(source)
        if not any(
            record.get("status") in {"accepted", "specialized"}
            for record in payloads
        ):
            return cls.load(source, persist=persist)
        library, checkpoint_sha256 = cls._hydrate_verified_checkpoint(
            source,
            persist=persist,
            encoded=encoded,
            payloads=payloads,
        )
        return _activate_checkpoint_library(
            library,
            checkpoint_sha256,
            operator=False,
        )

    @classmethod
    def _hydrate_verified_checkpoint(
        cls,
        source: Path,
        *,
        persist: bool,
        encoded: bytes | None = None,
        payloads: tuple[Mapping[str, object], ...] | None = None,
    ) -> tuple["RetrievalSkillLibrary", str]:
        if encoded is None or payloads is None:
            encoded, payloads = cls._read_current_checkpoint(source)
        checkpoint_sha256 = hashlib.sha256(encoded).hexdigest()
        witness_path = _checkpoint_witness_path(source, checkpoint_sha256)
        try:
            witness = json.loads(_safe_read_bytes(witness_path).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalSkillError(
                "active Retrieval Skill checkpoint has no valid immutable provenance witness"
            ) from error
        skills = tuple(
            RetrievalSkill._from_storage_payload(record) for record in payloads
        )
        active_origins = cls._validate_checkpoint_witness(
            witness,
            source=source,
            checkpoint_sha256=checkpoint_sha256,
            payloads=payloads,
            skills=skills,
        )
        library = object.__new__(cls)
        library.path = source
        library.persist = persist
        library._active_record_origins = dict(active_origins)
        library._file_sha256 = checkpoint_sha256
        library._skills = library._validated_index(
            skills, active_record_hashes=library._active_record_origins
        )
        return library, checkpoint_sha256

    @classmethod
    def migrate_legacy(
        cls, path: str | Path, *, persist: bool = True
    ) -> "RetrievalSkillLibrary":
        """Convert flat legacy rows to inactive candidates without granting authority."""
        return cls._hydrate_legacy(
            Path(path),
            persist=persist,
            authorize_historical_metrics=False,
        )

    @classmethod
    def _hydrate_legacy(
        cls,
        source: Path,
        *,
        persist: bool,
        authorize_historical_metrics: bool,
    ) -> "RetrievalSkillLibrary":
        try:
            encoded = _safe_read_bytes(source)
            raw = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalSkillError(f"invalid legacy Retrieval Skill file: {source}") from error
        if not isinstance(raw, list) or any(not isinstance(record, dict) for record in raw):
            raise RetrievalSkillError("legacy migration requires an array of flat rows")
        skills = tuple(
            RetrievalSkill._from_legacy_payload(
                record,
                authorize_historical_metrics=authorize_historical_metrics,
            )
            for record in raw
        )
        active_origins = {
            _record_digest(skill): "legacy_migration"
            for skill in skills
            if skill.is_active
        }
        library = object.__new__(cls)
        library.path = source
        library.persist = persist
        library._active_record_origins = dict(active_origins)
        library._file_sha256 = hashlib.sha256(encoded).hexdigest()
        library._skills = library._validated_index(
            skills, active_record_hashes=library._active_record_origins
        )
        return library

    @classmethod
    def from_release(cls, path: str | Path) -> "RetrievalSkillLibrary":
        """Hydrate the Skill history bound to a freshly verified immutable release."""
        from evolving_loop.retrieval_agent.policy import (
            RetrievalRelease,
        )

        return cls._from_loaded_release(RetrievalRelease.load(path))

    @classmethod
    def _from_loaded_release(cls, release: object) -> "RetrievalSkillLibrary":
        """Hydrate from one already verified, single-read release snapshot."""
        from evolving_loop.retrieval_agent.policy import (
            RetrievalRelease,
            _authorize_active_release_load,
        )

        if not isinstance(release, RetrievalRelease):
            raise RetrievalSkillError("invalid verified Retrieval release snapshot")
        payloads = tuple(release.skills)
        if any(not isinstance(record, Mapping) for record in payloads):
            raise RetrievalSkillError("release Skill rows must be objects")
        skills = tuple(
            RetrievalSkill._from_storage_payload(record) for record in payloads
        )
        indexed = cls._validated_index(
            skills,
            active_record_hashes=frozenset(
                _record_digest(skill) for skill in skills if skill.is_active
            ),
        )
        active_ids = {
            skill_id
            for skill_id, history in indexed.items()
            if history[-1].is_active
        }
        has_active_history = any(skill.is_active for skill in skills)
        if active_ids != set(release.genome.active_skill_ids):
            raise RetrievalSkillError(
                "release active_skill_ids do not match its active Skill history"
            )
        if has_active_history and release.manifest["state"] != "accepted":
            raise RetrievalSkillError("only an accepted release may activate Skills")
        if has_active_history:
            _authorize_active_release_load(release)
        active_origins = {
            _record_digest(skill): "verified_release"
            for skill in skills
            if skill.is_active
        }
        skills_path = release.path / "skills.json"
        library = object.__new__(cls)
        library.path = skills_path
        library.persist = False
        library._active_record_origins = dict(active_origins)
        library._file_sha256 = release.skills_file_sha256
        library._skills = indexed
        return (
            _activate_verified_release_library(library)
            if has_active_history
            else library
        )

    @classmethod
    def _validate_checkpoint_witness(
        cls,
        raw: object,
        *,
        source: Path,
        checkpoint_sha256: str,
        payloads: tuple[Mapping[str, object], ...],
        skills: tuple[RetrievalSkill, ...],
    ) -> dict[str, str]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "checkpoint_sha256",
            "checkpoint_path_sha256",
            "skills_sha256",
            "active_records",
        }:
            raise RetrievalSkillError("invalid Retrieval Skill checkpoint witness")
        if raw["schema_version"] != SKILL_CHECKPOINT_WITNESS_SCHEMA_VERSION:
            raise RetrievalSkillError("unsupported Retrieval Skill checkpoint witness")
        if raw["checkpoint_sha256"] != checkpoint_sha256:
            raise RetrievalSkillError("Retrieval Skill checkpoint witness hash mismatch")
        if raw["checkpoint_path_sha256"] != _checkpoint_path_digest(source):
            raise RetrievalSkillError("Retrieval Skill checkpoint witness path mismatch")
        if raw["skills_sha256"] != _canonical_digest(list(payloads)):
            raise RetrievalSkillError("Retrieval Skill checkpoint integrity hash mismatch")
        claimed = raw["active_records"]
        if not isinstance(claimed, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"sha256", "origin"}
            or not isinstance(item["sha256"], str)
            or item["origin"] not in {"evaluator_promotion", "legacy_migration"}
            for item in claimed
        ):
            raise RetrievalSkillError("invalid active-record provenance")
        origins = {str(item["sha256"]): str(item["origin"]) for item in claimed}
        if len(origins) != len(claimed) or claimed != [
            {"sha256": digest, "origin": origins[digest]}
            for digest in sorted(origins)
        ]:
            raise RetrievalSkillError("active-record provenance must be unique and sorted")
        actual_hashes = frozenset(
            _record_digest(skill) for skill in skills if skill.is_active
        )
        if actual_hashes != frozenset(origins):
            raise RetrievalSkillError("active Retrieval Skill provenance mismatch")
        for skill in skills:
            if skill.is_active:
                cls._validate_active_origin(skill, skills, origins[_record_digest(skill)])
        return origins

    @staticmethod
    def _validate_active_origin(
        skill: RetrievalSkill,
        skills: tuple[RetrievalSkill, ...],
        origin: str,
    ) -> None:
        if origin == "legacy_migration":
            if skill.status != "accepted" or skill.version != 1:
                raise RetrievalSkillError(
                    "legacy provenance may authorize only version-one accepted records"
                )
            return
        grouped = {
            item.version: item for item in skills if item.skill_id == skill.skill_id
        }
        parent = grouped.get(skill.parent_version or -1)
        tolerance = 1e-12
        if (
            skill.status != "accepted"
            or skill.version <= 1
            or parent is None
            or parent.status != "candidate"
            or len(set(skill.validated_task_ids)) < 3
            or len(set(skill.validated_entities)) < 2
            or skill.validation_smae_gain is None
            or skill.validation_srmse_gain is None
            or skill.validation_smae_gain < -tolerance
            or skill.validation_srmse_gain < -tolerance
            or not (
                skill.validation_smae_gain > tolerance
                or skill.validation_srmse_gain > tolerance
            )
        ):
            raise RetrievalSkillError(
                "evaluator checkpoint contains an active record without promotion gates"
            )

    @staticmethod
    def _validated_index(
        skills: Iterable[RetrievalSkill],
        *,
        active_record_hashes: Iterable[str] = (),
    ) -> dict[str, tuple[RetrievalSkill, ...]]:
        authorized = frozenset(active_record_hashes)
        grouped: dict[str, list[RetrievalSkill]] = {}
        for skill in skills:
            if not isinstance(skill, RetrievalSkill):
                raise RetrievalSkillError("library entries must be RetrievalSkill records")
            if skill.is_active and _record_digest(skill) not in authorized:
                raise RetrievalSkillError(
                    "active Retrieval Skills require verified release, migration, or evaluator provenance"
                )
            grouped.setdefault(skill.skill_id, []).append(skill)
        indexed: dict[str, tuple[RetrievalSkill, ...]] = {}
        for skill_id, history in grouped.items():
            ordered = tuple(sorted(history, key=lambda item: item.version))
            versions = tuple(item.version for item in ordered)
            if versions != tuple(range(1, len(ordered) + 1)):
                raise RetrievalSkillError(f"{skill_id} versions must be contiguous from one")
            for index, skill in enumerate(ordered):
                if index == 0:
                    if skill.parent_version is not None:
                        raise RetrievalSkillError(f"{skill_id} version one cannot have a parent")
                elif skill.parent_version != ordered[index - 1].version:
                    raise RetrievalSkillError(f"{skill_id} lineage must reference the previous version")
                if skill.name != ordered[0].name:
                    raise RetrievalSkillError(f"{skill_id} name must remain stable across versions")
            indexed[skill_id] = ordered
        return indexed

    def save(self) -> None:
        if getattr(self, "_read_only", False):
            raise RetrievalSkillError("read-only Retrieval Skill snapshots are immutable")
        if self.persist:
            if any(skill.is_active for skill in self.all()):
                _commit_authorized_library_update(self, self._skills)
            else:
                self._file_sha256 = self._write(self._skills)

    def _write(
        self,
        proposed: Mapping[str, tuple[RetrievalSkill, ...]],
        *,
        active_record_origins: Mapping[str, str] | None = None,
        _capability: object | None = None,
        _operation: str | None = None,
        _target: object | None = None,
    ) -> str:
        safe_path = _safe_artifact_path(self.path)
        origins = dict(
            self._active_record_origins
            if active_record_origins is None
            else active_record_origins
        )
        authorized = frozenset(origins)
        records = self._all_from(proposed)
        self._validated_index(records, active_record_hashes=authorized)
        skills_payload = [skill.to_payload() for skill in records]
        payload = {
            "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
            "skills": skills_payload,
        }
        active = frozenset(
            _record_digest(skill)
            for skill in records
            if skill.is_active
        )
        if active:
            if active != authorized or any(
                origin not in {"evaluator_promotion", "legacy_migration"}
                for origin in origins.values()
            ):
                raise RetrievalSkillError(
                    "active Retrieval Skills cannot be saved without verified provenance"
                )
            for skill in records:
                if skill.is_active:
                    self._validate_active_origin(
                        skill, records, origins[_record_digest(skill)]
                    )
            _consume_skill_capability(_capability, _operation or "", _target)
        encoded = (
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        checkpoint_sha256 = hashlib.sha256(encoded).hexdigest()
        safe_path, parent_descriptor = _open_artifact_parent(
            safe_path, create=True
        )
        temporary: str | None = None
        rollback_temporary: str | None = None
        previous_encoded: bytes | None = None
        witness_path: Path | None = None
        witness_encoded: bytes | None = None
        witness_directory_name: str | None = None
        witness_parent_descriptor: int | None = None
        witness_temporary: str | None = None
        witness_identity: tuple[int, int] | None = None
        witness_directory_created = False
        witness_directory_identity: tuple[int, int] | None = None
        previous_witness_name: str | None = None
        previous_witness_encoded: bytes | None = None
        checkpoint_identity: tuple[int, int] | None = None
        write_succeeded = False
        try:
            _revalidate_artifact_parent(safe_path, parent_descriptor)
            exists = _artifact_entry_exists(parent_descriptor, safe_path.name)
            if self._file_sha256 is not None:
                if not exists:
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint path was replaced or changed"
                    )
                previous_encoded = _read_artifact_entry(
                    parent_descriptor, safe_path.name
                )
                if (
                    hashlib.sha256(previous_encoded).hexdigest()
                    != self._file_sha256
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint path was replaced or changed"
                    )
            elif exists:
                raise RetrievalSkillError(
                    "retrieval Skill checkpoint path already exists without a loaded digest"
                )
            temporary = _unique_temporary(
                parent_descriptor, safe_path.name, encoded
            )
            checkpoint_snapshot = _read_optional_artifact_entry_snapshot(
                parent_descriptor, temporary
            )
            if (
                checkpoint_snapshot is None
                or checkpoint_snapshot[1] != encoded
            ):
                raise RetrievalSkillError(
                    "retrieval Skill checkpoint temporary changed"
                )
            checkpoint_identity = checkpoint_snapshot[0]
            if previous_encoded is not None:
                rollback_temporary = _unique_temporary(
                    parent_descriptor,
                    f"{safe_path.name}.rollback",
                    previous_encoded,
                )
            if active:
                witness_path = _checkpoint_witness_path(
                    safe_path, checkpoint_sha256
                )
                witness_encoded = _checkpoint_witness_bytes(
                    safe_path,
                    checkpoint_sha256,
                    skills_payload,
                    origins,
                )
                witness_directory_name = witness_path.parent.name
                expects_previous_witness = (
                    self._file_sha256 is not None
                    and _operation != "legacy_operator_migration"
                    and any(skill.is_active for skill in self.all())
                )
                if expects_previous_witness:
                    previous_witness_name = f"{self._file_sha256}.json"
                    previous_skills_payload = [
                        skill.to_payload() for skill in self.all()
                    ]
                    previous_witness_encoded = _checkpoint_witness_bytes(
                        safe_path,
                        self._file_sha256,
                        previous_skills_payload,
                        self._active_record_origins,
                    )
                (
                    witness_parent_descriptor,
                    witness_directory_created,
                    opened_witness_directory_identity,
                ) = _open_artifact_directory_entry(
                    parent_descriptor,
                    witness_directory_name,
                    create=True,
                )
                if witness_directory_created:
                    witness_directory_identity = opened_witness_directory_identity
                if expects_previous_witness:
                    _revalidate_artifact_parent(
                        witness_path, witness_parent_descriptor
                    )
                    if (
                        _read_optional_artifact_entry(
                            witness_parent_descriptor,
                            previous_witness_name,
                        )
                        != previous_witness_encoded
                    ):
                        raise RetrievalSkillError(
                            "previous Retrieval Skill checkpoint witness changed"
                        )
                    _revalidate_artifact_parent(
                        witness_path, witness_parent_descriptor
                    )
                witness_temporary = _unique_temporary(
                    witness_parent_descriptor,
                    witness_path.name,
                    witness_encoded,
                )
                temporary_snapshot = _read_optional_artifact_entry_snapshot(
                    witness_parent_descriptor,
                    witness_temporary,
                )
                if (
                    temporary_snapshot is None
                    or temporary_snapshot[1] != witness_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint witness temporary changed"
                    )
                proposed_witness_identity = temporary_snapshot[0]
                witness_before_link = _read_optional_artifact_entry_snapshot(
                    witness_parent_descriptor,
                    witness_path.name,
                )
                if witness_before_link is None:
                    witness_identity = proposed_witness_identity
                try:
                    _revalidate_artifact_parent(
                        witness_path, witness_parent_descriptor
                    )
                    try:
                        os.link(
                            witness_temporary,
                            witness_path.name,
                            src_dir_fd=witness_parent_descriptor,
                            dst_dir_fd=witness_parent_descriptor,
                            follow_symlinks=False,
                        )
                    except Exception as link_error:
                        linked_snapshot = _read_optional_artifact_entry_snapshot(
                            witness_parent_descriptor,
                            witness_path.name,
                        )
                        if (
                            witness_before_link is None
                            and linked_snapshot
                            == (proposed_witness_identity, witness_encoded)
                        ):
                            raise
                        if not isinstance(link_error, FileExistsError):
                            raise
                        if (
                            linked_snapshot is None
                            or linked_snapshot[1] != witness_encoded
                        ):
                            raise RetrievalSkillError(
                                "immutable Retrieval Skill checkpoint witness changed"
                            )
                finally:
                    if witness_temporary is not None:
                        try:
                            os.unlink(
                                witness_temporary,
                                dir_fd=witness_parent_descriptor,
                            )
                        except FileNotFoundError:
                            pass
                        witness_temporary = None
                _revalidate_artifact_parent(
                    witness_path, witness_parent_descriptor
                )
                if (
                    _read_artifact_entry(
                        witness_parent_descriptor, witness_path.name
                    )
                    != witness_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint witness publication failed"
                    )
                os.fsync(witness_parent_descriptor)
                _revalidate_artifact_parent(
                    witness_path, witness_parent_descriptor
                )
            _revalidate_artifact_parent(safe_path, parent_descriptor)
            if previous_encoded is None:
                try:
                    os.link(
                        temporary,
                        safe_path.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError as error:
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint appeared during no-replace commit"
                    ) from error
                os.unlink(temporary, dir_fd=parent_descriptor)
                temporary = None
            else:
                if (
                    _read_artifact_entry(parent_descriptor, safe_path.name)
                    != previous_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint changed before guarded replace"
                    )
                _revalidate_artifact_parent(safe_path, parent_descriptor)
                os.replace(
                    temporary,
                    safe_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                temporary = None
            _revalidate_artifact_parent(safe_path, parent_descriptor)
            if _read_artifact_entry(parent_descriptor, safe_path.name) != encoded:
                raise RetrievalSkillError(
                    "retrieval Skill checkpoint publication verification failed"
                )
            if (
                witness_path is not None
                and witness_encoded is not None
                and witness_parent_descriptor is not None
            ):
                _revalidate_artifact_parent(
                    witness_path, witness_parent_descriptor
                )
                if (
                    _read_artifact_entry(
                        witness_parent_descriptor, witness_path.name
                    )
                    != witness_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint witness changed during bundle commit"
                    )
                os.fsync(witness_parent_descriptor)
            os.fsync(parent_descriptor)
            _revalidate_artifact_parent(safe_path, parent_descriptor)
            if _read_artifact_entry(parent_descriptor, safe_path.name) != encoded:
                raise RetrievalSkillError(
                    "retrieval Skill checkpoint changed during bundle commit"
                )
            if (
                witness_path is not None
                and witness_encoded is not None
                and witness_parent_descriptor is not None
            ):
                _revalidate_artifact_parent(
                    witness_path, witness_parent_descriptor
                )
                if (
                    _read_artifact_entry(
                        witness_parent_descriptor, witness_path.name
                    )
                    != witness_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint witness changed during bundle commit"
                    )
                self._hydrate_verified_checkpoint(
                    safe_path,
                    persist=self.persist,
                    encoded=encoded,
                    payloads=tuple(skills_payload),
                )
                _revalidate_artifact_parent(
                    witness_path, witness_parent_descriptor
                )
                _revalidate_artifact_parent(safe_path, parent_descriptor)
            write_succeeded = True
        except Exception:
            try:
                if previous_encoded is None:
                    if checkpoint_identity is None:
                        raise RetrievalSkillError(
                            "first Retrieval Skill checkpoint ownership is unknown"
                        )
                    _unlink_owned_artifact_entry(
                        parent_descriptor,
                        safe_path.name,
                        checkpoint_identity,
                        encoded,
                    )
                    current_metadata = _artifact_entry_metadata(
                        parent_descriptor, safe_path.name
                    )
                    if current_metadata is not None and (
                        current_metadata.st_dev,
                        current_metadata.st_ino,
                    ) == checkpoint_identity:
                        raise RetrievalSkillError(
                            "first Retrieval Skill checkpoint publication could not roll back"
                        )
                elif _read_optional_artifact_entry(
                    parent_descriptor, safe_path.name
                ) != previous_encoded:
                    if rollback_temporary is None:
                        rollback_temporary = _unique_temporary(
                            parent_descriptor,
                            f"{safe_path.name}.rollback",
                            previous_encoded,
                        )
                    rollback_snapshot = _read_optional_artifact_entry_snapshot(
                        parent_descriptor, rollback_temporary
                    )
                    if (
                        rollback_snapshot is None
                        or rollback_snapshot[1] != previous_encoded
                    ):
                        rollback_temporary = None
                        raise RetrievalSkillError(
                            "retrieval Skill rollback artifact identity is unavailable"
                        )
                    observed = _read_optional_artifact_entry_snapshot(
                        parent_descriptor, safe_path.name
                    )
                    if observed is None:
                        rollback_temporary = None
                        raise RetrievalSkillError(
                            "retrieval Skill checkpoint disappeared before rollback"
                        )
                    try:
                        rollback_quarantine = (
                            _move_artifact_entry_to_quarantine(
                                parent_descriptor, safe_path.name
                            )
                        )
                    except Exception:
                        rollback_temporary = None
                        raise
                    if rollback_quarantine is None:
                        rollback_temporary = None
                        raise RetrievalSkillError(
                            "retrieval Skill checkpoint rollback ownership is ambiguous"
                        )
                    moved = _read_optional_artifact_entry_snapshot(
                        parent_descriptor, rollback_quarantine
                    )
                    if moved != observed:
                        rollback_temporary = None
                        if moved is not None:
                            _restore_quarantined_artifact_entry(
                                parent_descriptor,
                                rollback_quarantine,
                                safe_path.name,
                                expected_identity=moved[0],
                            )
                        raise RetrievalSkillError(
                            "retrieval Skill checkpoint changed during rollback quarantine"
                        )
                    try:
                        _rename_artifact_entry_noreplace(
                            parent_descriptor,
                            rollback_temporary,
                            safe_path.name,
                        )
                    except Exception:
                        restored = _read_optional_artifact_entry_snapshot(
                            parent_descriptor, safe_path.name
                        )
                        if restored != rollback_snapshot:
                            rollback_temporary = None
                            raise RetrievalSkillError(
                                "retrieval Skill checkpoint rollback name is occupied; "
                                "rollback and quarantined entries were retained"
                            )
                    rollback_temporary = None
                if previous_encoded is not None and (
                    _read_optional_artifact_entry(parent_descriptor, safe_path.name)
                    != previous_encoded
                ):
                    raise RetrievalSkillError(
                        "retrieval Skill checkpoint rollback verification failed"
                    )

                if (
                    witness_identity is not None
                    and witness_path is not None
                    and witness_encoded is not None
                    and witness_parent_descriptor is not None
                    and witness_path.name != previous_witness_name
                ):
                    _unlink_owned_artifact_entry(
                        witness_parent_descriptor,
                        witness_path.name,
                        witness_identity,
                        witness_encoded,
                    )
                    os.fsync(witness_parent_descriptor)

                parent_is_current = True
                try:
                    _revalidate_artifact_parent(safe_path, parent_descriptor)
                except RetrievalSkillError:
                    parent_is_current = False
                if (
                    parent_is_current
                    and witness_directory_name is not None
                    and (
                        previous_witness_encoded is not None
                        or _artifact_entry_exists(
                            parent_descriptor, witness_directory_name
                        )
                    )
                ):
                    visible_witness_descriptor, _created, _identity = (
                        _open_artifact_directory_entry(
                            parent_descriptor,
                            witness_directory_name,
                            create=previous_witness_encoded is not None,
                        )
                    )
                    try:
                        if (
                            witness_identity is not None
                            and witness_path is not None
                            and witness_encoded is not None
                            and witness_path.name != previous_witness_name
                        ):
                            _unlink_owned_artifact_entry(
                                visible_witness_descriptor,
                                witness_path.name,
                                witness_identity,
                                witness_encoded,
                            )
                        if (
                            previous_witness_name is not None
                            and previous_witness_encoded is not None
                            and _read_optional_artifact_entry(
                                visible_witness_descriptor,
                                previous_witness_name,
                            )
                            != previous_witness_encoded
                        ):
                            _replace_artifact_entry_bytes(
                                visible_witness_descriptor,
                                previous_witness_name,
                                previous_witness_encoded,
                            )
                        if (
                            previous_witness_name is not None
                            and _read_optional_artifact_entry(
                                visible_witness_descriptor,
                                previous_witness_name,
                            )
                            != previous_witness_encoded
                        ):
                            raise RetrievalSkillError(
                                "retrieval Skill witness rollback verification failed"
                            )
                        os.fsync(visible_witness_descriptor)
                    finally:
                        os.close(visible_witness_descriptor)
                if (
                    witness_directory_created
                    and witness_directory_name is not None
                    and witness_directory_identity is not None
                ):
                    _remove_owned_empty_artifact_directory(
                        parent_descriptor,
                        witness_directory_name,
                        witness_directory_identity,
                    )
                os.fsync(parent_descriptor)
            except Exception as rollback_error:
                raise RetrievalSkillError(
                    "Retrieval Skill checkpoint bundle rollback failed"
                ) from rollback_error
            raise
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            if rollback_temporary is not None:
                try:
                    os.unlink(rollback_temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            if witness_temporary is not None and witness_parent_descriptor is not None:
                try:
                    os.unlink(
                        witness_temporary,
                        dir_fd=witness_parent_descriptor,
                    )
                except FileNotFoundError:
                    pass
            if (
                not write_succeeded
                and witness_identity is not None
                and witness_path is not None
                and witness_encoded is not None
                and witness_parent_descriptor is not None
                and witness_path.name != previous_witness_name
            ):
                try:
                    _unlink_owned_artifact_entry(
                        witness_parent_descriptor,
                        witness_path.name,
                        witness_identity,
                        witness_encoded,
                    )
                    os.fsync(witness_parent_descriptor)
                except OSError:
                    pass
            if witness_parent_descriptor is not None:
                os.close(witness_parent_descriptor)
            if (
                not write_succeeded
                and witness_directory_created
                and witness_directory_name is not None
                and witness_directory_identity is not None
            ):
                try:
                    _remove_owned_empty_artifact_directory(
                        parent_descriptor,
                        witness_directory_name,
                        witness_directory_identity,
                    )
                except OSError:
                    pass
            os.close(parent_descriptor)
        return checkpoint_sha256

    @staticmethod
    def _all_from(
        skills: Mapping[str, tuple[RetrievalSkill, ...]]
    ) -> tuple[RetrievalSkill, ...]:
        return tuple(
            item
            for skill_id in sorted(skills)
            for item in skills[skill_id]
        )

    def apply_operations(self, operations: Iterable[RetrievalSkillOperation]) -> None:
        if getattr(self, "_read_only", False):
            raise RetrievalSkillError("read-only Retrieval Skill snapshots are immutable")
        proposed = {skill_id: tuple(history) for skill_id, history in self._skills.items()}
        for operation in operations:
            if not isinstance(operation, RetrievalSkillOperation):
                raise RetrievalSkillError("invalid retrieval skill operation")
            proposed = self._apply_one(proposed, operation)
            proposed = self._validated_index(
                self._all_from(proposed),
                active_record_hashes=self._active_record_origins,
            )
        if any(skill.is_active for skill in self._all_from(proposed)):
            _commit_authorized_library_update(self, proposed)
        else:
            file_sha256 = self._file_sha256
            if self.persist:
                file_sha256 = self._write(proposed)
            self._skills = proposed
            self._file_sha256 = file_sha256

    def _apply_one(
        self,
        proposed: dict[str, tuple[RetrievalSkill, ...]],
        operation: RetrievalSkillOperation,
    ) -> dict[str, tuple[RetrievalSkill, ...]]:
        if operation.kind == "add":
            skill = self._operation_skill(operation)
            if skill.skill_id in proposed:
                raise RetrievalSkillError(f"skill already exists: {skill.skill_id}")
            if skill.version != 1 or skill.parent_version is not None or skill.status != "candidate":
                raise RetrievalSkillError("add may create only a version-one candidate skill")
            return {**proposed, skill.skill_id: (skill,)}
        if operation.kind == "repair":
            current = self._current(proposed, operation.skill_id)
            requested = self._operation_skill(operation)
            if requested.status != "candidate":
                raise RetrievalSkillError(
                    "public repair may create only candidate skills; active promotion is trusted-evaluator only"
                )
            replacement = self._next_version(current, requested)
            return {**proposed, current.skill_id: (*proposed[current.skill_id], replacement)}
        if operation.kind == "specialize":
            current = self._current(proposed, operation.skill_id)
            replacement = self._next_version(current, self._operation_skill(operation))
            if replacement.status != "candidate":
                raise RetrievalSkillError(
                    "public specialize may create only candidate skills; active promotion is trusted-evaluator only"
                )
            stage_narrowed = current.stage == "both" and replacement.stage != "both"
            if not stage_narrowed and not replacement.applicability.is_narrower_than(current.applicability):
                raise RetrievalSkillError("specialize must narrow applicability or stage")
            return {**proposed, current.skill_id: (*proposed[current.skill_id], replacement)}
        if operation.kind == "quarantine":
            current = self._current(proposed, operation.skill_id)
            if current.status == "quarantined":
                raise RetrievalSkillError(f"skill is already quarantined: {current.skill_id}")
            reason = _text(operation.reason, "quarantine reason")
            quarantined = copy(current)
            for field, value in (
                ("version", current.version + 1),
                ("parent_version", current.version),
                ("status", "quarantined"),
                ("quarantine_reason", reason),
            ):
                object.__setattr__(quarantined, field, value)
            return {**proposed, current.skill_id: (*proposed[current.skill_id], quarantined)}
        if operation.kind == "merge":
            successor = self._operation_skill(operation)
            sources = tuple(sorted({_text(value, "merge source skill id") for value in operation.source_skill_ids}))
            if len(sources) < 2:
                raise RetrievalSkillError("merge requires at least two source skills")
            if successor.skill_id in proposed:
                raise RetrievalSkillError("merge successor must use a new stable identity")
            source_records = tuple(self._current(proposed, source) for source in sources)
            if any(skill.status == "quarantined" for skill in source_records):
                raise RetrievalSkillError("merge sources must be active or candidates")
            if successor.version != 1 or successor.parent_version is not None or successor.status != "candidate":
                raise RetrievalSkillError("merge must create a version-one candidate successor")
            successor = replace(successor, merged_from_skill_ids=sources)
            result = {**proposed, successor.skill_id: (successor,)}
            for source in sources:
                current = result[source][-1]
                quarantined = copy(current)
                for field, value in (
                    ("version", current.version + 1),
                    ("parent_version", current.version),
                    ("status", "quarantined"),
                    ("quarantine_reason", f"merged into {successor.skill_id}"),
                ):
                    object.__setattr__(quarantined, field, value)
                result[source] = (*result[source], quarantined)
            return result
        raise RetrievalSkillError(f"unknown retrieval skill operation: {operation.kind}")

    @staticmethod
    def _operation_skill(operation: RetrievalSkillOperation) -> RetrievalSkill:
        if operation.skill is None:
            raise RetrievalSkillError(f"{operation.kind} requires a skill")
        return operation.skill

    @staticmethod
    def _current(
        proposed: Mapping[str, tuple[RetrievalSkill, ...]], skill_id: str | None
    ) -> RetrievalSkill:
        key = _text(skill_id, "skill_id")
        if key not in proposed:
            raise RetrievalSkillError(f"unknown retrieval skill: {key}")
        return proposed[key][-1]

    @staticmethod
    def _next_version(current: RetrievalSkill, replacement: RetrievalSkill) -> RetrievalSkill:
        if replacement.skill_id != current.skill_id:
            raise RetrievalSkillError("repair and specialize must preserve skill_id")
        if replacement.name != current.name:
            raise RetrievalSkillError("repair and specialize must preserve the compatibility name")
        if replacement.quarantine_reason is not None and replacement.status != "quarantined":
            raise RetrievalSkillError("replacement may not carry a stale quarantine reason")
        return replace(
            replacement,
            version=current.version + 1,
            parent_version=current.version,
            merged_from_skill_ids=current.merged_from_skill_ids,
            quarantine_reason=(
                replacement.quarantine_reason if replacement.status == "quarantined" else None
            ),
        )

    def get_by_id(self, skill_id: str) -> RetrievalSkill | None:
        history = self._skills.get(skill_id)
        return history[-1] if history else None

    def history(self, skill_id: str) -> tuple[RetrievalSkill, ...]:
        return self._skills.get(skill_id, ())

    def get(self, name: str) -> RetrievalSkill | None:
        matches = [history[-1] for history in self._skills.values() if history[-1].name == name]
        return max(matches, key=lambda skill: (skill.version, skill.skill_id), default=None)

    def all(self) -> tuple[RetrievalSkill, ...]:
        return self._all_from(self._skills)

    def active_skills(self) -> tuple[RetrievalSkill, ...]:
        """Return current prompt-active identities without applying task selectors."""
        _require_active_library_authority(self)
        return tuple(
            skill
            for skill in sorted(
                (history[-1] for history in self._skills.values()),
                key=lambda item: item.skill_id,
            )
            if skill.is_active
        )

    def for_stage(
        self,
        stage: Literal["round1", "round2"],
        *,
        assumption_kinds: Iterable[str] = (),
        gap_types: Iterable[str] = (),
        temporal_relations: Iterable[str] = (),
    ) -> tuple[RetrievalSkill, ...]:
        if stage not in {"round1", "round2"}:
            raise RetrievalSkillError("prompt projection requires round1 or round2")
        return tuple(
            skill
            for skill in self.active_skills()
            if skill.stage in {stage, "both"}
            and skill.applicability.matches(
                assumption_kinds=assumption_kinds,
                gap_types=gap_types,
                temporal_relations=temporal_relations,
            )
        )

    def list_for_prompt(
        self,
        stage: Literal["round1", "round2"] = "round1",
        **applicability: Iterable[str],
    ) -> str:
        skills = self.for_stage(stage, **applicability)
        if not skills:
            return "(no active retrieval skills)"
        return "\n".join(
            "- {name} [{skill_id}@{version}]: {description}; query steps: {steps}; "
            "required chain fields: {fields}; counterevidence: {counterevidence}; failures: {failures}".format(
                name=skill.name,
                skill_id=skill.skill_id,
                version=skill.version,
                description=skill.description,
                steps=" | ".join(skill.query_steps) or "none",
                fields=", ".join(skill.required_chain_fields) or "none",
                counterevidence=skill.counterevidence_rule,
                failures=" | ".join(skill.failure_conditions) or "none",
            )
            for skill in skills
        )

    def add(self, skill: RetrievalSkill) -> None:
        """Compatibility wrapper: direct additions remain candidate-only."""
        if skill.skill_id in self._skills:
            self.apply_operations((RetrievalSkillOperation.repair(skill.skill_id, skill),))
        else:
            self.apply_operations((RetrievalSkillOperation.add(skill),))

    def record_use(self, name: str, smae: float, srmse: float) -> None:
        """Keep the legacy call site harmless; evaluator-owned credit is not skill state."""
        del name, smae, srmse

    def clone(
        self,
        *,
        persist: bool = False,
        read_only: bool = False,
    ) -> "RetrievalSkillLibrary":
        if persist and "verified_release" in self._active_record_origins.values():
            raise RetrievalSkillError("verified release clones must remain read-only")
        if persist and read_only:
            raise RetrievalSkillError("read-only Retrieval Skill snapshots cannot persist")
        clone = object.__new__(type(self))
        clone.path = self.path
        clone.persist = persist
        clone._read_only = read_only or getattr(self, "_read_only", False)
        clone._active_record_origins = dict(self._active_record_origins)
        clone._file_sha256 = self._file_sha256 if persist else None
        clone._skills = {
            skill_id: tuple(history) for skill_id, history in self._skills.items()
        }
        _inherit_active_library_authority(self, clone)
        return clone

    def replay_snapshot(
        self,
        skills: Iterable[RetrievalSkill],
        *,
        persist: bool = False,
    ) -> "RetrievalSkillLibrary":
        """Rebuild a read-only policy snapshot using this verified source's authority.

        Snapshot rows are data, not credentials: any active row must match an
        exact active record already authorized by this library's immutable
        release, migration, or evaluator-checkpoint provenance.
        """
        records = tuple(skills)
        active_hashes = frozenset(
            _record_digest(skill) for skill in records if skill.is_active
        )
        if not active_hashes.issubset(self._active_record_origins):
            raise RetrievalSkillError(
                "policy snapshot contains an active Skill absent from its verified source"
            )
        active_skill_ids = {
            skill.skill_id for skill in records if skill.is_active
        }
        for skill_id in active_skill_ids:
            supplied = tuple(
                _record_digest(skill)
                for skill in records
                if skill.skill_id == skill_id
            )
            verified = tuple(
                _record_digest(skill) for skill in self.history(skill_id)
            )
            if supplied != verified:
                raise RetrievalSkillError(
                    "policy snapshot changed a verified active Skill history"
                )
        origins = {
            digest: self._active_record_origins[digest] for digest in active_hashes
        }
        if persist and "verified_release" in origins.values():
            raise RetrievalSkillError("verified release snapshots must remain read-only")
        replay = object.__new__(type(self))
        replay.path = self.path
        replay.persist = persist
        replay._active_record_origins = origins
        replay._file_sha256 = self._file_sha256 if persist else None
        replay._skills = replay._validated_index(
            records, active_record_hashes=origins
        )
        _inherit_active_library_authority(self, replay)
        return replay

    def __len__(self) -> int:
        return len(self._skills)
