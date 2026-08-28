"""Train-only, typed evolution for the two-stage Retrieval subsystem.

The engine deliberately owns scheduling and audit state, while a trusted evaluator
owns inference and scoring.  Mutation receives only the current typed Genome and a
fixed scope contract; evaluator outputs never cross back into the mutation prompt.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import stat
import fcntl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

from common.llm import (
    JsonExtractionError,
    LLMClient,
    TransientLLMError,
    parse_json_object,
)
from common.metrics import linear_quantile
from evolving_loop.data import ContextTask

from .policy import RetrievalGenome, RetrievalPolicyError
from .skill_library import (
    RetrievalSkillError,
    RetrievalSkillLibrary,
    _move_artifact_entry_to_quarantine,
    _rename_artifact_entry_noreplace,
    _restore_quarantined_artifact_entry,
    _skill_library_cache_identity,
)


RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION = 1
RetrievalChildScope = Literal["A", "B", "C"]
CHILD_SCOPES: tuple[RetrievalChildScope, ...] = ("A", "B", "C")

_SCOPE_ALIASES: dict[str, RetrievalChildScope] = {
    "a": "A",
    "round1": "A",
    "round_1": "A",
    "b": "B",
    "chain": "B",
    "chains": "B",
    "c": "C",
    "round2": "C",
    "round_2": "C",
}
_PRIMARY_SCOPE_FIELDS: dict[RetrievalChildScope, frozenset[str]] = {
    "A": frozenset(
        {"round1_prompt", "round1_strategy", "max_selected_documents"}
    ),
    "B": frozenset(
        {
            "max_evidence_chains",
            "max_citations_per_chain",
            "require_counterevidence_search",
            "require_target_match",
            "require_temporal_overlap",
        }
    ),
    "C": frozenset(
        {"round2_prompt", "round2_strategy", "second_round_trigger"}
    ),
}
_SCOPE_SKILL_STAGE: dict[RetrievalChildScope, str] = {
    "A": "round1",
    "B": "both",
    "C": "round2",
}


class RetrievalEvolutionError(ValueError):
    """Raised when the Retrieval evolution protocol cannot continue safely."""


class RetrievalCheckpointError(RetrievalEvolutionError):
    """Raised when a checkpoint is malformed or belongs to different science."""


class RetrievalForecastingFailure(RetrievalEvolutionError):
    """A completed non-transient forecasting outcome that must not be retried."""

    _FIXED_MESSAGES = {
        "InferenceRuntimeFailure": "retrieval inference failed",
        "InvalidHarnessResult": "retrieval harness returned an invalid result",
        "TrustedScoringFailure": "trusted retrieval scoring failed",
        "MissingRetrievalDiagnostics": "trusted retrieval diagnostics are missing",
        "InvalidTrustedMetric": "trusted retrieval metrics are invalid",
        "EvaluatorExecutionFailure": "trusted evaluator execution failed",
        "InvalidEvaluatorResult": "trusted evaluator result is invalid",
        "ForecastingFailure": "trusted forecasting failed",
    }

    def __init__(self, error_type: str, message: str = "") -> None:
        del message
        category = (
            error_type
            if error_type in self._FIXED_MESSAGES
            else "ForecastingFailure"
        )
        self.error_type = category
        self.original_message = self._FIXED_MESSAGES[category]
        super().__init__(f"{category}:{self.original_message}")

    @property
    def rejection_reason(self) -> str:
        return f"forecasting_failure:{self.error_type}:{self.original_message}"


class RetrievalEvaluator(Protocol):
    """Trusted evaluation boundary consumed by :class:`RetrievalEvolutionEngine`."""

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        skill_library: RetrievalSkillLibrary | None,
        harness_factory: Callable[..., object] | None,
        persist: bool,
        writers_enabled: bool,
        evolver_enabled: bool,
        cache_keys: tuple["RetrievalInferenceCacheKey", ...],
        metric_cap: float,
    ) -> "RetrievalEvaluation": ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_checkpoint_path(path: Path) -> Path:
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
            raise RetrievalCheckpointError(
                f"cannot canonicalize Retrieval evolution system path: {path}"
            ) from error
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RetrievalCheckpointError(
                f"cannot inspect Retrieval evolution checkpoint path: {path}"
            ) from error
        if stat.S_ISLNK(mode):
            raise RetrievalCheckpointError(
                f"Retrieval evolution checkpoint path contains a symlink: {path}"
            )
    return absolute


def _safe_checkpoint_read(path: Path) -> bytes:
    try:
        safe, parent_descriptor = _open_checkpoint_parent(path, create=False)
    except FileNotFoundError as error:
        raise RetrievalCheckpointError(
            "Retrieval evolution checkpoint does not exist"
        ) from error
    try:
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        encoded = _read_checkpoint_entry(parent_descriptor, safe.name)
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        return encoded
    except RetrievalCheckpointError:
        raise
    except OSError as error:
        raise RetrievalCheckpointError(
            "Retrieval evolution checkpoint path changed while reading"
        ) from error
    finally:
        os.close(parent_descriptor)


def _open_checkpoint_parent(
    path: Path, *, create: bool
) -> tuple[Path, int]:
    safe = _safe_checkpoint_path(path)
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
        raise RetrievalCheckpointError(
            "cannot open Retrieval evolution checkpoint root"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint root is not a directory"
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
                raise RetrievalCheckpointError(
                    "Retrieval evolution checkpoint parent is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return safe, descriptor
    except FileNotFoundError:
        os.close(descriptor)
        raise
    except RetrievalCheckpointError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise RetrievalCheckpointError(
            "Retrieval evolution checkpoint parent path changed while opening"
        ) from error
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_checkpoint_parent(path: Path, descriptor: int) -> None:
    try:
        _safe, current_descriptor = _open_checkpoint_parent(path, create=False)
    except FileNotFoundError as error:
        raise RetrievalCheckpointError(
            "Retrieval evolution checkpoint parent path changed"
        ) from error
    try:
        expected = os.fstat(descriptor)
        current = os.fstat(current_descriptor)
        if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint parent directory changed"
            )
    finally:
        os.close(current_descriptor)


def _read_checkpoint_entry(parent_descriptor: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint is not a regular file"
            )
        return handle.read()


def _read_checkpoint_entry_snapshot(
    parent_descriptor: int, name: str
) -> tuple[tuple[int, int], bytes]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint is not a regular file"
            )
        return (metadata.st_dev, metadata.st_ino), handle.read()


def _checkpoint_entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise RetrievalCheckpointError(
            "Retrieval evolution checkpoint path contains a symlink"
        )
    return True


def _safe_checkpoint_exists(path: Path) -> bool:
    try:
        safe, parent_descriptor = _open_checkpoint_parent(path, create=False)
    except FileNotFoundError:
        return False
    try:
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        exists = _checkpoint_entry_exists(parent_descriptor, safe.name)
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        return exists
    finally:
        os.close(parent_descriptor)


def _unique_checkpoint_temporary(
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
        raise RetrievalCheckpointError(
            "cannot allocate a unique Retrieval evolution checkpoint temporary"
        )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Descriptor ownership has ended; retain the unique name rather than
        # risk unlinking a replacement installed during the failed write.
        raise
    return temporary


def _build_checkpoint_authority_boundary():
    current: dict[str, tuple[str, int]] = {}

    def path_key(path: Path) -> str:
        return _digest(str(_safe_checkpoint_path(path)))

    def register(path: Path, checkpoint_sha256: str) -> int:
        key = path_key(path)
        prior = current.get(key)
        epoch = 1 if prior is None else prior[1] + 1
        current[key] = (checkpoint_sha256, epoch)
        return epoch

    def require(path: Path, checkpoint_sha256: str) -> int:
        authorized = current.get(path_key(path))
        if authorized is None or authorized[0] != checkpoint_sha256:
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint failed current authority authentication"
            )
        return authorized[1]

    def authorize_for_operator(
        path: str | Path,
        *,
        expected_sha256: str,
        expected_epoch: int,
    ) -> int:
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
            or type(expected_epoch) is not int
            or expected_epoch < 1
        ):
            raise RetrievalCheckpointError(
                "operator checkpoint authority requires an exact trusted digest and epoch"
            )
        source = _safe_checkpoint_path(Path(path))
        checkpoint_sha256 = hashlib.sha256(_safe_checkpoint_read(source)).hexdigest()
        if checkpoint_sha256 != expected_sha256:
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint does not match the operator's trusted digest"
            )
        key = path_key(source)
        prior = current.get(key)
        requested = (expected_sha256, expected_epoch)
        if prior is not None and prior != requested and expected_epoch <= prior[1]:
            raise RetrievalCheckpointError(
                "Retrieval evolution checkpoint authority epoch is stale"
            )
        current[key] = requested
        return expected_epoch

    return register, require, authorize_for_operator


(
    _register_evolution_checkpoint,
    _require_evolution_checkpoint,
    _authorize_retrieval_evolution_checkpoint_for_operator,
) = _build_checkpoint_authority_boundary()
del _build_checkpoint_authority_boundary


@dataclass(frozen=True)
class _CheckpointAuthorityToken:
    checkpoint_sha256: str
    authority_epoch: int
    checkpoint_identity: tuple[int, int]


_UNSPECIFIED_CHECKPOINT_IDENTITY = object()
_UNSPECIFIED_AUTHORITY_RECORD_IDENTITY = object()


def _exact_checkpoint_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RetrievalCheckpointError(f"invalid {field}")
    return value


def _checkpoint_identity_payload(
    identity: tuple[int, int] | None,
) -> dict[str, int] | None:
    if identity is None:
        return None
    return {"st_dev": identity[0], "st_ino": identity[1]}


def _exact_checkpoint_identity(
    value: object,
    field: str,
    *,
    allow_none: bool = False,
) -> tuple[int, int] | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"st_dev", "st_ino"}
        or type(value["st_dev"]) is not int
        or type(value["st_ino"]) is not int
        or value["st_dev"] < 0
        or value["st_ino"] <= 0
    ):
        raise RetrievalCheckpointError(f"invalid {field}")
    return cast(tuple[int, int], (value["st_dev"], value["st_ino"]))


class _OperatorCheckpointAuthority:
    """Protected two-phase digest/epoch journal for cross-process resume."""

    def __init__(
        self,
        checkpoint_path: Path,
        authority_path: Path,
        authority_head_path: Path,
        *,
        authentication_key: bytes | None,
        expected_authority_anchor: tuple[int, str] | None = None,
        expected_checkpoint_identity: tuple[int, int] | None | object = (
            _UNSPECIFIED_CHECKPOINT_IDENTITY
        ),
        expected_authority_identity: tuple[int, int] | None | object = (
            _UNSPECIFIED_AUTHORITY_RECORD_IDENTITY
        ),
        expected_head_identity: tuple[int, int] | None | object = (
            _UNSPECIFIED_AUTHORITY_RECORD_IDENTITY
        ),
        expected_checkpoint_parent_identity: tuple[int, int] | None = None,
        expected_authority_parent_identity: tuple[int, int] | None = None,
    ) -> None:
        if (
            not isinstance(authentication_key, bytes)
            or len(authentication_key) < 32
        ):
            raise RetrievalCheckpointError(
                "checkpoint authority requires an operator authentication key"
            )
        self._authentication_key = authentication_key
        self.checkpoint_path = _safe_checkpoint_path(checkpoint_path)
        self.authority_path = _safe_checkpoint_path(authority_path)
        self.authority_head_path = _safe_checkpoint_path(authority_head_path)
        if self.checkpoint_path.parent == self.authority_path.parent:
            raise RetrievalCheckpointError(
                "checkpoint authority must be independently provisioned, not adjacent"
            )
        if (
            self.authority_head_path.parent != self.authority_path.parent
            or self.authority_head_path.name == self.authority_path.name
        ):
            raise RetrievalCheckpointError(
                "checkpoint authority head must be a distinct protected record"
            )
        try:
            safe, parent_descriptor = _open_checkpoint_parent(
                self.authority_path, create=False
            )
        except FileNotFoundError as error:
            raise RetrievalCheckpointError(
                "checkpoint authority parent must be independently provisioned"
            ) from error
        parent_metadata = os.fstat(parent_descriptor)
        authority_parent_identity = (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        )
        if (
            expected_authority_parent_identity is not None
            and authority_parent_identity != expected_authority_parent_identity
        ):
            os.close(parent_descriptor)
            raise RetrievalCheckpointError(
                "checkpoint authority parent identity changed after path validation"
            )
        if (
            parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            os.close(parent_descriptor)
            raise RetrievalCheckpointError(
                "checkpoint authority parent is not operator-protected"
            )
        try:
            fcntl.flock(parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(parent_descriptor)
            raise RetrievalCheckpointError(
                "checkpoint authority store is already in use"
            ) from error
        self.authority_path = safe
        self._parent_descriptor = parent_descriptor
        self._checkpoint_parent_identity = expected_checkpoint_parent_identity
        self._closed = False
        self._pending_token: _CheckpointAuthorityToken | None = None
        self._record_identities: dict[str, tuple[int, int] | None] = {
            self.authority_path.name: None,
            self.authority_head_path.name: None,
        }
        try:
            checkpoint_snapshot = self._checkpoint_snapshot()
            checkpoint_identity = (
                checkpoint_snapshot[0] if checkpoint_snapshot is not None else None
            )
            if (
                expected_checkpoint_identity
                is not _UNSPECIFIED_CHECKPOINT_IDENTITY
                and checkpoint_identity != expected_checkpoint_identity
            ):
                raise RetrievalCheckpointError(
                    "checkpoint identity changed after operator path validation"
                )
            self._checkpoint_identity = checkpoint_identity
            self._state = self._load_state()
            self._head = self._load_head()
            for expected, path in (
                (expected_authority_identity, self.authority_path),
                (expected_head_identity, self.authority_head_path),
            ):
                if (
                    expected is not _UNSPECIFIED_AUTHORITY_RECORD_IDENTITY
                    and self._record_identities[path.name] != expected
                ):
                    raise RetrievalCheckpointError(
                        "checkpoint authority record identity changed after path validation"
                    )
            self._require_external_anchor(expected_authority_anchor)
            self._reconcile()
        except Exception:
            self.close()
            raise

    def _checkpoint_snapshot(self) -> tuple[tuple[int, int], str] | None:
        try:
            safe, parent_descriptor = _open_checkpoint_parent(
                self.checkpoint_path, create=False
            )
        except FileNotFoundError:
            return None
        try:
            parent_metadata = os.fstat(parent_descriptor)
            if (
                self._checkpoint_parent_identity is not None
                and (parent_metadata.st_dev, parent_metadata.st_ino)
                != self._checkpoint_parent_identity
            ):
                raise RetrievalCheckpointError(
                    "checkpoint parent identity changed after path validation"
                )
            _revalidate_checkpoint_parent(safe, parent_descriptor)
            try:
                identity, encoded = _read_checkpoint_entry_snapshot(
                    parent_descriptor, safe.name
                )
            except FileNotFoundError:
                return None
            _revalidate_checkpoint_parent(safe, parent_descriptor)
            return identity, hashlib.sha256(encoded).hexdigest()
        finally:
            os.close(parent_descriptor)

    def _checkpoint_digest(self) -> str | None:
        snapshot = self._checkpoint_snapshot()
        return snapshot[1] if snapshot is not None else None

    def _default_state(self) -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": 4,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": None,
            "checkpoint_identity": None,
            "authority_epoch": 0,
            "authority_head": self._authority_head(None, 0, None),
            "pending": None,
        }
        return self._signed_journal(core)

    def _mac(self, domain: bytes, payload: Mapping[str, object]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hmac.new(
            self._authentication_key,
            domain + b"\0" + encoded,
            hashlib.sha256,
        ).hexdigest()

    def _authority_head(
        self,
        digest: str | None,
        epoch: int,
        checkpoint_identity: tuple[int, int] | None,
    ) -> str:
        return self._mac(
            b"retrieval-checkpoint-authority-head-v2",
            {
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": digest,
                "checkpoint_identity": _checkpoint_identity_payload(
                    checkpoint_identity
                ),
                "authority_epoch": epoch,
            },
        )

    def _signed_journal(
        self, state: Mapping[str, object]
    ) -> dict[str, object]:
        core = {key: value for key, value in state.items() if key != "journal_mac"}
        return {
            **core,
            "journal_mac": self._mac(
                b"retrieval-checkpoint-authority-journal-v2", core
            ),
        }

    def _head_state(
        self,
        digest: str | None,
        epoch: int,
        checkpoint_identity: tuple[int, int] | None,
    ) -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": 2,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": digest,
            "checkpoint_identity": _checkpoint_identity_payload(
                checkpoint_identity
            ),
            "authority_epoch": epoch,
            "authority_head": self._authority_head(
                digest, epoch, checkpoint_identity
            ),
        }
        return {
            **core,
            "head_mac": self._mac(
                b"retrieval-checkpoint-authority-monotonic-head-v2", core
            ),
        }

    def _require_external_anchor(
        self,
        expected: tuple[int, str] | None,
    ) -> None:
        pristine = (
            self._record_identities[self.authority_path.name] is None
            and self._record_identities[self.authority_head_path.name] is None
            and self._checkpoint_snapshot() is None
        )
        if expected is None:
            if pristine:
                return
            raise RetrievalCheckpointError(
                "checkpoint resume requires an external authority anchor"
            )
        if (
            type(expected) is not tuple
            or len(expected) != 2
            or type(expected[0]) is not int
            or expected[0] < 0
        ):
            raise RetrievalCheckpointError("invalid external authority anchor")
        expected_head = _exact_checkpoint_digest(
            expected[1], "external authority anchor head"
        )
        committed = (
            self._state["authority_epoch"],
            self._state["authority_head"],
        )
        if committed != (expected[0], expected_head):
            raise RetrievalCheckpointError(
                "external authority anchor does not match committed epoch and head"
            )

    @property
    def current_anchor(self) -> tuple[int, str]:
        return (
            cast(int, self._state["authority_epoch"]),
            cast(str, self._state["authority_head"]),
        )

    def _read_protected_record(
        self, path: Path, label: str
    ) -> dict[str, object] | None:
        try:
            identity, encoded = _read_checkpoint_entry_snapshot(
                self._parent_descriptor, path.name
            )
        except FileNotFoundError:
            self._record_identities[path.name] = None
            return None
        metadata = os.stat(
            path.name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        if (metadata.st_dev, metadata.st_ino) != identity:
            raise RetrievalCheckpointError(
                f"checkpoint authority {label} identity changed while reading"
            )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RetrievalCheckpointError(
                f"checkpoint authority {label} is not operator-protected"
            )
        try:
            raw = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalCheckpointError(
                f"invalid checkpoint authority {label}"
            ) from error
        if not isinstance(raw, dict):
            raise RetrievalCheckpointError(
                f"invalid checkpoint authority {label} schema"
            )
        self._record_identities[path.name] = identity
        return raw

    def _load_state(self) -> dict[str, object]:
        raw = self._read_protected_record(self.authority_path, "journal")
        if raw is None:
            if self._checkpoint_digest() is not None:
                raise RetrievalCheckpointError(
                    "checkpoint exists without independently protected authority"
                )
            return self._default_state()
        expected = {
            "schema_version",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_identity",
            "authority_epoch",
            "authority_head",
            "pending",
            "journal_mac",
        }
        if set(raw) != expected:
            raise RetrievalCheckpointError("invalid checkpoint authority schema")
        if raw["schema_version"] != 4:
            raise RetrievalCheckpointError("unsupported checkpoint authority schema")
        supplied_mac = raw["journal_mac"]
        if type(supplied_mac) is not str or not hmac.compare_digest(
            supplied_mac,
            self._signed_journal(raw)["journal_mac"],
        ):
            raise RetrievalCheckpointError(
                "checkpoint authority journal authentication failed"
            )
        if raw["checkpoint_path"] != str(self.checkpoint_path):
            raise RetrievalCheckpointError("checkpoint authority path binding mismatch")
        epoch = raw["authority_epoch"]
        digest = raw["checkpoint_sha256"]
        checkpoint_identity = _exact_checkpoint_identity(
            raw["checkpoint_identity"],
            "checkpoint authority identity",
            allow_none=True,
        )
        if type(epoch) is not int or epoch < 0:
            raise RetrievalCheckpointError("invalid checkpoint authority epoch")
        if epoch == 0:
            if digest is not None or checkpoint_identity is not None:
                raise RetrievalCheckpointError("invalid initial checkpoint authority")
        else:
            _exact_checkpoint_digest(digest, "checkpoint authority digest")
            if checkpoint_identity is None:
                raise RetrievalCheckpointError(
                    "invalid checkpoint authority identity"
                )
        expected_head = self._authority_head(
            cast(str | None, digest), epoch, checkpoint_identity
        )
        if raw["authority_head"] != expected_head:
            raise RetrievalCheckpointError(
                "checkpoint authority journal head authentication failed"
            )
        pending = raw["pending"]
        if pending is not None:
            if (
                not isinstance(pending, dict)
                or set(pending)
                != {
                    "checkpoint_sha256",
                    "checkpoint_identity",
                    "authority_epoch",
                    "authority_head",
                }
                or type(pending["authority_epoch"]) is not int
                or pending["authority_epoch"] != epoch + 1
            ):
                raise RetrievalCheckpointError("invalid pending checkpoint authority")
            _exact_checkpoint_digest(
                pending["checkpoint_sha256"],
                "pending checkpoint authority digest",
            )
            pending_identity = _exact_checkpoint_identity(
                pending["checkpoint_identity"],
                "pending checkpoint authority identity",
            )
            if pending["authority_head"] != self._authority_head(
                cast(str, pending["checkpoint_sha256"]),
                cast(int, pending["authority_epoch"]),
                pending_identity,
            ):
                raise RetrievalCheckpointError(
                    "pending checkpoint authority head authentication failed"
                )
        return raw

    def _load_head(self) -> dict[str, object] | None:
        raw = self._read_protected_record(self.authority_head_path, "head")
        if raw is None:
            return None
        expected = {
            "schema_version",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_identity",
            "authority_epoch",
            "authority_head",
            "head_mac",
        }
        if set(raw) != expected or raw["schema_version"] != 2:
            raise RetrievalCheckpointError(
                "invalid checkpoint authority head schema"
            )
        supplied_mac = raw["head_mac"]
        core = {key: value for key, value in raw.items() if key != "head_mac"}
        if type(supplied_mac) is not str or not hmac.compare_digest(
            supplied_mac,
            self._mac(
                b"retrieval-checkpoint-authority-monotonic-head-v2", core
            ),
        ):
            raise RetrievalCheckpointError(
                "checkpoint authority head authentication failed"
            )
        if raw["checkpoint_path"] != str(self.checkpoint_path):
            raise RetrievalCheckpointError(
                "checkpoint authority head path binding mismatch"
            )
        epoch = raw["authority_epoch"]
        digest = raw["checkpoint_sha256"]
        checkpoint_identity = _exact_checkpoint_identity(
            raw["checkpoint_identity"],
            "checkpoint authority head identity",
            allow_none=True,
        )
        if type(epoch) is not int or epoch < 0:
            raise RetrievalCheckpointError("invalid checkpoint authority head epoch")
        if epoch == 0:
            if digest is not None or checkpoint_identity is not None:
                raise RetrievalCheckpointError("invalid initial checkpoint authority head")
        else:
            _exact_checkpoint_digest(digest, "checkpoint authority head digest")
            if checkpoint_identity is None:
                raise RetrievalCheckpointError(
                    "invalid checkpoint authority head identity"
                )
        if raw["authority_head"] != self._authority_head(
            cast(str | None, digest), epoch, checkpoint_identity
        ):
            raise RetrievalCheckpointError(
                "checkpoint authority head authentication failed"
            )
        return raw

    def _write_record(
        self, path: Path, payload: Mapping[str, object]
    ) -> None:
        _revalidate_checkpoint_parent(path, self._parent_descriptor)
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        temporary = _unique_checkpoint_temporary(
            self._parent_descriptor,
            path.name,
            encoded,
        )
        staged_identity, staged_bytes = _read_checkpoint_entry_snapshot(
            self._parent_descriptor, temporary
        )
        if staged_bytes != encoded:
            raise RetrievalCheckpointError(
                "checkpoint authority staging bytes changed"
            )
        expected_identity = self._record_identities[path.name]
        try:
            try:
                current_identity, _current_bytes = (
                    _read_checkpoint_entry_snapshot(
                        self._parent_descriptor, path.name
                    )
                )
            except FileNotFoundError:
                current_identity = None
            if current_identity != expected_identity:
                raise RetrievalCheckpointError(
                    "checkpoint authority record identity changed before publication"
                )
            if expected_identity is not None:
                try:
                    quarantine = _move_artifact_entry_to_quarantine(
                        self._parent_descriptor, path.name
                    )
                except Exception as error:
                    raise RetrievalCheckpointError(
                        "checkpoint authority record quarantine failed"
                    ) from error
                if quarantine is None:
                    raise RetrievalCheckpointError(
                        "checkpoint authority record disappeared during publication"
                    )
                try:
                    moved_identity, _moved_bytes = (
                        _read_checkpoint_entry_snapshot(
                            self._parent_descriptor, quarantine
                        )
                    )
                except FileNotFoundError as error:
                    raise RetrievalCheckpointError(
                        "checkpoint authority quarantine identity is unavailable"
                    ) from error
                if moved_identity != expected_identity:
                    try:
                        _restore_quarantined_artifact_entry(
                            self._parent_descriptor,
                            quarantine,
                            path.name,
                            expected_identity=moved_identity,
                        )
                    except Exception as error:
                        raise RetrievalCheckpointError(
                            "checkpoint authority foreign replacement was retained"
                        ) from error
                    raise RetrievalCheckpointError(
                        "checkpoint authority record identity changed during publication"
                    )
            try:
                _rename_artifact_entry_noreplace(
                    self._parent_descriptor, temporary, path.name
                )
            except Exception as error:
                try:
                    published_identity, published_bytes = (
                        _read_checkpoint_entry_snapshot(
                            self._parent_descriptor, path.name
                        )
                    )
                except FileNotFoundError:
                    published_identity, published_bytes = None, None
                if (
                    published_identity != staged_identity
                    or published_bytes != encoded
                ):
                    raise RetrievalCheckpointError(
                        "checkpoint authority no-replace publication failed"
                    ) from error
            temporary = ""
            published_identity, published_bytes = _read_checkpoint_entry_snapshot(
                self._parent_descriptor, path.name
            )
            if (
                published_identity != staged_identity
                or published_bytes != encoded
            ):
                raise RetrievalCheckpointError(
                    "checkpoint authority publication identity changed"
                )
            os.fsync(self._parent_descriptor)
            _revalidate_checkpoint_parent(path, self._parent_descriptor)
            self._record_identities[path.name] = staged_identity
        except Exception:
            # The unique unpublished authority inode is retained on uncertainty.
            raise

    def _write_state(
        self, state: Mapping[str, object]
    ) -> dict[str, object]:
        signed = self._signed_journal(state)
        self._write_record(self.authority_path, signed)
        return signed

    def _write_head(
        self,
        digest: str | None,
        epoch: int,
        checkpoint_identity: tuple[int, int] | None,
    ) -> dict[str, object]:
        head = self._head_state(digest, epoch, checkpoint_identity)
        self._write_record(self.authority_head_path, head)
        return head

    def _reconcile(self) -> None:
        pending = self._state["pending"]
        checkpoint_snapshot = self._checkpoint_snapshot()
        if self._head is None:
            if (
                _checkpoint_entry_exists(
                    self._parent_descriptor, self.authority_path.name
                )
                or checkpoint_snapshot is not None
            ):
                raise RetrievalCheckpointError(
                    "checkpoint authority journal exists without authenticated head"
                )
            return
        head_value = self._head["authority_head"]
        if isinstance(pending, Mapping):
            current_head = self._state["authority_head"]
            pending_head = pending["authority_head"]
            if head_value not in {current_head, pending_head}:
                raise RetrievalCheckpointError(
                    "checkpoint authority journal replay conflicts with monotonic head"
                )
            pending_identity = _exact_checkpoint_identity(
                pending["checkpoint_identity"],
                "pending checkpoint authority identity",
            )
            pending_snapshot = (
                pending_identity,
                pending["checkpoint_sha256"],
            )
            current_identity = _exact_checkpoint_identity(
                self._state["checkpoint_identity"],
                "checkpoint authority identity",
                allow_none=True,
            )
            current_snapshot = (
                None
                if current_identity is None
                else (current_identity, self._state["checkpoint_sha256"])
            )
            if checkpoint_snapshot == pending_snapshot:
                if head_value == current_head:
                    self._head = self._write_head(
                        cast(str, pending["checkpoint_sha256"]),
                        cast(int, pending["authority_epoch"]),
                        pending_identity,
                    )
                self._state = self._write_state({
                    **self._state,
                    "checkpoint_sha256": pending["checkpoint_sha256"],
                    "checkpoint_identity": pending["checkpoint_identity"],
                    "authority_epoch": pending["authority_epoch"],
                    "authority_head": pending["authority_head"],
                    "pending": None,
                })
            elif checkpoint_snapshot == current_snapshot:
                if head_value != current_head:
                    raise RetrievalCheckpointError(
                        "checkpoint authority head advanced without durable checkpoint"
                    )
                self._state = self._write_state(
                    {**self._state, "pending": None}
                )
            else:
                raise RetrievalCheckpointError(
                    "pending checkpoint authority cannot reconcile durable identity or bytes"
                )
        elif head_value != self._state["authority_head"]:
            raise RetrievalCheckpointError(
                "checkpoint authority journal replay conflicts with monotonic head"
            )
        expected_digest = self._state["checkpoint_sha256"]
        expected_identity = _exact_checkpoint_identity(
            self._state["checkpoint_identity"],
            "checkpoint authority identity",
            allow_none=True,
        )
        expected_epoch = self._state["authority_epoch"]
        checkpoint_snapshot = self._checkpoint_snapshot()
        if expected_digest is None:
            if (
                checkpoint_snapshot is not None
                or expected_identity is not None
                or expected_epoch != 0
            ):
                raise RetrievalCheckpointError(
                    "checkpoint authority has no trusted initial state"
                )
        else:
            if checkpoint_snapshot != (expected_identity, expected_digest):
                raise RetrievalCheckpointError(
                    "checkpoint identity or digest does not match protected authority record"
                )
            self._checkpoint_identity = expected_identity
            _authorize_retrieval_evolution_checkpoint_for_operator(
                self.checkpoint_path,
                expected_sha256=cast(str, expected_digest),
                expected_epoch=cast(int, expected_epoch),
            )

    def prepare(
        self,
        checkpoint_sha256: str,
        *,
        checkpoint_identity: tuple[int, int],
    ) -> _CheckpointAuthorityToken:
        if self._closed or self._pending_token is not None:
            raise RetrievalCheckpointError("checkpoint authority transaction is unavailable")
        digest = _exact_checkpoint_digest(
            checkpoint_sha256, "prepared checkpoint digest"
        )
        checkpoint_snapshot = self._checkpoint_snapshot()
        current_checkpoint_identity = (
            checkpoint_snapshot[0] if checkpoint_snapshot is not None else None
        )
        checkpoint_digest = (
            checkpoint_snapshot[1] if checkpoint_snapshot is not None else None
        )
        if (
            current_checkpoint_identity != self._checkpoint_identity
            or checkpoint_digest != self._state["checkpoint_sha256"]
        ):
            raise RetrievalCheckpointError(
                "checkpoint identity changed before authority transaction"
            )
        intended_identity = _exact_checkpoint_identity(
            _checkpoint_identity_payload(checkpoint_identity),
            "prepared checkpoint identity",
        )
        assert intended_identity is not None
        token = _CheckpointAuthorityToken(
            digest,
            cast(int, self._state["authority_epoch"]) + 1,
            intended_identity,
        )
        if self._head is None:
            self._head = self._write_head(
                cast(str | None, self._state["checkpoint_sha256"]),
                cast(int, self._state["authority_epoch"]),
                _exact_checkpoint_identity(
                    self._state["checkpoint_identity"],
                    "checkpoint authority identity",
                    allow_none=True,
                ),
            )
        proposed = {
            **self._state,
            "pending": {
                "checkpoint_sha256": token.checkpoint_sha256,
                "checkpoint_identity": _checkpoint_identity_payload(
                    token.checkpoint_identity
                ),
                "authority_epoch": token.authority_epoch,
                "authority_head": self._authority_head(
                    token.checkpoint_sha256,
                    token.authority_epoch,
                    token.checkpoint_identity,
                ),
            },
        }
        self._state = self._write_state(proposed)
        self._pending_token = token
        return token

    def commit(self, token: _CheckpointAuthorityToken) -> int:
        if self._pending_token is not token:
            raise RetrievalCheckpointError("invalid checkpoint authority transaction")
        checkpoint_snapshot = self._checkpoint_snapshot()
        if (
            checkpoint_snapshot is None
            or checkpoint_snapshot
            != (token.checkpoint_identity, token.checkpoint_sha256)
        ):
            raise RetrievalCheckpointError(
                "checkpoint commit does not match prepared authority identity and digest"
            )
        committed_checkpoint_identity = checkpoint_snapshot[0]
        pending = cast(Mapping[str, object], self._state["pending"])
        self._head = self._write_head(
            token.checkpoint_sha256,
            token.authority_epoch,
            token.checkpoint_identity,
        )
        committed = {
            **self._state,
            "checkpoint_sha256": token.checkpoint_sha256,
            "checkpoint_identity": _checkpoint_identity_payload(
                token.checkpoint_identity
            ),
            "authority_epoch": token.authority_epoch,
            "authority_head": pending["authority_head"],
            "pending": None,
        }
        self._state = self._write_state(committed)
        self._checkpoint_identity = committed_checkpoint_identity
        self._pending_token = None
        _authorize_retrieval_evolution_checkpoint_for_operator(
            self.checkpoint_path,
            expected_sha256=token.checkpoint_sha256,
            expected_epoch=token.authority_epoch,
        )
        return token.authority_epoch

    def abort(self, token: _CheckpointAuthorityToken) -> int | None:
        if self._pending_token is not token:
            return None
        checkpoint_snapshot = self._checkpoint_snapshot()
        if checkpoint_snapshot == (
            token.checkpoint_identity,
            token.checkpoint_sha256,
        ):
            return self.commit(token)
        current_identity = _exact_checkpoint_identity(
            self._state["checkpoint_identity"],
            "checkpoint authority identity",
            allow_none=True,
        )
        current_snapshot = (
            None
            if current_identity is None
            else (current_identity, self._state["checkpoint_sha256"])
        )
        if checkpoint_snapshot != current_snapshot:
            raise RetrievalCheckpointError(
                "checkpoint authority abort found unknown durable identity or bytes"
            )
        rolled_back = {**self._state, "pending": None}
        self._state = self._write_state(rolled_back)
        self._pending_token = None
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self._parent_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._parent_descriptor)


def _open_retrieval_checkpoint_authority_for_operator(
    checkpoint_path: str | Path,
    authority_path: str | Path,
    authority_head_path: str | Path,
    *,
    authentication_key: bytes | None,
    expected_authority_anchor: tuple[int, str] | None = None,
    expected_checkpoint_identity: tuple[int, int] | None | object = (
        _UNSPECIFIED_CHECKPOINT_IDENTITY
    ),
    expected_authority_identity: tuple[int, int] | None | object = (
        _UNSPECIFIED_AUTHORITY_RECORD_IDENTITY
    ),
    expected_head_identity: tuple[int, int] | None | object = (
        _UNSPECIFIED_AUTHORITY_RECORD_IDENTITY
    ),
    expected_checkpoint_parent_identity: tuple[int, int] | None = None,
    expected_authority_parent_identity: tuple[int, int] | None = None,
) -> _OperatorCheckpointAuthority:
    return _OperatorCheckpointAuthority(
        Path(checkpoint_path),
        Path(authority_path),
        Path(authority_head_path),
        authentication_key=authentication_key,
        expected_authority_anchor=expected_authority_anchor,
        expected_checkpoint_identity=expected_checkpoint_identity,
        expected_authority_identity=expected_authority_identity,
        expected_head_identity=expected_head_identity,
        expected_checkpoint_parent_identity=expected_checkpoint_parent_identity,
        expected_authority_parent_identity=expected_authority_parent_identity,
    )


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalEvolutionError(f"{field_name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RetrievalEvolutionError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class RetrievalEvaluation:
    """The complete trusted Retrieval and final-system metric vector."""

    version: str
    task_count: int
    mean_final_smae: float
    mean_final_srmse: float
    mean_contextual_oracle_smae: float
    mean_contextual_oracle_srmse: float
    p90_smae: float
    p95_smae: float
    supporting_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    invalid_count: int
    catastrophic_count: int
    task_traces: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise RetrievalEvolutionError("evaluation version must be non-empty")
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise RetrievalEvolutionError("evaluation task_count must be positive")
        for name in (
            "mean_final_smae",
            "mean_final_srmse",
            "mean_contextual_oracle_smae",
            "mean_contextual_oracle_srmse",
            "p90_smae",
            "p95_smae",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise RetrievalEvolutionError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        for name in (
            "supporting_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise RetrievalEvolutionError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        for name in ("invalid_count", "catastrophic_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RetrievalEvolutionError(f"{name} must be a non-negative integer")
        if not isinstance(self.task_traces, tuple) or any(
            not isinstance(item, dict) for item in self.task_traces
        ):
            raise RetrievalEvolutionError("task_traces must be a tuple of trace objects")
        normalized_traces: list[dict[str, object]] = []
        for trace in self.task_traces:
            normalized_trace = dict(trace)
            normalized_trace["final_smae"] = _finite(
                trace.get("final_smae"), "task trace final_smae"
            )
            normalized_traces.append(normalized_trace)
        object.__setattr__(self, "task_traces", tuple(normalized_traces))

    def gate_failures(
        self,
        parent: "RetrievalEvaluation",
        tolerance: float,
        *,
        require_strict_contextual_gain: bool = True,
    ) -> tuple[str, ...]:
        tolerance = _finite(tolerance, "tolerance")
        if tolerance < 0:
            raise RetrievalEvolutionError("tolerance cannot be negative")
        failures: list[str] = []
        if self.task_count != parent.task_count:
            failures.append("task_count")
        if (
            self.mean_contextual_oracle_smae
            > parent.mean_contextual_oracle_smae + tolerance
        ):
            failures.append("mean_contextual_oracle_smae")
        if (
            self.mean_contextual_oracle_srmse
            > parent.mean_contextual_oracle_srmse + tolerance
        ):
            failures.append("mean_contextual_oracle_srmse")
        if require_strict_contextual_gain and not (
            self.mean_contextual_oracle_smae
            < parent.mean_contextual_oracle_smae - tolerance
            or self.mean_contextual_oracle_srmse
            < parent.mean_contextual_oracle_srmse - tolerance
        ):
            failures.append("strict_contextual_gain")
        if self.mean_final_smae > parent.mean_final_smae + tolerance:
            failures.append("mean_final_smae")
        if self.mean_final_srmse > parent.mean_final_srmse + tolerance:
            failures.append("mean_final_srmse")
        if self.p90_smae > parent.p90_smae + tolerance:
            failures.append("p90_smae")
        if self.p95_smae > parent.p95_smae + tolerance:
            failures.append("p95_smae")
        if self.supporting_recall < parent.supporting_recall - 0.02:
            failures.append("supporting_recall")
        if self.distractor_avoidance < parent.distractor_avoidance - 0.02:
            failures.append("distractor_avoidance")
        if self.exact_quote_validity != 1.0:
            failures.append("exact_quote_validity")
        if self.catastrophic_count > parent.catastrophic_count:
            failures.append("catastrophic_count")
        if self.invalid_count > parent.invalid_count:
            failures.append("invalid_count")
        return tuple(failures)

    def dev_accepts(
        self, parent: "RetrievalEvaluation", tolerance: float
    ) -> bool:
        return not self.gate_failures(
            parent,
            tolerance,
            require_strict_contextual_gain=True,
        )

    def pareto_safe(
        self, parent: "RetrievalEvaluation", tolerance: float
    ) -> bool:
        return not self.gate_failures(
            parent,
            tolerance,
            require_strict_contextual_gain=False,
        )

    @property
    def rank_key(self) -> tuple[float, ...]:
        """Lower is better; safety eligibility is checked separately."""
        return (
            self.mean_contextual_oracle_srmse,
            self.mean_contextual_oracle_smae,
            self.mean_final_srmse,
            self.mean_final_smae,
            self.p95_smae,
            self.p90_smae,
            -self.supporting_recall,
            -self.distractor_avoidance,
            -self.complete_chain_rate,
            float(self.catastrophic_count),
            float(self.invalid_count),
        )

    def summary(self) -> dict[str, object]:
        payload = self.to_payload()
        payload.pop("task_traces")
        return payload

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "task_count": self.task_count,
            "mean_final_smae": self.mean_final_smae,
            "mean_final_srmse": self.mean_final_srmse,
            "mean_contextual_oracle_smae": self.mean_contextual_oracle_smae,
            "mean_contextual_oracle_srmse": self.mean_contextual_oracle_srmse,
            "p90_smae": self.p90_smae,
            "p95_smae": self.p95_smae,
            "supporting_recall": self.supporting_recall,
            "distractor_avoidance": self.distractor_avoidance,
            "exact_quote_validity": self.exact_quote_validity,
            "complete_chain_rate": self.complete_chain_rate,
            "invalid_count": self.invalid_count,
            "catastrophic_count": self.catastrophic_count,
            "task_traces": list(self.task_traces),
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalEvaluation":
        fields = {
            "version",
            "task_count",
            "mean_final_smae",
            "mean_final_srmse",
            "mean_contextual_oracle_smae",
            "mean_contextual_oracle_srmse",
            "p90_smae",
            "p95_smae",
            "supporting_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
            "invalid_count",
            "catastrophic_count",
            "task_traces",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise RetrievalCheckpointError("invalid checkpoint evaluation")
        traces = raw["task_traces"]
        if not isinstance(traces, list) or any(not isinstance(item, dict) for item in traces):
            raise RetrievalCheckpointError("invalid checkpoint task traces")
        float_fields = {
            "mean_final_smae",
            "mean_final_srmse",
            "mean_contextual_oracle_smae",
            "mean_contextual_oracle_srmse",
            "p90_smae",
            "p95_smae",
            "supporting_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
        }
        if (
            type(raw["version"]) is not str
            or any(
                isinstance(raw[field], bool)
                or not isinstance(raw[field], (int, float))
                for field in float_fields
            )
            or any(
                type(raw[field]) is not int
                for field in ("task_count", "invalid_count", "catastrophic_count")
            )
            or any(
                type(trace.get("task_id")) is not str
                or isinstance(trace.get("final_smae"), bool)
                or not isinstance(trace.get("final_smae"), (int, float))
                for trace in traces
            )
        ):
            raise RetrievalCheckpointError(
                "checkpoint evaluation metric and trace types must be exact"
            )
        try:
            return cls(
                **{
                    **{key: raw[key] for key in fields if key != "task_traces"},
                    "task_traces": tuple(dict(item) for item in traces),
                }
            )
        except RetrievalEvolutionError as error:
            raise RetrievalCheckpointError(
                "checkpoint evaluation contains an invalid metric"
            ) from error


def combine_retrieval_evaluations(
    version: str, evaluations: Sequence[RetrievalEvaluation]
) -> RetrievalEvaluation:
    """Combine disjoint task batches without blending count-based safety fields."""
    if not evaluations:
        raise RetrievalEvolutionError("cannot combine an empty evaluation vector")
    total = sum(item.task_count for item in evaluations)

    def weighted(field_name: str) -> float:
        return sum(
            getattr(item, field_name) * item.task_count for item in evaluations
        ) / total

    traces = tuple(trace for item in evaluations for trace in item.task_traces)
    if len(traces) != total:
        raise RetrievalEvolutionError(
            "each aggregated task must have exactly one task trace"
        )
    trace_task_ids = tuple(trace.get("task_id") for trace in traces)
    if any(type(task_id) is not str for task_id in trace_task_ids) or len(
        set(trace_task_ids)
    ) != len(trace_task_ids):
        raise RetrievalEvolutionError(
            "aggregated task traces must provide unique exact task coverage"
        )
    trace_smae = [
        _finite(trace.get("final_smae"), "task trace final_smae")
        for trace in traces
    ]

    return RetrievalEvaluation(
        version=version,
        task_count=total,
        mean_final_smae=weighted("mean_final_smae"),
        mean_final_srmse=weighted("mean_final_srmse"),
        mean_contextual_oracle_smae=weighted(
            "mean_contextual_oracle_smae"
        ),
        mean_contextual_oracle_srmse=weighted(
            "mean_contextual_oracle_srmse"
        ),
        p90_smae=linear_quantile(trace_smae, 0.90),
        p95_smae=linear_quantile(trace_smae, 0.95),
        supporting_recall=weighted("supporting_recall"),
        distractor_avoidance=weighted("distractor_avoidance"),
        exact_quote_validity=weighted("exact_quote_validity"),
        complete_chain_rate=weighted("complete_chain_rate"),
        invalid_count=sum(item.invalid_count for item in evaluations),
        catastrophic_count=sum(item.catastrophic_count for item in evaluations),
        task_traces=traces,
    )


def _library_hash(library: RetrievalSkillLibrary | None) -> str:
    payload = [] if library is None else [skill.to_payload() for skill in library.all()]
    return _digest(payload)


@dataclass(frozen=True)
class RetrievalInferenceCacheKey:
    """All executable inputs that may change one task's trusted evaluation."""

    task_id: str
    task_sha256: str
    genome_sha256: str
    skill_library_sha256: str
    skill_authority_sha256: str
    verifier_sha256: str
    evaluator_sha256: str
    metric_sha256: str
    metric_cap: float
    harness_sha256: str
    scientific_inputs_sha256: str

    def digest(self) -> str:
        return _digest(
            {
                "task_id": self.task_id,
                "task_sha256": self.task_sha256,
                "genome_sha256": self.genome_sha256,
                "skill_library_sha256": self.skill_library_sha256,
                "skill_authority_sha256": self.skill_authority_sha256,
                "verifier_sha256": self.verifier_sha256,
                "evaluator_sha256": self.evaluator_sha256,
                "metric_sha256": self.metric_sha256,
                "metric_cap": self.metric_cap,
                "harness_sha256": self.harness_sha256,
                "scientific_inputs_sha256": self.scientific_inputs_sha256,
            }
        )


def build_inference_cache_key(
    task: ContextTask,
    genome: RetrievalGenome,
    skill_library: RetrievalSkillLibrary | None,
    *,
    verifier_hash: str,
    evaluator_hash: str,
    metric_hash: str,
    metric_cap: float,
    harness_hash: str,
    scientific_inputs_hash: str,
) -> RetrievalInferenceCacheKey:
    return RetrievalInferenceCacheKey(
        task_id=task.numeric.task_id,
        task_sha256=_digest(_task_science_payload(task)),
        genome_sha256=genome.fingerprint(),
        skill_library_sha256=_library_hash(skill_library),
        skill_authority_sha256=(
            _digest({"kind": "no_library"})
            if skill_library is None
            else _skill_library_cache_identity(skill_library)
        ),
        verifier_sha256=str(verifier_hash),
        evaluator_sha256=str(evaluator_hash),
        metric_sha256=str(metric_hash),
        metric_cap=_finite(metric_cap, "metric_cap"),
        harness_sha256=str(harness_hash),
        scientific_inputs_sha256=str(scientific_inputs_hash),
    )


def _normalize_scope(scope: str) -> RetrievalChildScope | None:
    if scope in CHILD_SCOPES:
        return cast(RetrievalChildScope, scope)
    return _SCOPE_ALIASES.get(str(scope).strip().lower())


def parse_scoped_child(
    parent: RetrievalGenome,
    proposal: Mapping[str, object],
    *,
    scope: str,
    skill_library: RetrievalSkillLibrary | None = None,
) -> RetrievalGenome | None:
    """Parse a complete Genome and reject any field outside one fixed scope.

    Skill activation is stage-owned as well: A may change Round 1 IDs, B only
    cross-stage IDs, and C only Round 2 IDs.  Unknown Skill identity therefore
    fails closed instead of being inferred from a name supplied by the model.
    """
    normalized = _normalize_scope(scope)
    if normalized is None or not isinstance(parent, RetrievalGenome):
        return None
    try:
        child = RetrievalGenome.from_payload(proposal)
    except (RetrievalPolicyError, TypeError, ValueError):
        return None
    if (
        child.schema_version != parent.schema_version
        or child.parent != parent.version
        or child.version == parent.version
    ):
        return None
    ignored = {"version", "parent"}
    changed = {
        field_name
        for field_name in parent.to_payload()
        if field_name not in ignored
        and getattr(child, field_name) != getattr(parent, field_name)
    }
    allowed = _PRIMARY_SCOPE_FIELDS[normalized] | {"active_skill_ids"}
    if not changed or not changed.issubset(allowed):
        return None
    eligible_by_id: dict[str, object] = {}
    if child.active_skill_ids:
        if skill_library is None:
            return None
        eligible_by_id = {
            skill.skill_id: skill for skill in skill_library.active_skills()
        }
        if not set(child.active_skill_ids).issubset(eligible_by_id):
            return None
    if "active_skill_ids" in changed:
        if skill_library is None:
            return None
        changed_skill_ids = set(parent.active_skill_ids).symmetric_difference(
            child.active_skill_ids
        )
        owned_stage = _SCOPE_SKILL_STAGE[normalized]
        eligible = {
            skill.skill_id: skill for skill in skill_library.active_skills()
        }
        for skill_id in changed_skill_ids:
            skill = eligible.get(skill_id)
            if skill is None or skill.stage != owned_stage:
                return None
    return child


@dataclass(frozen=True)
class RetrievalEvolutionConfig:
    generations: int = 3
    screen_tasks: int = 8
    promote: int = 2
    train_folds: int = 5
    tolerance: float = 1e-12
    random_seed: int = 7
    transient_retries: int = 2
    checkpoint_path: str | Path | None = None
    resume: bool = True
    dataset_split_hash: str | None = None
    verifier_hash: str | None = None
    evaluator_hash: str | None = None
    metric_hash: str | None = None
    mutation_model_hash: str | None = None
    harness_hash: str | None = None
    metric_cap: float = 5.0

    def __post_init__(self) -> None:
        for name, lower in (
            ("generations", 1),
            ("train_folds", 2),
            ("transient_retries", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise RetrievalEvolutionError(f"invalid {name}")
        if self.generations > 333:
            raise RetrievalEvolutionError("generations exceed the vNNN namespace")
        if self.screen_tasks != 8:
            raise RetrievalEvolutionError("Retrieval evolution screens exactly eight Train tasks")
        if (
            isinstance(self.promote, bool)
            or not isinstance(self.promote, int)
            or not 0 <= self.promote <= 2
        ):
            raise RetrievalEvolutionError("Retrieval evolution promotes at most two children")
        tolerance = _finite(self.tolerance, "tolerance")
        cap = _finite(self.metric_cap, "metric_cap")
        if tolerance < 0 or cap <= 0:
            raise RetrievalEvolutionError("invalid tolerance or metric_cap")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise RetrievalEvolutionError("invalid random_seed")
        if self.harness_hash is not None and (
            type(self.harness_hash) is not str or not self.harness_hash
        ):
            raise RetrievalEvolutionError("harness_hash must be a non-empty digest")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "metric_cap", cap)
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True)
class RetrievalGenerationTrace:
    generation: int
    parent_version: str
    parent_fingerprint: str
    child_versions: tuple[str, ...]
    child_fingerprints: tuple[str, ...]
    child_scopes: tuple[RetrievalChildScope, ...]
    child_proposals: tuple[dict[str, object], ...]
    screen_task_ids: tuple[str, ...]
    fold_entities: tuple[tuple[str, ...], ...]
    promoted_fingerprints: tuple[str, ...]
    train_winner_version: str
    train_winner_fingerprint: str
    rejection_reasons: dict[str, str]
    screen_summaries: dict[str, dict[str, object]]
    train_summaries: dict[str, dict[str, object]]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise RetrievalEvolutionError("generation must be a non-negative integer")
        if (
            len(self.child_versions) != 3
            or len(self.child_fingerprints) != 3
            or self.child_scopes != CHILD_SCOPES
            or len(self.child_proposals) != 3
            or any(type(item) is not dict for item in self.child_proposals)
            or len(set(self.child_versions)) != 3
            or len(set(self.child_fingerprints)) != 3
        ):
            raise RetrievalEvolutionError(
                "each generation must bind exactly the A/B/C child vector"
            )
        if len(self.promoted_fingerprints) > 2 or not set(
            self.promoted_fingerprints
        ).issubset(self.child_fingerprints):
            raise RetrievalEvolutionError("invalid promoted Child fingerprint vector")

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "parent_version": self.parent_version,
            "parent_fingerprint": self.parent_fingerprint,
            "child_versions": list(self.child_versions),
            "child_fingerprints": list(self.child_fingerprints),
            "child_scopes": list(self.child_scopes),
            "child_proposals": [dict(item) for item in self.child_proposals],
            "screen_task_ids": list(self.screen_task_ids),
            "fold_entities": [list(item) for item in self.fold_entities],
            "promoted_fingerprints": list(self.promoted_fingerprints),
            "train_winner_version": self.train_winner_version,
            "train_winner_fingerprint": self.train_winner_fingerprint,
            "rejection_reasons": dict(self.rejection_reasons),
            "screen_summaries": self.screen_summaries,
            "train_summaries": self.train_summaries,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalGenerationTrace":
        if not isinstance(raw, Mapping):
            raise RetrievalCheckpointError("invalid generation trace")
        expected = {
            "generation",
            "parent_version",
            "parent_fingerprint",
            "child_versions",
            "child_fingerprints",
            "child_scopes",
            "child_proposals",
            "screen_task_ids",
            "fold_entities",
            "promoted_fingerprints",
            "train_winner_version",
            "train_winner_fingerprint",
            "rejection_reasons",
            "screen_summaries",
            "train_summaries",
        }
        if set(raw) != expected:
            raise RetrievalCheckpointError("invalid generation trace schema")
        try:
            if type(raw["generation"]) is not int:
                raise ValueError
            string_fields = (
                "parent_version",
                "parent_fingerprint",
                "train_winner_version",
                "train_winner_fingerprint",
            )
            if any(type(raw[field]) is not str for field in string_fields):
                raise ValueError
            vector_fields = (
                "child_versions",
                "child_fingerprints",
                "child_scopes",
                "screen_task_ids",
                "promoted_fingerprints",
            )
            if any(type(raw[field]) is not list for field in vector_fields):
                raise ValueError
            if any(
                type(item) is not str
                for field in vector_fields
                for item in raw[field]
            ):
                raise ValueError
            scopes = tuple(raw["child_scopes"])
            if scopes != CHILD_SCOPES:
                raise ValueError
            proposals = raw["child_proposals"]
            if type(proposals) is not list or len(proposals) != 3 or any(
                type(item) is not dict for item in proposals
            ):
                raise ValueError
            fold_entities_raw = raw["fold_entities"]
            if type(fold_entities_raw) is not list or any(
                type(item) is not list
                or any(type(entity) is not str for entity in item)
                for item in fold_entities_raw
            ):
                raise ValueError
            rejection_raw = raw["rejection_reasons"]
            if type(rejection_raw) is not dict or any(
                type(key) is not str or type(value) is not str
                for key, value in rejection_raw.items()
            ):
                raise ValueError
            for field in ("screen_summaries", "train_summaries"):
                value = raw[field]
                if type(value) is not dict or any(
                    type(key) is not str or type(summary) is not dict
                    for key, summary in value.items()
                ):
                    raise ValueError
            return cls(
                generation=raw["generation"],
                parent_version=raw["parent_version"],
                parent_fingerprint=raw["parent_fingerprint"],
                child_versions=tuple(raw["child_versions"]),
                child_fingerprints=tuple(raw["child_fingerprints"]),
                child_scopes=cast(tuple[RetrievalChildScope, ...], scopes),
                child_proposals=tuple(dict(item) for item in proposals),
                screen_task_ids=tuple(raw["screen_task_ids"]),
                fold_entities=tuple(
                    tuple(item) for item in fold_entities_raw
                ),
                promoted_fingerprints=tuple(raw["promoted_fingerprints"]),
                train_winner_version=raw["train_winner_version"],
                train_winner_fingerprint=raw["train_winner_fingerprint"],
                rejection_reasons=dict(rejection_raw),
                screen_summaries={
                    key: dict(value)
                    for key, value in raw["screen_summaries"].items()
                },
                train_summaries={
                    key: dict(value)
                    for key, value in raw["train_summaries"].items()
                },
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid generation trace values") from error


@dataclass(frozen=True)
class RetrievalEvolutionResult:
    original_parent: RetrievalGenome
    train_winner: RetrievalGenome
    selected_genome: RetrievalGenome
    accepted: bool
    acceptance_reasons: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    parent_dev: RetrievalEvaluation | None
    child_dev: RetrievalEvaluation | None
    generations: tuple[RetrievalGenerationTrace, ...]
    trace: tuple[dict[str, object], ...]
    release_genome: RetrievalGenome | None
    release_published: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "original_parent": self.original_parent.to_payload(),
            "train_winner": self.train_winner.to_payload(),
            "selected_genome": self.selected_genome.to_payload(),
            "accepted": self.accepted,
            "acceptance_reasons": list(self.acceptance_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "parent_dev": self.parent_dev.to_payload() if self.parent_dev else None,
            "child_dev": self.child_dev.to_payload() if self.child_dev else None,
            "generations": [item.to_payload() for item in self.generations],
            "trace": list(self.trace),
            "release_genome": (
                self.release_genome.to_payload() if self.release_genome else None
            ),
            "release_published": self.release_published,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalEvolutionResult":
        expected = {
            "original_parent",
            "train_winner",
            "selected_genome",
            "accepted",
            "acceptance_reasons",
            "rejection_reasons",
            "parent_dev",
            "child_dev",
            "generations",
            "trace",
            "release_genome",
            "release_published",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise RetrievalCheckpointError("invalid completed result")
        if type(raw["accepted"]) is not bool or type(
            raw["release_published"]
        ) is not bool:
            raise RetrievalCheckpointError("invalid completed result boolean fields")
        try:
            if (
                any(
                    type(raw[field]) is not dict
                    for field in (
                        "original_parent",
                        "train_winner",
                        "selected_genome",
                    )
                )
                or any(
                    type(raw[field]) is not list
                    or any(type(item) is not str for item in raw[field])
                    for field in ("acceptance_reasons", "rejection_reasons")
                )
                or type(raw["generations"]) is not list
                or any(type(item) is not dict for item in raw["generations"])
                or type(raw["trace"]) is not list
                or any(type(item) is not dict for item in raw["trace"])
                or (
                    raw["parent_dev"] is not None
                    and type(raw["parent_dev"]) is not dict
                )
                or (
                    raw["child_dev"] is not None
                    and type(raw["child_dev"]) is not dict
                )
                or (
                    raw["release_genome"] is not None
                    and type(raw["release_genome"]) is not dict
                )
            ):
                raise TypeError
            parent_dev = (
                None
                if raw["parent_dev"] is None
                else RetrievalEvaluation.from_payload(raw["parent_dev"])
            )
            child_dev = (
                None
                if raw["child_dev"] is None
                else RetrievalEvaluation.from_payload(raw["child_dev"])
            )
            release_genome = (
                None
                if raw["release_genome"] is None
                else RetrievalGenome.from_payload(raw["release_genome"])
            )
            return cls(
                original_parent=RetrievalGenome.from_payload(raw["original_parent"]),
                train_winner=RetrievalGenome.from_payload(raw["train_winner"]),
                selected_genome=RetrievalGenome.from_payload(raw["selected_genome"]),
                accepted=raw["accepted"],
                acceptance_reasons=tuple(raw["acceptance_reasons"]),
                rejection_reasons=tuple(raw["rejection_reasons"]),
                parent_dev=parent_dev,
                child_dev=child_dev,
                generations=tuple(
                    RetrievalGenerationTrace.from_payload(item)
                    for item in raw["generations"]
                ),
                trace=tuple(dict(item) for item in raw["trace"]),
                release_genome=release_genome,
                release_published=raw["release_published"],
            )
        except (TypeError, ValueError, RetrievalPolicyError) as error:
            raise RetrievalCheckpointError("invalid completed result values") from error


def _task_science_payload(task: ContextTask) -> dict[str, object]:
    return {
        "task_id": task.numeric.task_id,
        "entity_name": task.numeric.entity_name,
        "history_values": list(task.numeric.history_values),
        "future_values": list(task.numeric.future_values),
        "prediction_length": task.numeric.prediction_length,
        "frequency": task.numeric.frequency,
        "seasonal_period": task.numeric.seasonal_period,
        "target_name": task.target_name,
        "target_description": task.target_description,
        "history_timestamps": list(task.history_timestamps),
        "future_timestamps": list(task.future_timestamps),
        "documents": [
            {
                "document_id": item.document_id,
                "content": item.content,
                "role": item.role,
                "subtype": item.subtype,
            }
            for item in task.documents
        ],
        "gt_evidence": list(task.gt_evidence),
        "labels_public": task.labels_public,
    }


class RetrievalEvolutionEngine:
    """Evolve one Retrieval coordinate without consulting Dev until the end."""

    def __init__(
        self,
        mutation_llm: LLMClient,
        evaluator: RetrievalEvaluator,
        config: RetrievalEvolutionConfig | None = None,
        *,
        skill_library: RetrievalSkillLibrary | None = None,
        harness_factory: Callable[..., object] | None = None,
        _checkpoint_authority: _OperatorCheckpointAuthority | None = None,
        _checkpoint_payload_validator: Callable[[object], None] | None = None,
        _checkpoint_parent_identity: tuple[int, int] | None = None,
    ) -> None:
        self.mutation_llm = mutation_llm
        if not callable(getattr(evaluator, "evaluate", None)):
            raise RetrievalEvolutionError(
                "trusted evaluator must implement the complete evaluate protocol"
            )
        self.evaluator = evaluator
        self.config = config or RetrievalEvolutionConfig()
        if harness_factory is not None and self.config.harness_hash is None:
            raise RetrievalEvolutionError(
                "harness_factory requires an explicit frozen harness_hash digest"
            )
        self.skill_library = skill_library
        self.harness_factory = harness_factory
        if _checkpoint_authority is not None and self.config.checkpoint_path is None:
            raise RetrievalEvolutionError(
                "checkpoint authority requires a checkpoint path"
            )
        if (
            _checkpoint_payload_validator is not None
            and not callable(_checkpoint_payload_validator)
        ):
            raise RetrievalEvolutionError("invalid checkpoint payload validator")
        self._checkpoint_authority = _checkpoint_authority
        self._checkpoint_parent_identity = _checkpoint_parent_identity
        self._checkpoint_payload_validator = _checkpoint_payload_validator
        self._scientific_inputs: dict[str, object] = {}
        self._original_parent: RetrievalGenome | None = None
        self._current_parent: RetrievalGenome | None = None
        self._next_generation = 0
        self._generations: list[RetrievalGenerationTrace] = []
        self._trace: list[dict[str, object]] = []
        self._evaluation_cache: dict[str, RetrievalEvaluation] = {}
        self._cache_records: dict[str, dict[str, object]] = {}
        self._terminal_outcomes: dict[str, dict[str, object]] = {}
        self._pending_children: dict[str, object] | None = None
        self._all_child_fingerprints: list[str] = []
        self._candidate_libraries: dict[str, RetrievalSkillLibrary | None] = {}
        self._checkpoint_file_sha256: str | None = None
        self._checkpoint_file_identity: tuple[int, int] | None = None
        self._checkpoint_authority_epoch: int | None = None

    def evolve(
        self,
        parent: RetrievalGenome,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> RetrievalEvolutionResult:
        train = tuple(train_tasks)
        dev = tuple(dev_tasks)
        self._validate_inputs(parent, train, dev)
        screen, remaining_folds = self._partition_train(train)
        self._scientific_inputs = self._science_signature(parent, train, dev)
        completed = self._load_checkpoint(
            parent,
            train,
            dev,
            screen,
            remaining_folds,
        )
        if completed is not None:
            return completed
        assert self._original_parent is not None
        assert self._current_parent is not None
        self._candidate_libraries.setdefault(
            self._original_parent.fingerprint(), self._clone_library(self.skill_library)
        )
        self._candidate_libraries.setdefault(
            self._current_parent.fingerprint(),
            self._candidate_libraries[self._original_parent.fingerprint()],
        )

        for generation in range(self._next_generation, self.config.generations):
            self._run_generation(generation, screen, remaining_folds)

        original = self._original_parent
        winner = self._current_parent
        parent_dev = self._evaluate_batch(
            original,
            dev,
            stage="parent_dev",
            readonly=True,
            library=self._readonly_library(original),
        )
        try:
            child_dev = self._evaluate_batch(
                winner,
                dev,
                stage="child_dev",
                readonly=True,
                library=self._readonly_library(winner),
            )
        except TransientLLMError:
            raise
        except RetrievalForecastingFailure as error:
            child_dev = None
            rejection_reasons = (f"child_dev_failure:{error.error_type}:{error.original_message}",)
        else:
            rejection_reasons = child_dev.gate_failures(
                parent_dev,
                self.config.tolerance,
                require_strict_contextual_gain=True,
            )
        accepted = child_dev is not None and not rejection_reasons
        self._event(
            "dev_completed",
            original_parent=original.version,
            train_winner=winner.version,
            accepted=accepted,
            rejection_reasons=list(rejection_reasons),
        )
        self._event(
            "release_accepted" if accepted else "release_rejected",
            genome=winner.version if accepted else original.version,
            publication_deferred=True,
        )
        result = RetrievalEvolutionResult(
            original_parent=original,
            train_winner=winner,
            selected_genome=winner if accepted else original,
            accepted=accepted,
            acceptance_reasons=("all_dev_gates_passed",) if accepted else (),
            rejection_reasons=tuple(rejection_reasons),
            parent_dev=parent_dev,
            child_dev=child_dev,
            generations=tuple(self._generations),
            trace=tuple(self._trace),
            release_genome=winner if accepted else None,
            release_published=False,
        )
        self._save_checkpoint(status="complete", result=result)
        return result

    def _validate_inputs(
        self,
        parent: RetrievalGenome,
        train: tuple[ContextTask, ...],
        dev: tuple[ContextTask, ...],
    ) -> None:
        if not isinstance(parent, RetrievalGenome):
            raise RetrievalEvolutionError("parent must be a RetrievalGenome")
        if int(parent.version[1:]) + self.config.generations * 3 > 999:
            raise RetrievalEvolutionError(
                "configured generations exceed the remaining vNNN namespace"
            )
        if len(train) != 80 or len(dev) != 20:
            raise RetrievalEvolutionError(
                "Retrieval evolution requires exactly 80 Train and 20 Dev tasks"
            )
        if any(not isinstance(task, ContextTask) for task in (*train, *dev)):
            raise RetrievalEvolutionError("all evolution cases must be ContextTask records")
        train_ids = [task.numeric.task_id for task in train]
        dev_ids = [task.numeric.task_id for task in dev]
        if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
            raise RetrievalEvolutionError("task IDs must be unique within each split")
        if set(train_ids).intersection(dev_ids):
            raise RetrievalEvolutionError("Train and Dev task IDs must be disjoint")

    def _science_signature(
        self,
        parent: RetrievalGenome,
        train: tuple[ContextTask, ...],
        dev: tuple[ContextTask, ...],
    ) -> dict[str, object]:
        split_content_hash = _digest(
            {
                "train": [_task_science_payload(task) for task in train],
                "dev": [_task_science_payload(task) for task in dev],
            }
        )
        evaluator_type = type(self.evaluator)
        llm_type = type(self.mutation_llm)
        return {
            "dataset_split_hash": self.config.dataset_split_hash
            or split_content_hash,
            "dataset_content_hash": split_content_hash,
            "verifier_hash": self.config.verifier_hash
            or str(getattr(self.evaluator, "verifier_hash", "retrieval-verifier-v1")),
            "evaluator_hash": self.config.evaluator_hash
            or str(
                getattr(
                    self.evaluator,
                    "evaluator_hash",
                    f"{evaluator_type.__module__}.{evaluator_type.__qualname__}",
                )
            ),
            "metric_hash": self.config.metric_hash or "drcik-point-metrics-v1",
            "mutation_model_hash": self.config.mutation_model_hash
            or f"{llm_type.__module__}.{llm_type.__qualname__}",
            "metric_cap": self.config.metric_cap,
            "random_seed": self.config.random_seed,
            "generations": self.config.generations,
            "screen_tasks": self.config.screen_tasks,
            "promote": self.config.promote,
            "train_folds": self.config.train_folds,
            "transient_retries": self.config.transient_retries,
            "tolerance": self.config.tolerance,
            "original_parent_fingerprint": parent.fingerprint(),
            "skill_library_hash": _library_hash(self.skill_library),
            "skill_library_authority": (
                _digest({"kind": "no_library"})
                if self.skill_library is None
                else _skill_library_cache_identity(self.skill_library)
            ),
            "harness_hash": self.config.harness_hash
            or _digest({"kind": "no_retrieval_harness"}),
        }

    def _partition_train(
        self, tasks: tuple[ContextTask, ...]
    ) -> tuple[tuple[ContextTask, ...], tuple[tuple[ContextTask, ...], ...]]:
        by_entity: dict[str, list[ContextTask]] = {}
        for task in tasks:
            by_entity.setdefault(task.numeric.entity_name, []).append(task)
        entities = sorted(by_entity)
        random.Random(self.config.random_seed).shuffle(entities)
        if len(entities) <= self.config.train_folds:
            raise RetrievalEvolutionError(
                "Train entity diversity cannot supply the configured fold count "
                "after entity-disjoint screening"
            )

        # Find a deterministic complete-entity subset of exactly eight cases.
        # Keeping the fewest entities for a given total maximizes fold diversity.
        subsets: dict[int, tuple[str, ...]] = {0: ()}
        for entity in entities:
            count = len(by_entity[entity])
            additions: dict[int, tuple[str, ...]] = {}
            for total, selected in tuple(subsets.items()):
                new_total = total + count
                if new_total > self.config.screen_tasks:
                    continue
                candidate = (*selected, entity)
                previous = subsets.get(new_total) or additions.get(new_total)
                if previous is None or len(candidate) < len(previous):
                    additions[new_total] = candidate
            subsets.update(additions)
        screen_entities = subsets.get(self.config.screen_tasks)
        if screen_entities is None or (
            len(entities) - len(screen_entities) < self.config.train_folds
        ):
            raise RetrievalEvolutionError(
                "Train entities cannot form exactly eight entity-disjoint screen cases "
                "while preserving the configured fold count"
            )
        selected = frozenset(screen_entities)
        remaining_entities = [entity for entity in entities if entity not in selected]
        if len(remaining_entities) < self.config.train_folds:
            raise RetrievalEvolutionError(
                "remaining Train entity diversity is below the configured fold count"
            )
        fold_entities: list[list[str]] = [
            [] for _ in range(self.config.train_folds)
        ]
        fold_sizes = [0 for _ in range(self.config.train_folds)]
        for entity in remaining_entities:
            fold_index = min(
                range(self.config.train_folds),
                key=lambda index: (fold_sizes[index], index),
            )
            fold_entities[fold_index].append(entity)
            fold_sizes[fold_index] += len(by_entity[entity])
        folds = tuple(
            tuple(
                task
                for entity in entity_group
                for task in by_entity[entity]
            )
            for entity_group in fold_entities
        )
        if any(not fold for fold in folds):
            raise RetrievalEvolutionError(
                "configured Train folds must all contain at least one entity"
            )
        screen = tuple(
            task
            for entity in screen_entities
            for task in by_entity[entity]
        )
        return screen, folds

    def _run_generation(
        self,
        generation: int,
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> None:
        assert self._current_parent is not None
        parent = self._current_parent
        self._event(
            "generation_started",
            generation=generation,
            parent=parent.version,
            parent_fingerprint=parent.fingerprint(),
        )
        parent_library = self._library_for(parent)
        children, proposal_rejections = self._children_for_generation(
            generation,
            parent,
            parent_library=parent_library,
        )
        parent_screen = self._evaluate_batch(
            parent,
            screen,
            stage=f"g{generation}_parent_screen_train",
            readonly=False,
            library=parent_library,
        )
        screen_evaluations: dict[str, RetrievalEvaluation] = {
            parent.version: parent_screen
        }
        rejection_reasons = dict(proposal_rejections)
        eligible: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
        for scope, child in children:
            child_library = self._library_for(child, source=parent_library)
            try:
                evaluation = self._evaluate_batch(
                    child,
                    screen,
                    stage=f"g{generation}_child_{scope}_screen_train",
                    readonly=False,
                    library=child_library,
                )
            except TransientLLMError:
                raise
            except RetrievalForecastingFailure as error:
                rejection_reasons[child.version] = error.rejection_reason
                self._event(
                    "screen_completed",
                    generation=generation,
                    child=child.version,
                    valid=False,
                    reason=rejection_reasons[child.version],
                )
                continue
            screen_evaluations[child.version] = evaluation
            failures = evaluation.gate_failures(
                parent_screen,
                self.config.tolerance,
                require_strict_contextual_gain=False,
            )
            if failures:
                rejection_reasons[child.version] = "screen_gate:" + ",".join(failures)
            else:
                eligible.append((scope, child))
            self._event(
                "screen_completed",
                generation=generation,
                child=child.version,
                valid=not failures,
                rejection_reasons=list(failures),
            )

        eligible.sort(
            key=lambda item: (
                screen_evaluations[item[1].version].rank_key,
                item[1].fingerprint(),
            )
        )
        promoted = eligible[: self.config.promote]
        promoted_versions = tuple(child.version for _scope, child in promoted)
        for _scope, child in eligible[self.config.promote :]:
            rejection_reasons[child.version] = "screen_rank:not_promoted"

        parent_fold_evaluations = tuple(
            self._evaluate_batch(
                parent,
                fold,
                stage=f"g{generation}_parent_train_fold_{index}",
                readonly=False,
                library=parent_library,
            )
            for index, fold in enumerate(remaining_folds)
        )
        parent_train = combine_retrieval_evaluations(
            parent.version, (parent_screen, *parent_fold_evaluations)
        )
        train_evaluations: dict[str, RetrievalEvaluation] = {
            parent.version: parent_train
        }
        full_candidates: list[RetrievalGenome] = []
        for _scope, child in promoted:
            child_library = self._library_for(child)
            child_folds: list[RetrievalEvaluation] = []
            try:
                for index, fold in enumerate(remaining_folds):
                    child_folds.append(
                        self._evaluate_batch(
                            child,
                            fold,
                            stage=f"g{generation}_child_train_fold_{index}",
                            readonly=False,
                            library=child_library,
                        )
                    )
            except TransientLLMError:
                raise
            except RetrievalForecastingFailure as error:
                rejection_reasons[child.version] = error.rejection_reason
                continue
            evaluation = combine_retrieval_evaluations(
                child.version,
                (screen_evaluations[child.version], *child_folds),
            )
            train_evaluations[child.version] = evaluation
            fold_safe = all(
                candidate_fold.pareto_safe(parent_fold, self.config.tolerance)
                for candidate_fold, parent_fold in zip(
                    child_folds, parent_fold_evaluations, strict=True
                )
            )
            failures = evaluation.gate_failures(
                parent_train,
                self.config.tolerance,
                require_strict_contextual_gain=True,
            )
            if not fold_safe:
                rejection_reasons[child.version] = "train_fold_vector:not_pareto_safe"
            elif failures:
                rejection_reasons[child.version] = "train_gate:" + ",".join(failures)
            else:
                full_candidates.append(child)

        winner = (
            min(
                full_candidates,
                key=lambda item: (
                    train_evaluations[item.version].rank_key,
                    item.fingerprint(),
                ),
            )
            if full_candidates
            else parent
        )
        for child in full_candidates:
            if child is not winner:
                rejection_reasons[child.version] = "train_rank:not_selected"

        pending_rows = (
            self._pending_children.get("children")
            if isinstance(self._pending_children, Mapping)
            else None
        )
        if not isinstance(pending_rows, list) or len(pending_rows) != 3:
            raise RetrievalEvolutionError(
                "generation must retain exactly three scoped child slots"
            )
        slots = tuple(
            next(
                row
                for row in pending_rows
                if isinstance(row, Mapping) and row.get("scope") == scope
            )
            for scope in CHILD_SCOPES
        )
        if any(
            type(row["version"]) is not str
            or type(row["fingerprint"]) is not str
            for row in slots
        ):
            raise RetrievalCheckpointError("pending Child slot types are invalid")
        child_versions = tuple(row["version"] for row in slots)
        child_fingerprints = tuple(row["fingerprint"] for row in slots)
        record = RetrievalGenerationTrace(
            generation=generation,
            parent_version=parent.version,
            parent_fingerprint=parent.fingerprint(),
            child_versions=child_versions,
            child_fingerprints=child_fingerprints,
            child_scopes=CHILD_SCOPES,
            child_proposals=tuple(dict(row["proposal"]) for row in slots),
            screen_task_ids=tuple(task.numeric.task_id for task in screen),
            fold_entities=tuple(
                tuple(sorted({task.numeric.entity_name for task in fold}))
                for fold in remaining_folds
            ),
            promoted_fingerprints=tuple(
                child.fingerprint() for _scope, child in promoted
            ),
            train_winner_version=winner.version,
            train_winner_fingerprint=winner.fingerprint(),
            rejection_reasons=rejection_reasons,
            screen_summaries={
                version: evaluation.summary()
                for version, evaluation in screen_evaluations.items()
            },
            train_summaries={
                version: evaluation.summary()
                for version, evaluation in train_evaluations.items()
            },
        )
        self._generations.append(record)
        self._current_parent = winner
        self._next_generation = generation + 1
        self._pending_children = None
        self._event(
            "generation_completed",
            generation=generation,
            parent=parent.version,
            train_winner=winner.version,
            promoted=list(promoted_versions),
            rejection_reasons=rejection_reasons,
        )
        self._save_checkpoint(status="running", result=None)

    def _children_for_generation(
        self,
        generation: int,
        parent: RetrievalGenome,
        *,
        parent_library: RetrievalSkillLibrary | None,
    ) -> tuple[list[tuple[RetrievalChildScope, RetrievalGenome]], dict[str, str]]:
        pending = self._pending_children
        if pending is not None:
            if (
                pending.get("generation") != generation
                or pending.get("parent_fingerprint") != parent.fingerprint()
            ):
                raise RetrievalCheckpointError("pending child checkpoint does not match Parent")
            raw_children = pending.get("children")
            if not isinstance(raw_children, list):
                raise RetrievalCheckpointError("invalid pending child checkpoint")
        else:
            raw_children = []
            self._pending_children = {
                "generation": generation,
                "parent_fingerprint": parent.fingerprint(),
                "children": raw_children,
            }

        children: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
        rejections: dict[str, str] = {}
        for index, scope in enumerate(CHILD_SCOPES, start=1):
            assert self._original_parent is not None
            version_number = (
                int(self._original_parent.version[1:]) + generation * 3 + index
            )
            version = f"v{version_number:03d}"
            existing = next(
                (
                    item
                    for item in raw_children
                    if isinstance(item, Mapping) and item.get("scope") == scope
                ),
                None,
            )
            if existing is None:
                proposal = self._request_child(
                    parent,
                    scope,
                    version,
                    generation,
                    skill_library=parent_library,
                )
                child = parse_scoped_child(
                    parent,
                    proposal,
                    scope=scope,
                    skill_library=parent_library,
                )
                if child is None or child.version != version:
                    raw_fingerprint = _digest(
                        {
                            "scope": scope,
                            "version": version,
                            "proposal": proposal,
                        }
                    )
                    raw_children.append(
                        {
                            "scope": scope,
                            "valid": False,
                            "version": version,
                            "fingerprint": raw_fingerprint,
                            "proposal": dict(proposal),
                        }
                    )
                    self._all_child_fingerprints.append(raw_fingerprint)
                    rejections[version] = "invalid_scoped_candidate"
                    self._event(
                        "child_proposed",
                        generation=generation,
                        scope=scope,
                        version=version,
                        fingerprint=raw_fingerprint,
                        valid=False,
                    )
                    self._save_checkpoint(status="running", result=None)
                    continue
                existing = {
                    "scope": scope,
                    "valid": True,
                    "version": child.version,
                    "fingerprint": child.fingerprint(),
                    "proposal": child.to_payload(),
                }
                raw_children.append(existing)
                self._all_child_fingerprints.append(child.fingerprint())
                self._event(
                    "child_proposed",
                    generation=generation,
                    scope=scope,
                    version=child.version,
                    fingerprint=child.fingerprint(),
                    valid=True,
                )
                self._save_checkpoint(status="running", result=None)
            if existing.get("valid") is False:
                rejections[version] = "invalid_scoped_candidate"
                continue
            proposal = existing.get("proposal")
            if not isinstance(proposal, Mapping):
                raise RetrievalCheckpointError("pending child has no complete proposal")
            child = parse_scoped_child(
                parent,
                proposal,
                scope=scope,
                skill_library=parent_library,
            )
            if (
                child is None
                or child.version != version
                or child.fingerprint() != existing.get("fingerprint")
            ):
                raise RetrievalCheckpointError("pending child fingerprint mismatch")
            children.append((scope, child))
        return children, rejections

    def _request_child(
        self,
        parent: RetrievalGenome,
        scope: RetrievalChildScope,
        version: str,
        generation: int,
        *,
        skill_library: RetrievalSkillLibrary | None,
    ) -> Mapping[str, object]:
        system = (
            "Return one complete typed Retrieval Genome JSON object. "
            "Obey the supplied immutable scope and host-owned schema."
        )
        payload = {
            "schema_version": 1,
            "generation": generation,
            "scope": scope,
            "required_version": version,
            "required_parent": parent.version,
            "mutable_fields": sorted(
                _PRIMARY_SCOPE_FIELDS[scope] | {"active_skill_ids"}
            ),
            "owned_skill_stage": _SCOPE_SKILL_STAGE[scope],
            "active_skill_catalog": [
                {
                    "skill_id": skill.skill_id,
                    "stage": skill.stage,
                    "applicability": skill.applicability.to_payload(),
                }
                for skill in (
                    () if skill_library is None else skill_library.active_skills()
                )
            ],
            "parent_genome": parent.to_payload(),
        }
        attempts = self.config.transient_retries + 1
        for attempt in range(attempts):
            try:
                response = self.mutation_llm.complete(
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }
                    ],
                    temperature=0.0,
                )
                try:
                    return parse_json_object(response.text)
                except JsonExtractionError as error:
                    del error
                    self._event(
                        "mutation_response_invalid",
                        generation=generation,
                        scope=scope,
                        error="invalid_mutation_response",
                    )
                    return {
                        "invalid_mutation_response": "invalid_mutation_response"
                    }
            except TransientLLMError as error:
                if attempt + 1 >= attempts:
                    self._event(
                        "transient_exhausted",
                        operation="mutation",
                        generation=generation,
                        scope=scope,
                        error="transient_model_failure",
                    )
                    self._save_checkpoint(status="running", result=None)
                    raise
                self._event(
                    "transient_retry",
                    operation="mutation",
                    generation=generation,
                    scope=scope,
                    attempt=attempt + 1,
                )
                self._save_checkpoint(status="running", result=None)

        raise AssertionError("unreachable mutation retry loop")

    def _evaluate_batch(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        readonly: bool,
        library: RetrievalSkillLibrary | None,
    ) -> RetrievalEvaluation:
        if not tasks:
            raise RetrievalEvolutionError("evaluation batches cannot be empty")
        verifier_hash = str(self._scientific_inputs["verifier_hash"])
        evaluator_hash = str(self._scientific_inputs["evaluator_hash"])
        metric_hash = str(self._scientific_inputs["metric_hash"])
        harness_hash = str(self._scientific_inputs["harness_hash"])
        scientific_inputs_hash = _digest(self._scientific_inputs)
        task_keys = tuple(
            build_inference_cache_key(
                task,
                genome,
                library,
                verifier_hash=verifier_hash,
                evaluator_hash=evaluator_hash,
                metric_hash=metric_hash,
                metric_cap=self.config.metric_cap,
                harness_hash=harness_hash,
                scientific_inputs_hash=scientific_inputs_hash,
            )
            for task in tasks
        )
        batch_key = _digest(
            {
                "stage": stage,
                "task_keys": [item.digest() for item in task_keys],
                "readonly": readonly,
            }
        )
        terminal = self._terminal_outcomes.get(batch_key)
        if terminal is not None:
            raise RetrievalForecastingFailure(
                cast(str, terminal["error_type"]),
                cast(str, terminal["error_message"]),
            )
        cached = self._evaluation_cache.get(batch_key)
        if cached is not None:
            self._event(
                "evaluation_cache_hit",
                stage=stage,
                genome=genome.version,
                task_count=len(tasks),
            )
            return cached

        kwargs: dict[str, object] = {
            "stage": stage,
            "skill_library": library,
            "harness_factory": self.harness_factory,
            "persist": False,
            "writers_enabled": False,
            "evolver_enabled": False,
            "cache_keys": task_keys,
            "metric_cap": self.config.metric_cap,
        }
        attempts = self.config.transient_retries + 1
        for attempt in range(attempts):
            try:
                result = self._invoke_evaluator(genome, tasks, kwargs)
                break
            except TransientLLMError as error:
                if attempt + 1 >= attempts:
                    self._event(
                        "transient_exhausted",
                        operation="evaluation",
                        stage=stage,
                        genome=genome.version,
                        error="transient_model_failure",
                    )
                    self._save_checkpoint(status="running", result=None)
                    raise
                self._event(
                    "transient_retry",
                    operation="evaluation",
                    stage=stage,
                    genome=genome.version,
                    attempt=attempt + 1,
                )
                self._save_checkpoint(status="running", result=None)
            except (TypeError, RetrievalSkillError):
                raise
            except Exception as error:
                failure = RetrievalForecastingFailure(
                    (
                        error.error_type
                        if isinstance(error, RetrievalForecastingFailure)
                        else "EvaluatorExecutionFailure"
                    )
                )
                self._record_terminal_outcome(
                    batch_key,
                    genome,
                    tasks,
                    task_keys,
                    stage,
                    failure,
                )
                raise failure from None
        else:
            raise AssertionError("unreachable evaluator retry loop")
        try:
            result = self._validate_evaluator_result(genome, tasks, result)
        except Exception:
            failure = RetrievalForecastingFailure("InvalidEvaluatorResult")
            self._record_terminal_outcome(
                batch_key,
                genome,
                tasks,
                task_keys,
                stage,
                failure,
            )
            raise failure from None
        self._evaluation_cache[batch_key] = result
        self._cache_records[batch_key] = {
            "stage": stage,
            "readonly": readonly,
            "genome": genome.to_payload(),
            "genome_fingerprint": genome.fingerprint(),
            "skill_library_sha256": task_keys[0].skill_library_sha256,
            "skill_authority_sha256": task_keys[0].skill_authority_sha256,
            "task_ids": [task.numeric.task_id for task in tasks],
            "task_cache_keys": [item.digest() for item in task_keys],
            "evaluation_sha256": _digest(result.to_payload()),
            "evaluation": result.to_payload(),
        }
        self._save_checkpoint(status="running", result=None)
        return result

    @staticmethod
    def _validate_evaluator_result(
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        result: object,
    ) -> RetrievalEvaluation:
        if not isinstance(result, RetrievalEvaluation):
            raise RetrievalEvolutionError("trusted evaluator returned an untyped result")
        if result.version != genome.version:
            raise RetrievalEvolutionError(
                "trusted evaluator returned the wrong Genome version"
            )
        if result.task_count != len(tasks):
            raise RetrievalEvolutionError(
                "trusted evaluator did not cover the complete task batch"
            )
        expected_task_ids = tuple(task.numeric.task_id for task in tasks)
        trace_task_ids = tuple(trace.get("task_id") for trace in result.task_traces)
        if (
            len(trace_task_ids) != len(expected_task_ids)
            or any(type(task_id) is not str for task_id in trace_task_ids)
            or set(trace_task_ids) != set(expected_task_ids)
        ):
            raise RetrievalEvolutionError(
                "trusted evaluator task traces do not provide exact task coverage"
            )
        trace_smae = tuple(
            _finite(trace.get("final_smae"), "task trace final_smae")
            for trace in result.task_traces
        )
        return replace(
            result,
            p90_smae=linear_quantile(trace_smae, 0.90),
            p95_smae=linear_quantile(trace_smae, 0.95),
        )

    def _record_terminal_outcome(
        self,
        batch_key: str,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        task_keys: tuple[RetrievalInferenceCacheKey, ...],
        stage: str,
        failure: RetrievalForecastingFailure,
    ) -> None:
        core: dict[str, object] = {
            "stage": stage,
            "readonly": stage in {"parent_dev", "child_dev"},
            "genome": genome.to_payload(),
            "genome_fingerprint": genome.fingerprint(),
            "skill_library_sha256": task_keys[0].skill_library_sha256,
            "skill_authority_sha256": task_keys[0].skill_authority_sha256,
            "task_ids": [task.numeric.task_id for task in tasks],
            "task_cache_keys": [item.digest() for item in task_keys],
            "error_type": failure.error_type,
            "error_message": failure.original_message,
        }
        self._terminal_outcomes[batch_key] = {
            **core,
            "outcome_sha256": _digest(core),
        }
        self._event(
            "forecasting_failure_completed",
            stage=stage,
            genome=genome.version,
            error_type=failure.error_type,
            error=failure.original_message,
        )
        self._save_checkpoint(status="running", result=None)

    def _invoke_evaluator(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        kwargs: dict[str, object],
    ) -> RetrievalEvaluation:
        return self.evaluator.evaluate(genome, tasks, **kwargs)

    def _library_for(
        self,
        genome: RetrievalGenome,
        *,
        source: RetrievalSkillLibrary | None = None,
    ) -> RetrievalSkillLibrary | None:
        fingerprint = genome.fingerprint()
        if fingerprint not in self._candidate_libraries:
            self._candidate_libraries[fingerprint] = self._clone_library(
                source if source is not None else self.skill_library
            )
        return self._candidate_libraries[fingerprint]

    def _readonly_library(
        self, genome: RetrievalGenome
    ) -> RetrievalSkillLibrary | None:
        return self._clone_library(self._library_for(genome))

    @staticmethod
    def _clone_library(
        library: RetrievalSkillLibrary | None,
    ) -> RetrievalSkillLibrary | None:
        return (
            None
            if library is None
            else library.clone(persist=False, read_only=True)
        )

    def _event(self, kind: str, **payload: object) -> None:
        self._trace.append({"kind": kind, **payload})

    def _checkpoint_path(self) -> Path | None:
        value = self.config.checkpoint_path
        return value if isinstance(value, Path) else None

    def _save_checkpoint(
        self,
        *,
        status: Literal["running", "complete"],
        result: RetrievalEvolutionResult | None,
    ) -> None:
        path = self._checkpoint_path()
        if path is None or self._original_parent is None or self._current_parent is None:
            return
        task_completion = [
            {key: value for key, value in record.items() if key != "evaluation"}
            | {"cache_key": cache_key}
            for cache_key, record in sorted(self._cache_records.items())
        ]
        payload = {
            "schema_version": RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION,
            "status": status,
            "scientific_inputs": self._scientific_inputs,
            "original_parent": self._original_parent.to_payload(),
            "original_parent_fingerprint": self._original_parent.fingerprint(),
            "current_parent": self._current_parent.to_payload(),
            "current_parent_fingerprint": self._current_parent.fingerprint(),
            "next_generation": self._next_generation,
            "pending_children": self._pending_children,
            "child_fingerprints": list(self._all_child_fingerprints),
            "task_completion": task_completion,
            "evaluation_cache": self._cache_records,
            "terminal_outcomes": self._terminal_outcomes,
            "generations": [item.to_payload() for item in self._generations],
            "trace": self._trace,
            "result": result.to_payload() if result is not None else None,
        }
        if self._checkpoint_payload_validator is not None:
            self._checkpoint_payload_validator(payload)
        encoded = (
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        path, parent_descriptor = _open_checkpoint_parent(path, create=True)
        temporary: str | None = None
        authority_token: _CheckpointAuthorityToken | None = None
        checkpoint_sha256 = hashlib.sha256(encoded).hexdigest()
        try:
            parent_metadata = os.fstat(parent_descriptor)
            if (
                self._checkpoint_parent_identity is not None
                and (parent_metadata.st_dev, parent_metadata.st_ino)
                != self._checkpoint_parent_identity
            ):
                raise RetrievalCheckpointError(
                    "Retrieval evolution checkpoint parent identity changed after preflight"
                )
            _revalidate_checkpoint_parent(path, parent_descriptor)
            try:
                current_identity, current_encoded = (
                    _read_checkpoint_entry_snapshot(
                        parent_descriptor, path.name
                    )
                )
            except FileNotFoundError:
                current_identity, current_encoded = None, None
            if self._checkpoint_file_sha256 is None and current_identity is not None:
                raise RetrievalCheckpointError(
                    "Retrieval evolution checkpoint already exists without a loaded digest"
                )
            if self._checkpoint_file_sha256 is not None:
                if (
                    current_identity != self._checkpoint_file_identity
                    or current_encoded is None
                    or hashlib.sha256(current_encoded).hexdigest()
                    != self._checkpoint_file_sha256
                ):
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint changed before guarded replace"
                    )
            temporary = _unique_checkpoint_temporary(
                parent_descriptor, path.name, encoded
            )
            staged_identity, staged_encoded = _read_checkpoint_entry_snapshot(
                parent_descriptor, temporary
            )
            if staged_encoded != encoded:
                raise RetrievalCheckpointError(
                    "Retrieval evolution checkpoint staging bytes changed"
                )
            if self._checkpoint_authority is not None:
                authority_token = self._checkpoint_authority.prepare(
                    checkpoint_sha256,
                    checkpoint_identity=staged_identity,
                )
            _revalidate_checkpoint_parent(path, parent_descriptor)
            if self._checkpoint_file_identity is not None:
                try:
                    quarantine = _move_artifact_entry_to_quarantine(
                        parent_descriptor, path.name
                    )
                except Exception as error:
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint quarantine failed"
                    ) from error
                if quarantine is None:
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint disappeared during guarded replace"
                    )
                moved_identity, _moved_encoded = _read_checkpoint_entry_snapshot(
                    parent_descriptor, quarantine
                )
                if moved_identity != self._checkpoint_file_identity:
                    try:
                        _restore_quarantined_artifact_entry(
                            parent_descriptor,
                            quarantine,
                            path.name,
                            expected_identity=moved_identity,
                        )
                    except Exception as error:
                        raise RetrievalCheckpointError(
                            "Retrieval evolution checkpoint foreign replacement was retained"
                        ) from error
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint identity changed during guarded replace"
                    )
            try:
                _rename_artifact_entry_noreplace(
                    parent_descriptor, temporary, path.name
                )
            except Exception as error:
                try:
                    published_identity, published_encoded = (
                        _read_checkpoint_entry_snapshot(
                            parent_descriptor, path.name
                        )
                    )
                except FileNotFoundError:
                    published_identity, published_encoded = None, None
                if (
                    published_identity != staged_identity
                    or published_encoded != encoded
                ):
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint no-replace publication failed"
                    ) from error
            temporary = None
            _revalidate_checkpoint_parent(path, parent_descriptor)
            published_identity, published_encoded = (
                _read_checkpoint_entry_snapshot(parent_descriptor, path.name)
            )
            if (
                published_identity != staged_identity
                or published_encoded != encoded
            ):
                raise RetrievalCheckpointError(
                    "Retrieval evolution checkpoint publication identity changed"
                )
            os.fsync(parent_descriptor)
            self._checkpoint_file_sha256 = checkpoint_sha256
            self._checkpoint_file_identity = staged_identity
            if self._checkpoint_authority is None:
                self._checkpoint_authority_epoch = _register_evolution_checkpoint(
                    path,
                    checkpoint_sha256,
                )
            else:
                assert authority_token is not None
                self._checkpoint_authority_epoch = (
                    self._checkpoint_authority.commit(authority_token)
                )
                authority_token = None
        except Exception:
            if self._checkpoint_authority is not None and authority_token is not None:
                recovered_epoch = self._checkpoint_authority.abort(authority_token)
                if recovered_epoch is not None:
                    self._checkpoint_file_sha256 = checkpoint_sha256
                    checkpoint_snapshot = self._checkpoint_authority._checkpoint_snapshot()
                    self._checkpoint_file_identity = (
                        checkpoint_snapshot[0]
                        if checkpoint_snapshot is not None
                        else None
                    )
                    self._checkpoint_authority_epoch = recovered_epoch
            raise
        finally:
            # Any unpublished unique name is deliberately retained on failure:
            # after descriptor ownership ends, name-based cleanup is unsafe.
            os.close(parent_descriptor)

    def _load_checkpoint(
        self,
        parent: RetrievalGenome,
        train: tuple[ContextTask, ...],
        dev: tuple[ContextTask, ...],
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> RetrievalEvolutionResult | None:
        path = self._checkpoint_path()
        if path is None:
            self._original_parent = parent
            self._current_parent = parent
            return None
        path = _safe_checkpoint_path(path)
        if not self.config.resume or not _safe_checkpoint_exists(path):
            self._original_parent = parent
            self._current_parent = parent
            return None
        try:
            safe, parent_descriptor = _open_checkpoint_parent(path, create=False)
            try:
                parent_metadata = os.fstat(parent_descriptor)
                if (
                    self._checkpoint_parent_identity is not None
                    and (parent_metadata.st_dev, parent_metadata.st_ino)
                    != self._checkpoint_parent_identity
                ):
                    raise RetrievalCheckpointError(
                        "Retrieval evolution checkpoint parent identity changed after preflight"
                    )
                _revalidate_checkpoint_parent(safe, parent_descriptor)
                self._checkpoint_file_identity, encoded = (
                    _read_checkpoint_entry_snapshot(
                        parent_descriptor, safe.name
                    )
                )
                _revalidate_checkpoint_parent(safe, parent_descriptor)
            finally:
                os.close(parent_descriptor)
            raw = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalCheckpointError("invalid Retrieval evolution checkpoint") from error
        self._checkpoint_file_sha256 = hashlib.sha256(encoded).hexdigest()
        self._checkpoint_authority_epoch = _require_evolution_checkpoint(
            path,
            self._checkpoint_file_sha256,
        )
        expected = {
            "schema_version",
            "status",
            "scientific_inputs",
            "original_parent",
            "original_parent_fingerprint",
            "current_parent",
            "current_parent_fingerprint",
            "next_generation",
            "pending_children",
            "child_fingerprints",
            "task_completion",
            "evaluation_cache",
            "terminal_outcomes",
            "generations",
            "trace",
            "result",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise RetrievalCheckpointError("invalid Retrieval evolution checkpoint schema")
        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"]
            != RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION
        ):
            raise RetrievalCheckpointError("unsupported Retrieval evolution checkpoint schema")
        if type(raw["scientific_inputs"]) is not dict:
            raise RetrievalCheckpointError("invalid checkpoint scientific input types")
        if _canonical_json(raw["scientific_inputs"]) != _canonical_json(
            self._scientific_inputs
        ):
            raise RetrievalCheckpointError("checkpoint scientific inputs do not match this run")
        try:
            original = RetrievalGenome.from_payload(raw["original_parent"])
            current = RetrievalGenome.from_payload(raw["current_parent"])
        except (RetrievalPolicyError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid checkpoint Parent Genome") from error
        if (
            type(raw["original_parent_fingerprint"]) is not str
            or type(raw["current_parent_fingerprint"]) is not str
            or type(raw["status"]) is not str
            or raw["status"] not in {"running", "complete"}
            or original.fingerprint() != raw["original_parent_fingerprint"]
            or current.fingerprint() != raw["current_parent_fingerprint"]
            or original.fingerprint() != parent.fingerprint()
        ):
            raise RetrievalCheckpointError("checkpoint Parent fingerprint mismatch")
        try:
            if (
                type(raw["generations"]) is not list
                or any(type(item) is not dict for item in raw["generations"])
                or type(raw["trace"]) is not list
                or any(type(item) is not dict for item in raw["trace"])
                or type(raw["evaluation_cache"]) is not dict
                or any(
                    type(key) is not str or type(value) is not dict
                    for key, value in raw["evaluation_cache"].items()
                )
                or type(raw["terminal_outcomes"]) is not dict
                or any(
                    type(key) is not str or type(value) is not dict
                    for key, value in raw["terminal_outcomes"].items()
                )
                or type(raw["child_fingerprints"]) is not list
                or any(
                    type(item) is not str for item in raw["child_fingerprints"]
                )
                or type(raw["task_completion"]) is not list
                or any(type(item) is not dict for item in raw["task_completion"])
                or type(raw["next_generation"]) is not int
            ):
                raise TypeError
            generations = [
                RetrievalGenerationTrace.from_payload(item)
                for item in raw["generations"]
            ]
            trace = [dict(item) for item in raw["trace"]]
            cache_records = {
                key: dict(value)
                for key, value in raw["evaluation_cache"].items()
            }
            terminal_outcomes = {
                key: dict(value)
                for key, value in raw["terminal_outcomes"].items()
            }
            child_fingerprints = list(raw["child_fingerprints"])
            task_completion = list(raw["task_completion"])
            next_generation = raw["next_generation"]
        except (AttributeError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid checkpoint execution state") from error
        expected_completion = [
            {key: value for key, value in record.items() if key != "evaluation"}
            | {"cache_key": cache_key}
            for cache_key, record in sorted(cache_records.items())
        ]
        if task_completion != expected_completion:
            raise RetrievalCheckpointError("checkpoint task completion binding mismatch")
        flattened = [
            fingerprint
            for generation in generations
            for fingerprint in generation.child_fingerprints
        ]
        pending = raw["pending_children"]
        if pending is not None:
            self._validate_pending_checkpoint(
                pending,
                original=original,
                current=current,
                next_generation=next_generation,
            )
            flattened.extend(
                item["fingerprint"] for item in pending["children"]
            )
        if flattened != child_fingerprints:
            raise RetrievalCheckpointError("checkpoint Child fingerprint binding mismatch")
        if (
            next_generation != len(generations)
            or tuple(item.generation for item in generations)
            != tuple(range(next_generation))
        ):
            raise RetrievalCheckpointError(
                "checkpoint generation cursor does not match completed records"
            )
        expected_current_fingerprint = (
            original.fingerprint()
            if not generations
            else generations[-1].train_winner_fingerprint
        )
        if current.fingerprint() != expected_current_fingerprint:
            raise RetrievalCheckpointError(
                "checkpoint current Parent does not match the generation winner"
            )
        status = raw["status"]
        if status == "complete" and (
            next_generation != self.config.generations or pending is not None
        ):
            raise RetrievalCheckpointError(
                "completed checkpoint has an invalid generation cursor"
            )
        if status == "running" and pending is not None and (
            pending.get("generation") != next_generation
            or pending.get("parent_fingerprint") != current.fingerprint()
        ):
            raise RetrievalCheckpointError(
                "pending checkpoint generation does not match the current Parent"
            )
        evaluation_cache = self._attest_execution_records(
            cache_records,
            terminal_outcomes,
            screen=screen,
            remaining_folds=remaining_folds,
            dev=dev,
        )
        self._original_parent = original
        self._current_parent = current
        self._next_generation = next_generation
        self._generations = generations
        self._trace = trace
        self._cache_records = cache_records
        self._evaluation_cache = evaluation_cache
        self._terminal_outcomes = terminal_outcomes
        self._pending_children = None if pending is None else dict(pending)
        self._all_child_fingerprints = child_fingerprints
        _, completed_outcomes = self._attest_completed_generations(
            original=original,
            current=current,
            generations=tuple(generations),
            trace=tuple(trace),
            screen=screen,
            remaining_folds=remaining_folds,
        )
        if status == "complete":
            if raw["result"] is None:
                raise RetrievalCheckpointError("completed checkpoint has no result")
            result = RetrievalEvolutionResult.from_payload(raw["result"])
            self._attest_completed_result(
                result,
                original=original,
                current=current,
                generations=tuple(generations),
                trace=tuple(trace),
                screen=screen,
                remaining_folds=remaining_folds,
            )
            return result
        if status != "running" or raw["result"] is not None:
            raise RetrievalCheckpointError("invalid checkpoint status")
        if not 0 <= next_generation <= self.config.generations:
            raise RetrievalCheckpointError("invalid checkpoint generation cursor")
        self._attest_running_stage_cursor(
            original=original,
            current=current,
            pending=pending,
            next_generation=next_generation,
            consumed=completed_outcomes,
            trace=tuple(trace),
            screen=screen,
            remaining_folds=remaining_folds,
        )
        return None

    def _attest_running_stage_cursor(
        self,
        *,
        original: RetrievalGenome,
        current: RetrievalGenome,
        pending: object,
        next_generation: int,
        consumed: set[str],
        trace: tuple[dict[str, object], ...],
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> None:
        all_outcomes = set(self._cache_records) | set(self._terminal_outcomes)

        def finish() -> None:
            if consumed != all_outcomes:
                raise RetrievalCheckpointError(
                    "running checkpoint evaluation stage cursor or coverage is invalid"
                )

        if any(
            event.get("kind") in {
                "dev_completed",
                "release_accepted",
                "release_rejected",
            }
            or (
                event.get("kind") == "generation_completed"
                and event.get("generation") not in range(next_generation)
            )
            for event in trace
        ):
            raise RetrievalCheckpointError(
                "running checkpoint audit stage cursor is invalid"
            )
        if pending is None:
            if next_generation == self.config.generations:
                parent_dev = self._checkpoint_outcome(
                    "parent_dev",
                    original,
                    consumed,
                    required=False,
                )
                if parent_dev is None:
                    finish()
                    return
                if isinstance(parent_dev, RetrievalForecastingFailure):
                    finish()
                    return
                self._checkpoint_outcome(
                    "child_dev",
                    current,
                    consumed,
                    required=False,
                )
            finish()
            return
        if next_generation >= self.config.generations:
            raise RetrievalCheckpointError(
                "running checkpoint cannot start a generation past the configured cursor"
            )
        assert isinstance(pending, dict)
        rows = pending["children"]
        assert isinstance(rows, list)
        if len(rows) < len(CHILD_SCOPES):
            finish()
            return

        parent_library = self._clone_library(self.skill_library)
        children: list[tuple[RetrievalChildScope, RetrievalGenome | None]] = []
        for scope, row in zip(CHILD_SCOPES, rows, strict=True):
            assert isinstance(row, dict)
            child = (
                parse_scoped_child(
                    current,
                    row["proposal"],
                    scope=scope,
                    skill_library=parent_library,
                )
                if row["valid"]
                else None
            )
            children.append((scope, child))

        parent_screen = self._checkpoint_outcome(
            f"g{next_generation}_parent_screen_train",
            current,
            consumed,
            required=False,
        )
        if parent_screen is None or isinstance(
            parent_screen, RetrievalForecastingFailure
        ):
            finish()
            return

        screen_evaluations: dict[str, RetrievalEvaluation] = {
            current.version: parent_screen
        }
        eligible: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
        for scope, child in children:
            if child is None:
                continue
            outcome = self._checkpoint_outcome(
                f"g{next_generation}_child_{scope}_screen_train",
                child,
                consumed,
                required=False,
            )
            if outcome is None:
                finish()
                return
            if isinstance(outcome, RetrievalForecastingFailure):
                continue
            screen_evaluations[child.version] = outcome
            if outcome.pareto_safe(parent_screen, self.config.tolerance):
                eligible.append((scope, child))

        eligible.sort(
            key=lambda item: (
                screen_evaluations[item[1].version].rank_key,
                item[1].fingerprint(),
            )
        )
        promoted = eligible[: self.config.promote]
        for fold_index, _fold in enumerate(remaining_folds):
            outcome = self._checkpoint_outcome(
                f"g{next_generation}_parent_train_fold_{fold_index}",
                current,
                consumed,
                required=False,
            )
            if outcome is None or isinstance(
                outcome, RetrievalForecastingFailure
            ):
                finish()
                return

        for _scope, child in promoted:
            for fold_index, _fold in enumerate(remaining_folds):
                outcome = self._checkpoint_outcome(
                    f"g{next_generation}_child_train_fold_{fold_index}",
                    child,
                    consumed,
                    required=False,
                )
                if outcome is None:
                    finish()
                    return
                if isinstance(outcome, RetrievalForecastingFailure):
                    break
        finish()

    def _stage_tasks(
        self,
        stage: str,
        *,
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
        dev: tuple[ContextTask, ...],
    ) -> tuple[tuple[ContextTask, ...], bool]:
        if stage in {"parent_dev", "child_dev"}:
            return dev, True
        for generation in range(self.config.generations):
            if stage == f"g{generation}_parent_screen_train" or stage in {
                f"g{generation}_child_{scope}_screen_train"
                for scope in CHILD_SCOPES
            }:
                return screen, False
            for index, fold in enumerate(remaining_folds):
                if stage in {
                    f"g{generation}_parent_train_fold_{index}",
                    f"g{generation}_child_train_fold_{index}",
                }:
                    return fold, False
        raise RetrievalCheckpointError(
            f"checkpoint contains an unknown evaluation stage: {stage}"
        )

    def _recomputed_task_keys(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
    ) -> tuple[RetrievalInferenceCacheKey, ...]:
        library = self._clone_library(self.skill_library)
        return tuple(
            build_inference_cache_key(
                task,
                genome,
                library,
                verifier_hash=self._scientific_inputs["verifier_hash"],
                evaluator_hash=self._scientific_inputs["evaluator_hash"],
                metric_hash=self._scientific_inputs["metric_hash"],
                metric_cap=self.config.metric_cap,
                harness_hash=str(self._scientific_inputs["harness_hash"]),
                scientific_inputs_hash=_digest(self._scientific_inputs),
            )
            for task in tasks
        )

    def _attest_execution_records(
        self,
        cache_records: dict[str, dict[str, object]],
        terminal_outcomes: dict[str, dict[str, object]],
        *,
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
        dev: tuple[ContextTask, ...],
    ) -> dict[str, RetrievalEvaluation]:
        evaluation_cache: dict[str, RetrievalEvaluation] = {}
        common_fields = {
            "stage",
            "readonly",
            "genome",
            "genome_fingerprint",
            "skill_library_sha256",
            "skill_authority_sha256",
            "task_ids",
            "task_cache_keys",
        }
        for cache_key, record in cache_records.items():
            if set(record) != common_fields | {
                "evaluation_sha256",
                "evaluation",
            }:
                raise RetrievalCheckpointError("invalid checkpoint evaluation cache")
            evaluation_cache[cache_key] = self._attest_one_execution_record(
                cache_key,
                record,
                terminal=False,
                screen=screen,
                remaining_folds=remaining_folds,
                dev=dev,
            )
        for cache_key, record in terminal_outcomes.items():
            if cache_key in cache_records or set(record) != common_fields | {
                "error_type",
                "error_message",
                "outcome_sha256",
            }:
                raise RetrievalCheckpointError("invalid terminal forecasting outcome")
            self._attest_one_execution_record(
                cache_key,
                record,
                terminal=True,
                screen=screen,
                remaining_folds=remaining_folds,
                dev=dev,
            )
        return evaluation_cache

    def _attest_one_execution_record(
        self,
        cache_key: str,
        record: dict[str, object],
        *,
        terminal: bool,
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
        dev: tuple[ContextTask, ...],
    ) -> RetrievalEvaluation | None:
        if (
            type(cache_key) is not str
            or type(record["stage"]) is not str
            or type(record["readonly"]) is not bool
            or type(record["genome"]) is not dict
            or type(record["genome_fingerprint"]) is not str
            or type(record["skill_library_sha256"]) is not str
            or type(record["skill_authority_sha256"]) is not str
            or type(record["task_ids"]) is not list
            or any(type(item) is not str for item in record["task_ids"])
            or type(record["task_cache_keys"]) is not list
            or any(type(item) is not str for item in record["task_cache_keys"])
        ):
            raise RetrievalCheckpointError("invalid checkpoint execution record types")
        try:
            genome = RetrievalGenome.from_payload(record["genome"])
        except (RetrievalPolicyError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid cached Genome") from error
        if genome.fingerprint() != record["genome_fingerprint"]:
            raise RetrievalCheckpointError("cached Genome fingerprint does not recompute")
        tasks, expected_readonly = self._stage_tasks(
            record["stage"],
            screen=screen,
            remaining_folds=remaining_folds,
            dev=dev,
        )
        expected_task_ids = [task.numeric.task_id for task in tasks]
        if (
            record["readonly"] is not expected_readonly
            or record["task_ids"] != expected_task_ids
        ):
            raise RetrievalCheckpointError(
                "checkpoint evaluation task coverage does not match its schedule"
            )
        task_keys = self._recomputed_task_keys(genome, tasks)
        expected_task_keys = [item.digest() for item in task_keys]
        if (
            record["task_cache_keys"] != expected_task_keys
            or record["skill_library_sha256"]
            != task_keys[0].skill_library_sha256
            or record["skill_authority_sha256"]
            != task_keys[0].skill_authority_sha256
        ):
            raise RetrievalCheckpointError(
                "checkpoint task cache identity does not recompute"
            )
        expected_cache_key = _digest(
            {
                "stage": record["stage"],
                "task_keys": expected_task_keys,
                "readonly": expected_readonly,
            }
        )
        if cache_key != expected_cache_key:
            raise RetrievalCheckpointError("checkpoint batch cache key does not recompute")
        if terminal:
            if (
                type(record["error_type"]) is not str
                or type(record["error_message"]) is not str
                or type(record["outcome_sha256"]) is not str
            ):
                raise RetrievalCheckpointError("invalid terminal outcome types")
            core = {
                key: value
                for key, value in record.items()
                if key != "outcome_sha256"
            }
            if record["outcome_sha256"] != _digest(core):
                raise RetrievalCheckpointError(
                    "terminal forecasting outcome digest does not recompute"
                )
            return None
        if (
            type(record["evaluation_sha256"]) is not str
            or type(record["evaluation"]) is not dict
            or record["evaluation_sha256"] != _digest(record["evaluation"])
        ):
            raise RetrievalCheckpointError(
                "checkpoint evaluation digest does not recompute"
            )
        evaluation = RetrievalEvaluation.from_payload(record["evaluation"])
        try:
            evaluation = self._validate_evaluator_result(genome, tasks, evaluation)
        except RetrievalEvolutionError as error:
            raise RetrievalCheckpointError(
                "checkpoint evaluation coverage is invalid"
            ) from error
        return evaluation

    def _checkpoint_outcome(
        self,
        stage: str,
        genome: RetrievalGenome,
        consumed: set[str],
        *,
        required: bool = True,
    ) -> RetrievalEvaluation | RetrievalForecastingFailure | None:
        fingerprint = genome.fingerprint()
        cache_matches = [
            (key, self._evaluation_cache[key])
            for key, record in self._cache_records.items()
            if record["stage"] == stage
            and record["genome_fingerprint"] == fingerprint
        ]
        terminal_matches = [
            (key, record)
            for key, record in self._terminal_outcomes.items()
            if record["stage"] == stage
            and record["genome_fingerprint"] == fingerprint
        ]
        if len(cache_matches) + len(terminal_matches) > 1:
            raise RetrievalCheckpointError(
                "checkpoint contains duplicate proposal evaluation outcomes"
            )
        if cache_matches:
            key, evaluation = cache_matches[0]
            consumed.add(key)
            return evaluation
        if terminal_matches:
            key, record = terminal_matches[0]
            consumed.add(key)
            return RetrievalForecastingFailure(
                cast(str, record["error_type"]),
                cast(str, record["error_message"]),
            )
        if required:
            raise RetrievalCheckpointError(
                f"checkpoint is missing completed evaluation outcome: {stage}"
            )
        return None

    def _attested_children(
        self,
        generation: RetrievalGenerationTrace,
        parent: RetrievalGenome,
        original: RetrievalGenome,
    ) -> tuple[tuple[RetrievalChildScope, RetrievalGenome | None], ...]:
        parent_library = self._clone_library(self.skill_library)
        children: list[tuple[RetrievalChildScope, RetrievalGenome | None]] = []
        for index, scope in enumerate(CHILD_SCOPES, start=1):
            expected_version = (
                f"v{int(original.version[1:]) + generation.generation * 3 + index:03d}"
            )
            proposal = generation.child_proposals[index - 1]
            child = parse_scoped_child(
                parent,
                proposal,
                scope=scope,
                skill_library=parent_library,
            )
            expected_fingerprint = (
                child.fingerprint()
                if child is not None and child.version == expected_version
                else _digest(
                    {
                        "scope": scope,
                        "version": expected_version,
                        "proposal": proposal,
                    }
                )
            )
            if (
                generation.child_scopes[index - 1] != scope
                or generation.child_versions[index - 1] != expected_version
                or generation.child_fingerprints[index - 1]
                != expected_fingerprint
            ):
                raise RetrievalCheckpointError(
                    "completed generation proposal fingerprint does not recompute"
                )
            if child is not None and child.version != expected_version:
                child = None
            children.append((scope, child))
        return tuple(children)

    def _attest_completed_generations(
        self,
        *,
        original: RetrievalGenome,
        current: RetrievalGenome,
        generations: tuple[RetrievalGenerationTrace, ...],
        trace: tuple[dict[str, object], ...],
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> tuple[RetrievalGenome, set[str]]:
        consumed: set[str] = set()
        replay_parent = original
        for generation_index, record in enumerate(generations):
            if (
                record.generation != generation_index
                or record.parent_version != replay_parent.version
                or record.parent_fingerprint != replay_parent.fingerprint()
                or record.screen_task_ids
                != tuple(task.numeric.task_id for task in screen)
                or record.fold_entities
                != tuple(
                    tuple(sorted({task.numeric.entity_name for task in fold}))
                    for fold in remaining_folds
                )
            ):
                raise RetrievalCheckpointError(
                    "completed generation schedule or Parent binding is invalid"
                )
            children = self._attested_children(record, replay_parent, original)
            parent_screen = self._checkpoint_outcome(
                f"g{generation_index}_parent_screen_train",
                replay_parent,
                consumed,
            )
            if not isinstance(parent_screen, RetrievalEvaluation):
                raise RetrievalCheckpointError(
                    "completed generation Parent screen is not a valid evaluation"
                )
            screen_evaluations: dict[str, RetrievalEvaluation] = {
                replay_parent.version: parent_screen
            }
            rejection_reasons: dict[str, str] = {}
            eligible: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
            for scope, child in children:
                version = record.child_versions[CHILD_SCOPES.index(scope)]
                if child is None:
                    rejection_reasons[version] = "invalid_scoped_candidate"
                    continue
                outcome = self._checkpoint_outcome(
                    f"g{generation_index}_child_{scope}_screen_train",
                    child,
                    consumed,
                )
                if isinstance(outcome, RetrievalForecastingFailure):
                    rejection_reasons[child.version] = outcome.rejection_reason
                    continue
                if not isinstance(outcome, RetrievalEvaluation):
                    raise RetrievalCheckpointError("missing Child screen evaluation")
                screen_evaluations[child.version] = outcome
                failures = outcome.gate_failures(
                    parent_screen,
                    self.config.tolerance,
                    require_strict_contextual_gain=False,
                )
                if failures:
                    rejection_reasons[child.version] = "screen_gate:" + ",".join(
                        failures
                    )
                else:
                    eligible.append((scope, child))
            eligible.sort(
                key=lambda item: (
                    screen_evaluations[item[1].version].rank_key,
                    item[1].fingerprint(),
                )
            )
            promoted = eligible[: self.config.promote]
            for _scope, child in eligible[self.config.promote :]:
                rejection_reasons[child.version] = "screen_rank:not_promoted"
            parent_fold_evaluations: list[RetrievalEvaluation] = []
            for fold_index, _fold in enumerate(remaining_folds):
                outcome = self._checkpoint_outcome(
                    f"g{generation_index}_parent_train_fold_{fold_index}",
                    replay_parent,
                    consumed,
                )
                if not isinstance(outcome, RetrievalEvaluation):
                    raise RetrievalCheckpointError(
                        "completed generation Parent fold is not evaluable"
                    )
                parent_fold_evaluations.append(outcome)
            parent_train = combine_retrieval_evaluations(
                replay_parent.version,
                (parent_screen, *parent_fold_evaluations),
            )
            train_evaluations: dict[str, RetrievalEvaluation] = {
                replay_parent.version: parent_train
            }
            full_candidates: list[RetrievalGenome] = []
            for _scope, child in promoted:
                child_folds: list[RetrievalEvaluation] = []
                terminal: RetrievalForecastingFailure | None = None
                for fold_index, _fold in enumerate(remaining_folds):
                    outcome = self._checkpoint_outcome(
                        f"g{generation_index}_child_train_fold_{fold_index}",
                        child,
                        consumed,
                    )
                    if isinstance(outcome, RetrievalForecastingFailure):
                        terminal = outcome
                        break
                    if not isinstance(outcome, RetrievalEvaluation):
                        raise RetrievalCheckpointError(
                            "completed promoted Child fold is missing"
                        )
                    child_folds.append(outcome)
                if terminal is not None:
                    rejection_reasons[child.version] = terminal.rejection_reason
                    continue
                evaluation = combine_retrieval_evaluations(
                    child.version,
                    (screen_evaluations[child.version], *child_folds),
                )
                train_evaluations[child.version] = evaluation
                fold_safe = all(
                    child_fold.pareto_safe(parent_fold, self.config.tolerance)
                    for child_fold, parent_fold in zip(
                        child_folds,
                        parent_fold_evaluations,
                        strict=True,
                    )
                )
                failures = evaluation.gate_failures(
                    parent_train,
                    self.config.tolerance,
                    require_strict_contextual_gain=True,
                )
                if not fold_safe:
                    rejection_reasons[child.version] = (
                        "train_fold_vector:not_pareto_safe"
                    )
                elif failures:
                    rejection_reasons[child.version] = "train_gate:" + ",".join(
                        failures
                    )
                else:
                    full_candidates.append(child)
            winner = (
                min(
                    full_candidates,
                    key=lambda item: (
                        train_evaluations[item.version].rank_key,
                        item.fingerprint(),
                    ),
                )
                if full_candidates
                else replay_parent
            )
            for child in full_candidates:
                if child.fingerprint() != winner.fingerprint():
                    rejection_reasons[child.version] = "train_rank:not_selected"
            if (
                record.promoted_fingerprints
                != tuple(child.fingerprint() for _scope, child in promoted)
                or record.train_winner_version != winner.version
                or record.train_winner_fingerprint != winner.fingerprint()
                or record.rejection_reasons != rejection_reasons
                or record.screen_summaries
                != {
                    version: evaluation.summary()
                    for version, evaluation in screen_evaluations.items()
                }
                or record.train_summaries
                != {
                    version: evaluation.summary()
                    for version, evaluation in train_evaluations.items()
                }
            ):
                raise RetrievalCheckpointError(
                    "completed generation selection trace does not replay"
                )
            completion_events = [
                event
                for event in trace
                if event.get("kind") == "generation_completed"
                and event.get("generation") == generation_index
            ]
            if len(completion_events) != 1 or completion_events[0] != {
                "kind": "generation_completed",
                "generation": generation_index,
                "parent": replay_parent.version,
                "train_winner": winner.version,
                "promoted": [child.version for _scope, child in promoted],
                "rejection_reasons": rejection_reasons,
            }:
                raise RetrievalCheckpointError(
                    "generation completion audit trace does not replay"
                )
            replay_parent = winner

        if replay_parent.fingerprint() != current.fingerprint():
            raise RetrievalCheckpointError(
                "checkpoint selected Parent does not replay from Train"
            )
        return replay_parent, consumed

    def _attest_completed_result(
        self,
        result: RetrievalEvolutionResult,
        *,
        original: RetrievalGenome,
        current: RetrievalGenome,
        generations: tuple[RetrievalGenerationTrace, ...],
        trace: tuple[dict[str, object], ...],
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> None:
        replay_parent, consumed = self._attest_completed_generations(
            original=original,
            current=current,
            generations=generations,
            trace=trace,
            screen=screen,
            remaining_folds=remaining_folds,
        )
        parent_dev = self._checkpoint_outcome(
            "parent_dev",
            original,
            consumed,
        )
        child_dev_outcome = self._checkpoint_outcome(
            "child_dev",
            replay_parent,
            consumed,
        )
        if not isinstance(parent_dev, RetrievalEvaluation):
            raise RetrievalCheckpointError("completed Parent Dev evaluation is missing")
        if isinstance(child_dev_outcome, RetrievalForecastingFailure):
            child_dev = None
            rejection_reasons = (
                f"child_dev_failure:{child_dev_outcome.error_type}:"
                f"{child_dev_outcome.original_message}",
            )
        elif isinstance(child_dev_outcome, RetrievalEvaluation):
            child_dev = child_dev_outcome
            rejection_reasons = child_dev.gate_failures(
                parent_dev,
                self.config.tolerance,
                require_strict_contextual_gain=True,
            )
        else:
            raise RetrievalCheckpointError("completed Child Dev evaluation is missing")
        accepted = child_dev is not None and not rejection_reasons
        selected = replay_parent if accepted else original
        release = replay_parent if accepted else None
        if (
            result.original_parent != original
            or result.train_winner != replay_parent
            or result.selected_genome != selected
            or result.accepted is not accepted
            or result.acceptance_reasons
            != (("all_dev_gates_passed",) if accepted else ())
            or result.rejection_reasons != tuple(rejection_reasons)
            or result.parent_dev != parent_dev
            or result.child_dev != child_dev
            or result.generations != generations
            or result.trace != trace
            or result.release_genome != release
            or result.release_published
        ):
            raise RetrievalCheckpointError(
                "completed checkpoint result or Dev gates do not replay"
            )
        if len(trace) < 2 or trace[-2:] != (
            {
                "kind": "dev_completed",
                "original_parent": original.version,
                "train_winner": replay_parent.version,
                "accepted": accepted,
                "rejection_reasons": list(rejection_reasons),
            },
            {
                "kind": "release_accepted" if accepted else "release_rejected",
                "genome": replay_parent.version if accepted else original.version,
                "publication_deferred": True,
            },
        ):
            raise RetrievalCheckpointError(
                "completed checkpoint Dev/release audit trace is incomplete"
            )
        if consumed != set(self._cache_records) | set(self._terminal_outcomes):
            raise RetrievalCheckpointError(
                "checkpoint contains execution outcomes outside the replayed schedule"
            )

    def _validate_pending_checkpoint(
        self,
        pending: object,
        *,
        original: RetrievalGenome,
        current: RetrievalGenome,
        next_generation: int,
    ) -> None:
        if type(pending) is not dict or set(pending) != {
            "generation",
            "parent_fingerprint",
            "children",
        }:
            raise RetrievalCheckpointError("invalid pending child state")
        if (
            type(pending["generation"]) is not int
            or pending["generation"] != next_generation
            or type(pending["parent_fingerprint"]) is not str
            or pending["parent_fingerprint"] != current.fingerprint()
            or type(pending["children"]) is not list
            or len(pending["children"]) > len(CHILD_SCOPES)
        ):
            raise RetrievalCheckpointError("invalid pending child generation binding")
        parent_library = self._clone_library(self.skill_library)
        for index, row in enumerate(pending["children"], start=1):
            if type(row) is not dict or set(row) != {
                "scope",
                "valid",
                "version",
                "fingerprint",
                "proposal",
            }:
                raise RetrievalCheckpointError("invalid pending child row schema")
            scope = CHILD_SCOPES[index - 1]
            version = (
                f"v{int(original.version[1:]) + next_generation * 3 + index:03d}"
            )
            if (
                type(row["scope"]) is not str
                or row["scope"] != scope
                or type(row["valid"]) is not bool
                or type(row["version"]) is not str
                or row["version"] != version
                or type(row["fingerprint"]) is not str
                or type(row["proposal"]) is not dict
            ):
                raise RetrievalCheckpointError(
                    "pending child rows must retain canonical A/B/C order and types"
                )
            child = parse_scoped_child(
                current,
                row["proposal"],
                scope=scope,
                skill_library=parent_library,
            )
            if row["valid"]:
                if (
                    child is None
                    or child.version != version
                    or child.fingerprint() != row["fingerprint"]
                ):
                    raise RetrievalCheckpointError(
                        "pending valid child fingerprint does not recompute"
                    )
            else:
                expected_fingerprint = _digest(
                    {
                        "scope": scope,
                        "version": version,
                        "proposal": row["proposal"],
                    }
                )
                if child is not None or row["fingerprint"] != expected_fingerprint:
                    raise RetrievalCheckpointError(
                        "pending invalid child fingerprint does not recompute"
                    )


__all__ = [
    "CHILD_SCOPES",
    "RetrievalCheckpointError",
    "RetrievalEvaluation",
    "RetrievalEvolutionConfig",
    "RetrievalEvolutionEngine",
    "RetrievalEvolutionError",
    "RetrievalEvolutionResult",
    "RetrievalGenerationTrace",
    "RetrievalInferenceCacheKey",
    "build_inference_cache_key",
    "combine_retrieval_evaluations",
    "parse_scoped_child",
]
