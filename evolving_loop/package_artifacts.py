"""Immutable caches, append-only trace artifacts, and exact resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from common.evolution_core.task_feedback import (
    TaskEvidenceProjection,
    TaskFeedbackError,
)
from evolving_loop.package_stage_runner import PackageStageEvidence


_CACHE_LAYERS: frozenset[str] = frozenset({"numerical", "retrieval", "decision"})
_STAGE_ORDER: tuple[str, ...] = (
    "screen8",
    "screen32",
    "build64",
    "calibration16",
    "dev20",
)
_SHA256_CHARS = set("0123456789abcdef")


class PackageArtifactError(ValueError):
    """Raised when an immutable artifact, cache entry, or checkpoint is violated."""


class PackageCacheMissError(RuntimeError):
    """Raised when a required cache entry is absent under ``cache_only``."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(path.parent), delete=False, prefix=".tmp-"
    )
    try:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise PackageArtifactError(f"immutable artifact already exists: {path.name}")
    _atomic_write(path, data)


# --------------------------------------------------------------------------
# immutable inference cache
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageCacheKey:
    layer: Literal["numerical", "retrieval", "decision"]
    task_sha256: str
    candidate_sha256: str
    dependency_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.layer not in _CACHE_LAYERS:
            raise PackageArtifactError("cache key layer is invalid")
        if not _is_sha256(self.task_sha256) or not _is_sha256(self.candidate_sha256):
            raise PackageArtifactError("cache key identities must be canonical SHA-256")
        dependencies = dict(self.dependency_fingerprints)
        if not dependencies or any(
            not isinstance(name, str) or not name or not _is_sha256(value)
            for name, value in dependencies.items()
        ):
            raise PackageArtifactError(
                "cache key dependency fingerprints must be canonical"
            )
        object.__setattr__(
            self,
            "dependency_fingerprints",
            dict(sorted(dependencies.items())),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "layer": self.layer,
            "task_sha256": self.task_sha256,
            "candidate_sha256": self.candidate_sha256,
            "dependency_fingerprints": dict(self.dependency_fingerprints),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_payload())


class PackageInferenceCache:
    """Write-once canonical cache with exact dependency validation."""

    def __init__(self, root: Path, *, cache_only: bool) -> None:
        if type(cache_only) is not bool:
            raise PackageArtifactError("cache_only must be a boolean")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_only = cache_only

    def _entry_path(self, key: PackageCacheKey) -> Path:
        return self.root / key.layer / f"{key.fingerprint}.json"

    def get_or_compute(
        self, key: PackageCacheKey, compute: Callable[[], object]
    ) -> object:
        if not isinstance(key, PackageCacheKey):
            raise PackageArtifactError("cache lookup requires a PackageCacheKey")
        if not callable(compute):
            raise PackageArtifactError("cache lookup requires a compute callback")
        path = self._entry_path(key)
        if path.exists():
            envelope = json.loads(path.read_text("utf-8"))
            if (
                envelope.get("key_sha256") == key.fingerprint
                and _canonical_json(envelope.get("key"))
                == _canonical_json(key.to_payload())
                and envelope.get("value_sha256")
                == hashlib.sha256(_canonical_json(envelope.get("value"))).hexdigest()
            ):
                return envelope["value"]
            raise PackageArtifactError(
                "cache entry does not bind its canonical key/value"
            )
        if self.cache_only:
            raise PackageCacheMissError(
                f"cache-only inference has no entry for {key.layer} {key.fingerprint[:12]}"
            )
        value = compute()
        envelope = {
            "schema_version": 1,
            "key": key.to_payload(),
            "key_sha256": key.fingerprint,
            "value": value,
            "value_sha256": hashlib.sha256(_canonical_json(value)).hexdigest(),
        }
        _write_once(path, _canonical_json(envelope))
        return value


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageCheckpoint:
    schema_version: int
    run_sha256: str
    schedule_sha256: str
    initial_bundle_payload: Mapping[str, object]
    current_bundle_payload: Mapping[str, object]
    completed_steps: tuple[Mapping[str, object], ...]
    candidate_fingerprints: tuple[str, ...]
    cache_fingerprints: tuple[str, ...]
    consumed_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PackageArtifactError("package checkpoint schema must be exactly one")
        if not _is_sha256(self.run_sha256) or not _is_sha256(self.schedule_sha256):
            raise PackageArtifactError(
                "checkpoint identities must be canonical SHA-256"
            )
        if not isinstance(self.initial_bundle_payload, Mapping) or not isinstance(
            self.current_bundle_payload, Mapping
        ):
            raise PackageArtifactError("checkpoint bundle payloads must be mappings")
        object.__setattr__(self, "completed_steps", tuple(self.completed_steps))
        object.__setattr__(
            self, "candidate_fingerprints", tuple(self.candidate_fingerprints)
        )
        object.__setattr__(self, "cache_fingerprints", tuple(self.cache_fingerprints))
        object.__setattr__(self, "consumed_stages", tuple(self.consumed_stages))
        if any(stage not in _STAGE_ORDER for stage in self.consumed_stages):
            raise PackageArtifactError("checkpoint consumed stages must be registered")
        if len(set(self.consumed_stages)) != len(self.consumed_stages):
            raise PackageArtifactError("checkpoint consumed stages must be unique")
        if self.completed_steps:
            last = self.completed_steps[-1]
            if _canonical_json(last.get("accepted_bundle_payload")) != _canonical_json(
                dict(self.current_bundle_payload)
            ):
                raise PackageArtifactError(
                    "checkpoint current bundle must equal the last completed step"
                )

    @property
    def next_coordinate_generation(self) -> int:
        return len(self.completed_steps)

    @property
    def next_stage(self) -> str:
        for stage in _STAGE_ORDER:
            if stage not in self.consumed_stages:
                return stage
        return _STAGE_ORDER[-1]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_sha256": self.run_sha256,
            "schedule_sha256": self.schedule_sha256,
            "initial_bundle_payload": dict(self.initial_bundle_payload),
            "current_bundle_payload": dict(self.current_bundle_payload),
            "completed_steps": [dict(step) for step in self.completed_steps],
            "candidate_fingerprints": list(self.candidate_fingerprints),
            "cache_fingerprints": list(self.cache_fingerprints),
            "consumed_stages": list(self.consumed_stages),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PackageCheckpoint":
        if not isinstance(payload, Mapping):
            raise PackageArtifactError("checkpoint payload must be a mapping")
        return cls(
            schema_version=payload.get("schema_version", 0),
            run_sha256=payload.get("run_sha256", ""),
            schedule_sha256=payload.get("schedule_sha256", ""),
            initial_bundle_payload=payload.get("initial_bundle_payload", {}),
            current_bundle_payload=payload.get("current_bundle_payload", {}),
            completed_steps=tuple(payload.get("completed_steps", ())),
            candidate_fingerprints=tuple(payload.get("candidate_fingerprints", ())),
            cache_fingerprints=tuple(payload.get("cache_fingerprints", ())),
            consumed_stages=tuple(payload.get("consumed_stages", ())),
        )


# --------------------------------------------------------------------------
# append-only artifact store
# --------------------------------------------------------------------------


class PackageArtifactStore:
    """Append-only run directory; also a durable Task-6 stage artifact sink."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "candidate_evidence").mkdir(exist_ok=True)
        (self.output_dir / "accepted_bundles").mkdir(exist_ok=True)
        (self.output_dir / "task_feedback").mkdir(exist_ok=True)
        self._stage_claims: dict[str, dict[str, object]] = {}
        claims_path = self.output_dir / "stage_claims.json"
        if claims_path.exists():
            self._stage_claims = json.loads(claims_path.read_text("utf-8"))

    # -- write-once run identity ----------------------------------------

    def write_run_manifest(self, manifest: Mapping[str, object]) -> None:
        _write_once(
            self.output_dir / "run_manifest.json", _canonical_json(dict(manifest))
        )

    def write_schedule(self, schedule_payload: Mapping[str, object]) -> None:
        _write_once(
            self.output_dir / "group_folds.json",
            _canonical_json(dict(schedule_payload)),
        )

    # -- append-only evidence -----------------------------------------

    def write_candidate_evidence(
        self,
        coordinate: str,
        generation: int,
        proposal_sha256: str,
        payload: Mapping[str, object],
    ) -> None:
        if not _is_sha256(proposal_sha256):
            raise PackageArtifactError("candidate evidence proposal must be canonical")
        name = f"{coordinate}-{generation}-{proposal_sha256}.json"
        _write_once(
            self.output_dir / "candidate_evidence" / name,
            _canonical_json(
                {
                    "coordinate": coordinate,
                    "generation": generation,
                    "proposal_sha256": proposal_sha256,
                    "payload": dict(payload),
                }
            ),
        )

    def append_coordinate_step(self, step_payload: Mapping[str, object]) -> None:
        step = dict(step_payload)
        trace_path = self.output_dir / "coordinate_trace.jsonl"
        existing = (
            trace_path.read_text("utf-8").splitlines() if trace_path.exists() else []
        )
        expected_generation = len(existing)
        if step.get("generation") != expected_generation:
            raise PackageArtifactError(
                "coordinate trace accepts only the next expected generation"
            )
        if existing:
            previous = json.loads(existing[-1])
            if step.get("parent_bytes_sha256") != previous.get("accepted_bytes_sha256"):
                raise PackageArtifactError(
                    "coordinate trace requires the exact preceding Parent fingerprint"
                )
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def publish_accepted_bundle(self, bundle_payload: Mapping[str, object]) -> str:
        data = _canonical_json(dict(bundle_payload))
        bundle_sha256 = hashlib.sha256(data).hexdigest()
        _write_once(
            self.output_dir / "accepted_bundles" / f"{bundle_sha256}.json", data
        )
        return bundle_sha256

    def write_task_feedback(
        self, generation: int, projection: TaskEvidenceProjection
    ) -> None:
        if type(generation) is not int or generation < 1:
            raise PackageArtifactError("task feedback generation must be positive")
        if type(projection) is not TaskEvidenceProjection:
            raise PackageArtifactError(
                "task feedback artifact requires an exact projection"
            )
        envelope = {
            "schema_version": 1,
            "generation": generation,
            "projection_sha256": projection.fingerprint,
            "identity": projection.to_identity_payload(),
        }
        _write_once(
            self.output_dir / "task_feedback" / f"task-feedback-{generation}.json",
            _canonical_json(envelope),
        )

    def load_task_feedback(self, generation: int) -> TaskEvidenceProjection:
        if type(generation) is not int or generation < 1:
            raise PackageArtifactError("task feedback generation must be positive")
        path = self.output_dir / "task_feedback" / f"task-feedback-{generation}.json"
        if not path.is_file():
            raise PackageArtifactError("task feedback artifact is missing")
        payload = json.loads(path.read_text("utf-8"))
        if (
            type(payload) is not dict
            or set(payload)
            != {
                "schema_version",
                "generation",
                "projection_sha256",
                "identity",
            }
            or payload["schema_version"] != 1
            or payload["generation"] != generation
        ):
            raise PackageArtifactError("task feedback artifact schema is invalid")
        try:
            projection = TaskEvidenceProjection.from_identity_payload(
                payload["identity"]
            )
        except TaskFeedbackError as error:
            raise PackageArtifactError("task feedback artifact is invalid") from error
        if payload["projection_sha256"] != projection.fingerprint:
            raise PackageArtifactError("task feedback artifact digest mismatch")
        return projection

    # -- stage claim ledger -----------------------------------------

    def _persist_claims(self) -> None:
        _atomic_write(
            self.output_dir / "stage_claims.json", _canonical_json(self._stage_claims)
        )

    def claim_stage(self, stage_id: str, *, candidate_sha256: str) -> None:
        if not _is_sha256(candidate_sha256):
            raise PackageArtifactError("stage claim candidate must be canonical")
        record = self._stage_claims.get(stage_id)
        if record is not None and record.get("committed"):
            raise PackageArtifactError(f"stage {stage_id} is already consumed")
        self._stage_claims[stage_id] = {
            "candidate_sha256": candidate_sha256,
            "committed": False,
        }
        self._persist_claims()

    def commit_stage(self, stage_id: str, *, evaluation_sha256: str) -> None:
        if not _is_sha256(evaluation_sha256):
            raise PackageArtifactError("stage commit evaluation must be canonical")
        record = self._stage_claims.get(stage_id)
        if record is None:
            raise PackageArtifactError(f"stage {stage_id} was never claimed")
        if record.get("committed"):
            raise PackageArtifactError(f"stage {stage_id} is already consumed")
        record["committed"] = True
        record["evaluation_sha256"] = evaluation_sha256
        self._persist_claims()

    def committed_stage_evaluation(self, stage_id: str) -> str | None:
        record = self._stage_claims.get(stage_id)
        if record is None or not record.get("committed"):
            return None
        return record.get("evaluation_sha256")

    # -- checkpoint --------------------------------------------------

    def write_checkpoint(self, checkpoint: PackageCheckpoint) -> None:
        if not isinstance(checkpoint, PackageCheckpoint):
            raise PackageArtifactError("write_checkpoint requires a PackageCheckpoint")
        _atomic_write(
            self.output_dir / "checkpoint.json",
            _canonical_json(checkpoint.to_payload()),
        )

    def load_checkpoint(
        self, *, expected_run_sha256: str, expected_schedule_sha256: str
    ) -> PackageCheckpoint:
        path = self.output_dir / "checkpoint.json"
        if not path.exists():
            raise PackageArtifactError("no package checkpoint to resume")
        checkpoint = PackageCheckpoint.from_payload(json.loads(path.read_text("utf-8")))
        if checkpoint.run_sha256 != expected_run_sha256:
            raise PackageArtifactError(
                "resume run identity does not match the checkpoint"
            )
        if checkpoint.schedule_sha256 != expected_schedule_sha256:
            raise PackageArtifactError(
                "resume schedule identity does not match the checkpoint"
            )
        trace_path = self.output_dir / "coordinate_trace.jsonl"
        if trace_path.exists():
            recorded = [
                json.loads(line)
                for line in trace_path.read_text("utf-8").splitlines()
                if line
            ]
            if len(recorded) < len(checkpoint.completed_steps):
                raise PackageArtifactError(
                    "checkpoint claims more steps than the trace"
                )
        return checkpoint

    # -- completion ------------------------------------------------

    def complete(
        self,
        final_bundle_payload: Mapping[str, object],
        *,
        accepted_steps: int | None = None,
        rejected_steps: int | None = None,
        formal_run: bool | None = None,
        full_chain_exercised: bool | None = None,
    ) -> str:
        if (self.output_dir / "evaluation_complete.json").exists():
            raise PackageArtifactError("this run directory is already complete")
        for name, value in (
            ("accepted_steps", accepted_steps),
            ("rejected_steps", rejected_steps),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise PackageArtifactError(f"{name} must be a non-negative integer")
        if formal_run is not None and type(formal_run) is not bool:
            raise PackageArtifactError("formal_run must be a boolean")
        if full_chain_exercised is not None and type(full_chain_exercised) is not bool:
            raise PackageArtifactError("full_chain_exercised must be a boolean")
        supplied_summary = (
            accepted_steps is not None,
            rejected_steps is not None,
            formal_run is not None,
        )
        if any(supplied_summary) and not all(supplied_summary):
            raise PackageArtifactError(
                "completion summary fields must be supplied together"
            )
        data = _canonical_json(dict(final_bundle_payload))
        _write_once(self.output_dir / "final_bundle.json", data)
        final_sha256 = hashlib.sha256(data).hexdigest()
        summary = {
            **(
                {
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "formal_run": formal_run,
                }
                if accepted_steps is not None
                and rejected_steps is not None
                and formal_run is not None
                else {}
            )
        }
        _atomic_write(
            self.output_dir / "evaluation_complete.json",
            _canonical_json(
                {
                    "schema_version": 1,
                    "status": (
                        "incomplete_chain"
                        if full_chain_exercised is False
                        else "complete"
                    ),
                    "final_bundle_sha256": final_sha256,
                    **summary,
                    **(
                        {"full_chain_exercised": full_chain_exercised}
                        if full_chain_exercised is not None
                        else {}
                    ),
                    "public_test_accessed": False,
                }
            ),
        )
        return final_sha256

    # -- Task-6 PackageStageArtifactSink adapters -------------------

    def record_candidate_evidence(
        self,
        target: str,
        generation: int,
        proposal_sha256: str,
        payload: Mapping[str, object],
    ) -> None:
        self.write_candidate_evidence(target, generation, proposal_sha256, payload)

    def record_stage_evidence(self, evidence: PackageStageEvidence) -> None:
        if not isinstance(evidence, PackageStageEvidence):
            raise PackageArtifactError("stage evidence must be a PackageStageEvidence")
        path = self.output_dir / "stage_evidence.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence.to_payload(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_acceptance_evidence(
        self, evidence_sha256: str, payload: Mapping[str, object]
    ) -> None:
        if not _is_sha256(evidence_sha256):
            raise PackageArtifactError("acceptance evidence digest must be canonical")
        if _digest(dict(payload)) != evidence_sha256:
            raise PackageArtifactError(
                "acceptance evidence digest does not bind its payload"
            )
        _write_once(
            self.output_dir / "accepted_bundles" / f"evidence-{evidence_sha256}.json",
            _canonical_json(dict(payload)),
        )

    def contains_evidence(self, evidence_sha256: str) -> bool:
        return (
            self.output_dir / "accepted_bundles" / f"evidence-{evidence_sha256}.json"
        ).exists()


__all__ = [
    "PackageArtifactError",
    "PackageArtifactStore",
    "PackageCacheKey",
    "PackageCacheMissError",
    "PackageCheckpoint",
    "PackageInferenceCache",
]
