"""Host-only active-source publication with sealed evidence and byte rollback."""
from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from pathlib import Path

from common.payload import strict_json_loads

from ..contracts import canonical_v2_bytes, fingerprint_payload
from ..store import _atomic_write, append_jsonl, write_atomic_json, write_once_json
from .contracts import SourceVariantV2
from .meta import SourceValidationV2


class SourceAuthorityError(ValueError):
    """Raised when a Host publication transition is not sealed and coherent."""


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise SourceAuthorityError(f"cannot read {path.name}") from error
    if not isinstance(payload, Mapping) or canonical_v2_bytes(payload) != raw:
        raise SourceAuthorityError(f"{path.name} is not canonical JSON")
    return payload


def _evidence_payload(stage: str, evidence: SourceValidationV2) -> dict[str, object]:
    return {
        "schema_version": 1, "stage": stage,
        "parent_source_sha256": evidence.parent_source_sha256,
        "finalist_source_sha256": evidence.finalist_source_sha256,
        "passed": evidence.passed, "reason": evidence.reason,
        "commitment_sha256": evidence.commitment_sha256,
        "replay_fingerprint": evidence.replay_fingerprint,
        "evaluation_fingerprints": list(evidence.evaluation_fingerprints),
    }


class SourceAuthorityV2:
    """The sole writer of the active source pointer and promotion history."""

    def __init__(self, root: str | Path, seed_source: SourceVariantV2) -> None:
        if not isinstance(seed_source, SourceVariantV2):
            raise TypeError("seed_source must be a SourceVariantV2")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.root / "active_source.json"
        self.sealed = self.root / "sealed"
        self.sealed.mkdir(exist_ok=True)
        self.checkpoint_path = self.root / "checkpoint.json"
        self.history_path = self.root / "promotion_history.jsonl"
        if self.active_path.exists():
            current = self.active_source()
            if current.protocol_fingerprint != seed_source.protocol_fingerprint or current.runtime_fingerprint != seed_source.runtime_fingerprint:
                raise SourceAuthorityError("active source commitments do not match seed")
        else:
            _atomic_write(self.active_path, seed_source.canonical_bytes())

    def active_source(self) -> SourceVariantV2:
        try:
            return SourceVariantV2.from_payload(_read_json(self.active_path))
        except ValueError as error:
            raise SourceAuthorityError("active pointer is invalid") from error

    def begin_canary(self, candidate: SourceVariantV2, evidence: SourceValidationV2) -> None:
        if not isinstance(candidate, SourceVariantV2) or not all(hasattr(evidence, name) for name in ("passed", "parent_source_sha256", "finalist_source_sha256", "reason", "commitment_sha256", "replay_fingerprint", "evaluation_fingerprints")):
            raise TypeError("candidate and evidence must be Source V2 values")
        parent = self.active_source()
        if not evidence.passed or evidence.parent_source_sha256 != parent.fingerprint() or evidence.finalist_source_sha256 != candidate.fingerprint():
            raise SourceAuthorityError("canary requires passed validation for active parent and candidate")
        if self.checkpoint_path.exists():
            raise SourceAuthorityError("a canary is already pending")
        self.stage_candidate(candidate)
        payload = _evidence_payload("validation", evidence)
        identity = fingerprint_payload(payload)
        evidence_path = self.sealed / f"{identity}.json"
        write_once_json(evidence_path, payload)
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != hashlib.sha256(canonical_v2_bytes(payload)).hexdigest():
            raise SourceAuthorityError("sealed evidence reread failed")
        checkpoint = {
            "schema_version": 1, "phase": "canary_pending",
            "candidate_source_sha256": candidate.fingerprint(),
            "parent_source_sha256": parent.fingerprint(), "validation_evidence_sha256": identity,
            "prior_pointer_b64": base64.b64encode(self.active_path.read_bytes()).decode("ascii"),
        }
        write_atomic_json(self.checkpoint_path, checkpoint)

    def finish_canary(self, result: SourceValidationV2) -> str:
        if not all(hasattr(result, name) for name in ("passed", "parent_source_sha256", "finalist_source_sha256", "reason", "commitment_sha256", "replay_fingerprint", "evaluation_fingerprints")):
            raise TypeError("result must be SourceValidationV2")
        if not self.checkpoint_path.exists():
            raise SourceAuthorityError("no canary is pending")
        checkpoint = _read_json(self.checkpoint_path)
        needed = {"schema_version", "phase", "candidate_source_sha256", "parent_source_sha256", "validation_evidence_sha256", "prior_pointer_b64"}
        if set(checkpoint) != needed or checkpoint["phase"] != "canary_pending":
            raise SourceAuthorityError("pending canary checkpoint is invalid")
        candidate_sha = checkpoint["candidate_source_sha256"]
        parent_sha = checkpoint["parent_source_sha256"]
        if result.parent_source_sha256 != parent_sha or result.finalist_source_sha256 != candidate_sha:
            raise SourceAuthorityError("canary evidence does not bind pending sources")
        payload = _evidence_payload("canary", result)
        evidence_sha = fingerprint_payload(payload)
        write_once_json(self.sealed / f"{evidence_sha}.json", payload)
        prior = base64.b64decode(str(checkpoint["prior_pointer_b64"]).encode("ascii"), validate=True)
        if result.passed:
            candidate_payload = _read_json(self.sealed / f"{checkpoint['validation_evidence_sha256']}.json")
            if candidate_payload.get("finalist_source_sha256") != candidate_sha:
                raise SourceAuthorityError("validation evidence pointer mismatch")
            # The Runner owns the source object; its pending candidate is reconstructed
            # from the caller's validated result identity in a companion source file.
            source_payload_path = self.root / "pending_source.json"
            candidate = SourceVariantV2.from_payload(_read_json(source_payload_path))
            if candidate.fingerprint() != candidate_sha:
                raise SourceAuthorityError("pending source does not match canary")
            _atomic_write(self.active_path, candidate.canonical_bytes())
            action = "activate"
            active = candidate_sha
        else:
            _atomic_write(self.active_path, prior)
            action = "rollback"
            active = parent_sha
        append_jsonl(self.history_path, {"schema_version": 1, "action": action,
                                         "parent_source_sha256": parent_sha,
                                         "candidate_source_sha256": candidate_sha,
                                         "evidence_sha256": evidence_sha})
        self.checkpoint_path.unlink()
        return active

    def stage_candidate(self, candidate: SourceVariantV2) -> None:
        """Persist the one candidate whose identity is referenced by a pending canary."""
        write_once_json(self.root / "pending_source.json", candidate.to_payload())


__all__ = ["SourceAuthorityError", "SourceAuthorityV2"]
