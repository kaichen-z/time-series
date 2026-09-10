"""Host-only acceptance authority and recoverable automatic Bundle publication.

These objects stay in the trusted kernel process. Proposers receive primitive
Train-only data through the separate sandbox boundary, never this module's
objects. A closed evaluation is issued by this kernel instance and is consumed
once with its live resource reservation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from common.payload import strict_json_loads

from .archive import ArchiveRecord, EvolutionArchive, verify_selected_lineage
from .budget import BudgetLedger, BudgetPlan, ResourceUse, StagePermit
from .bundle import (
    EvolutionBundleV2,
    MutationTarget,
    _seal_acceptance,
    validate_child_scope,
)
from .contracts import (
    KernelProtocolCommitment,
    _freeze_json_value,
    _reject_reserved_feedback_keys,
    _require_exact_schema,
    _require_mapping,
    _strict_json_value,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)
from .store import V2RunStore, write_atomic_json


class KernelAuthorityError(ValueError):
    """A transition or durable authority binding failed verification."""


def _read(path: Path, identity: str | None = None) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
        if not isinstance(payload, dict) or canonical_v2_bytes(payload) != raw:
            raise ValueError("artifact is not canonical JSON")
        if identity is not None and fingerprint_payload(payload) != identity:
            raise ValueError("artifact digest mismatch")
        return payload
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise KernelAuthorityError(f"cannot verify {path.name}: {error}") from error


def _runtime(value: Mapping[str, str]) -> object:
    runtime = _require_mapping(value, "runtime_fingerprints")
    if not runtime or any(not name for name in runtime):
        raise KernelAuthorityError("runtime_fingerprints must be non-empty")
    for name, digest in runtime.items():
        require_sha256(digest, name)
    _reject_reserved_feedback_keys(runtime, field="runtime_fingerprints")
    return _freeze_json_value(runtime)


def _dev(value: Mapping[str, object]) -> object:
    payload = _require_exact_schema(
        value, ("passed", "parent_metrics", "candidate_metrics"), field="dev_comparison"
    )
    if type(payload["passed"]) is not bool:
        raise KernelAuthorityError("dev_comparison.passed must be boolean")
    for name in ("parent_metrics", "candidate_metrics"):
        payload[name] = _require_mapping(payload[name], name)
    return _freeze_json_value(payload)


class _CanonicalContract:
    __slots__ = ()

    @classmethod
    def from_payload(cls, payload):
        return cls(
            **_require_exact_schema(
                payload, tuple(f.name for f in fields(cls)), field=cls.__name__
            )
        )

    def to_payload(self) -> dict[str, object]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        return _strict_json_value(payload)

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class ClosedEvaluation(_CanonicalContract):
    """Issuer-owned in-memory aggregates, bound to a single budget stage."""

    schema_version: int
    parent_bundle_sha256: str
    candidate_bundle_sha256: str
    protocol_fingerprint: str
    runtime_fingerprints: Mapping[str, str]
    reservation_sha256: str
    status: str
    train_evaluation_sha256: str
    dev_comparison: Mapping[str, object]
    resource_use: Mapping[str, object]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise KernelAuthorityError("closed evaluation schema_version must be 1")
        for name in (
            "parent_bundle_sha256",
            "candidate_bundle_sha256",
            "protocol_fingerprint",
            "reservation_sha256",
            "train_evaluation_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.status not in ("passed", "failed", "invalid"):
            raise KernelAuthorityError("closed evaluation requires a terminal status")
        object.__setattr__(
            self, "runtime_fingerprints", _runtime(self.runtime_fingerprints)
        )
        object.__setattr__(self, "dev_comparison", _dev(self.dev_comparison))
        use = ResourceUse.from_payload(self.resource_use)
        object.__setattr__(self, "resource_use", _freeze_json_value(use.to_payload()))

    def to_record_payload(self) -> dict[str, object]:
        """Durable identity record; raw Dev is persisted only in decision evidence."""
        payload = self.to_payload()
        payload["schema_version"] = 2
        payload["dev_comparison_sha256"] = fingerprint_payload(
            payload.pop("dev_comparison")
        )
        return payload

    @classmethod
    def _validated_record_payload(cls, payload):
        """Validate a durable record without requiring access to raw Dev values."""
        plain = _require_mapping(payload, "closed evaluation record")
        if (
            type(plain.get("schema_version")) is not int
            or plain.get("schema_version") != 2
        ):
            raise KernelAuthorityError("closed evaluation record schema must be 2")
        record = _require_exact_schema(
            plain,
            tuple(f.name for f in fields(cls) if f.name != "dev_comparison")
            + ("dev_comparison_sha256",),
            field="closed evaluation record",
        )
        for name in (
            "parent_bundle_sha256",
            "candidate_bundle_sha256",
            "protocol_fingerprint",
            "reservation_sha256",
            "train_evaluation_sha256",
            "dev_comparison_sha256",
        ):
            require_sha256(record[name], name)
        if record["status"] not in ("passed", "failed", "invalid"):
            raise KernelAuthorityError("closed evaluation requires a terminal status")
        _runtime(record["runtime_fingerprints"])
        ResourceUse.from_payload(record["resource_use"])
        return record

    @classmethod
    def from_record_payload(cls, payload, *, dev_comparison):
        """Reconstruct evaluator aggregates only from digest-bound decision evidence."""
        record = cls._validated_record_payload(payload)
        digest = record.pop("dev_comparison_sha256")
        if digest != fingerprint_payload(dev_comparison):
            raise KernelAuthorityError("closed evaluation Dev digest mismatch")
        record["schema_version"] = 1
        record["dev_comparison"] = dev_comparison
        return cls.from_payload(record)


@dataclass(frozen=True, slots=True)
class AcceptanceEvidence(_CanonicalContract):
    """Evaluator-only evidence binds a provisional Child; never a sealed SHA."""

    schema_version: int
    parent_bundle_sha256: str
    candidate_bundle_sha256: str
    target: MutationTarget
    protocol_fingerprint: str
    runtime_fingerprints: Mapping[str, str]
    evaluation_sha256: str
    train_evaluation_sha256: str
    dev_comparison: Mapping[str, object]
    budget_allowed: bool
    decision: str
    archive_snapshot_sha256: str
    scheduler_state_sha256: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise KernelAuthorityError("acceptance schema_version must be 1")
        for name in (
            "parent_bundle_sha256",
            "candidate_bundle_sha256",
            "protocol_fingerprint",
            "evaluation_sha256",
            "train_evaluation_sha256",
            "archive_snapshot_sha256",
            "scheduler_state_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.target not in ("numerical", "retrieval", "decision", "joint"):
            raise KernelAuthorityError("acceptance target is invalid")
        if (
            self.decision not in ("accept", "reject")
            or type(self.budget_allowed) is not bool
        ):
            raise KernelAuthorityError("acceptance decision is invalid")
        object.__setattr__(
            self, "runtime_fingerprints", _runtime(self.runtime_fingerprints)
        )
        object.__setattr__(self, "dev_comparison", _dev(self.dev_comparison))


@dataclass(frozen=True)
class _BundleArtifact:
    bundle: EvolutionBundleV2
    artifact_kind: str

    @property
    def schema_version(self):
        return self.bundle.schema_version

    def to_payload(self):
        return self.bundle.to_payload()

    def canonical_bytes(self):
        return self.bundle.canonical_bytes()

    def fingerprint(self):
        return self.bundle.fingerprint()


def _record(bundle, kind, parents, train=None):
    train = train or {
        "status": "seed",
        "train_objectives": {},
        "train_behavior_descriptors": {},
        "resource_use": ResourceUse().to_payload(),
    }
    return ArchiveRecord(
        bundle.schema_version,
        bundle.fingerprint(),
        kind,
        parents,
        "seed" if not parents else kind,
        bundle.protocol_fingerprint,
        bundle.runtime_fingerprints,
        train["train_behavior_descriptors"],
        train["train_objectives"],
        train["status"],
        train["resource_use"],
        (),
        (),
    )


class PromotionHost:
    """Single-writer publication with checkpoint-bound pending and committed events."""

    _EVENT_FIELDS = (
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "action",
        "prior_bundle_sha256",
        "bundle_sha256",
        "acceptance_evidence_sha256",
        "reason",
        "event_sha256",
    )
    _ROLLBACK_REASONS = frozenset(
        {"canary_failed", "integrity_failed", "safety_failed"}
    )

    def __init__(self, kernel: EvolutionKernel):
        self._kernel = kernel

    def _validate_event(self, event, previous, known, sequence):
        _require_exact_schema(event, self._EVENT_FIELDS, field="promotion event")
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        if event["event_sha256"] != fingerprint_payload(body):
            raise KernelAuthorityError("promotion event digest mismatch")
        if type(event["schema_version"]) is not int or event["schema_version"] != 1:
            raise KernelAuthorityError("promotion event schema mismatch")
        if type(event["sequence"]) is not int or event["sequence"] != sequence:
            raise KernelAuthorityError("promotion event sequence mismatch")
        prior = previous["bundle_sha256"] if previous else None
        previous_sha = previous["event_sha256"] if previous else None
        if (
            event["prior_bundle_sha256"] != prior
            or event["previous_event_sha256"] != previous_sha
        ):
            raise KernelAuthorityError("promotion event prior identity mismatch")
        require_sha256(event["bundle_sha256"], "promotion Bundle")
        if previous is None:
            if (
                event["action"],
                event["reason"],
                event["bundle_sha256"],
                event["acceptance_evidence_sha256"],
            ) != ("seed", "seed", self._kernel._seed_sha, None):
                raise KernelAuthorityError(
                    "promotion history must begin at committed seed"
                )
        elif event["action"] == "activate":
            matches = [
                t
                for t in self._kernel._transitions.values()
                if t["sealed_bundle_sha256"] == event["bundle_sha256"]
            ]
            if len(matches) != 1 or event["reason"] != "accepted":
                raise KernelAuthorityError(
                    "activation requires an authorized completed transition"
                )
            transition = matches[0]
            if (
                transition["decision"] != "accept"
                or transition["parent_bundle_sha256"] != prior
                or transition["acceptance_evidence_sha256"]
                != event["acceptance_evidence_sha256"]
            ):
                raise KernelAuthorityError("promotion acceptance binding mismatch")
        elif event["action"] == "rollback":
            if (
                event["bundle_sha256"] not in known
                or event["bundle_sha256"] == prior
                or event["reason"] not in self._ROLLBACK_REASONS
                or known[event["bundle_sha256"]] != event["acceptance_evidence_sha256"]
            ):
                raise KernelAuthorityError(
                    "rollback requires known prior active and closed reason"
                )
        else:
            raise KernelAuthorityError("invalid promotion action")

    def history(self) -> tuple[dict[str, object], ...]:
        """Verify the history chain independently of possibly corrupt source objects."""
        path = self._kernel.store.root / "promotion_history.jsonl"
        if not path.exists():
            return ()
        data = path.read_bytes()
        if not data or not data.endswith(b"\n"):
            raise KernelAuthorityError("promotion history is truncated")
        result, known = [], {}
        try:
            for line in data.splitlines(keepends=True):
                event = strict_json_loads(
                    line.decode("utf-8"), context="promotion history"
                )
                if canonical_v2_bytes(event) != line:
                    raise ValueError("noncanonical promotion event")
                self._validate_event(
                    event, result[-1] if result else None, known, len(result)
                )
                result.append(event)
                known[event["bundle_sha256"]] = event["acceptance_evidence_sha256"]
        except (ValueError, TypeError, UnicodeError) as error:
            raise KernelAuthorityError(f"invalid promotion history: {error}") from error
        return tuple(result)

    def _publication_state(self):
        kernel = self._kernel
        history = self.history()
        committed = kernel._committed_event_sha
        positions = [i for i, e in enumerate(history) if e["event_sha256"] == committed]
        index = positions[0] if len(positions) == 1 else -1
        if committed is not None and index < 0:
            raise KernelAuthorityError("committed publication event is missing")
        if kernel._active_sha != (
            history[index]["bundle_sha256"] if index >= 0 else None
        ):
            raise KernelAuthorityError("checkpoint committed active identity mismatch")
        prefix = history[: index + 1]
        tail = history[index + 1 :]
        pending = kernel._pending_event
        if pending is None:
            if tail:
                raise KernelAuthorityError(
                    "history contains an uncommitted, unauthorized event"
                )
        else:
            known = {
                e["bundle_sha256"]: e["acceptance_evidence_sha256"] for e in prefix
            }
            self._validate_event(
                pending, prefix[-1] if prefix else None, known, len(prefix)
            )
            if tail not in ((), (pending,)):
                raise KernelAuthorityError("pending publication does not match history")
        return history, bool(tail)

    def _pointer_sha(self):
        path = self._kernel.store.root / "accepted_bundle.json"
        return (
            EvolutionBundleV2.from_payload(_read(path)).fingerprint()
            if path.exists()
            else None
        )

    def reconcile(self) -> EvolutionBundleV2:
        kernel = self._kernel
        _, appended = self._publication_state()
        pending = kernel._pending_event
        if pending is None:
            if self._pointer_sha() != kernel._active_sha:
                raise KernelAuthorityError(
                    "active pointer drift after committed publication"
                )
            return kernel._verified_bundle(
                kernel._active_sha, recovery=kernel._terminal is not None
            )
        bundle = kernel._verified_bundle(
            pending["bundle_sha256"], recovery=kernel._terminal is not None
        )
        if self._pointer_sha() not in (kernel._active_sha, bundle.fingerprint()):
            raise KernelAuthorityError(
                "active pointer drift from explicit pending publication"
            )
        if not appended:
            kernel._persist_pending_intent(pending)
            kernel.store.append_promotion(pending)
        kernel._persist_pending_intent(pending)
        kernel.store.publish_active_bundle(bundle.to_payload())
        kernel._committed_event_sha = pending["event_sha256"]
        kernel._active_sha = bundle.fingerprint()
        kernel._pending_event = None
        kernel._checkpoint()
        return bundle

    def _publish(self, action, bundle, reason, history):
        kernel = self._kernel
        event = {
            "schema_version": 1,
            "sequence": len(history),
            "previous_event_sha256": history[-1]["event_sha256"] if history else None,
            "action": action,
            "prior_bundle_sha256": kernel._active_sha,
            "bundle_sha256": bundle.fingerprint(),
            "acceptance_evidence_sha256": bundle.acceptance_evidence_sha256,
            "reason": reason,
        }
        event["event_sha256"] = fingerprint_payload(event)
        kernel._pending_event = event
        kernel._checkpoint()
        return self.reconcile()

    def activate(self, bundle: EvolutionBundleV2) -> EvolutionBundleV2:
        self._kernel._require_mutable()
        verified = self._kernel._verified_bundle(bundle.fingerprint())
        if verified.acceptance_evidence_sha256 is None:
            raise KernelAuthorityError("activation requires a sealed Bundle")
        active = self.reconcile()
        if active.fingerprint() == verified.fingerprint():
            return active
        if verified.parent_bundle_sha256 != active.fingerprint():
            raise KernelAuthorityError("activation requires the active Parent")
        return self._publish("activate", verified, "accepted", self.history())

    def rollback(
        self, *, reason: str, bundle_sha256: str | None = None
    ) -> EvolutionBundleV2:
        kernel = self._kernel
        if reason not in self._ROLLBACK_REASONS:
            raise KernelAuthorityError(
                "rollback requires a closed canary/integrity reason"
            )
        history, _ = self._publication_state()
        if kernel._pending_event is not None or kernel._terminal is not None:
            return self.reconcile()
        if self._pointer_sha() != kernel._active_sha:
            raise KernelAuthorityError(
                "active pointer drift after committed publication"
            )
        if (
            bundle_sha256 is None
            and history[-1]["action"] == "rollback"
            and history[-1]["reason"] == reason
        ):
            return self.reconcile()
        target = bundle_sha256 or history[-1]["prior_bundle_sha256"]
        if target is None or target not in {e["bundle_sha256"] for e in history[:-1]}:
            raise KernelAuthorityError("rollback requires a known prior active Bundle")
        corrupt_source = False
        try:
            self.reconcile()
        except (KernelAuthorityError, ValueError):
            if reason not in ("integrity_failed", "safety_failed"):
                raise
            corrupt_source = True
        bundle = kernel._verified_bundle(target, recovery=corrupt_source)
        if corrupt_source:
            selected = verify_selected_lineage(kernel.store.root / "archive", target)
            kernel._archive_snapshot = selected["archive_snapshot_sha256"]
            kernel._terminal = {
                "status": "recovered_requires_new_epoch",
                "invalidated_bundle_sha256": kernel._active_sha,
                "destination_bundle_sha256": target,
                "destination_lineage": list(selected["lineage"]),
                "archive_snapshot_sha256": selected["archive_snapshot_sha256"],
                "reason": reason,
            }
        return self._publish("rollback", bundle, reason, history)


class EvolutionKernel:
    """Trusted evaluations, durable budget closures, and automatic publication."""

    _CHECKPOINT_FIELDS = (
        "schema_version",
        "budget",
        "archive_snapshot_sha256",
        "active_bundle_sha256",
        "committed_event_sha256",
        "pending_publication",
        "budget_closures",
        "completed_transitions",
        "terminal_recovery",
        "checkpoint_sha256",
    )

    def _configure(self, store, protocol, budget, seed, checkpoint_path=None):
        if type(protocol) is not KernelProtocolCommitment:
            raise KernelAuthorityError("kernel requires the full protocol commitment")
        if seed.generation != 0 or seed.protocol_fingerprint != protocol.fingerprint():
            raise KernelAuthorityError("seed protocol commitment mismatch")
        self.store, self.protocol, self.budget = store, protocol, budget
        self.checkpoint_path = self._resolve_checkpoint_path(store, checkpoint_path)
        self._seed_sha, self._runtimes = seed.fingerprint(), seed.runtime_fingerprints
        self._closed, self._permits, self._closures, self._transitions = {}, {}, {}, {}
        self._unpersisted_closures = {}
        self._active_sha = self._committed_event_sha = self._pending_event = None
        self._terminal = self._last_checkpoint_sha = None
        self.promotion_host = PromotionHost(self)

    @staticmethod
    def _resolve_checkpoint_path(store, checkpoint_path):
        path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else store.root / "checkpoint.json"
        )
        if path.name != "checkpoint.json" or not path.resolve().is_relative_to(
            store.root.resolve()
        ):
            raise KernelAuthorityError(
                "checkpoint path must be checkpoint.json within the run"
            )
        return path

    def __init__(
        self,
        store: V2RunStore,
        protocol: KernelProtocolCommitment,
        budget: BudgetLedger,
        *,
        seed: EvolutionBundleV2,
        checkpoint_path: str | Path | None = None,
    ):
        # A constructor never adopts or repairs an established/partial run.
        if any(path.is_file() for path in store.root.rglob("*")):
            raise KernelAuthorityError(
                "constructor requires a fresh run; use resume for established authority"
            )
        if (
            budget.charged_use != ResourceUse()
            or budget.checkpoint()["open_reservations"]
        ):
            raise KernelAuthorityError("fresh kernel requires an unused budget ledger")
        self._configure(store, protocol, budget, seed, checkpoint_path)
        store.write_run_manifest(self._manifest())
        store.write_budget_plan(budget.plan.to_payload())
        self.archive = EvolutionArchive(store.root / "archive")
        self.archive.append(
            _BundleArtifact(seed, "bundle"), _record(seed, "bundle", ())
        )
        self._archive_snapshot = self.archive.snapshot_sha256()
        self.promotion_host._publish("seed", seed, "seed", ())

    def _manifest(self):
        return {
            "system": "evolution_v2",
            "kernel_protocol": self.protocol.to_payload(),
            "runtime_fingerprints": dict(self._runtimes),
            "seed_bundle_sha256": self._seed_sha,
            "budget_plan_sha256": self.budget.plan.fingerprint(),
        }

    @classmethod
    def resume(
        cls,
        store: V2RunStore,
        plan: BudgetPlan,
        *,
        monotonic: Callable[[], float],
        checkpoint_path: str | Path | None = None,
    ):
        try:
            manifest = _read(store.root / "run_manifest.json")
            _read(store.root / "budget_plan.json", plan.fingerprint())
            checkpoint = _read(cls._resolve_checkpoint_path(store, checkpoint_path))
            _require_exact_schema(
                checkpoint, cls._CHECKPOINT_FIELDS, field="kernel checkpoint"
            )
            if (
                fingerprint_payload(
                    {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
                )
                != checkpoint["checkpoint_sha256"]
            ):
                raise ValueError("checkpoint digest mismatch")
            if (
                type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 2
            ):
                raise ValueError("checkpoint schema mismatch")
            protocol = KernelProtocolCommitment.from_payload(
                manifest["kernel_protocol"]
            )
            seed_sha = require_sha256(manifest["seed_bundle_sha256"], "seed SHA")
            seed = EvolutionBundleV2.from_payload(
                _read(store.root / "archive" / "objects" / f"{seed_sha}.json", seed_sha)
            )
            budget = BudgetLedger.resume(
                plan, checkpoint["budget"], monotonic=monotonic
            )
            if budget.checkpoint()["open_reservations"]:
                raise ValueError(
                    "open reservation cannot be resumed or reissued; recover in a new epoch"
                )
            kernel = cls.__new__(cls)
            kernel._configure(store, protocol, budget, seed, checkpoint_path)
            if any(manifest.get(k) != v for k, v in kernel._manifest().items()):
                raise ValueError("manifest protocol/authority mismatch")
            kernel._closures = dict(checkpoint["budget_closures"])
            kernel._transitions = dict(checkpoint["completed_transitions"])
            kernel._active_sha = checkpoint["active_bundle_sha256"]
            kernel._committed_event_sha = checkpoint["committed_event_sha256"]
            kernel._pending_event = checkpoint["pending_publication"]
            kernel._terminal = checkpoint["terminal_recovery"]
            kernel._archive_snapshot = checkpoint["archive_snapshot_sha256"]
            kernel._last_checkpoint_sha = checkpoint["checkpoint_sha256"]
            if kernel._terminal is not None:
                # Only finish an already authorized terminal recovery. No mutable
                # or partially verified archive escapes this path.
                kernel.archive = None
                kernel._validate_terminal()
                kernel.promotion_host.reconcile()
                raise KernelAuthorityError("recovered run requires a new epoch")
            kernel.archive = EvolutionArchive(store.root / "archive")
            kernel._verify_checkpoint_state(
                prior_elapsed_wall_seconds=checkpoint["budget"][
                    "prior_elapsed_wall_seconds"
                ]
            )
            kernel.promotion_host.reconcile()
            return kernel
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(
                f"resume authority verification failed: {error}"
            ) from error

    def _require_mutable(self):
        if self._terminal is not None:
            raise KernelAuthorityError("recovered run requires a new epoch")

    @staticmethod
    def stage_id(child: EvolutionBundleV2) -> str:
        return f"evaluation:{child.fingerprint()}"

    def reserve_evaluation(
        self, child: EvolutionBundleV2, estimate: ResourceUse
    ) -> StagePermit:
        """Issue the only valid permit instance and durably reserve its resources."""
        self._require_mutable()
        self.active_bundle()
        permit = self.budget.reserve_stage(self.stage_id(child), estimate)
        if permit.allowed:
            self._permits[permit.reservation_sha256] = permit
        self._checkpoint()
        return permit

    def _permit(self, child, permit):
        if (
            type(permit) is not StagePermit
            or not permit.allowed
            or permit.reason is not None
            or self._permits.get(permit.reservation_sha256) is not permit
        ):
            raise KernelAuthorityError(
                "transition requires the exact kernel-issued stage permit"
            )
        if not any(
            r["reservation_sha256"] == permit.reservation_sha256
            and r["stage_id"] == self.stage_id(child)
            for r in self.budget.checkpoint()["open_reservations"]
        ):
            raise KernelAuthorityError(
                "transition requires a live candidate-bound stage permit"
            )
        return permit.reservation_sha256

    def close_evaluation(
        self,
        parent: EvolutionBundleV2,
        child: EvolutionBundleV2,
        *,
        permit: StagePermit,
        status: str,
        train_objectives: Mapping[str, object],
        train_behavior_descriptors: Mapping[str, object],
        dev_comparison: Mapping[str, object],
        resource_use: ResourceUse,
    ) -> ClosedEvaluation:
        """Mint trusted immutable output. Train status and Dev values stay separate."""
        self._require_mutable()
        reservation = self._permit(child, permit)
        if reservation in self._closed:
            raise KernelAuthorityError("stage already has a closed evaluation")
        train = {
            "schema_version": 1,
            "parent_bundle_sha256": parent.fingerprint(),
            "candidate_bundle_sha256": child.fingerprint(),
            "status": status,
            "protocol_fingerprint": self.protocol.fingerprint(),
            "runtime_fingerprints": dict(self._runtimes),
            "train_objectives": _require_mapping(train_objectives, "train_objectives"),
            "train_behavior_descriptors": _require_mapping(
                train_behavior_descriptors, "train_behavior_descriptors"
            ),
            "resource_use": resource_use.to_payload(),
        }
        _reject_reserved_feedback_keys(train, field="train evaluation")
        result = ClosedEvaluation(
            1,
            parent.fingerprint(),
            child.fingerprint(),
            self.protocol.fingerprint(),
            self._runtimes,
            reservation,
            status,
            fingerprint_payload(train),
            dev_comparison,
            resource_use.to_payload(),
        )
        self._closed[reservation] = (result, train, permit)
        return result

    def _issued_evaluation(self, evaluation, permit):
        # Resolve authority by object identity, never caller-supplied hashes.
        for issued in self._closed.values():
            if evaluation is issued[0]:
                return issued
        for issued in self._closed.values():
            if permit is issued[2]:
                return issued
        raise KernelAuthorityError("closed evaluation was not issued by this kernel")

    def _account(self, issued):
        evaluation, train, permit = issued
        reservation = evaluation.reservation_sha256
        if (
            reservation in self._closures
            and reservation not in self._unpersisted_closures
        ):
            return self._load_closure(reservation)
        if reservation not in self._unpersisted_closures:
            before = self.budget.checkpoint()
            outcome = self.budget.close_stage(
                permit, ResourceUse.from_payload(evaluation.resource_use)
            )
            self._unpersisted_closures[reservation] = _freeze_json_value(
                {
                    "schema_version": 1,
                    "candidate_bundle_sha256": evaluation.candidate_bundle_sha256,
                    "parent_bundle_sha256": evaluation.parent_bundle_sha256,
                    "reservation_sha256": reservation,
                    "stage_id": f"evaluation:{evaluation.candidate_bundle_sha256}",
                    "evaluation_sha256": fingerprint_payload(
                        evaluation.to_record_payload()
                    ),
                    "resource_use": dict(evaluation.resource_use),
                    "allowed": outcome.allowed,
                    "reason": outcome.reason,
                    "budget_before": before,
                    "budget_after": self.budget.checkpoint(),
                }
            )
        # Preserve the original accounting result before any artifact write;
        # finally/retries must never close this reservation a second time.
        closure = self._unpersisted_closures[reservation]
        # Actual use is already charged even if one of these writes fails.
        self.store.write_evaluation(evaluation.candidate_bundle_sha256, "train", train)
        self.store.write_evaluation(
            evaluation.candidate_bundle_sha256, "closed", evaluation.to_record_payload()
        )
        self.store.write_evaluation(
            evaluation.candidate_bundle_sha256, "budget_closure", closure
        )
        self._closures[reservation] = {
            "candidate_bundle_sha256": evaluation.candidate_bundle_sha256,
            "closure_sha256": fingerprint_payload(closure),
        }
        self._checkpoint()
        del self._unpersisted_closures[reservation]
        return closure

    def _load_closure(self, reservation):
        ref = self._closures.get(reservation)
        if ref is None:
            raise KernelAuthorityError("missing durable budget closure")
        _require_exact_schema(
            ref,
            ("candidate_bundle_sha256", "closure_sha256"),
            field="closure reference",
        )
        require_sha256(ref["candidate_bundle_sha256"], "closure candidate")
        require_sha256(ref["closure_sha256"], "closure SHA")
        closure = _read(
            self.store.root
            / "evaluations"
            / ref["candidate_bundle_sha256"]
            / "budget_closure.json",
            ref["closure_sha256"],
        )
        _require_exact_schema(
            closure,
            (
                "schema_version",
                "candidate_bundle_sha256",
                "parent_bundle_sha256",
                "reservation_sha256",
                "stage_id",
                "evaluation_sha256",
                "resource_use",
                "allowed",
                "reason",
                "budget_before",
                "budget_after",
            ),
            field="budget closure",
        )
        if type(closure["schema_version"]) is not int or closure["schema_version"] != 1:
            raise KernelAuthorityError("budget closure schema mismatch")
        if (
            closure["candidate_bundle_sha256"] != ref["candidate_bundle_sha256"]
            or closure["reservation_sha256"] != reservation
            or closure["stage_id"] != f"evaluation:{ref['candidate_bundle_sha256']}"
        ):
            raise KernelAuthorityError("candidate-specific budget closure mismatch")
        closed_record = ClosedEvaluation._validated_record_payload(
            _read(
                self.store.root
                / "evaluations"
                / ref["candidate_bundle_sha256"]
                / "closed.json",
                closure["evaluation_sha256"],
            )
        )
        if (
            closed_record["candidate_bundle_sha256"]
            != closure["candidate_bundle_sha256"]
            or closed_record["parent_bundle_sha256"] != closure["parent_bundle_sha256"]
            or closed_record["reservation_sha256"] != closure["reservation_sha256"]
            or closed_record["resource_use"] != closure["resource_use"]
        ):
            raise KernelAuthorityError("budget closure closed-record binding mismatch")
        replay = BudgetLedger.resume(
            self.budget.plan, closure["budget_before"], monotonic=lambda: 0.0
        )
        opened = [
            r
            for r in replay.checkpoint()["open_reservations"]
            if r["reservation_sha256"] == reservation
        ]
        if len(opened) != 1 or opened[0]["stage_id"] != closure["stage_id"]:
            raise KernelAuthorityError("budget closure reservation binding mismatch")
        outcome = replay.close_stage(
            reservation, ResourceUse.from_payload(closure["resource_use"])
        )
        if type(closure["allowed"]) is not bool or (
            outcome.allowed,
            outcome.reason,
        ) != (closure["allowed"], closure["reason"]):
            raise KernelAuthorityError("budget closure outcome mismatch")
        BudgetLedger.resume(
            self.budget.plan, closure["budget_after"], monotonic=lambda: 0.0
        )
        if (
            closure["budget_after"]["prior_elapsed_wall_seconds"]
            < closure["budget_before"]["prior_elapsed_wall_seconds"]
        ):
            raise KernelAuthorityError("budget closure elapsed time moved backwards")
        expected = replay.checkpoint()
        if any(
            expected[k] != closure["budget_after"][k]
            for k in expected
            if k not in ("prior_elapsed_wall_seconds", "checkpoint_sha256")
        ):
            raise KernelAuthorityError("budget closure actual accounting mismatch")
        if reservation not in self.budget.checkpoint()["closed_reservation_sha256s"]:
            raise KernelAuthorityError("budget closure not in durable ledger")
        return closure

    def evaluate_transition(
        self,
        parent: EvolutionBundleV2,
        child: EvolutionBundleV2,
        *,
        target: MutationTarget,
        evaluation: ClosedEvaluation,
        permit: StagePermit | None = None,
    ) -> EvolutionBundleV2:
        issued = self._issued_evaluation(evaluation, permit)
        trusted, train, _ = issued
        try:
            self._require_mutable()
            active = self.active_bundle()
            if parent.fingerprint() != active.fingerprint():
                raise KernelAuthorityError("transition Parent is not active")
            self._permit(child, permit)
            validate_child_scope(parent, child, target)
            if (
                parent.protocol_fingerprint != self.protocol.fingerprint()
                or child.protocol_fingerprint != self.protocol.fingerprint()
                or parent.runtime_fingerprints != self._runtimes
            ):
                raise KernelAuthorityError("transition protocol/runtime mismatch")
            if (
                evaluation is not trusted
                or trusted.parent_bundle_sha256 != parent.fingerprint()
                or trusted.candidate_bundle_sha256 != child.fingerprint()
            ):
                raise KernelAuthorityError(
                    "closed evaluation authority or Bundle binding mismatch"
                )
            closure = self._account(issued)
            self.archive.append(
                _BundleArtifact(child, "bundle"),
                _record(child, "bundle", (parent.fingerprint(),), train),
            )
            self._archive_snapshot = self.archive.snapshot_sha256()
            decision = (
                "accept"
                if closure["allowed"]
                and trusted.status == "passed"
                and trusted.dev_comparison["passed"]
                else "reject"
            )
            evidence = AcceptanceEvidence(
                1,
                parent.fingerprint(),
                child.fingerprint(),
                target,
                self.protocol.fingerprint(),
                self._runtimes,
                closure["evaluation_sha256"],
                trusted.train_evaluation_sha256,
                trusted.dev_comparison,
                closure["allowed"],
                decision,
                self._archive_snapshot,
                parent.scheduler_state_sha256,
            )
            self.store.write_acceptance(evidence.fingerprint(), evidence.to_payload())
            self._transitions[child.fingerprint()] = {
                "parent_bundle_sha256": parent.fingerprint(),
                "acceptance_evidence_sha256": evidence.fingerprint(),
                "budget_closure_sha256": fingerprint_payload(closure),
                "decision": decision,
                "archive_snapshot_sha256": self._archive_snapshot,
                "sealed_bundle_sha256": None,
            }
            self._checkpoint()
            evidence = self.load_acceptance(evidence.fingerprint())
            self.store.append_progress(
                {
                    "candidate_bundle_sha256": child.fingerprint(),
                    "train_evaluation_sha256": trusted.train_evaluation_sha256,
                    "status": trusted.status,
                    "resource_use": dict(trusted.resource_use),
                }
            )
            if decision == "reject":
                return parent
            sealed = _seal_acceptance(
                parent,
                child,
                target,
                evidence,
                self._archive_snapshot,
                parent.scheduler_state_sha256,
            )
            self.archive.append(
                _BundleArtifact(sealed, "acceptance_seal"),
                _record(sealed, "acceptance_seal", (child.fingerprint(),), train),
            )
            self._archive_snapshot = self.archive.snapshot_sha256()
            self._transitions[child.fingerprint()][
                "sealed_bundle_sha256"
            ] = sealed.fingerprint()
            self._checkpoint()
            self.promotion_host.activate(sealed)
            return sealed
        except (ValueError, TypeError) as error:
            raise KernelAuthorityError(str(error)) from error
        finally:
            # Issuer-owned actual work is charged regardless of caller permit,
            # pointer integrity, scope, or evaluation-authenticity failures.
            self._account(issued)

    def load_acceptance(self, identity: str) -> AcceptanceEvidence:
        return self._load_acceptance(identity, recovery=False)

    def _load_acceptance(self, identity, *, recovery):
        try:
            require_sha256(identity, "acceptance evidence SHA")
            evidence = AcceptanceEvidence.from_payload(
                _read(self.store.root / "acceptance" / f"{identity}.json", identity)
            )
            ref = self._transitions.get(evidence.candidate_bundle_sha256)
            if ref is None or ref["acceptance_evidence_sha256"] != identity:
                raise ValueError("evidence lacks an authorized durable transition")
            evaluation = ClosedEvaluation.from_record_payload(
                _read(
                    self.store.root
                    / "evaluations"
                    / evidence.candidate_bundle_sha256
                    / "closed.json",
                    evidence.evaluation_sha256,
                ),
                dev_comparison=evidence.dev_comparison,
            )
            closure = self._load_closure(evaluation.reservation_sha256)
            if (
                ref["budget_closure_sha256"] != fingerprint_payload(closure)
                or closure["allowed"] != evidence.budget_allowed
                or closure["candidate_bundle_sha256"]
                != evidence.candidate_bundle_sha256
                or closure["parent_bundle_sha256"] != evidence.parent_bundle_sha256
                or closure["evaluation_sha256"] != evidence.evaluation_sha256
                or closure["resource_use"] != dict(evaluation.resource_use)
            ):
                raise ValueError("acceptance budget closure binding mismatch")
            train = _read(
                self.store.root
                / "evaluations"
                / evidence.candidate_bundle_sha256
                / "train.json",
                evidence.train_evaluation_sha256,
            )
            _reject_reserved_feedback_keys(train, field="train evaluation")
            for name in (
                "parent_bundle_sha256",
                "candidate_bundle_sha256",
                "protocol_fingerprint",
                "runtime_fingerprints",
                "train_evaluation_sha256",
                "dev_comparison",
            ):
                if getattr(evidence, name) != getattr(evaluation, name):
                    raise ValueError(f"acceptance evaluation {name} mismatch")
            for name in (
                "parent_bundle_sha256",
                "candidate_bundle_sha256",
                "protocol_fingerprint",
                "runtime_fingerprints",
                "status",
                "resource_use",
            ):
                if train[name] != getattr(evaluation, name):
                    raise ValueError(f"Train evaluation {name} mismatch")
            decision = (
                "accept"
                if closure["allowed"]
                and evaluation.status == "passed"
                and evidence.dev_comparison["passed"]
                else "reject"
            )
            if (
                evidence.decision != decision
                or ref["decision"] != decision
                or ref["parent_bundle_sha256"] != evidence.parent_bundle_sha256
                or ref["archive_snapshot_sha256"] != evidence.archive_snapshot_sha256
                or evidence.protocol_fingerprint != self.protocol.fingerprint()
                or evidence.runtime_fingerprints != self._runtimes
            ):
                raise ValueError(
                    "acceptance decision/protocol/runtime/transition mismatch"
                )
            provisional = self._archive_bundle(
                evidence.candidate_bundle_sha256, recovery=recovery
            )
            parent = self._archive_bundle(
                evidence.parent_bundle_sha256, recovery=recovery
            )
            validate_child_scope(parent, provisional, evidence.target)
            if (
                parent.protocol_fingerprint != self.protocol.fingerprint()
                or parent.runtime_fingerprints != self._runtimes
            ):
                raise ValueError("acceptance Parent protocol/runtime mismatch")
            prefix = hashlib.sha256()
            expected = _record(
                provisional, "bundle", (parent.fingerprint(),), train
            ).to_payload()
            for line in (
                (self.store.root / "archive" / "index.jsonl")
                .read_bytes()
                .splitlines(keepends=True)
            ):
                prefix.update(line)
                record = strict_json_loads(
                    line.decode("utf-8"), context="archive index"
                )["record"]
                if record["artifact_sha256"] == provisional.fingerprint():
                    if (
                        record != expected
                        or prefix.hexdigest() != evidence.archive_snapshot_sha256
                    ):
                        raise ValueError(
                            "acceptance provisional archive binding mismatch"
                        )
                    break
            return evidence
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(
                f"invalid acceptance evidence: {error}"
            ) from error

    def _archive_bundle(self, identity, *, recovery):
        require_sha256(identity, "Bundle SHA")
        if recovery:
            selected = verify_selected_lineage(self.store.root / "archive", identity)
            return EvolutionBundleV2.from_payload(selected["payloads"][identity])
        self.archive.lineage(identity)
        return EvolutionBundleV2.from_payload(
            _read(
                self.store.root / "archive" / "objects" / f"{identity}.json", identity
            )
        )

    def _verified_bundle(self, identity, *, recovery=False) -> EvolutionBundleV2:
        try:
            bundle = self._archive_bundle(identity, recovery=recovery)
            if (
                bundle.protocol_fingerprint != self.protocol.fingerprint()
                or bundle.runtime_fingerprints != self._runtimes
            ):
                raise ValueError("Bundle protocol/runtime commitment mismatch")
            if bundle.generation == 0:
                if identity != self._seed_sha:
                    raise ValueError("unknown seed Bundle")
                return bundle
            if bundle.acceptance_evidence_sha256 is None:
                raise ValueError("Bundle is not sealed")
            evidence = self._load_acceptance(
                bundle.acceptance_evidence_sha256, recovery=recovery
            )
            ref = self._transitions[evidence.candidate_bundle_sha256]
            if evidence.decision != "accept" or ref["sealed_bundle_sha256"] != identity:
                raise ValueError("sealed Bundle not authorized by completed transition")
            parent = self._verified_bundle(
                evidence.parent_bundle_sha256, recovery=recovery
            )
            provisional = self._archive_bundle(
                evidence.candidate_bundle_sha256, recovery=recovery
            )
            expected = _seal_acceptance(
                parent,
                provisional,
                evidence.target,
                evidence,
                evidence.archive_snapshot_sha256,
                evidence.scheduler_state_sha256,
            )
            if expected.canonical_bytes() != bundle.canonical_bytes():
                raise ValueError("sealed Bundle differs from acceptance evidence")
            for line in (
                (self.store.root / "archive" / "index.jsonl")
                .read_bytes()
                .splitlines(keepends=True)
            ):
                record = strict_json_loads(
                    line.decode("utf-8"), context="archive index"
                )["record"]
                if record["artifact_sha256"] == identity:
                    if record["artifact_kind"] != "acceptance_seal" or record[
                        "parent_sha256s"
                    ] != [provisional.fingerprint()]:
                        raise ValueError("acceptance seal archive lineage mismatch")
                    break
            return bundle
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(f"invalid archived Bundle: {error}") from error

    def active_bundle(self) -> EvolutionBundleV2:
        return self.promotion_host.reconcile()

    def finalize(self) -> None:
        """Seal finalization only after all live evaluations have closed cleanly."""
        self._require_mutable()
        self._verify_checkpoint_state()
        self.active_bundle()
        if self.budget.checkpoint()["open_reservations"]:
            raise KernelAuthorityError("cannot finalize with open evaluations")
        self.budget.begin_finalization()
        self._checkpoint()

    def _verify_checkpoint_state(self, *, prior_elapsed_wall_seconds=None):
        self.promotion_host._publication_state()
        prefix, snapshots = hashlib.sha256(), set()
        for line in (
            (self.store.root / "archive" / "index.jsonl")
            .read_bytes()
            .splitlines(keepends=True)
        ):
            prefix.update(line)
            snapshots.add(prefix.hexdigest())
        if self._archive_snapshot not in snapshots:
            raise KernelAuthorityError(
                "checkpoint archive snapshot is not a verified prefix"
            )
        total = ResourceUse()
        if prior_elapsed_wall_seconds is None:
            prior_elapsed_wall_seconds = self.budget.elapsed_wall_seconds
        for reservation in self._closures:
            closure = self._load_closure(reservation)
            if (
                prior_elapsed_wall_seconds
                < closure["budget_after"]["prior_elapsed_wall_seconds"]
            ):
                raise KernelAuthorityError(
                    "checkpoint elapsed time precedes a verified budget closure"
                )
            total = total + ResourceUse.from_payload(closure["resource_use"])
        if total != self.budget.charged_use or set(self._closures) != set(
            self.budget.checkpoint()["closed_reservation_sha256s"]
        ):
            raise KernelAuthorityError(
                "checkpoint budget closures do not account for the ledger"
            )
        for candidate, ref in self._transitions.items():
            require_sha256(candidate, "transition candidate")
            _require_exact_schema(
                ref,
                (
                    "parent_bundle_sha256",
                    "acceptance_evidence_sha256",
                    "budget_closure_sha256",
                    "decision",
                    "archive_snapshot_sha256",
                    "sealed_bundle_sha256",
                ),
                field="completed transition",
            )
            evidence = self.load_acceptance(ref["acceptance_evidence_sha256"])
            if evidence.candidate_bundle_sha256 != candidate:
                raise KernelAuthorityError("checkpoint transition candidate mismatch")
            if ref["sealed_bundle_sha256"] is not None:
                self._verified_bundle(ref["sealed_bundle_sha256"])

    def _validate_terminal(self):
        value = self._terminal
        _require_exact_schema(
            value,
            (
                "status",
                "invalidated_bundle_sha256",
                "destination_bundle_sha256",
                "destination_lineage",
                "archive_snapshot_sha256",
                "reason",
            ),
            field="terminal recovery",
        )
        if value["status"] != "recovered_requires_new_epoch" or value["reason"] not in (
            "integrity_failed",
            "safety_failed",
        ):
            raise KernelAuthorityError("invalid terminal recovery")
        history, _ = self.promotion_host._publication_state()
        event = self._pending_event or history[-1]
        if (
            event["action"] != "rollback"
            or event["reason"] != value["reason"]
            or event["prior_bundle_sha256"] != value["invalidated_bundle_sha256"]
            or event["bundle_sha256"] != value["destination_bundle_sha256"]
        ):
            raise KernelAuthorityError("terminal recovery is not bound to rollback")
        selected = verify_selected_lineage(
            self.store.root / "archive", value["destination_bundle_sha256"]
        )
        if (
            list(selected["lineage"]) != value["destination_lineage"]
            or selected["archive_snapshot_sha256"] != value["archive_snapshot_sha256"]
            or self._archive_snapshot != value["archive_snapshot_sha256"]
        ):
            raise KernelAuthorityError("terminal destination lineage/index mismatch")

    def _persist_pending_intent(self, pending):
        """Re-establish and verify durability at each publication write boundary."""
        if self._pending_event != pending:
            raise KernelAuthorityError("pending publication changed before persistence")
        self._checkpoint()
        durable = _read(self.checkpoint_path)
        digest = fingerprint_payload(
            {key: value for key, value in durable.items() if key != "checkpoint_sha256"}
        )
        if (
            digest != self._last_checkpoint_sha
            or durable.get("checkpoint_sha256") != digest
            or durable.get("pending_publication") != pending
            or durable.get("committed_event_sha256") != self._committed_event_sha
            or durable.get("active_bundle_sha256") != self._active_sha
        ):
            raise KernelAuthorityError(
                "pending publication checkpoint failed durable verification"
            )

    def _checkpoint(self):
        # Never derive authority from the active pointer: it may be the failure
        # being handled while authentic work still must be durably accounted.
        manifest = _read(self.store.root / "run_manifest.json")
        if any(manifest.get(k) != v for k, v in self._manifest().items()):
            raise KernelAuthorityError("run manifest protocol/authority mismatch")
        _read(self.store.root / "budget_plan.json", self.budget.plan.fingerprint())
        if self._last_checkpoint_sha is not None:
            existing = _read(self.checkpoint_path)
            digest = fingerprint_payload(
                {k: v for k, v in existing.items() if k != "checkpoint_sha256"}
            )
            if (
                digest != existing.get("checkpoint_sha256")
                or digest != self._last_checkpoint_sha
            ):
                raise KernelAuthorityError("checkpoint changed outside the kernel")
        checkpoint = {
            "schema_version": 2,
            "budget": self.budget.checkpoint(),
            "archive_snapshot_sha256": self._archive_snapshot,
            "active_bundle_sha256": self._active_sha,
            "committed_event_sha256": self._committed_event_sha,
            "pending_publication": self._pending_event,
            "budget_closures": self._closures,
            "completed_transitions": self._transitions,
            "terminal_recovery": self._terminal,
        }
        checkpoint["checkpoint_sha256"] = fingerprint_payload(checkpoint)
        if self.checkpoint_path == self.store.root / "checkpoint.json":
            self.store.write_checkpoint(checkpoint)
        else:
            write_atomic_json(self.checkpoint_path, checkpoint)
        self._last_checkpoint_sha = checkpoint["checkpoint_sha256"]


__all__ = [
    "AcceptanceEvidence",
    "ClosedEvaluation",
    "EvolutionKernel",
    "KernelAuthorityError",
    "PromotionHost",
]
