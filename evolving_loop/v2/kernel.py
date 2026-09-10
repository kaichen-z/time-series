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

from .archive import ArchiveRecord, EvolutionArchive
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
from .store import V2RunStore


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
    """Immutable evaluator-only aggregates, bound to a single budget stage."""

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
    """Single-writer Host: durable intent precedes atomic pointer replacement.

    History is a hash chain. Recovery only applies its last verified event when
    the pointer is either that event's prior or desired identity. Other drift
    fails closed; no heuristic repair or human-approval flag is supported.
    """

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

    def history(self) -> tuple[dict[str, object], ...]:
        path = self._kernel.store.root / "promotion_history.jsonl"
        if not path.exists():
            return ()
        data = path.read_bytes()
        if not data or not data.endswith(b"\n"):
            raise KernelAuthorityError("promotion history is truncated")
        result = []
        active = None
        known = set()
        for line in data.splitlines(keepends=True):
            try:
                event = strict_json_loads(
                    line.decode("utf-8"), context="promotion history"
                )
                _require_exact_schema(
                    event, self._EVENT_FIELDS, field="promotion event"
                )
                if canonical_v2_bytes(event) != line:
                    raise ValueError("noncanonical promotion event")
                body = {k: v for k, v in event.items() if k != "event_sha256"}
                if event["event_sha256"] != fingerprint_payload(body):
                    raise ValueError("promotion event digest mismatch")
                if (
                    type(event["schema_version"]) is not int
                    or event["schema_version"] != 1
                ):
                    raise ValueError("promotion event schema mismatch")
                if type(event["sequence"]) is not int or event["sequence"] != len(
                    result
                ):
                    raise ValueError("promotion event sequence mismatch")
                previous = result[-1]["event_sha256"] if result else None
                if (
                    event["previous_event_sha256"] != previous
                    or event["prior_bundle_sha256"] != active
                ):
                    raise ValueError("promotion history prior identity mismatch")
                bundle = self._kernel._verified_bundle(event["bundle_sha256"])
                if (
                    event["acceptance_evidence_sha256"]
                    != bundle.acceptance_evidence_sha256
                ):
                    raise ValueError("promotion acceptance evidence mismatch")
                if not result:
                    if (
                        event["action"] != "seed"
                        or bundle.fingerprint() != self._kernel._seed_sha
                        or event["reason"] != "seed"
                    ):
                        raise ValueError(
                            "promotion history must begin at committed seed"
                        )
                elif event["action"] == "activate":
                    if (
                        bundle.parent_bundle_sha256 != active
                        or event["reason"] != "accepted"
                    ):
                        raise ValueError("activation Parent mismatch")
                elif event["action"] == "rollback":
                    if (
                        bundle.fingerprint() not in known
                        or event["reason"] not in self._ROLLBACK_REASONS
                        or bundle.fingerprint() == active
                    ):
                        raise ValueError(
                            "rollback requires known prior active and closed reason"
                        )
                else:
                    raise ValueError("invalid promotion action")
                active = bundle.fingerprint()
                known.add(active)
                result.append(event)
            except (TypeError, ValueError, UnicodeError) as error:
                raise KernelAuthorityError(
                    f"invalid promotion history: {error}"
                ) from error
        return tuple(result)

    def reconcile(self) -> EvolutionBundleV2:
        history = self.history()
        if not history:
            raise KernelAuthorityError("missing promotion history")
        event = history[-1]
        bundle = self._kernel._verified_bundle(event["bundle_sha256"])
        path = self._kernel.store.root / "accepted_bundle.json"
        if path.exists():
            pointer = EvolutionBundleV2.from_payload(_read(path))
            if pointer.fingerprint() == bundle.fingerprint():
                return bundle
            if pointer.fingerprint() != event["prior_bundle_sha256"]:
                raise KernelAuthorityError(
                    "active pointer drift from promotion history"
                )
        elif event["action"] != "seed":
            raise KernelAuthorityError("active pointer missing after seed")
        self._kernel.store.publish_active_bundle(bundle.to_payload())
        return bundle

    def _publish(self, action, bundle, reason, history):
        event = {
            "schema_version": 1,
            "sequence": len(history),
            "previous_event_sha256": history[-1]["event_sha256"] if history else None,
            "action": action,
            "prior_bundle_sha256": history[-1]["bundle_sha256"] if history else None,
            "bundle_sha256": bundle.fingerprint(),
            "acceptance_evidence_sha256": bundle.acceptance_evidence_sha256,
            "reason": reason,
        }
        event["event_sha256"] = fingerprint_payload(event)
        self._kernel.store.append_promotion(event)
        self._kernel.store.publish_active_bundle(bundle.to_payload())
        return bundle

    def activate(self, bundle: EvolutionBundleV2) -> EvolutionBundleV2:
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
        if reason not in self._ROLLBACK_REASONS:
            raise KernelAuthorityError(
                "rollback requires a closed canary/integrity reason"
            )
        active = self.reconcile()
        history = self.history()
        if (
            bundle_sha256 is None
            and history[-1]["action"] == "rollback"
            and history[-1]["reason"] == reason
        ):
            self._kernel._checkpoint()
            return active
        target = bundle_sha256 or history[-1]["prior_bundle_sha256"]
        if target is None or target not in {
            event["bundle_sha256"] for event in history[:-1]
        }:
            raise KernelAuthorityError("rollback requires a known prior active Bundle")
        bundle = self._kernel._verified_bundle(target)
        if bundle.fingerprint() == active.fingerprint():
            return bundle
        result = self._publish("rollback", bundle, reason, history)
        self._kernel._checkpoint()
        return result


class EvolutionKernel:
    """Immutable protocol gate over trusted evaluations and a resource ledger."""

    def __init__(
        self,
        store: V2RunStore,
        protocol: KernelProtocolCommitment,
        budget: BudgetLedger,
        *,
        seed: EvolutionBundleV2,
    ):
        if type(protocol) is not KernelProtocolCommitment:
            raise KernelAuthorityError("kernel requires the full protocol commitment")
        if seed.generation != 0 or seed.protocol_fingerprint != protocol.fingerprint():
            raise KernelAuthorityError("seed protocol commitment mismatch")
        self.store, self.protocol, self.budget = store, protocol, budget
        self._seed_sha = seed.fingerprint()
        self._runtimes = seed.runtime_fingerprints
        self._closed: dict[str, tuple[ClosedEvaluation, dict[str, object]]] = {}
        manifest = {
            "system": "evolution_v2",
            "kernel_protocol": protocol.to_payload(),
            "runtime_fingerprints": dict(self._runtimes),
            "seed_bundle_sha256": self._seed_sha,
            "budget_plan_sha256": budget.plan.fingerprint(),
        }
        path = store.root / "run_manifest.json"
        if path.exists():
            existing = _read(path)
            if any(existing.get(key) != value for key, value in manifest.items()):
                raise KernelAuthorityError(
                    "run manifest protocol or authority mismatch"
                )
        else:
            store.write_run_manifest(manifest)
        store.write_budget_plan(budget.plan.to_payload())
        self.archive = EvolutionArchive(store.root / "archive")
        self.promotion_host = PromotionHost(self)
        history = self.promotion_host.history()
        checkpoint_path = store.root / "checkpoint.json"
        if checkpoint_path.exists():
            self._validate_checkpoint(_read(checkpoint_path), history)
        if not history:
            if (store.root / "accepted_bundle.json").exists():
                raise KernelAuthorityError(
                    "active pointer exists without promotion history"
                )
            if self.archive.index.exists():
                self._verified_bundle(self._seed_sha)
            else:
                self.archive.append(
                    _BundleArtifact(seed, "bundle"), _record(seed, "bundle", ())
                )
            self.promotion_host._publish("seed", seed, "seed", ())
        self.promotion_host.reconcile()
        self._checkpoint()

    @classmethod
    def resume(
        cls, store: V2RunStore, plan: BudgetPlan, *, monotonic: Callable[[], float]
    ):
        manifest = _read(store.root / "run_manifest.json")
        try:
            protocol = KernelProtocolCommitment.from_payload(
                manifest["kernel_protocol"]
            )
            seed_sha = require_sha256(manifest["seed_bundle_sha256"], "seed SHA")
            seed = EvolutionBundleV2.from_payload(
                _read(store.root / "archive" / "objects" / f"{seed_sha}.json", seed_sha)
            )
            checkpoint = _read(store.root / "checkpoint.json")
            budget = BudgetLedger.resume(
                plan, checkpoint["budget"], monotonic=monotonic
            )
            if fingerprint_payload(
                {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
            ) != checkpoint.get("checkpoint_sha256"):
                raise ValueError("kernel checkpoint digest mismatch")
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(
                f"resume authority verification failed: {error}"
            ) from error
        return cls(store, protocol, budget, seed=seed)

    @staticmethod
    def stage_id(child: EvolutionBundleV2) -> str:
        return f"evaluation:{child.fingerprint()}"

    def _permit(self, child, permit):
        if (
            type(permit) is not StagePermit
            or not permit.allowed
            or permit.reason is not None
        ):
            raise KernelAuthorityError("transition requires a live stage permit")
        reservations = self.budget.checkpoint()["open_reservations"]
        if not any(
            r["reservation_sha256"] == permit.reservation_sha256
            and r["stage_id"] == self.stage_id(child)
            for r in reservations
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
        """Seal trusted evaluator output; ``status`` describes Train only.

        The caller owns the trusted evaluation boundary. This method and its
        returned object must never be supplied to a proposer or candidate.
        Actual use is consumed by evaluate_transition, including rejection.
        """
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
        self._closed[reservation] = (result, train)
        return result

    def evaluate_transition(
        self,
        parent: EvolutionBundleV2,
        child: EvolutionBundleV2,
        *,
        target: MutationTarget,
        evaluation: ClosedEvaluation,
        permit: StagePermit | None = None,
    ) -> EvolutionBundleV2:
        # Active identity is checked first, but a live trusted evaluation is
        # still charged in finally if any subsequent authority check rejects it.
        active = self.active_bundle()
        reservation = self._permit(child, permit)
        issued = self._closed.get(reservation)
        if issued is None:
            raise KernelAuthorityError(
                "closed evaluation was not issued by this kernel"
            )
        trusted, train = issued
        closed = False
        try:
            if parent.fingerprint() != active.fingerprint():
                raise KernelAuthorityError("transition Parent is not active")
            validate_child_scope(parent, child, target)
            if (
                parent.protocol_fingerprint != self.protocol.fingerprint()
                or child.protocol_fingerprint != self.protocol.fingerprint()
            ):
                raise KernelAuthorityError("transition protocol commitment mismatch")
            if dict(parent.runtime_fingerprints) != dict(self._runtimes):
                raise KernelAuthorityError("transition runtime mismatch")
            if (
                evaluation is not trusted
                or evaluation.parent_bundle_sha256 != parent.fingerprint()
                or evaluation.candidate_bundle_sha256 != child.fingerprint()
            ):
                raise KernelAuthorityError(
                    "closed evaluation authority or Bundle binding mismatch"
                )
            outcome = self.budget.close_stage(
                permit, ResourceUse.from_payload(trusted.resource_use)
            )
            closed = True
            self._checkpoint()
            self.store.write_evaluation(child.fingerprint(), "train", train)
            self.store.write_evaluation(
                child.fingerprint(), "closed", evaluation.to_payload()
            )
            self.archive.append(
                _BundleArtifact(child, "bundle"),
                _record(child, "bundle", (parent.fingerprint(),), train),
            )
            snapshot = self.archive.snapshot_sha256()
            decision = (
                "accept"
                if outcome.allowed
                and evaluation.status == "passed"
                and evaluation.dev_comparison["passed"]
                else "reject"
            )
            evidence = AcceptanceEvidence(
                1,
                parent.fingerprint(),
                child.fingerprint(),
                target,
                self.protocol.fingerprint(),
                self._runtimes,
                evaluation.fingerprint(),
                evaluation.train_evaluation_sha256,
                evaluation.dev_comparison,
                outcome.allowed,
                decision,
                snapshot,
                parent.scheduler_state_sha256,
            )
            self.store.write_acceptance(evidence.fingerprint(), evidence.to_payload())
            evidence = self.load_acceptance(evidence.fingerprint())
            # Progress and archive carry Train results only, never Dev decisions.
            self.store.append_progress(
                {
                    "candidate_bundle_sha256": child.fingerprint(),
                    "train_evaluation_sha256": evaluation.train_evaluation_sha256,
                    "status": evaluation.status,
                    "resource_use": dict(evaluation.resource_use),
                }
            )
            if evidence.decision == "reject":
                return parent
            sealed = _seal_acceptance(
                parent, child, target, evidence, snapshot, parent.scheduler_state_sha256
            )
            self.archive.append(
                _BundleArtifact(sealed, "acceptance_seal"),
                _record(sealed, "acceptance_seal", (child.fingerprint(),), train),
            )
            self.promotion_host.activate(sealed)
            return sealed
        except (ValueError, TypeError) as error:
            raise KernelAuthorityError(str(error)) from error
        finally:
            if not closed:
                self.budget.close_stage(
                    permit, ResourceUse.from_payload(trusted.resource_use)
                )
            self._closed.pop(reservation, None)
            # Do not reconcile a pending publication here: a failed pointer
            # write must stay failed until explicit Host retry or restart.
            self._checkpoint()

    def load_acceptance(self, identity: str) -> AcceptanceEvidence:
        try:
            require_sha256(identity, "acceptance evidence SHA")
            evidence = AcceptanceEvidence.from_payload(
                _read(self.store.root / "acceptance" / f"{identity}.json", identity)
            )
            evaluation = ClosedEvaluation.from_payload(
                _read(
                    self.store.root
                    / "evaluations"
                    / evidence.candidate_bundle_sha256
                    / "closed.json",
                    evidence.evaluation_sha256,
                )
            )
            if (
                evaluation.reservation_sha256
                not in self.budget.checkpoint()["closed_reservation_sha256s"]
            ):
                raise ValueError("acceptance requires an accounted budget reservation")
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
                if evidence.budget_allowed
                and evaluation.status == "passed"
                and evidence.dev_comparison["passed"]
                else "reject"
            )
            if (
                evidence.decision != decision
                or evidence.protocol_fingerprint != self.protocol.fingerprint()
                or evidence.runtime_fingerprints != self._runtimes
            ):
                raise ValueError("acceptance decision/protocol/runtime mismatch")
            self.archive.lineage(evidence.candidate_bundle_sha256)
            parent = EvolutionBundleV2.from_payload(
                _read(
                    self.archive.objects / f"{evidence.parent_bundle_sha256}.json",
                    evidence.parent_bundle_sha256,
                )
            )
            provisional = EvolutionBundleV2.from_payload(
                _read(
                    self.archive.objects / f"{evidence.candidate_bundle_sha256}.json",
                    evidence.candidate_bundle_sha256,
                )
            )
            validate_child_scope(parent, provisional, evidence.target)
            if (
                parent.protocol_fingerprint != self.protocol.fingerprint()
                or parent.runtime_fingerprints != self._runtimes
            ):
                raise ValueError("acceptance Parent protocol/runtime mismatch")
            prefix = hashlib.sha256()
            expected_record = _record(
                provisional, "bundle", (parent.fingerprint(),), train
            ).to_payload()
            for line in self.archive.index.read_bytes().splitlines(keepends=True):
                prefix.update(line)
                record = strict_json_loads(
                    line.decode("utf-8"), context="archive index"
                )["record"]
                if record["artifact_sha256"] == provisional.fingerprint():
                    if (
                        record != expected_record
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

    def _verified_bundle(self, identity: str) -> EvolutionBundleV2:
        try:
            require_sha256(identity, "Bundle SHA")
            self.archive.lineage(identity)  # Revalidates every indexed object.
            bundle = EvolutionBundleV2.from_payload(
                _read(self.archive.objects / f"{identity}.json", identity)
            )
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
            evidence = self.load_acceptance(bundle.acceptance_evidence_sha256)
            if evidence.decision != "accept":
                raise ValueError("Bundle evidence does not accept")
            parent = self._verified_bundle(evidence.parent_bundle_sha256)
            provisional = EvolutionBundleV2.from_payload(
                _read(
                    self.archive.objects / f"{evidence.candidate_bundle_sha256}.json",
                    evidence.candidate_bundle_sha256,
                )
            )
            self.archive.lineage(evidence.candidate_bundle_sha256)
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
            prefix = hashlib.sha256()
            found_provisional = False
            for line in self.archive.index.read_bytes().splitlines(keepends=True):
                record = strict_json_loads(
                    line.decode("utf-8"), context="archive index"
                )["record"]
                prefix.update(line)
                if record["artifact_sha256"] == provisional.fingerprint():
                    if (
                        record["artifact_kind"] != "bundle"
                        or record["parent_sha256s"] != [parent.fingerprint()]
                        or prefix.hexdigest() != evidence.archive_snapshot_sha256
                    ):
                        raise ValueError(
                            "provisional archive snapshot/lineage mismatch"
                        )
                    found_provisional = True
                if record["artifact_sha256"] == identity:
                    if (
                        not found_provisional
                        or record["artifact_kind"] != "acceptance_seal"
                        or record["parent_sha256s"] != [provisional.fingerprint()]
                    ):
                        raise ValueError("acceptance seal archive lineage mismatch")
            return bundle
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(f"invalid archived Bundle: {error}") from error

    def active_bundle(self) -> EvolutionBundleV2:
        return self.promotion_host.reconcile()

    def _validate_checkpoint(self, checkpoint, history):
        try:
            _require_exact_schema(
                checkpoint,
                (
                    "schema_version",
                    "budget",
                    "archive_snapshot_sha256",
                    "active_bundle_sha256",
                    "checkpoint_sha256",
                ),
                field="kernel checkpoint",
            )
            if (
                type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 1
            ):
                raise ValueError("checkpoint schema mismatch")
            if (
                fingerprint_payload(
                    {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
                )
                != checkpoint["checkpoint_sha256"]
            ):
                raise ValueError("checkpoint digest mismatch")
            if not history or checkpoint["active_bundle_sha256"] not in (
                history[-1]["bundle_sha256"],
                history[-1]["prior_bundle_sha256"],
            ):
                raise ValueError("checkpoint active identity mismatch")
            prefix = hashlib.sha256()
            snapshots = {prefix.hexdigest()}
            for line in self.archive.index.read_bytes().splitlines(keepends=True):
                prefix.update(line)
                snapshots.add(prefix.hexdigest())
            if checkpoint["archive_snapshot_sha256"] not in snapshots:
                raise ValueError("checkpoint archive snapshot is not a verified prefix")
            durable = checkpoint["budget"]
            current = self.budget.checkpoint()
            for key in current:
                if (
                    key not in ("checkpoint_sha256", "prior_elapsed_wall_seconds")
                    and current[key] != durable[key]
                ):
                    raise ValueError("checkpoint budget does not match supplied ledger")
            if (
                current["prior_elapsed_wall_seconds"]
                < durable["prior_elapsed_wall_seconds"]
            ):
                raise ValueError("checkpoint budget elapsed time regressed")
        except (KeyError, TypeError, ValueError) as error:
            raise KernelAuthorityError(f"invalid checkpoint: {error}") from error

    def _checkpoint(self):
        path = self.store.root / "accepted_bundle.json"
        active_sha = fingerprint_payload(_read(path)) if path.exists() else None
        checkpoint = {
            "schema_version": 1,
            "budget": self.budget.checkpoint(),
            "archive_snapshot_sha256": self.archive.snapshot_sha256(),
            "active_bundle_sha256": active_sha,
        }
        checkpoint["checkpoint_sha256"] = fingerprint_payload(checkpoint)
        self.store.write_checkpoint(checkpoint)


__all__ = [
    "AcceptanceEvidence",
    "ClosedEvaluation",
    "EvolutionKernel",
    "KernelAuthorityError",
    "PromotionHost",
]
