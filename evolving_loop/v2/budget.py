"""Resumable, content-bound resource accounting for Evolution V2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .contracts import (
    EvolutionV2Config,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)


_FLOAT_RESOURCES = ("wall_seconds", "gpu_seconds")
_INTEGER_RESOURCES = (
    "task_executions",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "subprocesses",
    "artifact_bytes",
)
_RESOURCE_FIELDS = (
    "wall_seconds",
    "task_executions",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "gpu_seconds",
    "subprocesses",
    "artifact_bytes",
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "prior_elapsed_wall_seconds",
        "charged_use",
        "open_reservations",
        "closed_stage_ids",
        "finalization_started",
        "exhausted_reason",
        "checkpoint_sha256",
    }
)
_RESERVATION_FIELDS = frozenset({"stage_id", "estimate", "reservation_sha256"})


class BudgetContractError(ValueError):
    """Raised when a budget plan, permit, or checkpoint violates its contract."""


def _require_finite_float(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative float")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_exact_fields(
    payload: object, expected: frozenset[str], field_name: str
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
        raise BudgetContractError(
            f"{field_name} must be an object with the exact schema"
        )
    if set(payload) != expected:
        raise BudgetContractError(f"{field_name} must use the exact schema")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class ResourceUse:
    """Exact, non-negative accounting values for every governed resource."""

    wall_seconds: float = 0.0
    task_executions: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0
    subprocesses: int = 0
    artifact_bytes: int = 0

    def __post_init__(self) -> None:
        for name in _FLOAT_RESOURCES:
            _require_finite_float(getattr(self, name), name)
        for name in _INTEGER_RESOURCES:
            _require_non_negative_int(getattr(self, name), name)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return _RESOURCE_FIELDS

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ResourceUse":
        values = _require_exact_fields(
            payload, frozenset(_RESOURCE_FIELDS), "resource_use"
        )
        return cls(**values)  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in _RESOURCE_FIELDS}

    def __add__(self, other: "ResourceUse") -> "ResourceUse":
        if not isinstance(other, ResourceUse):
            return NotImplemented
        return ResourceUse(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in _RESOURCE_FIELDS
            }
        )


def _coerce_resource_use(value: object, field_name: str) -> ResourceUse:
    if isinstance(value, ResourceUse):
        return value
    if isinstance(value, Mapping):
        try:
            return ResourceUse(**dict(value))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid {field_name}: {error}") from error
    raise ValueError(f"{field_name} must be ResourceUse or a resource mapping")


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """Immutable time reserve and resource ceilings for one evolution run."""

    hard_limit_seconds: int
    finalization_reserve_fraction: float
    ceilings: ResourceUse
    schema_version: int = field(default=1, init=False)
    search_deadline_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        if type(self.hard_limit_seconds) is not int or self.hard_limit_seconds <= 0:
            raise ValueError("hard_limit_seconds must be a positive integer")
        reserve = self.finalization_reserve_fraction
        if type(reserve) is not float or not math.isfinite(reserve):
            raise ValueError("finalization_reserve_fraction must be a finite float")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("finalization_reserve_fraction must be in [0, 1)")
        if not isinstance(self.ceilings, ResourceUse):
            raise ValueError("ceilings must be ResourceUse")
        object.__setattr__(
            self,
            "search_deadline_seconds",
            self.hard_limit_seconds * (1.0 - reserve),
        )

    @classmethod
    def from_config(
        cls, config: EvolutionV2Config, *, ceilings: ResourceUse | Mapping[str, object]
    ) -> "BudgetPlan":
        if not isinstance(config, EvolutionV2Config):
            raise ValueError("config must be EvolutionV2Config")
        return cls(
            hard_limit_seconds=config.hard_limit_seconds,
            finalization_reserve_fraction=config.finalization_reserve_fraction,
            ceilings=_coerce_resource_use(ceilings, "ceilings"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hard_limit_seconds": self.hard_limit_seconds,
            "finalization_reserve_fraction": self.finalization_reserve_fraction,
            "search_deadline_seconds": self.search_deadline_seconds,
            "ceilings": self.ceilings.to_payload(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class StagePermit:
    """Typed decision returned when a stage is opened or closed."""

    allowed: bool
    reason: str | None
    reservation_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise ValueError("allowed must be a boolean")
        if self.reason is not None and (
            type(self.reason) is not str or not self.reason
        ):
            raise ValueError("reason must be a non-empty string or None")
        if self.reservation_sha256 is not None:
            require_sha256(self.reservation_sha256, "reservation_sha256")


@dataclass(frozen=True, slots=True)
class _OpenReservation:
    stage_id: str
    estimate: ResourceUse
    reservation_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "estimate": self.estimate.to_payload(),
            "reservation_sha256": self.reservation_sha256,
        }


class BudgetLedger:
    """Mutable Host ledger whose complete state round-trips through checkpoints."""

    def __init__(self, plan: BudgetPlan, *, monotonic: Callable[[], float]) -> None:
        if not isinstance(plan, BudgetPlan):
            raise ValueError("plan must be BudgetPlan")
        if not callable(monotonic):
            raise ValueError("monotonic must be callable")
        self.plan = plan
        self._monotonic = monotonic
        self._origin = self._read_clock()
        self._prior_elapsed_wall_seconds = 0.0
        self._charged_use = ResourceUse()
        self._open_by_stage: dict[str, _OpenReservation] = {}
        self._open_stage_by_sha: dict[str, str] = {}
        self._closed_stage_ids: set[str] = set()
        self._closed_reservation_sha256s: set[str] = set()
        self._finalization_started = False
        self._exhausted_reason: str | None = None

    def _read_clock(self) -> float:
        value = self._monotonic()
        if type(value) not in (int, float) or isinstance(value, bool):
            raise BudgetContractError("monotonic clock must return a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise BudgetContractError("monotonic clock must return a finite number")
        return numeric

    @property
    def elapsed_wall_seconds(self) -> float:
        delta = self._read_clock() - self._origin
        if delta < 0.0:
            raise BudgetContractError("monotonic clock moved backwards")
        return self._prior_elapsed_wall_seconds + delta

    @property
    def charged_use(self) -> ResourceUse:
        return self._charged_use

    @property
    def finalization_started(self) -> bool:
        return self._finalization_started

    def _pending_use(self) -> ResourceUse:
        total = ResourceUse()
        for reservation in self._open_by_stage.values():
            total = total + reservation.estimate
        return total

    def _denial_reason(self, estimate: ResourceUse) -> str | None:
        if self._exhausted_reason is not None:
            return self._exhausted_reason
        if self._finalization_started:
            return "finalization_started"
        elapsed = self.elapsed_wall_seconds
        if elapsed >= self.plan.search_deadline_seconds:
            return "finalization_reserve"
        pending = self._pending_use()
        if (
            elapsed + pending.wall_seconds + estimate.wall_seconds
            > self.plan.search_deadline_seconds
        ):
            return "finalization_reserve"
        committed = self._charged_use + pending + estimate
        for name in _RESOURCE_FIELDS:
            if getattr(committed, name) > getattr(self.plan.ceilings, name):
                return f"{name}_exhausted"
        return None

    def can_open_stage(
        self, estimate: ResourceUse | Mapping[str, object]
    ) -> StagePermit:
        requested = _coerce_resource_use(estimate, "estimate")
        reason = self._denial_reason(requested)
        return StagePermit(reason is None, reason, None)

    @staticmethod
    def _validate_stage_id(stage_id: object) -> str:
        if type(stage_id) is not str or not stage_id:
            raise ValueError("stage_id must be a non-empty string")
        return stage_id

    def _reservation_sha256(self, stage_id: str, estimate: ResourceUse) -> str:
        return fingerprint_payload(
            {
                "schema_version": 1,
                "plan_sha256": self.plan.fingerprint(),
                "stage_id": stage_id,
                "estimate": estimate.to_payload(),
            }
        )

    def reserve_stage(
        self, stage_id: str, estimate: ResourceUse | Mapping[str, object]
    ) -> StagePermit:
        identity = self._validate_stage_id(stage_id)
        requested = _coerce_resource_use(estimate, "estimate")
        if identity in self._open_by_stage:
            return StagePermit(False, "stage_already_open", None)
        if identity in self._closed_stage_ids:
            return StagePermit(False, "stage_closed", None)
        reason = self._denial_reason(requested)
        if reason is not None:
            self._closed_stage_ids.add(identity)
            return StagePermit(False, reason, None)
        reservation_sha256 = self._reservation_sha256(identity, requested)
        reservation = _OpenReservation(identity, requested, reservation_sha256)
        self._open_by_stage[identity] = reservation
        self._open_stage_by_sha[reservation_sha256] = identity
        return StagePermit(True, None, reservation_sha256)

    def _permit_sha(self, reservation: StagePermit | str) -> str:
        if isinstance(reservation, StagePermit):
            if not reservation.allowed or reservation.reservation_sha256 is None:
                raise BudgetContractError("reservation must be an allowed stage permit")
            return reservation.reservation_sha256
        try:
            return require_sha256(reservation, "reservation_sha256")
        except ValueError as error:
            raise BudgetContractError(str(error)) from error

    def close_stage(
        self,
        reservation: StagePermit | str,
        actual: ResourceUse | Mapping[str, object],
    ) -> StagePermit:
        consumed = _coerce_resource_use(actual, "actual")
        reservation_sha256 = self._permit_sha(reservation)
        if reservation_sha256 in self._closed_reservation_sha256s:
            return StagePermit(False, "stage_closed", None)
        stage_id = self._open_stage_by_sha.get(reservation_sha256)
        if stage_id is None:
            raise BudgetContractError("unknown reservation SHA")
        opened = self._open_by_stage.pop(stage_id)
        del self._open_stage_by_sha[reservation_sha256]
        self._closed_stage_ids.add(stage_id)
        self._closed_reservation_sha256s.add(reservation_sha256)
        self._charged_use = self._charged_use + consumed

        overrun = any(
            getattr(consumed, name) > getattr(opened.estimate, name)
            for name in _RESOURCE_FIELDS
        ) or any(
            getattr(self._charged_use, name) > getattr(self.plan.ceilings, name)
            for name in _RESOURCE_FIELDS
        )
        if overrun:
            self._exhausted_reason = "budget_overrun"
            return StagePermit(False, "budget_overrun", None)
        return StagePermit(True, None, None)

    def charge(self, actual: ResourceUse | Mapping[str, object]) -> StagePermit:
        consumed = _coerce_resource_use(actual, "actual")
        self._charged_use = self._charged_use + consumed
        for name in _RESOURCE_FIELDS:
            if getattr(self._charged_use, name) > getattr(self.plan.ceilings, name):
                self._exhausted_reason = f"{name}_exhausted"
                return StagePermit(False, self._exhausted_reason, None)
        return StagePermit(True, None, None)

    def begin_finalization(self) -> None:
        self._finalization_started = True

    def checkpoint(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": self.plan.fingerprint(),
            "prior_elapsed_wall_seconds": self.elapsed_wall_seconds,
            "charged_use": self._charged_use.to_payload(),
            "open_reservations": [
                self._open_by_stage[stage_id].to_payload()
                for stage_id in sorted(self._open_by_stage)
            ],
            "closed_stage_ids": sorted(self._closed_stage_ids),
            "finalization_started": self._finalization_started,
            "exhausted_reason": self._exhausted_reason,
        }
        payload["checkpoint_sha256"] = fingerprint_payload(payload)
        return payload

    to_checkpoint = checkpoint

    @staticmethod
    def checkpoint_sha256(checkpoint: Mapping[str, object]) -> str:
        payload = dict(checkpoint)
        payload.pop("checkpoint_sha256", None)
        return fingerprint_payload(payload)

    @classmethod
    def resume(
        cls,
        plan: BudgetPlan,
        checkpoint: Mapping[str, object],
        *,
        monotonic: Callable[[], float],
    ) -> "BudgetLedger":
        if not isinstance(plan, BudgetPlan):
            raise ValueError("plan must be BudgetPlan")
        values = _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "checkpoint")
        try:
            checkpoint_plan_sha = require_sha256(values["plan_sha256"], "plan SHA")
        except ValueError as error:
            raise BudgetContractError(str(error)) from error
        if checkpoint_plan_sha != plan.fingerprint():
            raise BudgetContractError(
                "checkpoint plan SHA does not match supplied plan"
            )
        try:
            claimed_checkpoint_sha = require_sha256(
                values["checkpoint_sha256"], "checkpoint SHA"
            )
        except ValueError as error:
            raise BudgetContractError(str(error)) from error
        if claimed_checkpoint_sha != cls.checkpoint_sha256(values):
            raise BudgetContractError("checkpoint SHA mismatch")
        if values["schema_version"] != 1 or type(values["schema_version"]) is not int:
            raise BudgetContractError("checkpoint schema_version must be exactly 1")
        try:
            prior_elapsed = _require_finite_float(
                values["prior_elapsed_wall_seconds"], "prior_elapsed_wall_seconds"
            )
            charged = ResourceUse.from_payload(values["charged_use"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise BudgetContractError(str(error)) from error
        finalization = values["finalization_started"]
        if type(finalization) is not bool:
            raise BudgetContractError("finalization_started must be a boolean")
        exhausted_reason = values["exhausted_reason"]
        if exhausted_reason is not None and (
            type(exhausted_reason) is not str or not exhausted_reason
        ):
            raise BudgetContractError("exhausted_reason must be a string or None")

        closed_value = values["closed_stage_ids"]
        if not isinstance(closed_value, list) or any(
            type(stage_id) is not str or not stage_id for stage_id in closed_value
        ):
            raise BudgetContractError("closed_stage_ids must be a list of stage IDs")
        if len(closed_value) != len(set(closed_value)):
            raise BudgetContractError("closed_stage_ids must be unique")
        closed = set(closed_value)

        open_value = values["open_reservations"]
        if not isinstance(open_value, list):
            raise BudgetContractError("open_reservations must be a list")
        open_reservations: list[_OpenReservation] = []
        seen_open: set[str] = set()
        for item in open_value:
            reservation_values = _require_exact_fields(
                item, _RESERVATION_FIELDS, "open reservation"
            )
            stage_id = cls._validate_stage_id(reservation_values["stage_id"])
            if stage_id in seen_open or stage_id in closed:
                raise BudgetContractError(
                    "checkpoint stage IDs must be unique and disjoint"
                )
            try:
                estimate = ResourceUse.from_payload(reservation_values["estimate"])  # type: ignore[arg-type]
                reservation_sha = require_sha256(
                    reservation_values["reservation_sha256"], "reservation SHA"
                )
            except (TypeError, ValueError) as error:
                raise BudgetContractError(str(error)) from error
            expected_sha = fingerprint_payload(
                {
                    "schema_version": 1,
                    "plan_sha256": plan.fingerprint(),
                    "stage_id": stage_id,
                    "estimate": estimate.to_payload(),
                }
            )
            if reservation_sha != expected_sha:
                raise BudgetContractError("reservation SHA mismatch")
            seen_open.add(stage_id)
            open_reservations.append(
                _OpenReservation(stage_id, estimate, reservation_sha)
            )

        ledger = cls(plan, monotonic=monotonic)
        ledger._prior_elapsed_wall_seconds = prior_elapsed
        ledger._charged_use = charged
        ledger._closed_stage_ids = closed
        ledger._finalization_started = finalization
        ledger._exhausted_reason = exhausted_reason  # type: ignore[assignment]
        for reservation in open_reservations:
            ledger._open_by_stage[reservation.stage_id] = reservation
            ledger._open_stage_by_sha[reservation.reservation_sha256] = (
                reservation.stage_id
            )

        if exhausted_reason is None:
            committed = ledger._charged_use + ledger._pending_use()
            if any(
                getattr(committed, name) > getattr(plan.ceilings, name)
                for name in _RESOURCE_FIELDS
            ):
                raise BudgetContractError(
                    "checkpoint resource use exceeds plan ceilings"
                )
        return ledger

    from_checkpoint = resume
