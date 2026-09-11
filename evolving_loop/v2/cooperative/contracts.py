"""Canonical artifacts for cooperative Evolution V2 research runs."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal

from evolving_loop.retrieval_agent.policy import RetrievalGenome

from ..bundle import EvolutionBundleV2, MutationTarget
from ..contracts import (
    _freeze_json_value,
    _require_exact_schema,
    _strict_json_value,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)


ARM_ORDER = ("numerical", "retrieval", "decision", "joint")
_ARM_SET = frozenset(ARM_ORDER)


def _schema_version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be exactly 1")


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite float")
    return value


def _nonempty_text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    return None if value is None else require_sha256(value, field)


def _json_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    plain = _strict_json_value(value, field=field)
    assert isinstance(plain, dict)
    return plain


def _json_object_rows(value: object, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of JSON objects")
    rows = tuple(_json_object(row, f"{field}[]") for row in value)
    return tuple(_freeze_json_value(row) for row in rows)  # type: ignore[return-value]


def _plain(value: object) -> object:
    if isinstance(value, _CanonicalContract):
        return value.to_payload()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class _CanonicalContract:
    __slots__ = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]):
        values = _require_exact_schema(
            payload,
            tuple(field.name for field in fields(cls)),
            field=cls.__name__,
        )
        plain = _strict_json_value(values, field=cls.__name__)
        assert isinstance(plain, dict)
        return cls(**plain)

    def to_payload(self) -> dict[str, object]:
        plain = _strict_json_value(
            {field.name: _plain(getattr(self, field.name)) for field in fields(self)},
            field=type(self).__name__,
        )
        assert isinstance(plain, dict)
        return plain

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class RetrievalModuleV2(_CanonicalContract):
    schema_version: int
    source_release_sha256: str
    genome_payload: Mapping[str, object]
    skills_payload: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        require_sha256(self.source_release_sha256, "source_release_sha256")
        genome = RetrievalGenome.from_payload(
            _json_object(self.genome_payload, "genome_payload")
        )
        skills = _json_object_rows(self.skills_payload, "skills_payload")
        skill_ids = tuple(
            row.get("skill_id") for row in skills if type(row.get("skill_id")) is str
        )
        if any(skill_id not in skill_ids for skill_id in genome.active_skill_ids):
            raise ValueError("active Retrieval Skill IDs must exist in skills_payload")
        object.__setattr__(
            self, "genome_payload", _freeze_json_value(genome.to_payload())
        )
        object.__setattr__(self, "skills_payload", skills)

    @property
    def genome(self) -> RetrievalGenome:
        return RetrievalGenome.from_payload(self.genome_payload)


@dataclass(frozen=True, slots=True)
class DecisionModuleV2(_CanonicalContract):
    schema_version: int
    prompt: str
    skills: tuple[Mapping[str, object], ...]
    enable_evidence_adjustments: bool
    max_evidence_adjustments: int
    aggregation: Literal["last", "mean"]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _nonempty_text(self.prompt, "prompt")
        object.__setattr__(self, "skills", _json_object_rows(self.skills, "skills"))
        if type(self.enable_evidence_adjustments) is not bool:
            raise ValueError("enable_evidence_adjustments must be a boolean")
        adjustments = _nonnegative_int(
            self.max_evidence_adjustments, "max_evidence_adjustments"
        )
        if adjustments > 3:
            raise ValueError("max_evidence_adjustments must be in [0, 3]")
        if self.aggregation not in {"last", "mean"}:
            raise ValueError("aggregation must be one of: last, mean")


@dataclass(frozen=True, slots=True)
class BundleCandidateV2(_CanonicalContract):
    schema_version: int
    target: MutationTarget
    parent_bundle_sha256: str
    operator: str
    numerical_release_sha256: str | None
    numerical_registry_sha256: str | None
    retrieval_release_sha256: str | None
    decision_policy_sha256: str | None

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if self.target not in _ARM_SET:
            raise ValueError("target must be numerical, retrieval, decision, or joint")
        require_sha256(self.parent_bundle_sha256, "parent_bundle_sha256")
        _nonempty_text(self.operator, "operator")
        for name in (
            "numerical_release_sha256",
            "numerical_registry_sha256",
            "retrieval_release_sha256",
            "decision_policy_sha256",
        ):
            _optional_sha256(getattr(self, name), name)

    def to_child(self, parent: EvolutionBundleV2) -> EvolutionBundleV2:
        if self.parent_bundle_sha256 != parent.fingerprint():
            raise ValueError("candidate Parent mismatch")
        changes: dict[str, object] = {}
        if self.numerical_release_sha256 is not None:
            if self.numerical_registry_sha256 is None:
                raise ValueError("Numerical candidate requires release and registry")
            changes["numerical"] = (
                self.numerical_release_sha256,
                self.numerical_registry_sha256,
            )
        elif self.numerical_registry_sha256 is not None:
            raise ValueError("Numerical candidate requires release and registry")
        if self.retrieval_release_sha256 is not None:
            changes["retrieval"] = self.retrieval_release_sha256
        if self.decision_policy_sha256 is not None:
            changes["decision"] = self.decision_policy_sha256
        expected = 2 if self.target == "joint" else 1
        if (self.target == "joint" and len(changes) < expected) or (
            self.target != "joint" and tuple(changes) != (self.target,)
        ):
            raise ValueError("candidate scope does not match target; joint needs at least two")
        return parent.provisional_child(self.target, changes)


@dataclass(frozen=True, slots=True)
class SchedulerArmStateV2(_CanonicalContract):
    attempts: int
    acceptances: int
    discounted_reward_sum: float
    discounted_cost_sum: float

    def __post_init__(self) -> None:
        attempts = _nonnegative_int(self.attempts, "attempts")
        acceptances = _nonnegative_int(self.acceptances, "acceptances")
        if acceptances > attempts:
            raise ValueError("acceptances must not exceed attempts")
        _finite_float(self.discounted_reward_sum, "discounted_reward_sum")
        cost = _finite_float(self.discounted_cost_sum, "discounted_cost_sum")
        if cost < 0.0:
            raise ValueError("discounted_cost_sum must be non-negative")


@dataclass(frozen=True, slots=True)
class CooperativeSchedulerStateV2(_CanonicalContract):
    schema_version: int
    mode: Literal["ucb", "thompson"]
    seed: int
    draw_counter: int
    completed_step: int
    discount: float
    arms: Mapping[str, SchedulerArmStateV2]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if self.mode not in {"ucb", "thompson"}:
            raise ValueError("mode must be ucb or thompson")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        _nonnegative_int(self.draw_counter, "draw_counter")
        _nonnegative_int(self.completed_step, "completed_step")
        discount = _finite_float(self.discount, "discount")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not isinstance(self.arms, Mapping) or not self.arms:
            raise ValueError("arms must be a non-empty object")
        names = tuple(self.arms)
        expected = tuple(name for name in ARM_ORDER if name in self.arms)
        if names != expected or any(name not in _ARM_SET for name in names):
            raise ValueError("arms must be an enabled subset in canonical order")
        parsed = {
            name: (
                value
                if isinstance(value, SchedulerArmStateV2)
                else SchedulerArmStateV2.from_payload(value)  # type: ignore[arg-type]
            )
            for name, value in self.arms.items()
        }
        object.__setattr__(self, "arms", _freeze_json_value(parsed))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "CooperativeSchedulerStateV2":
        values = _require_exact_schema(
            payload,
            tuple(field.name for field in fields(cls)),
            field="CooperativeSchedulerStateV2",
        )
        arms = values["arms"]
        if not isinstance(arms, Mapping):
            raise ValueError("arms must be an object")
        return cls(
            schema_version=values["schema_version"],  # type: ignore[arg-type]
            mode=values["mode"],  # type: ignore[arg-type]
            seed=values["seed"],  # type: ignore[arg-type]
            draw_counter=values["draw_counter"],  # type: ignore[arg-type]
            completed_step=values["completed_step"],  # type: ignore[arg-type]
            discount=values["discount"],  # type: ignore[arg-type]
            arms={
                name: SchedulerArmStateV2.from_payload(value)  # type: ignore[arg-type]
                for name, value in arms.items()
            },
        )


@dataclass(frozen=True, slots=True)
class CooperativeCheckpointV2(_CanonicalContract):
    schema_version: int
    config_sha256: str
    input_sha256s: Mapping[str, str]
    active_bundle_sha256: str
    scheduler_state: CooperativeSchedulerStateV2
    scheduler_state_sha256: str
    next_step: int
    accepted_steps: int
    rejected_steps: int
    completed_candidate_sha256s: tuple[str, ...]
    kernel_checkpoint_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        for name in (
            "config_sha256",
            "active_bundle_sha256",
            "scheduler_state_sha256",
            "kernel_checkpoint_sha256",
            "checkpoint_sha256",
        ):
            require_sha256(getattr(self, name), name)
        inputs = _json_object(self.input_sha256s, "input_sha256s")
        if not inputs:
            raise ValueError("input_sha256s must not be empty")
        for name, digest in inputs.items():
            _nonempty_text(name, "input name")
            require_sha256(digest, f"input_sha256s.{name}")
        state = self.scheduler_state
        if not isinstance(state, CooperativeSchedulerStateV2):
            state = CooperativeSchedulerStateV2.from_payload(state)  # type: ignore[arg-type]
        if state.fingerprint() != self.scheduler_state_sha256:
            raise ValueError("scheduler state SHA mismatch")
        next_step = _nonnegative_int(self.next_step, "next_step")
        accepted = _nonnegative_int(self.accepted_steps, "accepted_steps")
        rejected = _nonnegative_int(self.rejected_steps, "rejected_steps")
        if accepted + rejected != next_step:
            raise ValueError("accepted_steps and rejected_steps must equal next_step")
        if not isinstance(self.completed_candidate_sha256s, (list, tuple)):
            raise ValueError("completed_candidate_sha256s must be a list or tuple")
        completed = tuple(self.completed_candidate_sha256s)
        for digest in completed:
            require_sha256(digest, "completed_candidate_sha256s")
        if len(completed) != len(set(completed)):
            raise ValueError("completed_candidate_sha256s must be unique")
        object.__setattr__(self, "input_sha256s", _freeze_json_value(inputs))
        object.__setattr__(self, "scheduler_state", state)
        object.__setattr__(self, "completed_candidate_sha256s", completed)
        body = self.to_payload()
        body.pop("checkpoint_sha256")
        if fingerprint_payload(body) != self.checkpoint_sha256:
            raise ValueError("cooperative checkpoint body SHA mismatch")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CooperativeCheckpointV2":
        values = _require_exact_schema(
            payload,
            tuple(field.name for field in fields(cls)),
            field="CooperativeCheckpointV2",
        )
        return cls(
            **(values | {
                "scheduler_state": CooperativeSchedulerStateV2.from_payload(
                    values["scheduler_state"]  # type: ignore[arg-type]
                )
            })
        )  # type: ignore[arg-type]

    @classmethod
    def seal(cls, **body: object) -> "CooperativeCheckpointV2":
        plain = _strict_json_value(
            {name: _plain(value) for name, value in body.items()},
            field="CooperativeCheckpointV2",
        )
        assert isinstance(plain, dict)
        return cls.from_payload(
            plain | {"checkpoint_sha256": fingerprint_payload(plain)}
        )


@dataclass(frozen=True, slots=True)
class CooperativeRunResultV2(_CanonicalContract):
    schema_version: int
    status: str
    active_bundle_sha256: str
    scheduler_state_sha256: str
    attempted_arms: tuple[str, ...]
    accepted_steps: int
    rejected_steps: int
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        _nonempty_text(self.status, "status")
        require_sha256(self.active_bundle_sha256, "active_bundle_sha256")
        require_sha256(self.scheduler_state_sha256, "scheduler_state_sha256")
        if not isinstance(self.attempted_arms, (list, tuple)):
            raise ValueError("attempted_arms must be a list or tuple")
        attempted = tuple(self.attempted_arms)
        if any(type(arm) is not str or arm not in _ARM_SET for arm in attempted):
            raise ValueError("attempted_arms contains an unknown scheduler arm")
        accepted = _nonnegative_int(self.accepted_steps, "accepted_steps")
        rejected = _nonnegative_int(self.rejected_steps, "rejected_steps")
        if accepted + rejected != len(attempted):
            raise ValueError("result counts must equal attempted arms")
        if type(self.public_test_accessed) is not bool:
            raise ValueError("public_test_accessed must be a boolean")
        object.__setattr__(self, "attempted_arms", attempted)


__all__ = [
    "ARM_ORDER",
    "BundleCandidateV2",
    "CooperativeCheckpointV2",
    "CooperativeRunResultV2",
    "CooperativeSchedulerStateV2",
    "DecisionModuleV2",
    "RetrievalModuleV2",
    "SchedulerArmStateV2",
]
