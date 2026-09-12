"""Strict configuration for the bounded cooperative research prototype."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from common.payload import strict_json_loads

from ..budget import ResourceUse
from ..contracts import (
    EvolutionV2Config,
    _require_exact_schema,
    canonical_v2_bytes,
)


_FIELDS = (
    "schema_version",
    "control",
    "max_steps",
    "children_per_step",
    "discount",
    "task_cost_weight",
    "metric_cap",
    "acceptance_tolerance",
    "resource_ceilings",
)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class CooperativeConfigV2:
    schema_version: int
    control: EvolutionV2Config
    max_steps: int
    children_per_step: int
    discount: float
    task_cost_weight: float
    metric_cap: float
    acceptance_tolerance: float
    resource_ceilings: ResourceUse

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")
        if not isinstance(self.control, EvolutionV2Config):
            raise ValueError("control must be an EvolutionV2Config")
        if self.control.profile == "public":
            raise ValueError("cooperative control cannot use the Public profile")
        if self.control.runner != "production":
            raise ValueError("cooperative control runner must be production")
        if type(self.max_steps) is not int or self.max_steps != 4:
            raise ValueError("cooperative max_steps must be exactly 4")
        if type(self.children_per_step) is not int or self.children_per_step != 1:
            raise ValueError("cooperative children_per_step must be exactly 1")
        discount = _finite_number(self.discount, "discount")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        cost = _finite_number(self.task_cost_weight, "task_cost_weight")
        if cost < 0.0:
            raise ValueError("task_cost_weight must be non-negative")
        cap = _finite_number(self.metric_cap, "metric_cap")
        if cap <= 0.0:
            raise ValueError("metric_cap must be positive")
        tolerance = _finite_number(self.acceptance_tolerance, "acceptance_tolerance")
        if tolerance != 1e-12:
            raise ValueError("acceptance_tolerance must be exactly 1e-12")
        if not isinstance(self.resource_ceilings, ResourceUse):
            raise ValueError("resource_ceilings must be ResourceUse")
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "task_cost_weight", cost)
        object.__setattr__(self, "metric_cap", cap)
        object.__setattr__(self, "acceptance_tolerance", tolerance)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CooperativeConfigV2":
        values = _require_exact_schema(payload, _FIELDS, field="cooperative config")
        control = values["control"]
        ceilings = values["resource_ceilings"]
        if not isinstance(control, Mapping):
            raise ValueError("control must be an object with the exact schema")
        if not isinstance(ceilings, Mapping):
            raise ValueError(
                "resource_ceilings must be an object with the exact schema"
            )
        return cls(
            schema_version=values["schema_version"],  # type: ignore[arg-type]
            control=EvolutionV2Config.from_payload(control),
            max_steps=values["max_steps"],  # type: ignore[arg-type]
            children_per_step=values["children_per_step"],  # type: ignore[arg-type]
            discount=values["discount"],  # type: ignore[arg-type]
            task_cost_weight=values["task_cost_weight"],  # type: ignore[arg-type]
            metric_cap=values["metric_cap"],  # type: ignore[arg-type]
            acceptance_tolerance=values["acceptance_tolerance"],  # type: ignore[arg-type]
            resource_ceilings=ResourceUse.from_payload(ceilings),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "control": self.control.to_payload(),
            "max_steps": self.max_steps,
            "children_per_step": self.children_per_step,
            "discount": self.discount,
            "task_cost_weight": self.task_cost_weight,
            "metric_cap": self.metric_cap,
            "acceptance_tolerance": self.acceptance_tolerance,
            "resource_ceilings": self.resource_ceilings.to_payload(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())


def load_cooperative_config(path: str | Path) -> CooperativeConfigV2:
    source = Path(path)
    payload = strict_json_loads(
        source.read_text(encoding="utf-8"),
        context=f"Cooperative Evolution V2 config {source}",
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"Cooperative Evolution V2 config {source} must be an object")
    return CooperativeConfigV2.from_payload(payload)


__all__ = ["CooperativeConfigV2", "load_cooperative_config"]
