"""Shared final-pipeline metrics and acceptance gates for package evolution."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType

from common.metrics import linear_quantile


_SHA256_LENGTH = 64


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class PackageTaskScore:
    """One task's authoritative final Numerical→Retrieval→Decision score."""

    task_id: str
    entity_name: str
    final_smae: float
    final_srmse: float
    final_smae_raw: float
    final_srmse_raw: float
    final_forecast: tuple[float, ...]
    numerical_oracle_smae: float
    numerical_oracle_srmse: float
    numerical_candidate_count: int
    smae_clipped: bool
    srmse_clipped: bool
    invalid_count: int
    catastrophic_count: int
    fallback_count: int
    selected_candidate_id: str
    numerical_package_sha256: str
    final_retrieval_sha256: str
    final_decision_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("package task score requires a task ID")
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise ValueError("package task score requires an entity name")
        for field_name in (
            "final_smae",
            "final_srmse",
            "final_smae_raw",
            "final_srmse_raw",
            "numerical_oracle_smae",
            "numerical_oracle_srmse",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, float(value))
        try:
            forecast_values = tuple(self.final_forecast)
            if any(isinstance(value, bool) for value in forecast_values):
                raise ValueError
            forecast = tuple(float(value) for value in forecast_values)
        except (TypeError, ValueError) as error:
            raise ValueError("final_forecast must contain finite numbers") from error
        if not forecast or any(not math.isfinite(value) for value in forecast):
            raise ValueError("final_forecast must contain finite numbers")
        object.__setattr__(self, "final_forecast", forecast)
        if type(self.numerical_candidate_count) is not int or (
            self.numerical_candidate_count <= 0
        ):
            raise ValueError("numerical_candidate_count must be a positive integer")
        if (
            type(self.smae_clipped) is not bool
            or type(self.srmse_clipped) is not bool
        ):
            raise ValueError("package clipping markers must be booleans")
        if (
            self.final_smae > self.final_smae_raw
            or self.final_srmse > self.final_srmse_raw
        ):
            raise ValueError("capped package metrics cannot exceed their raw values")
        if self.smae_clipped != (self.final_smae < self.final_smae_raw) or (
            self.srmse_clipped != (self.final_srmse < self.final_srmse_raw)
        ):
            raise ValueError("package clipping markers must match capped metrics")
        for field_name in (
            "invalid_count",
            "catastrophic_count",
            "fallback_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        expected_catastrophic = int(
            self.final_smae_raw > 10.0 or self.final_srmse_raw > 10.0
        )
        if self.catastrophic_count != expected_catastrophic:
            raise ValueError(
                "catastrophic_count must reflect a raw scaled error above 10.0"
            )
        if (
            not isinstance(self.selected_candidate_id, str)
            or not self.selected_candidate_id
        ):
            raise ValueError("package task score requires a selected candidate")
        for field_name in (
            "numerical_package_sha256",
            "final_retrieval_sha256",
            "final_decision_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a canonical SHA-256")

    @property
    def joint(self) -> float:
        return (self.final_smae + self.final_srmse) / 2.0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _evaluation_fields(
    rows: tuple[PackageTaskScore, ...],
    expected_task_ids: tuple[str, ...],
) -> dict[str, int | float]:
    def mean(field_name: str) -> float:
        return (
            statistics.fmean(float(getattr(row, field_name)) for row in rows)
            if rows
            else 0.0
        )

    def quantile(field_name: str, probability: float) -> float:
        return (
            linear_quantile(
                [float(getattr(row, field_name)) for row in rows], probability
            )
            if rows
            else 0.0
        )

    mean_smae = mean("final_smae")
    mean_srmse = mean("final_srmse")
    return {
        "task_count": len(rows),
        "coverage": len(rows) / len(expected_task_ids),
        "mean_smae": mean_smae,
        "mean_srmse": mean_srmse,
        "mean_joint": (mean_smae + mean_srmse) / 2.0,
        "mean_smae_raw": mean("final_smae_raw"),
        "mean_srmse_raw": mean("final_srmse_raw"),
        "p90_smae": quantile("final_smae", 0.90),
        "p95_smae": quantile("final_smae", 0.95),
        "p90_srmse": quantile("final_srmse", 0.90),
        "p95_srmse": quantile("final_srmse", 0.95),
        "invalid_count": sum(row.invalid_count for row in rows),
        "catastrophic_count": sum(row.catastrophic_count for row in rows),
        "fallback_count": sum(row.fallback_count for row in rows),
        "clipped_count": sum(
            int(row.smae_clipped or row.srmse_clipped) for row in rows
        ),
    }


@dataclass(frozen=True)
class PackageGateConfig:
    tolerance: float = 1e-12
    minimum_relative_joint_gain: float = 0.005
    maximum_task_joint_regret: float = 0.25

    def __post_init__(self) -> None:
        for field_name in (
            "tolerance",
            "minimum_relative_joint_gain",
            "maximum_task_joint_regret",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, float(value))


@dataclass(frozen=True)
class PackageEvaluation:
    """Immutable aggregate whose primary fields alone govern acceptance."""

    candidate_sha256: str
    task_rows: tuple[PackageTaskScore, ...]
    expected_task_ids: tuple[str, ...]
    missing_task_ids: tuple[str, ...]
    task_count: int
    coverage: float
    mean_smae: float
    mean_srmse: float
    mean_joint: float
    mean_smae_raw: float
    mean_srmse_raw: float
    p90_smae: float
    p95_smae: float
    p90_srmse: float
    p95_srmse: float
    invalid_count: int
    catastrophic_count: int
    fallback_count: int
    clipped_count: int
    secondary_diagnostics: Mapping[str, float]
    public_test_accessed: bool = False

    def __post_init__(self) -> None:
        if not _is_sha256(self.candidate_sha256):
            raise ValueError("package evaluation candidate must be a canonical SHA-256")
        task_rows = tuple(self.task_rows)
        expected_task_ids = tuple(self.expected_task_ids)
        missing_task_ids = tuple(self.missing_task_ids)
        if any(not isinstance(row, PackageTaskScore) for row in task_rows):
            raise TypeError("package evaluation rows must be PackageTaskScore values")
        if any(
            not isinstance(task_id, str) or not task_id
            for task_id in expected_task_ids
        ):
            raise ValueError("package evaluation expected task IDs must be non-empty")
        if any(
            not isinstance(task_id, str) or not task_id
            for task_id in missing_task_ids
        ):
            raise ValueError("package evaluation missing task IDs must be non-empty")
        if (
            not expected_task_ids
            or len(expected_task_ids) != len(set(expected_task_ids))
        ):
            raise ValueError("package evaluation requires unique expected task IDs")
        row_ids = tuple(row.task_id for row in task_rows)
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("package evaluation task rows must be unique")
        if set(row_ids) - set(expected_task_ids):
            raise ValueError("package evaluation rows are outside expected membership")
        canonical_rows = tuple(sorted(task_rows, key=lambda row: row.task_id))
        canonical_expected = tuple(sorted(expected_task_ids))
        canonical_missing = tuple(sorted(set(canonical_expected) - set(row_ids)))
        object.__setattr__(self, "task_rows", canonical_rows)
        object.__setattr__(self, "expected_task_ids", canonical_expected)
        object.__setattr__(self, "missing_task_ids", canonical_missing)
        for field_name in (
            "coverage",
            "mean_smae",
            "mean_srmse",
            "mean_joint",
            "mean_smae_raw",
            "mean_srmse_raw",
            "p90_smae",
            "p95_smae",
            "p90_srmse",
            "p95_srmse",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, float(value))
        if self.coverage > 1.0:
            raise ValueError("package evaluation coverage cannot exceed one")
        for field_name in (
            "task_count",
            "invalid_count",
            "catastrophic_count",
            "fallback_count",
            "clipped_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if type(self.public_test_accessed) is not bool:
            raise ValueError("public_test_accessed must be a boolean")
        diagnostics = dict(self.secondary_diagnostics)
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for key, value in diagnostics.items()
        ):
            raise ValueError("secondary diagnostics must be finite named numbers")
        object.__setattr__(
            self,
            "secondary_diagnostics",
            MappingProxyType(
                {key: float(value) for key, value in sorted(diagnostics.items())}
            ),
        )
        derived = _evaluation_fields(canonical_rows, canonical_expected)
        if missing_task_ids != canonical_missing or any(
            getattr(self, field_name) != value
            for field_name, value in derived.items()
        ):
            raise ValueError(
                "package evaluation derived fields do not match task rows"
            )

    @classmethod
    def from_rows(
        cls,
        candidate_sha256: str,
        rows: Sequence[PackageTaskScore],
        expected_task_ids: Sequence[str],
        secondary_diagnostics: Mapping[str, float] | None = None,
    ) -> "PackageEvaluation":
        if not _is_sha256(candidate_sha256):
            raise ValueError("package evaluation candidate must be a canonical SHA-256")
        expected = tuple(expected_task_ids)
        if (
            not expected
            or any(not isinstance(task_id, str) or not task_id for task_id in expected)
            or len(expected) != len(set(expected))
        ):
            raise ValueError("package evaluation requires unique expected task IDs")
        expected = tuple(sorted(expected))
        scored = tuple(rows)
        if any(not isinstance(row, PackageTaskScore) for row in scored):
            raise TypeError("package evaluation rows must be PackageTaskScore values")
        row_ids = tuple(row.task_id for row in scored)
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("package evaluation task rows must be unique")
        unexpected = set(row_ids) - set(expected)
        if unexpected:
            raise ValueError("package evaluation rows are outside expected membership")
        scored = tuple(sorted(scored, key=lambda row: row.task_id))
        missing = tuple(sorted(set(expected) - set(row_ids)))

        return cls(
            candidate_sha256=candidate_sha256,
            task_rows=scored,
            expected_task_ids=expected,
            missing_task_ids=missing,
            **_evaluation_fields(scored, expected),
            secondary_diagnostics=secondary_diagnostics or {},
            public_test_accessed=False,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "candidate_sha256": self.candidate_sha256,
            "task_rows": [row.to_payload() for row in self.task_rows],
            "expected_task_ids": list(self.expected_task_ids),
            "missing_task_ids": list(self.missing_task_ids),
            "task_count": self.task_count,
            "coverage": self.coverage,
            "mean_smae": self.mean_smae,
            "mean_srmse": self.mean_srmse,
            "mean_joint": self.mean_joint,
            "mean_smae_raw": self.mean_smae_raw,
            "mean_srmse_raw": self.mean_srmse_raw,
            "p90_smae": self.p90_smae,
            "p95_smae": self.p95_smae,
            "p90_srmse": self.p90_srmse,
            "p95_srmse": self.p95_srmse,
            "invalid_count": self.invalid_count,
            "catastrophic_count": self.catastrophic_count,
            "fallback_count": self.fallback_count,
            "clipped_count": self.clipped_count,
            "secondary_diagnostics": dict(self.secondary_diagnostics),
            "public_test_accessed": self.public_test_accessed,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_payload())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def result_bytes(self) -> bytes:
        payload = self.to_payload()
        del payload["candidate_sha256"]
        return _canonical_json(payload)


def package_screen_failures(
    child: PackageEvaluation,
    parent: PackageEvaluation,
    config: PackageGateConfig,
) -> tuple[str, ...]:
    """Return primary screen-gate failures; diagnostics never grant acceptance."""
    if not isinstance(child, PackageEvaluation) or not isinstance(
        parent, PackageEvaluation
    ):
        raise TypeError("package gates require PackageEvaluation values")
    if not isinstance(config, PackageGateConfig):
        raise TypeError("package gates require PackageGateConfig")
    failures: list[str] = []
    if (
        child.coverage != 1.0
        or parent.coverage != 1.0
        or child.expected_task_ids != parent.expected_task_ids
        or tuple(row.task_id for row in child.task_rows)
        != tuple(row.task_id for row in parent.task_rows)
    ):
        failures.append("task_coverage")
    if child.mean_smae > parent.mean_smae + config.tolerance:
        failures.append("mean_smae")
    if child.mean_srmse > parent.mean_srmse + config.tolerance:
        failures.append("mean_srmse")
    for field_name in (
        "invalid_count",
        "catastrophic_count",
        "clipped_count",
        "fallback_count",
    ):
        if getattr(child, field_name) > getattr(parent, field_name):
            failures.append(field_name)
    if child.public_test_accessed:
        failures.append("public_test_accessed")
    return tuple(failures)


def package_full_gate_failures(
    child: PackageEvaluation,
    parent: PackageEvaluation,
    config: PackageGateConfig,
    *,
    stage: str,
    fold_manifest: object | None = None,
    initial: PackageEvaluation | None = None,
) -> tuple[str, ...]:
    """Return Build/Calibration/Dev acceptance failures."""
    if not isinstance(stage, str) or not stage:
        raise ValueError("package full gate requires a stage")
    failures = list(package_screen_failures(child, parent, config))
    relative_gain = (
        (parent.mean_joint - child.mean_joint) / parent.mean_joint
        if parent.mean_joint > 0.0
        else 0.0
    )
    if relative_gain < config.minimum_relative_joint_gain - config.tolerance:
        failures.append("minimum_relative_joint_gain")
    for field_name in (
        "p90_smae",
        "p95_smae",
        "p90_srmse",
        "p95_srmse",
    ):
        if getattr(child, field_name) > getattr(parent, field_name) + config.tolerance:
            failures.append(field_name)

    parent_rows = {row.task_id: row for row in parent.task_rows}
    child_rows = {row.task_id: row for row in child.task_rows}
    paired_ids = tuple(sorted(set(parent_rows).intersection(child_rows)))
    maximum_regret = max(
        (
            child_rows[task_id].joint - parent_rows[task_id].joint
            for task_id in paired_ids
        ),
        default=0.0,
    )
    if maximum_regret > config.maximum_task_joint_regret + config.tolerance:
        failures.append("maximum_task_joint_regret")

    if initial is not None:
        if not isinstance(initial, PackageEvaluation):
            raise TypeError("initial Toto authority must be a PackageEvaluation")
        if child.mean_smae > initial.mean_smae + config.tolerance:
            failures.append("initial_toto_mean_smae")
        if child.mean_srmse > initial.mean_srmse + config.tolerance:
            failures.append("initial_toto_mean_srmse")

    if stage.casefold().startswith("build"):
        if (
            fold_manifest is None
            or not hasattr(fold_manifest, "task_fold_map")
            or getattr(fold_manifest, "fold_count", None) != 5
        ):
            failures.append("fold_manifest")
        else:
            task_fold_map = dict(fold_manifest.task_fold_map)
            folds = set(task_fold_map.values())
            if (
                set(task_fold_map) != set(parent_rows)
                or set(task_fold_map) != set(child_rows)
                or folds != set(range(5))
            ):
                failures.append("fold_manifest")
            else:
                nonregressing = 0
                improving = 0
                for fold in range(5):
                    task_ids = tuple(
                        task_id
                        for task_id, assigned_fold in task_fold_map.items()
                        if assigned_fold == fold
                    )
                    parent_smae = statistics.fmean(
                        parent_rows[task_id].final_smae for task_id in task_ids
                    )
                    parent_srmse = statistics.fmean(
                        parent_rows[task_id].final_srmse for task_id in task_ids
                    )
                    child_smae = statistics.fmean(
                        child_rows[task_id].final_smae for task_id in task_ids
                    )
                    child_srmse = statistics.fmean(
                        child_rows[task_id].final_srmse for task_id in task_ids
                    )
                    safe = (
                        child_smae <= parent_smae + config.tolerance
                        and child_srmse <= parent_srmse + config.tolerance
                    )
                    if safe:
                        nonregressing += 1
                        if (
                            (child_smae + child_srmse) / 2.0
                            < (parent_smae + parent_srmse) / 2.0
                            - 1e-12
                        ):
                            improving += 1
                if nonregressing < 4:
                    failures.append("nonregressing_folds")
                if improving < 3:
                    failures.append("improving_folds")
    return tuple(dict.fromkeys(failures))


def package_rank_key(evaluation: PackageEvaluation) -> tuple[float, ...]:
    if not isinstance(evaluation, PackageEvaluation):
        raise TypeError("package rank key requires a PackageEvaluation")
    return (evaluation.mean_joint, evaluation.mean_srmse, evaluation.mean_smae)


__all__ = [
    "PackageEvaluation",
    "PackageGateConfig",
    "PackageTaskScore",
    "package_full_gate_failures",
    "package_rank_key",
    "package_screen_failures",
]
