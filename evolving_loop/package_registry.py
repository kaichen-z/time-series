"""Immutable task-to-Champion Numerical package registration."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from evolving_loop.data import ContextTask
from evolving_loop.numerical_two_stage import numerical_package_fingerprint
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint
from numerical_agent.evolution.numerical_package import NumericalForecastPackage
from numerical_agent.evolution.screening import profile_task


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHAMPION_KEYS = frozenset(
    {"champion_release", "champion_recipe", "champion_assumptions"}
)


class PackageRegistryError(ValueError):
    """Raised when a registered task/package authority is incomplete or changes."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _task_payload(task: ContextTask) -> dict[str, object]:
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


def task_registry_fingerprint(task: ContextTask) -> str:
    if not isinstance(task, ContextTask):
        raise PackageRegistryError("registry tasks must be ContextTask records")
    return _digest(_task_payload(task))


def _validate_task_package_binding(
    task: ContextTask,
    package: NumericalForecastPackage,
) -> None:
    if not isinstance(package, NumericalForecastPackage):
        raise PackageRegistryError(
            "registry entries require NumericalForecastPackage values"
        )
    expected_profile = profile_task(
        Task(
            task.numeric.task_id,
            tuple(task.numeric.history_values),
            task.numeric.prediction_length,
            task.numeric.frequency,
            (),
        )
    )
    if package.task_profile != expected_profile:
        raise PackageRegistryError("Numerical package task binding profile mismatch")
    expected_input = task_input_fingerprint(
        task_id=task.numeric.task_id,
        history=task.numeric.history_values,
        frequency=task.numeric.frequency,
        horizon=task.numeric.prediction_length,
    )
    if package.component_fingerprints.get("task_input") != expected_input:
        raise PackageRegistryError("Numerical package task binding input mismatch")
    components = dict(package.component_fingerprints)
    if _CHAMPION_KEYS - set(components) or any(
        _SHA256.fullmatch(components[key]) is None for key in _CHAMPION_KEYS
    ):
        raise PackageRegistryError(
            "registry requires complete Champion Numerical provenance"
        )


class FrozenNumericalPackageRegistry:
    """Read-only package authority bound to exact resolved evaluation tasks."""

    def __init__(
        self,
        entries: Sequence[tuple[ContextTask, NumericalForecastPackage]],
        *,
        release_sha256: str,
        expected_task_ids: Sequence[str],
    ) -> None:
        if not isinstance(release_sha256, str) or _SHA256.fullmatch(release_sha256) is None:
            raise PackageRegistryError("registry supply release must be canonical")
        expected = tuple(expected_task_ids)
        if (
            not expected
            or any(not isinstance(task_id, str) or not task_id for task_id in expected)
            or len(expected) != len(set(expected))
        ):
            raise PackageRegistryError("registry requires a complete task coverage universe")
        expected = tuple(sorted(expected))
        supplied = tuple(entries)
        if not supplied:
            raise PackageRegistryError("package registry cannot be empty")
        packages: dict[str, NumericalForecastPackage] = {}
        task_hashes: dict[str, str] = {}
        package_hashes: dict[str, str] = {}
        for entry in supplied:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise PackageRegistryError("registry entries must be task/package pairs")
            task, package = entry
            if not isinstance(task, ContextTask):
                raise PackageRegistryError("registry tasks must be ContextTask records")
            task_id = task.numeric.task_id
            if not isinstance(task_id, str) or not task_id:
                raise PackageRegistryError("registry task IDs must be non-empty")
            if task_id in packages:
                raise PackageRegistryError("package registry task IDs must be unique")
            _validate_task_package_binding(task, package)
            if package.component_fingerprints.get("numerical_supply_release") != release_sha256:
                raise PackageRegistryError(
                    "registry package Numerical supply release mismatch"
                )
            packages[task_id] = package
            task_hashes[task_id] = task_registry_fingerprint(task)
            package_hashes[task_id] = numerical_package_fingerprint(package)
        if tuple(sorted(packages)) != expected:
            raise PackageRegistryError("registry requires complete task coverage")
        manifest = {
            "schema_version": 1,
            "release_sha256": release_sha256,
            "task_ids": list(expected),
            "entries": [
                {
                    "task_id": task_id,
                    "task_sha256": task_hashes[task_id],
                    "package_sha256": package_hashes[task_id],
                    "champion_release_sha256": packages[
                        task_id
                    ].component_fingerprints["champion_release"],
                }
                for task_id in sorted(packages)
            ],
        }
        self._packages = MappingProxyType(packages)
        self._task_hashes = MappingProxyType(task_hashes)
        self._package_hashes = MappingProxyType(package_hashes)
        self.release_sha256 = release_sha256
        frozen_manifest = _freeze_json(manifest)
        if not isinstance(frozen_manifest, Mapping):  # pragma: no cover
            raise AssertionError("frozen package manifest must remain a mapping")
        self._manifest = frozen_manifest
        self.fingerprint = _digest(manifest)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._packages))

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._manifest

    def package_for(self, task: ContextTask) -> NumericalForecastPackage:
        if not isinstance(task, ContextTask):
            raise PackageRegistryError("package lookup requires a ContextTask")
        task_id = task.numeric.task_id
        package = self._packages.get(task_id)
        if package is None:
            raise PackageRegistryError("task is absent from the package registry")
        if task_registry_fingerprint(task) != self._task_hashes[task_id]:
            raise PackageRegistryError("Numerical package task binding changed")
        if numerical_package_fingerprint(package) != self._package_hashes[task_id]:
            raise PackageRegistryError("registered Numerical package changed")
        return package
