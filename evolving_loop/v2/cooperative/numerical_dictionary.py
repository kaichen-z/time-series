"""P3-owned selector artifacts over a frozen executable Numerical Dictionary."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType

from common.payload import canonical_json_bytes
from evolving_loop.data import ContextTask
from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyRelease,
    bound_numerical_package,
    build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.package_registry import task_registry_fingerprint
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage,
    RankedNumericalForecast,
)
from numerical_agent.evolution.task_shortlist import (
    TaskCandidateShortlistV1,
    TaskShortlistPolicyV1,
)
from numerical_agent.run_task_local_ensemble_evolution import task_input_sha256

from ..numerical_qd.adapters import FrozenNumericalArtifactsV2, import_numerical_seed

from ..contracts import (
    _require_exact_schema,
    canonical_v2_bytes,
    fingerprint_payload,
    require_sha256,
)


_SELECTOR_FIELDS = (
    "schema_version",
    "generation",
    "parent_selector_sha256",
    "target_candidates",
    "morphology_weight",
    "success_weight",
    "mean_error_weight",
    "p90_error_weight",
    "family_diversity_weight",
)
_WEIGHT_FIELDS = _SELECTOR_FIELDS[4:]
_WEIGHT_CYCLE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
_TARGET_CYCLE = {6: 7, 7: 8, 8: 6}


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class DictionarySelectorGenomeV2:
    """Canonical P3 policy for choosing a task-local Numerical shortlist."""

    schema_version: int
    generation: int
    parent_selector_sha256: str | None
    target_candidates: int
    morphology_weight: float
    success_weight: float
    mean_error_weight: float
    p90_error_weight: float
    family_diversity_weight: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self.generation == 0:
            if self.parent_selector_sha256 is not None:
                raise ValueError("seed Selector must not claim a Parent")
        elif self.parent_selector_sha256 is None:
            raise ValueError("evolved Selector requires a Parent")
        if self.parent_selector_sha256 is not None:
            require_sha256(self.parent_selector_sha256, "parent_selector_sha256")
        if type(self.target_candidates) is not int or not 6 <= self.target_candidates <= 8:
            raise ValueError("target_candidates must be an integer in [6, 8]")
        for field in _WEIGHT_FIELDS:
            object.__setattr__(self, field, _finite_nonnegative(getattr(self, field), field))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DictionarySelectorGenomeV2":
        values = _require_exact_schema(payload, _SELECTOR_FIELDS, field="Dictionary Selector Genome")
        return cls(**values)  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _SELECTOR_FIELDS}

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


def seed_selector_genome() -> DictionarySelectorGenomeV2:
    return DictionarySelectorGenomeV2(
        schema_version=1,
        generation=0,
        parent_selector_sha256=None,
        target_candidates=8,
        morphology_weight=1.0,
        success_weight=1.0,
        mean_error_weight=1.0,
        p90_error_weight=0.5,
        family_diversity_weight=0.25,
    )


def _next_weight(value: float) -> float:
    try:
        index = _WEIGHT_CYCLE.index(value)
    except ValueError:
        return min(_WEIGHT_CYCLE, key=lambda candidate: (abs(candidate - value), candidate))
    return _WEIGHT_CYCLE[(index + 1) % len(_WEIGHT_CYCLE)]


def mutate_selector_genome(
    parent: DictionarySelectorGenomeV2,
    step: int,
) -> DictionarySelectorGenomeV2:
    if type(parent) is not DictionarySelectorGenomeV2:
        raise TypeError("Selector Parent must be a DictionarySelectorGenomeV2")
    if type(step) is not int or step < 0:
        raise ValueError("Selector step must be a non-negative integer")
    coordinate = step % (len(_WEIGHT_FIELDS) + 1)
    changes: dict[str, object] = {
        "generation": parent.generation + 1,
        "parent_selector_sha256": parent.fingerprint(),
    }
    if coordinate < len(_WEIGHT_FIELDS):
        field = _WEIGHT_FIELDS[coordinate]
        changes[field] = _next_weight(float(getattr(parent, field)))
    else:
        changes["target_candidates"] = _TARGET_CYCLE[parent.target_candidates]
    return replace(parent, **changes)


@dataclass(frozen=True, slots=True)
class P3NumericalDictionaryV2:
    """Canonical identity plus exact in-memory executable inputs for P3."""

    schema_version: int
    source_pair_sha256s: tuple[str, ...]
    task_sha256s: Mapping[str, str]
    anchor_name: str
    anchor_release_sha256: str
    alternatives: tuple[NumericalAlternativeSpec, ...]
    runtime_fingerprints: Mapping[str, str]
    public_test_accessed: bool
    _package_inputs: Mapping[str, Mapping[str, RankedNumericalForecast]] = field(
        repr=False,
        compare=False,
    )
    _source_packages: Mapping[str, NumericalForecastPackage] = field(
        repr=False,
        compare=False,
    )
    _template_release: NumericalSupplyRelease = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("P3 Numerical Dictionary schema_version must be 1")
        if (
            type(self.source_pair_sha256s) is not tuple
            or self.source_pair_sha256s != tuple(sorted(set(self.source_pair_sha256s)))
            or not self.source_pair_sha256s
        ):
            raise ValueError("P3 Numerical Dictionary source pairs must be sorted and unique")
        for identity in self.source_pair_sha256s:
            require_sha256(identity, "source pair SHA")
        if type(self.anchor_name) is not str or not self.anchor_name.strip():
            raise ValueError("P3 Numerical Dictionary requires an Anchor name")
        require_sha256(self.anchor_release_sha256, "anchor release SHA")
        if self.public_test_accessed is not False:
            raise ValueError("Public Test access is forbidden")
        tasks = dict(self.task_sha256s)
        if not tasks or tuple(tasks) != tuple(sorted(tasks)):
            raise ValueError("P3 Numerical Dictionary tasks must be sorted")
        for task_id, identity in tasks.items():
            if type(task_id) is not str or not task_id:
                raise ValueError("P3 Numerical Dictionary task IDs must be non-empty")
            require_sha256(identity, "task SHA")
        alternatives = tuple(self.alternatives)
        names = tuple(item.candidate_id for item in alternatives)
        if names != tuple(sorted(set(names))):
            raise ValueError("P3 Numerical Dictionary alternatives must be sorted and unique")
        runtime = dict(self.runtime_fingerprints)
        for identity in runtime.values():
            require_sha256(identity, "runtime fingerprint")
        inputs = dict(self._package_inputs)
        sources = dict(self._source_packages)
        if set(inputs) != set(tasks) or set(sources) != set(tasks):
            raise ValueError("P3 Numerical Dictionary inputs must bind every task")
        frozen_inputs = {}
        allowed = set(names) | {self.anchor_name}
        for task_id in sorted(inputs):
            values = dict(inputs[task_id])
            if self.anchor_name not in values or not set(values) <= allowed:
                raise ValueError("P3 Numerical Dictionary task inputs require the Anchor")
            if any(type(item) is not RankedNumericalForecast for item in values.values()):
                raise ValueError("P3 Numerical Dictionary inputs must be ranked forecasts")
            frozen_inputs[task_id] = MappingProxyType(dict(sorted(values.items())))
            if type(sources[task_id]) is not NumericalForecastPackage:
                raise ValueError("P3 Numerical Dictionary source packages are invalid")
        if type(self._template_release) is not NumericalSupplyRelease:
            raise ValueError("P3 Numerical Dictionary template release is invalid")
        object.__setattr__(self, "task_sha256s", MappingProxyType(tasks))
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(self, "runtime_fingerprints", MappingProxyType(dict(sorted(runtime.items()))))
        object.__setattr__(self, "_package_inputs", MappingProxyType(frozen_inputs))
        object.__setattr__(self, "_source_packages", MappingProxyType(sources))

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return (self.anchor_name,) + tuple(item.candidate_id for item in self.alternatives)

    def package_inputs_for(self, task: ContextTask) -> Mapping[str, RankedNumericalForecast]:
        if type(task) is not ContextTask:
            raise TypeError("P3 Numerical Dictionary requires an exact ContextTask")
        task_id = task.numeric.task_id
        if self.task_sha256s.get(task_id) != task_registry_fingerprint(task):
            raise ValueError("P3 Numerical Dictionary task identity changed")
        return self._package_inputs[task_id]

    def source_package_for(self, task: ContextTask) -> NumericalForecastPackage:
        self.package_inputs_for(task)
        return self._source_packages[task.numeric.task_id]

    @property
    def template_release(self) -> NumericalSupplyRelease:
        return self._template_release

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_pair_sha256s": list(self.source_pair_sha256s),
            "task_sha256s": dict(self.task_sha256s),
            "anchor_name": self.anchor_name,
            "anchor_release_sha256": self.anchor_release_sha256,
            "alternatives": [item.to_payload() for item in self.alternatives],
            "runtime_fingerprints": dict(self.runtime_fingerprints),
            "public_test_accessed": self.public_test_accessed,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


def _pair_identity(pair: FrozenNumericalArtifactsV2) -> str:
    return fingerprint_payload(
        {
            "release": pair.release.fingerprint,
            "registry": pair.registry.fingerprint,
            "envelope": pair.envelope.fingerprint(),
        }
    )


def build_p3_numerical_dictionary(
    pairs: tuple[FrozenNumericalArtifactsV2, ...],
    tasks: tuple[ContextTask, ...],
) -> P3NumericalDictionaryV2:
    """Union compatible P2 executable artifacts without inheriting selection."""
    values = tuple(pairs)
    resolved_tasks = tuple(sorted(tuple(tasks), key=lambda task: task.numeric.task_id))
    if not values or any(type(pair) is not FrozenNumericalArtifactsV2 for pair in values):
        raise ValueError("P3 Numerical Dictionary requires frozen P2 pairs")
    if not resolved_tasks or any(type(task) is not ContextTask for task in resolved_tasks):
        raise ValueError("P3 Numerical Dictionary requires exact Host tasks")
    task_ids = tuple(task.numeric.task_id for task in resolved_tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("P3 Numerical Dictionary task IDs must be unique")

    first = values[0]
    anchor_payload = first.release.to_payload()["anchor_release_payload"]
    anchor_release_sha = fingerprint_payload(anchor_payload)
    runtime = dict(first.release.runtime_fingerprints)
    specs: dict[str, NumericalAlternativeSpec] = {}
    package_inputs: dict[str, dict[str, RankedNumericalForecast]] = {
        task_id: {} for task_id in task_ids
    }
    source_packages = {
        task.numeric.task_id: first.registry.package_for(task)
        for task in resolved_tasks
    }
    anchor_name: str | None = None

    for pair in values:
        if pair.registry.task_ids != tuple(sorted(task_ids)):
            raise ValueError("P3 Numerical Dictionary task universe changed")
        if dict(pair.release.runtime_fingerprints) != runtime:
            raise ValueError("P3 Numerical Dictionary runtime identity changed")
        if fingerprint_payload(pair.release.to_payload()["anchor_release_payload"]) != anchor_release_sha:
            raise ValueError("P3 Numerical Dictionary Anchor release changed")
        restored = pair.envelope.restore(resolved_tasks)
        if restored.fingerprint != pair.registry.fingerprint:
            raise ValueError("P3 Numerical Dictionary frozen registry changed")
        for spec in pair.release.alternatives:
            existing = specs.get(spec.candidate_id)
            if existing is not None and canonical_v2_bytes(existing.to_payload()) != canonical_v2_bytes(spec.to_payload()):
                raise ValueError("P3 Numerical Dictionary candidate specification changed")
            specs[spec.candidate_id] = spec
        for task in resolved_tasks:
            package = pair.registry.package_for(task)
            anchor = package.protected_baseline
            if anchor_name is None:
                anchor_name = anchor.name
            if anchor.name != anchor_name:
                raise ValueError("P3 Numerical Dictionary Anchor identity changed")
            existing_anchor = package_inputs[task.numeric.task_id].get(anchor.name)
            if existing_anchor is not None and replace(existing_anchor, rank=1) != replace(anchor, rank=1):
                raise ValueError("P3 Numerical Dictionary Anchor forecast changed")
            package_inputs[task.numeric.task_id][anchor.name] = anchor
            for item in package.ranked_alternatives:
                existing = package_inputs[task.numeric.task_id].get(item.name)
                if existing is not None and replace(existing, rank=1) != replace(item, rank=1):
                    raise ValueError("P3 Numerical Dictionary candidate forecast changed")
                package_inputs[task.numeric.task_id][item.name] = item

    assert anchor_name is not None
    allowed = set(specs) | {anchor_name}
    for task_id in task_ids:
        package_inputs[task_id] = {
            name: item for name, item in package_inputs[task_id].items() if name in allowed
        }
    return P3NumericalDictionaryV2(
        schema_version=1,
        source_pair_sha256s=tuple(sorted(_pair_identity(pair) for pair in values)),
        task_sha256s=MappingProxyType(
            {task.numeric.task_id: task_registry_fingerprint(task) for task in resolved_tasks}
        ),
        anchor_name=anchor_name,
        anchor_release_sha256=anchor_release_sha,
        alternatives=tuple(specs[name] for name in sorted(specs)),
        runtime_fingerprints=MappingProxyType(runtime),
        public_test_accessed=False,
        _package_inputs=MappingProxyType(package_inputs),
        _source_packages=MappingProxyType(source_packages),
        _template_release=first.release,
    )


def _bounded_score(value: float) -> float:
    return min(5.0, max(0.0, float(value))) if math.isfinite(float(value)) else 5.0


def _ordered_shortlist(
    dictionary: P3NumericalDictionaryV2,
    genome: DictionarySelectorGenomeV2,
    task: ContextTask,
) -> tuple[str, ...]:
    inputs = dictionary.package_inputs_for(task)
    anchor = dictionary.anchor_name
    remaining = [item for name, item in inputs.items() if name != anchor and item.diagnostics.eligible]
    family_counts = {inputs[anchor].family: 1}
    chosen = [anchor]

    def score(item: RankedNumericalForecast) -> tuple[float, str]:
        diagnostic = item.diagnostics
        success_penalty = 1.0 / (1.0 + max(0, diagnostic.successful_folds))
        value = (
            genome.morphology_weight * _bounded_score(diagnostic.recent_joint_scaled_error)
            + genome.success_weight * success_penalty
            + genome.mean_error_weight * _bounded_score(diagnostic.median_joint_scaled_error)
            + genome.p90_error_weight * _bounded_score(diagnostic.worst_joint_scaled_error)
            + genome.family_diversity_weight * family_counts.get(item.family, 0)
        )
        return value, item.name

    while remaining and len(chosen) < genome.target_candidates:
        remaining.sort(key=score)
        item = remaining.pop(0)
        chosen.append(item.name)
        family_counts[item.family] = family_counts.get(item.family, 0) + 1
    return tuple(chosen)


def _selector_release(
    dictionary: P3NumericalDictionaryV2,
    genome: DictionarySelectorGenomeV2,
) -> NumericalSupplyRelease:
    payload = dictionary.template_release.to_payload()
    payload.update(
        schema_version=2,
        version=f"n{900 + genome.generation:03d}",
        parent_sha256=dictionary.template_release.fingerprint,
        alternatives=[item.to_payload() for item in dictionary.alternatives],
        source_fingerprints={
            **dict(dictionary.template_release.source_fingerprints),
            "dictionary": dictionary.fingerprint(),
            "p3_dictionary": dictionary.fingerprint(),
            "p3_selector": genome.fingerprint(),
        },
    )
    return parse_numerical_supply_release(payload)


def _task_history_sha(task: ContextTask) -> str:
    numeric = task.numeric
    return task_input_sha256(
        RuntimeTask(
            numeric.task_id,
            numeric.history_values,
            numeric.prediction_length,
            numeric.frequency,
            (),
        )
    )


def _diagnostics_payload(
    task: ContextTask,
    task_sha256: str,
    names: tuple[str, ...],
    inputs: Mapping[str, RankedNumericalForecast],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": task.numeric.task_id,
        "task_input_sha256": task_sha256,
        "rows": [
            {
                "candidate_name": name,
                "failure_reason": None,
                "diagnostic": asdict(inputs[name].diagnostics),
            }
            for name in sorted(names)
        ],
        "public_test_accessed": False,
    }


def materialize_selector_pair(
    dictionary: P3NumericalDictionaryV2,
    genome: DictionarySelectorGenomeV2,
    tasks: tuple[ContextTask, ...],
) -> FrozenNumericalArtifactsV2:
    """Materialize one P3 Selector into an atomic release/registry pair."""
    if type(dictionary) is not P3NumericalDictionaryV2:
        raise TypeError("P3 selection requires a P3NumericalDictionaryV2")
    if type(genome) is not DictionarySelectorGenomeV2:
        raise TypeError("P3 selection requires a DictionarySelectorGenomeV2")
    resolved_tasks = tuple(sorted(tuple(tasks), key=lambda task: task.numeric.task_id))
    if tuple(task.numeric.task_id for task in resolved_tasks) != tuple(dictionary.task_sha256s):
        raise ValueError("P3 Selector task universe differs from its Dictionary")
    release = _selector_release(dictionary, genome)
    policy = TaskShortlistPolicyV1()

    def build(task: ContextTask, supplied: NumericalSupplyRelease) -> NumericalForecastPackage:
        inputs = dictionary.package_inputs_for(task)
        names = _ordered_shortlist(dictionary, genome, task)
        exclusions = tuple(
            sorted(
                (
                    name,
                    "unavailable" if name not in inputs or not inputs[name].diagnostics.eligible else "ranked_out",
                )
                for name in dictionary.candidate_names
                if name not in names
            )
        )
        task_sha = _task_history_sha(task)
        shortlist = TaskCandidateShortlistV1(
            1,
            task_sha,
            dictionary.fingerprint(),
            policy.fingerprint(),
            names,
            exclusions,
            len(names) < policy.minimum_candidates,
            False,
        )
        diagnostics = _diagnostics_payload(task, task_sha, names, inputs)
        diagnostics_sha = __import__("hashlib").sha256(
            canonical_json_bytes(diagnostics)
        ).hexdigest()
        return bound_numerical_package(
            dictionary.source_package_for(task),
            supplied,
            {name: inputs[name] for name in names},
            history=task.numeric.history_values,
            shortlist=shortlist,
            hindcast_diagnostics_sha256=diagnostics_sha,
            hindcast_diagnostics=diagnostics,
        )

    registry = build_package_registry(resolved_tasks, release, build)
    envelope = import_numerical_seed(release, registry, tasks=resolved_tasks).envelope
    return FrozenNumericalArtifactsV2(
        release,
        registry,
        envelope,
        (genome.fingerprint(),),
    )


__all__ = [
    "DictionarySelectorGenomeV2",
    "P3NumericalDictionaryV2",
    "build_p3_numerical_dictionary",
    "materialize_selector_pair",
    "mutate_selector_genome",
    "seed_selector_genome",
]
