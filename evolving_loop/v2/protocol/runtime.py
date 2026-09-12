"""Host-owned executable adapters for frozen infrastructure protocols."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from evolving_loop.data import ContextTask, _to_context_task
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry, task_registry_fingerprint
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import fingerprint_payload, require_sha256
from evolving_loop.v2.cooperative.adapters import CooperativeArtifactCatalog, CooperativePipelineAdapter
from evolving_loop.v2.numerical_qd.adapters import FrozenNumericalArtifactsV2

from .contracts import InfrastructureProtocolV2, KINDS


_IMPLEMENTATIONS = MappingProxyType({
    ("backbone", "last_value", 1): "last_value",
    ("backbone", "history_mean", 1): "history_mean",
    ("loader", "canonical_json", 1): "canonical_json",
    ("loader", "alternate_history_json", 1): "alternate_history_json",
    ("loader", "changed_history_json", 1): "changed_history_json",
    ("verifier_strategy", "exact_support", 1): "exact_support",
    ("verifier_strategy", "deduplicate_support", 1): "deduplicate_support",
    ("diagnostic_metric", "forecast_spread", 1): "forecast_spread",
    ("diagnostic_metric", "absolute_movement", 1): "absolute_movement",
    ("schema_migration", "identity_envelope", 1): "identity_envelope",
    ("schema_migration", "envelope_v2", 1): "envelope_v2",
})


def _freeze_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return MappingProxyType(dict(value))


def _finite_series(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numeric values")
    return result


def _forecast(implementation: str, history: Sequence[float], horizon: int) -> tuple[float, ...]:
    values = _finite_series(history, "history")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("forecast horizon must be a positive integer")
    value = values[-1] if implementation == "last_value" else sum(values) / len(values)
    return (value,) * horizon


def _diagnostic(
    implementation: str, history: Sequence[float], forecast: Sequence[float]
) -> dict[str, float]:
    old, new = _finite_series(history, "history"), _finite_series(forecast, "forecast")
    if implementation == "forecast_spread":
        value = max(new) - min(new)
    elif implementation == "absolute_movement":
        value = abs(new[-1] - old[-1])
    else:  # pragma: no cover - registry closure prevents this path
        raise ValueError("unknown diagnostic implementation")
    if not math.isfinite(value):
        raise ValueError("diagnostic must be finite")
    return {implementation: float(value)}


@dataclass(frozen=True, slots=True)
class ProtocolHostInputs:
    """Host-only inputs; no candidate may supply or mutate these authorities."""

    raw_fixture_records: Mapping[str, object]
    canonical_records: Mapping[str, object]
    catalog: CooperativeArtifactCatalog
    frozen_numerical: FrozenNumericalArtifactsV2
    retrieval_factory: Callable
    decision_factory: Callable
    committed_task_metadata: Mapping[str, object]
    l0_commitment_sha256: str
    primary_metric_cap: float
    baseline_verifier: Callable[[dict], bool]
    known_supports: Mapping[str, object]
    retrieval_skill_library: RetrievalSkillLibrary | None = None
    empty_skill_path: str | Path = "unused-protocol-retrieval-skills.json"

    def __post_init__(self) -> None:
        for field in ("raw_fixture_records", "canonical_records", "committed_task_metadata", "known_supports"):
            object.__setattr__(self, field, _freeze_mapping(getattr(self, field), field))
        if type(self.catalog) is not CooperativeArtifactCatalog:
            raise TypeError("protocol host requires a CooperativeArtifactCatalog")
        if type(self.frozen_numerical) is not FrozenNumericalArtifactsV2:
            raise TypeError("protocol host requires frozen Numerical artifacts")
        if not callable(self.retrieval_factory) or not callable(self.decision_factory):
            raise ValueError("protocol host factories must be callable")
        if not callable(self.baseline_verifier):
            raise ValueError("protocol host baseline_verifier must be callable")
        require_sha256(self.l0_commitment_sha256, "l0_commitment_sha256")
        if (
            isinstance(self.primary_metric_cap, bool)
            or not isinstance(self.primary_metric_cap, (int, float))
            or not math.isfinite(self.primary_metric_cap)
            or self.primary_metric_cap <= 0.0
        ):
            raise ValueError("primary_metric_cap must be positive and finite")
        object.__setattr__(self, "primary_metric_cap", float(self.primary_metric_cap))
        if self.retrieval_skill_library is not None and type(self.retrieval_skill_library) is not RetrievalSkillLibrary:
            raise TypeError("retrieval_skill_library must be a RetrievalSkillLibrary")
        object.__setattr__(self, "empty_skill_path", Path(self.empty_skill_path))


class ProtocolRuntimeRegistry:
    """Resolve only the fixed Host implementation map before any execution."""

    def resolve(
        self, protocol: InfrastructureProtocolV2, host_inputs: ProtocolHostInputs
    ) -> "ProtocolRuntime":
        if type(protocol) is not InfrastructureProtocolV2:
            raise TypeError("protocol must be an InfrastructureProtocolV2")
        if type(host_inputs) is not ProtocolHostInputs:
            raise TypeError("host_inputs must be ProtocolHostInputs")
        if protocol.l0_commitment_sha256 != host_inputs.l0_commitment_sha256:
            raise ValueError("protocol L0 commitment does not match host")
        resolved: dict[str, str] = {}
        for component in protocol.components:
            key = (component.kind, component.implementation_id, component.implementation_version)
            implementation = _IMPLEMENTATIONS.get(key)
            if component.artifact_schema_version != 1 or implementation is None:
                raise ValueError(f"unresolved protocol component: {key}")
            resolved[component.kind] = implementation
        if tuple(resolved) != KINDS:
            raise ValueError("protocol components are not a complete canonical map")
        return ProtocolRuntime(protocol, host_inputs, MappingProxyType(resolved))


@dataclass(frozen=True, slots=True)
class ProtocolRuntime:
    protocol: InfrastructureProtocolV2
    host_inputs: ProtocolHostInputs
    implementations: Mapping[str, str]

    def _canonical_task(self, task_id: str) -> ContextTask:
        try:
            record = self.host_inputs.canonical_records[task_id]
        except KeyError as error:
            raise ValueError(f"canonical task record is missing: {task_id}") from error
        if type(record) is ContextTask:
            task = record
        elif isinstance(record, Mapping):
            task = _to_context_task(dict(record))
        else:
            raise ValueError("canonical task records must be ContextTask or JSON objects")
        if task.numeric.task_id != task_id:
            raise ValueError("canonical record task ID mismatch")
        return task

    def _committed_hash(self, task_id: str) -> str:
        try:
            value = self.host_inputs.committed_task_metadata[task_id]
        except KeyError as error:
            raise ValueError(f"committed task metadata is missing: {task_id}") from error
        if isinstance(value, Mapping):
            value = value.get("task_sha256")
        return require_sha256(value, f"committed task SHA for {task_id}")

    def load_tasks(self) -> tuple[ContextTask, ...]:
        loader = self.implementations["loader"]
        if loader == "changed_history_json":
            raise ValueError("changed history fixtures are incompatible with the frozen registry")
        records = self.host_inputs.raw_fixture_records
        if not records:
            raise ValueError("protocol host has no raw fixture records")
        tasks = []
        for task_id in self.host_inputs.committed_task_metadata:
            if task_id not in records:
                raise ValueError(f"raw fixture record is missing: {task_id}")
            raw = records[task_id]
            if type(raw) is ContextTask:
                raw_task_id = raw.numeric.task_id
            elif isinstance(raw, Mapping):
                raw_task_id = raw.get("benchmark_id", raw.get("task_id"))
            else:
                raise ValueError("raw fixture records must be ContextTask or JSON objects")
            if raw_task_id != task_id:
                raise ValueError("raw fixture task ID mismatch")
            # alternate_history_json is deliberately an alias: it never supplies
            # bytes to the registry; the full canonical Host record does.
            task = self._canonical_task(task_id)
            if task_registry_fingerprint(task) != self._committed_hash(task_id):
                raise ValueError("canonical task does not match committed metadata")
            tasks.append(task)
        if len(tasks) != len({task.numeric.task_id for task in tasks}):
            raise ValueError("loaded task IDs must be unique")
        return tuple(tasks)

    def _pipeline(self) -> CooperativePipelineAdapter:
        return CooperativePipelineAdapter(
            self.host_inputs.catalog,
            self.host_inputs.retrieval_factory,
            self.host_inputs.decision_factory,
            metric_cap=self.host_inputs.primary_metric_cap,
            retrieval_skill_library=self.host_inputs.retrieval_skill_library,
            empty_skill_path=self.host_inputs.empty_skill_path,
        )

    def _derived_registry(
        self, source: FrozenNumericalArtifactsV2, tasks: tuple[ContextTask, ...]
    ) -> FrozenNumericalPackageRegistry:
        entries = []
        backbone = self.implementations["backbone"]
        for task in tasks:
            package = source.registry.package_for(task)
            forecast = _forecast(backbone, task.numeric.history_values, task.numeric.prediction_length)
            alternatives = tuple(replace(item, forecast=forecast) for item in package.ranked_alternatives)
            protected = next(item for item in alternatives if item.name == package.protected_baseline.name)
            selection = replace(package.selection_decision, forecast=forecast)
            entries.append((task, replace(
                package,
                final_forecast=forecast,
                protected_baseline=protected,
                ranked_alternatives=alternatives,
                selection_decision=selection,
            )))
        return FrozenNumericalPackageRegistry(
            entries,
            release_sha256=source.release.fingerprint,
            expected_task_ids=tuple(task.numeric.task_id for task in tasks),
        )

    def evaluate(
        self, bundle: EvolutionBundleV2, tasks: Sequence[ContextTask], stage: str
    ) -> PackageEvaluation:
        if type(bundle) is not EvolutionBundleV2:
            raise TypeError("bundle must be an EvolutionBundleV2")
        if type(stage) is not str or stage not in {"train", "dev"}:
            raise ValueError("protocol runtime stage must be exactly train or dev")
        canonical_tasks = self.load_tasks()
        if tuple(tasks) != canonical_tasks:
            raise ValueError("runtime evaluation requires exactly canonical loaded tasks")
        source = self.host_inputs.catalog.resolve_numerical(
            bundle.numerical_release_sha256, bundle.numerical_registry_sha256
        )
        retrieval = self.host_inputs.catalog.resolve_retrieval(bundle.retrieval_release_sha256)
        decision = self.host_inputs.catalog.resolve_decision(bundle.decision_policy_sha256)
        pipeline = self._pipeline()
        skills = pipeline._skills_for(retrieval)
        evaluation = PackagePipelineEvaluator._evaluate_components(
            candidate_sha256=fingerprint_payload({
                "protocol_sha256": self.protocol.fingerprint(),
                "bundle_sha256": bundle.fingerprint(),
            }),
            registry=self._derived_registry(source, canonical_tasks),
            tasks=canonical_tasks,
            stage=stage,
            retrieval_factory=lambda: self.host_inputs.retrieval_factory(retrieval.genome, skills),
            decision_factory=lambda: self.host_inputs.decision_factory(decision),
            metric_cap=self.host_inputs.primary_metric_cap,
            expected_retrieval_sha256=retrieval.genome.fingerprint(),
            expected_decision_prompt_sha256=hashlib.sha256(decision.prompt.encode("utf-8")).hexdigest(),
        )
        diagnostics = dict(evaluation.secondary_diagnostics)
        values = [self.diagnostics(task.numeric.history_values, row.final_forecast) for task, row in zip(canonical_tasks, evaluation.task_rows, strict=True)]
        name = self.implementations["diagnostic_metric"]
        diagnostics[name] = sum(value[name] for value in values) / len(values)
        return PackageEvaluation.from_rows(
            evaluation.candidate_sha256, evaluation.task_rows, evaluation.expected_task_ids, diagnostics
        )

    def verify(self, evidence: dict) -> bool:
        if type(evidence) is not dict or not self.host_inputs.baseline_verifier(evidence):
            return False
        document_id = evidence.get("document_id")
        if type(document_id) is not str:
            return False
        supports = evidence.get("support_ids", evidence.get("support_id"))
        if type(supports) is str:
            supplied = (supports,)
        elif isinstance(supports, (tuple, list)) and all(type(item) is str for item in supports):
            supplied = tuple(supports)
        else:
            return False
        known = self.host_inputs.known_supports.get(document_id)
        if type(known) is str:
            allowed = {known}
        elif isinstance(known, (tuple, list, set, frozenset)):
            allowed = set(known)
        else:
            return False
        if not supplied or not set(supplied) <= allowed:
            return False
        return self.implementations["verifier_strategy"] == "deduplicate_support" or len(supplied) == len(set(supplied))

    def diagnostics(self, history: tuple[float, ...], forecast: tuple[float, ...]) -> dict[str, float]:
        return _diagnostic(self.implementations["diagnostic_metric"], history, forecast)


def migrate_envelope(payload: dict, target_version: int) -> dict:
    """Validate an embedded artifact identity before a pure v1→v2 adaptation."""
    if type(payload) is not dict or type(target_version) is not int or target_version not in (1, 2):
        raise ValueError("payload and target_version must use supported exact schemas")
    fields = set(payload)
    v1 = {"schema_version", "artifact", "artifact_sha256"}
    v2 = {"schema_version", "content", "content_sha256"}
    if fields == v1 and payload.get("schema_version") == 1:
        artifact, digest = payload["artifact"], require_sha256(payload["artifact_sha256"], "artifact_sha256")
        if not isinstance(artifact, Mapping) or fingerprint_payload(artifact) != digest:
            raise ValueError("v1 artifact SHA does not match embedded content")
        return {"schema_version": 1, "artifact": dict(artifact), "artifact_sha256": digest} if target_version == 1 else {"schema_version": 2, "content": dict(artifact), "content_sha256": digest}
    if fields == v2 and payload.get("schema_version") == 2:
        content, digest = payload["content"], require_sha256(payload["content_sha256"], "content_sha256")
        if target_version != 2:
            raise ValueError("v2 envelopes cannot migrate backwards")
        if not isinstance(content, Mapping) or fingerprint_payload(content) != digest:
            raise ValueError("v2 content SHA does not match embedded content")
        return {"schema_version": 2, "content": dict(content), "content_sha256": digest}
    raise ValueError("envelope must use exactly the supported v1 or v2 schema")
