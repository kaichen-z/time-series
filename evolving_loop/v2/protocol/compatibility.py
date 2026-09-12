"""Host-only compatibility replay and sealed L1 protocol decisions."""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_registry import task_registry_fingerprint
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from evolving_loop.v2.store import write_once_json

from .contracts import InfrastructureProtocolV2
from .runtime import ProtocolHostInputs, ProtocolRuntimeRegistry


_CORPUS_FIELDS = (
    "schema_version", "l0_commitment_sha256", "train_task_sha256s", "dev_task_sha256s",
    "archive_bundle_sha256s", "artifact_sha256s", "verifier_fixture_sha256",
)
_EVIDENCE_FIELDS = (
    "schema_version", "old_protocol_sha256", "proposed_protocol_sha256", "corpus_sha256",
    "runtime_fingerprint", "checks", "train_rows", "migration_mapping", "sealed_dev_sha256",
)
_DECISION_FIELDS = ("schema_version", "decision", "reason_codes")
_CHECK_KEYS = ("l0", "tasks", "firewall", "artifacts", "verifier", "train", "dev")
_REASONS = ("l0_mismatch", "task_hash_mismatch", "label_boundary", "artifact_migration", "verifier_failure", "train_regression", "dev_regression")


def _exact(payload: object, fields: tuple[str, ...], name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != set(fields) or any(type(key) is not str for key in payload):
        raise ValueError(f"{name} must use the exact schema")
    return dict(payload)


def _sha_tuple(value: object, name: str, *, exact: int | None = None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an ordered SHA-256 list")
    result = tuple(require_sha256(item, name) for item in value)
    if len(result) != len(set(result)) or (exact is not None and len(result) != exact):
        raise ValueError(f"{name} must have unique required entries")
    return result


@dataclass(frozen=True, slots=True)
class CompatibilityCorpusV2:
    schema_version: int
    l0_commitment_sha256: str
    train_task_sha256s: tuple[str, ...]
    dev_task_sha256s: tuple[str, ...]
    archive_bundle_sha256s: tuple[str, ...]
    artifact_sha256s: tuple[str, ...]
    verifier_fixture_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        require_sha256(self.l0_commitment_sha256, "l0_commitment_sha256")
        train = _sha_tuple(self.train_task_sha256s, "train_task_sha256s", exact=4)
        dev = _sha_tuple(self.dev_task_sha256s, "dev_task_sha256s", exact=1)
        bundles = _sha_tuple(self.archive_bundle_sha256s, "archive_bundle_sha256s", exact=2)
        artifacts = _sha_tuple(self.artifact_sha256s, "artifact_sha256s")
        if set(train) & set(dev):
            raise ValueError("Train and Dev task identities must be disjoint")
        if not set(bundles) <= set(artifacts):
            raise ValueError("artifact_sha256s must close archive Bundles")
        require_sha256(self.verifier_fixture_sha256, "verifier_fixture_sha256")
        object.__setattr__(self, "train_task_sha256s", train)
        object.__setattr__(self, "dev_task_sha256s", dev)
        object.__setattr__(self, "archive_bundle_sha256s", bundles)
        object.__setattr__(self, "artifact_sha256s", artifacts)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CompatibilityCorpusV2":
        values = _exact(payload, _CORPUS_FIELDS, "compatibility corpus")
        return cls(values["schema_version"], values["l0_commitment_sha256"], values["train_task_sha256s"], values["dev_task_sha256s"], values["archive_bundle_sha256s"], values["artifact_sha256s"], values["verifier_fixture_sha256"])  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) if not isinstance(getattr(self, name), tuple) else list(getattr(self, name)) for name in _CORPUS_FIELDS}

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class CompatibilityEvidenceV2:
    schema_version: int
    old_protocol_sha256: str
    proposed_protocol_sha256: str
    corpus_sha256: str
    runtime_fingerprint: str
    checks: Mapping[str, bool]
    train_rows: tuple[Mapping[str, object], ...]
    migration_mapping: tuple[Mapping[str, str], ...]
    sealed_dev_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for field in _EVIDENCE_FIELDS[1:5]:
            require_sha256(getattr(self, field), field)
        checks = dict(self.checks)
        if set(checks) != set(_CHECK_KEYS) or any(type(value) is not bool for value in checks.values()):
            raise ValueError("checks must contain exactly the canonical boolean checks")
        rows = tuple(dict(row) for row in self.train_rows)
        for row in rows:
            expected = {"protocol_sha256", "bundle_sha256", "stage", "coverage", "mean_smae", "mean_srmse", "invalid_count", "catastrophic_count", "evaluation_sha256"}
            if set(row) != expected or row["stage"] != "train":
                raise ValueError("train rows must use the exact public-safe aggregate schema")
            require_sha256(row["protocol_sha256"], "train protocol_sha256")
            require_sha256(row["bundle_sha256"], "train bundle_sha256")
            require_sha256(row["evaluation_sha256"], "train evaluation_sha256")
        mapping = tuple(dict(item) for item in self.migration_mapping)
        for item in mapping:
            if set(item) != {"old_envelope_sha256", "new_envelope_sha256"}:
                raise ValueError("migration mapping must use exact envelope SHA fields")
            require_sha256(item["old_envelope_sha256"], "old_envelope_sha256")
            require_sha256(item["new_envelope_sha256"], "new_envelope_sha256")
        if self.sealed_dev_sha256 is not None:
            require_sha256(self.sealed_dev_sha256, "sealed_dev_sha256")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "train_rows", rows)
        object.__setattr__(self, "migration_mapping", mapping)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CompatibilityEvidenceV2":
        values = _exact(payload, _EVIDENCE_FIELDS, "compatibility evidence")
        return cls(values["schema_version"], values["old_protocol_sha256"], values["proposed_protocol_sha256"], values["corpus_sha256"], values["runtime_fingerprint"], values["checks"], tuple(values["train_rows"]), tuple(values["migration_mapping"]), values["sealed_dev_sha256"])  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "old_protocol_sha256": self.old_protocol_sha256,
            "proposed_protocol_sha256": self.proposed_protocol_sha256, "corpus_sha256": self.corpus_sha256,
            "runtime_fingerprint": self.runtime_fingerprint, "checks": dict(self.checks),
            "train_rows": [dict(row) for row in self.train_rows], "migration_mapping": [dict(row) for row in self.migration_mapping],
            "sealed_dev_sha256": self.sealed_dev_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class ProtocolDecisionV2:
    schema_version: int
    decision: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.decision not in {"accept", "reject"} or not self.reason_codes:
            raise ValueError("invalid protocol decision")
        if self.decision == "accept" and self.reason_codes != ("compatible",):
            raise ValueError("accepted decision must be compatible")
        if self.decision == "reject" and any(reason not in _REASONS for reason in self.reason_codes):
            raise ValueError("invalid protocol rejection reason")

    def to_payload(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "decision": self.decision, "reason_codes": list(self.reason_codes)}

    def canonical_bytes(self) -> bytes:
        return canonical_v2_bytes(self.to_payload())

    def fingerprint(self) -> str:
        return fingerprint_payload(self.to_payload())


@dataclass(frozen=True, slots=True)
class CompatibilityHostInputsV2:
    """Additional immutable Host-only authorities needed for archive replay."""
    runtime_inputs: ProtocolHostInputs
    split_manifest: Mapping[str, object]
    archive_bundles: Mapping[str, EvolutionBundleV2]
    artifact_envelopes: Mapping[str, Mapping[str, object]]
    verifier_fixtures: tuple[Mapping[str, object], ...]
    runtime_fingerprint: str
    evaluation_cache: MutableMapping[tuple[str, str, str, str, str], PackageEvaluation] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_inputs, ProtocolHostInputs):
            raise TypeError("compatibility Host inputs require ProtocolHostInputs")
        require_sha256(self.runtime_fingerprint, "runtime_fingerprint")
        if len(self.verifier_fixtures) != 4 or not all(isinstance(item, Mapping) for item in self.verifier_fixtures):
            raise ValueError("compatibility Host inputs require four verifier fixtures")


def nonregressing(old: PackageEvaluation, new: PackageEvaluation) -> bool:
    return (
        new.coverage == 1.0
        and new.invalid_count <= old.invalid_count
        and new.catastrophic_count <= old.catastrophic_count
        and new.mean_smae <= old.mean_smae + 1e-12
        and new.mean_srmse <= old.mean_srmse + 1e-12
    )


def _finite(evaluation: PackageEvaluation) -> bool:
    values = (evaluation.coverage, evaluation.mean_smae, evaluation.mean_srmse, evaluation.mean_joint)
    return all(math.isfinite(value) for value in values) and all(
        type(value) is int and value >= 0 for value in (evaluation.invalid_count, evaluation.catastrophic_count)
    )


def _safe_row(protocol_sha: str, bundle_sha: str, evaluation: PackageEvaluation) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha, "bundle_sha256": bundle_sha, "stage": "train",
        "coverage": evaluation.coverage, "mean_smae": evaluation.mean_smae, "mean_srmse": evaluation.mean_srmse,
        "invalid_count": evaluation.invalid_count, "catastrophic_count": evaluation.catastrophic_count,
        "evaluation_sha256": evaluation.fingerprint,
    }


def _selected_tasks(runtime, wanted: tuple[str, ...]):
    loaded = runtime.load_tasks()
    by_sha = {task_registry_fingerprint(task): task for task in loaded}
    if set(by_sha) != {task_registry_fingerprint(task) for task in loaded} or any(sha not in by_sha for sha in wanted):
        raise ValueError("task hash mismatch")
    return tuple(by_sha[sha] for sha in wanted)


def _unwrap(payload: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    if payload.get("schema_version") == 1 and set(payload) == {"schema_version", "artifact", "artifact_sha256"}:
        content, digest = payload["artifact"], payload["artifact_sha256"]
    elif payload.get("schema_version") == 2 and set(payload) == {"schema_version", "content", "content_sha256"}:
        content, digest = payload["content"], payload["content_sha256"]
    else:
        raise ValueError("unrecognized artifact envelope")
    if not isinstance(content, Mapping) or fingerprint_payload(content) != digest:
        raise ValueError("artifact envelope identity mismatch")
    return content, require_sha256(digest, "embedded artifact SHA")


def _evidence(old: InfrastructureProtocolV2, proposed: InfrastructureProtocolV2, corpus: CompatibilityCorpusV2, host: CompatibilityHostInputsV2, checks: dict[str, bool], rows=(), mappings=(), sealed=None) -> CompatibilityEvidenceV2:
    return CompatibilityEvidenceV2(1, old.fingerprint(), proposed.fingerprint(), corpus.fingerprint(), host.runtime_fingerprint, checks, tuple(rows), tuple(mappings), sealed)


def check_compatibility(old: InfrastructureProtocolV2, proposed: InfrastructureProtocolV2, corpus: CompatibilityCorpusV2, registry: ProtocolRuntimeRegistry, *, host_inputs: CompatibilityHostInputsV2, sealed_store: Path) -> CompatibilityEvidenceV2:
    if not isinstance(old, InfrastructureProtocolV2) or not isinstance(proposed, InfrastructureProtocolV2):
        raise TypeError("compatibility requires infrastructure protocols")
    if not isinstance(corpus, CompatibilityCorpusV2) or not isinstance(host_inputs, CompatibilityHostInputsV2):
        raise TypeError("compatibility requires canonical corpus and Host inputs")
    if not isinstance(registry, ProtocolRuntimeRegistry):
        raise TypeError("compatibility requires ProtocolRuntimeRegistry")
    split = host_inputs.split_manifest
    public = split.get("public_task_sha256s") if isinstance(split, Mapping) else None
    if not isinstance(public, (list, tuple)) or public:
        raise ValueError("Public membership is forbidden in the compatibility corpus")
    if tuple(split.get("train_task_sha256s", ())) != corpus.train_task_sha256s or tuple(split.get("dev_task_sha256s", ())) != corpus.dev_task_sha256s:
        raise ValueError("compatibility split manifest is not explicitly verified")
    checks = {key: False for key in _CHECK_KEYS}
    host = host_inputs.runtime_inputs
    if old.l0_commitment_sha256 != proposed.l0_commitment_sha256 or old.l0_commitment_sha256 != corpus.l0_commitment_sha256 or host.l0_commitment_sha256 != corpus.l0_commitment_sha256:
        return _evidence(old, proposed, corpus, host_inputs, checks)
    try:
        old_runtime, proposed_runtime = registry.resolve(old, host), registry.resolve(proposed, host)
    except (TypeError, ValueError):
        return _evidence(old, proposed, corpus, host_inputs, checks)
    checks["l0"] = True
    try:
        old_train, proposed_train = _selected_tasks(old_runtime, corpus.train_task_sha256s), _selected_tasks(proposed_runtime, corpus.train_task_sha256s)
        old_dev, proposed_dev = _selected_tasks(old_runtime, corpus.dev_task_sha256s), _selected_tasks(proposed_runtime, corpus.dev_task_sha256s)
        if tuple(task_registry_fingerprint(task) for task in old_train) != corpus.train_task_sha256s or tuple(task_registry_fingerprint(task) for task in proposed_train) != corpus.train_task_sha256s:
            raise ValueError("task hash mismatch")
    except ValueError:
        return _evidence(old, proposed, corpus, host_inputs, checks)
    checks["tasks"] = True
    # P3 agents receive their own projections; labels remain only in this Host scorer.
    checks["firewall"] = True
    mappings: list[dict[str, str]] = []
    try:
        for old_sha, envelope in sorted(host_inputs.artifact_envelopes.items()):
            require_sha256(old_sha, "artifact envelope SHA")
            if fingerprint_payload(envelope) != old_sha or old_sha not in corpus.artifact_sha256s:
                raise ValueError("artifact envelope SHA mismatch")
            original, digest = _unwrap(envelope)
            old_target = 1 if old_runtime.implementations["schema_migration"] == "identity_envelope" else 2
            new_target = 1 if proposed_runtime.implementations["schema_migration"] == "identity_envelope" else 2
            old_migrated = old_runtime.migrate_envelope(dict(envelope), old_target)
            new_migrated = proposed_runtime.migrate_envelope(dict(envelope), new_target)
            if _unwrap(old_migrated) != (original, digest) or _unwrap(new_migrated) != (original, digest) or proposed_runtime.migrate_envelope(new_migrated, new_target) != new_migrated:
                raise ValueError("artifact migration mismatch")
            mappings.append({"old_envelope_sha256": old_sha, "new_envelope_sha256": fingerprint_payload(new_migrated)})
    except ValueError:
        return _evidence(old, proposed, corpus, host_inputs, checks, mappings=mappings)
    checks["artifacts"] = True
    expected = (True, False, False, False)
    if tuple(old_runtime.verify(dict(item)) for item in host_inputs.verifier_fixtures) != expected or tuple(proposed_runtime.verify(dict(item)) for item in host_inputs.verifier_fixtures) != expected:
        return _evidence(old, proposed, corpus, host_inputs, checks, mappings=mappings)
    checks["verifier"] = True
    cache = host_inputs.evaluation_cache if host_inputs.evaluation_cache is not None else {}
    rows: list[dict[str, object]] = []
    try:
        for bundle_sha in corpus.archive_bundle_sha256s:
            bundle = host_inputs.archive_bundles[bundle_sha]
            if bundle.fingerprint() != bundle_sha:
                raise ValueError("archive bundle SHA mismatch")
            values = []
            for protocol, runtime, tasks in ((old, old_runtime, old_train), (proposed, proposed_runtime, proposed_train)):
                key = (protocol.fingerprint(), host_inputs.runtime_fingerprint, corpus.fingerprint(), bundle_sha, "train")
                evaluation = cache.get(key)
                if evaluation is None:
                    evaluation = runtime.evaluate(bundle, tasks, "train")
                    cache[key] = evaluation
                if not isinstance(evaluation, PackageEvaluation) or not _finite(evaluation):
                    raise ValueError("nonfinite train aggregate")
                values.append(evaluation)
                rows.append(_safe_row(protocol.fingerprint(), bundle_sha, evaluation))
            if not _finite(values[0]) or not nonregressing(values[0], values[1]):
                raise ValueError("train regression")
    except (KeyError, ValueError):
        return _evidence(old, proposed, corpus, host_inputs, checks, rows=rows, mappings=mappings)
    checks["train"] = True
    try:
        bundle_sha = corpus.archive_bundle_sha256s[0]
        bundle = host_inputs.archive_bundles[bundle_sha]
        dev_values = []
        for protocol, runtime, tasks in ((old, old_runtime, old_dev), (proposed, proposed_runtime, proposed_dev)):
            key = (protocol.fingerprint(), host_inputs.runtime_fingerprint, corpus.fingerprint(), bundle_sha, "dev")
            evaluation = cache.get(key)
            if evaluation is None:
                evaluation = runtime.evaluate(bundle, tasks, "dev")
                cache[key] = evaluation
            if not isinstance(evaluation, PackageEvaluation) or not _finite(evaluation):
                raise ValueError("nonfinite Dev aggregate")
            dev_values.append(evaluation)
        if not nonregressing(dev_values[0], dev_values[1]):
            raise ValueError("Dev regression")
        payload = {"schema_version": 1, "old_evaluation": dev_values[0].to_payload(), "proposed_evaluation": dev_values[1].to_payload()}
        sealed = fingerprint_payload(payload)
        path = Path(sealed_store) / "sealed" / f"{sealed}.json"
        write_once_json(path, payload)
        if path.read_bytes() != canonical_v2_bytes(payload):
            raise ValueError("sealed Dev readback mismatch")
    except (KeyError, ValueError):
        return _evidence(old, proposed, corpus, host_inputs, checks, rows=rows, mappings=mappings)
    checks["dev"] = True
    return _evidence(old, proposed, corpus, host_inputs, checks, rows=rows, mappings=mappings, sealed=sealed)


def decide_protocol(evidence: CompatibilityEvidenceV2) -> ProtocolDecisionV2:
    if not isinstance(evidence, CompatibilityEvidenceV2):
        raise TypeError("decision requires compatibility evidence")
    if all(evidence.checks.values()):
        return ProtocolDecisionV2(1, "accept", ("compatible",))
    reasons = tuple(reason for key, reason in zip(_CHECK_KEYS, _REASONS, strict=True) if not evidence.checks[key])
    return ProtocolDecisionV2(1, "reject", reasons)
