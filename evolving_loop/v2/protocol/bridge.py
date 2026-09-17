"""Production Host bridge from a sealed P3 closure to Protocol V2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evolving_loop.package_registry import task_registry_fingerprint

from ..contracts import fingerprint_payload
from ..real.host import RealHostRuntimeV2, select_real_task_projection
from .compatibility import CompatibilityCorpusV2, CompatibilityHostInputsV2
from .contracts import InfrastructureProtocolV2, ProtocolComponentV2
from .runtime import ProtocolHostInputs, ProtocolRuntimeRegistry


_P5_UNAVAILABLE = "p5_handoff_unavailable"
_VERIFIER_DOCUMENT = "p5-verifier-document"
_VERIFIER_SUPPORT = "p5-verifier-support"


@dataclass(frozen=True, slots=True)
class ProtocolRunCaseV2:
    """Generic protocol-run inputs produced from a verified P3 closure."""

    status: str
    config: dict[str, object] | None
    input_manifest: dict[str, object] | None
    registry: ProtocolRuntimeRegistry | None
    inputs: CompatibilityHostInputsV2 | None
    corpus: CompatibilityCorpusV2 | None
    seed_protocol: InfrastructureProtocolV2 | None

    def run(self, output_dir: Path, *, stop_after: int | None = None) -> dict:
        """Execute this case through the generic Protocol runner."""
        if (
            self.status != "protocol_case_ready"
            or self.config is None
            or self.input_manifest is None
            or self.registry is None
            or self.inputs is None
        ):
            return {"schema_version": 1, "status": _P5_UNAVAILABLE}
        from .runner import run_protocol_evolution

        return run_protocol_evolution(
            Path(output_dir),
            self.config,
            self.input_manifest,
            self.registry,
            host_inputs=self.inputs,
            stop_after=stop_after,
        )

def _seed_protocol(l0_commitment: str) -> InfrastructureProtocolV2:
    return InfrastructureProtocolV2(
        1,
        1,
        None,
        l0_commitment,
        (
            ProtocolComponentV2("backbone", "frozen_numerical", 1, 1),
            ProtocolComponentV2("loader", "canonical_json", 1, 1),
            ProtocolComponentV2("verifier_strategy", "exact_support", 1, 1),
            ProtocolComponentV2("diagnostic_metric", "forecast_spread", 1, 1),
            ProtocolComponentV2("schema_migration", "identity_envelope", 1, 1),
        ),
    )


def _replacement_templates() -> tuple[ProtocolComponentV2, ...]:
    return (
        ProtocolComponentV2("backbone", "history_mean", 1, 1),
        ProtocolComponentV2("loader", "alternate_history_json", 1, 1),
        ProtocolComponentV2("verifier_strategy", "deduplicate_support", 1, 1),
        ProtocolComponentV2("diagnostic_metric", "absolute_movement", 1, 1),
        ProtocolComponentV2("schema_migration", "envelope_v2", 1, 2),
    )


def _unavailable() -> ProtocolRunCaseV2:
    return ProtocolRunCaseV2(_P5_UNAVAILABLE, None, None, None, None, None, None)


def _second_bundle_for_compatibility(closure: P3BundleClosureV2):
    """Select a distinct, authenticated P3 candidate for compatibility."""
    active = closure.active_bundle
    active_evidence = active.acceptance_evidence_sha256
    accepted = None
    fallback = None
    for bundle, receipt in zip(
        closure.closed_candidate_bundles,
        closure.acceptance_evidence,
        strict=True,
    ):
        candidate_sha = bundle.fingerprint()
        if receipt.candidate_bundle_sha256 != candidate_sha:
            raise ValueError("P3 closure acceptance evidence does not bind its Bundle")
        if candidate_sha != active.fingerprint() and fallback is None:
            fallback = bundle
        if (
            candidate_sha != active.fingerprint()
            and receipt.decision == "accept"
            and receipt.fingerprint() == active_evidence
        ):
            accepted = bundle
    return accepted or fallback


def _acceptance_evidence(closure: P3BundleClosureV2) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for bundle, receipt in zip(
        closure.closed_candidate_bundles, closure.acceptance_evidence, strict=True
    ):
        if receipt.candidate_bundle_sha256 != bundle.fingerprint():
            raise ValueError("P3 closure acceptance evidence does not bind its Bundle")
        evidence[receipt.fingerprint()] = receipt.to_payload()
    return evidence


def _projection(task) -> dict[str, object]:
    """Return the deliberately label-free agent-facing task projection."""
    return {
        "task_id": task.numeric.task_id,
        "entity_name": task.numeric.entity_name,
        "target_name": task.target_name,
        "target_description": task.target_description,
        "history_values": list(task.numeric.history_values),
        "history_timestamps": list(task.history_timestamps),
        "prediction_length": task.numeric.prediction_length,
        "frequency": task.numeric.frequency,
        "documents": [
            {"document_id": document.document_id, "content": document.content}
            for document in task.documents
        ],
    }


def build_protocol_case_from_p3(
    closure: "P3BundleClosureV2",
    host: RealHostRuntimeV2,
    *,
    hard_limit_seconds: int,
) -> ProtocolRunCaseV2:
    """Create a closed two-Bundle Protocol V2 corpus without CLI dispatch."""
    # Import lazily: real.bridges reaches the root CLI, which imports the
    # protocol package while the P3 module is still initializing.
    from ..real.bridges import P3BundleClosureV2

    if type(closure) is not P3BundleClosureV2:
        raise TypeError("protocol bridge requires a sealed P3 closure")
    if type(host) is not RealHostRuntimeV2:
        raise TypeError("protocol bridge requires a real Host runtime")
    if type(hard_limit_seconds) is not int or hard_limit_seconds < 0:
        raise ValueError("protocol bridge hard_limit_seconds must be non-negative (0 means unlimited)")
    if closure.runtime_identity != host.resource_reporter_sha256:
        raise ValueError("protocol bridge P3 runtime commitment does not match Host")
    second = _second_bundle_for_compatibility(closure)
    if not closure.p5_handoff_available or second is None:
        return _unavailable()
    bundles = (closure.active_bundle, second)
    if len({bundle.fingerprint() for bundle in bundles}) != 2:
        return _unavailable()
    for bundle in bundles:
        closure.catalog.resolve_numerical(
            bundle.numerical_release_sha256, bundle.numerical_registry_sha256
        )
        closure.catalog.resolve_retrieval(bundle.retrieval_release_sha256)
        closure.catalog.resolve_decision(bundle.decision_policy_sha256)
    tasks = tuple(closure.train_tasks) + tuple(closure.dev_tasks)
    train, dev = select_real_task_projection(
        host.train_tasks,
        host.dev_tasks,
        train_size=host.projection_train_size,
        dev_size=host.projection_dev_size,
    )
    expected_total = host.projection_train_size + host.projection_dev_size
    if (
        len(tasks) != expected_total
        or tuple(closure.train_tasks) != train
        or tuple(closure.dev_tasks) != dev
    ):
        raise ValueError("protocol bridge requires the sealed P3 Train/Dev projection")
    task_shas = tuple(task_registry_fingerprint(task) for task in tasks)
    if len(set(task_shas)) != expected_total:
        raise ValueError("protocol bridge task identities must be unique")
    evidence = _acceptance_evidence(closure)
    envelopes = {
        fingerprint_payload(
            {
                "schema_version": 1,
                "artifact": bundle.to_payload(),
                "artifact_sha256": bundle.fingerprint(),
            }
        ): {
            "schema_version": 1,
            "artifact": bundle.to_payload(),
            "artifact_sha256": bundle.fingerprint(),
        }
        for bundle in bundles
    }
    fixtures = (
        {
            "baseline": True,
            "document_id": _VERIFIER_DOCUMENT,
            "support_id": _VERIFIER_SUPPORT,
        },
        {
            "baseline": True,
            "document_id": _VERIFIER_DOCUMENT,
            "support_id": "missing-support",
        },
        {
            "baseline": True,
            "document_id": _VERIFIER_DOCUMENT,
            "support_id": _VERIFIER_SUPPORT,
            "evaluation": "private",
        },
        {
            "baseline": True,
            "document_id": "fabricated-document",
            "support_id": _VERIFIER_SUPPORT,
        },
    )
    runtime_inputs = ProtocolHostInputs(
        raw_fixture_records={task.numeric.task_id: task for task in tasks},
        canonical_records={task.numeric.task_id: task for task in tasks},
        catalog=closure.catalog,
        frozen_numerical=closure.numerical,
        retrieval_factory=host.retrieval_factory,
        decision_factory=host.decision_factory,
        committed_task_metadata={
            task.numeric.task_id: task_registry_fingerprint(task) for task in tasks
        },
        l0_commitment_sha256=closure.active_bundle.protocol_fingerprint,
        primary_metric_cap=closure.metric_cap,
        baseline_verifier=lambda item: item.get("baseline") is True
        and "evaluation" not in item,
        known_supports={_VERIFIER_DOCUMENT: (_VERIFIER_SUPPORT,)},
        retrieval_skill_library=host.retrieval_skill_library,
        empty_skill_path=Path("p5-empty-retrieval-skills.json"),
        fixture_scope="ordinary",
    )
    train_n = len(tuple(closure.train_tasks))
    split = {
        "train_task_sha256s": task_shas[:train_n],
        "dev_task_sha256s": task_shas[train_n:],
        "public_task_sha256s": (),
    }
    corpus = CompatibilityCorpusV2(
        1,
        closure.active_bundle.protocol_fingerprint,
        task_shas[:train_n],
        task_shas[train_n:],
        tuple(bundle.fingerprint() for bundle in bundles),
        tuple(bundle.fingerprint() for bundle in bundles) + tuple(envelopes),
        fingerprint_payload({"fixtures": list(fixtures)}),
    )
    inputs = CompatibilityHostInputsV2(
        runtime_inputs,
        split,
        {bundle.fingerprint(): bundle for bundle in bundles},
        envelopes,
        fixtures,
        closure.runtime_identity,
        {task_registry_fingerprint(task): _projection(task) for task in tasks},
        bundle_acceptance_evidence=evidence,
    )
    seed = _seed_protocol(closure.active_bundle.protocol_fingerprint)
    templates = _replacement_templates()
    config = {
        "schema_version": 1,
        "profile": "real",
        "seed": 0,
        "max_proposals": len(templates),
        "hard_limit_seconds": hard_limit_seconds,
    }
    manifest = {
        "schema_version": 1,
        "l0_commitment": closure.active_bundle.protocol_fingerprint,
        "runtime_fingerprint": closure.runtime_identity,
        "corpus": corpus.to_payload(),
        "seed_protocol": seed.to_payload(),
        "host_input_files": [],
        "frozen_bundle_sha256": closure.active_bundle.fingerprint(),
        "replacement_templates": [template.to_payload() for template in templates],
    }
    return ProtocolRunCaseV2(
        "protocol_case_ready",
        config,
        manifest,
        ProtocolRuntimeRegistry(),
        inputs,
        corpus,
        seed,
    )


__all__ = ["ProtocolRunCaseV2", "build_protocol_case_from_p3"]
