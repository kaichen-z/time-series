"""Closed, typed material/control taxonomy for Host-owned artifact accounting.

Kinds are selected by typed writers, never by the absence/presence of marker
keys. Unknown kinds and payloads outside their exact contract fail closed.
Control records may reference/duplicate material only under the store's usual
verified immutable graph; they cannot authorize new executable material.
"""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from common.payload import strict_json_loads
from ..budget import _CHECKPOINT_FIELDS as BUDGET_FIELDS
from ..bundle import EvolutionBundleV2
from ..contracts import _require_exact_schema, canonical_v2_bytes
from .config import NumericalQDConfigV2
from .contracts import (HyperbandRungV2, HyperbandStateV2, HyperbandTaskResultV2,
    NumericalEvaluationV2, NumericalGenomeV2, NumericalInventoryV2, NumericalMutationPolicyV2,
    NumericalProposerPromptV2, NumericalQDCheckpointV2, NumericalQDEntryV2, RungManifestV2)
from .map_elites import NumericalQDArchive


class ArtifactKindV2(str, Enum):
    SOURCE = "source"
    CONFIG = "config"
    GENOME = "genome"
    INVENTORY = "inventory"
    SCREENING_POLICY = "screening_policy"
    COMBINED_POLICY = "combined_policy"
    RECIPE_POLICY = "recipe_policy"
    STRUCTURAL_POLICY = "structural_policy"
    MUTATION_POLICY = "mutation_policy"
    PROMPT = "prompt"
    PROPOSER_REQUEST = "proposer_request"
    PROPOSAL_ATTEMPT = "proposal_attempt"
    EXECUTABLE_CHILD = "executable_child"
    EVALUATION = "evaluation"
    RUNG_MANIFEST = "rung_manifest"
    TASK_RESULT = "task_result"
    QD_ENTRY = "qd_entry"
    CELL_SUBSET = "cell_subset"
    FROZEN_PAIR = "frozen_pair"
    BOOTSTRAP_FORECAST = "bootstrap_forecast"
    BUNDLE = "bundle"
    QD_ARCHIVE = "qd_archive"
    HYPERBAND_STATE = "hyperband_state"
    RUNG_RECORD = "rung_record"
    BUDGET_CHECKPOINT = "budget_checkpoint"
    KERNEL_CHECKPOINT = "kernel_checkpoint"
    RUNNER_CHECKPOINT = "runner_checkpoint"
    MANIFEST = "manifest"
    GENERATION_STATUS = "generation_status"
    PARTIAL_RUNG = "partial_rung"
    BOOTSTRAP_PREFLIGHT = "bootstrap_preflight"
    BOOTSTRAP_ADMISSION = "bootstrap_admission"
    BOOTSTRAP_CLOSURE = "bootstrap_closure"
    BOOTSTRAP_REPLAY = "bootstrap_replay"
    BOOTSTRAP_RECEIPT = "bootstrap_receipt"


@dataclass(frozen=True)
class ArtifactSpecV2:
    billable: bool
    contract: type | None = None
    payload_fields: tuple[str, ...] | None = None


K = ArtifactKindV2
ARTIFACT_KINDS = MappingProxyType({
    K.SOURCE: ArtifactSpecV2(True),
    K.CONFIG: ArtifactSpecV2(True, NumericalQDConfigV2),
    K.GENOME: ArtifactSpecV2(True, NumericalGenomeV2),
    K.INVENTORY: ArtifactSpecV2(True, NumericalInventoryV2),
    K.SCREENING_POLICY: ArtifactSpecV2(True, payload_fields=("metric_policy", "entries", "fallback_names")),
    K.COMBINED_POLICY: ArtifactSpecV2(True, payload_fields=("policies",)),
    K.RECIPE_POLICY: ArtifactSpecV2(True, payload_fields=("name", "kind", "parents", "fallback_parent", "assumptions")),
    K.STRUCTURAL_POLICY: ArtifactSpecV2(True, payload_fields=("schema_version", "operator", "parents", "applicability_cells")),
    K.MUTATION_POLICY: ArtifactSpecV2(True, NumericalMutationPolicyV2),
    K.PROMPT: ArtifactSpecV2(True, NumericalProposerPromptV2),
    K.PROPOSER_REQUEST: ArtifactSpecV2(True),
    K.PROPOSAL_ATTEMPT: ArtifactSpecV2(True, payload_fields=("context", "batch")),
    K.EXECUTABLE_CHILD: ArtifactSpecV2(True, payload_fields=("materialized_numerical_child",)),
    K.EVALUATION: ArtifactSpecV2(True, NumericalEvaluationV2),
    K.RUNG_MANIFEST: ArtifactSpecV2(True, RungManifestV2),
    K.TASK_RESULT: ArtifactSpecV2(True, HyperbandTaskResultV2),
    K.QD_ENTRY: ArtifactSpecV2(True, NumericalQDEntryV2),
    K.CELL_SUBSET: ArtifactSpecV2(True, payload_fields=("schema_version", "kind", "cell", "parent_manifest_sha256",
        "split_sha256", "protocol_sha256", "task_ids", "tasks")),
    K.FROZEN_PAIR: ArtifactSpecV2(True, payload_fields=("supply", "registry")),
    K.BOOTSTRAP_FORECAST: ArtifactSpecV2(True, payload_fields=("admission_sha256", "status", "forecast")),
    K.BUNDLE: ArtifactSpecV2(False, EvolutionBundleV2),
    K.QD_ARCHIVE: ArtifactSpecV2(False, NumericalQDArchive),
    K.HYPERBAND_STATE: ArtifactSpecV2(False, HyperbandStateV2),
    K.RUNG_RECORD: ArtifactSpecV2(False, HyperbandRungV2),
    K.BUDGET_CHECKPOINT: ArtifactSpecV2(False, payload_fields=tuple(BUDGET_FIELDS)),
    K.KERNEL_CHECKPOINT: ArtifactSpecV2(False),
    K.RUNNER_CHECKPOINT: ArtifactSpecV2(False, NumericalQDCheckpointV2),
    K.MANIFEST: ArtifactSpecV2(False, payload_fields=("schema_version", "system")),
    K.GENERATION_STATUS: ArtifactSpecV2(False, payload_fields=("numerical_qd_step",)),
    K.PARTIAL_RUNG: ArtifactSpecV2(False, payload_fields=("closed_partial_rung",)),
    K.BOOTSTRAP_PREFLIGHT: ArtifactSpecV2(False, payload_fields=("schema_version", "stage", "seed_supply_sha256",
        "protocol_sha256", "budget_plan_sha256", "input_sha256s", "estimate")),
    K.BOOTSTRAP_ADMISSION: ArtifactSpecV2(False, payload_fields=("sequence", "previous_sha256", "kind", "arguments", "charged_use")),
    K.BOOTSTRAP_CLOSURE: ArtifactSpecV2(False, payload_fields=("sequence", "previous_sha256", "kind", "cache_sha256")),
    K.BOOTSTRAP_REPLAY: ArtifactSpecV2(False, payload_fields=("sequence", "previous_sha256", "kind", "replay_index", "cache_sha256")),
    K.BOOTSTRAP_RECEIPT: ArtifactSpecV2(False, payload_fields=("schema_version", "preflight_sha256", "segment_index",
        "previous_receipt_sha256", "replay_total", "replay_consumed", "status", "reason", "events", "chain_sha256",
        "resource_use", "reservation_sha256", "allowed", "closure_reason", "budget_before", "budget_after")),
})
_TYPES = {spec.contract: kind for kind, spec in ARTIFACT_KINDS.items() if spec.contract is not None}


def artifact_kind(value):
    """Only exact registered Python contracts can infer their writer kind."""
    try:
        return _TYPES[type(value)]
    except KeyError as error:
        raise ValueError("artifact writer requires an explicit registered kind") from error


def validate_artifact(kind, raw):
    kind = ArtifactKindV2(kind)
    spec = ARTIFACT_KINDS[kind]
    if type(raw) is not bytes:
        raise TypeError("artifact accounting requires exact bytes")
    if kind is K.SOURCE:
        raw.decode("utf-8")
        return spec
    payload = strict_json_loads(raw.decode("utf-8"), context="typed artifact")
    if type(payload) is not dict or canonical_v2_bytes(payload) != raw:
        raise ValueError("artifact requires canonical JSON object bytes")
    if spec.contract is not None:
        spec.contract.from_payload(payload)
    elif spec.payload_fields is not None:
        _require_exact_schema(payload, spec.payload_fields, field=kind.value)
    if kind is K.PROPOSER_REQUEST:
        from .proposers import primitive_proposer_request
        primitive_proposer_request(**payload)
    elif kind is K.KERNEL_CHECKPOINT:
        from ..kernel import EvolutionKernel
        _require_exact_schema(payload, EvolutionKernel._CHECKPOINT_FIELDS, field=kind.value)
    elif kind is K.GENERATION_STATUS:
        from .contracts import TrainMutationFeedbackV2
        row = _require_exact_schema(payload["numerical_qd_step"], ("generation", "status", "active_bundle_sha256",
            "winner_genome_sha256", "proposal_attempt_sha256", "train_feedback"), field=kind.value)
        TrainMutationFeedbackV2.from_payload(row["train_feedback"])
        if type(row["generation"]) is not int or type(row["status"]) is not str:
            raise ValueError("generation control requires primitive status")
    elif kind is K.PARTIAL_RUNG:
        from .contracts import HyperbandBudgetOutcomeV2
        row = _require_exact_schema(payload["closed_partial_rung"], ("state", "manifest", "reason", "task_results",
            "unpersisted_task_results", "budget_outcome"), field=kind.value)
        HyperbandStateV2.from_payload(row["state"])
        RungManifestV2.from_payload(row["manifest"])
        HyperbandBudgetOutcomeV2.from_payload(row["budget_outcome"])
        for item in row["task_results"]:
            _require_exact_schema(item, ("task_sha256", "result"), field="partial task reference")
            HyperbandTaskResultV2.from_payload(item["result"])
    return spec


def billable_size(kind, raw):
    return len(raw) if validate_artifact(kind, raw).billable else 0
