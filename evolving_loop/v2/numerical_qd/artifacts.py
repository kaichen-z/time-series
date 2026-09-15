"""Closed, typed material/control taxonomy for Host-owned artifact accounting.

Kinds are selected by typed writers, never by the absence/presence of marker
keys. Unknown kinds and payloads outside their exact contract fail closed.
Control records may reference/duplicate material only under the store's usual
verified immutable graph; they cannot authorize new executable material.
"""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from pathlib import PurePosixPath

from common.payload import strict_json_loads
from ..budget import BudgetLedger, ResourceUse, _CHECKPOINT_FIELDS as BUDGET_FIELDS
from ..bundle import EvolutionBundleV2
from ..contracts import _require_exact_schema, canonical_v2_bytes, fingerprint_payload, require_sha256
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
    TASK_SHORTLIST = "task_shortlist"
    SHORTLIST_POLICY = "shortlist_policy"
    SHORTLIST_INDEX = "shortlist_index"
    HINDCAST_DIAGNOSTICS = "hindcast_diagnostics"
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
    K.TASK_SHORTLIST: ArtifactSpecV2(True, payload_fields=("schema_version", "task_input_sha256",
        "dictionary_sha256", "policy_sha256", "candidate_names", "exclusion_reasons",
        "shortlist_underfilled", "public_test_accessed")),
    K.SHORTLIST_POLICY: ArtifactSpecV2(True, payload_fields=("schema_version", "minimum_candidates",
        "target_candidates", "maximum_candidates")),
    K.SHORTLIST_INDEX: ArtifactSpecV2(True, payload_fields=("schema_version", "policy_sha256",
        "entries", "public_test_accessed")),
    K.HINDCAST_DIAGNOSTICS: ArtifactSpecV2(True, payload_fields=("schema_version", "task_id",
        "task_input_sha256", "rows", "public_test_accessed")),
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


def validate_unpersisted_task(row):
    _require_exact_schema(row, ("task_sha256", "task_id", "candidate_sha256", "cache_key", "result_sha256",
        "status", "resource_use"), field="unpersisted partial task")
    for field in ("task_sha256", "candidate_sha256", "cache_key", "result_sha256"):
        require_sha256(row[field], field)
    if type(row["task_id"]) is not str or not row["task_id"] or row["status"] != "persistence_denied":
        raise ValueError("unpersisted partial task requires primitive identity and persistence_denied status")
    return ResourceUse.from_payload(row["resource_use"])


def validate_material_receipts(rows):
    """Metadata only: a Host-successful material write, never its payload."""
    if type(rows) is not list:
        raise ValueError("material receipts must be a list")
    paths = set()
    for row in rows:
        _require_exact_schema(row, ("kind", "relative_path", "content_sha256", "size_bytes"), field="material receipt")
        if not ARTIFACT_KINDS[ArtifactKindV2(row["kind"])].billable:
            raise ValueError("material receipt must reference a billable kind")
        path = row["relative_path"]
        if (type(path) is not str or not path or PurePosixPath(path).is_absolute()
                or PurePosixPath(path).as_posix() != path or ".." in PurePosixPath(path).parts or path in paths):
            raise ValueError("material receipt requires a unique canonical relative path")
        require_sha256(row["content_sha256"], "material content SHA")
        if type(row["size_bytes"]) is not int or row["size_bytes"] <= 0:
            raise ValueError("material receipt size must be a positive integer")
        paths.add(path)
    return rows


def _optional_reason(value):
    if value is not None and (type(value) is not str or not value):
        raise ValueError("receipt reason must be None or a nonempty string")


def _bootstrap_budget(value):
    budget = _require_exact_schema(value, BUDGET_FIELDS, field="bootstrap budget checkpoint")
    if type(budget["schema_version"]) is not int or budget["schema_version"] != 1:
        raise ValueError("bootstrap budget schema mismatch")
    for field in ("plan_sha256", "checkpoint_sha256"):
        require_sha256(budget[field], field)
    if BudgetLedger.checkpoint_sha256(budget) != budget["checkpoint_sha256"]:
        raise ValueError("bootstrap budget SHA mismatch")
    elapsed = budget["prior_elapsed_wall_seconds"]
    if type(elapsed) is not float or elapsed < 0.0 or type(budget["finalization_started"]) is not bool:
        raise ValueError("bootstrap budget clock/finalization mismatch")
    _optional_reason(budget["exhausted_reason"])
    ResourceUse.from_payload(budget["charged_use"])
    for field in ("closed_stage_ids", "closed_reservation_sha256s"):
        rows = budget[field]
        if (type(rows) is not list or any(type(row) is not str or not row for row in rows)
                or rows != sorted(set(rows))):
            raise ValueError("bootstrap closed identities must be sorted unique strings")
    for sha in budget["closed_reservation_sha256s"]:
        require_sha256(sha, "closed reservation SHA")
    if len(budget["closed_reservation_sha256s"]) > len(budget["closed_stage_ids"]):
        raise ValueError("bootstrap closed reservation count mismatch")
    if type(budget["open_reservations"]) is not list:
        raise ValueError("bootstrap open reservations must be a list")
    stages = []
    for reservation in budget["open_reservations"]:
        _require_exact_schema(reservation, ("stage_id", "estimate", "reservation_sha256"), field="bootstrap reservation")
        stage = reservation["stage_id"]
        if type(stage) is not str or not stage or stage in budget["closed_stage_ids"]:
            raise ValueError("bootstrap reservation stage mismatch")
        ResourceUse.from_payload(reservation["estimate"])
        sha = require_sha256(reservation["reservation_sha256"], "bootstrap reservation SHA")
        if (sha in budget["closed_reservation_sha256s"] or sha != fingerprint_payload({"schema_version": 1,
                "plan_sha256": budget["plan_sha256"], "stage_id": stage, "estimate": reservation["estimate"]})):
            raise ValueError("bootstrap reservation identity mismatch")
        stages.append(stage)
    if stages != sorted(set(stages)):
        raise ValueError("bootstrap open stages must be sorted and unique")
    return budget


def _bootstrap_receipt(payload):
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 2:
        raise ValueError("bootstrap receipt schema mismatch")
    if type(payload["status"]) is not str or payload["status"] not in {"passed", "failed", "stopped", "interrupted"}:
        raise ValueError("bootstrap receipt status mismatch")
    for field in ("reason", "closure_reason"):
        _optional_reason(payload[field])
    if (type(payload["allowed"]) is not bool
            or payload["allowed"] != (payload["closure_reason"] is None)):
        raise ValueError("bootstrap receipt closure decision mismatch")
    status, reason, allowed = payload["status"], payload["reason"], payload["allowed"]
    if ((status == "passed" and (not allowed or reason is not None))
            or (status != "passed" and reason is None)
            or (status == "interrupted" and not allowed)
            or (not allowed and reason != payload["closure_reason"])):
        raise ValueError("bootstrap receipt status/outcome mismatch")
    for field in ("preflight_sha256", "reservation_sha256", "chain_sha256"):
        require_sha256(payload[field], field)
    for field in ("segment_index", "replay_total", "replay_consumed"):
        if type(payload[field]) is not int or payload[field] < 0:
            raise ValueError("bootstrap receipt counters must be nonnegative integers")
    previous = payload["previous_receipt_sha256"]
    if previous is not None:
        require_sha256(previous, "bootstrap previous receipt SHA")
    if ((payload["segment_index"] == 0) != (previous is None)
            or payload["replay_consumed"] > payload["replay_total"]
            or (previous is None and payload["replay_total"] != 0)
            or (payload["status"] == "passed" and payload["replay_consumed"] != payload["replay_total"])):
        raise ValueError("bootstrap receipt replay prefix counters mismatch")
    events = payload["events"]
    if type(events) is not list:
        raise ValueError("bootstrap events must be a list")
    for sha in events:
        require_sha256(sha, "bootstrap event SHA")
    if payload["replay_consumed"] > len(events):
        raise ValueError("bootstrap receipt replay event count mismatch")
    if (len(events) != len(set(events))
            or payload["chain_sha256"] != (events[-1] if events else previous or payload["preflight_sha256"])):
        raise ValueError("bootstrap receipt event chain mismatch")
    use = ResourceUse.from_payload(payload["resource_use"])
    before, after = (_bootstrap_budget(payload[field]) for field in ("budget_before", "budget_after"))
    stage = "seed_bootstrap:" + payload["preflight_sha256"] + (f":{payload['segment_index']}" if previous else "")
    opened = before["open_reservations"]
    if (len(opened) != 1 or opened[0]["stage_id"] != stage
            or opened[0]["reservation_sha256"] != payload["reservation_sha256"] or after["open_reservations"]
            or before["plan_sha256"] != after["plan_sha256"]
            or before["finalization_started"] != after["finalization_started"]
            or after["prior_elapsed_wall_seconds"] < before["prior_elapsed_wall_seconds"]
            or ResourceUse.from_payload(before["charged_use"]) + use != ResourceUse.from_payload(after["charged_use"])
            or after["closed_stage_ids"] != sorted([*before["closed_stage_ids"], stage])
            or after["closed_reservation_sha256s"] != sorted([*before["closed_reservation_sha256s"], payload["reservation_sha256"]])):
        raise ValueError("bootstrap receipt exact closure mismatch")


_TASK4_KINDS = (K.TASK_SHORTLIST, K.SHORTLIST_POLICY, K.HINDCAST_DIAGNOSTICS)


def task4_artifact_kind(payload):
    """Only the three exact Task 4 schemas retain their original encoding."""
    for kind in _TASK4_KINDS:
        if set(payload) == set(ARTIFACT_KINDS[kind].payload_fields):
            return kind
    return None


def artifact_bytes(kind, payload):
    if kind is not None and ArtifactKindV2(kind) in _TASK4_KINDS:
        from common.payload import canonical_json_bytes
        return canonical_json_bytes(payload)
    return canonical_v2_bytes(payload)


def validate_artifact(kind, raw):
    kind = ArtifactKindV2(kind)
    spec = ARTIFACT_KINDS[kind]
    if type(raw) is not bytes:
        raise TypeError("artifact accounting requires exact bytes")
    if kind is K.SOURCE:
        raw.decode("utf-8")
        return spec
    payload = strict_json_loads(raw.decode("utf-8"), context="typed artifact")
    if type(payload) is not dict or artifact_bytes(kind, payload) != raw:
        raise ValueError("artifact requires canonical JSON object bytes")
    if spec.contract is not None:
        spec.contract.from_payload(payload)
    elif spec.payload_fields is not None:
        _require_exact_schema(payload, spec.payload_fields, field=kind.value)
    if kind is K.TASK_SHORTLIST:
        from numerical_agent.evolution.task_shortlist import TaskCandidateShortlistV1
        TaskCandidateShortlistV1.from_payload(payload)
    elif kind is K.SHORTLIST_POLICY:
        from numerical_agent.evolution.task_shortlist import TaskShortlistPolicyV1
        TaskShortlistPolicyV1.from_payload(payload)
    elif kind is K.SHORTLIST_INDEX:
        from numerical_agent.run_task_local_ensemble_evolution import parse_shortlist_index
        parse_shortlist_index(payload)
    elif kind is K.HINDCAST_DIAGNOSTICS:
        from numerical_agent.run_task_local_ensemble_evolution import parse_diagnostics_payload
        parse_diagnostics_payload(payload)
    if kind is K.PROPOSER_REQUEST:
        from .proposers import primitive_proposer_request
        primitive_proposer_request(**payload)
    elif kind is K.BOOTSTRAP_PREFLIGHT:
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or payload["stage"] != "seed_bootstrap":
            raise ValueError("bootstrap preflight schema mismatch")
        for field in ("seed_supply_sha256", "protocol_sha256", "budget_plan_sha256"):
            require_sha256(payload[field], field)
        if type(payload["input_sha256s"]) is not dict or not payload["input_sha256s"]:
            raise ValueError("bootstrap inputs must be committed identities")
        for digest in payload["input_sha256s"].values():
            require_sha256(digest, "bootstrap input identity")
        ResourceUse.from_payload(payload["estimate"])
    elif kind in (K.BOOTSTRAP_ADMISSION, K.BOOTSTRAP_CLOSURE, K.BOOTSTRAP_REPLAY):
        expected = {K.BOOTSTRAP_ADMISSION: "admission", K.BOOTSTRAP_CLOSURE: "closed_forecast", K.BOOTSTRAP_REPLAY: "replayed_forecast"}
        if payload["kind"] != expected[kind] or type(payload["sequence"]) is not int or payload["sequence"] < 0:
            raise ValueError("bootstrap event identity mismatch")
        require_sha256(payload["previous_sha256"], "bootstrap previous event SHA")
        if kind is K.BOOTSTRAP_ADMISSION:
            if type(payload["arguments"]) is not list:
                raise ValueError("bootstrap dispatch arguments must be a list")
            ResourceUse.from_payload(payload["charged_use"])
        else:
            require_sha256(payload["cache_sha256"], "bootstrap cache SHA")
        if kind is K.BOOTSTRAP_REPLAY and (type(payload["replay_index"]) is not int or payload["replay_index"] < 0):
            raise ValueError("bootstrap replay counter must be a nonnegative integer")
    elif kind is K.BOOTSTRAP_RECEIPT:
        _bootstrap_receipt(payload)
    elif kind is K.KERNEL_CHECKPOINT:
        from ..kernel import EvolutionKernel
        _require_exact_schema(payload, EvolutionKernel._CHECKPOINT_FIELDS, field=kind.value)
    elif kind is K.GENERATION_STATUS:
        from .contracts import TrainMutationFeedbackV2
        legacy_fields = ("generation", "status", "active_bundle_sha256",
            "winner_genome_sha256", "proposal_attempt_sha256", "train_feedback")
        raw_row = payload["numerical_qd_step"]
        if isinstance(raw_row, dict) and set(raw_row) == set(legacy_fields):
            fields = legacy_fields
        else:
            fields = (*legacy_fields, "materialization_failures")
            if isinstance(raw_row, dict) and "mutation_prompt_population_sha256" in raw_row:
                fields = (*fields, "mutation_prompt_population_sha256")
        row = _require_exact_schema(raw_row, fields, field=kind.value)
        TrainMutationFeedbackV2.from_payload(row["train_feedback"])
        if type(row["generation"]) is not int or type(row["status"]) is not str:
            raise ValueError("generation control requires primitive status")
        failures = row.get("materialization_failures", [])
        if type(failures) is not list:
            raise ValueError("materialization failures must be a list")
        for failure in failures:
            failure = _require_exact_schema(
                failure, ("member_id", "error_type", "message"),
                field="materialization failure",
            )
            if (any(type(failure[name]) is not str or not failure[name]
                    for name in ("member_id", "error_type", "message"))
                    or len(failure["message"].encode("utf-8")) > 512):
                raise ValueError("materialization failure fields must be bounded text")
    elif kind is K.PARTIAL_RUNG:
        from .contracts import HyperbandBudgetOutcomeV2
        row = _require_exact_schema(payload["closed_partial_rung"], ("state", "manifest_sha256", "reason", "task_results",
            "unpersisted_task_results", "budget_outcome"), field=kind.value)
        HyperbandStateV2.from_payload(row["state"])
        if row["manifest_sha256"] is not None:
            require_sha256(row["manifest_sha256"], "partial manifest SHA")
        HyperbandBudgetOutcomeV2.from_payload(row["budget_outcome"])
        for item in row["task_results"]:
            _require_exact_schema(item, ("task_sha256", "result"), field="partial task reference")
            HyperbandTaskResultV2.from_payload(item["result"])
        if type(row["unpersisted_task_results"]) is not list:
            raise ValueError("unpersisted partial tasks must be a list")
        for item in row["unpersisted_task_results"]:
            validate_unpersisted_task(item)
    return spec


def billable_size(kind, raw):
    return len(raw) if validate_artifact(kind, raw).billable else 0
