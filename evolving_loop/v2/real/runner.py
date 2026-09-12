"""Root-owned, resumable orchestration for bounded real V2 evolution."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from common.payload import strict_json_loads

from ..budget import BudgetLedger, BudgetPlan, ResourceUse
from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from ..store import V2RunStore, write_once_json
from .contracts import (
    PROFILE_SCHEDULES,
    RealEvolutionCheckpointV2,
    RealEvolutionManifestV2,
    RealRunResultV2,
    RealStageRecordV2,
)


_STAGES = ("p2", "p3", "p4", "p5")
_PHASES = {
    "p2": ("P2_RUNNING", "P2_SEALED"),
    "p3": ("P3_RUNNING", "P3_SEALED"),
    "p4": ("P4_RUNNING", "P4_SEALED"),
    "p5": ("P5_RUNNING", "P5_SEALED"),
}


class RealRunnerError(ValueError):
    """Raised when a root real-run artifact cannot be safely adopted."""


def _json_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if hasattr(value, "to_payload"):
        value = value.to_payload()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    # Canonical serialization is the shared strict JSON validator.
    raw = canonical_v2_bytes(value)
    copied = strict_json_loads(raw.decode("utf-8"), context=field)
    assert isinstance(copied, dict)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class SealedStageV2:
    """Validated child boundary accepted by the root state machine."""

    completion_sha256: str
    handoff_payload: Mapping[str, object]
    summary: Mapping[str, object]
    public_test_accessed: bool

    def __post_init__(self) -> None:
        require_sha256(self.completion_sha256, "completion_sha256")
        object.__setattr__(
            self, "handoff_payload", _json_mapping(self.handoff_payload, field="handoff_payload")
        )
        object.__setattr__(self, "summary", _json_mapping(self.summary, field="summary"))
        if type(self.public_test_accessed) is not bool:
            raise ValueError("public_test_accessed must be a boolean")


@dataclass(frozen=True, slots=True)
class RealStageContextV2:
    """Narrow Host capability supplied to one child bridge."""

    stage: str
    output_dir: Path
    grant_seconds: int
    manifest: RealEvolutionManifestV2
    manifest_sha256: str
    model_binding_sha256: str
    handoffs: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("stage must be p2, p3, p4, or p5")
        if type(self.grant_seconds) is not int or self.grant_seconds <= 0:
            raise ValueError("grant_seconds must be positive")
        require_sha256(self.manifest_sha256, "manifest_sha256")
        require_sha256(self.model_binding_sha256, "model_binding_sha256")
        for stage, digest in self.handoffs.items():
            if stage not in _STAGES:
                raise ValueError("handoffs must use real stage names")
            require_sha256(digest, f"handoffs.{stage}")
        object.__setattr__(self, "handoffs", MappingProxyType(dict(self.handoffs)))


@dataclass(frozen=True, slots=True)
class RealStagePorts:
    """Typed seams for the four existing project runners and their seals."""

    run_p2: Callable[[RealStageContextV2], object]
    seal_p2: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p3: Callable[[RealStageContextV2], object]
    seal_p3: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p4: Callable[[RealStageContextV2], object]
    seal_p4: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p5: Callable[[RealStageContextV2], object]
    seal_p5: Callable[[RealStageContextV2, object], SealedStageV2]
    begin_finalization: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_p2", "seal_p2", "run_p3", "seal_p3", "run_p4", "seal_p4", "run_p5", "seal_p5",
        ):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")
        if self.begin_finalization is not None and not callable(self.begin_finalization):
            raise ValueError("begin_finalization must be callable or None")


def _read_canonical(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise RealRunnerError(f"missing or invalid canonical artifact: {path}") from error
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise RealRunnerError(f"artifact must be canonical JSON: {path}")
    return value


def _plan(manifest: RealEvolutionManifestV2) -> BudgetPlan:
    schedule = PROFILE_SCHEDULES[manifest.profile]
    return BudgetPlan(
        schedule.total_seconds,
        float(schedule.finalization_reserve_seconds / schedule.total_seconds),
        ResourceUse(wall_seconds=float(schedule.total_seconds)),
    )


def _root_manifest(manifest: RealEvolutionManifestV2, plan: BudgetPlan) -> dict[str, object]:
    return {
        "system": "evolution_v2",
        "schema_version": 1,
        "kind": "real_root_run",
        "real_manifest": manifest.to_payload(),
        "real_manifest_sha256": manifest.fingerprint(),
        "model_binding_sha256": manifest.model.fingerprint(),
        "budget_plan_sha256": plan.fingerprint(),
    }


def _load_state(
    root: Path, manifest: RealEvolutionManifestV2, plan: BudgetPlan, monotonic: Callable[[], float]
) -> tuple[V2RunStore, BudgetLedger, RealEvolutionCheckpointV2 | None]:
    store = V2RunStore.create(root)
    expected_manifest = _root_manifest(manifest, plan)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if _read_canonical(manifest_path) != expected_manifest:
            raise RealRunnerError("root manifest does not match the supplied real manifest")
    else:
        store.write_run_manifest(expected_manifest)

    budget_path = root / "budget_plan.json"
    if budget_path.exists():
        if _read_canonical(budget_path) != plan.to_payload():
            raise RealRunnerError("root budget plan does not match the supplied profile")
    else:
        store.write_budget_plan(plan.to_payload())

    checkpoint_path = root / "checkpoint.json"
    if not checkpoint_path.exists():
        return store, BudgetLedger(plan, monotonic=monotonic), None
    checkpoint = RealEvolutionCheckpointV2.from_payload(_read_canonical(checkpoint_path))
    try:
        ledger = BudgetLedger.resume(plan, checkpoint.budget_checkpoint, monotonic=monotonic)
    except ValueError as error:
        raise RealRunnerError(f"invalid root budget checkpoint: {error}") from error
    return store, ledger, checkpoint


def _checkpoint(
    store: V2RunStore,
    ledger: BudgetLedger,
    *,
    phase: str,
    records: list[RealStageRecordV2],
    active_stage: str | None,
    carry_seconds: int,
    handoff_sha256s: Mapping[str, str],
    completion_sha256: str | None,
) -> RealEvolutionCheckpointV2:
    checkpoint = RealEvolutionCheckpointV2(
        phase,
        tuple(records),
        active_stage,
        carry_seconds,
        ledger.checkpoint(),
        dict(handoff_sha256s),
        completion_sha256,
    )
    store.write_checkpoint(checkpoint.to_payload())
    return checkpoint


def _result(
    status: str,
    manifest: RealEvolutionManifestV2,
    records: list[RealStageRecordV2],
    completion_sha256: str | None = None,
) -> RealRunResultV2:
    return RealRunResultV2(
        status, manifest.fingerprint(), manifest.model.fingerprint(), tuple(records), completion_sha256, False
    )


def _previous_progress_sha(root: Path) -> str | None:
    path = root / "progress.jsonl"
    if not path.exists():
        return None
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        return None
    raw = lines[-1]
    value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise RealRunnerError("progress rows must be canonical JSON")
    return fingerprint_payload(value)


def _append_progress_once(store: V2RunStore, root: Path, row: Mapping[str, object]) -> None:
    """Append one transition row, tolerating only an interrupted identical retry."""
    path = root / "progress.jsonl"
    expected = canonical_v2_bytes(row)
    if path.exists():
        for raw in path.read_bytes().splitlines(keepends=True):
            value = strict_json_loads(raw.decode("utf-8"), context=str(path))
            if type(value) is not dict or canonical_v2_bytes(value) != raw:
                raise RealRunnerError("progress rows must be canonical JSON")
            if raw == expected:
                return
    store.append_progress(row)


def _stage_status(sealed: SealedStageV2) -> str:
    value = sealed.summary.get("status", "complete")
    if value not in {"complete", "incomplete", "failed"}:
        raise RealRunnerError("sealed stage summary status must be complete, incomplete, or failed")
    return value


def _stage_output(
    root: Path,
    stage: str,
    completion_sha256: str,
    sealed_public_test_accessed: bool,
) -> None:
    completion = _read_canonical(root / stage / "evaluation_complete.json")
    if fingerprint_payload(completion) != completion_sha256:
        raise RealRunnerError(f"{stage} completion digest does not match its sealed boundary")
    completion_public = completion.get("public_test_accessed")
    if type(completion_public) is not bool:
        raise RealRunnerError(f"{stage} completion public access evidence is missing")
    if completion_public or sealed_public_test_accessed or completion_public != sealed_public_test_accessed:
        raise RealRunnerError(f"{stage} completion public access evidence must be false and match its seal")


def _sealed_handoff_payload(stage: str, sealed: SealedStageV2) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": stage,
        "completion_sha256": sealed.completion_sha256,
        "handoff_payload": dict(sealed.handoff_payload),
        "summary": dict(sealed.summary),
        "public_test_accessed": sealed.public_test_accessed,
    }


def _validate_sealed_records(
    root: Path, records: list[RealStageRecordV2], handoffs: Mapping[str, str]
) -> bool:
    if tuple(record.stage for record in records) != _STAGES[: len(records)]:
        raise RealRunnerError("root stage records are not a sealed prefix")
    public_accessed = False
    for record in records:
        if record.status != "complete" or record.completion_sha256 is None:
            raise RealRunnerError("only complete sealed records can be revalidated")
        claimed = handoffs.get(record.stage)
        if claimed is None:
            raise RealRunnerError(f"missing root handoff for {record.stage}")
        handoff = _read_canonical(root / "handoffs" / f"{record.stage}.json")
        if fingerprint_payload(handoff) != claimed:
            raise RealRunnerError(f"{record.stage} root handoff digest mismatch")
        if handoff.get("stage") != record.stage or handoff.get("completion_sha256") != record.completion_sha256:
            raise RealRunnerError(f"{record.stage} root handoff does not bind its completion")
        public = handoff.get("public_test_accessed")
        if type(public) is not bool:
            raise RealRunnerError(f"{record.stage} public access evidence is missing")
        _stage_output(root, record.stage, record.completion_sha256, public)
        public_accessed = public_accessed or public
    return public_accessed


def _persist_run_result(root: Path, stage: str, result: object) -> None:
    try:
        payload = _json_mapping(result, field="stage result")
    except (TypeError, ValueError):
        return
    write_once_json(root / stage / "root_run_result.json", payload)


def _read_run_result(root: Path, stage: str) -> object | None:
    path = root / stage / "root_run_result.json"
    return _read_canonical(path) if path.exists() else None


def _context(
    root: Path,
    stage: str,
    grant_seconds: int,
    manifest: RealEvolutionManifestV2,
    handoffs: Mapping[str, str],
) -> RealStageContextV2:
    child_root = root / stage
    child_root.mkdir(parents=True, exist_ok=True)
    return RealStageContextV2(
        stage,
        child_root,
        grant_seconds,
        manifest,
        manifest.fingerprint(),
        manifest.model.fingerprint(),
        handoffs,
    )


def _grant(
    manifest: RealEvolutionManifestV2, ledger: BudgetLedger, stage: str, carry_seconds: int
) -> int:
    schedule = PROFILE_SCHEDULES[manifest.profile]
    index = _STAGES.index(stage)
    base = schedule.allocations[stage]
    later_base = sum(schedule.allocations[name] for name in _STAGES[index + 1 :])
    search_remaining = (
        schedule.total_seconds - schedule.finalization_reserve_seconds - ledger.charged_use.wall_seconds
    )
    return int(min(base + carry_seconds, search_remaining - later_base))


def _terminal_active_failure(
    *,
    root: Path,
    store: V2RunStore,
    ledger: BudgetLedger,
    reservation: object,
    stage: str,
    grant_seconds: int,
    records: list[RealStageRecordV2],
    handoffs: Mapping[str, str],
) -> None:
    """Conservatively close a returned-but-unverifiable child as failed."""
    ledger.close_stage(reservation, ResourceUse(wall_seconds=float(grant_seconds)))
    records.append(
        RealStageRecordV2(
            stage,
            grant_seconds,
            grant_seconds,
            "failed",
            None,
            _previous_progress_sha(root),
        )
    )
    _checkpoint(
        store,
        ledger,
        phase="FAILED",
        records=records,
        active_stage=None,
        carry_seconds=0,
        handoff_sha256s=handoffs,
        completion_sha256=None,
    )


def _summary(
    manifest: RealEvolutionManifestV2,
    plan: BudgetPlan,
    records: list[RealStageRecordV2],
    handoffs: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": manifest.fingerprint(),
        "model_binding_sha256": manifest.model.fingerprint(),
        "budget_plan_sha256": plan.fingerprint(),
        "stage_records": [record.to_payload() for record in records],
        "handoff_sha256s": dict(handoffs),
        "public_test_accessed": False,
    }


def run_real_evolution(
    output_dir: str | Path,
    manifest: RealEvolutionManifestV2,
    ports: RealStagePorts,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> RealRunResultV2:
    """Run or safely resume one immutable real P2→P5 root epoch."""
    if not isinstance(manifest, RealEvolutionManifestV2):
        raise TypeError("manifest must be RealEvolutionManifestV2")
    if not isinstance(ports, RealStagePorts):
        raise TypeError("ports must be RealStagePorts")
    if not callable(monotonic):
        raise TypeError("monotonic must be callable")

    root = Path(output_dir)
    plan = _plan(manifest)
    store, ledger, checkpoint = _load_state(root, manifest, plan, monotonic)
    records = list(checkpoint.stage_records) if checkpoint is not None else []
    handoffs = dict(checkpoint.handoff_sha256s) if checkpoint is not None else {}
    carry = checkpoint.carry_seconds if checkpoint is not None else 0

    if checkpoint is not None and checkpoint.phase == "COMPLETE":
        if checkpoint.completion_sha256 is None or checkpoint.active_stage is not None:
            raise RealRunnerError("completed root checkpoint is malformed")
        public = _validate_sealed_records(root, records, handoffs)
        if public or len(records) != len(_STAGES) or not ledger.finalization_started:
            raise RealRunnerError("completed root violates its sealed finalization boundary")
        result = RealRunResultV2.from_payload(_read_canonical(root / "evaluation_complete.json"))
        expected_result = _result("complete", manifest, records, checkpoint.completion_sha256)
        if result != expected_result:
            raise RealRunnerError("completed root result does not match checkpoint")
        summary = _read_canonical(root / "result_summary.json")
        if summary != _summary(manifest, plan, records, handoffs):
            raise RealRunnerError("completed root summary does not match its checkpointed bindings")
        if fingerprint_payload(summary) != checkpoint.completion_sha256:
            raise RealRunnerError("completed root summary digest does not match checkpoint")
        return result

    if checkpoint is not None and checkpoint.phase in {"INCOMPLETE", "FAILED"} and checkpoint.active_stage is None:
        return _result("incomplete" if checkpoint.phase == "INCOMPLETE" else "failed", manifest, records)

    if checkpoint is not None and checkpoint.active_stage is not None:
        stage = checkpoint.active_stage
        open_rows = checkpoint.budget_checkpoint.get("open_reservations", [])
        active = next((row for row in open_rows if row.get("stage_id") == stage), None)
        if not isinstance(active, Mapping):
            raise RealRunnerError("active root stage has no budget reservation")
        estimate = ResourceUse.from_payload(active["estimate"])
        if estimate.wall_seconds <= 0.0 or not estimate.wall_seconds.is_integer():
            raise RealRunnerError("active root stage grant must be a positive whole second")
        grant_seconds = int(estimate.wall_seconds)
        if stage != _STAGES[len(records)]:
            raise RealRunnerError("active root stage does not follow sealed records")
        context = _context(root, stage, grant_seconds, manifest, handoffs)
        prior_result = _read_run_result(root, stage)
        if prior_result is None:
            ledger.close_stage(active["reservation_sha256"], estimate)
            records.append(RealStageRecordV2(stage, grant_seconds, grant_seconds, "incomplete", None, _previous_progress_sha(root)))
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None,
                        carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        seal = getattr(ports, f"seal_{stage}")
        try:
            sealed = seal(context, prior_result)
            if not isinstance(sealed, SealedStageV2):
                raise RealRunnerError("stage seal must return SealedStageV2")
            _stage_status(sealed)
            _stage_output(root, stage, sealed.completion_sha256, sealed.public_test_accessed)
        except Exception:
            _terminal_active_failure(
                root=root, store=store, ledger=ledger, reservation=active["reservation_sha256"],
                stage=stage, grant_seconds=grant_seconds, records=records, handoffs=handoffs,
            )
            raise
        closure = ledger.close_stage(active["reservation_sha256"], estimate)
        if not closure.allowed:
            records.append(RealStageRecordV2(stage, grant_seconds, grant_seconds, "failed", sealed.completion_sha256, _previous_progress_sha(root)))
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None,
                        carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)
        handoff_payload = _sealed_handoff_payload(stage, sealed)
        handoff_sha = fingerprint_payload(handoff_payload)
        write_once_json(root / "handoffs" / f"{stage}.json", handoff_payload)
        record = RealStageRecordV2(stage, grant_seconds, grant_seconds, _stage_status(sealed), sealed.completion_sha256, _previous_progress_sha(root))
        records.append(record)
        handoffs[stage] = handoff_sha
        row = {"schema_version": 1, "stage": stage, "stage_record_sha256": record.fingerprint(),
               "completion_sha256": sealed.completion_sha256, "handoff_sha256": handoff_sha, "status": record.status}
        _append_progress_once(store, root, row)
        if record.status != "complete":
            _checkpoint(store, ledger, phase="INCOMPLETE" if record.status == "incomplete" else "FAILED", records=records,
                        active_stage=None, carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result(record.status, manifest, records)
        carry = 0
        _checkpoint(store, ledger, phase=_PHASES[stage][1], records=records, active_stage=None,
                    carry_seconds=carry, handoff_sha256s=handoffs, completion_sha256=None)

    if records:
        public = _validate_sealed_records(root, records, handoffs)
        if public:
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)

    for stage in _STAGES[len(records) :]:
        grant_seconds = _grant(manifest, ledger, stage, carry)
        if grant_seconds <= 0:
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None, carry_seconds=0,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        permit = ledger.reserve_stage(stage, ResourceUse(wall_seconds=float(grant_seconds)))
        if not permit.allowed:
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None, carry_seconds=0,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        _checkpoint(store, ledger, phase=_PHASES[stage][0], records=records, active_stage=stage,
                    carry_seconds=carry, handoff_sha256s=handoffs, completion_sha256=None)
        context = _context(root, stage, grant_seconds, manifest, handoffs)
        start = float(monotonic())
        result = getattr(ports, f"run_{stage}")(context)
        end = float(monotonic())
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise RealRunnerError("stage monotonic interval is invalid")
        charged_seconds = math.ceil(end - start)
        _persist_run_result(root, stage, result)
        try:
            sealed = getattr(ports, f"seal_{stage}")(context, result)
            if not isinstance(sealed, SealedStageV2):
                raise RealRunnerError("stage seal must return SealedStageV2")
            stage_status = _stage_status(sealed)
            _stage_output(root, stage, sealed.completion_sha256, sealed.public_test_accessed)
            closure = ledger.close_stage(permit, ResourceUse(wall_seconds=float(charged_seconds)))
        except Exception:
            _terminal_active_failure(
                root=root, store=store, ledger=ledger, reservation=permit,
                stage=stage, grant_seconds=grant_seconds, records=records, handoffs=handoffs,
            )
            raise
        if not closure.allowed:
            stage_status = "failed"
        handoff_payload = _sealed_handoff_payload(stage, sealed)
        handoff_sha = fingerprint_payload(handoff_payload)
        write_once_json(root / "handoffs" / f"{stage}.json", handoff_payload)
        record = RealStageRecordV2(stage, grant_seconds, charged_seconds, stage_status, sealed.completion_sha256, _previous_progress_sha(root))
        records.append(record)
        handoffs[stage] = handoff_sha
        _append_progress_once(
            store,
            root,
            {"schema_version": 1, "stage": stage, "stage_record_sha256": record.fingerprint(),
             "completion_sha256": sealed.completion_sha256, "handoff_sha256": handoff_sha, "status": stage_status},
        )
        carry = max(0, grant_seconds - charged_seconds)
        phase = _PHASES[stage][1] if stage_status == "complete" else ("INCOMPLETE" if stage_status == "incomplete" else "FAILED")
        _checkpoint(store, ledger, phase=phase, records=records, active_stage=None, carry_seconds=carry,
                    handoff_sha256s=handoffs, completion_sha256=None)
        if stage_status != "complete":
            return _result(stage_status, manifest, records)
        if sealed.public_test_accessed:
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)

    if len(records) != len(_STAGES):
        raise RealRunnerError("root reached finalization without four sealed stages")
    ledger.begin_finalization()
    _checkpoint(store, ledger, phase="FINALIZING", records=records, active_stage=None, carry_seconds=carry,
                handoff_sha256s=handoffs, completion_sha256=None)
    if ports.begin_finalization is not None:
        ports.begin_finalization()
    public = _validate_sealed_records(root, records, handoffs)
    if public:
        _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                    handoff_sha256s=handoffs, completion_sha256=None)
        return _result("failed", manifest, records)
    summary = _summary(manifest, plan, records, handoffs)
    completion_sha = fingerprint_payload(summary)
    write_once_json(root / "result_summary.json", summary)
    result = _result("complete", manifest, records, completion_sha)
    store.write_completion(result.to_payload())
    _checkpoint(store, ledger, phase="COMPLETE", records=records, active_stage=None, carry_seconds=carry,
                handoff_sha256s=handoffs, completion_sha256=completion_sha)
    return result


__all__ = [
    "RealRunnerError",
    "RealStageContextV2",
    "RealStagePorts",
    "SealedStageV2",
    "run_real_evolution",
]
