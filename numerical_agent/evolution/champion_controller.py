"""Build evolution plus the durable 64/16/20 Champion lifecycle."""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Callable, Literal, NoReturn, cast

from common.data import Task
from common.payload import canonical_json_bytes, strict_json_loads

from .champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    _parse_fitted_policy,
    champion_fingerprint,
    parse_champion_recipe,
    parse_champion_release,
)
from .champion_evidence import (
    _InvalidAttemptAggregate,
    ChampionComparison,
    ChampionEvidenceError,
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionScore,
    ChampionTaskRow,
    MorphologyAggregate,
    ProposerEvidence,
    _ProposerComparison,
    _assert_sanitized,
    compare_champion,
    sanitize_build_evidence,
    score_policy,
)
from .champion_proposal import ChampionProposalError, expand_recipe
from .champion_runtime import execute_champion
from .numerical_selector import CandidateDiagnostics


_FORMAL_SIZES = (64, 16, (8, 32, 64))
_SMOKE_SIZES = (8, 2, (4, 8))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_STAGE = re.compile(r"build_generation_([1-9][0-9]*)\Z")
_PROFILE_FEATURES = frozenset({
    "history_length",
    "horizon",
    "horizon_ratio",
    "zero_fraction",
    "trend_strength",
    "periodicity_strength",
    "periodicity_confidence",
    "outlier_fraction",
    "noise_relative_scale",
    "stationarity_score",
    "recent_regime_confidence",
    "intermittency_adi",
    "intermittency_cv2",
})


class ChampionControllerError(ValueError):
    """Build evolution input or evidence violates its fixed authority."""


class ChampionLifecycleError(ChampionControllerError):
    """Formal Train/Dev lifecycle state violates its fixed authority."""


def _fail(message: str) -> NoReturn:
    raise ChampionControllerError(message)


def _canonical_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _lifecycle_fail(message: str) -> NoReturn:
    raise ChampionLifecycleError(message)


@dataclass(frozen=True)
class TrainPartitions:
    """Exact entity-disjoint internal partitions of the formal Train set."""

    build: tuple[Task, ...]
    calibration: tuple[Task, ...]


def partition_train_tasks(
    tasks: tuple[Task, ...] | list[Task],
    *,
    build_size: int = 64,
    calibration_size: int = 16,
    seed: int,
) -> TrainPartitions:
    """Choose an exact SHA-seeded entity-group subset for Calibration."""
    if type(build_size) is not int or type(calibration_size) is not int:
        _lifecycle_fail("partition sizes must be exact integers")
    if build_size <= 0 or calibration_size <= 0:
        _lifecycle_fail("partition sizes must be positive")
    if type(seed) is not int:
        _lifecycle_fail("partition seed must be an exact integer")
    if type(tasks) not in {tuple, list}:
        _lifecycle_fail("Train tasks must be an exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], tasks))
    if len(snapshot) != build_size + calibration_size:
        _lifecycle_fail("Train tasks do not match the exact partition sizes")

    task_ids: dict[str, str] = {}
    entity_ids: dict[str, str] = {}
    groups: dict[str, list[Task]] = {}
    for raw_task in snapshot:
        if type(raw_task) is not Task:
            _lifecycle_fail("Train partitions require exact Task values")
        task = cast(Task, raw_task)
        if type(task.task_id) is not str or not task.task_id:
            _lifecycle_fail("Train task IDs must be nonempty exact strings")
        canonical_task_id = _canonical_identity(task.task_id)
        if canonical_task_id in task_ids:
            _lifecycle_fail("Train tasks contain duplicate task IDs")
        task_ids[canonical_task_id] = task.task_id
        if (
            type(task.entity_name) is not str
            or not task.entity_name.strip()
            or _canonical_identity(task.entity_name) == "unknown"
        ):
            _lifecycle_fail("Train tasks contain an unknown entity")
        canonical_entity = _canonical_identity(task.entity_name)
        prior_entity = entity_ids.setdefault(canonical_entity, task.entity_name)
        if prior_entity != task.entity_name:
            _lifecycle_fail("Train tasks contain duplicate entity identities")
        groups.setdefault(task.entity_name, []).append(task)

    ordered_groups = tuple(
        sorted(
            (
                (
                    hashlib.sha256(
                        f"{seed}{entity_name}".encode("utf-8")
                    ).hexdigest(),
                    entity_name,
                    tuple(sorted(group, key=lambda item: item.task_id)),
                )
                for entity_name, group in groups.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
    )
    subsets: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, _, group) in enumerate(ordered_groups):
        next_subsets = dict(subsets)
        for size, selected in subsets.items():
            candidate_size = size + len(group)
            if candidate_size <= calibration_size and candidate_size not in next_subsets:
                next_subsets[candidate_size] = selected + (index,)
        subsets = next_subsets
    selected_groups = subsets.get(calibration_size)
    if selected_groups is None:
        _lifecycle_fail(
            "Train entity groups cannot form the exact entity-disjoint partition"
        )
    calibration_indexes = set(selected_groups)
    build = tuple(
        task
        for index, (_, _, group) in enumerate(ordered_groups)
        if index not in calibration_indexes
        for task in group
    )
    calibration = tuple(
        task
        for index, (_, _, group) in enumerate(ordered_groups)
        if index in calibration_indexes
        for task in group
    )
    if len(build) != build_size or len(calibration) != calibration_size:
        _lifecycle_fail("entity-disjoint partition produced the wrong exact sizes")
    if {task.entity_name for task in build} & {
        task.entity_name for task in calibration
    }:
        _lifecycle_fail("entity-disjoint partition contains entity overlap")
    return TrainPartitions(build=build, calibration=calibration)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _lifecycle_fail(f"{label} must be a canonical SHA-256 digest")
    return cast(str, value)


def _validated_hash_inventory(
    values: object, label: str
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or not values:
        _lifecycle_fail(f"{label} must be a nonempty exact tuple")
    inventory = cast(tuple[object, ...], values)
    parsed: list[tuple[str, str]] = []
    canonical_names: set[str] = set()
    for raw_pair in inventory:
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact name/hash pairs")
        name, digest = cast(tuple[object, object], raw_pair)
        if type(name) is not str or not name or not name.isidentifier():
            _lifecycle_fail(f"{label} names must be public identifiers")
        canonical_name = _canonical_identity(name)
        if canonical_name in canonical_names:
            _lifecycle_fail(f"{label} contains a duplicate identity")
        canonical_names.add(canonical_name)
        parsed.append((name, _require_sha256(digest, f"{label} hash")))
    result = tuple(parsed)
    if result != tuple(sorted(result)):
        _lifecycle_fail(f"{label} must use canonical sorted order")
    return result


def _validated_membership(
    values: object,
    label: str,
    *,
    expected_size: int,
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or len(values) != expected_size:
        _lifecycle_fail(f"{label} must contain exactly {expected_size} tasks")
    membership = cast(tuple[object, ...], values)
    parsed: list[tuple[str, str]] = []
    task_ids: set[str] = set()
    entity_aliases: dict[str, str] = {}
    for raw_pair in membership:
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact task/entity pairs")
        task_id, entity_name = cast(tuple[object, object], raw_pair)
        if type(task_id) is not str or not task_id:
            _lifecycle_fail(f"{label} contains an invalid task ID")
        canonical_task = _canonical_identity(task_id)
        if canonical_task in task_ids:
            _lifecycle_fail(f"{label} contains duplicate task IDs")
        task_ids.add(canonical_task)
        if (
            type(entity_name) is not str
            or not entity_name.strip()
            or _canonical_identity(entity_name) == "unknown"
        ):
            _lifecycle_fail(f"{label} contains an unknown entity")
        canonical_entity = _canonical_identity(entity_name)
        prior = entity_aliases.setdefault(canonical_entity, entity_name)
        if prior != entity_name:
            _lifecycle_fail(f"{label} contains duplicate entity identities")
        parsed.append((task_id, entity_name))
    return tuple(parsed)


@dataclass(frozen=True)
class ChampionRunManifest:
    """Complete pre-execution identity of one formal 64/16/20 run."""

    schema_version: int
    partition_seed: int
    source_hashes: tuple[tuple[str, str], ...]
    train_tasks: tuple[tuple[str, str], ...]
    dev_tasks: tuple[tuple[str, str], ...]
    split_manifest_fingerprint: str
    build_tasks: tuple[tuple[str, str], ...]
    calibration_tasks: tuple[tuple[str, str], ...]
    dictionary_hashes: tuple[tuple[str, str], ...]
    forecast_store_fingerprint: str
    metric_policy_fingerprint: str
    proposal_model: str
    proposal_config_fingerprint: str
    schedule_fingerprint: str
    numeric_grid_fingerprint: str
    candidate_minimum_gain: float
    research_target_gain: float
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("run manifest schema_version must be exactly one")
        if type(self.partition_seed) is not int:
            _lifecycle_fail("run manifest partition seed must be an exact integer")
        _validated_hash_inventory(self.source_hashes, "source hashes")
        _validated_hash_inventory(self.dictionary_hashes, "Dictionary hashes")
        train = _validated_membership(
            self.train_tasks, "Train membership", expected_size=80
        )
        dev = _validated_membership(self.dev_tasks, "Dev membership", expected_size=20)
        build = _validated_membership(
            self.build_tasks, "Build membership", expected_size=64
        )
        calibration = _validated_membership(
            self.calibration_tasks,
            "Calibration membership",
            expected_size=16,
        )
        train_map = dict(train)
        if set(build) & set(calibration):
            _lifecycle_fail("Build and Calibration membership overlap")
        if dict(build) | dict(calibration) != train_map:
            _lifecycle_fail("Build and Calibration must exactly partition Train")
        if {entity for _, entity in build} & {
            entity for _, entity in calibration
        }:
            _lifecycle_fail("Build and Calibration must be entity-disjoint")
        if {_canonical_identity(task_id) for task_id, _ in train} & {
            _canonical_identity(task_id) for task_id, _ in dev
        }:
            _lifecycle_fail("Train and Dev contain duplicate task IDs")
        if {_canonical_identity(entity) for _, entity in train} & {
            _canonical_identity(entity) for _, entity in dev
        }:
            _lifecycle_fail("Train and Dev contain duplicate entities")
        for name in (
            "split_manifest_fingerprint",
            "forecast_store_fingerprint",
            "metric_policy_fingerprint",
            "proposal_config_fingerprint",
            "schedule_fingerprint",
            "numeric_grid_fingerprint",
            "runtime_fingerprint",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.proposal_model) is not str or not self.proposal_model.strip():
            _lifecycle_fail("proposal model identity must be a nonempty exact string")
        for name in ("candidate_minimum_gain", "research_target_gain"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                _lifecycle_fail(f"{name} must be a finite nonnegative float")
        if self.research_target_gain < self.candidate_minimum_gain:
            _lifecycle_fail("research target cannot be below candidate threshold")

    def _identity_payload(self) -> dict[str, object]:
        return {
            entry.name: getattr(self, entry.name)
            for entry in fields(self)
        }

    @property
    def input_fingerprint(self) -> str:
        return champion_fingerprint(self._identity_payload())

    def to_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["input_fingerprint"] = self.input_fingerprint
        return payload


def _validated_feedback_list(
    value: object,
    expected_fields: frozenset[str],
    label: str,
) -> tuple[dict[str, object], ...]:
    if type(value) is not list:
        _lifecycle_fail(f"checkpoint {label} feedback must be an exact JSON array")
    result: list[dict[str, object]] = []
    for raw_item in cast(list[object], value):
        if type(raw_item) is not dict or set(raw_item) != expected_fields:
            _lifecycle_fail(f"checkpoint {label} feedback has a malformed schema")
        result.append(cast(dict[str, object], raw_item))
    return tuple(result)


def _parse_sanitized_feedback(payload: dict[str, object]) -> ProposerEvidence:
    if set(payload) != {
        "label",
        "independent_generalization_claim",
        "morphology",
        "comparisons",
        "invalid_attempts",
    }:
        _lifecycle_fail("checkpoint feedback has a malformed schema")
    if (
        payload["label"] != "adaptive_train_build_diagnostic"
        or payload["independent_generalization_claim"] is not False
    ):
        _lifecycle_fail("checkpoint feedback violates Build-only authority")
    morphology_payloads = _validated_feedback_list(
        payload["morphology"],
        frozenset(MorphologyAggregate.__dataclass_fields__),
        "morphology",
    )
    comparison_payloads = _validated_feedback_list(
        payload["comparisons"],
        frozenset(_ProposerComparison.__dataclass_fields__),
        "comparison",
    )
    invalid_payloads = _validated_feedback_list(
        payload["invalid_attempts"],
        frozenset(_InvalidAttemptAggregate.__dataclass_fields__),
        "invalid-attempt",
    )
    try:
        morphology = tuple(
            MorphologyAggregate(**item)  # type: ignore[arg-type]
            for item in morphology_payloads
        )
        comparisons: list[_ProposerComparison] = []
        count_fields = {
            "smae_clipped_count",
            "srmse_clipped_count",
            "improved_folds",
            "total_folds",
            "wins",
            "ties",
            "losses",
        }
        raw_fields = {
            "p90_smae_raw",
            "p95_smae_raw",
            "p90_srmse_raw",
            "p95_srmse_raw",
        }
        for item in comparison_payloads:
            if (
                type(item["candidate_name"]) is not str
                or not cast(str, item["candidate_name"]).isidentifier()
            ):
                _lifecycle_fail("checkpoint comparison candidate is malformed")
            for name, value in item.items():
                if name == "candidate_name":
                    continue
                if name in count_fields:
                    if type(value) is not int or value < 0:
                        _lifecycle_fail("checkpoint comparison count is malformed")
                elif name in raw_fields and value == "positive_infinity":
                    continue
                elif type(value) is not float or not math.isfinite(value):
                    _lifecycle_fail("checkpoint comparison metric is malformed")
            comparisons.append(
                _ProposerComparison(**item)  # type: ignore[arg-type]
            )
        invalid_attempts = tuple(
            _InvalidAttemptAggregate(**item)  # type: ignore[arg-type]
            for item in invalid_payloads
        )
        evidence = ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=morphology,
            comparisons=tuple(comparisons),
            invalid_attempts=invalid_attempts,
        )
        ProposerEvidence.__post_init__(evidence)
        _assert_sanitized(payload)
    except ChampionLifecycleError:
        raise
    except Exception as error:
        raise ChampionLifecycleError("checkpoint feedback is malformed") from error
    champion_fingerprint(payload)
    return evidence


@dataclass(frozen=True)
class ChampionCheckpoint:
    """Immutable post-Build state from which Calibration may run once."""

    schema_version: int
    input_fingerprint: str
    completed_stage: str
    active_parent: ChampionRelease
    proposal_archive: tuple[FittedChampionPolicy, ...]
    sanitized_feedback: ProposerEvidence

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("checkpoint schema_version must be exactly one")
        _require_sha256(self.input_fingerprint, "checkpoint input fingerprint")
        if type(self.completed_stage) is not str or _CHECKPOINT_STAGE.fullmatch(
            self.completed_stage
        ) is None:
            _lifecycle_fail("checkpoint completed stage is invalid")
        if type(self.active_parent) is not ChampionRelease:
            _lifecycle_fail("checkpoint active Parent must be an exact release")
        _validated_parent(self.active_parent)
        if (
            type(self.proposal_archive) is not tuple
            or len(self.proposal_archive) > 3
        ):
            _lifecycle_fail("checkpoint proposal archive cannot exceed three policies")
        policy_ids: list[str] = []
        for policy in self.proposal_archive:
            if type(policy) is not FittedChampionPolicy:
                _lifecycle_fail("checkpoint archive contains an invalid policy")
            FittedChampionPolicy.__post_init__(policy)
            policy_ids.append(champion_fingerprint(policy))
        if len(policy_ids) != len(set(policy_ids)):
            _lifecycle_fail("checkpoint archive contains duplicate policies")
        if type(self.sanitized_feedback) is not ProposerEvidence:
            _lifecycle_fail("checkpoint feedback must be exact sanitized evidence")
        try:
            ProposerEvidence.__post_init__(self.sanitized_feedback)
            _assert_sanitized(self.sanitized_feedback.to_payload())
        except Exception as error:
            raise ChampionLifecycleError("checkpoint feedback is malformed") from error

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_fingerprint": self.input_fingerprint,
            "completed_stage": self.completed_stage,
            "active_parent": self.active_parent.to_payload(),
            "proposal_archive": [
                policy.to_payload() for policy in self.proposal_archive
            ],
            "sanitized_feedback": self.sanitized_feedback.to_payload(),
        }


@dataclass(frozen=True)
class ChampionStageCandidateReport:
    policy_sha256: str
    evidence_fingerprint: str
    comparison: ChampionComparison | None
    invalid_reason: Literal["unscorable_child"] | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.policy_sha256, "reported policy fingerprint")
        _require_sha256(self.evidence_fingerprint, "reported evidence fingerprint")
        if self.comparison is None:
            if self.invalid_reason != "unscorable_child":
                _lifecycle_fail("invalid stage evidence requires a typed reason")
        else:
            if type(self.comparison) is not ChampionComparison:
                _lifecycle_fail("stage report contains an invalid comparison")
            ChampionComparison.__post_init__(self.comparison)
            if self.invalid_reason is not None:
                _lifecycle_fail("scored stage evidence cannot carry an invalid reason")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_sha256": self.policy_sha256,
            "evidence_fingerprint": self.evidence_fingerprint,
            "comparison": (
                asdict(self.comparison) if self.comparison is not None else None
            ),
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class ChampionEvaluationReport:
    schema_version: int
    input_fingerprint: str
    split: Literal["calibration", "dev"]
    parent_sha256: str
    candidates: tuple[ChampionStageCandidateReport, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("evaluation report schema_version must be exactly one")
        _require_sha256(self.input_fingerprint, "report input fingerprint")
        _require_sha256(self.parent_sha256, "report Parent fingerprint")
        if type(self.split) is not str or self.split not in {"calibration", "dev"}:
            _lifecycle_fail("evaluation report has an invalid split")
        if type(self.candidates) is not tuple or not self.candidates:
            _lifecycle_fail("evaluation report requires candidate evidence")
        for candidate in self.candidates:
            if type(candidate) is not ChampionStageCandidateReport:
                _lifecycle_fail("evaluation report candidate is malformed")
            ChampionStageCandidateReport.__post_init__(candidate)
        fingerprints = tuple(item.policy_sha256 for item in self.candidates)
        if len(fingerprints) != len(set(fingerprints)):
            _lifecycle_fail("evaluation report contains duplicate candidates")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_fingerprint": self.input_fingerprint,
            "split": self.split,
            "parent_sha256": self.parent_sha256,
            "candidates": [item.to_payload() for item in self.candidates],
        }


@dataclass(frozen=True)
class ChampionEvolutionOutcome:
    manifest: ChampionRunManifest
    checkpoint: ChampionCheckpoint
    release: ChampionRelease
    build_result: BuildEvolutionResult | None
    calibration_report: ChampionEvaluationReport | None
    dev_report: ChampionEvaluationReport | None


def canonical_release_bytes(release: ChampionRelease) -> bytes:
    """Return the exact canonical bytes used by release publication."""
    if type(release) is not ChampionRelease:
        _lifecycle_fail("release must be an exact ChampionRelease")
    _validated_parent(release)
    return canonical_json_bytes(release.to_payload())


def _reject_nonfinite_json(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        _lifecycle_fail("stored JSON contains a nonfinite number")
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                _lifecycle_fail("stored JSON contains a non-string key")
            _reject_nonfinite_json(item)
    elif type(value) is list:
        for item in cast(list[object], value):
            _reject_nonfinite_json(item)


class ChampionArtifactStore:
    """Canonical, atomic, immutable lifecycle artifact storage."""

    def __init__(self, root: str | Path) -> None:
        raw_root = Path(root)
        absolute_root = raw_root.absolute()
        cursor = Path(absolute_root.anchor)
        for component in absolute_root.parts[1:]:
            cursor /= component
            if cursor.is_symlink():
                _lifecycle_fail("artifact root cannot use a symlink or path alias")
        raw_root.mkdir(parents=True, exist_ok=True)
        if not raw_root.is_dir() or raw_root.is_symlink():
            _lifecycle_fail("artifact root must be an exact directory")
        self.root = raw_root.resolve()

    def _path(self, name: str) -> Path:
        if (
            type(name) is not str
            or not name
            or Path(name).name != name
            or not name.endswith(".json")
        ):
            _lifecycle_fail("artifact name must be a simple JSON filename")
        path = self.root / name
        if path.is_symlink():
            _lifecycle_fail("artifact path cannot be a symlink or path alias")
        if path.exists() and path.stat().st_nlink != 1:
            _lifecycle_fail("artifact path cannot be a hardlink alias")
        return path

    def _sync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_create(self, path: Path, payload: dict[str, object]) -> None:
        if path.exists() or path.is_symlink():
            _lifecycle_fail(f"immutable artifact already exists: {path.name}")
        content = canonical_json_bytes(payload)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as error:
                raise ChampionLifecycleError(
                    f"immutable artifact already exists: {path.name}"
                ) from error
            self._sync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _atomic_replace(self, path: Path, content: bytes) -> None:
        if path.is_symlink():
            _lifecycle_fail("release path cannot be a symlink or path alias")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            self._sync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _read_payload(self, path: Path) -> dict[str, object]:
        if path.is_symlink() or not path.is_file():
            _lifecycle_fail(f"artifact is missing or aliased: {path.name}")
        if path.stat().st_nlink != 1:
            _lifecycle_fail(f"artifact is an immutable hardlink alias: {path.name}")
        try:
            raw = path.read_bytes()
            decoded = raw.decode("utf-8")
            payload = strict_json_loads(decoded, context=str(path))
        except (OSError, UnicodeError, ValueError) as error:
            raise ChampionLifecycleError(
                f"artifact contains malformed JSON: {path.name}"
            ) from error
        if type(payload) is not dict:
            _lifecycle_fail(f"artifact must contain an exact JSON object: {path.name}")
        result = cast(dict[str, object], payload)
        _reject_nonfinite_json(result)
        try:
            canonical = canonical_json_bytes(result)
        except (TypeError, ValueError) as error:
            raise ChampionLifecycleError(
                f"artifact is not canonical JSON: {path.name}"
            ) from error
        if raw != canonical:
            _lifecycle_fail(f"artifact is not canonical JSON: {path.name}")
        return result

    def bind_manifest(self, manifest: ChampionRunManifest) -> Path:
        if type(manifest) is not ChampionRunManifest:
            _lifecycle_fail("manifest must be an exact ChampionRunManifest")
        ChampionRunManifest.__post_init__(manifest)
        path = self._path("run_manifest.json")
        if not path.exists():
            self._atomic_create(path, manifest.to_payload())
            return path
        if canonical_json_bytes(self._read_payload(path)) != canonical_json_bytes(
            manifest.to_payload()
        ):
            _lifecycle_fail("run manifest input drifted from durable authority")
        return path

    def write_checkpoint(self, checkpoint: ChampionCheckpoint) -> Path:
        if type(checkpoint) is not ChampionCheckpoint:
            _lifecycle_fail("checkpoint must be an exact ChampionCheckpoint")
        ChampionCheckpoint.__post_init__(checkpoint)
        path = self._path("checkpoint.json")
        self._atomic_create(path, checkpoint.to_payload())
        return path

    def load_checkpoint(self) -> ChampionCheckpoint | None:
        path = self._path("checkpoint.json")
        if not path.exists():
            return None
        payload = self._read_payload(path)
        if set(payload) != {
            "schema_version",
            "input_fingerprint",
            "completed_stage",
            "active_parent",
            "proposal_archive",
            "sanitized_feedback",
        }:
            _lifecycle_fail("checkpoint JSON has a malformed schema")
        try:
            archive_payload = payload["proposal_archive"]
            if type(archive_payload) is not list:
                _lifecycle_fail("checkpoint proposal archive must be a JSON array")
            feedback = payload["sanitized_feedback"]
            if type(feedback) is not dict:
                _lifecycle_fail("checkpoint feedback must be a JSON object")
            return ChampionCheckpoint(
                schema_version=payload["schema_version"],  # type: ignore[arg-type]
                input_fingerprint=payload["input_fingerprint"],  # type: ignore[arg-type]
                completed_stage=payload["completed_stage"],  # type: ignore[arg-type]
                active_parent=parse_champion_release(payload["active_parent"]),
                proposal_archive=tuple(
                    _parse_fitted_policy(item)
                    for item in cast(list[object], archive_payload)
                ),
                sanitized_feedback=_parse_sanitized_feedback(
                    cast(dict[str, object], feedback)
                ),
            )
        except ChampionLifecycleError:
            raise
        except Exception as error:
            raise ChampionLifecycleError("checkpoint JSON is malformed") from error

    def report_exists(self, split: Literal["calibration", "dev"]) -> bool:
        if type(split) is not str or split not in {"calibration", "dev"}:
            _lifecycle_fail("artifact store has no authority for this split")
        return self._path(f"{split}_report.json").exists()

    def write_report(self, report: ChampionEvaluationReport) -> Path:
        if type(report) is not ChampionEvaluationReport:
            _lifecycle_fail("report must be an exact ChampionEvaluationReport")
        ChampionEvaluationReport.__post_init__(report)
        path = self._path(f"{report.split}_report.json")
        self._atomic_create(path, report.to_payload())
        return path

    def ensure_release(self, release: ChampionRelease) -> Path:
        path = self._path("champion_release.json")
        expected = canonical_release_bytes(release)
        if not path.exists():
            return self.publish_release(release)
        payload = self._read_payload(path)
        try:
            stored = parse_champion_release(payload)
        except Exception as error:
            raise ChampionLifecycleError("stored Champion release is malformed") from error
        if canonical_release_bytes(stored) != expected:
            _lifecycle_fail("active release drifted from the exact Parent")
        return path

    def publish_release(self, release: ChampionRelease) -> Path:
        content = canonical_release_bytes(release)
        release_id = champion_fingerprint(release)
        archive_root = self.root / "releases"
        if archive_root.exists() and archive_root.is_symlink():
            _lifecycle_fail("release archive cannot be a symlink or path alias")
        archive_root.mkdir(exist_ok=True)
        archive_path = archive_root / f"{release_id}.json"
        if archive_path.is_symlink():
            _lifecycle_fail("release archive path cannot be a symlink or alias")
        if archive_path.exists():
            if canonical_json_bytes(self._read_payload(archive_path)) != canonical_json_bytes(
                release.to_payload()
            ):
                _lifecycle_fail("immutable release archive drifted")
        else:
            self._atomic_create(archive_path, release.to_payload())
        current = self._path("champion_release.json")
        self._atomic_replace(current, content)
        return current


@dataclass(frozen=True)
class ChampionEvolutionConfig:
    """Pre-registered Build schedule and candidate gates."""

    build_size: int = 64
    calibration_size: int = 16
    screen_sizes: tuple[int, ...] = (8, 32, 64)
    generations: int = 1
    minimum_recipes: int = 5
    maximum_recipes: int = 10
    maximum_full_build_children: int = 3
    maximum_finalists: int = 3
    candidate_minimum_gain: float = 0.005
    research_target_gain: float = 0.05
    gate_config: ChampionGateConfig = field(
        default_factory=lambda: ChampionGateConfig(minimum_improved_folds=4)
    )
    build_task_ids: tuple[str, ...] = ()
    screen_task_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if type(self.build_size) is not int or type(self.calibration_size) is not int:
            _fail("Build and Calibration sizes must be exact integers")
        if type(self.screen_sizes) is not tuple or any(
            type(size) is not int for size in self.screen_sizes
        ):
            _fail("screen sizes must be an exact integer tuple")
        sizing = (self.build_size, self.calibration_size, self.screen_sizes)
        if sizing not in {_FORMAL_SIZES, _SMOKE_SIZES}:
            _fail("only the formal 64/16 or deterministic 8/2 smoke schedule is allowed")
        for name in (
            "generations",
            "minimum_recipes",
            "maximum_recipes",
            "maximum_full_build_children",
            "maximum_finalists",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"{name} must be an exact positive integer")
        if not 5 <= self.minimum_recipes <= self.maximum_recipes <= 10:
            _fail("each generation must propose five through ten recipes")
        if self.maximum_full_build_children > 3 or self.maximum_finalists > 3:
            _fail("full Build and finalist bounds cannot exceed three")
        for name in ("candidate_minimum_gain", "research_target_gain"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                _fail(f"{name} must be a finite nonnegative float")
        if self.research_target_gain < self.candidate_minimum_gain:
            _fail("research target cannot be below the candidate threshold")
        if type(self.gate_config) is not ChampionGateConfig:
            _fail("gate_config must be an exact ChampionGateConfig")
        ChampionGateConfig.__post_init__(self.gate_config)
        if type(self.build_task_ids) is not tuple or any(
            type(task_id) is not str or not task_id for task_id in self.build_task_ids
        ):
            _fail("Build task membership must be an exact string tuple")
        if self.build_task_ids and (
            len(self.build_task_ids) != self.build_size
            or len(self.build_task_ids) != len(set(self.build_task_ids))
        ):
            _fail("Build task membership must contain the fixed unique universe")
        if type(self.screen_task_ids) is not tuple or any(
            type(stage) is not tuple for stage in self.screen_task_ids
        ):
            _fail("screen membership must be an exact tuple of exact tuples")
        if self.screen_task_ids:
            if len(self.screen_task_ids) != len(self.screen_sizes):
                _fail("screen membership must cover every fixed stage")
            for size, stage in zip(self.screen_sizes, self.screen_task_ids, strict=True):
                if (
                    len(stage) != size
                    or any(type(task_id) is not str or not task_id for task_id in stage)
                    or len(stage) != len(set(stage))
                ):
                    _fail("each screen stage must have its exact unique membership")
            for earlier, later in zip(
                self.screen_task_ids, self.screen_task_ids[1:], strict=False
            ):
                if not set(earlier).issubset(later):
                    _fail("screen membership must be successively nested")

    @property
    def fingerprint(self) -> str:
        """Fingerprint schedule, gates, thresholds, and fixed task membership."""
        return champion_fingerprint(self)

    @property
    def screen_membership_sha256(self) -> str:
        return champion_fingerprint({"screen_task_ids": self.screen_task_ids})


@dataclass(frozen=True)
class BuildAttempt:
    """The latest trusted Build evidence for one expanded fitted Child."""

    fitted_id: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...]
    comparison: ChampionComparison | None
    status: Literal[
        "invalid", "pruned", "finalist", "below_candidate_threshold"
    ]
    invalid_reason: Literal["unscorable_child"] | None = None
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass(frozen=True)
class BuildGeneration:
    """One structural proposal generation evaluated against the same Parent."""

    number: int
    mutation_parent_sha256: str
    config_fingerprint: str
    stage_counts: tuple[int, ...]
    full_build_children: int
    attempts: tuple[BuildAttempt, ...]
    finalists: tuple[FittedChampionPolicy, ...]
    feedback: ProposerEvidence


@dataclass(frozen=True)
class BuildEvolutionResult:
    """Build diagnostics and shortlist; active_parent is never a Build Child."""

    active_parent: object
    generations: tuple[BuildGeneration, ...]
    shortlist: tuple[FittedChampionPolicy, ...]
    config: ChampionEvolutionConfig
    config_fingerprint: str
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass
class _AttemptState:
    fitted_id: str
    score_name: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...] = ()
    comparison: ChampionComparison | None = None
    evidence_rows: tuple[ChampionTaskRow, ...] = ()
    invalid_reason: Literal["unscorable_child"] | None = None


def _capture_object_graph(
    value: object,
) -> tuple[tuple[object, tuple[tuple[str, object], ...]], ...]:
    """Capture exact dataclass fields so hostile callbacks can be rolled back."""
    captured: list[tuple[object, tuple[tuple[str, object], ...]]] = []
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        if is_dataclass(item) and not isinstance(item, type):
            seen.add(identity)
            values = tuple((entry.name, getattr(item, entry.name)) for entry in fields(item))
            captured.append((item, values))
            for _, nested in values:
                visit(nested)
            return
        if type(item) in {tuple, list}:
            seen.add(identity)
            for nested in cast(tuple[object, ...] | list[object], item):
                visit(nested)
        elif type(item) is dict:
            seen.add(identity)
            for key, nested in cast(dict[object, object], item).items():
                visit(key)
                visit(nested)

    visit(value)
    return tuple(captured)


def _restore_object_graph(
    captured: tuple[tuple[object, tuple[tuple[str, object], ...]], ...],
) -> None:
    for item, values in captured:
        field_names = {name for name, _ in values}
        try:
            extra_names = tuple(set(vars(item)) - field_names)
        except TypeError:
            extra_names = ()
        for name in extra_names:
            object.__delattr__(item, name)
        for name, value in values:
            object.__setattr__(item, name, value)


def _fingerprint_changed(value: object, expected: str) -> bool:
    try:
        return champion_fingerprint(value) != expected
    except Exception:
        return True


def _validated_parent(
    parent: object,
) -> tuple[ChampionRecipe, FittedChampionPolicy | None]:
    def validate_recipe(recipe: object) -> ChampionRecipe:
        if type(recipe) is not ChampionRecipe:
            _fail("active Parent contains an invalid recipe")
        canonical = cast(ChampionRecipe, recipe)
        for assumption in canonical.assumptions:
            if type(assumption) is not EvolutionAssumption:
                _fail("active Parent contains an invalid assumption")
            EvolutionAssumption.__post_init__(assumption)
        ChampionRecipe.__post_init__(canonical)
        return canonical

    try:
        if type(parent) is ChampionRelease:
            ChampionRelease.__post_init__(parent)
            FittedChampionPolicy.__post_init__(parent.policy)
            return validate_recipe(parent.policy.recipe), parent.policy
        if type(parent) is FittedChampionPolicy:
            FittedChampionPolicy.__post_init__(parent)
            return validate_recipe(parent.recipe), parent
        if type(parent) is ChampionRecipe:
            return validate_recipe(parent), None
    except ChampionControllerError:
        raise
    except Exception as error:
        raise ChampionControllerError("active Parent violates its frozen contract") from error
    _fail("active Parent must be an exact ChampionRelease, policy, or recipe")
    raise AssertionError("unreachable")


def _validated_rows(
    rows: object,
    *,
    build_size: int,
) -> tuple[tuple[ChampionTaskRow, ...], tuple[str, ...], tuple[str, ...]]:
    if type(rows) not in {tuple, list} or not rows:
        _fail("Build rows must be a nonempty exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], rows))
    seen_keys: set[tuple[str, str]] = set()
    task_ids: list[str] = []
    identities: dict[str, tuple[object, object, int, str, object]] = {}
    candidate_tasks: dict[str, set[str]] = {}
    normalized_candidates: dict[str, str] = {}
    for raw_row in snapshot:
        if type(raw_row) is not ChampionTaskRow:
            _fail("Build rows must contain exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionControllerError("Build row violates its trusted contract") from error
        if row.split != "build":
            _fail("Build evolution accepts Build rows only")
        normalized_name = _canonical_identity(row.candidate_name)
        prior_name = normalized_candidates.setdefault(
            normalized_name, row.candidate_name
        )
        if prior_name != row.candidate_name:
            _fail("normalized row inventory contains a candidate collision")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _fail("Build rows contain a duplicate candidate/task key")
        seen_keys.add(key)
        if row.task_id not in identities:
            task_ids.append(row.task_id)
        identity = (row.profile, row.truth, row.fold, row.split, row.history)
        previous = identities.setdefault(row.task_id, identity)
        if identity != previous:
            _fail(
                "Build row universe drifted across profile, truth, fold, split, "
                "or history"
            )
        candidate_tasks.setdefault(row.candidate_name, set()).add(row.task_id)
    if len(task_ids) != build_size:
        _fail("Build rows do not contain the configured task universe")
    universe = set(task_ids)
    if any(task_set != universe for task_set in candidate_tasks.values()):
        _fail("every materialized candidate must cover the fixed Build universe")
    return (
        cast(tuple[ChampionTaskRow, ...], snapshot),
        tuple(task_ids),
        tuple(candidate_tasks),
    )


def _bound_config(
    config: ChampionEvolutionConfig,
    task_ids: tuple[str, ...],
) -> ChampionEvolutionConfig:
    if config.build_task_ids and config.build_task_ids != task_ids:
        _fail("Build task membership drifted from the pre-registered universe")
    screens = config.screen_task_ids or tuple(
        task_ids[:size] for size in config.screen_sizes
    )
    bound = _clone_config(
        config,
        build_task_ids=task_ids,
        screen_task_ids=screens,
    )
    if set(bound.screen_task_ids[-1]) != set(task_ids):
        _fail("the final screen must equal the complete fixed Build universe")
    if any(not set(stage).issubset(task_ids) for stage in bound.screen_task_ids):
        _fail("screen membership contains a task outside fixed Build")
    return bound


def _clone_gate(config: ChampionGateConfig) -> ChampionGateConfig:
    return ChampionGateConfig(
        **{entry.name: getattr(config, entry.name) for entry in fields(config)}
    )


def _clone_config(
    config: ChampionEvolutionConfig,
    *,
    build_task_ids: tuple[str, ...] | None = None,
    screen_task_ids: tuple[tuple[str, ...], ...] | None = None,
) -> ChampionEvolutionConfig:
    """Detach every nested registered value from caller-owned aliases."""
    return ChampionEvolutionConfig(
        build_size=config.build_size,
        calibration_size=config.calibration_size,
        screen_sizes=tuple(config.screen_sizes),
        generations=config.generations,
        minimum_recipes=config.minimum_recipes,
        maximum_recipes=config.maximum_recipes,
        maximum_full_build_children=config.maximum_full_build_children,
        maximum_finalists=config.maximum_finalists,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        gate_config=_clone_gate(config.gate_config),
        build_task_ids=(
            tuple(config.build_task_ids)
            if build_task_ids is None
            else tuple(build_task_ids)
        ),
        screen_task_ids=(
            tuple(tuple(stage) for stage in config.screen_task_ids)
            if screen_task_ids is None
            else tuple(tuple(stage) for stage in screen_task_ids)
        ),
    )


def _require_config_fingerprint(
    config: ChampionEvolutionConfig,
    expected: str,
) -> None:
    if _fingerprint_changed(config, expected):
        _fail("registered Champion evolution config mutated")


def _feature_value(policy: FittedChampionPolicy, task_row: ChampionTaskRow, index: int) -> float:
    assumption = policy.recipe.assumptions[index]
    feature = assumption.feature
    profile = task_row.profile
    try:
        if feature == "horizon_ratio":
            value: object = profile.horizon / profile.history_length
        elif feature in _PROFILE_FEATURES:
            value = getattr(profile, feature)
        else:
            _fail("fitted Child references an unsupported Build feature")
    except (ArithmeticError, AttributeError, TypeError, ValueError) as error:
        raise ChampionControllerError("fitted Child feature is invalid") from error
    if type(value) not in {int, float}:
        _fail("fitted Child feature must be exactly numeric")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail("fitted Child feature must be finite")
    return number


def _policy_forecast(
    policy: FittedChampionPolicy,
    task_rows: dict[str, ChampionTaskRow],
) -> tuple[tuple[float, ...] | None, str | None]:
    fallback = task_rows.get(policy.recipe.fallback_parent)
    if fallback is None or fallback.forecast is None:
        return None, "fallback_unavailable"
    parent_rows = tuple(task_rows.get(name) for name in policy.recipe.parents)
    enriched = fallback.history is not None or fallback.diagnostic is not None
    if enriched or any(
        row is not None and (row.history is not None or row.diagnostic is not None)
        for row in parent_rows
    ):
        if fallback.history is None:
            return None, "history_unavailable"
        forecasts: dict[str, tuple[float, ...]] = {}
        diagnostics: dict[str, CandidateDiagnostics] = {}
        for name, row in zip(policy.recipe.parents, parent_rows, strict=True):
            if row is None or row.forecast is None:
                return None, "parent_forecast_unavailable"
            if row.history != fallback.history:
                return None, "history_mismatch"
            if row.diagnostic is None or not _valid_history_diagnostic(row.diagnostic):
                return None, "history_diagnostic_unavailable"
            forecasts[name] = row.forecast
            diagnostics[name] = _runtime_diagnostic(row.diagnostic)
        execution = execute_champion(
            policy,
            forecasts,
            diagnostics,
            fallback.profile,
            fallback.history,
            fallback.profile.horizon,
        )
        if execution.fallback_reason not in {None, "assumption_not_satisfied"}:
            return None, f"runtime_{execution.fallback_reason}"
        return execution.forecast, None

    parents: dict[str, tuple[float, ...]] = {}
    for name in policy.recipe.parents:
        row = task_rows.get(name)
        if row is None or row.forecast is None:
            return fallback.forecast, None
        parents[name] = row.forecast
    for index, (assumption, (_, threshold)) in enumerate(
        zip(policy.recipe.assumptions, policy.thresholds, strict=True)
    ):
        value = _feature_value(policy, fallback, index)
        matches = value >= threshold if assumption.direction == "above" else value < threshold
        if not matches:
            return fallback.forecast, None

    kind = policy.recipe.kind
    if kind == "select":
        return parents[policy.recipe.parents[0]], None
    if kind == "route":
        routed = policy.recipe.assumptions[0].candidate_name
        if any(item.candidate_name != routed for item in policy.recipe.assumptions):
            return None, "invalid_route"
        return parents[routed], None
    left_name, right_name = policy.recipe.parents
    left, right = parents[left_name], parents[right_name]
    if kind == "horizon_route":
        split = int(len(left) * policy.horizon_split)
        return left[:split] + right[split:], None
    if kind == "weighted":
        return tuple(
            left[index] * policy.weights[0] + right[index] * policy.weights[1]
            for index in range(len(left))
        ), None
    if kind == "median":
        return tuple(
            left[index] / 2.0 + right[index] / 2.0 for index in range(len(left))
        ), None
    if kind == "bounded_overlay":
        return None, "history_scale_unavailable"
    return None, "unsupported_operator"


def _valid_history_diagnostic(diagnostic: object) -> bool:
    if type(diagnostic) is not ChampionHistoryDiagnostic:
        return False
    try:
        ChampionHistoryDiagnostic.__post_init__(diagnostic)
    except (ChampionEvidenceError, TypeError, ValueError):
        return False
    return True


def _runtime_diagnostic(
    diagnostic: ChampionHistoryDiagnostic,
) -> CandidateDiagnostics:
    """Construct Task 3 input with no fold objects, forecasts, truths, or cache."""
    return CandidateDiagnostics(
        name=diagnostic.name,
        family=diagnostic.family,
        folds=(),
        successful_folds=diagnostic.successful_folds,
        eligible=diagnostic.eligible,
        reason_code=diagnostic.reason_code,
        median_mase=diagnostic.median_mase,
        recent_mase=diagnostic.recent_mase,
        worst_mase=diagnostic.worst_mase,
        mase_mad=diagnostic.mase_mad,
        median_mae=diagnostic.median_mae,
        median_smape=diagnostic.median_smape,
        median_rmsse=diagnostic.median_rmsse,
        normalized_bias=diagnostic.normalized_bias,
        slope_error=diagnostic.slope_error,
        phase_error=diagnostic.phase_error,
        amplitude_ratio=diagnostic.amplitude_ratio,
        explosion=diagnostic.explosion,
        fold_forecasts=(),
        fold_truths=(),
        cache_key="",
        long_horizon_fold=None,
        long_horizon_coverage=diagnostic.long_horizon_coverage,
        median_joint_scaled_error=diagnostic.median_joint_scaled_error,
        recent_joint_scaled_error=diagnostic.recent_joint_scaled_error,
        worst_joint_scaled_error=diagnostic.worst_joint_scaled_error,
        median_smae=diagnostic.median_smae,
        recent_smae=diagnostic.recent_smae,
        worst_smae=diagnostic.worst_smae,
        smae_mad=diagnostic.smae_mad,
        median_srmse=diagnostic.median_srmse,
        recent_srmse=diagnostic.recent_srmse,
        worst_srmse=diagnostic.worst_srmse,
        srmse_mad=diagnostic.srmse_mad,
        worst_smae_raw=diagnostic.worst_smae_raw,
        worst_srmse_raw=diagnostic.worst_srmse_raw,
    )


def _materialize_policy(
    policy: FittedChampionPolicy,
    score_name: str,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    *,
    split: Literal["build", "calibration", "dev"] = "build",
) -> tuple[ChampionTaskRow, ...]:
    by_task: dict[str, dict[str, ChampionTaskRow]] = {task_id: {} for task_id in task_ids}
    identity: dict[str, ChampionTaskRow] = {}
    selected = set(task_ids)
    for row in rows:
        if row.task_id not in selected:
            continue
        by_task[row.task_id][row.candidate_name] = row
        identity.setdefault(row.task_id, row)
    materialized: list[ChampionTaskRow] = []
    for task_id in task_ids:
        base = identity[task_id]
        try:
            forecast, failure = _policy_forecast(policy, by_task[task_id])
        except (ArithmeticError, ChampionControllerError, OverflowError, ValueError):
            forecast, failure = None, "invalid_policy_materialization"
        if forecast is not None and any(not math.isfinite(value) for value in forecast):
            forecast, failure = None, "invalid_policy_materialization"
        materialized.append(
            ChampionTaskRow(
                task_id=task_id,
                candidate_name=score_name,
                profile=base.profile,
                truth=base.truth,
                forecast=forecast,
                failure_reason=failure if forecast is None else None,
                fold=base.fold,
                split=split,
            )
        )
    return tuple(materialized)


def _parent_rows(
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
) -> tuple[str, tuple[ChampionTaskRow, ...]]:
    supplied = tuple(
        row
        for row in rows
        if row.task_id in set(task_ids) and row.candidate_name == parent_recipe.name
    )
    if supplied:
        if len(supplied) != len(task_ids):
            _fail("supplied Parent rows do not cover fixed screen membership")
        return parent_recipe.name, supplied
    if parent_policy is None:
        _fail("an unfitted Parent recipe requires trusted materialized Parent rows")
    split = rows[0].split
    if split not in {"build", "calibration", "dev"}:
        _fail("Parent materialization received an unauthorized split")
    return parent_recipe.name, _materialize_policy(
        parent_policy,
        parent_recipe.name,
        rows,
        task_ids,
        split=cast(Literal["build", "calibration", "dev"], split),
    )


def _stage_gate(config: ChampionEvolutionConfig, *, final: bool) -> ChampionGateConfig:
    del final
    return config.gate_config


def _evaluate_stage(
    state: _AttemptState,
    *,
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    gate: ChampionGateConfig,
) -> _AttemptState:
    parent_name, parent_rows = _parent_rows(
        parent_recipe, parent_policy, rows, task_ids
    )
    child_rows = _materialize_policy(state.policy, state.score_name, rows, task_ids)
    evidence_rows = parent_rows + child_rows
    try:
        parent_score = score_policy(evidence_rows, parent_name)
    except (ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionControllerError("trusted Build scoring rejected the Parent") from error
    try:
        child_score = score_policy(evidence_rows, state.score_name)
        comparison = compare_champion(parent_score, child_score, gate)
    except (ChampionEvidenceError, TypeError, ValueError):
        return replace(
            state,
            stage_task_counts=state.stage_task_counts + (len(task_ids),),
            comparison=None,
            evidence_rows=evidence_rows,
            invalid_reason="unscorable_child",
        )
    return replace(
        state,
        stage_task_counts=state.stage_task_counts + (len(task_ids),),
        comparison=comparison,
        evidence_rows=evidence_rows,
        invalid_reason=None,
    )


def _rank(state: _AttemptState) -> tuple[float, float, float, str]:
    if state.comparison is None:
        _fail("attempt has no trusted comparison")
    comparison = cast(ChampionComparison, state.comparison)
    return (
        -comparison.joint_improvement,
        comparison.mean_delta_smae,
        comparison.mean_delta_srmse,
        state.fitted_id,
    )


def _diverse(
    states: tuple[_AttemptState, ...],
    *,
    limit: int,
) -> tuple[_AttemptState, ...]:
    ranked = tuple(sorted(states, key=_rank))
    selected: list[_AttemptState] = []
    seen_kinds: set[str] = set()
    seen_structures: set[tuple[str, tuple[str, ...]]] = set()
    for state in ranked:
        kind = state.policy.recipe.kind
        if kind in seen_kinds:
            continue
        selected.append(state)
        seen_kinds.add(kind)
        seen_structures.add((kind, state.policy.recipe.parents))
        if len(selected) == limit:
            return tuple(selected)
    for state in ranked:
        structure = (state.policy.recipe.kind, state.policy.recipe.parents)
        if state in selected or structure in seen_structures:
            continue
        selected.append(state)
        seen_structures.add(structure)
        if len(selected) == limit:
            break
    return tuple(selected)


def _feedback(states: tuple[_AttemptState, ...]) -> ProposerEvidence:
    scored_states = tuple(
        state
        for state in states
        if state.comparison is not None and state.evidence_rows
    )
    invalid_states = tuple(
        state
        for state in states
        if state.comparison is None and state.invalid_reason is not None
    )
    if len(scored_states) + len(invalid_states) != len(states):
        _fail("every Build attempt requires genuine or typed invalid feedback")
    invalid_attempts: list[_InvalidAttemptAggregate] = []
    for state in invalid_states:
        if (
            champion_fingerprint(state.policy) != state.fitted_id
            or not state.stage_task_counts
        ):
            _fail("invalid Build attempt feedback lost its host binding")
        invalid_attempts.append(
            _InvalidAttemptAggregate(
                structure_sha256=state.fitted_id,
                kind=state.policy.recipe.kind,
                stage_support=state.stage_task_counts[-1],
                reason_code=state.invalid_reason,
            )
        )
    grouped: dict[tuple[str, ...], list[_AttemptState]] = {}
    for state in scored_states:
        comparison = cast(ChampionComparison, state.comparison)
        task_ids = tuple(
            row.task_id
            for row in state.evidence_rows
            if row.candidate_name == comparison.parent_name
        )
        grouped.setdefault(task_ids, []).append(state)
    evidence_parts: list[ProposerEvidence] = []
    for group in grouped.values():
        parent_name = cast(ChampionComparison, group[0].comparison).parent_name
        parent_rows = tuple(
            row for row in group[0].evidence_rows if row.candidate_name == parent_name
        )
        child_rows = tuple(
            row
            for state in group
            for row in state.evidence_rows
            if row.candidate_name == state.score_name
        )
        comparisons = tuple(cast(ChampionComparison, state.comparison) for state in group)
        try:
            evidence_parts.append(
                sanitize_build_evidence(parent_rows + child_rows, comparisons)
            )
        except (ChampionEvidenceError, TypeError, ValueError) as error:
            raise ChampionControllerError("Build feedback could not be sanitized") from error
    return ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            sorted(
                (item for part in evidence_parts for item in part.morphology),
                key=lambda item: (item.group_id, item.candidate_name),
            )
        ),
        comparisons=tuple(
            sorted(
                (item for part in evidence_parts for item in part.comparisons),
                key=lambda item: item.candidate_name,
            )
        ),
        invalid_attempts=tuple(
            sorted(
                invalid_attempts,
                key=lambda item: (item.structure_sha256, item.reason_code),
            )
        ),
    )


def _call_proposer(
    proposer: Callable[[object, ProposerEvidence], object],
    parent: object,
    feedback: ProposerEvidence,
    *,
    minimum: int,
    maximum: int,
) -> tuple[ChampionRecipe, ...]:
    if not callable(proposer):
        _fail("proposer must be callable")
    try:
        proposed = proposer(parent, feedback)
    except Exception as error:
        raise ChampionControllerError("structural proposer callback failed") from error
    if type(proposed) not in {tuple, list}:
        _fail("proposer must return an exact tuple or list")
    recipes = tuple(cast(tuple[object, ...] | list[object], proposed))
    if not minimum <= len(recipes) <= maximum:
        _fail("proposer must return the configured five through ten recipes")
    validated: list[ChampionRecipe] = []
    for raw_recipe in recipes:
        if type(raw_recipe) is not ChampionRecipe:
            _fail("proposer output must contain exact ChampionRecipe values")
        recipe = cast(ChampionRecipe, raw_recipe)
        try:
            ChampionRecipe.__post_init__(recipe)
            detached = parse_champion_recipe(recipe.to_payload())
        except Exception as error:
            raise ChampionControllerError(
                "proposer returned an invalid Champion recipe"
            ) from error
        validated.append(detached)
    recipe_ids = tuple(champion_fingerprint(recipe) for recipe in validated)
    recipe_names = tuple(recipe.name for recipe in validated)
    canonical_names = tuple(_canonical_identity(name) for name in recipe_names)
    if len(recipe_ids) != len(set(recipe_ids)) or len(canonical_names) != len(
        set(canonical_names)
    ):
        _fail("proposer returned duplicate policy identities")
    for recipe in validated:
        canonical_parents = tuple(
            _canonical_identity(name) for name in recipe.parents
        )
        canonical_assumptions = tuple(
            _canonical_identity(item.assumption_id) for item in recipe.assumptions
        )
        if len(canonical_parents) != len(set(canonical_parents)) or len(
            canonical_assumptions
        ) != len(set(canonical_assumptions)):
            _fail("proposer returned duplicate policy identities")
    return tuple(validated)


def _stored_policies(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[FittedChampionPolicy, ...]:
    values = [
        attempt.policy
        for generation in generations
        for attempt in generation.attempts
    ]
    values.extend(state.policy for state in archive)
    unique: list[FittedChampionPolicy] = []
    seen: set[int] = set()
    for policy in values:
        if id(policy) in seen:
            continue
        seen.add(id(policy))
        unique.append(policy)
    return tuple(unique)


def _stored_policy_fingerprints(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[tuple[FittedChampionPolicy, str], ...]:
    return tuple(
        (policy, champion_fingerprint(policy))
        for policy in _stored_policies(generations, archive)
    )


def _stored_policy_changed(
    fingerprints: tuple[tuple[FittedChampionPolicy, str], ...],
) -> bool:
    return any(
        _fingerprint_changed(policy, expected)
        for policy, expected in fingerprints
    )


def _validate_stored_policy_ids(generations: list[BuildGeneration]) -> None:
    for generation in generations:
        for attempt in generation.attempts:
            if champion_fingerprint(attempt.policy) != attempt.fitted_id:
                _fail("stored fitted policy fingerprint drifted")


def run_build_evolution(
    parent: object,
    rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
    proposer: Callable[[object, ProposerEvidence], object],
    config: ChampionEvolutionConfig,
) -> BuildEvolutionResult:
    """Run Build-only structural evolution without replacing ``parent``."""
    if type(config) is not ChampionEvolutionConfig:
        _fail("config must be an exact ChampionEvolutionConfig")
    ChampionEvolutionConfig.__post_init__(config)
    input_config_sha256 = config.fingerprint
    parent_recipe, parent_policy = _validated_parent(parent)
    snapshot, task_ids, row_candidate_names = _validated_rows(
        rows, build_size=config.build_size
    )
    bound_config = _bound_config(config, task_ids)
    bound_config_sha256 = bound_config.fingerprint
    parent_sha256 = champion_fingerprint(parent)
    rows_sha256 = champion_fingerprint(snapshot)
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    seen_recipe_ids: set[str] = {champion_fingerprint(parent_recipe)}
    seen_policy_names: set[str] = {_canonical_identity(parent_recipe.name)}
    reserved_row_names = {
        _canonical_identity(candidate_name)
        for candidate_name in row_candidate_names
    }
    seen_fitted_ids: set[str] = set()
    generations: list[BuildGeneration] = []
    archive: list[_AttemptState] = []

    for generation_number in range(1, bound_config.generations + 1):
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        parent_state = _capture_object_graph(parent)
        row_state = _capture_object_graph(snapshot)
        feedback_state = _capture_object_graph(feedback)
        config_state = _capture_object_graph(config)
        bound_config_state = _capture_object_graph(bound_config)
        stored_fingerprints = _stored_policy_fingerprints(generations, archive)
        stored_state = _capture_object_graph(
            tuple(policy for policy, _ in stored_fingerprints)
        )
        feedback_sha256 = champion_fingerprint(feedback)
        parent_changed = row_changed = feedback_changed = False
        config_changed = bound_config_changed = stored_changed = False
        callback_error: Exception | None = None
        recipes: tuple[ChampionRecipe, ...] | None = None
        try:
            try:
                recipes = _call_proposer(
                    proposer,
                    parent,
                    feedback,
                    minimum=bound_config.minimum_recipes,
                    maximum=bound_config.maximum_recipes,
                )
            except Exception as error:
                callback_error = error
            finally:
                parent_changed = _fingerprint_changed(parent, parent_sha256)
                row_changed = _fingerprint_changed(snapshot, rows_sha256)
                feedback_changed = _fingerprint_changed(feedback, feedback_sha256)
                config_changed = _fingerprint_changed(config, input_config_sha256)
                bound_config_changed = _fingerprint_changed(
                    bound_config, bound_config_sha256
                )
                stored_changed = _stored_policy_changed(stored_fingerprints)
        finally:
            _restore_object_graph(parent_state)
            _restore_object_graph(row_state)
            _restore_object_graph(feedback_state)
            _restore_object_graph(config_state)
            _restore_object_graph(bound_config_state)
            _restore_object_graph(stored_state)
        if parent_changed:
            _fail("proposer attempted to mutate the active Parent")
        if row_changed:
            _fail("proposer attempted to mutate the fixed Build rows")
        if feedback_changed:
            _fail("proposer attempted to mutate sanitized Build feedback")
        if config_changed or bound_config_changed:
            _fail("proposer attempted to mutate the registered config")
        if stored_changed:
            _fail("proposer attempted to mutate an archived fitted policy")
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        if callback_error is not None:
            raise callback_error
        if recipes is None:
            _fail("structural proposer returned no recipes")
        validated_recipes = cast(tuple[ChampionRecipe, ...], recipes)

        states: list[_AttemptState] = []
        for recipe in validated_recipes:
            recipe_id = champion_fingerprint(recipe)
            policy_name = _canonical_identity(recipe.name)
            if policy_name in reserved_row_names:
                _fail("proposal policy ID collides with materialized row inventory")
            if recipe_id in seen_recipe_ids or policy_name in seen_policy_names:
                _fail("duplicate policy ID across Build generations")
            seen_recipe_ids.add(recipe_id)
            seen_policy_names.add(policy_name)
            try:
                policies = expand_recipe(recipe, snapshot)
            except (ChampionProposalError, TypeError, ValueError) as error:
                raise ChampionControllerError(
                    "host numeric expansion rejected a recipe"
                ) from error
            for policy in policies:
                fitted_id = champion_fingerprint(policy)
                if fitted_id in seen_fitted_ids:
                    _fail("duplicate fitted policy ID across Build generations")
                seen_fitted_ids.add(fitted_id)
                states.append(
                    _AttemptState(
                        fitted_id=fitted_id,
                        score_name=f"build_child_{fitted_id}",
                        policy=policy,
                    )
                )
        if not states:
            _fail("host expansion produced an empty Build stage")

        active = tuple(states)
        for stage_index, stage_task_ids in enumerate(bound_config.screen_task_ids):
            if not active:
                break
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            _validate_stored_policy_ids(generations)
            evaluated = tuple(
                _evaluate_stage(
                    state,
                    parent_recipe=parent_recipe,
                    parent_policy=parent_policy,
                    rows=snapshot,
                    task_ids=stage_task_ids,
                    gate=_stage_gate(
                        bound_config,
                        final=stage_index == len(bound_config.screen_task_ids) - 1,
                    ),
                )
                for state in active
            )
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            updates = {state.fitted_id: state for state in evaluated}
            states = [updates.get(state.fitted_id, state) for state in states]
            safe = tuple(
                state
                for state in evaluated
                if state.comparison is not None and state.comparison.accepted
            )
            if stage_index == len(bound_config.screen_task_ids) - 1:
                active = safe
            elif stage_index == len(bound_config.screen_task_ids) - 2:
                active = _diverse(
                    safe, limit=bound_config.maximum_full_build_children
                )
            else:
                survivor_count = max(
                    bound_config.maximum_full_build_children,
                    (len(safe) + 1) // 2,
                )
                active = _diverse(safe, limit=survivor_count)

        full_build_children = sum(
            bool(state.stage_task_counts)
            and state.stage_task_counts[-1] == bound_config.build_size
            for state in states
        )
        eligible = tuple(
            state
            for state in active
            if state.comparison is not None
            and state.comparison.joint_improvement
            >= bound_config.candidate_minimum_gain
        )
        finalists = _diverse(eligible, limit=bound_config.maximum_finalists)
        finalist_ids = {state.fitted_id for state in finalists}
        attempts = tuple(
            BuildAttempt(
                fitted_id=state.fitted_id,
                policy=state.policy,
                stage_task_counts=state.stage_task_counts,
                comparison=state.comparison,
                status=(
                    "invalid"
                    if state.invalid_reason is not None
                    else "finalist"
                    if state.fitted_id in finalist_ids
                    else "below_candidate_threshold"
                    if state in active
                    else "pruned"
                ),
                invalid_reason=state.invalid_reason,
            )
            for state in states
        )
        generation_feedback = _feedback(tuple(states))
        feedback = ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=feedback.morphology + generation_feedback.morphology,
            comparisons=feedback.comparisons + generation_feedback.comparisons,
            invalid_attempts=tuple(
                sorted(
                    feedback.invalid_attempts + generation_feedback.invalid_attempts,
                    key=lambda item: (item.structure_sha256, item.reason_code),
                )
            ),
        )
        generations.append(
            BuildGeneration(
                number=generation_number,
                mutation_parent_sha256=parent_sha256,
                config_fingerprint=bound_config.fingerprint,
                stage_counts=bound_config.screen_sizes,
                full_build_children=full_build_children,
                attempts=attempts,
                finalists=tuple(state.policy for state in finalists),
                feedback=generation_feedback,
            )
        )
        archive.extend(finalists)

    shortlist = _diverse(tuple(archive), limit=bound_config.maximum_finalists)
    _require_config_fingerprint(config, input_config_sha256)
    _require_config_fingerprint(bound_config, bound_config_sha256)
    _validate_stored_policy_ids(generations)
    if champion_fingerprint(parent) != parent_sha256:
        _fail("Build evolution attempted to mutate the active Parent")
    if champion_fingerprint(snapshot) != rows_sha256:
        _fail("Build evolution attempted to mutate the fixed Build rows")
    return BuildEvolutionResult(
        active_parent=parent,
        generations=tuple(generations),
        shortlist=tuple(state.policy for state in shortlist),
        config=bound_config,
        config_fingerprint=bound_config.fingerprint,
    )


def _validated_lifecycle_tasks(
    tasks: object,
    *,
    label: str,
    expected_membership: tuple[tuple[str, str], ...],
) -> tuple[Task, ...]:
    if type(tasks) not in {tuple, list}:
        _lifecycle_fail(f"{label} tasks must be an exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], tasks))
    if len(snapshot) != len(expected_membership):
        _lifecycle_fail(f"{label} task count drifted from the run manifest")
    validated: list[Task] = []
    for raw_task in snapshot:
        if type(raw_task) is not Task:
            _lifecycle_fail(f"{label} tasks must contain exact Task values")
        task = cast(Task, raw_task)
        if type(task.task_id) is not str or not task.task_id:
            _lifecycle_fail(f"{label} contains an invalid task ID")
        if (
            type(task.entity_name) is not str
            or not task.entity_name.strip()
            or _canonical_identity(task.entity_name) == "unknown"
        ):
            _lifecycle_fail(f"{label} contains an unknown entity")
        if type(task.history_values) is not tuple or not task.history_values:
            _lifecycle_fail(f"{label} task history must be a nonempty exact tuple")
        if type(task.future_values) is not tuple or not task.future_values:
            _lifecycle_fail(f"{label} task truth must be a nonempty exact tuple")
        for series_name, values in (
            ("history", task.history_values),
            ("truth", task.future_values),
        ):
            if any(
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                for value in values
            ):
                _lifecycle_fail(f"{label} task {series_name} must be exactly finite")
        if (
            type(task.prediction_length) is not int
            or task.prediction_length <= 0
            or task.prediction_length != len(task.future_values)
        ):
            _lifecycle_fail(f"{label} task horizon is malformed")
        if type(task.frequency) is not str or not task.frequency.strip():
            _lifecycle_fail(f"{label} task frequency is malformed")
        if task.seasonal_period is not None and type(task.seasonal_period) is not str:
            _lifecycle_fail(f"{label} task seasonal period is malformed")
        validated.append(task)
    result = tuple(validated)
    membership = tuple((task.task_id, task.entity_name) for task in result)
    if membership != expected_membership:
        _lifecycle_fail(f"{label} task identity or entity drifted from the manifest")
    _validated_membership(membership, f"{label} membership", expected_size=len(result))
    return result


def _validated_stage_rows(
    rows: object,
    *,
    tasks: tuple[Task, ...],
    split: Literal["build", "calibration", "dev"],
) -> tuple[ChampionTaskRow, ...]:
    if type(rows) not in {tuple, list} or not rows:
        _lifecycle_fail(f"{split} row provider returned no exact rows")
    snapshot = tuple(cast(tuple[object, ...] | list[object], rows))
    by_task = {task.task_id: task for task in tasks}
    if len(by_task) != len(tasks):
        _lifecycle_fail(f"{split} tasks contain duplicate IDs")
    seen_keys: set[tuple[str, str]] = set()
    normalized_candidates: dict[str, str] = {}
    candidate_tasks: dict[str, set[str]] = {}
    for raw_row in snapshot:
        if type(raw_row) is not ChampionTaskRow:
            _lifecycle_fail(f"{split} rows require exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionLifecycleError(f"{split} row is malformed") from error
        if row.split != split:
            _lifecycle_fail(f"{split} row provider crossed a split boundary")
        task = by_task.get(row.task_id)
        if task is None:
            _lifecycle_fail(f"{split} rows contain a foreign task ID")
        if (
            row.truth != task.future_values
            or row.history != task.history_values
            or row.profile.history_length != len(task.history_values)
            or row.profile.horizon != task.prediction_length
            or row.profile.frequency != task.frequency
        ):
            _lifecycle_fail(f"{split} row identity or label projection drifted")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _lifecycle_fail(f"{split} rows contain duplicate candidate/task IDs")
        seen_keys.add(key)
        normalized = _canonical_identity(row.candidate_name)
        prior_name = normalized_candidates.setdefault(normalized, row.candidate_name)
        if prior_name != row.candidate_name:
            _lifecycle_fail(f"{split} rows contain duplicate candidate identities")
        candidate_tasks.setdefault(row.candidate_name, set()).add(row.task_id)
    universe = set(by_task)
    if any(candidate_ids != universe for candidate_ids in candidate_tasks.values()):
        _lifecycle_fail(f"every {split} candidate must cover the exact task universe")
    return cast(tuple[ChampionTaskRow, ...], snapshot)


def _call_row_provider(
    provider: Callable[[tuple[Task, ...], str], object],
    tasks: tuple[Task, ...],
    split: Literal["build", "calibration", "dev"],
    *,
    protected: tuple[object, ...],
) -> tuple[ChampionTaskRow, ...]:
    if not callable(provider):
        _lifecycle_fail("row provider must be callable")

    def fingerprint_state() -> str:
        def safe_value(value: object) -> object:
            if type(value) is ChampionCheckpoint:
                return cast(ChampionCheckpoint, value).to_payload()
            if type(value) is ChampionRunManifest:
                return cast(ChampionRunManifest, value).to_payload()
            if type(value) is ProposerEvidence:
                return cast(ProposerEvidence, value).to_payload()
            if type(value) in {tuple, list}:
                return tuple(
                    safe_value(item)
                    for item in cast(tuple[object, ...] | list[object], value)
                )
            if type(value) is dict:
                return {
                    cast(str, key): safe_value(item)
                    for key, item in cast(dict[object, object], value).items()
                }
            return value

        return champion_fingerprint(safe_value((tasks, protected)))

    protected_state = _capture_object_graph((tasks, protected))
    protected_sha256 = fingerprint_state()
    callback_error: Exception | None = None
    supplied: object = None
    changed = False
    try:
        try:
            supplied = provider(tasks, split)
        except Exception as error:
            callback_error = error
        finally:
            try:
                changed = fingerprint_state() != protected_sha256
            except Exception:
                changed = True
    finally:
        _restore_object_graph(protected_state)
    if changed:
        _lifecycle_fail(f"{split} row provider mutated bound lifecycle inputs")
    if callback_error is not None:
        raise ChampionLifecycleError(f"{split} row provider failed") from callback_error
    return _validated_stage_rows(supplied, tasks=tasks, split=split)


@dataclass(frozen=True)
class _StageEvaluationState:
    policy: FittedChampionPolicy
    parent_score: ChampionScore
    child_score: ChampionScore | None
    comparison: ChampionComparison | None
    evidence_fingerprint: str


def _evaluate_lifecycle_stage(
    parent: ChampionRelease,
    policies: tuple[FittedChampionPolicy, ...],
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    *,
    split: Literal["calibration", "dev"],
    gate: ChampionGateConfig,
) -> tuple[_StageEvaluationState, ...]:
    parent_recipe, parent_policy = _validated_parent(parent)
    parent_name, materialized_parent = _parent_rows(
        parent_recipe, parent_policy, rows, task_ids
    )
    try:
        parent_score = score_policy(materialized_parent, parent_name)
    except (ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionLifecycleError(
            f"trusted {split} scoring rejected the exact Parent"
        ) from error
    states: list[_StageEvaluationState] = []
    for policy in policies:
        policy_sha256 = champion_fingerprint(policy)
        child_name = f"{split}_child_{policy_sha256}"
        child_rows = _materialize_policy(
            policy,
            child_name,
            rows,
            task_ids,
            split=split,
        )
        evidence_fingerprint = champion_fingerprint(
            {
                "input_rows": rows,
                "parent_rows": materialized_parent,
                "child_rows": child_rows,
                "gate": gate,
                "policy": policy,
                "split": split,
            }
        )
        try:
            child_score = score_policy(child_rows, child_name)
            comparison = compare_champion(parent_score, child_score, gate)
        except (ChampionEvidenceError, TypeError, ValueError):
            child_score = None
            comparison = None
        states.append(
            _StageEvaluationState(
                policy=policy,
                parent_score=parent_score,
                child_score=child_score,
                comparison=comparison,
                evidence_fingerprint=evidence_fingerprint,
            )
        )
    return tuple(states)


def _stage_report(
    manifest: ChampionRunManifest,
    parent: ChampionRelease,
    split: Literal["calibration", "dev"],
    states: tuple[_StageEvaluationState, ...],
) -> ChampionEvaluationReport:
    return ChampionEvaluationReport(
        schema_version=1,
        input_fingerprint=manifest.input_fingerprint,
        split=split,
        parent_sha256=champion_fingerprint(parent),
        candidates=tuple(
            ChampionStageCandidateReport(
                policy_sha256=champion_fingerprint(state.policy),
                evidence_fingerprint=state.evidence_fingerprint,
                comparison=state.comparison,
                invalid_reason=(
                    "unscorable_child" if state.comparison is None else None
                ),
            )
            for state in states
        ),
    )


def _fresh_comparisons(
    states: tuple[_StageEvaluationState, ...],
    gate: ChampionGateConfig,
) -> tuple[tuple[FittedChampionPolicy, ChampionComparison], ...]:
    fresh: list[tuple[FittedChampionPolicy, ChampionComparison]] = []
    for state in states:
        if state.child_score is None:
            continue
        try:
            comparison = compare_champion(
                state.parent_score,
                state.child_score,
                gate,
            )
        except (ChampionEvidenceError, TypeError, ValueError) as error:
            raise ChampionLifecycleError(
                "bound lifecycle comparison could not be recomputed"
            ) from error
        if comparison.accepted:
            fresh.append((state.policy, comparison))
    return tuple(fresh)


def _lifecycle_rank(
    value: tuple[FittedChampionPolicy, ChampionComparison],
) -> tuple[float, float, float, str]:
    policy, comparison = value
    return (
        -comparison.joint_improvement,
        comparison.mean_delta_smae,
        comparison.mean_delta_srmse,
        champion_fingerprint(policy),
    )


def _combined_feedback(result: BuildEvolutionResult) -> ProposerEvidence:
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            item
            for generation in result.generations
            for item in generation.feedback.morphology
        ),
        comparisons=tuple(
            item
            for generation in result.generations
            for item in generation.feedback.comparisons
        ),
        invalid_attempts=tuple(
            sorted(
                (
                    item
                    for generation in result.generations
                    for item in generation.feedback.invalid_attempts
                ),
                key=lambda item: (item.structure_sha256, item.reason_code),
            )
        ),
    )
    return feedback


class ChampionEvolutionController:
    """Run the formal one-shot Build, Calibration, and Dev lifecycle."""

    def __init__(
        self,
        *,
        manifest: ChampionRunManifest,
        config: ChampionEvolutionConfig,
        proposer: Callable[[object, ProposerEvidence], object],
        row_provider: Callable[[tuple[Task, ...], str], object],
        artifact_store: ChampionArtifactStore,
    ) -> None:
        if type(manifest) is not ChampionRunManifest:
            _lifecycle_fail("manifest must be an exact ChampionRunManifest")
        ChampionRunManifest.__post_init__(manifest)
        if type(config) is not ChampionEvolutionConfig:
            _lifecycle_fail("config must be an exact ChampionEvolutionConfig")
        ChampionEvolutionConfig.__post_init__(config)
        if type(artifact_store) is not ChampionArtifactStore:
            _lifecycle_fail("artifact store must be an exact ChampionArtifactStore")
        if not callable(proposer) or not callable(row_provider):
            _lifecycle_fail("lifecycle callbacks must be callable")
        self.manifest = manifest
        self.config = config
        self.proposer = proposer
        self.row_provider = row_provider
        self.artifact_store = artifact_store

    def _validate_bindings(
        self,
        parent: ChampionRelease,
        train_tasks: object,
    ) -> tuple[tuple[Task, ...], TrainPartitions]:
        if type(parent) is not ChampionRelease:
            _lifecycle_fail("formal evolution Parent must be an exact ChampionRelease")
        _validated_parent(parent)
        ChampionRunManifest.__post_init__(self.manifest)
        ChampionEvolutionConfig.__post_init__(self.config)
        if (
            self.config.build_size != 64
            or self.config.calibration_size != 16
            or self.config.screen_sizes != (8, 32, 64)
        ):
            _lifecycle_fail("formal lifecycle requires the exact 64/16 schedule")
        if self.config.fingerprint != self.manifest.schedule_fingerprint:
            _lifecycle_fail("schedule fingerprint drifted from the run manifest")
        if (
            self.config.candidate_minimum_gain
            != self.manifest.candidate_minimum_gain
            or self.config.research_target_gain
            != self.manifest.research_target_gain
        ):
            _lifecycle_fail("threshold inputs drifted from the run manifest")
        if (
            parent.metric_policy_fingerprint
            != self.manifest.metric_policy_fingerprint
        ):
            _lifecycle_fail("metric policy drifted from the exact Parent")
        dictionary = dict(self.manifest.dictionary_hashes)
        if any(dictionary.get(name) != digest for name, digest in parent.source_hashes):
            _lifecycle_fail("Dictionary sources drifted from the exact Parent")
        train = _validated_lifecycle_tasks(
            train_tasks,
            label="Train",
            expected_membership=self.manifest.train_tasks,
        )
        parts = partition_train_tasks(
            train,
            build_size=64,
            calibration_size=16,
            seed=self.manifest.partition_seed,
        )
        build_membership = tuple(
            (task.task_id, task.entity_name) for task in parts.build
        )
        calibration_membership = tuple(
            (task.task_id, task.entity_name) for task in parts.calibration
        )
        if (
            build_membership != self.manifest.build_tasks
            or calibration_membership != self.manifest.calibration_tasks
        ):
            _lifecycle_fail("internal split membership drifted from the run manifest")
        build_ids = tuple(task.task_id for task in parts.build)
        if self.config.build_task_ids != build_ids:
            _lifecycle_fail("Build task membership drifted from the schedule")
        if (
            not self.config.screen_task_ids
            or self.config.screen_task_ids[-1] != build_ids
        ):
            _lifecycle_fail("screen membership drifted from the Build partition")
        return train, parts

    def _require_checkpoint(
        self,
        checkpoint: ChampionCheckpoint,
        parent: ChampionRelease,
    ) -> None:
        if checkpoint.input_fingerprint != self.manifest.input_fingerprint:
            _lifecycle_fail("checkpoint belongs to a stale or foreign run")
        expected_stage = f"build_generation_{self.config.generations}"
        if checkpoint.completed_stage != expected_stage:
            _lifecycle_fail("checkpoint completed stage is stale or foreign")
        if champion_fingerprint(checkpoint.active_parent) != champion_fingerprint(parent):
            _lifecycle_fail("checkpoint active Parent is stale or foreign")

    def _require_durable_state(
        self,
        checkpoint: ChampionCheckpoint,
        parent: ChampionRelease,
    ) -> None:
        self.artifact_store.bind_manifest(self.manifest)
        self.artifact_store.ensure_release(parent)
        stored = self.artifact_store.load_checkpoint()
        if stored is None:
            _lifecycle_fail("durable checkpoint disappeared before a boundary")
        self._require_checkpoint(stored, parent)
        if canonical_json_bytes(stored.to_payload()) != canonical_json_bytes(
            checkpoint.to_payload()
        ):
            _lifecycle_fail("durable checkpoint drifted before a lifecycle boundary")

    def evolve(
        self,
        parent: ChampionRelease,
        train_tasks: tuple[Task, ...] | list[Task],
        dev_tasks: object,
    ) -> ChampionEvolutionOutcome:
        """Run one formal lifecycle; Dev stays unopened until fresh Calibration acceptance."""
        parent_sha256 = champion_fingerprint(parent)
        config_sha256 = self.config.fingerprint
        manifest_sha256 = self.manifest.input_fingerprint
        _, parts = self._validate_bindings(parent, train_tasks)
        self.artifact_store.bind_manifest(self.manifest)
        if self.artifact_store.report_exists("calibration") or self.artifact_store.report_exists(
            "dev"
        ):
            _lifecycle_fail("one-shot lifecycle report cannot be overwritten or replayed")
        self.artifact_store.ensure_release(parent)
        checkpoint = self.artifact_store.load_checkpoint()
        build_result: BuildEvolutionResult | None = None
        if checkpoint is None:
            build_rows = _call_row_provider(
                self.row_provider,
                parts.build,
                "build",
                protected=(parent, self.config, self.manifest),
            )
            try:
                build_result = run_build_evolution(
                    parent,
                    build_rows,
                    self.proposer,
                    self.config,
                )
            except ChampionControllerError:
                raise
            except Exception as error:
                raise ChampionLifecycleError("Build evolution failed") from error
            checkpoint = ChampionCheckpoint(
                schema_version=1,
                input_fingerprint=self.manifest.input_fingerprint,
                completed_stage=f"build_generation_{self.config.generations}",
                active_parent=parent,
                proposal_archive=build_result.shortlist,
                sanitized_feedback=_combined_feedback(build_result),
            )
            self.artifact_store.write_checkpoint(checkpoint)
        else:
            self._require_checkpoint(checkpoint, parent)

        self._require_checkpoint(checkpoint, parent)
        self._require_durable_state(checkpoint, parent)
        if not checkpoint.proposal_archive:
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=None,
                dev_report=None,
            )
        if (
            champion_fingerprint(parent) != parent_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted before Calibration")
        calibration_rows = _call_row_provider(
            self.row_provider,
            parts.calibration,
            "calibration",
            protected=(parent, self.config, self.manifest, checkpoint),
        )
        calibration_states = _evaluate_lifecycle_stage(
            parent,
            checkpoint.proposal_archive,
            calibration_rows,
            tuple(task.task_id for task in parts.calibration),
            split="calibration",
            gate=self.config.gate_config,
        )
        calibration_report = _stage_report(
            self.manifest,
            parent,
            "calibration",
            calibration_states,
        )
        self.artifact_store.write_report(calibration_report)
        self._require_durable_state(checkpoint, parent)
        if (
            champion_fingerprint(parent) != parent_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted at Calibration boundary")
        calibration_accepted = _fresh_comparisons(
            calibration_states, self.config.gate_config
        )
        if not calibration_accepted:
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=calibration_report,
                dev_report=None,
            )
        train_winner, _ = min(calibration_accepted, key=_lifecycle_rank)

        self._require_durable_state(checkpoint, parent)
        dev = _validated_lifecycle_tasks(
            dev_tasks,
            label="Dev",
            expected_membership=self.manifest.dev_tasks,
        )
        dev_rows = _call_row_provider(
            self.row_provider,
            dev,
            "dev",
            protected=(parent, train_winner, self.config, self.manifest, checkpoint),
        )
        dev_states = _evaluate_lifecycle_stage(
            parent,
            (train_winner,),
            dev_rows,
            tuple(task.task_id for task in dev),
            split="dev",
            gate=self.config.gate_config,
        )
        dev_report = _stage_report(self.manifest, parent, "dev", dev_states)
        self.artifact_store.write_report(dev_report)
        self._require_durable_state(checkpoint, parent)
        if (
            champion_fingerprint(parent) != parent_sha256
            or champion_fingerprint(train_winner)
            != dev_report.candidates[0].policy_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted at Dev boundary")
        dev_accepted = _fresh_comparisons(dev_states, self.config.gate_config)
        if not dev_accepted:
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=calibration_report,
                dev_report=dev_report,
            )
        if _canonical_identity(train_winner.recipe.name) in {
            _canonical_identity(item) for item in parent.lineage
        }:
            _lifecycle_fail("accepted release lineage contains a replayed policy ID")
        release = ChampionRelease(
            policy=train_winner,
            source_hashes=self.manifest.dictionary_hashes,
            metric_policy_fingerprint=self.manifest.metric_policy_fingerprint,
            lineage=parent.lineage + (train_winner.recipe.name,),
        )
        self.artifact_store.publish_release(release)
        return ChampionEvolutionOutcome(
            manifest=self.manifest,
            checkpoint=checkpoint,
            release=release,
            build_result=build_result,
            calibration_report=calibration_report,
            dev_report=dev_report,
        )


__all__ = [
    "BuildAttempt",
    "BuildEvolutionResult",
    "BuildGeneration",
    "ChampionArtifactStore",
    "ChampionCheckpoint",
    "ChampionControllerError",
    "ChampionEvaluationReport",
    "ChampionEvolutionController",
    "ChampionEvolutionConfig",
    "ChampionEvolutionOutcome",
    "ChampionLifecycleError",
    "ChampionRunManifest",
    "ChampionStageCandidateReport",
    "TrainPartitions",
    "canonical_release_bytes",
    "partition_train_tasks",
    "run_build_evolution",
]
