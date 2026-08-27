"""Train-only, typed evolution for the two-stage Retrieval subsystem.

The engine deliberately owns scheduling and audit state, while a trusted evaluator
owns inference and scoring.  Mutation receives only the current typed Genome and a
fixed scope contract; evaluator outputs never cross back into the mutation prompt.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from common.llm import (
    JsonExtractionError,
    LLMClient,
    TransientLLMError,
    parse_json_object,
)
from evolving_loop.data import ContextTask

from .policy import RetrievalGenome, RetrievalPolicyError
from .skill_library import RetrievalSkillLibrary


RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION = 1
RetrievalChildScope = Literal["A", "B", "C"]
CHILD_SCOPES: tuple[RetrievalChildScope, ...] = ("A", "B", "C")

_SCOPE_ALIASES: dict[str, RetrievalChildScope] = {
    "a": "A",
    "round1": "A",
    "round_1": "A",
    "b": "B",
    "chain": "B",
    "chains": "B",
    "c": "C",
    "round2": "C",
    "round_2": "C",
}
_PRIMARY_SCOPE_FIELDS: dict[RetrievalChildScope, frozenset[str]] = {
    "A": frozenset(
        {"round1_prompt", "round1_strategy", "max_selected_documents"}
    ),
    "B": frozenset(
        {
            "max_evidence_chains",
            "max_citations_per_chain",
            "require_counterevidence_search",
            "require_target_match",
            "require_temporal_overlap",
        }
    ),
    "C": frozenset(
        {"round2_prompt", "round2_strategy", "second_round_trigger"}
    ),
}
_SCOPE_SKILL_STAGE: dict[RetrievalChildScope, str] = {
    "A": "round1",
    "B": "both",
    "C": "round2",
}


class RetrievalEvolutionError(ValueError):
    """Raised when the Retrieval evolution protocol cannot continue safely."""


class RetrievalCheckpointError(RetrievalEvolutionError):
    """Raised when a checkpoint is malformed or belongs to different science."""


class RetrievalEvaluator(Protocol):
    """Trusted evaluation boundary consumed by :class:`RetrievalEvolutionEngine`."""

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        **kwargs: object,
    ) -> "RetrievalEvaluation": ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _callable_identity(value: object | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalEvolutionError(f"{field_name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RetrievalEvolutionError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class RetrievalEvaluation:
    """The complete trusted Retrieval and final-system metric vector."""

    version: str
    task_count: int
    mean_final_smae: float
    mean_final_srmse: float
    mean_contextual_oracle_smae: float
    mean_contextual_oracle_srmse: float
    p90_smae: float
    p95_smae: float
    supporting_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    invalid_count: int
    catastrophic_count: int
    task_traces: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise RetrievalEvolutionError("evaluation version must be non-empty")
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise RetrievalEvolutionError("evaluation task_count must be positive")
        for name in (
            "mean_final_smae",
            "mean_final_srmse",
            "mean_contextual_oracle_smae",
            "mean_contextual_oracle_srmse",
            "p90_smae",
            "p95_smae",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise RetrievalEvolutionError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        for name in (
            "supporting_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise RetrievalEvolutionError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        for name in ("invalid_count", "catastrophic_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RetrievalEvolutionError(f"{name} must be a non-negative integer")
        if not isinstance(self.task_traces, tuple) or any(
            not isinstance(item, dict) for item in self.task_traces
        ):
            raise RetrievalEvolutionError("task_traces must be a tuple of trace objects")

    def gate_failures(
        self,
        parent: "RetrievalEvaluation",
        tolerance: float,
        *,
        require_strict_contextual_gain: bool = True,
    ) -> tuple[str, ...]:
        tolerance = _finite(tolerance, "tolerance")
        if tolerance < 0:
            raise RetrievalEvolutionError("tolerance cannot be negative")
        failures: list[str] = []
        if self.task_count != parent.task_count:
            failures.append("task_count")
        if (
            self.mean_contextual_oracle_smae
            > parent.mean_contextual_oracle_smae + tolerance
        ):
            failures.append("mean_contextual_oracle_smae")
        if (
            self.mean_contextual_oracle_srmse
            > parent.mean_contextual_oracle_srmse + tolerance
        ):
            failures.append("mean_contextual_oracle_srmse")
        if require_strict_contextual_gain and not (
            self.mean_contextual_oracle_smae
            < parent.mean_contextual_oracle_smae - tolerance
            or self.mean_contextual_oracle_srmse
            < parent.mean_contextual_oracle_srmse - tolerance
        ):
            failures.append("strict_contextual_gain")
        if self.mean_final_smae > parent.mean_final_smae + tolerance:
            failures.append("mean_final_smae")
        if self.mean_final_srmse > parent.mean_final_srmse + tolerance:
            failures.append("mean_final_srmse")
        if self.p90_smae > parent.p90_smae + tolerance:
            failures.append("p90_smae")
        if self.p95_smae > parent.p95_smae + tolerance:
            failures.append("p95_smae")
        if self.supporting_recall < parent.supporting_recall - 0.02:
            failures.append("supporting_recall")
        if self.distractor_avoidance < parent.distractor_avoidance - 0.02:
            failures.append("distractor_avoidance")
        if self.exact_quote_validity != 1.0:
            failures.append("exact_quote_validity")
        if self.catastrophic_count > parent.catastrophic_count:
            failures.append("catastrophic_count")
        if self.invalid_count > parent.invalid_count:
            failures.append("invalid_count")
        return tuple(failures)

    def dev_accepts(
        self, parent: "RetrievalEvaluation", tolerance: float
    ) -> bool:
        return not self.gate_failures(
            parent,
            tolerance,
            require_strict_contextual_gain=True,
        )

    def pareto_safe(
        self, parent: "RetrievalEvaluation", tolerance: float
    ) -> bool:
        return not self.gate_failures(
            parent,
            tolerance,
            require_strict_contextual_gain=False,
        )

    @property
    def rank_key(self) -> tuple[float, ...]:
        """Lower is better; safety eligibility is checked separately."""
        return (
            self.mean_contextual_oracle_srmse,
            self.mean_contextual_oracle_smae,
            self.mean_final_srmse,
            self.mean_final_smae,
            self.p95_smae,
            self.p90_smae,
            -self.supporting_recall,
            -self.distractor_avoidance,
            -self.complete_chain_rate,
            float(self.catastrophic_count),
            float(self.invalid_count),
        )

    def summary(self) -> dict[str, object]:
        payload = self.to_payload()
        payload.pop("task_traces")
        return payload

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "task_count": self.task_count,
            "mean_final_smae": self.mean_final_smae,
            "mean_final_srmse": self.mean_final_srmse,
            "mean_contextual_oracle_smae": self.mean_contextual_oracle_smae,
            "mean_contextual_oracle_srmse": self.mean_contextual_oracle_srmse,
            "p90_smae": self.p90_smae,
            "p95_smae": self.p95_smae,
            "supporting_recall": self.supporting_recall,
            "distractor_avoidance": self.distractor_avoidance,
            "exact_quote_validity": self.exact_quote_validity,
            "complete_chain_rate": self.complete_chain_rate,
            "invalid_count": self.invalid_count,
            "catastrophic_count": self.catastrophic_count,
            "task_traces": list(self.task_traces),
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalEvaluation":
        fields = {
            "version",
            "task_count",
            "mean_final_smae",
            "mean_final_srmse",
            "mean_contextual_oracle_smae",
            "mean_contextual_oracle_srmse",
            "p90_smae",
            "p95_smae",
            "supporting_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
            "invalid_count",
            "catastrophic_count",
            "task_traces",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise RetrievalCheckpointError("invalid checkpoint evaluation")
        traces = raw["task_traces"]
        if not isinstance(traces, list) or any(not isinstance(item, dict) for item in traces):
            raise RetrievalCheckpointError("invalid checkpoint task traces")
        return cls(
            **{
                **{key: raw[key] for key in fields if key != "task_traces"},
                "task_traces": tuple(dict(item) for item in traces),
            }
        )


def combine_retrieval_evaluations(
    version: str, evaluations: Sequence[RetrievalEvaluation]
) -> RetrievalEvaluation:
    """Combine disjoint task batches without blending count-based safety fields."""
    if not evaluations:
        raise RetrievalEvolutionError("cannot combine an empty evaluation vector")
    total = sum(item.task_count for item in evaluations)

    def weighted(field_name: str) -> float:
        return sum(
            getattr(item, field_name) * item.task_count for item in evaluations
        ) / total

    traces = tuple(trace for item in evaluations for trace in item.task_traces)
    trace_smae = [
        float(trace["final_smae"])
        for trace in traces
        if isinstance(trace.get("final_smae"), (int, float))
        and not isinstance(trace.get("final_smae"), bool)
    ]

    def percentile(values: list[float], probability: float, fallback: float) -> float:
        if not values:
            return fallback
        ordered = sorted(values)
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    return RetrievalEvaluation(
        version=version,
        task_count=total,
        mean_final_smae=weighted("mean_final_smae"),
        mean_final_srmse=weighted("mean_final_srmse"),
        mean_contextual_oracle_smae=weighted(
            "mean_contextual_oracle_smae"
        ),
        mean_contextual_oracle_srmse=weighted(
            "mean_contextual_oracle_srmse"
        ),
        p90_smae=percentile(
            trace_smae,
            0.90,
            max(item.p90_smae for item in evaluations),
        ),
        p95_smae=percentile(
            trace_smae,
            0.95,
            max(item.p95_smae for item in evaluations),
        ),
        supporting_recall=weighted("supporting_recall"),
        distractor_avoidance=weighted("distractor_avoidance"),
        exact_quote_validity=weighted("exact_quote_validity"),
        complete_chain_rate=weighted("complete_chain_rate"),
        invalid_count=sum(item.invalid_count for item in evaluations),
        catastrophic_count=sum(item.catastrophic_count for item in evaluations),
        task_traces=traces,
    )


def _library_hash(library: RetrievalSkillLibrary | None) -> str:
    payload = [] if library is None else [skill.to_payload() for skill in library.all()]
    return _digest(payload)


@dataclass(frozen=True)
class RetrievalInferenceCacheKey:
    """All executable inputs that may change one task's trusted evaluation."""

    task_id: str
    genome_sha256: str
    skill_library_sha256: str
    verifier_sha256: str
    evaluator_sha256: str

    def digest(self) -> str:
        return _digest(
            {
                "task_id": self.task_id,
                "genome_sha256": self.genome_sha256,
                "skill_library_sha256": self.skill_library_sha256,
                "verifier_sha256": self.verifier_sha256,
                "evaluator_sha256": self.evaluator_sha256,
            }
        )


def build_inference_cache_key(
    task: ContextTask,
    genome: RetrievalGenome,
    skill_library: RetrievalSkillLibrary | None,
    *,
    verifier_hash: str,
    evaluator_hash: str,
) -> RetrievalInferenceCacheKey:
    return RetrievalInferenceCacheKey(
        task_id=task.numeric.task_id,
        genome_sha256=genome.fingerprint(),
        skill_library_sha256=_library_hash(skill_library),
        verifier_sha256=str(verifier_hash),
        evaluator_sha256=str(evaluator_hash),
    )


def _normalize_scope(scope: str) -> RetrievalChildScope | None:
    if scope in CHILD_SCOPES:
        return cast(RetrievalChildScope, scope)
    return _SCOPE_ALIASES.get(str(scope).strip().lower())


def parse_scoped_child(
    parent: RetrievalGenome,
    proposal: Mapping[str, object],
    *,
    scope: str,
    skill_library: RetrievalSkillLibrary | None = None,
) -> RetrievalGenome | None:
    """Parse a complete Genome and reject any field outside one fixed scope.

    Skill activation is stage-owned as well: A may change Round 1 IDs, B only
    cross-stage IDs, and C only Round 2 IDs.  Unknown Skill identity therefore
    fails closed instead of being inferred from a name supplied by the model.
    """
    normalized = _normalize_scope(scope)
    if normalized is None or not isinstance(parent, RetrievalGenome):
        return None
    try:
        child = RetrievalGenome.from_payload(proposal)
    except (RetrievalPolicyError, TypeError, ValueError):
        return None
    if (
        child.schema_version != parent.schema_version
        or child.parent != parent.version
        or child.version == parent.version
    ):
        return None
    ignored = {"version", "parent"}
    changed = {
        field_name
        for field_name in parent.to_payload()
        if field_name not in ignored
        and getattr(child, field_name) != getattr(parent, field_name)
    }
    allowed = _PRIMARY_SCOPE_FIELDS[normalized] | {"active_skill_ids"}
    if not changed or not changed.issubset(allowed):
        return None
    if "active_skill_ids" in changed:
        if skill_library is None:
            return None
        changed_skill_ids = set(parent.active_skill_ids).symmetric_difference(
            child.active_skill_ids
        )
        owned_stage = _SCOPE_SKILL_STAGE[normalized]
        for skill_id in changed_skill_ids:
            skill = skill_library.get_by_id(skill_id)
            if skill is None or skill.stage != owned_stage:
                return None
    return child


@dataclass(frozen=True)
class RetrievalEvolutionConfig:
    generations: int = 3
    screen_tasks: int = 8
    promote: int = 2
    train_folds: int = 5
    tolerance: float = 1e-12
    random_seed: int = 7
    transient_retries: int = 2
    checkpoint_path: str | Path | None = None
    resume: bool = True
    dataset_split_hash: str | None = None
    verifier_hash: str | None = None
    evaluator_hash: str | None = None
    metric_hash: str | None = None
    mutation_model_hash: str | None = None
    metric_cap: float = 5.0

    def __post_init__(self) -> None:
        for name, lower in (
            ("generations", 1),
            ("train_folds", 2),
            ("transient_retries", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise RetrievalEvolutionError(f"invalid {name}")
        if self.generations > 333:
            raise RetrievalEvolutionError("generations exceed the vNNN namespace")
        if self.screen_tasks != 8:
            raise RetrievalEvolutionError("Retrieval evolution screens exactly eight Train tasks")
        if (
            isinstance(self.promote, bool)
            or not isinstance(self.promote, int)
            or not 0 <= self.promote <= 2
        ):
            raise RetrievalEvolutionError("Retrieval evolution promotes at most two children")
        tolerance = _finite(self.tolerance, "tolerance")
        cap = _finite(self.metric_cap, "metric_cap")
        if tolerance < 0 or cap <= 0:
            raise RetrievalEvolutionError("invalid tolerance or metric_cap")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise RetrievalEvolutionError("invalid random_seed")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "metric_cap", cap)
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True)
class RetrievalGenerationTrace:
    generation: int
    parent_version: str
    parent_fingerprint: str
    child_versions: tuple[str, ...]
    child_fingerprints: tuple[str, ...]
    child_scopes: tuple[RetrievalChildScope, ...]
    screen_task_ids: tuple[str, ...]
    fold_entities: tuple[tuple[str, ...], ...]
    promoted_fingerprints: tuple[str, ...]
    train_winner_version: str
    train_winner_fingerprint: str
    rejection_reasons: dict[str, str]
    screen_summaries: dict[str, dict[str, object]]
    train_summaries: dict[str, dict[str, object]]

    def __post_init__(self) -> None:
        if (
            len(self.child_versions) != 3
            or len(self.child_fingerprints) != 3
            or self.child_scopes != CHILD_SCOPES
            or len(set(self.child_versions)) != 3
            or len(set(self.child_fingerprints)) != 3
        ):
            raise RetrievalEvolutionError(
                "each generation must bind exactly the A/B/C child vector"
            )
        if len(self.promoted_fingerprints) > 2 or not set(
            self.promoted_fingerprints
        ).issubset(self.child_fingerprints):
            raise RetrievalEvolutionError("invalid promoted Child fingerprint vector")

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "parent_version": self.parent_version,
            "parent_fingerprint": self.parent_fingerprint,
            "child_versions": list(self.child_versions),
            "child_fingerprints": list(self.child_fingerprints),
            "child_scopes": list(self.child_scopes),
            "screen_task_ids": list(self.screen_task_ids),
            "fold_entities": [list(item) for item in self.fold_entities],
            "promoted_fingerprints": list(self.promoted_fingerprints),
            "train_winner_version": self.train_winner_version,
            "train_winner_fingerprint": self.train_winner_fingerprint,
            "rejection_reasons": dict(self.rejection_reasons),
            "screen_summaries": self.screen_summaries,
            "train_summaries": self.train_summaries,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalGenerationTrace":
        if not isinstance(raw, Mapping):
            raise RetrievalCheckpointError("invalid generation trace")
        expected = {
            "generation",
            "parent_version",
            "parent_fingerprint",
            "child_versions",
            "child_fingerprints",
            "child_scopes",
            "screen_task_ids",
            "fold_entities",
            "promoted_fingerprints",
            "train_winner_version",
            "train_winner_fingerprint",
            "rejection_reasons",
            "screen_summaries",
            "train_summaries",
        }
        if set(raw) != expected:
            raise RetrievalCheckpointError("invalid generation trace schema")
        try:
            scopes = tuple(raw["child_scopes"])
            if any(scope not in CHILD_SCOPES for scope in scopes):
                raise ValueError
            return cls(
                generation=int(raw["generation"]),
                parent_version=str(raw["parent_version"]),
                parent_fingerprint=str(raw["parent_fingerprint"]),
                child_versions=tuple(str(item) for item in raw["child_versions"]),
                child_fingerprints=tuple(
                    str(item) for item in raw["child_fingerprints"]
                ),
                child_scopes=cast(tuple[RetrievalChildScope, ...], scopes),
                screen_task_ids=tuple(str(item) for item in raw["screen_task_ids"]),
                fold_entities=tuple(
                    tuple(str(entity) for entity in item)
                    for item in raw["fold_entities"]
                ),
                promoted_fingerprints=tuple(
                    str(item) for item in raw["promoted_fingerprints"]
                ),
                train_winner_version=str(raw["train_winner_version"]),
                train_winner_fingerprint=str(raw["train_winner_fingerprint"]),
                rejection_reasons={
                    str(key): str(value)
                    for key, value in raw["rejection_reasons"].items()
                },
                screen_summaries={
                    str(key): dict(value)
                    for key, value in raw["screen_summaries"].items()
                },
                train_summaries={
                    str(key): dict(value)
                    for key, value in raw["train_summaries"].items()
                },
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid generation trace values") from error


@dataclass(frozen=True)
class RetrievalEvolutionResult:
    original_parent: RetrievalGenome
    train_winner: RetrievalGenome
    selected_genome: RetrievalGenome
    accepted: bool
    acceptance_reasons: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    parent_dev: RetrievalEvaluation | None
    child_dev: RetrievalEvaluation | None
    generations: tuple[RetrievalGenerationTrace, ...]
    trace: tuple[dict[str, object], ...]
    release_genome: RetrievalGenome | None
    release_published: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "original_parent": self.original_parent.to_payload(),
            "train_winner": self.train_winner.to_payload(),
            "selected_genome": self.selected_genome.to_payload(),
            "accepted": self.accepted,
            "acceptance_reasons": list(self.acceptance_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "parent_dev": self.parent_dev.to_payload() if self.parent_dev else None,
            "child_dev": self.child_dev.to_payload() if self.child_dev else None,
            "generations": [item.to_payload() for item in self.generations],
            "trace": list(self.trace),
            "release_genome": (
                self.release_genome.to_payload() if self.release_genome else None
            ),
            "release_published": self.release_published,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalEvolutionResult":
        expected = {
            "original_parent",
            "train_winner",
            "selected_genome",
            "accepted",
            "acceptance_reasons",
            "rejection_reasons",
            "parent_dev",
            "child_dev",
            "generations",
            "trace",
            "release_genome",
            "release_published",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise RetrievalCheckpointError("invalid completed result")
        if not isinstance(raw["accepted"], bool) or not isinstance(
            raw["release_published"], bool
        ):
            raise RetrievalCheckpointError("invalid completed result boolean fields")
        try:
            parent_dev = (
                None
                if raw["parent_dev"] is None
                else RetrievalEvaluation.from_payload(raw["parent_dev"])
            )
            child_dev = (
                None
                if raw["child_dev"] is None
                else RetrievalEvaluation.from_payload(raw["child_dev"])
            )
            release_genome = (
                None
                if raw["release_genome"] is None
                else RetrievalGenome.from_payload(raw["release_genome"])
            )
            return cls(
                original_parent=RetrievalGenome.from_payload(raw["original_parent"]),
                train_winner=RetrievalGenome.from_payload(raw["train_winner"]),
                selected_genome=RetrievalGenome.from_payload(raw["selected_genome"]),
                accepted=raw["accepted"],
                acceptance_reasons=tuple(
                    str(item) for item in raw["acceptance_reasons"]
                ),
                rejection_reasons=tuple(
                    str(item) for item in raw["rejection_reasons"]
                ),
                parent_dev=parent_dev,
                child_dev=child_dev,
                generations=tuple(
                    RetrievalGenerationTrace.from_payload(item)
                    for item in raw["generations"]
                ),
                trace=tuple(dict(item) for item in raw["trace"]),
                release_genome=release_genome,
                release_published=raw["release_published"],
            )
        except (TypeError, ValueError, RetrievalPolicyError) as error:
            raise RetrievalCheckpointError("invalid completed result values") from error


def _task_science_payload(task: ContextTask) -> dict[str, object]:
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


class RetrievalEvolutionEngine:
    """Evolve one Retrieval coordinate without consulting Dev until the end."""

    def __init__(
        self,
        mutation_llm: LLMClient,
        evaluator: RetrievalEvaluator | Callable[..., RetrievalEvaluation],
        config: RetrievalEvolutionConfig | None = None,
        *,
        skill_library: RetrievalSkillLibrary | None = None,
        harness_factory: Callable[..., object] | None = None,
    ) -> None:
        self.mutation_llm = mutation_llm
        self.evaluator = evaluator
        self.config = config or RetrievalEvolutionConfig()
        self.skill_library = skill_library
        self.harness_factory = harness_factory
        self._scientific_inputs: dict[str, object] = {}
        self._original_parent: RetrievalGenome | None = None
        self._current_parent: RetrievalGenome | None = None
        self._next_generation = 0
        self._generations: list[RetrievalGenerationTrace] = []
        self._trace: list[dict[str, object]] = []
        self._evaluation_cache: dict[str, RetrievalEvaluation] = {}
        self._cache_records: dict[str, dict[str, object]] = {}
        self._pending_children: dict[str, object] | None = None
        self._all_child_fingerprints: list[str] = []
        self._candidate_libraries: dict[str, RetrievalSkillLibrary | None] = {}

    def evolve(
        self,
        parent: RetrievalGenome,
        train_tasks: Sequence[ContextTask],
        dev_tasks: Sequence[ContextTask],
    ) -> RetrievalEvolutionResult:
        train = tuple(train_tasks)
        dev = tuple(dev_tasks)
        self._validate_inputs(parent, train, dev)
        self._scientific_inputs = self._science_signature(parent, train, dev)
        completed = self._load_checkpoint(parent)
        if completed is not None:
            return completed
        assert self._original_parent is not None
        assert self._current_parent is not None
        self._candidate_libraries.setdefault(
            self._original_parent.fingerprint(), self._clone_library(self.skill_library)
        )
        self._candidate_libraries.setdefault(
            self._current_parent.fingerprint(),
            self._candidate_libraries[self._original_parent.fingerprint()],
        )

        screen = train[: self.config.screen_tasks]
        remaining_folds = self._entity_folds(train[self.config.screen_tasks :])
        for generation in range(self._next_generation, self.config.generations):
            self._run_generation(generation, screen, remaining_folds)

        original = self._original_parent
        winner = self._current_parent
        parent_dev = self._evaluate_batch(
            original,
            dev,
            stage="parent_dev",
            readonly=True,
            library=self._readonly_library(original),
        )
        try:
            child_dev = self._evaluate_batch(
                winner,
                dev,
                stage="child_dev",
                readonly=True,
                library=self._readonly_library(winner),
            )
        except TransientLLMError:
            raise
        except Exception as error:
            child_dev = None
            rejection_reasons = (f"child_dev_failure:{type(error).__name__}:{error}",)
        else:
            rejection_reasons = child_dev.gate_failures(
                parent_dev,
                self.config.tolerance,
                require_strict_contextual_gain=True,
            )
        accepted = child_dev is not None and not rejection_reasons
        self._event(
            "dev_completed",
            original_parent=original.version,
            train_winner=winner.version,
            accepted=accepted,
            rejection_reasons=list(rejection_reasons),
        )
        self._event(
            "release_accepted" if accepted else "release_rejected",
            genome=winner.version if accepted else original.version,
            publication_deferred=True,
        )
        result = RetrievalEvolutionResult(
            original_parent=original,
            train_winner=winner,
            selected_genome=winner if accepted else original,
            accepted=accepted,
            acceptance_reasons=("all_dev_gates_passed",) if accepted else (),
            rejection_reasons=tuple(rejection_reasons),
            parent_dev=parent_dev,
            child_dev=child_dev,
            generations=tuple(self._generations),
            trace=tuple(self._trace),
            release_genome=winner if accepted else None,
            release_published=False,
        )
        self._save_checkpoint(status="complete", result=result)
        return result

    def _validate_inputs(
        self,
        parent: RetrievalGenome,
        train: tuple[ContextTask, ...],
        dev: tuple[ContextTask, ...],
    ) -> None:
        if not isinstance(parent, RetrievalGenome):
            raise RetrievalEvolutionError("parent must be a RetrievalGenome")
        if int(parent.version[1:]) + self.config.generations * 3 > 999:
            raise RetrievalEvolutionError(
                "configured generations exceed the remaining vNNN namespace"
            )
        if len(train) != 80 or len(dev) != 20:
            raise RetrievalEvolutionError(
                "Retrieval evolution requires exactly 80 Train and 20 Dev tasks"
            )
        if any(not isinstance(task, ContextTask) for task in (*train, *dev)):
            raise RetrievalEvolutionError("all evolution cases must be ContextTask records")
        train_ids = [task.numeric.task_id for task in train]
        dev_ids = [task.numeric.task_id for task in dev]
        if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
            raise RetrievalEvolutionError("task IDs must be unique within each split")
        if set(train_ids).intersection(dev_ids):
            raise RetrievalEvolutionError("Train and Dev task IDs must be disjoint")

    def _science_signature(
        self,
        parent: RetrievalGenome,
        train: tuple[ContextTask, ...],
        dev: tuple[ContextTask, ...],
    ) -> dict[str, object]:
        split_content_hash = _digest(
            {
                "train": [_task_science_payload(task) for task in train],
                "dev": [_task_science_payload(task) for task in dev],
            }
        )
        evaluator_type = type(self.evaluator)
        llm_type = type(self.mutation_llm)
        return {
            "dataset_split_hash": self.config.dataset_split_hash
            or split_content_hash,
            "dataset_content_hash": split_content_hash,
            "verifier_hash": self.config.verifier_hash
            or str(getattr(self.evaluator, "verifier_hash", "retrieval-verifier-v1")),
            "evaluator_hash": self.config.evaluator_hash
            or str(
                getattr(
                    self.evaluator,
                    "evaluator_hash",
                    f"{evaluator_type.__module__}.{evaluator_type.__qualname__}",
                )
            ),
            "metric_hash": self.config.metric_hash or "drcik-point-metrics-v1",
            "mutation_model_hash": self.config.mutation_model_hash
            or f"{llm_type.__module__}.{llm_type.__qualname__}",
            "metric_cap": self.config.metric_cap,
            "random_seed": self.config.random_seed,
            "generations": self.config.generations,
            "screen_tasks": self.config.screen_tasks,
            "promote": self.config.promote,
            "train_folds": self.config.train_folds,
            "tolerance": self.config.tolerance,
            "original_parent_fingerprint": parent.fingerprint(),
            "skill_library_hash": _library_hash(self.skill_library),
            "harness_factory": _callable_identity(self.harness_factory),
        }

    def _entity_folds(
        self, tasks: tuple[ContextTask, ...]
    ) -> tuple[tuple[ContextTask, ...], ...]:
        by_entity: dict[str, list[ContextTask]] = {}
        for task in tasks:
            by_entity.setdefault(task.numeric.entity_name, []).append(task)
        entities = sorted(by_entity)
        random.Random(self.config.random_seed).shuffle(entities)
        fold_count = min(self.config.train_folds, len(entities))
        if fold_count < 1:
            raise RetrievalEvolutionError("remaining Train cases cannot be empty")
        fold_entities: list[list[str]] = [[] for _ in range(fold_count)]
        for index, entity in enumerate(entities):
            fold_entities[index % fold_count].append(entity)
        return tuple(
            tuple(
                task
                for entity in entity_group
                for task in by_entity[entity]
            )
            for entity_group in fold_entities
        )

    def _run_generation(
        self,
        generation: int,
        screen: tuple[ContextTask, ...],
        remaining_folds: tuple[tuple[ContextTask, ...], ...],
    ) -> None:
        assert self._current_parent is not None
        parent = self._current_parent
        self._event(
            "generation_started",
            generation=generation,
            parent=parent.version,
            parent_fingerprint=parent.fingerprint(),
        )
        children, proposal_rejections = self._children_for_generation(
            generation, parent
        )
        parent_library = self._library_for(parent)
        parent_screen = self._evaluate_batch(
            parent,
            screen,
            stage=f"g{generation}_parent_screen_train",
            readonly=False,
            library=parent_library,
        )
        screen_evaluations: dict[str, RetrievalEvaluation] = {
            parent.version: parent_screen
        }
        rejection_reasons = dict(proposal_rejections)
        eligible: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
        for scope, child in children:
            child_library = self._library_for(child, source=parent_library)
            try:
                evaluation = self._evaluate_batch(
                    child,
                    screen,
                    stage=f"g{generation}_child_{scope}_screen_train",
                    readonly=False,
                    library=child_library,
                )
            except TransientLLMError:
                raise
            except Exception as error:
                rejection_reasons[child.version] = (
                    f"forecasting_failure:{type(error).__name__}:{error}"
                )
                self._event(
                    "screen_completed",
                    generation=generation,
                    child=child.version,
                    valid=False,
                    reason=rejection_reasons[child.version],
                )
                continue
            screen_evaluations[child.version] = evaluation
            failures = evaluation.gate_failures(
                parent_screen,
                self.config.tolerance,
                require_strict_contextual_gain=False,
            )
            if failures:
                rejection_reasons[child.version] = "screen_gate:" + ",".join(failures)
            else:
                eligible.append((scope, child))
            self._event(
                "screen_completed",
                generation=generation,
                child=child.version,
                valid=not failures,
                rejection_reasons=list(failures),
            )

        eligible.sort(
            key=lambda item: (
                screen_evaluations[item[1].version].rank_key,
                item[1].fingerprint(),
            )
        )
        promoted = eligible[: self.config.promote]
        promoted_versions = {child.version for _scope, child in promoted}
        for _scope, child in eligible[self.config.promote :]:
            rejection_reasons[child.version] = "screen_rank:not_promoted"

        parent_fold_evaluations = tuple(
            self._evaluate_batch(
                parent,
                fold,
                stage=f"g{generation}_parent_train_fold_{index}",
                readonly=False,
                library=parent_library,
            )
            for index, fold in enumerate(remaining_folds)
        )
        parent_train = combine_retrieval_evaluations(
            parent.version, (parent_screen, *parent_fold_evaluations)
        )
        train_evaluations: dict[str, RetrievalEvaluation] = {
            parent.version: parent_train
        }
        full_candidates: list[RetrievalGenome] = []
        for _scope, child in promoted:
            child_library = self._library_for(child)
            child_folds: list[RetrievalEvaluation] = []
            try:
                for index, fold in enumerate(remaining_folds):
                    child_folds.append(
                        self._evaluate_batch(
                            child,
                            fold,
                            stage=f"g{generation}_child_train_fold_{index}",
                            readonly=False,
                            library=child_library,
                        )
                    )
            except TransientLLMError:
                raise
            except Exception as error:
                rejection_reasons[child.version] = (
                    f"forecasting_failure:{type(error).__name__}:{error}"
                )
                continue
            evaluation = combine_retrieval_evaluations(
                child.version,
                (screen_evaluations[child.version], *child_folds),
            )
            train_evaluations[child.version] = evaluation
            fold_safe = all(
                candidate_fold.pareto_safe(parent_fold, self.config.tolerance)
                for candidate_fold, parent_fold in zip(
                    child_folds, parent_fold_evaluations, strict=True
                )
            )
            failures = evaluation.gate_failures(
                parent_train,
                self.config.tolerance,
                require_strict_contextual_gain=True,
            )
            if not fold_safe:
                rejection_reasons[child.version] = "train_fold_vector:not_pareto_safe"
            elif failures:
                rejection_reasons[child.version] = "train_gate:" + ",".join(failures)
            else:
                full_candidates.append(child)

        winner = (
            min(
                full_candidates,
                key=lambda item: (
                    train_evaluations[item.version].rank_key,
                    item.fingerprint(),
                ),
            )
            if full_candidates
            else parent
        )
        for child in full_candidates:
            if child is not winner:
                rejection_reasons[child.version] = "train_rank:not_selected"

        pending_rows = (
            self._pending_children.get("children")
            if isinstance(self._pending_children, Mapping)
            else None
        )
        if not isinstance(pending_rows, list) or len(pending_rows) != 3:
            raise RetrievalEvolutionError(
                "generation must retain exactly three scoped child slots"
            )
        slots = tuple(
            next(
                row
                for row in pending_rows
                if isinstance(row, Mapping) and row.get("scope") == scope
            )
            for scope in CHILD_SCOPES
        )
        child_versions = tuple(str(row["version"]) for row in slots)
        child_fingerprints = tuple(str(row["fingerprint"]) for row in slots)
        record = RetrievalGenerationTrace(
            generation=generation,
            parent_version=parent.version,
            parent_fingerprint=parent.fingerprint(),
            child_versions=child_versions,
            child_fingerprints=child_fingerprints,
            child_scopes=CHILD_SCOPES,
            screen_task_ids=tuple(task.numeric.task_id for task in screen),
            fold_entities=tuple(
                tuple(sorted({task.numeric.entity_name for task in fold}))
                for fold in remaining_folds
            ),
            promoted_fingerprints=tuple(
                child.fingerprint() for _scope, child in promoted
            ),
            train_winner_version=winner.version,
            train_winner_fingerprint=winner.fingerprint(),
            rejection_reasons=rejection_reasons,
            screen_summaries={
                version: evaluation.summary()
                for version, evaluation in screen_evaluations.items()
            },
            train_summaries={
                version: evaluation.summary()
                for version, evaluation in train_evaluations.items()
            },
        )
        self._generations.append(record)
        self._current_parent = winner
        self._next_generation = generation + 1
        self._pending_children = None
        self._event(
            "generation_completed",
            generation=generation,
            parent=parent.version,
            train_winner=winner.version,
            promoted=list(promoted_versions),
            rejection_reasons=rejection_reasons,
        )
        self._save_checkpoint(status="running", result=None)

    def _children_for_generation(
        self, generation: int, parent: RetrievalGenome
    ) -> tuple[list[tuple[RetrievalChildScope, RetrievalGenome]], dict[str, str]]:
        pending = self._pending_children
        if pending is not None:
            if (
                pending.get("generation") != generation
                or pending.get("parent_fingerprint") != parent.fingerprint()
            ):
                raise RetrievalCheckpointError("pending child checkpoint does not match Parent")
            raw_children = pending.get("children")
            if not isinstance(raw_children, list):
                raise RetrievalCheckpointError("invalid pending child checkpoint")
        else:
            raw_children = []
            self._pending_children = {
                "generation": generation,
                "parent_fingerprint": parent.fingerprint(),
                "children": raw_children,
            }

        children: list[tuple[RetrievalChildScope, RetrievalGenome]] = []
        rejections: dict[str, str] = {}
        for index, scope in enumerate(CHILD_SCOPES, start=1):
            assert self._original_parent is not None
            version_number = (
                int(self._original_parent.version[1:]) + generation * 3 + index
            )
            version = f"v{version_number:03d}"
            existing = next(
                (
                    item
                    for item in raw_children
                    if isinstance(item, Mapping) and item.get("scope") == scope
                ),
                None,
            )
            if existing is None:
                proposal = self._request_child(parent, scope, version, generation)
                child = parse_scoped_child(
                    parent,
                    proposal,
                    scope=scope,
                    skill_library=self.skill_library,
                )
                if child is None or child.version != version:
                    raw_fingerprint = _digest(
                        {
                            "scope": scope,
                            "version": version,
                            "proposal": proposal,
                        }
                    )
                    raw_children.append(
                        {
                            "scope": scope,
                            "valid": False,
                            "version": version,
                            "fingerprint": raw_fingerprint,
                            "proposal": dict(proposal),
                        }
                    )
                    self._all_child_fingerprints.append(raw_fingerprint)
                    rejections[version] = "invalid_scoped_candidate"
                    self._event(
                        "child_proposed",
                        generation=generation,
                        scope=scope,
                        version=version,
                        fingerprint=raw_fingerprint,
                        valid=False,
                    )
                    self._save_checkpoint(status="running", result=None)
                    continue
                existing = {
                    "scope": scope,
                    "valid": True,
                    "version": child.version,
                    "fingerprint": child.fingerprint(),
                    "proposal": child.to_payload(),
                }
                raw_children.append(existing)
                self._all_child_fingerprints.append(child.fingerprint())
                self._event(
                    "child_proposed",
                    generation=generation,
                    scope=scope,
                    version=child.version,
                    fingerprint=child.fingerprint(),
                    valid=True,
                )
                self._save_checkpoint(status="running", result=None)
            if not bool(existing.get("valid")):
                rejections[version] = "invalid_scoped_candidate"
                continue
            proposal = existing.get("proposal")
            if not isinstance(proposal, Mapping):
                raise RetrievalCheckpointError("pending child has no complete proposal")
            child = parse_scoped_child(
                parent,
                proposal,
                scope=scope,
                skill_library=self.skill_library,
            )
            if (
                child is None
                or child.version != version
                or child.fingerprint() != existing.get("fingerprint")
            ):
                raise RetrievalCheckpointError("pending child fingerprint mismatch")
            children.append((scope, child))
        return children, rejections

    def _request_child(
        self,
        parent: RetrievalGenome,
        scope: RetrievalChildScope,
        version: str,
        generation: int,
    ) -> Mapping[str, object]:
        system = (
            "Return one complete typed Retrieval Genome JSON object. "
            "Obey the supplied immutable scope and host-owned schema."
        )
        payload = {
            "schema_version": 1,
            "generation": generation,
            "scope": scope,
            "required_version": version,
            "required_parent": parent.version,
            "mutable_fields": sorted(
                _PRIMARY_SCOPE_FIELDS[scope] | {"active_skill_ids"}
            ),
            "owned_skill_stage": _SCOPE_SKILL_STAGE[scope],
            "parent_genome": parent.to_payload(),
        }
        attempts = self.config.transient_retries + 1
        for attempt in range(attempts):
            try:
                response = self.mutation_llm.complete(
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }
                    ],
                    temperature=0.0,
                )
                try:
                    return parse_json_object(response.text)
                except JsonExtractionError as error:
                    self._event(
                        "mutation_response_invalid",
                        generation=generation,
                        scope=scope,
                        error=str(error),
                    )
                    return {
                        "invalid_mutation_response": f"{type(error).__name__}:{error}"
                    }
            except TransientLLMError as error:
                if attempt + 1 >= attempts:
                    self._event(
                        "transient_exhausted",
                        operation="mutation",
                        generation=generation,
                        scope=scope,
                        error=str(error),
                    )
                    self._save_checkpoint(status="running", result=None)
                    raise
                self._event(
                    "transient_retry",
                    operation="mutation",
                    generation=generation,
                    scope=scope,
                    attempt=attempt + 1,
                )
                self._save_checkpoint(status="running", result=None)

        raise AssertionError("unreachable mutation retry loop")

    def _evaluate_batch(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        readonly: bool,
        library: RetrievalSkillLibrary | None,
    ) -> RetrievalEvaluation:
        if not tasks:
            raise RetrievalEvolutionError("evaluation batches cannot be empty")
        verifier_hash = str(self._scientific_inputs["verifier_hash"])
        evaluator_hash = str(self._scientific_inputs["evaluator_hash"])
        task_keys = tuple(
            build_inference_cache_key(
                task,
                genome,
                library,
                verifier_hash=verifier_hash,
                evaluator_hash=evaluator_hash,
            )
            for task in tasks
        )
        batch_key = _digest(
            {
                "stage": stage,
                "task_keys": [item.digest() for item in task_keys],
                "readonly": readonly,
            }
        )
        cached = self._evaluation_cache.get(batch_key)
        if cached is not None:
            self._event(
                "evaluation_cache_hit",
                stage=stage,
                genome=genome.version,
                task_count=len(tasks),
            )
            return cached

        kwargs: dict[str, object] = {
            "stage": stage,
            "skill_library": library,
            "harness_factory": self.harness_factory,
            "persist": False,
            "writers_enabled": not readonly,
            "evolver_enabled": not readonly,
            "cache_keys": task_keys,
            "metric_cap": self.config.metric_cap,
        }
        attempts = self.config.transient_retries + 1
        for attempt in range(attempts):
            try:
                result = self._invoke_evaluator(genome, tasks, kwargs)
                break
            except TransientLLMError as error:
                if attempt + 1 >= attempts:
                    self._event(
                        "transient_exhausted",
                        operation="evaluation",
                        stage=stage,
                        genome=genome.version,
                        error=str(error),
                    )
                    self._save_checkpoint(status="running", result=None)
                    raise
                self._event(
                    "transient_retry",
                    operation="evaluation",
                    stage=stage,
                    genome=genome.version,
                    attempt=attempt + 1,
                )
                self._save_checkpoint(status="running", result=None)
        else:
            raise AssertionError("unreachable evaluator retry loop")
        if not isinstance(result, RetrievalEvaluation):
            raise RetrievalEvolutionError("trusted evaluator returned an untyped result")
        if result.version != genome.version:
            raise RetrievalEvolutionError("trusted evaluator returned the wrong Genome version")
        if result.task_count != len(tasks):
            raise RetrievalEvolutionError("trusted evaluator did not cover the complete task batch")
        expected_task_ids = tuple(task.numeric.task_id for task in tasks)
        trace_task_ids = tuple(trace.get("task_id") for trace in result.task_traces)
        if (
            len(trace_task_ids) != len(expected_task_ids)
            or any(not isinstance(task_id, str) for task_id in trace_task_ids)
            or set(trace_task_ids) != set(expected_task_ids)
        ):
            raise RetrievalEvolutionError(
                "trusted evaluator task traces do not provide exact task coverage"
            )
        self._evaluation_cache[batch_key] = result
        self._cache_records[batch_key] = {
            "stage": stage,
            "genome_fingerprint": genome.fingerprint(),
            "task_ids": [task.numeric.task_id for task in tasks],
            "task_cache_keys": [item.digest() for item in task_keys],
            "evaluation_sha256": _digest(result.to_payload()),
            "evaluation": result.to_payload(),
        }
        self._save_checkpoint(status="running", result=None)
        return result

    def _invoke_evaluator(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        kwargs: dict[str, object],
    ) -> RetrievalEvaluation:
        method = getattr(self.evaluator, "evaluate", self.evaluator)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            accepted = kwargs
        else:
            parameters = signature.parameters.values()
            has_var_kwargs = any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters
            )
            accepted = (
                kwargs
                if has_var_kwargs
                else {key: value for key, value in kwargs.items() if key in signature.parameters}
            )
        return method(genome, tasks, **accepted)

    def _library_for(
        self,
        genome: RetrievalGenome,
        *,
        source: RetrievalSkillLibrary | None = None,
    ) -> RetrievalSkillLibrary | None:
        fingerprint = genome.fingerprint()
        if fingerprint not in self._candidate_libraries:
            self._candidate_libraries[fingerprint] = self._clone_library(
                source if source is not None else self.skill_library
            )
        return self._candidate_libraries[fingerprint]

    def _readonly_library(
        self, genome: RetrievalGenome
    ) -> RetrievalSkillLibrary | None:
        return self._clone_library(self._library_for(genome))

    @staticmethod
    def _clone_library(
        library: RetrievalSkillLibrary | None,
    ) -> RetrievalSkillLibrary | None:
        return None if library is None else library.clone(persist=False)

    def _event(self, kind: str, **payload: object) -> None:
        self._trace.append({"kind": kind, **payload})

    def _checkpoint_path(self) -> Path | None:
        value = self.config.checkpoint_path
        return value if isinstance(value, Path) else None

    def _save_checkpoint(
        self,
        *,
        status: Literal["running", "complete"],
        result: RetrievalEvolutionResult | None,
    ) -> None:
        path = self._checkpoint_path()
        if path is None or self._original_parent is None or self._current_parent is None:
            return
        task_completion = [
            {key: value for key, value in record.items() if key != "evaluation"}
            | {"cache_key": cache_key}
            for cache_key, record in sorted(self._cache_records.items())
        ]
        payload = {
            "schema_version": RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION,
            "status": status,
            "scientific_inputs": self._scientific_inputs,
            "original_parent": self._original_parent.to_payload(),
            "original_parent_fingerprint": self._original_parent.fingerprint(),
            "current_parent": self._current_parent.to_payload(),
            "current_parent_fingerprint": self._current_parent.fingerprint(),
            "next_generation": self._next_generation,
            "pending_children": self._pending_children,
            "child_fingerprints": list(self._all_child_fingerprints),
            "task_completion": task_completion,
            "evaluation_cache": self._cache_records,
            "generations": [item.to_payload() for item in self._generations],
            "trace": self._trace,
            "result": result.to_payload() if result is not None else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_checkpoint(
        self, parent: RetrievalGenome
    ) -> RetrievalEvolutionResult | None:
        path = self._checkpoint_path()
        if path is None or not path.exists() or not self.config.resume:
            self._original_parent = parent
            self._current_parent = parent
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalCheckpointError("invalid Retrieval evolution checkpoint") from error
        expected = {
            "schema_version",
            "status",
            "scientific_inputs",
            "original_parent",
            "original_parent_fingerprint",
            "current_parent",
            "current_parent_fingerprint",
            "next_generation",
            "pending_children",
            "child_fingerprints",
            "task_completion",
            "evaluation_cache",
            "generations",
            "trace",
            "result",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise RetrievalCheckpointError("invalid Retrieval evolution checkpoint schema")
        if raw["schema_version"] != RETRIEVAL_EVOLUTION_CHECKPOINT_SCHEMA_VERSION:
            raise RetrievalCheckpointError("unsupported Retrieval evolution checkpoint schema")
        if _canonical_json(raw["scientific_inputs"]) != _canonical_json(
            self._scientific_inputs
        ):
            raise RetrievalCheckpointError("checkpoint scientific inputs do not match this run")
        try:
            original = RetrievalGenome.from_payload(raw["original_parent"])
            current = RetrievalGenome.from_payload(raw["current_parent"])
        except (RetrievalPolicyError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid checkpoint Parent Genome") from error
        if (
            original.fingerprint() != raw["original_parent_fingerprint"]
            or current.fingerprint() != raw["current_parent_fingerprint"]
            or original.fingerprint() != parent.fingerprint()
        ):
            raise RetrievalCheckpointError("checkpoint Parent fingerprint mismatch")
        try:
            generations = [
                RetrievalGenerationTrace.from_payload(item)
                for item in raw["generations"]
            ]
            trace = [dict(item) for item in raw["trace"]]
            cache_records = {
                str(key): dict(value)
                for key, value in raw["evaluation_cache"].items()
            }
            child_fingerprints = [str(item) for item in raw["child_fingerprints"]]
            task_completion = list(raw["task_completion"])
            next_generation = int(raw["next_generation"])
        except (AttributeError, TypeError, ValueError) as error:
            raise RetrievalCheckpointError("invalid checkpoint execution state") from error
        expected_completion = [
            {key: value for key, value in record.items() if key != "evaluation"}
            | {"cache_key": cache_key}
            for cache_key, record in sorted(cache_records.items())
        ]
        if task_completion != expected_completion:
            raise RetrievalCheckpointError("checkpoint task completion binding mismatch")
        flattened = [
            fingerprint
            for generation in generations
            for fingerprint in generation.child_fingerprints
        ]
        pending = raw["pending_children"]
        if pending is not None:
            if not isinstance(pending, Mapping) or not isinstance(
                pending.get("children"), list
            ):
                raise RetrievalCheckpointError("invalid pending child state")
            flattened.extend(
                str(item.get("fingerprint"))
                for item in pending["children"]
                if isinstance(item, Mapping)
            )
        if flattened != child_fingerprints:
            raise RetrievalCheckpointError("checkpoint Child fingerprint binding mismatch")
        if (
            next_generation != len(generations)
            or tuple(item.generation for item in generations)
            != tuple(range(next_generation))
        ):
            raise RetrievalCheckpointError(
                "checkpoint generation cursor does not match completed records"
            )
        expected_current_fingerprint = (
            original.fingerprint()
            if not generations
            else generations[-1].train_winner_fingerprint
        )
        if current.fingerprint() != expected_current_fingerprint:
            raise RetrievalCheckpointError(
                "checkpoint current Parent does not match the generation winner"
            )
        status = raw["status"]
        if status == "complete" and (
            next_generation != self.config.generations or pending is not None
        ):
            raise RetrievalCheckpointError(
                "completed checkpoint has an invalid generation cursor"
            )
        if status == "running" and pending is not None and (
            pending.get("generation") != next_generation
            or pending.get("parent_fingerprint") != current.fingerprint()
        ):
            raise RetrievalCheckpointError(
                "pending checkpoint generation does not match the current Parent"
            )
        evaluation_cache: dict[str, RetrievalEvaluation] = {}
        for cache_key, record in cache_records.items():
            if set(record) != {
                "stage",
                "genome_fingerprint",
                "task_ids",
                "task_cache_keys",
                "evaluation_sha256",
                "evaluation",
            }:
                raise RetrievalCheckpointError("invalid checkpoint evaluation cache")
            if record["evaluation_sha256"] != _digest(record["evaluation"]):
                raise RetrievalCheckpointError(
                    "checkpoint evaluation digest does not match task completion"
                )
            evaluation_cache[cache_key] = RetrievalEvaluation.from_payload(
                record["evaluation"]
            )
        self._original_parent = original
        self._current_parent = current
        self._next_generation = next_generation
        self._generations = generations
        self._trace = trace
        self._cache_records = cache_records
        self._evaluation_cache = evaluation_cache
        self._pending_children = None if pending is None else dict(pending)
        self._all_child_fingerprints = child_fingerprints
        if status == "complete":
            if raw["result"] is None:
                raise RetrievalCheckpointError("completed checkpoint has no result")
            result = RetrievalEvolutionResult.from_payload(raw["result"])
            if (
                result.original_parent.fingerprint() != original.fingerprint()
                or result.train_winner.fingerprint() != current.fingerprint()
                or result.generations != tuple(generations)
                or result.trace != tuple(trace)
                or result.release_published
            ):
                raise RetrievalCheckpointError("completed checkpoint result binding mismatch")
            return result
        if status != "running" or raw["result"] is not None:
            raise RetrievalCheckpointError("invalid checkpoint status")
        if not 0 <= next_generation <= self.config.generations:
            raise RetrievalCheckpointError("invalid checkpoint generation cursor")
        return None


__all__ = [
    "CHILD_SCOPES",
    "RetrievalCheckpointError",
    "RetrievalEvaluation",
    "RetrievalEvolutionConfig",
    "RetrievalEvolutionEngine",
    "RetrievalEvolutionError",
    "RetrievalEvolutionResult",
    "RetrievalGenerationTrace",
    "RetrievalInferenceCacheKey",
    "build_inference_cache_key",
    "combine_retrieval_evaluations",
    "parse_scoped_child",
]
