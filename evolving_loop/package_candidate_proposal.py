"""Typed and redacted package-coordinate proposal contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol

from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateState,
    PackageCoordinateTarget,
)
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TASK_IDENTITY = re.compile(r"(?i)\btask[_-]?\d")
_SUMMARY_KEYS = frozenset(
    {
        "task_count",
        "mean_smae",
        "mean_srmse",
        "mean_joint",
        "p90_smae",
        "p95_smae",
        "p90_srmse",
        "p95_srmse",
        "invalid_count",
        "catastrophic_count",
        "clipped_count",
        "fallback_count",
        "coverage",
    }
)
_REJECTED_SUMMARY_KEYS = _SUMMARY_KEYS | frozenset({"candidate_sha256"})
_GATE_NAMES = frozenset(
    {
        "task_coverage",
        "mean_smae",
        "mean_srmse",
        "invalid_count",
        "catastrophic_count",
        "clipped_count",
        "fallback_count",
        "public_test_accessed",
        "minimum_relative_joint_gain",
        "p90_smae",
        "p95_smae",
        "p90_srmse",
        "p95_srmse",
        "maximum_task_joint_regret",
        "initial_toto_mean_smae",
        "initial_toto_mean_srmse",
        "fold_manifest",
        "nonregressing_folds",
        "improving_folds",
    }
)
_INVALID_REASONS = frozenset(
    {
        "invalid_schema",
        "duplicate_child",
        "unknown_candidate",
        "materialization_failed",
        "cross_coordinate_change",
    }
)
_FORBIDDEN_KEY_PARTS = (
    "task_id",
    "task_trace",
    "per_task",
    "entity",
    "forecast",
    "future_value",
    "truth",
    "document",
    "quote",
    "residual",
)
_STRUCTURE_KEYS = frozenset(
    {
        "structure_sha256",
        "recipe_sha256",
        "candidate_sha256",
        "proposal_sha256",
        "behavior_sha256",
        "target",
        "kind",
        "candidate_name",
        "parents",
        "fallback_parent",
        "assumptions",
        "stage_support",
        "invalid_reason",
    }
)
_STRUCTURE_KINDS = frozenset(
    {
        "select",
        "route",
        "horizon_route",
        "weighted",
        "median",
        "bounded_overlay",
    }
)
_ASSUMPTION_KEYS = frozenset(
    {"candidate_name", "feature", "direction", "horizon_region", "operator"}
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_safe_mapping(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(
                fragment in key.casefold() for fragment in _FORBIDDEN_KEY_PARTS
            ):
                raise ValueError(f"{context} contains a forbidden task-level field")
            _validate_safe_mapping(item, context=context)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_safe_mapping(item, context=context)
        return
    if value is None or type(value) in {str, int, float, bool}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{context} must contain finite values")
        if type(value) is str and _TASK_IDENTITY.search(value):
            raise ValueError(f"{context} contains a forbidden task identity")
        return
    raise ValueError(f"{context} must contain canonical JSON values")


def _validate_structure(structure: Mapping[str, object]) -> None:
    if not structure or set(structure) - _STRUCTURE_KEYS:
        raise ValueError("proposal structure must use the closed structural schema")
    digest_keys = tuple(key for key in structure if key.endswith("_sha256"))
    if not digest_keys or any(
        not isinstance(structure[key], str)
        or _SHA256.fullmatch(structure[key]) is None  # type: ignore[arg-type]
        for key in digest_keys
    ):
        raise ValueError("proposal structure requires a closed SHA-256 identity")
    kind = structure.get("kind")
    if kind is not None and kind not in _STRUCTURE_KINDS:
        raise ValueError("proposal structure kind must use the closed operator set")
    target = structure.get("target")
    if target is not None and target not in {"numerical", "retrieval", "decision"}:
        raise ValueError("proposal structure target must use the coordinate set")
    reason = structure.get("invalid_reason")
    if reason is not None and reason not in _INVALID_REASONS:
        raise ValueError("proposal structure reason must use the closed reason set")
    support = structure.get("stage_support")
    if support is not None and (type(support) is not int or support <= 0):
        raise ValueError("proposal structure support must be a positive integer")
    for key in ("candidate_name", "fallback_parent"):
        name = structure.get(key)
        if name is not None and (
            not isinstance(name, str) or not name.isidentifier() or name.startswith("_")
        ):
            raise ValueError("proposal structure names must be closed identifiers")
    parents = structure.get("parents")
    if parents is not None and (
        not isinstance(parents, (tuple, list))
        or not parents
        or any(
            not isinstance(parent, str)
            or not parent.isidentifier()
            or parent.startswith("_")
            for parent in parents
        )
    ):
        raise ValueError("proposal structure parents must be closed identifiers")
    assumptions = structure.get("assumptions")
    if assumptions is not None:
        if not isinstance(assumptions, (tuple, list)) or any(
            not isinstance(item, Mapping) or set(item) != _ASSUMPTION_KEYS
            for item in assumptions
        ):
            raise ValueError("proposal assumptions must use the closed structural schema")
        for assumption in assumptions:
            _validate_safe_mapping(assumption, context="proposal assumption")
            if assumption.get("operator") not in _STRUCTURE_KINDS:
                raise ValueError("proposal assumption operator must be closed")


def _summary(evaluation: PackageEvaluation) -> dict[str, float | int]:
    if not isinstance(evaluation, PackageEvaluation):
        raise TypeError("package proposal feedback requires PackageEvaluation values")
    return {key: getattr(evaluation, key) for key in sorted(_SUMMARY_KEYS)}


@dataclass(frozen=True)
class PackageProposalFeedback:
    """The complete aggregate-only evidence permitted into proposal models."""

    parent_summary: Mapping[str, float | int]
    rejected_summaries: tuple[Mapping[str, float | int | str], ...]
    gate_names: tuple[str, ...]
    structures: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_summary, Mapping) or set(
            self.parent_summary
        ) - _SUMMARY_KEYS:
            raise ValueError("parent summary must use the aggregate allowlist")
        parent = dict(self.parent_summary)
        for key, value in parent.items():
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(f"parent aggregate {key} must be finite and numeric")
        if type(self.rejected_summaries) is not tuple:
            raise ValueError("rejected summaries must be an exact tuple")
        rejected: list[Mapping[str, float | int | str]] = []
        for summary in self.rejected_summaries:
            if not isinstance(summary, Mapping) or set(summary) - _REJECTED_SUMMARY_KEYS:
                raise ValueError("rejected summary must use the aggregate allowlist")
            row = dict(summary)
            candidate = row.get("candidate_sha256")
            if candidate is not None and (
                not isinstance(candidate, str) or _SHA256.fullmatch(candidate) is None
            ):
                raise ValueError("rejected candidate identity must be canonical")
            for key, value in row.items():
                if key == "candidate_sha256":
                    continue
                if type(value) not in {int, float} or not math.isfinite(float(value)):
                    raise ValueError(f"rejected aggregate {key} must be finite and numeric")
            rejected.append(MappingProxyType(dict(sorted(row.items()))))
        if type(self.gate_names) is not tuple or any(
            type(name) is not str or name not in _GATE_NAMES
            for name in self.gate_names
        ):
            raise ValueError("proposal feedback gate names must use the closed allowlist")
        if type(self.structures) is not tuple or any(
            not isinstance(structure, Mapping) for structure in self.structures
        ):
            raise ValueError("proposal structures must be an exact mapping tuple")
        structures: list[Mapping[str, object]] = []
        for structure in self.structures:
            _validate_safe_mapping(structure, context="proposal structure")
            _validate_structure(structure)
            structures.append(_freeze(structure))  # type: ignore[arg-type]
        structures.sort(key=lambda item: _canonical_json(item))
        rejected.sort(key=lambda item: _canonical_json(item))
        object.__setattr__(self, "parent_summary", MappingProxyType(dict(sorted(parent.items()))))
        object.__setattr__(self, "rejected_summaries", tuple(rejected))
        object.__setattr__(self, "gate_names", tuple(sorted(set(self.gate_names))))
        object.__setattr__(self, "structures", tuple(structures))

    @classmethod
    def from_evaluations(
        cls,
        *,
        parent: PackageEvaluation,
        rejected_children: Sequence[PackageEvaluation] = (),
        gate_names: Sequence[str] = (),
        structures: Sequence[Mapping[str, object]] = (),
    ) -> "PackageProposalFeedback":
        rejected = tuple(rejected_children)
        if any(not isinstance(item, PackageEvaluation) for item in rejected):
            raise TypeError("rejected proposal feedback requires PackageEvaluation values")
        return cls(
            parent_summary=_summary(parent),
            rejected_summaries=tuple(
                {"candidate_sha256": item.candidate_sha256, **_summary(item)}
                for item in rejected
            ),
            gate_names=tuple(gate_names),
            structures=tuple(structures),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "parent_summary": dict(self.parent_summary),
            "rejected_summaries": [dict(item) for item in self.rejected_summaries],
            "gate_names": list(self.gate_names),
            "structures": [_plain(item) for item in self.structures],
        }


@dataclass(frozen=True)
class PackageCandidate:
    slot: int
    target: PackageCoordinateTarget
    state: PackageCoordinateState
    proposal_sha256: str
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.slot) is not int or not 0 <= self.slot < 3:
            raise ValueError("package proposal slot must be one of 0, 1, or 2")
        if self.target not in {"numerical", "retrieval", "decision"}:
            raise ValueError("package proposal target is invalid")
        if not isinstance(self.state, PackageCoordinateState):
            raise ValueError("package proposal requires a coordinate state")
        if not isinstance(self.proposal_sha256, str) or _SHA256.fullmatch(
            self.proposal_sha256
        ) is None:
            raise ValueError("package proposal fingerprint must be canonical")
        if self.invalid_reason is not None and self.invalid_reason not in _INVALID_REASONS:
            raise ValueError("package proposal invalid reason must use the closed reason set")
        if self.invalid_reason is None and self.state.bundle.coordinate != self.target:
            raise ValueError("valid package proposal state must identify its target")
        if (
            self.invalid_reason is None
            and self.state.bundle.acceptance_evidence_sha256 is not None
        ):
            raise ValueError("valid package proposal must not carry acceptance evidence")


class PackageCandidateProposer(Protocol):
    def propose(
        self,
        parent: PackageCoordinateState,
        feedback: PackageProposalFeedback,
        *,
        generation: int,
        child_count: int,
    ) -> tuple[PackageCandidate, ...]: ...


def proposal_fingerprint(
    *,
    target: PackageCoordinateTarget,
    generation: int,
    slot: int,
    payload: object,
) -> str:
    return _digest(
        {
            "target": target,
            "generation": generation,
            "slot": slot,
            "payload": payload,
        }
    )


def embed_retrieval_candidate(
    policy: HarnessPolicy,
    genome: RetrievalGenome,
    skills: RetrievalSkillLibrary,
    *,
    changelog: str,
) -> HarnessPolicy:
    """Embed a canonical candidate snapshot without accepted-release authority."""
    if not isinstance(policy, HarnessPolicy) or not isinstance(genome, RetrievalGenome):
        raise ValueError("Retrieval candidate embedding requires typed policy and Genome")
    if not isinstance(skills, RetrievalSkillLibrary) or not getattr(
        skills, "_read_only", False
    ):
        raise ValueError("Retrieval candidate embedding requires a read-only Skill snapshot")
    skill_payload = [item.to_payload() for item in skills.all()]
    manifest = {
        "schema_version": 1,
        "version": genome.version,
        "parent": genome.parent,
        "genome_sha256": genome.fingerprint(),
        "round1_prompt_sha256": hashlib.sha256(genome.round1_prompt.encode("utf-8")).hexdigest(),
        "round2_prompt_sha256": hashlib.sha256(genome.round2_prompt.encode("utf-8")).hexdigest(),
        "skills_sha256": _digest(skill_payload),
        "state": "candidate",
        "train_dev_split_sha256": None,
        "verifier_sha256": None,
        "evaluator_sha256": None,
        "metric_sha256": None,
        "metric_cap": None,
        "resource_budgets": {
            "max_selected_documents": genome.max_selected_documents,
            "max_evidence_chains": genome.max_evidence_chains,
            "max_citations_per_chain": genome.max_citations_per_chain,
        },
        "train_summary": None,
        "dev_summary": None,
        "acceptance_reason": "not_evaluated_candidate",
        "audit_sha256": None,
    }
    embedded = {
        "genome": genome.to_payload(),
        "round1_prompt": genome.round1_prompt,
        "round2_prompt": genome.round2_prompt,
        "skills": skill_payload,
        "manifest": manifest,
    }
    return replace(
        policy,
        version=genome.version,
        parent=genome.parent,
        retrieval_prompt=genome.round1_prompt,
        retrieval_skills=tuple(skill_payload),
        changelog=changelog,
        retrieval_release_payload=embedded,
        retrieval_release_sha256=HarnessPolicy.retrieval_payload_fingerprint(embedded),
        retrieval_skill_source=skills,
    )


__all__ = [
    "PackageCandidate",
    "PackageCandidateProposer",
    "PackageProposalFeedback",
    "embed_retrieval_candidate",
    "proposal_fingerprint",
]
