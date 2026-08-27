"""Trusted, post-resolution Retrieval diagnostics and marginal credit."""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

from common.metrics import drcik_point_metrics
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
)
from evolving_loop.retrieval_agent.verifier import _verified_quote_spans

if TYPE_CHECKING:
    from evolving_loop.harness import CandidatePoolSnapshot, HarnessResult


@dataclass(frozen=True)
class SkillCredit:
    skill_id: str
    marginal_smae_gain: float
    marginal_srmse_gain: float


@dataclass(frozen=True)
class SkillNecessity:
    skill_id: str
    necessary: bool
    omitted_smae_regret: float = 0.0
    omitted_srmse_regret: float = 0.0


@dataclass(frozen=True)
class EvidenceChainCredit:
    chain_id: str
    numeric_eligible: bool
    marginal_smae_gain: float
    marginal_srmse_gain: float
    used_skill_ids: tuple[str, ...] = ()
    skill_credit: tuple[SkillCredit, ...] = ()


@dataclass(frozen=True)
class RetrievalTaskDiagnostics:
    supporting_recall: float
    gt_evidence_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    contextual_oracle_smae_gain: float
    contextual_oracle_srmse_gain: float
    invalid_count: int
    catastrophic_count: int
    chain_credit: tuple[EvidenceChainCredit, ...]


@dataclass(frozen=True)
class RetrievalCreditReport:
    coding_oracle_smae: float
    coding_oracle_srmse: float
    contextual_oracle_smae: float
    contextual_oracle_srmse: float
    final_smae: float
    final_srmse: float
    decision_smae_regret: float
    decision_srmse_regret: float
    chains: tuple[EvidenceChainCredit, ...]
    diagnostics: RetrievalTaskDiagnostics


@dataclass(frozen=True)
class RetrievalSkillTaskEvidence:
    """One pre-label leave-one-out replay scored on a resolved Train task."""

    skill_id: str
    task_id: str
    entity_name: str
    split: str
    exact_quote_validity: float
    without_skill_smae: float
    without_skill_srmse: float
    with_skill_smae: float
    with_skill_srmse: float
    added_catastrophic_count: int
    necessary: bool

    def __post_init__(self) -> None:
        if not self.skill_id or not self.task_id or not self.entity_name:
            raise ValueError("Skill evidence requires Skill, task, and entity IDs")
        if self.split not in {"train", "dev"}:
            raise ValueError("Skill evidence split must be train or dev")
        metrics = (
            self.exact_quote_validity,
            self.without_skill_smae,
            self.without_skill_srmse,
            self.with_skill_smae,
            self.with_skill_srmse,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics):
            raise ValueError("Skill evidence metrics must be finite")
        if not 0.0 <= self.exact_quote_validity <= 1.0:
            raise ValueError("exact quote validity must be in [0, 1]")
        if isinstance(self.added_catastrophic_count, bool) or self.added_catastrophic_count < 0:
            raise ValueError("added catastrophic count must be non-negative")
        if not isinstance(self.necessary, bool):
            raise ValueError("Skill necessity must be boolean")


def _score_pool(
    truth: Sequence[float], snapshot: "CandidatePoolSnapshot"
) -> tuple[dict[str, float | bool], int]:
    scored: list[tuple[str, dict[str, float | bool]]] = []
    invalid = 0
    for candidate in snapshot.candidates:
        try:
            score = drcik_point_metrics(truth, candidate.forecast)
        except (TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        scored.append((candidate.candidate_id, score))
    if not scored:
        raise ValueError("candidate pool snapshot has no scoreable forecasts")
    _candidate_id, oracle = min(
        scored,
        key=lambda item: (
            float(item[1]["srmse"]),
            float(item[1]["smae"]),
            item[0],
        ),
    )
    return oracle, invalid


def _fallback_snapshots(result: "HarnessResult") -> tuple["CandidatePoolSnapshot", ...]:
    from evolving_loop.harness import CandidatePoolEntry, CandidatePoolSnapshot

    coding_ids = {
        candidate.program.name for candidate in getattr(result.coding, "candidates", ())
    }
    baseline_by_id = {
        candidate.candidate_id: CandidatePoolEntry(
            candidate.candidate_id, candidate.forecast
        )
        for candidate in result.candidates
        if candidate.candidate_id in coding_ids
    }
    baseline = tuple(baseline_by_id.values())
    if not baseline:
        baseline = tuple(
            {
                candidate.candidate_id: CandidatePoolEntry(
                    candidate.candidate_id, candidate.forecast
                )
                for candidate in result.candidates
            }.values()
        )
    snapshots = [CandidatePoolSnapshot(None, baseline)]
    pool = list(baseline)
    chains = tuple(
        chain
        for chain in getattr(getattr(result, "retrieval_card", None), "chains", ())
        if chain.numeric_eligible
    )
    baseline_ids = {item.candidate_id for item in baseline}
    contextual = tuple(
        {
            candidate.candidate_id: candidate
            for candidate in result.candidates
            if candidate.candidate_id not in baseline_ids
        }.values()
    )
    for index, chain in enumerate(chains):
        if index < len(contextual):
            candidate = contextual[index]
            pool.append(CandidatePoolEntry(candidate.candidate_id, candidate.forecast))
        snapshots.append(CandidatePoolSnapshot(chain.chain_id, tuple(pool)))
    if not chains and contextual:
        for index, candidate in enumerate(contextual):
            pool.append(CandidatePoolEntry(candidate.candidate_id, candidate.forecast))
            snapshots.append(CandidatePoolSnapshot(f"legacy_evidence_{index}", tuple(pool)))
    return tuple(snapshots)


def _chain_ledger(result: "HarnessResult") -> tuple[object, ...]:
    card = getattr(result, "retrieval_card", None)
    return tuple(card.chains) if card is not None else ()


def _retrieved_document_ids(result: "HarnessResult") -> tuple[str, ...]:
    card = getattr(result, "retrieval_card", None)
    if card is not None:
        return tuple(card.selected_document_ids)
    retrieval = getattr(result, "retrieval", None)
    return tuple(getattr(retrieval, "selected_document_ids", ()))


def _normalized_tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def _retrieval_quality(
    task: ContextTask,
    result: "HarnessResult",
    chains: tuple[object, ...],
) -> tuple[float, float, float, float, float, int]:
    retrieved = set(_retrieved_document_ids(result))
    supporting = {document.document_id for document in task.documents if document.role == "supporting"}
    distractors = {document.document_id for document in task.documents if document.role == "distractor"}
    supporting_recall = len(retrieved & supporting) / len(supporting) if supporting else 1.0
    distractor_avoidance = (
        1.0 - len(retrieved & distractors) / len(distractors) if distractors else 1.0
    )

    recovered_text = " ".join(
        value
        for chain in chains
        for value in (
            str(getattr(chain, "claim", "")),
            *(citation.exact_quote for citation in getattr(chain, "citations", ())),
        )
    )
    recovered_tokens = _normalized_tokens(recovered_text)
    gt_evidence_recall = (
        sum(
            1
            for evidence in task.gt_evidence
            if (tokens := _normalized_tokens(evidence)) and tokens.issubset(recovered_tokens)
        )
        / len(task.gt_evidence)
        if task.gt_evidence
        else 1.0
    )

    documents = {document.document_id: document.content for document in task.documents}
    citations = tuple(
        citation for chain in chains for citation in getattr(chain, "citations", ())
    )
    valid_quotes = sum(
        bool(_verified_quote_spans(citation.exact_quote, documents.get(citation.document_id, "")))
        for citation in citations
    )
    card = getattr(result, "retrieval_card", None)
    rejections = tuple(getattr(card, "rejected", ())) if card is not None else tuple(
        getattr(getattr(result, "retrieval", None), "rejected", ())
    )
    ungrounded = sum(str(reason).startswith("ungrounded_quote:") for reason in rejections)
    quote_attempts = len(citations) + ungrounded
    exact_quote_validity = valid_quotes / quote_attempts if quote_attempts else 1.0

    evidence_chains = chains
    if card is not None:
        evidence_chains = tuple(card.round1.chains) + (
            tuple(card.round2.chains) if card.round2 is not None else ()
        )
    complete_chain_rate = (
        sum(bool(getattr(chain, "numeric_eligible", False)) for chain in evidence_chains)
        / len(evidence_chains)
        if evidence_chains
        else 0.0
    )
    return (
        supporting_recall,
        gt_evidence_recall,
        distractor_avoidance,
        exact_quote_validity,
        complete_chain_rate,
        len(rejections),
    )


def assign_chain_credit(
    task: ContextTask, result: "HarnessResult"
) -> RetrievalCreditReport:
    """Assign contextual-oracle pool gain only after public labels resolve."""
    if not task.labels_public or not task.numeric.future_values:
        raise ValueError("Retrieval credit requires resolved public labels")
    snapshots = tuple(getattr(result, "candidate_pool_snapshots", ())) or _fallback_snapshots(result)
    if not snapshots or snapshots[0].after_chain_id is not None:
        raise ValueError("candidate snapshots must start with the numeric-only pool")
    truth = task.numeric.future_values
    scored_snapshots = [_score_pool(truth, snapshot) for snapshot in snapshots]
    coding_score = scored_snapshots[0][0]
    contextual_score = scored_snapshots[-1][0]
    final_score = drcik_point_metrics(truth, result.forecast)
    chains = _chain_ledger(result)
    snapshot_index = 1
    chain_credit: list[EvidenceChainCredit] = []
    previous = coding_score
    for chain in chains:
        marginal_smae = 0.0
        marginal_srmse = 0.0
        if chain.numeric_eligible:
            if snapshot_index >= len(snapshots):
                raise ValueError("missing candidate-pool snapshot for numeric chain")
            snapshot = snapshots[snapshot_index]
            if snapshot.after_chain_id != chain.chain_id:
                raise ValueError("candidate-pool snapshots do not follow verified chain order")
            current = scored_snapshots[snapshot_index][0]
            marginal_smae = float(previous["smae"]) - float(current["smae"])
            marginal_srmse = float(previous["srmse"]) - float(current["srmse"])
            previous = current
            snapshot_index += 1
        used_skill_ids = tuple(chain.used_skill_ids)
        direct_skill_credit = (
            (
                SkillCredit(used_skill_ids[0], marginal_smae, marginal_srmse),
            )
            if len(used_skill_ids) == 1 and chain.numeric_eligible
            else ()
        )
        chain_credit.append(
            EvidenceChainCredit(
                chain_id=chain.chain_id,
                numeric_eligible=bool(chain.numeric_eligible),
                marginal_smae_gain=marginal_smae,
                marginal_srmse_gain=marginal_srmse,
                used_skill_ids=used_skill_ids,
                skill_credit=direct_skill_credit,
            )
        )
    if chains and snapshot_index != len(snapshots):
        raise ValueError("candidate-pool snapshots contain unverified chain additions")

    quality = _retrieval_quality(task, result, chains)
    final_snapshot = snapshots[-1]
    catastrophic_count = 0
    invalid_candidates = scored_snapshots[-1][1]
    for candidate in final_snapshot.candidates:
        try:
            score = drcik_point_metrics(truth, candidate.forecast)
        except (TypeError, ValueError, OverflowError):
            continue
        if bool(score["smae_clipped"]) or bool(score["srmse_clipped"]):
            catastrophic_count += 1
    diagnostics = RetrievalTaskDiagnostics(
        supporting_recall=quality[0],
        gt_evidence_recall=quality[1],
        distractor_avoidance=quality[2],
        exact_quote_validity=quality[3],
        complete_chain_rate=quality[4],
        contextual_oracle_smae_gain=(
            float(coding_score["smae"]) - float(contextual_score["smae"])
        ),
        contextual_oracle_srmse_gain=(
            float(coding_score["srmse"]) - float(contextual_score["srmse"])
        ),
        invalid_count=quality[5] + invalid_candidates,
        catastrophic_count=catastrophic_count,
        chain_credit=tuple(chain_credit),
    )
    return RetrievalCreditReport(
        coding_oracle_smae=float(coding_score["smae"]),
        coding_oracle_srmse=float(coding_score["srmse"]),
        contextual_oracle_smae=float(contextual_score["smae"]),
        contextual_oracle_srmse=float(contextual_score["srmse"]),
        final_smae=float(final_score["smae"]),
        final_srmse=float(final_score["srmse"]),
        decision_smae_regret=float(final_score["smae"]) - float(contextual_score["smae"]),
        decision_srmse_regret=float(final_score["srmse"]) - float(contextual_score["srmse"]),
        chains=tuple(chain_credit),
        diagnostics=diagnostics,
    )


def validate_skill_necessity(
    task: ContextTask,
    result: "HarnessResult",
    chain_id: str,
    *,
    tolerance: float = 1e-12,
) -> tuple[SkillNecessity, ...]:
    """Score explicit pre-label leave-one-Skill-out replays, failing closed if absent."""
    if not task.labels_public or not task.numeric.future_values:
        raise ValueError("Skill necessity requires resolved public labels")
    chain = next(
        (item for item in _chain_ledger(result) if item.chain_id == chain_id),
        None,
    )
    if chain is None:
        raise ValueError(f"unknown evidence chain: {chain_id}")
    full = next(
        (
            snapshot
            for snapshot in getattr(result, "candidate_pool_snapshots", ())
            if snapshot.after_chain_id == chain_id
        ),
        None,
    )
    if full is None:
        raise ValueError("missing full candidate-pool snapshot for Skill replay")
    full_score = _score_pool(task.numeric.future_values, full)[0]
    replays = {
        (replay.chain_id, replay.skill_id): replay.snapshot
        for replay in getattr(result, "skill_leave_one_out_snapshots", ())
    }
    validated = []
    for skill_id in chain.used_skill_ids:
        omitted = replays.get((chain_id, skill_id))
        if omitted is None:
            validated.append(SkillNecessity(skill_id, False))
            continue
        omitted_score = _score_pool(task.numeric.future_values, omitted)[0]
        smae_regret = float(omitted_score["smae"]) - float(full_score["smae"])
        srmse_regret = float(omitted_score["srmse"]) - float(full_score["srmse"])
        necessary = (
            smae_regret >= -tolerance
            and srmse_regret >= -tolerance
            and (smae_regret > tolerance or srmse_regret > tolerance)
        )
        validated.append(
            SkillNecessity(skill_id, necessary, smae_regret, srmse_regret)
        )
    return tuple(validated)


def promote_retrieval_skills(
    library: RetrievalSkillLibrary,
    evidence: Iterable[RetrievalSkillTaskEvidence],
    *,
    tolerance: float = 1e-12,
) -> tuple[str, ...]:
    """Activate candidate Skills only after all cross-Train gates pass."""
    grouped: dict[str, list[RetrievalSkillTaskEvidence]] = {}
    for row in evidence:
        if not isinstance(row, RetrievalSkillTaskEvidence):
            raise TypeError("promotion evidence must be RetrievalSkillTaskEvidence")
        grouped.setdefault(row.skill_id, []).append(row)
    operations = []
    promoted = []
    for skill_id in sorted(grouped):
        rows = grouped[skill_id]
        current = library.get_by_id(skill_id)
        task_ids = tuple(dict.fromkeys(row.task_id for row in rows))
        entities = tuple(sorted({row.entity_name for row in rows}))
        if (
            current is None
            or current.status != "candidate"
            or any(row.split != "train" for row in rows)
            or len(task_ids) < 3
            or len(entities) < 2
            or len(task_ids) != len(rows)
            or any(row.exact_quote_validity != 1.0 for row in rows)
            or any(not row.necessary for row in rows)
            or any(row.added_catastrophic_count != 0 for row in rows)
        ):
            continue
        without_smae = statistics.fmean(row.without_skill_smae for row in rows)
        without_srmse = statistics.fmean(row.without_skill_srmse for row in rows)
        with_smae = statistics.fmean(row.with_skill_smae for row in rows)
        with_srmse = statistics.fmean(row.with_skill_srmse for row in rows)
        smae_gain = without_smae - with_smae
        srmse_gain = without_srmse - with_srmse
        if not (
            smae_gain >= -tolerance
            and srmse_gain >= -tolerance
            and (smae_gain > tolerance or srmse_gain > tolerance)
        ):
            continue
        accepted = replace(
            current,
            status="accepted",
            validated_task_ids=task_ids,
            validated_entities=entities,
            validation_smae_gain=smae_gain,
            validation_srmse_gain=srmse_gain,
        )
        operations.append(RetrievalSkillOperation.repair(skill_id, accepted))
        promoted.append(skill_id)
    if operations:
        library.apply_operations(operations)
    return tuple(promoted)


__all__ = [
    "EvidenceChainCredit",
    "RetrievalCreditReport",
    "RetrievalSkillTaskEvidence",
    "RetrievalTaskDiagnostics",
    "SkillCredit",
    "SkillNecessity",
    "assign_chain_credit",
    "promote_retrieval_skills",
    "validate_skill_necessity",
]
