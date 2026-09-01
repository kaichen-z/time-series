"""Typed Retrieval-card quality scoring shared by legacy and package hosts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.schemas import FinalRetrievalCard
from evolving_loop.retrieval_agent.verifier import _verified_quote_spans


@dataclass(frozen=True)
class RetrievalCardQuality:
    supporting_recall: float
    gt_evidence_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    rejection_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "supporting_recall",
            "gt_evidence_recall",
            "distractor_avoidance",
            "exact_quote_validity",
            "complete_chain_rate",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            normalized = float(value)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
            object.__setattr__(self, field_name, normalized)
        if (
            isinstance(self.rejection_count, bool)
            or not isinstance(self.rejection_count, int)
            or self.rejection_count < 0
        ):
            raise ValueError("rejection_count must be a non-negative integer")


def _normalized_tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def score_retrieval_card_quality(
    task: ContextTask,
    card: FinalRetrievalCard,
) -> RetrievalCardQuality:
    """Score only verified card artifacts against trusted resolved annotations."""
    if not isinstance(task, ContextTask):
        raise TypeError("Retrieval quality requires a ContextTask")
    if not isinstance(card, FinalRetrievalCard):
        raise TypeError("Retrieval quality requires a FinalRetrievalCard")
    retrieved = set(card.selected_document_ids)
    supporting = {
        document.document_id
        for document in task.documents
        if document.role == "supporting"
    }
    distractors = {
        document.document_id
        for document in task.documents
        if document.role == "distractor"
    }
    supporting_recall = (
        len(retrieved & supporting) / len(supporting) if supporting else 1.0
    )
    distractor_avoidance = (
        1.0 - len(retrieved & distractors) / len(distractors)
        if distractors
        else 1.0
    )

    recovered_text = " ".join(
        value
        for chain in card.chains
        for value in (
            chain.claim,
            *(citation.exact_quote for citation in chain.citations),
        )
    )
    recovered_tokens = _normalized_tokens(recovered_text)
    gt_evidence_recall = (
        sum(
            1
            for evidence in task.gt_evidence
            if (tokens := _normalized_tokens(evidence))
            and tokens.issubset(recovered_tokens)
        )
        / len(task.gt_evidence)
        if task.gt_evidence
        else 1.0
    )

    citations = tuple(
        citation for chain in card.chains for citation in chain.citations
    )
    documents = {
        document.document_id: document.content for document in task.documents
    }
    valid_quotes = sum(
        bool(
            _verified_quote_spans(
                citation.exact_quote,
                documents.get(citation.document_id, ""),
            )
        )
        for citation in citations
    )
    audit_rounds = tuple(
        round_result
        for round_result in (card.round1, card.round2)
        if round_result is not None
    )
    audited_attempts = sum(item.quote_attempt_count for item in audit_rounds)
    audited_valid = sum(item.valid_quote_count for item in audit_rounds)
    if audited_attempts:
        exact_quote_validity = audited_valid / audited_attempts
    else:
        ungrounded = sum(
            str(reason).startswith("ungrounded_quote:")
            for reason in card.rejected
        )
        quote_attempts = len(citations) + ungrounded
        exact_quote_validity = (
            valid_quotes / quote_attempts if quote_attempts else 1.0
        )

    evidence_chains = tuple(card.round1.chains) + (
        tuple(card.round2.chains) if card.round2 is not None else ()
    )
    complete_chain_rate = (
        sum(chain.numeric_eligible for chain in evidence_chains)
        / len(evidence_chains)
        if evidence_chains
        else 0.0
    )
    return RetrievalCardQuality(
        supporting_recall=supporting_recall,
        gt_evidence_recall=gt_evidence_recall,
        distractor_avoidance=distractor_avoidance,
        exact_quote_validity=exact_quote_validity,
        complete_chain_rate=complete_chain_rate,
        rejection_count=len(card.rejected),
    )
