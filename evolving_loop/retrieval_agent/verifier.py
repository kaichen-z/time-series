"""Host-owned verification for untrusted two-stage Retrieval evidence."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Mapping, Sequence

from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.schemas import (
    EvidenceChain,
    EvidenceCitation,
    FinalRetrievalCard,
    RetrievalContractError,
    RetrievalRoundResult,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("−", "-").split())


def _verified_quote_spans(quote: str, document: str) -> tuple[str, ...]:
    """Accept a whole exact quote or independently exact, non-trivial sentences."""
    normalized_document = _normalize(document)
    if not quote:
        return ()
    if _normalize(quote) in normalized_document:
        return (quote,)
    spans = tuple(
        dict.fromkeys(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", quote)
            if part.strip()
        )
    )
    if len(spans) < 2 or any(
        len(_normalize(span)) < 32 or _normalize(span) not in normalized_document
        for span in spans
    ):
        return ()
    return spans


def stable_chain_id(chain: EvidenceChain) -> str:
    """Return an identity independent of model-provided IDs and document ordering."""
    payload = {
        "claim": _normalize(chain.claim),
        "citations": sorted(
            (_normalize(item.document_id), _normalize(item.exact_quote))
            for item in chain.citations
        ),
        "entity_match": chain.entity_match,
        "target_match": chain.target_match,
        "mechanism": chain.mechanism,
        "start": chain.start_timestamp,
        "end": chain.end_timestamp,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "chain_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _deduplicate_raw_citations(payload: object) -> object:
    """Remove repeated untrusted citations before strict schema parsing."""
    if not isinstance(payload, Mapping):
        return payload
    normalized = dict(payload)
    for field in ("evidence_chains", "counterevidence"):
        raw_chains = normalized.get(field)
        if not isinstance(raw_chains, (list, tuple)):
            continue
        chains: list[object] = []
        for raw_chain in raw_chains:
            if not isinstance(raw_chain, Mapping):
                chains.append(raw_chain)
                continue
            chain = dict(raw_chain)
            citations = chain.get("citations")
            if isinstance(citations, (list, tuple)):
                seen: set[tuple[object, object]] = set()
                deduplicated = []
                for citation in citations:
                    if not isinstance(citation, Mapping):
                        deduplicated.append(citation)
                        continue
                    key = (citation.get("document_id"), citation.get("exact_quote"))
                    if key not in seen:
                        seen.add(key)
                        deduplicated.append(citation)
                chain["citations"] = deduplicated
            chains.append(chain)
        normalized[field] = chains
    return normalized


def _missing(existing: Sequence[str], *additions: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*existing, *additions)))


def _citation_spans(
    task: ContextTask, citations: Sequence[EvidenceCitation]
) -> tuple[tuple[EvidenceCitation, ...], tuple[str, ...]]:
    documents = {document.document_id: document.content for document in task.documents}
    accepted: list[EvidenceCitation] = []
    rejected: list[str] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        document = documents.get(citation.document_id)
        spans = _verified_quote_spans(citation.exact_quote, document) if document is not None else ()
        if not spans:
            rejected.append(f"ungrounded_quote:{citation.document_id}")
            continue
        for span in spans:
            key = (citation.document_id, span)
            if key not in seen:
                seen.add(key)
                accepted.append(EvidenceCitation(citation.document_id, span))
    return tuple(accepted), tuple(rejected)


def _has_term(text: str, term: str) -> bool:
    return _normalize(term) in _normalize(text)


def _verify_chain(
    task: ContextTask,
    chain: EvidenceChain,
    *,
    allowed_skill_ids: frozenset[str],
    allowed_assumption_ids: frozenset[str],
    counterevidence: bool,
) -> tuple[EvidenceChain, tuple[str, ...]]:
    citations, rejected = _citation_spans(task, chain.citations)
    quote_text = " ".join(item.exact_quote for item in citations)
    missing: list[str] = list(chain.missing_links)
    if not citations:
        missing.append("citation")
    if not _has_term(quote_text, task.numeric.entity_name):
        missing.append("entity")
    if not _has_term(quote_text, task.target_name):
        missing.append("target")
    if chain.mechanism == "irrelevant":
        missing.append("mechanism")
    if chain.direction == "unknown":
        missing.append("direction")
    future = set(task.future_timestamps)
    if (
        chain.temporal_relation != "overlaps_future"
        or chain.start_timestamp not in future
        or chain.end_timestamp not in future
    ):
        missing.append("forecast_window")
    if chain.magnitude_kind in {"unknown", "none"} or chain.magnitude_value is None:
        missing.append("magnitude")
    elif not re.search(r"\d", re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", "", quote_text)):
        missing.append("magnitude")
    elif chain.magnitude_kind == "multiplier" and not 0.0 < chain.magnitude_value <= 21.0:
        missing.append("multiplier")
    unknown_skills = tuple(
        item for item in chain.used_skill_ids if item not in allowed_skill_ids
    )
    if unknown_skills:
        missing.append("skill")
        rejected += tuple(f"unknown_retrieval_skill:{item}" for item in unknown_skills)
    unknown_assumptions = tuple(
        item for item in chain.addressed_assumption_ids
        if item not in allowed_assumption_ids
    )
    if unknown_assumptions:
        missing.append("assumption")
        rejected += tuple(f"unknown_assumption:{item}" for item in unknown_assumptions)
    missing_links = _missing(missing)
    verified = replace(
        chain,
        chain_id=chain.chain_id,
        entity_match="entity" not in missing_links,
        target_match="target" not in missing_links,
        citations=citations,
        missing_links=missing_links,
        used_skill_ids=tuple(item for item in chain.used_skill_ids if item in allowed_skill_ids),
        addressed_assumption_ids=tuple(
            item for item in chain.addressed_assumption_ids if item in allowed_assumption_ids
        ),
        numeric_eligible=False,
    )
    numeric_eligible = not missing_links and not counterevidence
    verified = replace(verified, numeric_eligible=numeric_eligible)
    return replace(verified, chain_id=stable_chain_id(verified)), rejected


def _verify_collection(
    task: ContextTask,
    chains: Sequence[EvidenceChain],
    *,
    allowed_skill_ids: frozenset[str],
    allowed_assumption_ids: frozenset[str],
    counterevidence: bool,
) -> tuple[tuple[EvidenceChain, ...], tuple[str, ...]]:
    verified: list[EvidenceChain] = []
    rejected: list[str] = []
    identities: set[str] = set()
    for chain in chains:
        item, item_rejected = _verify_chain(
            task,
            chain,
            allowed_skill_ids=allowed_skill_ids,
            allowed_assumption_ids=allowed_assumption_ids,
            counterevidence=counterevidence,
        )
        rejected.extend(item_rejected)
        if item.chain_id in identities:
            rejected.append("duplicate_chain_identity")
            continue
        identities.add(item.chain_id)
        verified.append(item)
    return tuple(verified), tuple(rejected)


def verify_round_result(
    task: ContextTask,
    payload: Mapping[str, object],
    *,
    stage: str,
    allowed_skill_ids: Sequence[str],
    allowed_assumption_ids: Sequence[str],
) -> RetrievalRoundResult:
    """Parse and deterministically verify one untrusted Retrieval stage response."""
    if stage not in {"round1", "round2"}:
        raise ValueError("stage must be round1 or round2")
    try:
        raw = RetrievalRoundResult.from_payload(_deduplicate_raw_citations(payload))
    except RetrievalContractError as error:
        return RetrievalRoundResult((), (), ("invalid_retrieval_payload",), False, rejected=(str(error),))
    skill_ids = frozenset(allowed_skill_ids)
    assumption_ids = frozenset(allowed_assumption_ids)
    chains, chain_rejected = _verify_collection(
        task,
        raw.chains,
        allowed_skill_ids=skill_ids,
        allowed_assumption_ids=assumption_ids,
        counterevidence=False,
    )
    counter, counter_rejected = _verify_collection(
        task,
        raw.counterevidence,
        allowed_skill_ids=skill_ids,
        allowed_assumption_ids=assumption_ids,
        counterevidence=True,
    )
    all_ids = {item.chain_id for item in chains}
    unique_counter = tuple(item for item in counter if item.chain_id not in all_ids)
    if len(unique_counter) != len(counter):
        counter_rejected += ("duplicate_chain_identity",)
    rejected = tuple(dict.fromkeys((*raw.rejected, *chain_rejected, *counter_rejected)))
    return RetrievalRoundResult(
        chains=chains,
        counterevidence=unique_counter,
        missing_information=raw.missing_information,
        sufficient=raw.sufficient and bool(chains),
        gaps=raw.gaps,
        rejected=rejected,
        unresolved_contradictions=raw.unresolved_contradictions,
    )


def merge_verified_rounds(
    round1: RetrievalRoundResult,
    round2: RetrievalRoundResult | None,
) -> FinalRetrievalCard:
    """Append Round 2 verification without allowing it to rewrite Round 1."""
    if round2 is None:
        round2_chains: tuple[EvidenceChain, ...] = ()
        round2_counter: tuple[EvidenceChain, ...] = ()
        rejected: list[str] = list(round1.rejected)
    else:
        round2_chains = round2.chains
        round2_counter = round2.counterevidence
        rejected = [*round1.rejected, *round2.rejected]
    merged: list[EvidenceChain] = list(round1.chains)
    identities = {item.chain_id for item in merged}
    for chain in round2_chains:
        if chain.chain_id in identities:
            if chain != next(item for item in merged if item.chain_id == chain.chain_id):
                rejected.append("round2_chain_identity_conflict")
            continue
        identities.add(chain.chain_id)
        merged.append(chain)
    selected = tuple(dict.fromkeys(
        citation.document_id for chain in merged for citation in chain.citations
    ))
    unresolved = tuple(dict.fromkeys(
        value
        for result in (round1, round2)
        if result is not None
        for value in result.unresolved_contradictions
    ))
    return FinalRetrievalCard(
        round1=round1,
        round2=round2,
        chains=tuple(merged),
        selected_document_ids=selected,
        rejected=tuple(dict.fromkeys(rejected)),
        unresolved_contradictions=unresolved,
        complete=bool(merged) and all(item.numeric_eligible for item in merged),
    )


__all__ = [
    "_verified_quote_spans",
    "merge_verified_rounds",
    "stable_chain_id",
    "verify_round_result",
]
