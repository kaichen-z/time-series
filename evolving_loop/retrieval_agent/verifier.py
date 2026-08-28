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
    RetrievalGap,
    RetrievalRoundResult,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("−", "-").split())


_CONTRACTION_EXPANSIONS = {
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "won't": "will not",
    "can't": "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
}
_CONTRACTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(item) for item in _CONTRACTION_EXPANSIONS) + r")\b"
)


def _normalize_semantics(text: str) -> str:
    """Expand explicit contractions for matching without altering evidence text."""
    normalized = _normalize(text).replace("‘", "'").replace("’", "'")
    return _CONTRACTION_PATTERN.sub(
        lambda match: _CONTRACTION_EXPANSIONS[match.group(0)],
        normalized,
    )


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
        "entity": _normalize(chain.canonical_entity),
        "target": _normalize(chain.canonical_target),
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
                    document_id = citation.get("document_id")
                    exact_quote = citation.get("exact_quote")
                    if not isinstance(document_id, str) or not isinstance(exact_quote, str):
                        deduplicated.append(citation)
                        continue
                    key = (document_id, exact_quote)
                    if key not in seen:
                        seen.add(key)
                        deduplicated.append(citation)
                chain["citations"] = deduplicated
            chains.append(chain)
        normalized[field] = chains
    return normalized


def _raw_quote_audit(
    task: ContextTask, payload: object
) -> tuple[int, int]:
    """Count submitted quote attempts before any wire-level deduplication."""
    if not isinstance(payload, Mapping):
        return 0, 0
    documents = {document.document_id: document.content for document in task.documents}
    attempts = 0
    valid = 0
    for field in ("evidence_chains", "counterevidence"):
        chains = payload.get(field, ())
        if not isinstance(chains, (list, tuple)):
            continue
        for chain in chains:
            if not isinstance(chain, Mapping):
                continue
            citations = chain.get("citations", ())
            if not isinstance(citations, (list, tuple)):
                continue
            for citation in citations:
                if not isinstance(citation, Mapping):
                    continue
                attempts += 1
                document_id = citation.get("document_id")
                quote = citation.get("exact_quote")
                if (
                    isinstance(document_id, str)
                    and isinstance(quote, str)
                    and _verified_quote_spans(quote, documents.get(document_id, ""))
                ):
                    valid += 1
    return attempts, valid


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


def _entity_target_spans(
    task: ContextTask, citations: Sequence[EvidenceCitation]
) -> tuple[EvidenceCitation, ...]:
    return tuple(
        citation
        for citation in citations
        if _has_term(citation.exact_quote, task.numeric.entity_name)
        and _has_term(citation.exact_quote, task.target_name)
    )


def _has_direction(span: str, direction: str) -> bool:
    normalized = _normalize(span)
    terms = {
        "up": ("increase", "increased", "rises", "rise", "boost", "gain", "higher"),
        "down": ("decrease", "decreased", "drop", "decline", "lower", "reduce"),
        "stable": ("stable", "unchanged", "constant", "flat"),
    }
    for term in terms.get(direction, ()):
        for match in re.finditer(rf"\b{re.escape(term)}\b", normalized):
            prefix = normalized[max(0, match.start() - 48):match.start()]
            if re.search(r"\b(?:not|never|without|no\s+longer)(?:\s+\w+){0,2}\s*$", prefix):
                continue
            return True
    return False


def _has_mechanism(span: str, mechanism: str) -> bool:
    normalized = _normalize(span)
    terms = {
        "observation": ("measurement", "recording", "sensor", "logging", "reported"),
        "latent_process": ("process", "demand", "production", "consumption", "behavior"),
        "future_driver": ("future", "scheduled", "will ", "planned", "upcoming"),
        "regime": ("regime", "temporary", "recovery", "resumes", "ended"),
    }
    return any(term in normalized for term in terms.get(mechanism, ()))


def _without_timestamps(text: str) -> str:
    return re.sub(
        r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b",
        "",
        text,
    )


_TIMESTAMP = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"


def _cited_intervals(span: str) -> tuple[tuple[str, str], ...]:
    """Extract only explicit inclusive ranges; unknown or multiple ranges fail closed."""
    return tuple(
        re.findall(
            rf"\bfrom\s+({_TIMESTAMP})\s+(?:through|to)\s+({_TIMESTAMP})\b",
            span,
            flags=re.IGNORECASE,
        )
    )


_EVENT_STATUS = (
    r"(?:abort(?:ed|ion)?|cancel(?:led|ed|lation)?|called\s+off|"
    r"postpon(?:ed|ement)?|defer(?:red|ral)?|suspend(?:ed|sion)?|"
    r"reschedul(?:ed|ing)|withdrawn|withdrew|withdraw(?:al)?)"
)
_PASSIVE_AUXILIARY = re.compile(
    r"\b(?:was|were|is|are|has\s+been|have\s+been|had\s+been|"
    r"will\s+be|would\s+be)\s+(?:(not|never)\s+)?$"
)
_EVENT_TOKEN_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "cause", "caused", "causes", "causing", "documented", "driver", "event",
    "for", "from", "future", "had", "has", "have", "higher", "in", "increase",
    "increased", "increases", "increasing", "is", "it", "its", "lower", "of",
    "on", "percent", "planned", "prior", "scheduled", "says", "stable", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "upcoming", "was", "were", "will", "with", "would",
})


def _event_tokens(text: str, task: ContextTask) -> frozenset[str]:
    excluded = {
        *_EVENT_TOKEN_STOPWORDS,
        *re.findall(r"[a-z]+", _normalize(task.numeric.entity_name)),
        *re.findall(r"[a-z]+", _normalize(task.target_name)),
    }
    return frozenset(
        token
        for token in re.findall(r"[a-z]+", _normalize(text))
        if token not in excluded
        and not re.fullmatch(_EVENT_STATUS, token)
    )


def _clause_tail(text: str) -> str:
    return re.split(
        r"[;,:]|\b(?:but|however|although|while|whereas)\b",
        text,
        flags=re.IGNORECASE,
    )[-1]


def _status_event_mentions(sentence: str, task: ContextTask) -> tuple[frozenset[str], ...]:
    """Return affected events for active cancellation/non-occurrence statements."""
    normalized = _normalize_semantics(sentence)
    mentions: list[frozenset[str]] = []
    for match in re.finditer(rf"\b{_EVENT_STATUS}\b", normalized):
        prefix = normalized[:match.start()]
        if re.search(
            r"\b(?:did|does|do|will|would|should|has|have|had|can|could)\s+"
            r"(?:not|never)\s+$",
            prefix,
        ):
            continue
        passive = (
            None
            if match.group(0) == "rescheduling"
            else _PASSIVE_AUXILIARY.search(prefix)
        )
        if passive is not None:
            if passive.group(1) is not None:
                continue
            event_text = _clause_tail(prefix[:passive.start()])
        else:
            event_text = re.split(
                r"[.;,:]|\b(?:but|however|although|while|whereas)\b",
                normalized[match.end():],
                maxsplit=1,
            )[0]
        mentions.append(_event_tokens(event_text, task))

    nonoccurrence_patterns = (
        r"\b(?:will|would|should|did|does|do|can|could)\s+not\s+"
        r"(?:occur|happen|take\s+place|go\s+ahead)\b",
        r"\b(?:no\s+longer|never)\s+"
        r"(?:occurs?|occurred|happens?|happened|takes?\s+place|took\s+place|goes?\s+ahead)\b",
    )
    for pattern in nonoccurrence_patterns:
        for match in re.finditer(pattern, normalized):
            mentions.append(_event_tokens(_clause_tail(normalized[:match.start()]), task))
    return tuple(mentions)


def _claim_event_tokens(task: ContextTask, claim: str, span: str) -> frozenset[str]:
    tokens = set(_event_tokens(claim, task))
    entity = _normalize(task.numeric.entity_name)
    target = _normalize(task.target_name)
    for sentence in re.split(r"(?<=[.!?])\s+", _normalize(span)):
        if entity not in sentence or target not in sentence:
            continue
        entity_start = sentence.find(entity)
        prefix = sentence[:entity_start]
        if re.search(r"\b(?:cause|causes|caused|causing|drive|drives|driven)\b", prefix):
            tokens.update(_event_tokens(prefix, task))
    return frozenset(tokens)


def _has_cancellation(task: ContextTask, claim: str, span: str) -> bool:
    """Match cancellation only to the claimed event; unresolved references fail closed."""
    claimed_event = _claim_event_tokens(task, claim, span)
    for sentence in re.split(r"(?<=[.!?])\s+", span):
        for status_event in _status_event_mentions(sentence, task):
            if not status_event or not claimed_event:
                return True
            if status_event & claimed_event:
                return True
    return False


def _numbers(text: str) -> tuple[float, ...]:
    return tuple(
        float(item.replace(",", ""))
        for item in re.findall(r"(?<![\w.-])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.-])", _without_timestamps(text))
    )


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-6)


def _magnitude_matches(span: str, kind: str, value: float) -> bool:
    normalized = _normalize(span)
    if kind == "multiplier":
        factor_tokens = re.findall(
            r"(?<![\w.])((?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?))\s*(?:x\b|times\b|multipl\w*)",
            normalized.replace("×", "x"),
        )
        return any(_close(float(token.replace(",", "")), value) for token in factor_tokens)
    numbers = _numbers(span)
    if not numbers:
        return False
    if kind == "relative":
        if not any(marker in normalized for marker in ("%", "percent", "percentage", "rate")):
            return False
        expected = (value * 100.0,)
    else:
        expected = (value,)
    return any(_close(number, candidate) for number in numbers for candidate in expected)


def _magnitude_is_domain_safe(task: ContextTask, kind: str, value: float) -> bool:
    if kind == "absolute":
        history_scale = max((abs(item) for item in task.numeric.history_values), default=0.0)
        return 0.0 < value <= max(1.0, history_scale) * 100.0
    if kind == "relative":
        return 0.0 < value <= 1.0
    return 0.0 < value <= 21.0 if kind == "multiplier" else False


def _canonical_adjustment(kind: str, value: float, direction: str) -> float | None:
    if direction not in {"up", "down"}:
        return None
    sign = 1.0 if direction == "up" else -1.0
    if kind in {"absolute", "relative"}:
        return sign * value
    if kind == "multiplier":
        adjustment = value - 1.0
        if (direction == "up" and adjustment <= 0.0) or (
            direction == "down" and adjustment >= 0.0
        ):
            return None
        return adjustment
    return None


def _verify_chain(
    task: ContextTask,
    chain: EvidenceChain,
    *,
    allowed_skill_ids: frozenset[str],
    allowed_assumption_ids: frozenset[str],
    counterevidence: bool,
) -> tuple[EvidenceChain, tuple[str, ...]]:
    citations, rejected = _citation_spans(task, chain.citations)
    anchors = _entity_target_spans(task, citations)
    missing: list[str] = list(chain.missing_links)
    if not citations:
        missing.append("citation")
    if not any(_has_term(item.exact_quote, task.numeric.entity_name) for item in citations):
        missing.append("entity")
    if not any(_has_term(item.exact_quote, task.target_name) for item in citations):
        missing.append("target")
    if chain.mechanism == "irrelevant" or not any(
        _has_mechanism(item.exact_quote, chain.mechanism) for item in anchors
    ):
        missing.append("mechanism")
    if chain.direction == "unknown" or not any(
        _has_direction(item.exact_quote, chain.direction) for item in anchors
    ):
        missing.append("direction")
    future_indexes = {timestamp: index for index, timestamp in enumerate(task.future_timestamps)}
    cited_intervals = {
        interval
        for item in anchors
        for interval in _cited_intervals(item.exact_quote)
    }
    if (
        chain.temporal_relation != "overlaps_future"
        or chain.start_timestamp not in future_indexes
        or chain.end_timestamp not in future_indexes
        or future_indexes.get(chain.start_timestamp, 0) > future_indexes.get(chain.end_timestamp, -1)
        or cited_intervals != {(chain.start_timestamp, chain.end_timestamp)}
    ):
        missing.append("forecast_window")
    if any(_has_cancellation(task, chain.claim, item.exact_quote) for item in anchors):
        missing.append("causal_status")
    if chain.magnitude_kind in {"unknown", "none"} or chain.magnitude_value is None:
        missing.append("magnitude")
    elif not _magnitude_is_domain_safe(task, chain.magnitude_kind, chain.magnitude_value):
        missing.append("multiplier" if chain.magnitude_kind == "multiplier" else "magnitude_value")
    elif not any(
        _magnitude_matches(item.exact_quote, chain.magnitude_kind, chain.magnitude_value)
        for item in anchors
    ):
        missing.append("magnitude")
    canonical_adjustment = _canonical_adjustment(
        chain.magnitude_kind,
        chain.magnitude_value,
        chain.direction,
    ) if chain.magnitude_value is not None else None
    if chain.magnitude_value is not None and canonical_adjustment is None:
        missing.append("magnitude_value")
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
        canonical_entity=task.numeric.entity_name,
        canonical_target=task.target_name,
        legacy_adjustment_value=canonical_adjustment,
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
    submitted_conflicts: frozenset[str] = frozenset(),
) -> tuple[tuple[EvidenceChain, ...], tuple[str, ...]]:
    verified: list[EvidenceChain] = []
    rejected: list[str] = []
    identities: set[str] = set()
    for chain in chains:
        if chain.chain_id in submitted_conflicts:
            rejected.append("round2_chain_identity_conflict")
            continue
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
    prior_round1: RetrievalRoundResult | None = None,
) -> RetrievalRoundResult:
    """Parse and deterministically verify one untrusted Retrieval stage response."""
    if stage not in {"round1", "round2"}:
        raise ValueError("stage must be round1 or round2")
    quote_attempt_count, valid_quote_count = _raw_quote_audit(task, payload)
    try:
        raw = RetrievalRoundResult.from_payload(_deduplicate_raw_citations(payload))
    except (RetrievalContractError, TypeError, ValueError) as error:
        return RetrievalRoundResult(
            (), (), ("invalid_retrieval_payload",), False,
            rejected=(str(error),),
            quote_attempt_count=quote_attempt_count,
            valid_quote_count=valid_quote_count,
        )
    skill_ids = frozenset(allowed_skill_ids)
    assumption_ids = frozenset(allowed_assumption_ids)
    submitted_conflicts = (
        frozenset(item.chain_id for item in (*prior_round1.chains, *prior_round1.counterevidence))
        if stage == "round2" and prior_round1 is not None
        else frozenset()
    )
    chains, chain_rejected = _verify_collection(
        task,
        raw.chains,
        allowed_skill_ids=skill_ids,
        allowed_assumption_ids=assumption_ids,
        counterevidence=False,
        submitted_conflicts=submitted_conflicts,
    )
    counter, counter_rejected = _verify_collection(
        task,
        raw.counterevidence,
        allowed_skill_ids=skill_ids,
        allowed_assumption_ids=assumption_ids,
        counterevidence=True,
        submitted_conflicts=submitted_conflicts,
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
        quote_attempt_count=quote_attempt_count,
        valid_quote_count=valid_quote_count,
    )


def merge_verified_rounds(
    round1: RetrievalRoundResult,
    round2: RetrievalRoundResult | None,
    *,
    gaps: Sequence[RetrievalGap] = (),
) -> FinalRetrievalCard:
    """Append Round 2 verification without allowing it to rewrite Round 1."""
    final_gaps = tuple(gaps)
    if any(not isinstance(item, RetrievalGap) for item in final_gaps):
        raise RetrievalContractError("invalid final retrieval gaps")
    gap_ids = [item.assumption_id for item in final_gaps]
    if len(gap_ids) != len(set(gap_ids)):
        raise RetrievalContractError("duplicate final retrieval gap assumption_id")
    if round2 is None:
        round2_chains: tuple[EvidenceChain, ...] = ()
        round2_counter: tuple[EvidenceChain, ...] = ()
        rejected: list[str] = list(round1.rejected)
    else:
        round2_chains = round2.chains
        round2_counter = round2.counterevidence
        rejected = [*round1.rejected, *round2.rejected]
    merged: list[EvidenceChain] = [*round1.chains, *round1.counterevidence]
    identities = {item.chain_id for item in merged}
    for chain in round2_chains:
        if chain.chain_id in identities:
            if chain != next(item for item in merged if item.chain_id == chain.chain_id):
                rejected.append("round2_chain_identity_conflict")
            continue
        identities.add(chain.chain_id)
        merged.append(chain)
    for chain in round2_counter:
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
        complete=bool(round1.chains or round2_chains) and all(
            item.numeric_eligible for item in (*round1.chains, *round2_chains)
        ),
        gaps=final_gaps,
    )


__all__ = [
    "_verified_quote_spans",
    "merge_verified_rounds",
    "stable_chain_id",
    "verify_round_result",
]
