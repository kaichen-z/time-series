"""Strict, label-free contracts for the two Retrieval inference stages.

The objects in this module are deliberately independent of the Numerical Agent.  They
are the boundary between untrusted JSON and host-owned verification code: every input
mapping is checked for its complete key set before values are converted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from collections.abc import Mapping, Sequence

from evolving_loop.data import ContextTask


class RetrievalContractError(ValueError):
    """Raised when an inference payload violates a Retrieval contract."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_TEMPORAL_RELATIONS = frozenset(
    {"historical", "overlaps_future", "ended_before_future", "unknown"}
)
_MECHANISMS = frozenset(
    {"observation", "latent_process", "future_driver", "regime", "irrelevant"}
)
_DIRECTIONS = frozenset({"up", "down", "stable", "unknown"})
_MAGNITUDE_KINDS = frozenset(
    {"absolute", "relative", "multiplier", "rate", "explicit", "unknown", "none"}
)
_STANCES = frozenset({"supports", "support", "challenges", "challenge", "unresolved", "neutral"})
_ASSUMPTION_KINDS = frozenset(
    {
        "trend_persistence",
        "trend_reversal",
        "level_persistence",
        "seasonality",
        "periodicity",
        "future_event",
        "regime_change",
        "regime_persistence",
        "anomaly_reversion",
        "history_defect",
        "other",
        "unknown",
    }
)
_GAP_TYPES = frozenset(
    {
        "continuation_or_reversal",
        "missing_start",
        "missing_end",
        "missing_magnitude",
        "missing_mechanism",
        "missing_target_link",
        "missing_entity_link",
        "counterevidence",
        "contradiction",
        "other",
    }
)
_PRIORITIES = frozenset({"high", "medium", "low"})


def _mapping(raw: object, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise RetrievalContractError(f"{context} must be an object")
    return raw


def _exact(
    raw: object,
    required: set[str],
    *,
    optional: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> Mapping[str, object]:
    value = _mapping(raw, context)
    if any(not isinstance(key, str) for key in value):
        raise RetrievalContractError(f"invalid {context} keys")
    keys = set(value)
    allowed = required | optional
    if keys != allowed and not (required <= keys <= allowed):
        unknown = keys - allowed
        missing = required - keys
        if unknown:
            raise RetrievalContractError(
                f"forbidden {context} field: {sorted(unknown)[0]}"
            )
        raise RetrievalContractError(f"missing {context} field: {sorted(missing)[0]}")
    return value


def _text(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise RetrievalContractError(f"invalid {field}")
    result = value.strip()
    if nonempty and not result:
        raise RetrievalContractError(f"invalid {field}")
    return result


def _identifier(value: object, field: str) -> str:
    result = _text(value, field)
    if not _IDENTIFIER.fullmatch(result):
        raise RetrievalContractError(f"invalid {field} identifier")
    return result


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise RetrievalContractError(f"invalid {field}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalContractError(f"invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise RetrievalContractError(f"{field} must be finite")
    return result


def _finite_or_none(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field)


def _timestamp(value: object, field: str) -> str | None:
    if value is None:
        return None
    result = _text(value, field)
    try:
        datetime.fromisoformat(result.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise RetrievalContractError(f"invalid {field} timestamp") from error
    return result


def _text_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetrievalContractError(f"invalid {field}")
    return tuple(_text(item, field) for item in value)


def _identifier_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetrievalContractError(f"invalid {field}")
    result = tuple(_identifier(item, field) for item in value)
    if len(result) != len(set(result)):
        raise RetrievalContractError(f"duplicate {field}")
    return result


def _enum(value: object, field: str, values: frozenset[str]) -> str:
    result = _text(value, field)
    if result not in values:
        raise RetrievalContractError(f"invalid {field}: {result}")
    return result


@dataclass(frozen=True)
class RetrievalAssumption:
    assumption_id: str
    kind: str
    claim: str
    failure_condition: str

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalAssumption":
        required = {"assumption_id", "kind", "claim", "failure_condition"}
        value = _exact(raw, required, context="round-two")
        kind = _enum(value["kind"], "assumption kind", _ASSUMPTION_KINDS)
        return cls(
            assumption_id=_identifier(value["assumption_id"], "assumption_id"),
            kind=kind,
            claim=_text(value["claim"], "claim"),
            failure_condition=_text(value["failure_condition"], "failure_condition"),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "assumption_id": self.assumption_id,
            "kind": self.kind,
            "claim": self.claim,
            "failure_condition": self.failure_condition,
        }


@dataclass(frozen=True)
class EvidenceCitation:
    document_id: str
    exact_quote: str

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "EvidenceCitation":
        value = _exact(raw, {"document_id", "exact_quote"}, context="evidence citation")
        return cls(
            document_id=_identifier(value["document_id"], "document_id"),
            exact_quote=_text(value["exact_quote"], "exact_quote"),
        )

    def to_payload(self) -> dict[str, str]:
        return {"document_id": self.document_id, "exact_quote": self.exact_quote}


@dataclass(frozen=True)
class EvidenceChain:
    chain_id: str
    claim: str
    entity_match: bool
    target_match: bool
    temporal_relation: str
    mechanism: str
    direction: str
    magnitude_kind: str
    magnitude_value: float | None
    start_timestamp: str | None
    end_timestamp: str | None
    citations: tuple[EvidenceCitation, ...]
    missing_links: tuple[str, ...]
    used_skill_ids: tuple[str, ...]
    addressed_assumption_ids: tuple[str, ...]
    stance: str
    numeric_eligible: bool

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "EvidenceChain":
        required = {
            "chain_id", "claim", "entity_match", "target_match", "temporal_relation",
            "mechanism", "direction", "magnitude_kind", "magnitude_value",
            "start_timestamp", "end_timestamp", "citations", "missing_links",
            "used_skill_ids", "addressed_assumption_ids", "stance", "numeric_eligible",
        }
        value = _exact(raw, required, context="evidence chain")
        raw_citations = value["citations"]
        if not isinstance(raw_citations, (list, tuple)) or not raw_citations:
            raise RetrievalContractError("evidence chain citations must be non-empty")
        citations = tuple(EvidenceCitation.from_payload(item) for item in raw_citations)
        citation_keys = [(item.document_id, item.exact_quote) for item in citations]
        if len(citation_keys) != len(set(citation_keys)):
            raise RetrievalContractError("duplicate evidence citation")
        magnitude_kind = _enum(value["magnitude_kind"], "magnitude kind", _MAGNITUDE_KINDS)
        magnitude_value = _finite_or_none(value["magnitude_value"], "magnitude_value")
        if magnitude_kind not in {"unknown", "none"} and magnitude_value is None:
            raise RetrievalContractError("magnitude value required for magnitude kind")
        if magnitude_kind in {"unknown", "none"} and magnitude_value is not None:
            raise RetrievalContractError("magnitude value forbidden for unknown magnitude")
        start = _timestamp(value["start_timestamp"], "start_timestamp")
        end = _timestamp(value["end_timestamp"], "end_timestamp")
        if (start is None) != (end is None):
            raise RetrievalContractError("inclusive timestamps must be provided together")
        if start is not None and end is not None:
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                if start_dt > end_dt:
                    raise RetrievalContractError("start_timestamp after end_timestamp")
            except (TypeError, ValueError) as error:  # defensive for mixed aware/naive values
                raise RetrievalContractError("invalid timestamp ordering") from error
        return cls(
            chain_id=_identifier(value["chain_id"], "chain_id"),
            claim=_text(value["claim"], "claim"),
            entity_match=_bool(value["entity_match"], "entity_match"),
            target_match=_bool(value["target_match"], "target_match"),
            temporal_relation=_enum(value["temporal_relation"], "temporal relation", _TEMPORAL_RELATIONS),
            mechanism=_enum(value["mechanism"], "mechanism", _MECHANISMS),
            direction=_enum(value["direction"], "direction", _DIRECTIONS),
            magnitude_kind=magnitude_kind,
            magnitude_value=magnitude_value,
            start_timestamp=start,
            end_timestamp=end,
            citations=citations,
            missing_links=_text_list(value["missing_links"], "missing_links"),
            used_skill_ids=_identifier_list(value["used_skill_ids"], "used_skill_ids"),
            addressed_assumption_ids=_identifier_list(
                value["addressed_assumption_ids"], "addressed_assumption_ids"
            ),
            stance=_enum(value["stance"], "stance", _STANCES),
            numeric_eligible=_bool(value["numeric_eligible"], "numeric_eligible"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "claim": self.claim,
            "entity_match": self.entity_match,
            "target_match": self.target_match,
            "temporal_relation": self.temporal_relation,
            "mechanism": self.mechanism,
            "direction": self.direction,
            "magnitude_kind": self.magnitude_kind,
            "magnitude_value": self.magnitude_value,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "citations": [item.to_payload() for item in self.citations],
            "missing_links": list(self.missing_links),
            "used_skill_ids": list(self.used_skill_ids),
            "addressed_assumption_ids": list(self.addressed_assumption_ids),
            "stance": self.stance,
            "numeric_eligible": self.numeric_eligible,
        }


@dataclass(frozen=True)
class RetrievalGap:
    assumption_id: str
    gap_type: str
    missing_information: str
    priority: str

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalGap":
        value = _exact(
            raw,
            {"assumption_id", "gap_type", "missing_information", "priority"},
            context="retrieval gap",
        )
        return cls(
            assumption_id=_identifier(value["assumption_id"], "assumption_id"),
            gap_type=_enum(value["gap_type"], "gap type", _GAP_TYPES),
            missing_information=_text(value["missing_information"], "missing_information"),
            priority=_enum(value["priority"], "priority", _PRIORITIES),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "assumption_id": self.assumption_id,
            "gap_type": self.gap_type,
            "missing_information": self.missing_information,
            "priority": self.priority,
        }


def _chains(value: object, field: str) -> tuple[EvidenceChain, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetrievalContractError(f"invalid {field}")
    result = tuple(EvidenceChain.from_payload(item) for item in value)
    ids = [item.chain_id for item in result]
    if len(ids) != len(set(ids)):
        raise RetrievalContractError(f"duplicate chain_id in {field}")
    return result


@dataclass(frozen=True)
class RetrievalRoundResult:
    chains: tuple[EvidenceChain, ...]
    counterevidence: tuple[EvidenceChain, ...]
    missing_information: tuple[str, ...]
    sufficient: bool
    gaps: tuple[RetrievalGap, ...] = ()
    rejected: tuple[str, ...] = ()
    unresolved_contradictions: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalRoundResult":
        value = _exact(
            raw,
            {"evidence_chains", "counterevidence", "missing_information", "sufficient"},
            optional={"gaps", "rejected", "unresolved_contradictions"},
            context="retrieval round",
        )
        chains = _chains(value["evidence_chains"], "evidence_chains")
        counterevidence = _chains(value["counterevidence"], "counterevidence")
        ids = [item.chain_id for item in chains + counterevidence]
        if len(ids) != len(set(ids)):
            raise RetrievalContractError("duplicate chain_id across evidence and counterevidence")
        raw_gaps = value.get("gaps", ())
        if not isinstance(raw_gaps, (list, tuple)):
            raise RetrievalContractError("invalid gaps")
        gaps = tuple(RetrievalGap.from_payload(item) for item in raw_gaps)
        gap_ids = [item.assumption_id for item in gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise RetrievalContractError("duplicate retrieval gap assumption_id")
        return cls(
            chains=chains,
            counterevidence=counterevidence,
            missing_information=_text_list(value["missing_information"], "missing_information"),
            sufficient=_bool(value["sufficient"], "sufficient"),
            gaps=gaps,
            rejected=_text_list(value.get("rejected", ()), "rejected"),
            unresolved_contradictions=_text_list(
                value.get("unresolved_contradictions", ()), "unresolved_contradictions"
            ),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "evidence_chains": [item.to_payload() for item in self.chains],
            "counterevidence": [item.to_payload() for item in self.counterevidence],
            "missing_information": list(self.missing_information),
            "sufficient": self.sufficient,
        }
        if self.gaps:
            payload["gaps"] = [item.to_payload() for item in self.gaps]
        if self.rejected:
            payload["rejected"] = list(self.rejected)
        if self.unresolved_contradictions:
            payload["unresolved_contradictions"] = list(self.unresolved_contradictions)
        return payload

    @property
    def evidence_chains(self) -> tuple[EvidenceChain, ...]:
        """The wire-format spelling of :attr:`chains`."""
        return self.chains


@dataclass(frozen=True)
class FinalRetrievalCard:
    round1: RetrievalRoundResult
    round2: RetrievalRoundResult | None
    chains: tuple[EvidenceChain, ...]
    selected_document_ids: tuple[str, ...]
    rejected: tuple[str, ...]
    unresolved_contradictions: tuple[str, ...]
    complete: bool

    @property
    def completeness(self) -> bool:
        """Compatibility spelling for consumers that call the status completeness."""
        return self.complete

    @property
    def merged_chains(self) -> tuple[EvidenceChain, ...]:
        """The descriptive spelling of the deterministic merged chain ledger."""
        return self.chains

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "FinalRetrievalCard":
        value = _exact(
            raw,
            {
                "round1", "round2", "chains", "selected_document_ids", "rejected",
                "unresolved_contradictions", "complete",
            },
            context="final retrieval card",
        )
        round1 = RetrievalRoundResult.from_payload(_mapping(value["round1"], "round1"))
        raw_round2 = value["round2"]
        round2 = None if raw_round2 is None else RetrievalRoundResult.from_payload(
            _mapping(raw_round2, "round2")
        )
        chains = _chains(value["chains"], "merged chains")
        selected = _identifier_list(value["selected_document_ids"], "selected_document_ids")
        return cls(
            round1=round1,
            round2=round2,
            chains=chains,
            selected_document_ids=selected,
            rejected=_text_list(value["rejected"], "rejected"),
            unresolved_contradictions=_text_list(
                value["unresolved_contradictions"], "unresolved_contradictions"
            ),
            complete=_bool(value["complete"], "complete"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "round1": self.round1.to_payload(),
            "round2": self.round2.to_payload() if self.round2 is not None else None,
            "chains": [item.to_payload() for item in self.chains],
            "selected_document_ids": list(self.selected_document_ids),
            "rejected": list(self.rejected),
            "unresolved_contradictions": list(self.unresolved_contradictions),
            "complete": self.complete,
        }


def _skill_payload(skill: object) -> dict[str, object]:
    """Project a skill to advice fields; never expose outcome/evaluation fields."""
    if isinstance(skill, Mapping):
        source = skill
        get = source.get
    else:
        get = lambda name, default=None: getattr(skill, name, default)
    skill_id = get("skill_id", get("name", ""))
    result = {
        "skill_id": _identifier(skill_id, "skill_id"),
        "name": _text(get("name", skill_id), "skill name"),
        "description": _text(get("description", ""), "skill description"),
    }
    for key in ("applicability", "query_strategy", "verification_rule"):
        item = get(key)
        if item is not None:
            result[key] = _text(item, f"skill {key}")
    return result


def _task_target(task: ContextTask) -> dict[str, object]:
    """Construct the explicit safe target view instead of copying task internals."""
    view = task.retrieval_view()
    future = tuple(str(item) for item in view.get("future_timestamps", ()))
    return {
        "entity_name": _text(view.get("entity_name", ""), "entity_name"),
        "target_name": _text(view.get("target_name", ""), "target_name"),
        "description": _text(view.get("target_description", ""), "target_description", nonempty=False),
        "frequency": _text(view.get("frequency", ""), "frequency"),
        "forecast_window": [future[0], future[-1]] if future else [],
    }


def _task_documents(task: ContextTask) -> list[dict[str, str]]:
    view = task.retrieval_view()
    documents = view.get("documents", ())
    if not isinstance(documents, (list, tuple)):
        raise RetrievalContractError("invalid documents")
    result: list[dict[str, str]] = []
    ids: set[str] = set()
    for raw in documents:
        item = _exact(raw, {"document_id", "content"}, context="retrieval document")
        document_id = _identifier(item["document_id"], "document_id")
        if document_id in ids:
            raise RetrievalContractError("duplicate document_id")
        ids.add(document_id)
        result.append({"document_id": document_id, "content": _text(item["content"], "document content", nonempty=False)})
    return result


def build_round1_payload(task: ContextTask, *, skills: Sequence[object] = ()) -> dict[str, object]:
    """Build the assumption-blind input visible to Round 1."""
    return {
        "target": _task_target(task),
        "documents": _task_documents(task),
        "retrieval_skills": [_skill_payload(skill) for skill in skills],
    }


def build_round2_payload(
    task: ContextTask,
    round1: RetrievalRoundResult,
    gaps: Sequence[RetrievalGap | Mapping[str, object]],
    assumptions: Sequence[RetrievalAssumption | Mapping[str, object]],
    skills: Sequence[object] = (),
) -> dict[str, object]:
    """Build Round 2 input from verified evidence and explicitly sanitized records."""
    parsed_gaps = tuple(
        item if isinstance(item, RetrievalGap) else RetrievalGap.from_payload(item)
        for item in gaps
    )
    parsed_assumptions = tuple(
        item if isinstance(item, RetrievalAssumption) else RetrievalAssumption.from_payload(item)
        for item in assumptions
    )
    assumption_ids = [item.assumption_id for item in parsed_assumptions]
    if len(assumption_ids) != len(set(assumption_ids)):
        raise RetrievalContractError("duplicate assumption_id")
    unknown_gaps = set(item.assumption_id for item in parsed_gaps) - set(assumption_ids)
    if unknown_gaps:
        raise RetrievalContractError("retrieval gap references unknown assumption")
    return {
        "target": _task_target(task),
        "documents": _task_documents(task),
        "round1": round1.to_payload(),
        "gaps": [item.to_payload() for item in parsed_gaps],
        "assumptions": [item.to_payload() for item in parsed_assumptions],
        "retrieval_skills": [_skill_payload(skill) for skill in skills],
    }


__all__ = [
    "EvidenceChain",
    "EvidenceCitation",
    "FinalRetrievalCard",
    "RetrievalAssumption",
    "RetrievalContractError",
    "RetrievalGap",
    "RetrievalRoundResult",
    "build_round1_payload",
    "build_round2_payload",
]
