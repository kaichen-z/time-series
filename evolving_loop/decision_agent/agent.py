"""Decision Agent: choose an executed candidate; never invent forecast values."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from evolving_loop.retrieval_agent.agent import RetrievalResult
from evolving_loop.retrieval_agent.schemas import (
    RetrievalAssumption,
    RetrievalContractError,
    RetrievalGap,
)
from evolving_loop.decision_agent.skill_library import DecisionSkillLibrary
from common.llm import JsonExtractionError, LLMClient, parse_json_object

DECISION_PROMPT = """You are the Decision Agent in a time-series forecasting harness.
Choose among candidates that were already executed and historically hindcast. You cannot write
new values or edit a trajectory. The safe host default is the candidate with the lowest historical
hindcast sRMSE, with sMAE as the tie-breaker. Override it only when verified task evidence
specifically falsifies its assumption or supports another candidate. An override must cite verified
document IDs. If selecting an evidence-adjusted candidate, cite every document used to construct
that candidate.

Judge the complete evidence chain, not isolated quotes. A history-only hindcast cannot reject a
future event absent from history: when verified same-entity evidence jointly supplies the event,
causal mechanism, target, magnitude, and forecast window, prefer the matching executed
evidence-adjusted candidate unless equally strong counterevidence remains. Conversely, abstain when
the chain changes entity or target, ends before the forecast, is contradicted, or lacks a magnitude
or window. If one concrete missing discriminator or unresolved contradiction could change the
selected trajectory, set request_more_retrieval=true; do this only for a named gap, not generic
uncertainty.

When requesting more retrieval, emit one or more exact named gap records using only supplied
Morphology assumption IDs:
{"assumption_id": "a_trend", "gap_type": "continuation_or_reversal",
"missing_information": "Evidence of continuation or reversal", "priority": "high"}.
Use only the enumerated gap types and high, medium, or low priority; never copy candidate fields,
forecast values, scores, source code, or candidate IDs into a gap.

A historical defect can falsify a numeric candidate only when its verified affected window overlaps
the supplied visible history. Earlier defects outside that history must not influence selection.
When a history_cleaned candidate is present, the host has already replayed the same executable
programs on evidence-cleaned history and required both sMAE and sRMSE to be non-worse, with at least
one strict improvement over the raw-host replay on the same targets. Select it only when its cited
observation evidence supports the stated window and repair semantics; cite every source_document_id
attached to the candidate.

Treat measurement/software/sensor errors as observation-layer evidence: they can invalidate
extrapolation of the corrupted historical pattern but do not imply the real process had the same
movement. Treat promotions or demand shocks as latent-process evidence only for their documented
time window. A resolved historical event is not a reason to apply a future multiplier.

Return exactly one JSON object:
{"selected_candidate_id": "candidate_name", "supporting_document_ids": ["doc_1"],
"rationale": "why verified evidence justifies this selection",
"request_more_retrieval": false, "gaps": [], "used_skill_names": []}

If validated decision skills are supplied, use only applicable rules and report their exact names
in used_skill_names. A skill never overrides citation, provenance, or safe-host validation.
"""

_DECISION_RESPONSE_FIELDS = frozenset(
    {
        "selected_candidate_id",
        "supporting_document_ids",
        "rationale",
        "request_more_retrieval",
        "gaps",
        "used_skill_names",
    }
)
_DECISION_TEXT_FIELDS = frozenset({"selected_candidate_id", "rationale"})
_DECISION_TEXT_LIST_FIELDS = frozenset(
    {"supporting_document_ids", "used_skill_names"}
)


@dataclass(frozen=True)
class DecisionCandidate:
    candidate_id: str
    forecast: tuple[float, ...]
    assumption: str
    failure_condition: str
    hindcast_smae: float | None = None
    hindcast_srmse: float | None = None
    source_document_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    hindcast_smape: float | None = None

    def __post_init__(self) -> None:
        """Load pre-sMAE candidates without changing current ranking semantics."""
        fallback = self.hindcast_smape if self.hindcast_smape is not None else 0.0
        if self.hindcast_smae is None:
            object.__setattr__(self, "hindcast_smae", float(fallback))
        if self.hindcast_srmse is None:
            object.__setattr__(self, "hindcast_srmse", float(fallback))
        if self.hindcast_smape is None:
            object.__setattr__(self, "hindcast_smape", float(self.hindcast_smae))


@dataclass(frozen=True)
class DecisionResult:
    selected: DecisionCandidate
    host_default_id: str
    requested_more_retrieval: bool
    rationale: str
    supporting_document_ids: tuple[str, ...]
    llm_override_accepted: bool
    rejection_reason: str | None = None
    used_skill_names: tuple[str, ...] = ()
    gaps: tuple[RetrievalGap, ...] = ()


class DecisionAgent:
    def __init__(
        self,
        llm: LLMClient,
        library: DecisionSkillLibrary | None = None,
        *,
        prompt: str = DECISION_PROMPT,
    ) -> None:
        self.llm = llm
        self.library = library
        self.prompt = prompt

    def run(
        self,
        candidates: tuple[DecisionCandidate, ...],
        retrieval: RetrievalResult,
        *,
        host_default_id: str | None = None,
        prior_decisions: tuple[DecisionResult, ...] = (),
        round_index: int = 0,
        assumptions: tuple[RetrievalAssumption, ...] = (),
    ) -> DecisionResult:
        if not candidates:
            raise ValueError("Decision Agent requires at least one executed candidate")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        host_default = by_id.get(host_default_id) if host_default_id else None
        if host_default is None:
            host_default = min(
                candidates,
                key=lambda item: (item.hindcast_srmse, item.hindcast_smae),
            )
        payload = {
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "forecast": list(item.forecast),
                    "assumption": item.assumption,
                    "failure_condition": item.failure_condition,
                    "hindcast_smae": item.hindcast_smae,
                    "hindcast_srmse": item.hindcast_srmse,
                    "source_document_ids": list(item.source_document_ids),
                    "tags": list(item.tags),
                }
                for item in candidates
            ],
            "host_default_id": host_default.candidate_id,
            "verified_evidence": [asdict(item) for item in retrieval.evidence],
            "verified_impacts": [asdict(item) for item in retrieval.impacts],
            "validated_decision_skills": (
                self.library.list_for_prompt()
                if self.library is not None
                else "(decision skill library disabled)"
            ),
            "decision_round": round_index + 1,
            "prior_decisions": [
                {
                    "selected_candidate_id": item.selected.candidate_id,
                    "rationale": item.rationale,
                    "supporting_document_ids": list(item.supporting_document_ids),
                    "rejection_reason": item.rejection_reason,
                }
                for item in prior_decisions
            ],
            "morphology_assumptions": [item.to_payload() for item in assumptions],
        }
        response = self.llm.complete(
            system=self.prompt,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
        )
        try:
            choice = parse_json_object(response.text)
        except JsonExtractionError as error:
            return self._fallback(host_default, f"invalid_decision_json:{error}")

        missing_fields = _DECISION_RESPONSE_FIELDS - set(choice)
        field_errors = _decision_field_errors(choice)
        unknown_fields = set(choice) - _DECISION_RESPONSE_FIELDS
        raw_requested_more = choice.get("request_more_retrieval", False)
        request_error = (
            None
            if isinstance(raw_requested_more, bool)
            else "request_more_retrieval must be a boolean"
        )
        gaps, gap_error = _parse_gaps(choice.get("gaps", ()), assumptions)
        interface_reasons = []
        if missing_fields:
            interface_reasons.append(
                "invalid_decision_response_schema:missing fields "
                + ",".join(sorted(missing_fields))
            )
        if field_errors:
            interface_reasons.append(
                "invalid_decision_response_schema:" + ",".join(field_errors)
            )
        if gap_error is not None:
            interface_reasons.append("invalid_retrieval_gaps:" + gap_error)
        if unknown_fields:
            interface_reasons.append(
                "forbidden_decision_fields:" + ",".join(sorted(unknown_fields))
            )
        if request_error is not None:
            interface_reasons.append("invalid_retrieval_request:" + request_error)

        raw_candidate_id = choice.get("selected_candidate_id")
        candidate_id = raw_candidate_id if isinstance(raw_candidate_id, str) else ""
        chosen = by_id.get(candidate_id)
        if chosen is None:
            reason = ";".join(interface_reasons) or "unknown_candidate"
            return self._fallback(host_default, reason)
        raw_citations = choice.get("supporting_document_ids")
        cited = (
            tuple(raw_citations)
            if _is_text_list(raw_citations)
            else ()
        )
        verified_ids = {item.document_id for item in retrieval.evidence}
        if not set(cited).issubset(verified_ids):
            return self._fallback(host_default, "unverified_decision_citation")
        override = chosen.candidate_id != host_default.candidate_id
        if override and not cited:
            return self._fallback(host_default, "override_requires_task_evidence")
        if chosen.source_document_ids and not set(chosen.source_document_ids).issubset(cited):
            return self._fallback(host_default, "adjusted_candidate_requires_matching_citations")
        used_skills = []
        unknown_skills = []
        raw_used_skills = choice.get("used_skill_names")
        for name in raw_used_skills if _is_text_list(raw_used_skills) else ():
            if self.library is not None and self.library.get(name) is not None:
                used_skills.append(name)
            else:
                unknown_skills.append(name)
        reasons = list(interface_reasons)
        if unknown_skills:
            reasons.append("unknown_decision_skills:" + ",".join(unknown_skills))
        gap_interface_valid = (
            not missing_fields
            and not field_errors
            and gap_error is None
            and not unknown_fields
            and request_error is None
        )
        if not gap_interface_valid:
            gaps = ()
        requested_more = (
            gap_interface_valid and raw_requested_more is True and bool(gaps)
        )
        return DecisionResult(
            selected=chosen,
            host_default_id=host_default.candidate_id,
            requested_more_retrieval=requested_more,
            rationale=(
                choice["rationale"]
                if isinstance(choice.get("rationale"), str)
                else ""
            ),
            supporting_document_ids=cited,
            llm_override_accepted=override,
            used_skill_names=tuple(used_skills),
            rejection_reason=";".join(reasons) if reasons else None,
            gaps=gaps,
        )

    @staticmethod
    def _fallback(default: DecisionCandidate, reason: str) -> DecisionResult:
        return DecisionResult(
            selected=default,
            host_default_id=default.candidate_id,
            requested_more_retrieval=False,
            rationale="Preserve the best historically validated executable candidate.",
            supporting_document_ids=(),
            llm_override_accepted=False,
            rejection_reason=reason,
            used_skill_names=(),
            gaps=(),
        )


def _decision_field_errors(choice: dict[str, object]) -> tuple[str, ...]:
    errors = []
    for field in sorted(_DECISION_TEXT_FIELDS & set(choice)):
        if not isinstance(choice[field], str):
            errors.append(f"{field} must be a string")
    for field in sorted(_DECISION_TEXT_LIST_FIELDS & set(choice)):
        if not _is_text_list(choice[field]):
            errors.append(f"{field} must be a list of strings")
    return tuple(errors)


def _is_text_list(raw: object) -> bool:
    return isinstance(raw, list) and all(isinstance(item, str) for item in raw)


def _parse_gaps(
    raw: object,
    assumptions: tuple[RetrievalAssumption, ...],
) -> tuple[tuple[RetrievalGap, ...], str | None]:
    """Project only strict named gaps from the richer Decision context."""
    if not isinstance(raw, list):
        return (), "gaps must be a list"
    allowed_ids = {item.assumption_id for item in assumptions}
    try:
        gaps = tuple(RetrievalGap.from_payload(item) for item in raw)
        identities = [item.assumption_id for item in gaps]
        if len(identities) != len(set(identities)):
            raise RetrievalContractError("duplicate retrieval gap assumption_id")
        if not set(identities).issubset(allowed_ids):
            raise RetrievalContractError("retrieval gap references unknown assumption")
    except (RetrievalContractError, TypeError, ValueError) as error:
        return (), str(error)
    return gaps, None
