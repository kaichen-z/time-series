"""Decision Agent: choose an executed candidate; never invent forecast values."""
from __future__ import annotations

import json
from dataclasses import dataclass

from evolving_agent.llm import JsonExtractionError, LLMClient, parse_json_object
from evolving_agent.retrieval_agent.agent import RetrievalResult

DECISION_PROMPT = """You are the Decision Agent in a time-series forecasting harness.
Choose among candidates that were already executed and historically hindcast. You cannot write
new values or edit a trajectory. The safe host default is the candidate with the lowest historical
hindcast error. Override it only when verified task evidence specifically falsifies its assumption
or supports another candidate. An override must cite verified document IDs. If selecting an
evidence-adjusted candidate, cite every document used to construct that candidate.

Treat measurement/software/sensor errors as observation-layer evidence: they can invalidate
extrapolation of the corrupted historical pattern but do not imply the real process had the same
movement. Treat promotions or demand shocks as latent-process evidence only for their documented
time window. A resolved historical event is not a reason to apply a future multiplier.

Return exactly one JSON object:
{"selected_candidate_id": "candidate_name", "supporting_document_ids": ["doc_1"],
"rationale": "why verified evidence justifies this selection",
"request_more_retrieval": false}
"""


@dataclass(frozen=True)
class DecisionCandidate:
    candidate_id: str
    forecast: tuple[float, ...]
    assumption: str
    failure_condition: str
    hindcast_smape: float
    source_document_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionResult:
    selected: DecisionCandidate
    host_default_id: str
    requested_more_retrieval: bool
    rationale: str
    supporting_document_ids: tuple[str, ...]
    llm_override_accepted: bool
    rejection_reason: str | None = None


class DecisionAgent:
    def __init__(self, llm: LLMClient, *, prompt: str = DECISION_PROMPT) -> None:
        self.llm = llm
        self.prompt = prompt

    def run(
        self,
        candidates: tuple[DecisionCandidate, ...],
        retrieval: RetrievalResult,
    ) -> DecisionResult:
        if not candidates:
            raise ValueError("Decision Agent requires at least one executed candidate")
        host_default = min(candidates, key=lambda item: item.hindcast_smape)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        payload = {
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "forecast": list(item.forecast),
                    "assumption": item.assumption,
                    "failure_condition": item.failure_condition,
                    "hindcast_smape": item.hindcast_smape,
                    "source_document_ids": list(item.source_document_ids),
                    "tags": list(item.tags),
                }
                for item in candidates
            ],
            "host_default_id": host_default.candidate_id,
            "verified_evidence": [item.__dict__ for item in retrieval.evidence],
            "verified_impacts": [item.__dict__ for item in retrieval.impacts],
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

        candidate_id = str(choice.get("selected_candidate_id", ""))
        chosen = by_id.get(candidate_id)
        if chosen is None:
            return self._fallback(host_default, "unknown_candidate")
        cited = tuple(str(value) for value in choice.get("supporting_document_ids", ()))
        verified_ids = {item.document_id for item in retrieval.evidence}
        if not set(cited).issubset(verified_ids):
            return self._fallback(host_default, "unverified_decision_citation")
        override = chosen.candidate_id != host_default.candidate_id
        if override and not cited:
            return self._fallback(host_default, "override_requires_task_evidence")
        if chosen.source_document_ids and not set(chosen.source_document_ids).issubset(cited):
            return self._fallback(host_default, "adjusted_candidate_requires_matching_citations")
        return DecisionResult(
            selected=chosen,
            host_default_id=host_default.candidate_id,
            requested_more_retrieval=bool(choice.get("request_more_retrieval", False)),
            rationale=str(choice.get("rationale", "")),
            supporting_document_ids=cited,
            llm_override_accepted=override,
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
        )
