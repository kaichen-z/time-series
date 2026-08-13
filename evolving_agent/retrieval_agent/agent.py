"""Hypothesis-guided retrieval with deterministic citation verification."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from evolving_agent.coding_agent.evolution import ValidatedProgram
from evolving_agent.data import ContextTask
from evolving_agent.llm import JsonExtractionError, LLMClient, parse_json_object

RETRIEVAL_PROMPT = """You are the Retrieval Agent for contextual time-series forecasting.
Read the task corpus and the Coding Agent's competing, numbers-only assumptions. Retrieve only
evidence that can distinguish those assumptions or change the future forecast. Ignore documents
about other entities, other time windows, unrelated operations, or unsupported forecasts.

For every claim, copy an exact quote from one named document. Then classify where the event acts:
- observation: measurement, logging, sensor, or software-recording mechanism
- latent_process: the real target-generating process changed
- future_driver: a scheduled event overlaps the future horizon
- regime: evidence that a temporary regime ended or a stable regime resumes
- irrelevant

Do not turn every textual increase/decrease into a numeric edit. A quantitative edit is allowed
only when the quote explicitly gives its magnitude and time window. Return exactly one JSON object:
{"query": "...", "selected_document_ids": ["doc_1"],
"evidence": [{"document_id": "doc_1", "claim": "...", "exact_quote": "..."}],
"impacts": [{"source_document_ids": ["doc_1"], "mechanism_layer": "observation",
"temporal_relation": "historical|overlaps_future|ended_before_future|unknown",
"direction": "up|down|stable|unknown", "permanence": "temporary|permanent|unknown",
"adjustment_kind": "preserve|multiply|add|none", "adjustment_value": null,
"start_timestamp": null, "end_timestamp": null, "rationale": "..."}],
"sufficient": true, "missing_information": []}

For multiply, adjustment_value is the signed fractional change: 0.20 means +20 percent and
-0.20 means -20 percent. For add, it is an absolute change in target units.
"""


@dataclass(frozen=True)
class Evidence:
    document_id: str
    claim: str
    exact_quote: str


@dataclass(frozen=True)
class EvidenceImpact:
    source_document_ids: tuple[str, ...]
    mechanism_layer: str
    temporal_relation: str
    direction: str
    permanence: str
    adjustment_kind: str
    adjustment_value: float | None
    start_timestamp: str | None
    end_timestamp: str | None
    rationale: str


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    selected_document_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    impacts: tuple[EvidenceImpact, ...]
    sufficient: bool
    missing_information: tuple[str, ...]
    rejected: tuple[str, ...] = ()


class RetrievalAgent:
    def __init__(self, llm: LLMClient, *, prompt: str = RETRIEVAL_PROMPT) -> None:
        self.llm = llm
        self.prompt = prompt

    def run(
        self, task: ContextTask, candidates: tuple[ValidatedProgram, ...]
    ) -> RetrievalResult:
        assumptions = [
            {
                "candidate_id": candidate.program.name,
                "assumption": candidate.program.assumption,
                "failure_condition": candidate.program.failure_condition,
                "hindcast_smape": candidate.hindcast_smape,
            }
            for candidate in candidates
        ]
        payload = task.retrieval_view()
        payload["coding_hypotheses"] = assumptions
        response = self.llm.complete(
            system=self.prompt,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
        )
        try:
            result = parse_json_object(response.text)
        except JsonExtractionError as error:
            return RetrievalResult("", (), (), (), False, ("invalid_agent_response",), (str(error),))
        return self._verify(task, result)

    def _verify(self, task: ContextTask, result: dict) -> RetrievalResult:
        documents = {document.document_id: document for document in task.documents}
        accepted = []
        rejected = []
        for item in result.get("evidence", ()):
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id", ""))
            quote = str(item.get("exact_quote", "")).strip()
            document = documents.get(document_id)
            if document is None or not quote or _normalize(quote) not in _normalize(document.content):
                rejected.append(f"ungrounded_quote:{document_id}")
                continue
            accepted.append(Evidence(document_id, str(item.get("claim", "")), quote))

        accepted_ids = {item.document_id for item in accepted}
        impacts = []
        for raw in result.get("impacts", ()):
            if not isinstance(raw, dict):
                continue
            sources = tuple(str(value) for value in raw.get("source_document_ids", ()))
            if not sources or not set(sources).issubset(accepted_ids):
                rejected.append("impact_without_verified_citation")
                continue
            kind = str(raw.get("adjustment_kind", "none"))
            value = raw.get("adjustment_value")
            value = float(value) if isinstance(value, (int, float)) else None
            permanence = str(raw.get("permanence", "unknown"))
            start = _optional_text(raw.get("start_timestamp"))
            end = _optional_text(raw.get("end_timestamp"))
            if kind in {"multiply", "add"}:
                quoted = " ".join(item.exact_quote for item in accepted if item.document_id in sources)
                if value is None or not re.search(r"\d", quoted):
                    rejected.append("quantitative_impact_without_explicit_magnitude")
                    kind, value = "none", None
                elif permanence == "temporary" and (not start or not end):
                    rejected.append("temporary_impact_without_complete_window")
                    kind, value = "none", None
            impacts.append(
                EvidenceImpact(
                    source_document_ids=sources,
                    mechanism_layer=str(raw.get("mechanism_layer", "irrelevant")),
                    temporal_relation=str(raw.get("temporal_relation", "unknown")),
                    direction=str(raw.get("direction", "unknown")),
                    permanence=permanence,
                    adjustment_kind=kind,
                    adjustment_value=value,
                    start_timestamp=start,
                    end_timestamp=end,
                    rationale=str(raw.get("rationale", "")),
                )
            )
        selected = tuple(
            document_id
            for document_id in result.get("selected_document_ids", ())
            if document_id in accepted_ids
        )
        return RetrievalResult(
            query=str(result.get("query", "")),
            selected_document_ids=selected,
            evidence=tuple(accepted),
            impacts=tuple(impacts),
            sufficient=bool(result.get("sufficient", False)) and bool(accepted),
            missing_information=tuple(str(value) for value in result.get("missing_information", ())),
            rejected=tuple(rejected),
        )


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("−", "-").split())


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
