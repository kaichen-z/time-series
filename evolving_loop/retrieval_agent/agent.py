"""Hypothesis-guided retrieval with deterministic citation verification."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from evolving_loop.coding_agent.evolution import ValidatedProgram
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from common.llm import JsonExtractionError, LLMClient, parse_json_object

RETRIEVAL_PROMPT = """You are the Retrieval Agent for contextual time-series forecasting.
Read the entire task corpus and the Coding Agent's competing, numbers-only assumptions. Build the
evidence ledger before judging any numeric candidate, so candidate wording does not create
confirmation bias. Ignore documents about other entities, other time windows, unrelated operations,
or unsupported forecasts. Do not use document order, IDs, or corpus position as relevance signals.

Documents may split one effect across a causal chain: one may give an event and window, another the
mechanism, and another the target magnitude. Combine them into one impact only when exact quotes
consistently refer to the same entity, target, event, and forecast window; include every link in
source_document_ids. Search explicitly for negation, containment, cancellation, recovery, and
contradictory chains before declaring an impact sufficient. A nearby event that explicitly did not
affect the target is counterevidence, not a forecast driver. Prefer a coherent multi-document chain
over an isolated plausible statement.

For every claim, copy an exact quote from one named document. Then classify where the event acts:
- observation: measurement, logging, sensor, or software-recording mechanism
- latent_process: the real target-generating process changed
- future_driver: a scheduled event overlaps the future horizon
- regime: evidence that a temporary regime ended or a stable regime resumes
- irrelevant

Use typed operators in adjustment_kind. `multiply` and `add` are future target edits. The operators
`history_mask` and `history_repair` describe a historical observation defect that ended before the
forecast; they invalidate corrupted-history assumptions but never directly change future values.
Use `preserve` or `none` when no numeric operation is justified.
For `history_repair`, a non-null adjustment_value is the signed per-sample correction accumulated
from the first affected sample: step k receives k * adjustment_value. Leave it null for any other
repair form.

Do not turn every textual increase/decrease into a numeric edit. A quantitative future edit is
allowed only when verified quotes jointly provide its magnitude, target, causal link, and affected
window. Cross-document composition is allowed under the same-chain rule above. For "N times usual",
multiply adjustment_value is N-1 (five times means 4.0); for "drops by P percent", use -P/100.
When duration is given, start_timestamp and end_timestamp must be the first and last affected
sample timestamps, never an exclusive boundary. For example, hourly data "from 18:00 for 3 hours"
or "18:00 to 21:00" affects 18:00, 19:00, and 20:00, so end_timestamp is 20:00; include 21:00
only when the evidence explicitly says "through 21:00" or otherwise makes it an affected sample.
Return exactly one JSON object:
{"query": "...", "selected_document_ids": ["doc_1"],
"evidence": [{"document_id": "doc_1", "claim": "...", "exact_quote": "..."}],
"impacts": [{"source_document_ids": ["doc_1"], "mechanism_layer": "observation",
"temporal_relation": "historical|overlaps_future|ended_before_future|unknown",
"direction": "up|down|stable|unknown", "permanence": "temporary|permanent|unknown",
"adjustment_kind": "preserve|multiply|add|history_mask|history_repair|none",
"adjustment_value": null,
"start_timestamp": null, "end_timestamp": null, "rationale": "..."}],
"sufficient": true, "missing_information": [], "used_skill_names": []}

For multiply, adjustment_value is the signed fractional change: 0.20 means +20 percent and
-0.20 means -20 percent. For add, it is an absolute change in target units.

If validated retrieval skills are supplied, use only those that apply. Report their exact names
in used_skill_names. Skills are advice, not evidence; every claim still needs a document quote.
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
    used_skill_names: tuple[str, ...] = ()


class RetrievalAgent:
    def __init__(
        self,
        llm: LLMClient,
        library: RetrievalSkillLibrary | None = None,
        *,
        prompt: str = RETRIEVAL_PROMPT,
    ) -> None:
        self.llm = llm
        self.library = library
        self.prompt = prompt

    def run(
        self,
        task: ContextTask,
        candidates: tuple[ValidatedProgram, ...],
        *,
        prior: RetrievalResult | None = None,
        round_index: int = 0,
    ) -> RetrievalResult:
        assumptions = [
            {
                "candidate_id": candidate.program.name,
                "assumption": candidate.program.assumption,
                "failure_condition": candidate.program.failure_condition,
                "hindcast_smae": candidate.hindcast_smae,
                "hindcast_srmse": candidate.hindcast_srmse,
            }
            for candidate in candidates
        ]
        payload = task.retrieval_view()
        payload["retrieval_round"] = round_index + 1
        if prior is not None:
            payload["prior_retrieval"] = {
                "selected_document_ids": list(prior.selected_document_ids),
                "claims": [item.claim for item in prior.evidence],
                "missing_information": list(prior.missing_information),
                "rejections": list(prior.rejected),
                "instruction": "Fill unresolved gaps, seek counterevidence, and avoid duplicating prior claims.",
            }
        payload["coding_hypotheses"] = assumptions
        payload["validated_retrieval_skills"] = (
            self.library.list_for_prompt()
            if self.library is not None
            else "(retrieval skill library disabled)"
        )
        response = self.llm.complete(
            system=self.prompt,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
        )
        try:
            result = parse_json_object(response.text)
        except JsonExtractionError as error:
            return RetrievalResult("", (), (), (), False, ("invalid_agent_response",), (str(error),), ())
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
            spans = _verified_quote_spans(quote, document.content) if document is not None else ()
            if not spans:
                rejected.append(f"ungrounded_quote:{document_id}")
                continue
            accepted.extend(
                Evidence(document_id, str(item.get("claim", "")), span)
                for span in spans
            )

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
            if kind not in {
                "preserve",
                "multiply",
                "add",
                "history_mask",
                "history_repair",
                "none",
            }:
                rejected.append("unknown_adjustment_kind")
                kind, value = "none", None
            permanence = str(raw.get("permanence", "unknown"))
            start = _optional_text(raw.get("start_timestamp"))
            end = _optional_text(raw.get("end_timestamp"))
            if kind in {"multiply", "add"}:
                quoted = " ".join(item.exact_quote for item in accepted if item.document_id in sources)
                if value is None or not math.isfinite(value) or not re.search(r"\d", quoted):
                    rejected.append("quantitative_impact_without_explicit_magnitude")
                    kind, value = "none", None
                elif kind == "multiply" and not -1.0 <= value <= 20.0:
                    rejected.append("implausible_multiplicative_adjustment")
                    kind, value = "none", None
                elif permanence == "temporary" and (not start or not end):
                    rejected.append("temporary_impact_without_complete_window")
                    kind, value = "none", None
            elif kind in {"history_mask", "history_repair"}:
                if not start or not end:
                    rejected.append("history_adjustment_without_complete_window")
                    kind, value = "none", None
                elif not any(start <= timestamp <= end for timestamp in task.history_timestamps):
                    rejected.append("history_adjustment_outside_visible_history")
                    kind, value = "none", None
                elif kind == "history_repair" and value is not None:
                    quoted = " ".join(
                        item.exact_quote for item in accepted if item.document_id in sources
                    )
                    if not math.isfinite(value) or not re.search(r"\d", quoted):
                        rejected.append("history_repair_without_explicit_finite_rate")
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
        used_skills = []
        for name in result.get("used_skill_names", ()):
            name = str(name)
            if self.library is not None and self.library.get(name) is not None:
                used_skills.append(name)
            else:
                rejected.append(f"unknown_retrieval_skill:{name}")
        return RetrievalResult(
            query=str(result.get("query", "")),
            selected_document_ids=selected,
            evidence=tuple(accepted),
            impacts=tuple(impacts),
            sufficient=bool(result.get("sufficient", False)) and bool(accepted),
            missing_information=tuple(str(value) for value in result.get("missing_information", ())),
            rejected=tuple(rejected),
            used_skill_names=tuple(used_skills),
        )


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("−", "-").split())


def _verified_quote_spans(quote: str, document: str) -> tuple[str, ...]:
    """Accept a full quote or multiple independently exact, non-trivial sentences."""
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
    if len(spans) < 2:
        return ()
    if any(
        len(_normalize(span)) < 32 or _normalize(span) not in normalized_document
        for span in spans
    ):
        return ()
    return spans


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
