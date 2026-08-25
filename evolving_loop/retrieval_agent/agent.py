"""Hypothesis-guided retrieval with deterministic citation verification."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime

from evolving_loop.coding_agent.evolution import ValidatedProgram
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from common.llm import JsonExtractionError, LLMClient, parse_json_object

RETRIEVAL_PROMPT = """You are the Retrieval Agent for contextual time-series forecasting.
Read the task corpus and the Coding Agent's competing, numbers-only assumptions. Retrieve only
evidence that can distinguish those assumptions or change the future forecast. Ignore documents
about other entities, other time windows, unrelated operations, or unsupported forecasts.

Use the supplied historical timestamps and values as a consistency filter. A document whose
described trajectory, scale, timing, entity, or target conflicts with the observed series is a
time-series distractor unless another cited document explicitly supplies the missing bridge.
Never use or infer future target values.

For every claim, copy an exact quote from one named document. Then classify where the event acts:
- observation: measurement, logging, sensor, or software-recording mechanism
- latent_process: the real target-generating process changed
- future_driver: a scheduled event overlaps the future horizon
- regime: evidence that a temporary regime ended or a stable regime resumes
- irrelevant

Do not turn every textual increase/decrease into a numeric edit. A quantitative edit is allowed
only when the quote explicitly gives its magnitude and time window. Preserve exact entity, target,
unit, magnitude, interval, permanence, and modal qualifiers; do not rewrite a permanent level step
as a continuing trend. If a claim needs multiple documents, cite all of them and set sufficient to
false until the bridge is complete. Return exactly one JSON object:
{"query": "...", "selected_document_ids": ["doc_1"],
"evidence": [{"document_id": "doc_1", "claim": "...", "exact_quote": "..."}],
"impacts": [{"source_document_ids": ["doc_1"], "mechanism_layer": "observation",
"temporal_relation": "historical|overlaps_future|ended_before_future|unknown",
"direction": "up|down|stable|unknown", "permanence": "temporary|permanent|unknown",
"adjustment_kind": "preserve|multiply|add|none", "adjustment_value": null,
"start_timestamp": null, "end_timestamp": null, "rationale": "..."}],
"sufficient": true, "missing_information": [], "used_skill_names": []}

For multiply, adjustment_value is the signed fractional change: 0.20 means +20 percent and
-0.20 means -20 percent. For add, it is an absolute change in target units.

If validated retrieval skills are supplied, use only those that apply. Report their exact names
in used_skill_names. Skills are advice, not evidence; every claim still needs a document quote.
On a follow-up round, use prior_retrieval and prior_decision_feedback to resolve the named gap.
The response must be a complete final snapshot, not a delta: repeat every prior selected document,
evidence item, and impact that remains valid, include corrections and new bridge evidence, and omit
anything now ruled out. Report only missing_information that remains after the complete snapshot.
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
        decision_feedback: dict[str, object] | None = None,
        round_index: int = 0,
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
        payload["retrieval_round"] = round_index + 1
        if prior is not None:
            payload["prior_retrieval"] = {
                "selected_document_ids": list(prior.selected_document_ids),
                "evidence": [asdict(item) for item in prior.evidence],
                "impacts": [asdict(item) for item in prior.impacts],
                "missing_information": list(prior.missing_information),
                "rejections": list(prior.rejected),
                "instruction": (
                    "Return a complete final snapshot. Repeat every still-valid prior "
                    "document, evidence item, and impact; add corrections or bridge "
                    "evidence; omit anything ruled out. Do not return a delta."
                ),
            }
        if decision_feedback is not None:
            payload["prior_decision_feedback"] = decision_feedback
        payload["coding_hypotheses"] = assumptions
        payload["validated_retrieval_skills"] = (
            self.library.list_for_prompt()
            if self.library is not None
            else "(retrieval skill library disabled)"
        )
        response = self.llm.complete(
            system=self.prompt,
            messages=[
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0.0,
        )
        try:
            result = parse_json_object(response.text)
        except JsonExtractionError as error:
            return RetrievalResult(
                "", (), (), (), False, ("invalid_agent_response",), (str(error),), ()
            )
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
            if (
                document is None
                or not quote
                or _normalize(quote) not in _normalize(document.content)
            ):
                rejected.append(f"ungrounded_quote:{document_id}")
                continue
            accepted.append(Evidence(document_id, str(item.get("claim", "")), quote))

        accepted_ids = {item.document_id for item in accepted}
        impacts = []
        blocked_impact = False
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
                quoted = " ".join(
                    item.exact_quote for item in accepted if item.document_id in sources
                )
                if value is None or not _magnitude_is_quoted(
                    value, kind, str(raw.get("direction", "unknown")), quoted
                ):
                    rejected.append("quantitative_impact_without_matching_magnitude")
                    kind, value = "none", None
                    blocked_impact = True
                elif (
                    not start
                    or not end
                    or not _timestamp_is_quoted(start, quoted)
                    or not _timestamp_is_quoted(end, quoted)
                ):
                    rejected.append("quantitative_impact_without_quoted_window")
                    kind, value = "none", None
                    blocked_impact = True
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
        # Every verified evidence document is a final citation.  Keeping a
        # second model-controlled subset would let Decision use evidence that
        # is omitted from exported citations and retrieval diagnostics.
        selected = tuple(dict.fromkeys(item.document_id for item in accepted))
        used_skills = []
        for name in result.get("used_skill_names", ()):
            name = str(name)
            if self.library is not None and self.library.get(name) is not None:
                used_skills.append(name)
            else:
                rejected.append(f"unknown_retrieval_skill:{name}")
        missing_information = tuple(
            str(value) for value in result.get("missing_information", ())
        )
        return RetrievalResult(
            query=str(result.get("query", "")),
            selected_document_ids=selected,
            evidence=tuple(accepted),
            impacts=tuple(impacts),
            sufficient=(
                bool(result.get("sufficient", False))
                and bool(accepted)
                and not blocked_impact
                and not missing_information
            ),
            missing_information=missing_information,
            rejected=tuple(rejected),
            used_skill_names=tuple(used_skills),
        )


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("−", "-").split())


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(?P<number>[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
    r"\s*(?P<percent>%|percent|per\s+cent)?",
    re.IGNORECASE,
)


_UP_WORDS = re.compile(
    r"\b(?:add(?:ed|s|ing)?|boost(?:ed|s|ing)?|gain(?:ed|s|ing)?|grow(?:s|ing|n)?|"
    r"higher|increase(?:d|s|ing)?|raise(?:d|s|ing)?|rise|rises|rising|rose|up)\b",
    re.IGNORECASE,
)
_DOWN_WORDS = re.compile(
    r"\b(?:cut|cuts|cutting|decrease(?:d|s|ing)?|decline(?:d|s|ing)?|down|drop(?:ped|s|ping)?|"
    r"lower|loss|reduce(?:d|s|ing)?|reduction|shrink|shrinks|shrinking)\b",
    re.IGNORECASE,
)
_TEMPORAL_UNIT = re.compile(
    r"^\s*[- ]?(?:business\s+)?"
    r"(?:second|minute|hour|day|week|month|year|interval|reading)s?\b",
    re.IGNORECASE,
)
_DATE_OR_TIME = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*|\s+)\d{4}\b",
    re.IGNORECASE,
)


def _magnitude_is_quoted(value: float, kind: str, direction: str, quote: str) -> bool:
    """Verify an explicitly directed magnitude near its source wording."""
    expected_sign = 1 if value > 0 else -1 if value < 0 else 0
    if expected_sign == 0 or _direction_sign(direction) != expected_sign:
        return False
    date_or_time_spans = [match.span() for match in _DATE_OR_TIME.finditer(quote)]
    for match in _NUMBER_WITH_UNIT.finditer(quote):
        if any(
            match.start() < span_end and span_start < match.end()
            for span_start, span_end in date_or_time_spans
        ):
            continue
        quoted = float(match.group("number").replace(",", ""))
        if kind == "multiply":
            if not match.group("percent"):
                continue
            quoted /= 100.0
        elif match.group("percent"):
            continue
        if not math.isclose(abs(quoted), abs(value), rel_tol=1e-6, abs_tol=1e-9):
            continue
        if _TEMPORAL_UNIT.search(quote[match.end() : match.end() + 32]):
            continue
        if match.group("number").startswith(("+", "-")):
            quoted_sign = 1 if quoted > 0 else -1
        else:
            context = quote[max(0, match.start() - 80) : match.end() + 80]
            quoted_sign = _direction_sign(context)
        if quoted_sign == expected_sign:
            return True
    return False


def _direction_sign(text: str) -> int:
    up = bool(_UP_WORDS.search(text))
    down = bool(_DOWN_WORDS.search(text))
    return 1 if up and not down else -1 if down and not up else 0


def _timestamp_is_quoted(timestamp: str, quote: str) -> bool:
    """Require the literal date, plus the literal time for intraday boundaries."""
    lowered = quote.lower()
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return _normalize(timestamp) in _normalize(quote)
    day = str(parsed.day)
    date_candidates = {
        parsed.strftime("%Y-%m-%d"),
        f"{parsed.strftime('%B')} {day}, {parsed.year}",
        f"{parsed.strftime('%B')} {day} {parsed.year}",
        f"{parsed.strftime('%b')} {day}, {parsed.year}",
        f"{parsed.month}/{parsed.day}/{parsed.year}",
    }
    date_matches = any(candidate.lower() in lowered for candidate in date_candidates)
    has_intraday_time = parsed.time().replace(tzinfo=None) != datetime.min.time()
    if has_intraday_time:
        time_candidates = {parsed.strftime("%H:%M"), parsed.strftime("%H:%M:%S")}
        return date_matches and any(
            candidate.lower() in lowered for candidate in time_candidates
        )
    return date_matches
