from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, time

from .models import Diagnosis, Evidence, EvidenceImpact, ForecastTask, RetrievedDocument


ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?\b")
NATURAL_DATE_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}"
)
NATURAL_DATE_RE = re.compile(NATURAL_DATE_PATTERN, re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
UP_WORDS = {
    "increase", "increased", "increases", "increasing", "higher", "rise", "rises",
    "rising", "growth", "grow", "grows", "surge", "spike", "elevated", "boost",
}
DOWN_WORDS = {
    "decrease", "decreased", "decreases", "decreasing", "lower", "decline", "declines",
    "fall", "falls", "drop", "drops", "reduction", "reduced", "suppress", "suppresses",
}
RESOLVED_PHRASES = (
    "returned to baseline", "return to baseline", "returned to normal", "return to normal",
    "restored normal", "restored to normal", "resolved", "fixed", "patched", "removed",
    "no longer", "would not happen again", "will not happen again", "concluded", "ended",
    "expired", "reinstated", "stabilized",
)
TEMPORARY_WORDS = {"temporary", "temporarily", "campaign", "promotion", "discount", "event"}
PERMANENT_WORDS = {"permanent", "permanently", "lasting", "structural", "persistent", "sustained"}
FUTURE_WORDS = {"future", "forecast", "upcoming", "scheduled", "projected", "expected", "will"}


def _parse_timestamp(value: str, end_of_day: bool = False) -> datetime:
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)", "", value, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    try:
        parsed = datetime.fromisoformat(cleaned.replace("T", " "))
    except ValueError:
        parsed = datetime.strptime(cleaned, "%B %d %Y")
    if len(value) == 10 and end_of_day:
        return datetime.combine(parsed.date(), time.max)
    return parsed


def _event_window(text: str) -> tuple[str | None, str | None]:
    range_match = re.search(
        r"(?is)(?:from|between)\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)"
        r".{0,40}?(?:to|through|until|and)\s+"
        r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)",
        text,
    )
    if range_match:
        return range_match.group(1), range_match.group(2)
    natural_range = re.search(
        rf"(?is)(?:from|between)\s+({NATURAL_DATE_PATTERN})"
        rf".{{0,40}}?(?:to|through|until|and)\s+({NATURAL_DATE_PATTERN})",
        text,
    )
    if natural_range:
        return (
            _parse_timestamp(natural_range.group(1)).date().isoformat(),
            _parse_timestamp(natural_range.group(2)).date().isoformat(),
        )
    dates = ISO_DATE_RE.findall(text)
    dates.extend(
        _parse_timestamp(value).date().isoformat()
        for value in NATURAL_DATE_RE.findall(text)
    )
    if not dates:
        return None, None
    if len(dates) == 1:
        return dates[0], None
    parsed = sorted((_parse_timestamp(value), value) for value in dates)
    return parsed[0][1], parsed[-1][1]


def _direction(text: str) -> str:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    up = len(tokens & UP_WORDS)
    down = len(tokens & DOWN_WORDS)
    return "up" if up > down else "down" if down > up else "unclear"


def _permanence(text: str) -> str:
    lower = text.lower()
    tokens = set(re.findall(r"[a-z]+", lower))
    if any(phrase in lower for phrase in RESOLVED_PHRASES):
        return "resolved"
    if tokens & PERMANENT_WORDS:
        return "permanent"
    if tokens & TEMPORARY_WORDS:
        return "temporary"
    return "unspecified"


def _quantified_effect(text: str) -> tuple[str | None, float | None, str]:
    for sentence in SENTENCE_RE.split(text):
        lower = sentence.lower()
        direction = _direction(sentence)
        multiplier = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:x|times?)\b", lower)
        if multiplier:
            value = float(multiplier.group(1))
            if direction == "down" and value > 0:
                value = 1.0 / value
            return "multiplier", value, sentence.strip()[:500]
        word_multiplier = re.search(r"\b(twice|double|triple)\b", lower)
        if word_multiplier:
            value = {"twice": 2.0, "double": 2.0, "triple": 3.0}[word_multiplier.group(1)]
            if direction == "down":
                value = 1.0 / value
            return "multiplier", value, sentence.strip()[:500]
        if direction == "unclear":
            continue
        percent = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\b", lower)
        if percent:
            signed = float(percent.group(1)) / 100.0
            if direction == "down":
                signed = -signed
            return "percentage", signed, sentence.strip()[:500]
        absolute = re.search(r"\b(?:by|of)\s+([-+]?\d+(?:\.\d+)?)\s+(?:units?|points?|items?)\b", lower)
        if absolute:
            signed = float(absolute.group(1))
            if direction == "down":
                signed = -abs(signed)
            return "absolute_additive", signed, sentence.strip()[:500]
    return None, None, ""


def _forecast_relation(
    task: ForecastTask,
    start_timestamp: str | None,
    end_timestamp: str | None,
    text: str,
    permanence: str,
) -> str:
    future_start = _parse_timestamp(task.future_timestamps[0])
    future_end = _parse_timestamp(task.future_timestamps[-1], end_of_day=True)
    history_end = _parse_timestamp(task.history_timestamps[-1], end_of_day=True)
    start = _parse_timestamp(start_timestamp) if start_timestamp else None
    end = _parse_timestamp(end_timestamp, end_of_day=True) if end_timestamp else None
    if permanence == "permanent" and start and start <= history_end:
        return "embedded_in_history"
    if permanence == "permanent" and start and end is None:
        end = future_end
    if end and end < future_start:
        return "ended_before_forecast"
    if start and start > future_end:
        return "after_forecast"
    if (start and start <= future_end) or (end and end >= future_start):
        return "overlaps_forecast"
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return "forecast_relevant_undated" if tokens & FUTURE_WORDS else "historical_or_uncertain"


class EvidenceToForecastAgent:
    """Translate accepted prose into conservative, auditable numerical effects."""

    EVENT_KEYWORDS = {
        "anomaly": {"anomaly", "bug", "error", "malfunction", "inflated", "incident"},
        "resolution": {"resolved", "fixed", "patch", "update", "restored", "stabilized"},
        "promotion": {"promotion", "discount", "campaign", "markdown", "pricing"},
        "external_driver": {"weather", "policy", "maintenance", "outage", "firmware", "event"},
        "forecast_regime": {"forecast", "baseline", "seasonality", "periodic", "cycle", "trajectory"},
    }

    @staticmethod
    def _event_type(text: str) -> str:
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        scored = [
            (len(tokens & keywords), event_type)
            for event_type, keywords in EvidenceToForecastAgent.EVENT_KEYWORDS.items()
        ]
        score, event_type = max(scored, default=(0, "general"))
        return event_type if score else "general"

    def translate(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        retrieved: list[RetrievedDocument],
        evidence: list[Evidence],
    ) -> list[EvidenceImpact]:
        claims_by_document: dict[str, list[str]] = {}
        for item in evidence:
            claims_by_document.setdefault(item.document_id, []).append(item.claim)

        impacts: list[EvidenceImpact] = []
        for item in retrieved:
            document = item.document.agent_view()
            claims = claims_by_document.get(document.document_id, [])
            text = "\n".join((*claims, document.text))
            start_timestamp, end_timestamp = _event_window(text)
            direction = _direction(" ".join(claims) or document.text)
            permanence = _permanence(text)
            if permanence == "resolved" and start_timestamp and end_timestamp is None:
                end_timestamp = start_timestamp
                start_timestamp = None
            elif permanence == "temporary" and start_timestamp and end_timestamp is None:
                end_timestamp = start_timestamp
            relation = _forecast_relation(
                task, start_timestamp, end_timestamp, text, permanence
            )
            event_type = self._event_type(text)
            adjustment_kind, adjustment_value, quantified_claim = _quantified_effect(text)

            if relation == "ended_before_forecast" or (
                permanence == "resolved" and relation != "overlaps_forecast"
            ):
                adjustment_kind = "return_to_baseline"
                adjustment_value = 0.0
                rationale = "The evidence says the effect ended or was resolved before the forecast; do not extrapolate it."
                confidence = 0.85
            elif relation == "after_forecast":
                adjustment_kind = "outside_horizon"
                adjustment_value = 0.0
                rationale = "The event starts after the requested forecast horizon."
                confidence = 0.9
            elif relation == "embedded_in_history":
                adjustment_kind = "already_in_baseline"
                adjustment_value = 0.0
                rationale = (
                    "The permanent shift began during the observed history, so the recent numerical baseline already contains it."
                )
                confidence = 0.8
            elif adjustment_kind is not None:
                rationale = f"Apply the explicit quantified effect: {quantified_claim}"
                confidence = 0.9
            elif relation in {"overlaps_forecast", "forecast_relevant_undated"} and direction != "unclear":
                adjustment_kind = "standardized_additive"
                adjustment_value = 0.25 if direction == "up" else -0.25
                rationale = (
                    "No magnitude is stated; apply a conservative quarter-residual directional adjustment."
                )
                confidence = 0.55
            else:
                adjustment_kind = "qualitative_only"
                adjustment_value = None
                rationale = "The evidence is retained for reasoning but does not justify a numerical adjustment."
                confidence = 0.5

            impacts.append(
                EvidenceImpact(
                    source_document_ids=(document.document_id,),
                    event_type=event_type,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    direction=direction,
                    permanence=permanence,
                    forecast_relation=relation,
                    adjustment_kind=adjustment_kind,
                    adjustment_value=adjustment_value,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

        # Merge exact duplicates so several documents describing the same event
        # do not compound the same numerical effect.
        merged: dict[tuple[object, ...], EvidenceImpact] = {}
        for impact in impacts:
            key = (
                impact.event_type,
                impact.start_timestamp,
                impact.end_timestamp,
                impact.direction,
                impact.permanence,
                impact.forecast_relation,
                impact.adjustment_kind,
                impact.adjustment_value,
            )
            previous = merged.get(key)
            if previous is None:
                merged[key] = impact
            else:
                merged[key] = replace(
                    previous,
                    source_document_ids=tuple(
                        dict.fromkeys((*previous.source_document_ids, *impact.source_document_ids))
                    ),
                    confidence=max(previous.confidence, impact.confidence),
                )
        return list(merged.values())
