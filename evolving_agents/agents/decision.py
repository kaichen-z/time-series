"""The Decision Agent: judges candidates against evidence, then blends survivors with coded weights."""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, replace

from dr_cik.llm import JsonExtractionError, LLMClient, parse_json_object
from dr_cik.models import EvidenceItem, Forecast

from ..harness.trace import TraceEvent, emit, emit_llm_response
from ..models import AgentStep, Bundle, CodingCandidate, DecisionAuditEntry, DecisionOutput
from .common import extract_reasoning, render_fewshot_block, render_system_prompt

logger = logging.getLogger(__name__)

CONTRADICTION_SCHEMA = (
    'Respond with exactly one JSON object: {"contradicts": true | false, "reason": "<one sentence>"}'
)


@dataclass(frozen=True)
class DecisionAgentConfig:
    """Judging temperature and how surviving candidates are weighted."""

    temperature: float = 0.0
    softmax_beta: float = 1.0
    max_output_tokens: int = 400


def compute_weights(candidates: tuple[CodingCandidate, ...], beta: float = 1.0) -> dict[str, float]:
    """Softmax the candidates' hindcast errors into weights that sum to 1.

    Computed here in ordinary Python, never asked of the model: the same rule that keeps forecast
    numbers out of LLM free text applies to the numbers that combine them.
    """
    if not candidates:
        return {}
    scores = [candidate.hindcast_score for candidate in candidates]
    known = [score for score in scores if score is not None]
    fallback = statistics.fmean(known) if known else 0.0
    errors = [score if score is not None else fallback for score in scores]

    # Lower error is better, so negate before the softmax; shift for numerical stability.
    logits = [-beta * error for error in errors]
    ceiling = max(logits)
    weights = [math.exp(value - ceiling) for value in logits]
    total = sum(weights)
    if total <= 0:
        even = 1.0 / len(candidates)
        return {candidate.hypothesis.hypothesis_id: even for candidate in candidates}
    return {candidate.hypothesis.hypothesis_id: weight / total for candidate, weight in zip(candidates, weights)}


def blend(candidates: tuple[CodingCandidate, ...], weights: dict[str, float]) -> Forecast:
    """Combine candidate forecasts into one weighted mean trajectory."""
    usable = [candidate for candidate in candidates if candidate.forecast is not None]
    if not usable:
        raise ValueError("cannot blend: no candidate produced a forecast")
    horizon = len(usable[0].forecast.mean)
    mean = tuple(
        sum(weights.get(candidate.hypothesis.hypothesis_id, 0.0) * candidate.forecast.mean[step] for candidate in usable)
        for step in range(horizon)
    )
    samples = tuple(candidate.forecast.mean for candidate in usable)
    method = "decision:blend(" + ",".join(f"{candidate.hypothesis.hypothesis_id}={weights.get(candidate.hypothesis.hypothesis_id, 0.0):.2f}" for candidate in usable) + ")"
    return Forecast(mean=mean, samples=samples, method=method)


class DecisionAgent:
    """Discards candidates that the evidence contradicts, then weights and blends what survives."""

    def __init__(self, llm: LLMClient, bundle: Bundle, config: DecisionAgentConfig | None = None) -> None:
        self.llm = llm
        self.bundle = bundle
        base = config or DecisionAgentConfig()
        settings = bundle.hyperparameters
        self.config = replace(
            base,
            temperature=float(settings.get("temperature", base.temperature)),
            softmax_beta=float(settings.get("softmax_beta", base.softmax_beta)),
        )

    def _judge(self, candidate: CodingCandidate, item: EvidenceItem, task_id: str, generation: int | None) -> tuple[bool, str]:
        """Ask whether one piece of evidence contradicts one candidate's assumption."""
        prompt_parts = [
            f"Candidate {candidate.hypothesis.hypothesis_id}\nAssumption: {candidate.hypothesis.assumption_text}",
            f"Evidence: {item.claim}\nCited documents: {', '.join(item.source_doc_ids) or '(none)'}",
        ]
        fewshots = render_fewshot_block(self.bundle)
        if fewshots:
            prompt_parts.append(fewshots)
        prompt_parts.append(CONTRADICTION_SCHEMA)

        response = self.llm.complete(
            system=render_system_prompt(self.bundle),
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        reasoning, answer = extract_reasoning(response.text)
        emit_llm_response(task_id, "decision", answer, reasoning, model_id=getattr(self.llm, "model_id", "?"), generation=generation)
        try:
            parsed = parse_json_object(answer)
        except JsonExtractionError:
            # An unreadable judgement must not silently discard a sound candidate.
            logger.warning("decision[%s]: unreadable contradiction judgement, keeping the candidate", task_id)
            return False, "judgement unreadable; candidate kept"
        return bool(parsed.get("contradicts")), str(parsed.get("reason", ""))[:300]

    def decide(
        self,
        candidates: tuple[CodingCandidate, ...],
        evidence: tuple[EvidenceItem, ...],
        task_id: str = "-",
        generation: int | None = None,
        allow_revision: bool = True,
    ) -> DecisionOutput:
        """Judge every candidate against every claim, then weight and blend the survivors."""
        usable = tuple(candidate for candidate in candidates if candidate.forecast is not None)
        if not usable:
            raise ValueError("decide() needs at least one candidate with a forecast")

        emit(TraceEvent(task_id=task_id, agent="decision", event_type="agent_start", generation=generation, detail={"candidates": len(usable), "evidence": len(evidence)}))
        audit: list[DecisionAuditEntry] = []
        steps: list[AgentStep] = []
        survivors: list[CodingCandidate] = []
        calls = 0

        for candidate in usable:
            contradicting: list[str] = []
            reasons: list[str] = []
            for index, item in enumerate(evidence):
                contradicts, reason = self._judge(candidate, item, task_id, generation)
                calls += 1
                if contradicts:
                    contradicting.extend(item.source_doc_ids or (f"evidence_{index}",))
                    reasons.append(reason)
            kept = not contradicting
            if kept:
                survivors.append(candidate)
            audit.append(
                DecisionAuditEntry(
                    candidate_id=candidate.hypothesis.hypothesis_id,
                    kept=kept,
                    reason="; ".join(reasons) if reasons else "no evidence contradicted this assumption",
                    contradicting_evidence_ids=tuple(contradicting),
                )
            )

        revision_request = None
        if not survivors:
            # Everything was contradicted: still produce a forecast, but ask for a candidate that fits.
            if allow_revision and evidence:
                revision_request = (
                    "Every candidate was contradicted. Produce a hypothesis that accounts for: "
                    + "; ".join(item.claim for item in evidence[:3])
                )
            survivors = list(usable)
            steps.append(AgentStep(step_index=0, kind="all_contradicted", payload={"revision_requested": revision_request is not None}))

        chosen = tuple(survivors)
        weights = compute_weights(chosen, beta=self.config.softmax_beta)
        forecast = blend(chosen, weights)
        logger.info("decision[%s]: %d/%d candidate(s) survived, %d judgement call(s)", task_id, len(chosen), len(usable), calls)
        emit(TraceEvent(task_id=task_id, agent="decision", event_type="agent_end", generation=generation, detail={"survivors": len(chosen), "revision": revision_request is not None}))
        return DecisionOutput(
            final_forecast=forecast,
            weights=weights,
            audit=tuple(audit),
            revision_request=revision_request,
            steps=tuple(steps),
            llm_call_count=calls,
        )
