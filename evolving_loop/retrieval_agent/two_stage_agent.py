"""Assumption-blind Round 1 and sanitized gap-directed Round 2 retrieval."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from common.llm import LLMClient, TransientLLMError, parse_json_object
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.schemas import (
    RetrievalAssumption,
    RetrievalGap,
    RetrievalRoundResult,
    build_round1_payload,
    build_round2_payload,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkill,
    RetrievalSkillLibrary,
    _trusted_train_shadow_skills,
)
from evolving_loop.retrieval_agent.verifier import verify_round_result


_ROUND1_STRATEGY_PLANS = {
    "timeline_first": {
        "ordered_objectives": (
            "Anchor evidence to the forecast window before composing causal claims.",
            "Order event evidence chronologically and flag ended events.",
            "Search explicitly for cancellations, postponements, and reversals.",
        ),
        "selection_features": (
            "forecast_window_overlap",
            "explicit_event_dates",
            "entity_target_phrase",
        ),
    },
    "entity_first": {
        "ordered_objectives": (
            "Resolve the exact entity and target phrase before considering an event.",
            "Reject evidence about neighboring entities or similarly named targets.",
            "Then verify mechanism, magnitude, and forecast-window coverage.",
        ),
        "selection_features": (
            "entity_target_phrase",
            "canonical_name_boundaries",
            "forecast_window_overlap",
        ),
    },
    "contrastive": {
        "ordered_objectives": (
            "Build separate support and challenge hypotheses for each material event.",
            "Seek matched counterevidence before declaring the ledger sufficient.",
            "Retain unresolved contradictions rather than averaging them away.",
        ),
        "selection_features": (
            "support_challenge_pairing",
            "cancellation_or_reversal",
            "entity_target_phrase",
        ),
    },
}

_ROUND2_STRATEGY_PLANS = {
    "counterevidence_first": {
        "ordered_objectives": (
            "Search first for evidence that invalidates or limits each named assumption.",
            "Prioritize cancellation, postponement, reversal, containment, or recovery.",
            "Report unresolved assumptions when no exact counterevidence exists.",
        ),
        "selection_features": (
            "assumption_failure_condition",
            "counterevidence",
            "named_gap_priority",
        ),
    },
    "gap_first": {
        "ordered_objectives": (
            "Process named gaps in host-provided priority order.",
            "Fill only the missing link stated for each gap.",
            "Keep evidence attached to its addressed assumption ID.",
        ),
        "selection_features": (
            "named_gap_priority",
            "missing_information",
            "assumption_id",
        ),
    },
    "causal_chain_first": {
        "ordered_objectives": (
            "Complete entity, target, mechanism, window, and magnitude links in that order.",
            "Prefer exact evidence that closes an incomplete verified Round 1 chain.",
            "Preserve contradictions and do not overwrite Round 1 evidence.",
        ),
        "selection_features": (
            "incomplete_chain_fields",
            "verified_round1",
            "causal_link_completeness",
        ),
    },
}


def _query_plan(stage: str, strategy: str) -> dict[str, list[str]]:
    plans = _ROUND1_STRATEGY_PLANS if stage == "round1" else _ROUND2_STRATEGY_PLANS
    plan = plans[strategy]
    return {
        "ordered_objectives": list(plan["ordered_objectives"]),
        "selection_features": list(plan["selection_features"]),
    }


def _canonical_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(
        tokens[index : index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


def _content_rank(content: str, target: Mapping[str, object]) -> tuple[object, ...]:
    """Rank content using only the sanitized Retrieval target/query projection."""
    tokens = _canonical_tokens(content)
    token_set = frozenset(tokens)
    entity = _canonical_tokens(str(target.get("entity_name", "")))
    target_name = _canonical_tokens(str(target.get("target_name", "")))
    description = _canonical_tokens(str(target.get("description", "")))
    frequency = _canonical_tokens(str(target.get("frequency", "")))
    raw_window = target.get("forecast_window", ())
    window = (
        tuple(str(item) for item in raw_window)
        if isinstance(raw_window, (list, tuple))
        else ()
    )
    query_tokens = frozenset((*entity, *target_name, *description, *frequency))
    entity_match = _contains_phrase(tokens, entity)
    target_match = _contains_phrase(tokens, target_name)
    window_hits = sum(
        1
        for timestamp in window
        if unicodedata.normalize("NFKC", timestamp).casefold()
        in unicodedata.normalize("NFKC", content).casefold()
    )
    description_overlap = len(token_set.intersection(description))
    query_overlap = len(token_set.intersection(query_tokens))
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return (
        -int(entity_match and target_match),
        -int(entity_match),
        -int(target_match),
        -window_hits,
        -description_overlap,
        -query_overlap,
        content_sha256,
        content,
    )


def _select_documents(
    payload: Mapping[str, object], budget: int
) -> list[dict[str, str]]:
    """Select whole exact-content groups without using IDs or input positions."""
    raw_documents = payload.get("documents", ())
    target = payload.get("target", {})
    if not isinstance(raw_documents, (list, tuple)) or not isinstance(target, Mapping):
        return []
    grouped: dict[str, list[dict[str, str]]] = {}
    for raw in raw_documents:
        if not isinstance(raw, Mapping):
            continue
        document_id = raw.get("document_id")
        content = raw.get("content")
        if not isinstance(document_id, str) or not isinstance(content, str):
            continue
        grouped.setdefault(content, []).append(
            {"document_id": document_id, "content": content}
        )

    selected: list[dict[str, str]] = []
    remaining = max(int(budget), 0)
    for content, group in sorted(
        grouped.items(), key=lambda item: _content_rank(item[0], target)
    ):
        if len(group) > remaining:
            continue
        # IDs only canonicalize presentation inside an already-selected exact-content
        # group. They never affect group relevance or membership.
        selected.extend(sorted(group, key=lambda item: item["document_id"]))
        remaining -= len(group)
        if remaining == 0:
            break
    return selected


class TwoStageRetrievalAgent:
    """Execute two fixed Retrieval stages behind host-owned payload boundaries."""

    def __init__(
        self,
        llm: LLMClient,
        genome: RetrievalGenome,
        skills: RetrievalSkillLibrary,
    ) -> None:
        self.llm = llm
        self.genome = genome
        self.skills = skills

    def run_round1(self, task: ContextTask) -> RetrievalRoundResult:
        skills = self._skills("round1")
        payload = build_round1_payload(task, skills=self._skill_payloads(skills))
        payload["query_plan"] = _query_plan("round1", self.genome.round1_strategy)
        payload["documents"] = _select_documents(
            payload, self.genome.max_selected_documents
        )
        raw = self._complete(self.genome.round1_prompt, payload, stage="round1")
        if isinstance(raw, RetrievalRoundResult):
            return raw
        bounded = self._bounded_response(raw)
        verified = verify_round_result(
            task,
            bounded,
            stage="round1",
            allowed_skill_ids=tuple(item.skill_id for item in skills),
            allowed_assumption_ids=(),
            allowed_document_ids=tuple(
                item["document_id"] for item in payload["documents"]
            ),
        )
        if verified.gaps:
            verified = replace(
                verified,
                gaps=(),
                rejected=tuple(dict.fromkeys((*verified.rejected, "round1_gaps_forbidden"))),
            )
        return verified

    def run_round2(
        self,
        task: ContextTask,
        round1: RetrievalRoundResult,
        gaps: tuple[RetrievalGap, ...],
        assumptions: tuple[RetrievalAssumption, ...],
    ) -> RetrievalRoundResult:
        skills = self._skills("round2", assumptions=assumptions, gaps=gaps)
        payload = build_round2_payload(
            task,
            round1,
            gaps,
            assumptions,
            self._skill_payloads(skills),
        )
        payload["query_plan"] = _query_plan("round2", self.genome.round2_strategy)
        payload["documents"] = _select_documents(
            payload, self.genome.max_selected_documents
        )
        raw = self._complete(self.genome.round2_prompt, payload, stage="round2")
        if isinstance(raw, RetrievalRoundResult):
            return raw
        return verify_round_result(
            task,
            self._bounded_response(raw),
            stage="round2",
            allowed_skill_ids=tuple(item.skill_id for item in skills),
            allowed_assumption_ids=tuple(item.assumption_id for item in assumptions),
            allowed_document_ids=tuple(
                item["document_id"] for item in payload["documents"]
            ),
            prior_round1=round1,
        )

    def _complete(
        self,
        prompt: str,
        payload: Mapping[str, object],
        *,
        stage: str,
    ) -> Mapping[str, object] | RetrievalRoundResult:
        try:
            response = self.llm.complete(
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                ],
                temperature=0.0,
            )
            return parse_json_object(response.text)
        except TransientLLMError:
            raise
        except Exception as error:
            return RetrievalRoundResult(
                (),
                (),
                (f"invalid_{stage}_response",),
                False,
                rejected=(
                    f"invalid_{stage}_response",
                    f"{stage}_failure:{type(error).__name__}",
                ),
            )

    def _bounded_response(self, raw: Mapping[str, object]) -> dict[str, object]:
        payload = dict(raw)
        remaining = self.genome.max_evidence_chains
        for field in ("evidence_chains", "counterevidence"):
            collection = payload.get(field)
            if not isinstance(collection, (list, tuple)):
                continue
            bounded = []
            for item in collection[:remaining]:
                if isinstance(item, Mapping):
                    item = dict(item)
                    citations = item.get("citations")
                    if isinstance(citations, (list, tuple)):
                        item["citations"] = list(
                            citations[: self.genome.max_citations_per_chain]
                        )
                bounded.append(item)
            payload[field] = bounded
            remaining -= len(bounded)
        return payload

    def _skills(
        self,
        stage: str,
        *,
        assumptions: Iterable[RetrievalAssumption] = (),
        gaps: Iterable[RetrievalGap] = (),
    ) -> tuple[RetrievalSkill, ...]:
        active_ids = frozenset(self.genome.active_skill_ids)
        if not active_ids:
            return ()
        assumption_kinds = tuple(value.kind for value in assumptions)
        gap_types = tuple(value.gap_type for value in gaps)
        projected = {
            item.skill_id: item
            for item in (
                *self.skills.for_stage(
                    stage,
                    assumption_kinds=assumption_kinds,
                    gap_types=gap_types,
                ),
                *_trusted_train_shadow_skills(
                    self.skills,
                    stage,
                    assumption_kinds=assumption_kinds,
                    gap_types=gap_types,
                ),
            )
            if item.skill_id in active_ids
        }
        return tuple(projected[skill_id] for skill_id in sorted(projected))

    @staticmethod
    def _skill_payloads(skills: Sequence[RetrievalSkill]) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "skill_id": item.skill_id,
                "name": item.name,
                "description": item.description,
                "query_strategy": " | ".join(item.query_steps) or "No additional query steps.",
                "verification_rule": item.counterevidence_rule,
            }
            for item in skills
        )


__all__ = ["TwoStageRetrievalAgent"]
