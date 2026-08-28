"""Assumption-blind Round 1 and sanitized gap-directed Round 2 retrieval."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import json

from common.llm import LLMClient, parse_json_object
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
        payload["documents"] = payload["documents"][: self.genome.max_selected_documents]
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
        payload["documents"] = payload["documents"][: self.genome.max_selected_documents]
        raw = self._complete(self.genome.round2_prompt, payload, stage="round2")
        if isinstance(raw, RetrievalRoundResult):
            return raw
        return verify_round_result(
            task,
            self._bounded_response(raw),
            stage="round2",
            allowed_skill_ids=tuple(item.skill_id for item in skills),
            allowed_assumption_ids=tuple(item.assumption_id for item in assumptions),
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
