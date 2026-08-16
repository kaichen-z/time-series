"""Convert successful resolved trajectories into reusable Retrieval/Decision skills."""
from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import asdict, dataclass

from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.evaluation import ResolvedOutcome
from evolving_loop.harness import HarnessResult
from evolving_loop.retrieval_agent.skill_library import RetrievalSkill, RetrievalSkillLibrary
from common.llm import JsonExtractionError, LLMClient, parse_json_object

SKILL_LEARNING_PROMPT = """You are the post-outcome Skill Curator for a forecasting harness.
Read a resolved public training trajectory. Generalize only successful behavior into reusable
skills. A skill must describe a cross-task strategy, not memorize the task. Never include task IDs,
document IDs, entity names, exact timestamps, realized future values, or task-specific numeric
answers. Do not weaken exact-quote, provenance, temporal-window, or evidence safety checks.

Return exactly one JSON object. Use null when the corresponding module was not successful enough:
{"retrieval_skill": {"name": "short_snake_case", "description": "...",
"applicability": "when this strategy applies", "query_strategy": "general search procedure",
"verification_rule": "general evidence check"},
"decision_skill": {"name": "short_snake_case", "description": "...",
"applicability": "when this rule applies", "decision_rule": "general selection rule",
"failure_condition": "when the rule should not be used"}}
"""


@dataclass(frozen=True)
class SkillLearningConfig:
    minimum_retrieval_score: float = 0.6
    maximum_decision_regret: float = 1e-8


@dataclass(frozen=True)
class SkillLearningResult:
    retrieval_skill_name: str | None
    decision_skill_name: str | None
    retrieval_eligible: bool
    decision_eligible: bool
    rejection_reasons: tuple[str, ...] = ()


class OutcomeSkillLearner:
    """Write skills only after labels resolve and deterministic gates pass."""

    def __init__(
        self,
        llm: LLMClient,
        retrieval_library: RetrievalSkillLibrary,
        decision_library: DecisionSkillLibrary,
        config: SkillLearningConfig | None = None,
    ) -> None:
        self.llm = llm
        self.retrieval_library = retrieval_library
        self.decision_library = decision_library
        self.config = config or SkillLearningConfig()

    def learn(
        self,
        task: ContextTask,
        result: HarnessResult,
        outcome: ResolvedOutcome,
    ) -> SkillLearningResult:
        if not task.labels_public:
            raise ValueError("skill learning is forbidden for hidden/unreleased labels")
        retrieval_score = statistics.fmean(
            (outcome.retrieval_precision, outcome.supporting_recall, outcome.distractor_avoidance)
        )
        for name in result.retrieval.used_skill_names:
            self.retrieval_library.record_use(name, retrieval_score)
        decision_score = max(0.0, 1.0 - outcome.decision_selection_regret / 200.0)
        for name in result.decision.used_skill_names:
            self.decision_library.record_use(name, decision_score)
        retrieval_eligible = (
            retrieval_score >= self.config.minimum_retrieval_score
            and bool(result.retrieval.evidence)
        )
        decision_eligible = (
            len(result.candidates) >= 2
            and outcome.decision_selection_regret <= self.config.maximum_decision_regret
        )
        if not retrieval_eligible and not decision_eligible:
            return SkillLearningResult(None, None, False, False, ("no_module_passed_validation",))

        payload = {
            "module_eligibility": {
                "retrieval": retrieval_eligible,
                "decision": decision_eligible,
            },
            "resolved_metrics": {
                "retrieval_quality": retrieval_score,
                "decision_selection_regret": outcome.decision_selection_regret,
                "final_smape": outcome.final_smape,
            },
            "retrieval_trace": {
                "query": result.retrieval.query,
                "verified_claims": [item.claim for item in result.retrieval.evidence],
                "mechanism_layers": [item.mechanism_layer for item in result.retrieval.impacts],
                "rejections": list(result.retrieval.rejected),
            },
            "decision_trace": {
                "candidate_assumptions": [item.assumption for item in result.candidates],
                "candidate_tags": [list(item.tags) for item in result.candidates],
                "selected_tags": list(result.decision.selected.tags),
                "host_default_was_overridden": result.decision.llm_override_accepted,
                "safety_rejection": result.decision.rejection_reason,
            },
        }
        response = self.llm.complete(
            system=SKILL_LEARNING_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
        )
        try:
            proposal = parse_json_object(response.text)
        except JsonExtractionError as error:
            return SkillLearningResult(
                None, None, retrieval_eligible, decision_eligible, (f"invalid_skill_json:{error}",)
            )

        rejected = []
        retrieval_name = None
        decision_name = None
        if retrieval_eligible and isinstance(proposal.get("retrieval_skill"), dict):
            record = proposal["retrieval_skill"]
            reason = self._invalid_record(task, record, ("name", "description", "applicability", "query_strategy", "verification_rule"))
            if reason:
                rejected.append(f"retrieval:{reason}")
            else:
                retrieval_name = record["name"]
                self.retrieval_library.add(
                    RetrievalSkill(
                        skill_id=str(uuid.uuid4()),
                        name=record["name"],
                        description=record["description"],
                        applicability=record["applicability"],
                        query_strategy=record["query_strategy"],
                        verification_rule=record["verification_rule"],
                        created_from_task=task.numeric.task_id,
                        validation_score=retrieval_score,
                    )
                )
        if decision_eligible and isinstance(proposal.get("decision_skill"), dict):
            record = proposal["decision_skill"]
            reason = self._invalid_record(task, record, ("name", "description", "applicability", "decision_rule", "failure_condition"))
            if reason:
                rejected.append(f"decision:{reason}")
            else:
                decision_name = record["name"]
                self.decision_library.add(
                    DecisionSkill(
                        skill_id=str(uuid.uuid4()),
                        name=record["name"],
                        description=record["description"],
                        applicability=record["applicability"],
                        decision_rule=record["decision_rule"],
                        failure_condition=record["failure_condition"],
                        created_from_task=task.numeric.task_id,
                        validation_score=decision_score,
                    )
                )
        return SkillLearningResult(
            retrieval_name,
            decision_name,
            retrieval_eligible,
            decision_eligible,
            tuple(rejected),
        )

    @staticmethod
    def _invalid_record(task: ContextTask, record: dict, fields: tuple[str, ...]) -> str | None:
        if not all(isinstance(record.get(field), str) and record[field].strip() for field in fields):
            return "missing_required_field"
        combined = " ".join(record[field] for field in fields).lower()
        forbidden = [task.numeric.task_id, task.numeric.entity_name]
        forbidden.extend(document.document_id for document in task.documents)
        forbidden.extend(task.history_timestamps)
        forbidden.extend(task.future_timestamps)
        if any(value and value.lower() in combined for value in forbidden):
            return "task_specific_identifier"
        return None
