"""Convert successful resolved trajectories into reusable Retrieval/Decision skills."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from evolving_loop.data import ContextTask
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.evaluation import ResolvedOutcome
from evolving_loop.harness import HarnessResult
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillLibrary,
)
from evolving_loop.retrieval_agent.credit import (
    RetrievalSkillTaskEvidence,
    promote_retrieval_skills,
)
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
    minimum_retrieval_gain: float = 0.0
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
        retrieval_smae_gain = (
            outcome.coding_oracle_smae - outcome.contextual_oracle_smae
        )
        retrieval_srmse_gain = (
            outcome.coding_oracle_srmse - outcome.contextual_oracle_srmse
        )
        for name in result.retrieval.used_skill_names:
            self.retrieval_library.record_use(
                name, retrieval_smae_gain, retrieval_srmse_gain
            )
        for name in result.decision.used_skill_names:
            self.decision_library.record_use(
                name,
                -outcome.decision_selection_smae_regret,
                -outcome.decision_selection_srmse_regret,
            )
        retrieval_eligible = (
            retrieval_smae_gain >= self.config.minimum_retrieval_gain
            and retrieval_srmse_gain >= self.config.minimum_retrieval_gain
            and (
                retrieval_smae_gain > self.config.minimum_retrieval_gain
                or retrieval_srmse_gain > self.config.minimum_retrieval_gain
            )
            and bool(result.retrieval.evidence)
        )
        decision_eligible = (
            len(result.candidates) >= 2
            and float(outcome.decision_selection_smae_regret or 0.0)
            <= self.config.maximum_decision_regret
            and float(outcome.decision_selection_srmse_regret or 0.0)
            <= self.config.maximum_decision_regret
            and outcome.decision_selection_mae_regret
            <= self.config.maximum_decision_regret
        )
        if not retrieval_eligible and not decision_eligible:
            return SkillLearningResult(None, None, False, False, ("no_module_passed_validation",))

        payload = {
            "module_eligibility": {
                "retrieval": retrieval_eligible,
                "decision": decision_eligible,
            },
            "resolved_metrics": {
                "retrieval_smae_gain": retrieval_smae_gain,
                "retrieval_srmse_gain": retrieval_srmse_gain,
                "decision_selection_smae_regret": (
                    outcome.decision_selection_smae_regret
                ),
                "decision_selection_srmse_regret": (
                    outcome.decision_selection_srmse_regret
                ),
                "final_smae": outcome.final_smae,
                "final_srmse": outcome.final_srmse,
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
                        skill_id=f"retrieval_{uuid.uuid4().hex}",
                        version=1,
                        parent_version=None,
                        stage="both",
                        status="candidate",
                        name=record["name"],
                        description=record["description"],
                        applicability=RetrievalApplicability(),
                        query_steps=(record["query_strategy"],),
                        required_chain_fields=(),
                        counterevidence_rule=record["verification_rule"],
                        failure_conditions=(),
                        validated_task_ids=(task.numeric.task_id,),
                        validated_entities=(task.numeric.entity_name,),
                        validation_smae_gain=retrieval_smae_gain,
                        validation_srmse_gain=retrieval_srmse_gain,
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
                        validation_smae=-outcome.decision_selection_smae_regret,
                        validation_srmse=-outcome.decision_selection_srmse_regret,
                    )
                )
        return SkillLearningResult(
            retrieval_name,
            decision_name,
            retrieval_eligible,
            decision_eligible,
            tuple(rejected),
        )

    def promote_retrieval_candidates(
        self,
        evidence: tuple[RetrievalSkillTaskEvidence, ...],
    ) -> tuple[str, ...]:
        """Apply evaluator-owned cross-Train gates after task aggregation."""
        return promote_retrieval_skills(self.retrieval_library, evidence)

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
