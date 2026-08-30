"""Package-native Numerical -> two-stage Retrieval -> Decision orchestration.

This boundary deliberately consumes a completed :class:`NumericalForecastPackage`.
It never re-runs the Numerical or Morphology agents, never creates a new forecast, and
never gives Retrieval access to candidate identities, scores, or Numerical internals.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from common.data import Task as ContextNumericTask
from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.llm import TransientLLMError
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import (
    DecisionAgent,
    DecisionCandidate,
    DecisionResult,
)
from evolving_loop.retrieval_agent.agent import RetrievalResult
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.schemas import (
    FinalRetrievalCard,
    RetrievalAssumption,
    RetrievalContractError,
    RetrievalGap,
    RetrievalRoundResult,
    build_round1_payload,
    build_round2_payload,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkillLibrary,
    _trusted_train_shadow_skill_ids,
)
from evolving_loop.retrieval_agent.two_stage_agent import (
    TwoStageRetrievalAgent,
    _select_documents,
)
from evolving_loop.retrieval_agent.verifier import (
    merge_verified_rounds,
    verify_round_result,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.morphology import AssumptionGrounding
from numerical_agent.evolution.numerical_handoff import (
    safe_retrieval_projection,
    task_input_fingerprint,
)
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage,
    valid_forecast,
)
from numerical_agent.evolution.screening import profile_task


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_NUMERICAL_FINGERPRINTS = frozenset(
    {
        "metric_policy_fingerprint",
        "task_input",
        "task_profile",
        "screening_policy",
        "active_dictionary",
        "combined_policies",
        "decision_policy",
        "hindcast_config",
        "morphology_card",
    }
)
_BRIDGE_CONTRACT = {
    "schema_version": 1,
    "topology": "round1_decide_optional_round2_decide",
    "round1_assumption_blind": True,
    "round2_assumption_fields": [
        "assumption_id",
        "kind",
        "claim",
        "failure_condition",
    ],
    "selection_boundary": "materialized_ranked_alternatives_only",
    "safe_default": "single_package_selection_else_protected_baseline",
}
_RETRIEVAL_VERIFIER_CONTRACT = {
    "schema_version": 1,
    "implementation": "host_verified_evidence_chain_merge",
    "round1_identity_precedence": True,
    "exact_quote_required": True,
    "verified_citations_only": True,
}
_DECISION_HOST_CONTRACT = {
    "schema_version": 1,
    "implementation": "executed_candidate_verified_citation_gate",
    "unmaterialized_candidate_rejected": True,
    "override_requires_verified_evidence": True,
    "forecast_values_immutable": True,
}
_MAX_ASSUMPTIONS = 7
_MAX_TEXT_LENGTHS = {
    "assumption_id": 128,
    "kind": 64,
    "claim": 512,
    "failure_condition": 512,
}


@dataclass(frozen=True)
class NumericalTwoStageResult:
    """Immutable output of the package-native two-stage inference path."""

    numerical: NumericalForecastPackage
    retrieval_card: FinalRetrievalCard
    retrieval: RetrievalResult
    provisional_decision: DecisionResult
    final_decision: DecisionResult
    forecast: tuple[float, ...]
    fingerprints: Mapping[str, str]
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.numerical, NumericalForecastPackage):
            raise ValueError("result requires a NumericalForecastPackage")
        if not isinstance(self.retrieval_card, FinalRetrievalCard):
            raise ValueError("result requires a final Retrieval card")
        if not isinstance(self.retrieval, RetrievalResult):
            raise ValueError("result requires a verified Retrieval result")
        if not isinstance(self.provisional_decision, DecisionResult) or not isinstance(
            self.final_decision, DecisionResult
        ):
            raise ValueError("result requires provisional and final Decision artifacts")
        forecast = tuple(float(value) for value in self.forecast)
        if not valid_forecast(forecast, self.numerical.task_profile.horizon):
            raise ValueError("result forecast must be finite and match the Numerical horizon")
        if forecast != self.final_decision.selected.forecast:
            raise ValueError("result forecast must equal the final materialized Decision")
        if self.retrieval != self.retrieval_card.to_legacy_result():
            raise ValueError("result Retrieval projection must match the final card")
        fingerprints = dict(self.fingerprints)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not _SHA256.fullmatch(value)
            for key, value in fingerprints.items()
        ):
            raise ValueError("result fingerprints must be canonical SHA-256 strings")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason
        ):
            raise ValueError("fallback_reason must be a non-empty string or None")
        object.__setattr__(self, "forecast", forecast)
        object.__setattr__(
            self,
            "fingerprints",
            MappingProxyType(dict(sorted(fingerprints.items()))),
        )


def run_numerical_two_stage(
    task: ContextTask,
    numerical: NumericalForecastPackage,
    retrieval: TwoStageRetrievalAgent,
    decision: DecisionAgent,
) -> NumericalTwoStageResult:
    """Run fixed two-stage Retrieval and Decision over one frozen Numerical package."""
    _validate_inputs(task, numerical, retrieval, decision)
    retrieval_task = _sanitized_context_task(task)
    execution_retrieval = _frozen_retrieval_executor(retrieval)
    candidates = _decision_candidates(numerical)
    host_default = _safe_default(numerical, candidates)

    assumptions, handoff_failure = _validated_handoff(numerical)
    execution_fingerprints = _execution_fingerprints(
        retrieval_task,
        numerical,
        execution_retrieval,
        decision,
        candidates,
    )
    fallback_reason = handoff_failure

    round1 = _run_round1(execution_retrieval, retrieval_task)
    if _fatal_round_failure(round1, "round1"):
        round1 = replace(round1, chains=(), counterevidence=())
        fallback_reason = fallback_reason or "invalid_round1_response"
    round1_card = merge_verified_rounds(round1, None)
    if handoff_failure is not None:
        round1_card = _record_rejection(round1_card, handoff_failure)
    provisional_retrieval = round1_card.to_legacy_result()
    provisional, provisional_failure = _run_decision(
        decision,
        candidates,
        provisional_retrieval,
        host_default=host_default,
        assumptions=assumptions,
        round_index=0,
    )
    if provisional_failure is not None:
        fallback_reason = fallback_reason or provisional_failure

    round2: RetrievalRoundResult | None = None
    sent_gaps = ()
    if (
        fallback_reason is None
        and assumptions
        and _should_run_round2(
            execution_retrieval.genome.second_round_trigger,
            round1,
            provisional,
        )
    ):
        sent_gaps = provisional.gaps
        # TwoStageRetrievalAgent propagates TransientLLMError and converts all other
        # completion/contract failures into a typed, auditable stage result.
        round2 = _run_round2(
            execution_retrieval,
            retrieval_task,
            round1,
            sent_gaps,
            assumptions,
        )
        if _fatal_round_failure(round2, "round2"):
            round2 = replace(round2, chains=(), counterevidence=())
            fallback_reason = "invalid_round2_response"
        else:
            had_verified_evidence = bool(round2.chains or round2.counterevidence)
            binding_kind = (
                "gap"
                if sent_gaps
                or execution_retrieval.genome.second_round_trigger == "on_named_gap"
                else "assumption"
            )
            round2 = _bind_round2_to_scope(
                round2,
                gaps=sent_gaps,
                assumptions=assumptions,
                binding_kind=binding_kind,
            )
            if not round2.chains and not round2.counterevidence:
                fallback_reason = (
                    f"round2_no_{binding_kind}_bound_evidence"
                    if had_verified_evidence
                    else "round2_no_verified_evidence"
                )

    card = merge_verified_rounds(round1, round2, gaps=sent_gaps)
    if fallback_reason is not None:
        card = _record_rejection(card, fallback_reason)
    final_retrieval = card.to_legacy_result()
    final, final_failure = _run_decision(
        decision,
        candidates,
        final_retrieval,
        host_default=host_default,
        assumptions=assumptions,
        round_index=1,
        prior=(provisional,),
    )
    if final_failure is not None:
        fallback_reason = fallback_reason or final_failure
    if fallback_reason is not None:
        final = _fallback_decision(host_default, fallback_reason)

    materialized = {item.candidate_id: item for item in candidates}
    selected = materialized.get(final.selected.candidate_id)
    if selected is None or selected.forecast != final.selected.forecast:
        fallback_reason = fallback_reason or "unmaterialized_final_decision"
        final = _fallback_decision(host_default, fallback_reason)

    return NumericalTwoStageResult(
        numerical=numerical,
        retrieval_card=card,
        retrieval=final_retrieval,
        provisional_decision=provisional,
        final_decision=final,
        forecast=final.selected.forecast,
        fingerprints=_completed_fingerprints(
            execution_fingerprints,
            card,
            provisional,
            final,
        ),
        fallback_reason=fallback_reason,
    )


def _sanitized_context_task(task: ContextTask) -> ContextTask:
    """Create the only ContextTask object that a replaceable Retrieval agent sees."""
    numeric = task.numeric
    return ContextTask(
        numeric=ContextNumericTask(
            task_id=str(numeric.task_id),
            history_values=tuple(float(value) for value in numeric.history_values),
            future_values=(),
            prediction_length=int(numeric.prediction_length),
            frequency=str(numeric.frequency),
            seasonal_period=(
                None
                if numeric.seasonal_period is None
                else str(numeric.seasonal_period)
            ),
            entity_name=str(numeric.entity_name),
        ),
        target_name=str(task.target_name),
        target_description=str(task.target_description),
        history_timestamps=tuple(str(value) for value in task.history_timestamps),
        future_timestamps=tuple(str(value) for value in task.future_timestamps),
        documents=tuple(
            Document(
                document_id=str(item.document_id),
                content=str(item.content),
                role=None,
                subtype=None,
            )
            for item in task.documents
        ),
        gt_evidence=(),
        labels_public=False,
    )


def _frozen_retrieval_executor(
    retrieval: TwoStageRetrievalAgent,
) -> TwoStageRetrievalAgent:
    """Freeze the complete Retrieval execution scope before replaceable calls."""
    caller_genome_fingerprint = retrieval.genome.fingerprint()
    if not isinstance(caller_genome_fingerprint, str) or not _SHA256.fullmatch(
        caller_genome_fingerprint
    ):
        raise ValueError("Retrieval Genome fingerprint is not canonical")
    genome = RetrievalGenome.from_payload(
        RetrievalGenome.to_payload(retrieval.genome)
    )
    if genome.fingerprint() != caller_genome_fingerprint:
        raise ValueError("Retrieval Genome fingerprint does not match its payload")
    if type(retrieval.skills) is not RetrievalSkillLibrary:
        raise TypeError("Retrieval Skill library must be canonical")
    skill_snapshot = RetrievalSkillLibrary.all(retrieval.skills)
    skills = RetrievalSkillLibrary.frozen_execution_snapshot(
        retrieval.skills,
    )
    if type(skills) is not RetrievalSkillLibrary or (
        RetrievalSkillLibrary.all(skills) != skill_snapshot
    ):
        raise ValueError("Retrieval Skill snapshot drifted during preflight")
    return TwoStageRetrievalAgent(retrieval.llm, genome, skills)


def _validate_inputs(
    task: ContextTask,
    numerical: NumericalForecastPackage,
    retrieval: TwoStageRetrievalAgent,
    decision: DecisionAgent,
) -> None:
    if not isinstance(task, ContextTask):
        raise TypeError("task must be a ContextTask")
    if not isinstance(numerical, NumericalForecastPackage):
        raise TypeError("numerical must be a NumericalForecastPackage")
    if not isinstance(retrieval, TwoStageRetrievalAgent):
        raise TypeError("retrieval must be a TwoStageRetrievalAgent")
    if not isinstance(decision, DecisionAgent):
        raise TypeError("decision must be a DecisionAgent")
    if task.numeric.prediction_length != numerical.task_profile.horizon:
        raise ValueError("Numerical package horizon does not match the ContextTask horizon")
    expected_profile = profile_task(
        Task(
            task.numeric.task_id,
            tuple(task.numeric.history_values),
            task.numeric.prediction_length,
            task.numeric.frequency,
            (),
        )
    )
    if expected_profile != numerical.task_profile:
        raise ValueError("Numerical package TaskProfile does not match the ContextTask history")

    fingerprints = dict(numerical.component_fingerprints)
    missing = _REQUIRED_NUMERICAL_FINGERPRINTS - set(fingerprints)
    if missing:
        raise ValueError(
            "Numerical package is missing canonical component fingerprints: "
            + ", ".join(sorted(missing))
        )
    if fingerprints["metric_policy_fingerprint"] != METRIC_POLICY_FINGERPRINT:
        raise ValueError("Numerical package metric policy fingerprint mismatch")
    for name in _REQUIRED_NUMERICAL_FINGERPRINTS - {"metric_policy_fingerprint"}:
        if not _SHA256.fullmatch(fingerprints[name]):
            raise ValueError(f"Numerical package {name} fingerprint is not canonical")
    expected_input_fingerprint = task_input_fingerprint(
        task_id=task.numeric.task_id,
        history=task.numeric.history_values,
        frequency=task.numeric.frequency,
        horizon=task.numeric.prediction_length,
    )
    if fingerprints["task_input"] != expected_input_fingerprint:
        raise ValueError("Numerical package task input fingerprint mismatch")
    expected_profile_fingerprint = _fingerprint(expected_profile.to_public_payload())
    if fingerprints["task_profile"] != expected_profile_fingerprint:
        raise ValueError("Numerical package task profile fingerprint mismatch")
    expected_morphology_fingerprint = (
        numerical.morphology_card.fingerprint
        if numerical.morphology_card is not None
        else _fingerprint({"enabled": False})
    )
    if fingerprints["morphology_card"] != expected_morphology_fingerprint:
        raise ValueError("Numerical package Morphology card fingerprint mismatch")
    accepted = tuple(numerical.accepted_assumptions)
    if accepted and numerical.morphology_card is None:
        raise ValueError("accepted assumptions require a frozen Morphology card")
    card_assumptions = {
        item.assumption_id: item
        for item in (
            numerical.morphology_card.assumptions
            if numerical.morphology_card is not None
            else ()
        )
    }
    if any(
        type(item) is not AssumptionGrounding
        or type(card_assumptions.get(item.assumption_id)) is not AssumptionGrounding
        or card_assumptions[item.assumption_id] != item
        for item in accepted
    ):
        raise ValueError("accepted assumption is not bound to the frozen Morphology card")


def _decision_candidates(
    numerical: NumericalForecastPackage,
) -> tuple[DecisionCandidate, ...]:
    accepted = tuple(numerical.accepted_assumptions)
    candidates: list[DecisionCandidate] = []
    for alternative in numerical.ranked_alternatives:
        diagnostic = alternative.diagnostics
        if not valid_forecast(alternative.forecast, numerical.task_profile.horizon):
            continue
        if not math.isfinite(diagnostic.median_smae) or not math.isfinite(
            diagnostic.median_srmse
        ):
            continue
        grounding = next(
            (
                item
                for item in accepted
                if alternative.name in item.candidate_names
            ),
            None,
        )
        if grounding is None:
            assumption = (
                f"This materialized {alternative.family} candidate remains valid under "
                "its history-only validation."
            )
            failure_condition = (
                "The future regime differs from the history-only validation regime."
            )
        else:
            assumption = grounding.claim
            failure_condition = grounding.failure_condition
        candidates.append(
            DecisionCandidate(
                candidate_id=alternative.name,
                forecast=alternative.forecast,
                assumption=assumption,
                failure_condition=failure_condition,
                hindcast_smae=diagnostic.median_smae,
                hindcast_srmse=diagnostic.median_srmse,
                tags=("numerical_package", alternative.family),
            )
        )
    if not candidates:
        raise ValueError("Numerical package has no materialized alternatives with valid metrics")
    return tuple(candidates)


def _safe_default(
    numerical: NumericalForecastPackage,
    candidates: Sequence[DecisionCandidate],
) -> DecisionCandidate:
    by_id = {item.candidate_id: item for item in candidates}
    selected = tuple(numerical.selection_decision.selected)
    if len(selected) == 1 and selected[0] in by_id:
        return by_id[selected[0]]
    protected = by_id.get(numerical.protected_baseline.name)
    if protected is None:
        raise ValueError("Numerical protected baseline is not a valid materialized alternative")
    return protected


def _validated_handoff(
    numerical: NumericalForecastPackage,
) -> tuple[tuple[RetrievalAssumption, ...], str | None]:
    raw_handoff = tuple(numerical.retrieval_handoff)
    if not raw_handoff:
        return (), "empty_retrieval_handoff"
    if len(raw_handoff) > _MAX_ASSUMPTIONS:
        return (), "invalid_retrieval_handoff"
    try:
        safe, _trace, expected = safe_retrieval_projection(
            numerical.accepted_assumptions,
            numerical.rejected_assumptions,
        )
        if safe != numerical.accepted_assumptions:
            raise ValueError("accepted assumption has no safe Retrieval projection")
        if tuple(dict(item) for item in raw_handoff) != tuple(
            dict(item) for item in expected
        ):
            raise ValueError("Retrieval handoff does not match accepted assumptions")
        assumptions = tuple(
            RetrievalAssumption.from_payload(item) for item in raw_handoff
        )
        if len({item.assumption_id for item in assumptions}) != len(assumptions):
            raise ValueError("duplicate Retrieval assumption IDs")
        for item in assumptions:
            payload = item.to_payload()
            if any(
                len(payload[field]) > limit
                for field, limit in _MAX_TEXT_LENGTHS.items()
            ):
                raise ValueError("Retrieval assumption text exceeds its host budget")
    except (TypeError, ValueError):
        return (), "invalid_retrieval_handoff"
    return assumptions, None


def _run_round1(
    retrieval: TwoStageRetrievalAgent,
    task: ContextTask,
) -> RetrievalRoundResult:
    try:
        raw = retrieval.run_round1(task)
        return _host_verify_retrieval_round(
            retrieval,
            task,
            raw,
            stage="round1",
        )
    except TransientLLMError:
        raise
    except Exception as error:
        return RetrievalRoundResult(
            (),
            (),
            ("invalid_retrieval_payload",),
            False,
            rejected=(
                "invalid_round1_response",
                f"round1_failure:{type(error).__name__}",
            ),
        )


def _run_round2(
    retrieval: TwoStageRetrievalAgent,
    task: ContextTask,
    round1: RetrievalRoundResult,
    gaps: tuple[RetrievalGap, ...],
    assumptions: tuple[RetrievalAssumption, ...],
) -> RetrievalRoundResult:
    try:
        raw = retrieval.run_round2(task, round1, gaps, assumptions)
        return _host_verify_retrieval_round(
            retrieval,
            task,
            raw,
            stage="round2",
            round1=round1,
            gaps=gaps,
            assumptions=assumptions,
        )
    except TransientLLMError:
        raise
    except Exception as error:
        return RetrievalRoundResult(
            (),
            (),
            ("invalid_retrieval_payload",),
            False,
            rejected=(
                "invalid_round2_response",
                f"round2_failure:{type(error).__name__}",
            ),
        )


def _host_verify_retrieval_round(
    retrieval: TwoStageRetrievalAgent,
    task: ContextTask,
    raw: RetrievalRoundResult,
    *,
    stage: str,
    round1: RetrievalRoundResult | None = None,
    gaps: tuple[RetrievalGap, ...] = (),
    assumptions: tuple[RetrievalAssumption, ...] = (),
) -> RetrievalRoundResult:
    """Replay typed agent output through the canonical host-owned verifier."""
    if type(raw) is not RetrievalRoundResult:
        raise RetrievalContractError("noncanonical retrieval round result")
    skills = TwoStageRetrievalAgent._skills(
        retrieval,
        stage,
        assumptions=assumptions,
        gaps=gaps,
    )
    skill_payloads = TwoStageRetrievalAgent._skill_payloads(skills)
    if stage == "round1":
        scope_payload = build_round1_payload(task, skills=skill_payloads)
    elif stage == "round2" and round1 is not None:
        scope_payload = build_round2_payload(
            task,
            round1,
            gaps,
            assumptions,
            skill_payloads,
        )
    else:
        raise ValueError("Round 2 host verification requires verified Round 1")
    selected_documents = _select_documents(
        scope_payload,
        retrieval.genome.max_selected_documents,
    )
    raw_payload = RetrievalRoundResult.to_payload(raw)
    dropped: list[str] = []
    for field in ("evidence_chains", "counterevidence"):
        collection = raw_payload.get(field, ())
        if not isinstance(collection, (list, tuple)):
            continue
        retained: list[object] = []
        for item in collection:
            citations = item.get("citations") if isinstance(item, Mapping) else None
            if isinstance(citations, (list, tuple)) and citations:
                retained.append(item)
                continue
            chain_id = (
                item.get("chain_id", "unknown")
                if isinstance(item, Mapping)
                else "unknown"
            )
            dropped.append(f"host_dropped_citationless_chain:{chain_id}")
        raw_payload[field] = retained
    if dropped:
        prior_rejected = raw_payload.get("rejected", ())
        if not isinstance(prior_rejected, (list, tuple)):
            prior_rejected = ()
        raw_payload["rejected"] = list(
            dict.fromkeys((*prior_rejected, *dropped))
        )
    bounded = TwoStageRetrievalAgent._bounded_response(retrieval, raw_payload)
    verified = verify_round_result(
        task,
        bounded,
        stage=stage,
        allowed_skill_ids=tuple(item.skill_id for item in skills),
        allowed_assumption_ids=(
            ()
            if stage == "round1"
            else tuple(item.assumption_id for item in assumptions)
        ),
        allowed_document_ids=tuple(
            item["document_id"] for item in selected_documents
        ),
        prior_round1=round1 if stage == "round2" else None,
    )
    max_quote_attempts = (
        retrieval.genome.max_evidence_chains
        * retrieval.genome.max_citations_per_chain
    )
    verified = replace(
        verified,
        quote_attempt_count=min(
            max_quote_attempts,
            max(raw.quote_attempt_count, verified.quote_attempt_count),
        ),
    )
    if stage == "round1" and verified.gaps:
        verified = replace(
            verified,
            gaps=(),
            rejected=tuple(
                dict.fromkeys((*verified.rejected, "round1_gaps_forbidden"))
            ),
        )
    return verified


def _run_decision(
    decision: DecisionAgent,
    candidates: tuple[DecisionCandidate, ...],
    retrieval: RetrievalResult,
    *,
    host_default: DecisionCandidate,
    assumptions: tuple[RetrievalAssumption, ...],
    round_index: int,
    prior: tuple[DecisionResult, ...] = (),
) -> tuple[DecisionResult, str | None]:
    try:
        result = decision.run(
            candidates,
            retrieval,
            host_default_id=host_default.candidate_id,
            prior_decisions=prior,
            round_index=round_index,
            assumptions=assumptions,
        )
        return _validate_decision_result(
            result,
            decision=decision,
            candidates=candidates,
            retrieval=retrieval,
            host_default=host_default,
            assumptions=assumptions,
        )
    except TransientLLMError:
        raise
    except Exception as error:
        reason = f"decision_failure:{type(error).__name__}"
        return _fallback_decision(host_default, reason), reason


def _validate_decision_result(
    result: object,
    *,
    decision: DecisionAgent,
    candidates: tuple[DecisionCandidate, ...],
    retrieval: RetrievalResult,
    host_default: DecisionCandidate,
    assumptions: tuple[RetrievalAssumption, ...],
) -> tuple[DecisionResult, str | None]:
    """Revalidate an untrusted Decision artifact against host-owned objects."""
    if type(result) is not DecisionResult:
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason
    if result.rejection_reason is not None:
        reason = "decision_contract_rejected"
        return _fallback_decision(host_default, reason), reason

    by_id = {item.candidate_id: item for item in candidates}
    canonical = by_id.get(getattr(result.selected, "candidate_id", None))
    if (
        type(result.selected) is not DecisionCandidate
        or canonical is None
        or result.selected != canonical
        or result.host_default_id != host_default.candidate_id
        or type(result.requested_more_retrieval) is not bool
        or type(result.llm_override_accepted) is not bool
        or not isinstance(result.rationale, str)
    ):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason

    citations = result.supporting_document_ids
    if (
        type(citations) is not tuple
        or any(not isinstance(item, str) or not item for item in citations)
        or len(citations) != len(set(citations))
    ):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason
    verified_ids = {item.document_id for item in retrieval.evidence}
    if not set(citations).issubset(verified_ids) or not set(
        canonical.source_document_ids
    ).issubset(citations):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason

    override = canonical.candidate_id != host_default.candidate_id
    if (
        result.llm_override_accepted is not override
        or (override and not citations)
    ):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason

    used_skills = result.used_skill_names
    if (
        type(used_skills) is not tuple
        or any(not isinstance(item, str) or not item for item in used_skills)
        or len(used_skills) != len(set(used_skills))
        or (
            decision.library is None
            and bool(used_skills)
        )
        or (
            decision.library is not None
            and any(decision.library.get(item) is None for item in used_skills)
        )
    ):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason

    gaps = result.gaps
    allowed_assumptions = {item.assumption_id for item in assumptions}
    try:
        canonical_gaps = tuple(
            RetrievalGap.from_payload(item.to_payload())
            for item in gaps
            if type(item) is RetrievalGap
        )
    except (RetrievalContractError, TypeError, ValueError):
        canonical_gaps = ()
    if (
        type(gaps) is not tuple
        or any(type(item) is not RetrievalGap for item in gaps)
        or canonical_gaps != gaps
        or len(gaps) != len({item.assumption_id for item in gaps})
        or any(item.assumption_id not in allowed_assumptions for item in gaps)
        or result.requested_more_retrieval is not bool(gaps)
    ):
        reason = "invalid_decision_result"
        return _fallback_decision(host_default, reason), reason

    return replace(result, selected=canonical), None


def _fallback_decision(default: DecisionCandidate, reason: str) -> DecisionResult:
    return DecisionResult(
        selected=default,
        host_default_id=default.candidate_id,
        requested_more_retrieval=False,
        rationale="Preserve the frozen Numerical safe default.",
        supporting_document_ids=(),
        llm_override_accepted=False,
        rejection_reason=reason,
        used_skill_names=(),
        gaps=(),
    )


def _should_run_round2(
    trigger: str,
    round1: RetrievalRoundResult,
    provisional: DecisionResult,
) -> bool:
    if trigger == "never":
        return False
    if trigger == "always":
        return True
    if trigger == "on_named_gap":
        return provisional.requested_more_retrieval and bool(provisional.gaps)
    if trigger == "on_incomplete_chain":
        return (
            not round1.sufficient
            or not round1.chains
            or any(item.missing_links or not item.numeric_eligible for item in round1.chains)
        )
    raise ValueError(f"unknown second-round trigger: {trigger}")


def _fatal_round_failure(result: RetrievalRoundResult, stage: str) -> bool:
    return (
        f"invalid_{stage}_response" in result.rejected
        or "invalid_retrieval_payload" in result.missing_information
    )


def _bind_round2_to_scope(
    result: RetrievalRoundResult,
    *,
    gaps: tuple[RetrievalGap, ...],
    assumptions: tuple[RetrievalAssumption, ...],
    binding_kind: str,
) -> RetrievalRoundResult:
    """Keep only Round 2 chains bound to the host-supplied retrieval scope."""
    if binding_kind == "gap":
        allowed = {item.assumption_id for item in gaps}
    elif binding_kind == "assumption":
        allowed = {item.assumption_id for item in assumptions}
    else:
        raise ValueError("unknown Round 2 binding kind")

    def bound(chain: object) -> bool:
        addressed = set(getattr(chain, "addressed_assumption_ids", ()))
        return bool(addressed) and addressed.issubset(allowed)

    chains = tuple(item for item in result.chains if bound(item))
    counterevidence = tuple(
        item for item in result.counterevidence if bound(item)
    )
    removed = tuple(
        f"round2_chain_not_{binding_kind}_bound:{item.chain_id}"
        for item in (*result.chains, *result.counterevidence)
        if not bound(item)
    )
    return replace(
        result,
        chains=chains,
        counterevidence=counterevidence,
        rejected=tuple(dict.fromkeys((*result.rejected, *removed))),
    )


def _record_rejection(
    card: FinalRetrievalCard,
    reason: str,
) -> FinalRetrievalCard:
    return replace(card, rejected=tuple(dict.fromkeys((*card.rejected, reason))))


def _execution_fingerprints(
    task: ContextTask,
    numerical: NumericalForecastPackage,
    retrieval: TwoStageRetrievalAgent,
    decision: DecisionAgent,
    candidates: tuple[DecisionCandidate, ...],
) -> Mapping[str, str]:
    skill_payloads = [
        item.to_payload()
        for item in sorted(
            retrieval.skills.all(),
            key=lambda value: (value.skill_id, value.version),
        )
    ]
    skills_fingerprint = _fingerprint(skill_payloads)
    active_ids = frozenset(retrieval.genome.active_skill_ids)
    train_shadow_ids = tuple(
        skill_id
        for skill_id in _trusted_train_shadow_skill_ids(retrieval.skills)
        if skill_id in active_ids
    )
    result = {
        "bridge_contract": _fingerprint(_BRIDGE_CONTRACT),
        "context_projection": _fingerprint(task.retrieval_view()),
        "decision_candidates": _fingerprint(
            [_decision_candidate_payload(item) for item in candidates]
        ),
        "decision_host_contract": _fingerprint(_DECISION_HOST_CONTRACT),
        "metric_policy": METRIC_POLICY_FINGERPRINT,
        "morphology_projection": _fingerprint(
            {
                "morphology_card_fingerprint": (
                    numerical.morphology_card.fingerprint
                    if numerical.morphology_card is not None
                    else None
                ),
                "accepted_assumptions": [
                    item.to_payload() for item in numerical.accepted_assumptions
                ],
                "retrieval_handoff": [
                    dict(item) for item in numerical.retrieval_handoff
                ],
            }
        ),
        "numerical_package": _numerical_package_fingerprint(numerical),
        "retrieval_verifier_contract": _fingerprint(_RETRIEVAL_VERIFIER_CONTRACT),
        "retrieval_genome": retrieval.genome.fingerprint(),
        "retrieval_skills": skills_fingerprint,
        "retrieval_skill_scope": _fingerprint(
            {
                "schema_version": 1,
                "retrieval_genome_sha256": retrieval.genome.fingerprint(),
                "retrieval_skills_sha256": skills_fingerprint,
                "trusted_train_shadow_skill_ids": list(train_shadow_ids),
            }
        ),
        "decision_prompt": hashlib.sha256(decision.prompt.encode("utf-8")).hexdigest(),
        "decision_skills": _fingerprint(
            [
                asdict(item)
                for item in sorted(
                    decision.library.all() if decision.library is not None else (),
                    key=lambda value: (value.skill_id, value.name),
                )
            ]
        ),
    }
    for name, value in numerical.component_fingerprints.items():
        key = f"numerical_{name.removesuffix('_fingerprint')}"
        if key in result:
            raise ValueError(f"Numerical component fingerprint key conflicts with {key}")
        result[key] = (
            value
            if _SHA256.fullmatch(value)
            else _fingerprint({"component": name, "identity": value})
        )
    return _freeze_fingerprints(result)


def _completed_fingerprints(
    execution: Mapping[str, str],
    retrieval_card: FinalRetrievalCard,
    provisional: DecisionResult,
    final: DecisionResult,
) -> Mapping[str, str]:
    """Bind the exact verified Retrieval and host-validated Decision artifacts."""
    result = dict(execution)
    result.update(
        {
            "final_retrieval_artifact": _fingerprint(retrieval_card.to_payload()),
            "provisional_decision_artifact": _fingerprint(
                _decision_result_payload(provisional)
            ),
            "final_decision_artifact": _fingerprint(_decision_result_payload(final)),
        }
    )
    return _freeze_fingerprints(result)


def _decision_result_payload(result: DecisionResult) -> Mapping[str, object]:
    return {
        "selected": _decision_candidate_payload(result.selected),
        "host_default_id": result.host_default_id,
        "requested_more_retrieval": result.requested_more_retrieval,
        "rationale": result.rationale,
        "supporting_document_ids": list(result.supporting_document_ids),
        "llm_override_accepted": result.llm_override_accepted,
        "rejection_reason": result.rejection_reason,
        "used_skill_names": list(result.used_skill_names),
        "gaps": [item.to_payload() for item in result.gaps],
    }


def _decision_candidate_payload(
    candidate: DecisionCandidate,
) -> Mapping[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "forecast": list(candidate.forecast),
        "assumption": candidate.assumption,
        "failure_condition": candidate.failure_condition,
        "hindcast_smae": candidate.hindcast_smae,
        "hindcast_srmse": candidate.hindcast_srmse,
        "source_document_ids": list(candidate.source_document_ids),
        "tags": list(candidate.tags),
        "hindcast_smape": candidate.hindcast_smape,
    }


def _numerical_package_fingerprint(numerical: NumericalForecastPackage) -> str:
    """Bind the exact safe runtime projection without serializing internal folds."""
    selection = numerical.selection_decision
    return _fingerprint(
        {
            "task_id": numerical.task_profile.task_id,
            "task_input_fingerprint": numerical.component_fingerprints["task_input"],
            "task_profile": numerical.task_profile.to_public_payload(),
            "morphology_card_fingerprint": (
                numerical.morphology_card.fingerprint
                if numerical.morphology_card is not None
                else None
            ),
            "accepted_assumptions": [
                item.to_payload() for item in numerical.accepted_assumptions
            ],
            "active_candidate_names": list(numerical.active_candidate_names),
            "selection": {
                "mode": selection.mode,
                "selected": list(selection.selected),
                "weights": list(selection.weights),
                "forecast": list(selection.forecast),
                "baseline_name": selection.baseline_name,
            },
            "protected_baseline": {
                "name": numerical.protected_baseline.name,
                "forecast": list(numerical.protected_baseline.forecast),
            },
            "ranked_alternatives": [
                {
                    "rank": item.rank,
                    "name": item.name,
                    "family": item.family,
                    "forecast": list(item.forecast),
                    "median_smae": _canonical_metric(item.diagnostics.median_smae),
                    "median_srmse": _canonical_metric(item.diagnostics.median_srmse),
                }
                for item in numerical.ranked_alternatives
            ],
            "retrieval_handoff": [dict(item) for item in numerical.retrieval_handoff],
            "component_fingerprints": dict(numerical.component_fingerprints),
        }
    )


def _canonical_metric(value: float) -> float | str:
    number = float(value)
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "nan"
    return "+inf" if number > 0.0 else "-inf"


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_fingerprints(values: Mapping[str, str]) -> Mapping[str, str]:
    result = dict(values)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not _SHA256.fullmatch(value)
        for key, value in result.items()
    ):
        raise ValueError("execution fingerprint map contains a noncanonical fingerprint")
    return MappingProxyType(dict(sorted(result.items())))


__all__ = ["NumericalTwoStageResult", "run_numerical_two_stage"]
