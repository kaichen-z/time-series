from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict
from typing import Any

from .codex_agents import CodexCLIClient, IMPACT_SCHEMA, _task_payload
from .models import (
    CandidateDecision,
    Diagnosis,
    Evidence,
    EvidenceImpact,
    ForecastCandidate,
    ForecastTask,
    QueryAction,
    RetrievedDocument,
)
from .regimes import RegimeNormalizationAgent
from .triad import CandidateValidation, CodingForecastAgent, DecisionForecastAgent


CODING_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_families", "assumptions", "information_needs"],
    "properties": {
        "candidate_families": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "string",
                "enum": ["backbone", "statistical", "history_robust", "level"],
            },
        },
        "assumptions": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["family", "assumption"],
                "properties": {
                    "family": {
                        "type": "string",
                        "enum": ["backbone", "statistical", "history_robust", "level"],
                    },
                    "assumption": {"type": "string", "maxLength": 1200},
                },
            },
        },
        "information_needs": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 500},
        },
    },
}


CODING_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selected_candidate_ids", "rationale"],
    "properties": {
        "selected_candidate_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "rationale": {"type": "string", "maxLength": 1800},
    },
}


RETRIEVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "query",
        "rationale",
        "selected_document_ids",
        "evidence",
        "impacts",
        "sufficient",
    ],
    "properties": {
        "query": {"type": "string", "maxLength": 1200},
        "rationale": {"type": "string", "maxLength": 1800},
        "selected_document_ids": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
        "evidence": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["document_id", "claim", "exact_quote", "confidence"],
                "properties": {
                    "document_id": {"type": "string"},
                    "claim": {"type": "string", "maxLength": 1500},
                    "exact_quote": {"type": "string", "maxLength": 2400},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "impacts": IMPACT_SCHEMA["properties"]["impacts"],
        "sufficient": {"type": "boolean"},
    },
}


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "selected_candidate_id",
        "request_more_retrieval",
        "request_new_candidates",
        "rationale",
        "supporting_document_ids",
    ],
    "properties": {
        "selected_candidate_id": {"type": "string"},
        "request_more_retrieval": {"type": "boolean"},
        "request_new_candidates": {"type": "boolean"},
        "rationale": {"type": "string", "maxLength": 2200},
        "supporting_document_ids": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
    },
}


CODING_AGENT_PROMPT = """You are the Coding Agent in a contextual time-series forecasting system.
Read task.json, but do not inspect any context documents. Your job is to propose a small,
diverse set of executable numerical hypotheses from the historical numbers only. Select from:
backbone (the configured TSFM), statistical (transparent trend/seasonality), history_robust
(winsorize possible observation artifacts before forecasting), and level (local-regime mean).
Do not output future values and do not pretend to know future events. State the falsifiable
assumption behind each selected family and list the textual information that would distinguish
the hypotheses. Avoid selecting redundant families merely to increase the count."""


RETRIEVAL_AGENT_PROMPT = """You are the Retrieval Agent in a contextual time-series forecasting
system. Read task.json and candidate_hypotheses.json, then search the local documents/ directory.
Retrieve evidence that can distinguish the competing numerical hypotheses for the exact entity,
target variable, history cutoff, and forecast window. Filter wrong entities, wrong dates,
post-cutoff hindsight, irrelevant operational details, and documents that merely assert an
unsupported future time-series shape. An observation/recording bug and a real demand event are
not interchangeable: identify whether evidence concerns the latent process, the observation
mechanism, or a future regime. Every accepted claim needs a verbatim exact_quote from its cited
document. Translate accepted evidence into typed impacts, but never invent a date or magnitude;
use qualitative_only when the source is not quantitative. Encode percentage adjustments as a
fraction (for example, 4 percent is 0.04, not 4). Under the Dr-CiK corpus contract, an undated
document is assumed available at the history cutoff. A pre-cutoff plan, schedule, or analytic
forecast about the future is eligible evidence; do not call it hindsight merely because it
describes dates inside the forecast window. Only an explicit publication/report date after the
cutoff makes it hindsight. Search specifically for forward-looking plans or regime statements
covering the forecast window. Historical explanations alone are not sufficient when the corpus
may contain eligible information about the future window. Do not read outside this workspace."""


DECISION_AGENT_PROMPT = """You are the Decision Agent in a contextual time-series forecasting
system. Read candidates.json and evidence.json. Select exactly one existing candidate_id; never
invent or directly edit forecast values. Rolling-validation scores measure historical numerical
fit, while verified evidence tests whether a candidate's assumptions remain valid in the future.
Prefer a candidate only when both sources of information support it. Treat unsupported narrative
predictions and unverified magnitudes as weak evidence. If the current evidence cannot distinguish
plausible candidates, request more retrieval. Evidence that only explains historical anomalies is
not sufficient when no evidence yet addresses the forecast window and documents remain. If verified
active evidence has no executable candidate, request new candidates. Cite only supplied document
IDs. The hidden future and benchmark labels are unavailable and must never be inferred."""


def _candidate_family(candidate: ForecastCandidate) -> str:
    for family in ("history_robust", "statistical", "level", "backbone"):
        if family in candidate.tags:
            return family
    return "backbone"


class CodexCodingForecastAgent(CodingForecastAgent):
    """Codex proposes hypotheses; deterministic tools execute and backtest them."""

    def __init__(self, *args: Any, client: CodexCLIClient, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = client
        self.information_needs: tuple[str, ...] = ()
        self.normal_regime_agent = RegimeNormalizationAgent()

    @staticmethod
    def _workspace(task: ForecastTask, diagnosis: Diagnosis) -> dict[str, str]:
        return {
            "task.json": json.dumps(
                {
                    "task": _task_payload(task, diagnosis),
                    "history_timestamps": task.history_timestamps,
                    "history_values": task.history_values,
                    "future_timestamps": task.future_timestamps,
                    "available_program_families": list(CodingForecastAgent.PROGRAM_FAMILIES),
                },
                ensure_ascii=False,
                indent=2,
            )
        }

    def initial_candidates(
        self, task: ForecastTask, diagnosis: Diagnosis
    ) -> tuple[list[ForecastCandidate], str]:
        candidates, method = super().initial_candidates(task, diagnosis)
        result = self.client.complete(
            f"triad_coding_plan_{task.benchmark_id}",
            CODING_AGENT_PROMPT,
            CODING_PLAN_SCHEMA,
            workspace_files=self._workspace(task, diagnosis),
        )
        if not result:
            return candidates, method
        selected_families = {
            str(value) for value in result.get("candidate_families", [])
            if str(value) in self.PROGRAM_FAMILIES
        }
        selected_families.add("backbone")
        assumptions = {
            str(item.get("family")): str(item.get("assumption", "")).strip()
            for item in result.get("assumptions", [])
            if str(item.get("assumption", "")).strip()
        }
        selected = []
        for candidate in candidates:
            family = _candidate_family(candidate)
            if family not in selected_families:
                continue
            if family in assumptions:
                candidate = type(candidate)(
                    **{**asdict(candidate), "assumption": assumptions[family]}
                )
            selected.append(candidate)
        self.information_needs = tuple(
            str(value).strip() for value in result.get("information_needs", [])
            if str(value).strip()
        )
        return selected or candidates, method

    def expand_candidates(
        self,
        task: ForecastTask,
        candidates: list[ForecastCandidate],
        impacts: list[EvidenceImpact],
        round_index: int,
    ) -> list[ForecastCandidate]:
        expanded = super().expand_candidates(task, candidates, impacts, round_index)
        regime_impacts = [
            impact
            for impact in impacts
            if impact.event_type == "forecast_regime"
            and impact.forecast_relation in {"overlaps_forecast", "forecast_relevant_undated"}
            and impact.confidence >= 0.6
            and (
                "baseline" in impact.rationale.lower()
                or "seasonal" in impact.rationale.lower()
                or "normal" in impact.rationale.lower()
            )
        ]
        if regime_impacts:
            diagnosis = self.diagnoser.diagnose(task)
            baseline = next(
                (item for item in candidates if "backbone" in item.tags), None
            )
            if baseline is not None:
                regime_evidence = [
                    Evidence(
                        document_id=source_id,
                        claim=(
                            f"For {task.entity_name} {task.target_name}, verified evidence "
                            "says the forecast should follow the historical baseline and seasonality. "
                            + impact.rationale
                        ),
                        matched_terms=(),
                        confidence=impact.confidence,
                        effect_direction=impact.direction,
                        effect_window="forecast",
                        entity=task.entity_name,
                        target_variable=task.target_name,
                        evidence_quote=impact.rationale,
                        provenance_valid=True,
                    )
                    for impact in regime_impacts
                    for source_id in impact.source_document_ids
                ]
                projection = self.normal_regime_agent.project(
                    task, diagnosis, baseline.values, regime_evidence
                )
                if projection is not None:
                    differences = [
                        abs(task.history_values[index] - task.history_values[index - 1])
                        for index in range(1, len(task.history_values))
                    ]
                    scale = max(statistics.fmean(differences), 1e-8)
                    validation = CandidateValidation(
                        historical_score=1.0 / (1.0 + projection.validation_mae / scale),
                        mae=projection.validation_mae,
                        scaled_mae=projection.validation_mae / scale,
                        folds=1,
                    )
                    expanded.append(
                        self._candidate(
                            f"c_r{round_index}_normal_regime",
                            round_index,
                            f"normal_regime_harmonic_p{projection.seasonal_period}",
                            projection.values,
                            (
                                "Verified forecast-window evidence predicts a return to the "
                                "normal historical baseline and seasonality; test a backtested "
                                "trend-harmonic projection blended with the backbone."
                            ),
                            (
                                "evidence_adjusted",
                                "forecast_regime",
                                "normal_regime",
                                "history_backtested",
                            ),
                            validation,
                            (baseline.candidate_id,),
                            projection.source_document_ids,
                        )
                    )
                    expanded = self._deduplicate(expanded)
        if len(expanded) == len(candidates):
            return expanded
        payload = {
            "round": round_index,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "program_id": item.program_id,
                    "assumption": item.assumption,
                    "tags": item.tags,
                    "historical_score": item.historical_score,
                    "source_document_ids": item.source_document_ids,
                }
                for item in expanded
            ],
            "verified_impacts": [asdict(item) for item in impacts],
        }
        result = self.client.complete(
            f"triad_coding_revision_{task.benchmark_id}_{round_index}",
            "You are the Coding Agent. Read candidate_workspace.json. Choose which "
            "already-executed candidates remain logically distinct and evidence-grounded. "
            "Do not invent IDs or future values.",
            CODING_SELECTION_SCHEMA,
            workspace_files={"candidate_workspace.json": json.dumps(payload, ensure_ascii=False, indent=2)},
        )
        if not result:
            return expanded
        known = {item.candidate_id for item in expanded}
        selected_ids = {
            str(value) for value in result.get("selected_candidate_ids", [])
            if str(value) in known
        }
        selected_ids.update(
            item.candidate_id for item in expanded if "backbone" in item.tags
        )
        return [item for item in expanded if item.candidate_id in selected_ids] or expanded


class _StoredImpactTranslator:
    def __init__(self) -> None:
        self.by_key: dict[tuple[tuple[str, ...], str, str | None, str | None], EvidenceImpact] = {}

    def add(self, impacts: list[EvidenceImpact]) -> None:
        for impact in impacts:
            key = (
                impact.source_document_ids,
                impact.event_type,
                impact.start_timestamp,
                impact.end_timestamp,
            )
            self.by_key[key] = impact

    def translate(self, *_args: Any, **_kwargs: Any) -> list[EvidenceImpact]:
        return list(self.by_key.values())


class CodexRetrievalStreamAgent:
    """Codex searches the full local corpus and returns grounded evidence."""

    def __init__(self, client: CodexCLIClient, information_needs_source: Any) -> None:
        self.client = client
        self.information_needs_source = information_needs_source
        self.impact_agent = _StoredImpactTranslator()

    @staticmethod
    def _workspace(
        task: ForecastTask,
        diagnosis: Diagnosis,
        candidates: list[ForecastCandidate],
        seen_ids: set[str],
    ) -> dict[str, str]:
        files: dict[str, str] = {}
        manifest = []
        for index, document in enumerate(task.documents):
            if document.document_id in seen_ids:
                continue
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.document_id)
            path = f"documents/{index:04d}-{safe_id}.md"
            files[path] = document.text
            manifest.append({"document_id": document.document_id, "path": path})
        files["task.json"] = json.dumps(
            {
                "task": _task_payload(task, diagnosis),
                "history_cutoff": task.history_timestamps[-1],
                "documents": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        files["candidate_hypotheses.json"] = json.dumps(
            [
                {
                    "candidate_id": item.candidate_id,
                    "program_id": item.program_id,
                    "assumption": item.assumption,
                    "tags": item.tags,
                    "historical_score": item.historical_score,
                }
                for item in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )
        return files

    @staticmethod
    def _parse_impacts(raw_impacts: list[dict[str, Any]], known: set[str]) -> list[EvidenceImpact]:
        output = []
        for raw in raw_impacts:
            sources = tuple(dict.fromkeys(
                str(value) for value in raw.get("source_document_ids", [])
                if str(value) in known
            ))
            if not sources:
                continue
            try:
                adjustment_value = (
                    float(raw["adjustment_value"])
                    if raw.get("adjustment_value") is not None
                    else None
                )
                if (
                    str(raw.get("adjustment_kind")) == "percentage"
                    and adjustment_value is not None
                    and 1.0 < abs(adjustment_value) <= 100.0
                ):
                    adjustment_value /= 100.0
                output.append(EvidenceImpact(
                    source_document_ids=sources,
                    event_type=str(raw["event_type"]),
                    start_timestamp=raw.get("start_timestamp"),
                    end_timestamp=raw.get("end_timestamp"),
                    direction=str(raw["direction"]),
                    permanence=str(raw["permanence"]),
                    forecast_relation=str(raw["forecast_relation"]),
                    adjustment_kind=str(raw["adjustment_kind"]),
                    adjustment_value=adjustment_value,
                    confidence=float(raw["confidence"]),
                    rationale="Codex retrieval impact: " + str(raw["rationale"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return output

    def retrieve(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        candidates: list[ForecastCandidate],
        top_k: int,
        seen_ids: set[str],
    ) -> tuple[QueryAction, list[RetrievedDocument], list[RetrievedDocument], list[Evidence]]:
        needs = tuple(getattr(self.information_needs_source, "information_needs", ()))
        prompt = RETRIEVAL_AGENT_PROMPT
        if needs:
            prompt += "\nCoding Agent information needs:\n- " + "\n- ".join(needs)
        result = self.client.complete(
            f"triad_retrieval_{task.benchmark_id}_{len(seen_ids)}",
            prompt,
            RETRIEVAL_SCHEMA,
            workspace_files=self._workspace(task, diagnosis, candidates, seen_ids),
        )
        fallback_query = f"{task.entity_name} {task.target_name} forecast evidence"
        if not result:
            action = QueryAction("codex_retrieval", "Find hypothesis-discriminating evidence", fallback_query, "Codex retrieval failed; no evidence accepted.")
            return action, [], [], []
        by_id = {document.document_id: document.agent_view() for document in task.documents}
        selected_ids = []
        for value in result.get("selected_document_ids", []):
            document_id = str(value)
            if document_id in by_id and document_id not in seen_ids and document_id not in selected_ids:
                selected_ids.append(document_id)
        evidence: list[Evidence] = []
        for item in result.get("evidence", []):
            document_id = str(item.get("document_id", ""))
            quote = str(item.get("exact_quote", ""))
            if document_id not in by_id or document_id in seen_ids or not quote or quote not in by_id[document_id].text:
                continue
            if document_id not in selected_ids:
                selected_ids.append(document_id)
            evidence.append(Evidence(
                document_id=document_id,
                claim=str(item.get("claim", "")),
                matched_terms=(),
                confidence=float(item.get("confidence", 0.0)),
                effect_direction="unknown",
                effect_window="forecast_or_history",
                gap_id="candidate_disagreement",
                entity=task.entity_name,
                target_variable=task.target_name,
                evidence_quote=quote,
                provenance_valid=True,
            ))
        grounded_ids = {item.document_id for item in evidence}
        selected_ids = [value for value in selected_ids if value in grounded_ids][:top_k]
        selected_set = set(selected_ids)
        evidence = [item for item in evidence if item.document_id in selected_set]
        impacts = self._parse_impacts(result.get("impacts", []), selected_set)
        self.impact_agent.add(impacts)
        retrieved = [
            RetrievedDocument(by_id[document_id], 1.0, rank)
            for rank, document_id in enumerate(selected_ids, start=1)
        ]
        action = QueryAction(
            question_id="candidate_disagreement",
            question="Which evidence distinguishes the executable forecast hypotheses?",
            query=str(result.get("query", fallback_query)),
            rationale=str(result.get("rationale", "Codex searched the local corpus.")),
        )
        return action, retrieved, retrieved, evidence


class CodexDecisionForecastAgent(DecisionForecastAgent):
    """Codex selects; the host validates the ID and computes the final series."""

    def __init__(self, *args: Any, client: CodexCLIClient, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = client

    def decide(
        self,
        candidates: list[ForecastCandidate],
        impacts: list[EvidenceImpact],
        round_index: int,
        max_rounds: int,
        documents_remaining: bool,
    ) -> CandidateDecision:
        fallback = super().decide(candidates, impacts, round_index, max_rounds, documents_remaining)
        payload = {
            "round": round_index,
            "max_rounds": max_rounds,
            "documents_remaining": documents_remaining,
            "candidates": [asdict(item) for item in candidates],
            "verified_evidence_impacts": [asdict(item) for item in impacts],
            "host_assessments": [asdict(item) for item in fallback.assessments],
        }
        result = self.client.complete(
            f"triad_decision_{round_index}_{len(candidates)}",
            DECISION_AGENT_PROMPT,
            DECISION_SCHEMA,
            workspace_files={
                "candidates.json": json.dumps(payload, ensure_ascii=False, indent=2),
                "evidence.json": json.dumps([asdict(item) for item in impacts], ensure_ascii=False, indent=2),
            },
        )
        if not result:
            return fallback
        compatible = {
            item.candidate_id for item in fallback.assessments if item.evidence_compatible
        }
        selected_id = str(result.get("selected_candidate_id", ""))
        if selected_id not in compatible:
            return fallback
        can_continue = round_index < max_rounds and documents_remaining
        return CandidateDecision(
            selected_candidate_ids=(selected_id,),
            selected_weights=(1.0,),
            assessments=fallback.assessments,
            request_more_retrieval=bool(result.get("request_more_retrieval")) and can_continue,
            request_new_candidates=(
                fallback.request_new_candidates
                or (bool(result.get("request_new_candidates")) and round_index < max_rounds)
            ),
            rationale="Codex decision: " + str(result.get("rationale", "")),
        )
