from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .agents import ProbabilisticForecastAgent, TimeSeriesDiagnosisAgent
from .backbones import ChronosBackboneConfig, TimesFMBackboneConfig, build_forecast_backbone
from .codex_agents import CodexCLIClient, CodexCLIConfig
from .explicit_values import ExplicitValueValidation, ExplicitValueValidator
from .metrics import forecast_metrics, retrieval_metrics
from .models import Evidence, ForecastTask, RetrievedDocument, RunResult
from .regimes import RegimeNormalizationAgent
from .workspace import ForecastWorkspaceExecutor, RevisionPlannerAgent


CODEX_DIRECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["research_report", "cited_document_ids", "evidence", "forecast_values"],
    "properties": {
        "research_report": {"type": "string"},
        "cited_document_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "document_ids", "exact_quotes"],
                "properties": {
                    "claim": {"type": "string"},
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "exact_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "forecast_values": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
}


CODEX_CONTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["research_report", "cited_document_ids", "evidence", "forecast_contract"],
    "properties": {
        "research_report": {"type": "string"},
        "cited_document_ids": CODEX_DIRECT_SCHEMA["properties"]["cited_document_ids"],
        "evidence": CODEX_DIRECT_SCHEMA["properties"]["evidence"],
        "forecast_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "regime",
                "expected_behavior",
                "seasonality",
                "anomalous_history_windows",
                "future_event_windows",
                "confidence",
                "rationale",
            ],
            "properties": {
                "regime": {
                    "type": "string",
                    "enum": [
                        "normal_seasonal",
                        "normal_nonseasonal",
                        "temporary_future_event",
                        "permanent_shift",
                        "explicit_future_values",
                        "insufficient_evidence",
                    ],
                },
                "expected_behavior": {"type": "string"},
                "seasonality": {
                    "type": "string",
                    "enum": ["preserve", "suppress", "unclear"],
                },
                "anomalous_history_windows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start", "end", "reason"],
                        "properties": {
                            "start": {"type": ["string", "null"]},
                            "end": {"type": ["string", "null"]},
                            "reason": {"type": "string"},
                        },
                    },
                },
                "future_event_windows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start", "end", "direction", "reason"],
                        "properties": {
                            "start": {"type": ["string", "null"]},
                            "end": {"type": ["string", "null"]},
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down", "stable", "unclear"],
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        },
    },
}


@dataclass(frozen=True)
class CodexDirectConfig:
    num_samples: int = 100
    seed: int = 7
    backbone: str = "chronos"
    chronos_model_id: str = "amazon/chronos-bolt-small"
    chronos_device_map: str = "cpu"
    chronos_max_context: int = 2048
    chronos_max_horizon: int = 1024
    chronos_cache_dir: str | None = None
    chronos_local_files_only: bool = False
    timesfm_model_id: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_max_context: int = 4096
    timesfm_max_horizon: int = 1024
    timesfm_cache_dir: str | None = None
    timesfm_local_files_only: bool = False
    allow_statistical_fallback: bool = False
    codex_binary: str = "codex"
    codex_model: str | None = None
    codex_cache_dir: str = "outputs/codex-cache"
    codex_timeout_seconds: int = 300
    codex_reasoning_effort: str = "high"
    explicit_value_weight: float = 0.75


class CodexDirectBaseline:
    """Full-corpus Codex research plus unconstrained direct forecast baseline.

    This intentionally omits the proposed system's retriever, verifier,
    evidence-to-impact translator, memory, and restricted revision workspace.
    """

    def __init__(self, config: CodexDirectConfig | None = None) -> None:
        self.config = config or CodexDirectConfig()
        self.diagnosis_agent = TimeSeriesDiagnosisAgent()
        backbone = build_forecast_backbone(
            self.config.backbone,
            chronos_config=ChronosBackboneConfig(
                model_id=self.config.chronos_model_id,
                device_map=self.config.chronos_device_map,
                max_context=self.config.chronos_max_context,
                max_horizon=self.config.chronos_max_horizon,
                cache_dir=self.config.chronos_cache_dir,
                local_files_only=self.config.chronos_local_files_only,
            ),
            timesfm_config=TimesFMBackboneConfig(
                model_id=self.config.timesfm_model_id,
                max_context=self.config.timesfm_max_context,
                max_horizon=self.config.timesfm_max_horizon,
                cache_dir=self.config.timesfm_cache_dir,
                local_files_only=self.config.timesfm_local_files_only,
            ),
            allow_statistical_fallback=self.config.allow_statistical_fallback,
        )
        self.forecast_agent = ProbabilisticForecastAgent(backbone)
        self.codex = CodexCLIClient(
            CodexCLIConfig(
                binary=self.config.codex_binary,
                model=self.config.codex_model,
                cache_dir=self.config.codex_cache_dir,
                timeout_seconds=self.config.codex_timeout_seconds,
                reasoning_effort=self.config.codex_reasoning_effort,
            )
        )

    @staticmethod
    def _workspace(task: ForecastTask, baseline: tuple[float, ...]) -> dict[str, str]:
        manifest: list[dict[str, str]] = []
        files: dict[str, str] = {}
        for index, document in enumerate(task.documents):
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.document_id)
            filename = f"documents/{index:04d}-{safe_id}.md"
            manifest.append({"document_id": document.document_id, "path": filename})
            files[filename] = document.text
        files["task.json"] = json.dumps(
            {
                "benchmark_id": task.benchmark_id,
                "entity": task.entity_name,
                "target_variable": task.target_name,
                "target_description": task.target_description,
                "frequency": task.frequency,
                "prediction_length": task.prediction_length,
                "history_timestamps": task.history_timestamps,
                "history_values": task.history_values,
                "future_timestamps": task.future_timestamps,
                "chronos_baseline": baseline,
                "documents": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        return files

    def run(self, task: ForecastTask) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        baseline, baseline_method = self.forecast_agent.baseline(task, diagnosis)
        prompt = (
            "You are the Codex-Direct baseline for contextual time-series forecasting. "
            "Read task.json. Search the local documents/ corpus autonomously with shell reading "
            "and search tools. Identify evidence that was available by the history cutoff and is "
            "relevant to the exact entity, target, and forecast window. Reject wrong entities, "
            "wrong dates, hindsight, and misleading time-series claims. Then directly produce one "
            "forecast value for every future_timestamps entry. The Chronos forecast in task.json "
            "is the numerical baseline; revise it only when the discovered evidence warrants it. "
            "Citations must use document_id values from task.json and exact_quotes must be copied "
            "verbatim. Return exactly prediction_length forecast_values. Do not read anything "
            "outside this temporary workspace."
        )
        before = self.codex.stats()
        result = self.codex.complete(
            "codex_direct",
            prompt,
            CODEX_DIRECT_SCHEMA,
            workspace_files=self._workspace(task, baseline),
        )
        after = self.codex.stats()

        known = {document.document_id: document.agent_view() for document in task.documents}
        cited_ids: list[str] = []
        if result:
            for document_id in result.get("cited_document_ids", []):
                if document_id in known and document_id not in cited_ids:
                    cited_ids.append(document_id)
            for item in result.get("evidence", []):
                for document_id in item.get("document_ids", []):
                    if document_id in known and document_id not in cited_ids:
                        cited_ids.append(document_id)
        retrieved = [
            RetrievedDocument(known[document_id], 1.0, rank)
            for rank, document_id in enumerate(cited_ids, start=1)
        ]

        evidence: list[Evidence] = []
        if result:
            for item in result.get("evidence", []):
                source_ids = [item_id for item_id in item.get("document_ids", []) if item_id in known]
                if not source_ids:
                    continue
                quotes = [str(value) for value in item.get("exact_quotes", []) if str(value)]
                evidence.append(
                    Evidence(
                        document_id=source_ids[0],
                        claim=str(item.get("claim", "")),
                        matched_terms=(),
                        confidence=1.0,
                        effect_direction="unknown",
                        effect_window="forecast",
                        evidence_quote="\n".join(quotes),
                        provenance_valid=all(
                            any(quote in known[source_id].text for source_id in source_ids)
                            for quote in quotes
                        ),
                    )
                )

        proposed = result.get("forecast_values", []) if result else []
        valid_forecast = (
            len(proposed) == task.prediction_length
            and all(isinstance(value, (int, float)) for value in proposed)
        )
        mean = tuple(float(value) for value in proposed) if valid_forecast else baseline
        if min(task.history_values) >= 0:
            mean = tuple(max(0.0, value) for value in mean)
        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        forecast = self.forecast_agent.forecast_from_mean(
            task=task,
            diagnosis=diagnosis,
            mean=mean,
            baseline_mean=baseline,
            baseline_method=f"{baseline_method}+codex-direct" if valid_forecast else baseline_method,
            num_samples=self.config.num_samples,
            seed=task_seed,
        )
        metrics = retrieval_metrics(task, retrieved, evidence)
        metrics.update(forecast_metrics(task, forecast))
        metrics.update(
            {
                "codex_calls": float(after["calls"] - before["calls"]),
                "codex_cache_hits": float(after["cache_hits"] - before["cache_hits"]),
                "codex_failures": float(after["failures"] - before["failures"]),
                "codex_latency_seconds": float(after["latency_seconds"] - before["latency_seconds"]),
                "codex_valid_forecast": float(valid_forecast),
            }
        )
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=retrieved,
            evidence=evidence,
            forecast=forecast,
            metrics=metrics,
            loop_trace=[
                {
                    "system": "codex-direct-baseline",
                    "codex_model": self.config.codex_model or "cli-default",
                    "codex_reasoning_effort": self.config.codex_reasoning_effort,
                    "research_report": result.get("research_report", "") if result else "",
                    "cited_document_ids": cited_ids,
                    "codex": after,
                    "fallback_to_backbone": not valid_forecast,
                }
            ],
        )

    def run_many(self, tasks: list[ForecastTask]) -> list[RunResult]:
        return [self.run(task) for task in tasks]


class CodexContractSystem(CodexDirectBaseline):
    """Codex proposes a falsifiable regime contract; numeric tools forecast.

    Unlike ``CodexDirectBaseline``, Codex never emits the future trajectory.
    Text may open a candidate-model gate, but every magnitude comes from a
    history-only backtest and every edit goes through the restricted workspace.
    """

    def __init__(self, config: CodexDirectConfig | None = None) -> None:
        super().__init__(config)
        self.regime_agent = RegimeNormalizationAgent()
        self.explicit_value_validator = ExplicitValueValidator()
        self.revision_planner = RevisionPlannerAgent()
        self.workspace_executor = ForecastWorkspaceExecutor()

    @staticmethod
    def _parse_research(
        task: ForecastTask,
        result: dict | None,
    ) -> tuple[list[RetrievedDocument], list[Evidence]]:
        known = {document.document_id: document.agent_view() for document in task.documents}
        cited_ids: list[str] = []
        if result:
            for document_id in result.get("cited_document_ids", []):
                if document_id in known and document_id not in cited_ids:
                    cited_ids.append(document_id)
            for item in result.get("evidence", []):
                for document_id in item.get("document_ids", []):
                    if document_id in known and document_id not in cited_ids:
                        cited_ids.append(document_id)
        retrieved = [
            RetrievedDocument(known[document_id], 1.0, rank)
            for rank, document_id in enumerate(cited_ids, start=1)
        ]
        evidence: list[Evidence] = []
        if not result:
            return retrieved, evidence
        for item in result.get("evidence", []):
            source_ids = [item_id for item_id in item.get("document_ids", []) if item_id in known]
            quotes = [str(value) for value in item.get("exact_quotes", []) if str(value)]
            grounded_sources = [
                source_id
                for source_id in source_ids
                if quotes and all(quote in known[source_id].text for quote in quotes)
            ]
            if not grounded_sources:
                continue
            evidence.append(
                Evidence(
                    document_id=grounded_sources[0],
                    claim=str(item.get("claim", "")),
                    matched_terms=(),
                    confidence=1.0,
                    effect_direction="unknown",
                    effect_window="forecast_contract",
                    entity=task.entity_name,
                    target_variable=task.target_name,
                    evidence_quote="\n".join(quotes),
                    provenance_valid=True,
                )
            )
        return retrieved, evidence

    def run(self, task: ForecastTask) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        baseline, baseline_method = self.forecast_agent.baseline(task, diagnosis)
        prompt = (
            "You are the evidence and hypothesis component of a contextual time-series "
            "forecasting system. Read task.json and autonomously search documents/. Do not "
            "produce future numerical values. Instead, return a falsifiable forecast_contract "
            "describing the regime that numerical tools should test. Use normal_seasonal only "
            "when grounded evidence says temporary anomalies have ended and the forecast should "
            "follow historical baseline/seasonality. Reject wrong entities, variables, dates, "
            "post-cutoff hindsight, and misleading time-series claims. Under this benchmark's "
            "corpus contract, an undated document is available at the history cutoff unless an "
            "explicit issue/publication date proves otherwise. Every accepted claim needs at "
            "least one exact verbatim quote and valid document_id. The numerical layer will "
            "independently backtest candidate trajectories using history only."
        )
        before = self.codex.stats()
        result = self.codex.complete(
            "codex_contract",
            prompt,
            CODEX_CONTRACT_SCHEMA,
            workspace_files=self._workspace(task, baseline),
        )
        after = self.codex.stats()
        retrieved, evidence = self._parse_research(task, result)
        contract = result.get("forecast_contract", {}) if result else {}

        workspace = self.workspace_executor.initialize(task, baseline, baseline_method)
        projection = None
        revision_record = None
        explicit_validation: ExplicitValueValidation | None = None
        explicit_revision_records = []
        if (
            contract.get("regime") == "normal_seasonal"
            and contract.get("seasonality") == "preserve"
            and float(contract.get("confidence", 0.0)) >= 0.6
            and evidence
        ):
            # The structured contract opens the gate. The synthetic claim only
            # names the contract semantics; provenance remains the exact quotes
            # and source IDs produced above.
            contract_evidence = [
                Evidence(
                    document_id=item.document_id,
                    claim=(
                        f"For {task.entity_name} {task.target_name}, grounded evidence says "
                        "the future should return to historical baseline and seasonality."
                    ),
                    matched_terms=(),
                    confidence=min(item.confidence, float(contract["confidence"])),
                    effect_direction="stable",
                    effect_window="forecast_contract",
                    entity=task.entity_name,
                    target_variable=task.target_name,
                    evidence_quote=item.evidence_quote,
                    provenance_valid=item.provenance_valid,
                )
                for item in evidence
            ]
            projection = self.regime_agent.project(
                task,
                diagnosis,
                workspace.baseline_values,
                contract_evidence,
            )
            if projection is not None:
                proposal = self.revision_planner.regime_override(
                    workspace,
                    projection.values,
                    projection.source_document_ids,
                    min(projection.confidence, float(contract["confidence"])),
                    projection.rationale,
                )
                revision_record = self.workspace_executor.apply(workspace, proposal)

        if (
            contract.get("regime") == "explicit_future_values"
            and float(contract.get("confidence", 0.0)) >= 0.6
            and evidence
        ):
            explicit_validation = self.explicit_value_validator.validate(
                task,
                diagnosis,
                workspace.baseline_values,
                retrieved,
                evidence,
            )
            timestamp_indices = {
                timestamp: index for index, timestamp in enumerate(task.future_timestamps)
            }
            for timestamp, value in explicit_validation.accepted_points.items():
                proposal = self.revision_planner.point_override(
                    workspace,
                    timestamp_indices[timestamp],
                    value,
                    self.config.explicit_value_weight,
                    explicit_validation.accepted_sources[timestamp],
                )
                record = self.workspace_executor.apply(workspace, proposal)
                explicit_revision_records.append(record)

        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        forecast = self.forecast_agent.forecast_from_mean(
            task=task,
            diagnosis=diagnosis,
            mean=tuple(workspace.final_values),
            baseline_mean=workspace.baseline_values,
            baseline_method=f"{workspace.baseline_method}+codex-contract",
            num_samples=self.config.num_samples,
            seed=task_seed,
            impact_adjustments=self.workspace_executor.adjustments(workspace),
            revision_records=workspace.revision_records,
        )
        metrics = retrieval_metrics(task, retrieved, evidence)
        metrics.update(forecast_metrics(task, forecast))
        metrics.update(
            {
                "codex_calls": float(after["calls"] - before["calls"]),
                "codex_cache_hits": float(after["cache_hits"] - before["cache_hits"]),
                "codex_failures": float(after["failures"] - before["failures"]),
                "codex_latency_seconds": float(after["latency_seconds"] - before["latency_seconds"]),
                "contract_confidence": float(contract.get("confidence", 0.0)),
                "contract_revision_applied": float(
                    (revision_record is not None and revision_record.accepted)
                    or any(record.accepted for record in explicit_revision_records)
                ),
            }
        )
        if projection is not None:
            metrics.update(
                {
                    "candidate_validation_mae": projection.validation_mae,
                    "candidate_seasonal_naive_mae": projection.seasonal_naive_mae,
                    "candidate_blend_weight": projection.blend_weight,
                }
            )
        if explicit_validation is not None:
            metrics.update(
                {
                    "explicit_points_considered": float(
                        len(explicit_validation.decisions)
                    ),
                    "explicit_points_accepted": float(
                        len(explicit_validation.accepted_points)
                    ),
                    "explicit_point_coverage": (
                        len(explicit_validation.accepted_points)
                        / task.prediction_length
                    ),
                }
            )
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=retrieved,
            evidence=evidence,
            forecast=forecast,
            metrics=metrics,
            loop_trace=[
                {
                    "system": "codex-contract",
                    "codex_model": self.config.codex_model or "cli-default",
                    "codex_reasoning_effort": self.config.codex_reasoning_effort,
                    "research_report": result.get("research_report", "") if result else "",
                    "forecast_contract": contract,
                    "cited_document_ids": [item.document.document_id for item in retrieved],
                    "candidate_projection": (
                        {
                            "validation_mae": projection.validation_mae,
                            "seasonal_naive_mae": projection.seasonal_naive_mae,
                            "blend_weight": projection.blend_weight,
                            "rationale": projection.rationale,
                        }
                        if projection is not None
                        else None
                    ),
                    "revision_record": (
                        {
                            "accepted": revision_record.accepted,
                            "reason": revision_record.reason,
                            "affected_steps": revision_record.affected_steps,
                            "mean_absolute_change": revision_record.mean_absolute_change,
                        }
                        if revision_record is not None
                        else None
                    ),
                    "explicit_value_validation": (
                        {
                            "decisions": [
                                {
                                    "timestamp": item.timestamp,
                                    "value": item.value,
                                    "source_document_ids": list(
                                        item.source_document_ids
                                    ),
                                    "accepted": item.accepted,
                                    "reason": item.reason,
                                    "baseline_value": item.baseline_value,
                                    "standardized_deviation": item.standardized_deviation,
                                }
                                for item in explicit_validation.decisions
                            ],
                            "accepted_points": explicit_validation.accepted_points,
                        }
                        if explicit_validation is not None
                        else None
                    ),
                    "codex": after,
                }
            ],
            workspace=workspace,
        )
