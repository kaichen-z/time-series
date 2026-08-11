from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import (
    AgentBeliefState,
    Diagnosis,
    Evidence,
    EvidenceImpact,
    ForecastTask,
    QueryAction,
    RetrievedDocument,
)


QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "rationale"],
    "properties": {
        "query": {"type": "string", "minLength": 3, "maxLength": 1200},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "document_id",
                    "accepted",
                    "confidence",
                    "reason",
                    "event_types",
                    "evidence_quotes",
                ],
                "properties": {
                    "document_id": {"type": "string"},
                    "accepted": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "maxLength": 1000},
                    "event_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "anomaly",
                                "resolution",
                                "temporary_event",
                                "external_driver",
                                "forecast_regime",
                            ],
                        },
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 1200},
                    },
                },
            },
        }
    },
}

IMPACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["impacts"],
    "properties": {
        "impacts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_document_ids",
                    "event_type",
                    "start_timestamp",
                    "end_timestamp",
                    "direction",
                    "permanence",
                    "forecast_relation",
                    "adjustment_kind",
                    "adjustment_value",
                    "confidence",
                    "rationale",
                ],
                "properties": {
                    "source_document_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "anomaly",
                            "resolution",
                            "promotion",
                            "external_driver",
                            "forecast_regime",
                            "general",
                        ],
                    },
                    "start_timestamp": {"type": ["string", "null"]},
                    "end_timestamp": {"type": ["string", "null"]},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "stable", "unclear"],
                    },
                    "permanence": {
                        "type": "string",
                        "enum": ["resolved", "temporary", "permanent", "unspecified"],
                    },
                    "forecast_relation": {
                        "type": "string",
                        "enum": [
                            "ended_before_forecast",
                            "after_forecast",
                            "embedded_in_history",
                            "overlaps_forecast",
                            "forecast_relevant_undated",
                            "historical_or_uncertain",
                        ],
                    },
                    "adjustment_kind": {
                        "type": "string",
                        "enum": [
                            "multiplier",
                            "percentage",
                            "absolute_additive",
                            "standardized_additive",
                            "return_to_baseline",
                            "outside_horizon",
                            "already_in_baseline",
                            "qualitative_only",
                        ],
                    },
                    "adjustment_value": {"type": ["number", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 1500},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class CodexCLIConfig:
    binary: str = "codex"
    model: str | None = None
    cache_dir: str = "outputs/codex-cache"
    timeout_seconds: int = 180
    max_document_characters: int = 12000
    reasoning_effort: str = "low"


class CodexCLIClient:
    """Small, schema-constrained adapter around ``codex exec``.

    The model receives task data through stdin, runs in a read-only ephemeral
    sandbox, and cannot directly change the forecasting workspace. Failures
    return ``None`` so callers can fall back to deterministic agents.
    """

    def __init__(self, config: CodexCLIConfig | None = None) -> None:
        self.config = config or CodexCLIConfig()
        self.cache_dir = Path(self.config.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.total_latency_seconds = 0.0
        self.last_error: str | None = None

    def stats(self) -> dict[str, float | int | str | None]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "latency_seconds": round(self.total_latency_seconds, 3),
            "last_error": self.last_error,
        }

    def _cache_path(
        self,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        workspace_files: dict[str, str] | None = None,
    ) -> Path:
        material = json.dumps(
            {
                "stage": stage,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "prompt": prompt,
                "schema": schema,
                "workspace_files": workspace_files or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return self.cache_dir / f"{stage}-{hashlib.sha256(material).hexdigest()}.json"

    def complete(
        self,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        workspace_files: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        cache_path = self._cache_path(stage, prompt, schema, workspace_files)
        if cache_path.exists():
            try:
                self.cache_hits += 1
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        self.calls += 1
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="drcik-codex-") as directory:
                temporary = Path(directory)
                schema_path = temporary / "schema.json"
                output_path = temporary / "result.json"
                for filename, contents in (workspace_files or {}).items():
                    relative = Path(filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"unsafe Codex workspace path: {filename}")
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(contents, encoding="utf-8")
                schema_path.write_text(
                    json.dumps(schema, ensure_ascii=False), encoding="utf-8"
                )
                command = [
                    self.config.binary,
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--cd",
                    directory,
                    "-c",
                    f'model_reasoning_effort="{self.config.reasoning_effort}"',
                ]
                if self.config.model:
                    command.extend(("--model", self.config.model))
                command.append("-")
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0 or not output_path.exists():
                    detail = completed.stderr.strip().splitlines()
                    raise RuntimeError(detail[-1][:500] if detail else "codex exec failed")
                result = json.loads(output_path.read_text(encoding="utf-8"))
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.last_error = None
                return result
        except (
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            self.failures += 1
            self.last_error = f"{type(error).__name__}: {error}"[:700]
            return None
        finally:
            self.total_latency_seconds += time.monotonic() - started


def _task_payload(task: ForecastTask, diagnosis: Diagnosis) -> dict[str, Any]:
    return {
        "benchmark_id": task.benchmark_id,
        "entity": task.entity_name,
        "target_variable": task.target_name,
        "target_description": task.target_description,
        "frequency": task.frequency,
        "history_window": [task.history_timestamps[0], task.history_timestamps[-1]],
        "forecast_window": [task.future_timestamps[0], task.future_timestamps[-1]],
        "diagnosis": {
            "trend": diagnosis.trend,
            "seasonal_period": diagnosis.seasonal_period,
            "seasonal_strength": diagnosis.seasonal_strength,
        },
    }


class CodexQueryPlannerAgent:
    def __init__(self, client: CodexCLIClient) -> None:
        self.client = client

    def refine(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        state: AgentBeliefState,
        action: QueryAction,
    ) -> QueryAction:
        payload = {
            "task": _task_payload(task, diagnosis),
            "information_gap": asdict(state.forecast_gaps[action.question_id]),
            "current_query": action.query,
            "previous_queries": [item.query for item in state.query_history[-4:]],
            "rejection_reasons": state.rejected_reasons,
        }
        prompt = (
            "You are the query-planning component of a contextual time-series forecasting "
            "agent. Return one concise corpus-search query that maximizes the chance of finding "
            "entity-specific, target-specific, temporally relevant causal evidence for the stated "
            "gap. Include exact entity and target names and useful event/date terms. Prospective "
            "plans, schedules, and forecasts published by the cutoff are eligible context, so do "
            "not exclude them merely because they are projections. Do not answer the forecasting "
            "question. Do not use tools or inspect files.\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = self.client.complete("query", prompt, QUERY_SCHEMA)
        if not result or not str(result.get("query", "")).strip():
            return action
        return QueryAction(
            question_id=action.question_id,
            question=action.question,
            query=str(result["query"]).strip(),
            rationale=(
                action.rationale
                + " Codex query refinement: "
                + str(result.get("rationale", "")).strip()
            )[:1800],
        )


class CodexEvidenceJudgeAgent:
    def __init__(self, client: CodexCLIClient) -> None:
        self.client = client

    def judge(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        action: QueryAction,
        retrieved: list[RetrievedDocument],
    ) -> dict[str, dict[str, Any]] | None:
        documents = [
            {
                "document_id": item.document.document_id,
                "text": item.document.text[: self.client.config.max_document_characters],
            }
            for item in retrieved
        ]
        payload = {
            "task": _task_payload(task, diagnosis),
            "information_gap": {
                "id": action.question_id,
                "question": action.question,
                "query": action.query,
            },
            "documents": documents,
        }
        prompt = (
            "You are a strict evidence verifier for time-series forecasting. For every supplied "
            "document, decide whether it directly helps answer the current information gap for "
            "the exact entity, target variable, and relevant time window. Reject distractors, "
            "wrong entities, wrong dates, post-cutoff hindsight, unattributed claims, and claims "
            "about another variable. A prospective plan, schedule, analytic forecast, or expected "
            "future regime stated in a document available by the history cutoff is valid contextual "
            "evidence; identify it as prospective in the reason and calibrate confidence instead "
            "of rejecting it as circular. Under this benchmark's corpus contract, an undated "
            "document is assumed available at the cutoff: infer post-cutoff hindsight only from an "
            "explicit publication, issue, or report date after the cutoff, not from narrative tense "
            "or the event timestamps alone. Evidence quotes must be copied exactly from that document "
            "and should be self-contained enough to name the target or entity when possible. Return "
            "one decision per document. Do not use tools or inspect files.\n\n"
            "If a relevant document contains explicit timestamp-value forecasts within the "
            "requested horizon, include the exact table rows or value lines in evidence_quotes; "
            "these quotes are pinned through context compression.\n\n"
            "INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)
        )
        result = self.client.complete("verify", prompt, VERIFICATION_SCHEMA)
        if not result:
            return None
        known = {item.document.document_id for item in retrieved}
        decisions = {
            str(item.get("document_id")): item
            for item in result.get("decisions", [])
            if str(item.get("document_id")) in known
        }
        return decisions if decisions else None


class CodexEvidenceToForecastAgent:
    def __init__(self, client: CodexCLIClient, fallback: Any) -> None:
        self.client = client
        self.fallback = fallback

    def translate(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        retrieved: list[RetrievedDocument],
        evidence: list[Evidence],
    ) -> list[EvidenceImpact]:
        fallback_impacts = self.fallback.translate(task, diagnosis, retrieved, evidence)
        if not retrieved or not evidence:
            return fallback_impacts
        payload = {
            "task": _task_payload(task, diagnosis),
            "accepted_evidence": [asdict(item) for item in evidence],
            "documents": [
                {
                    "document_id": item.document.document_id,
                    "text": item.document.text[: self.client.config.max_document_characters],
                }
                for item in retrieved
            ],
        }
        prompt = (
            "You translate verified textual evidence into auditable effects on a future time-series "
            "forecast. Use only the supplied evidence. Never invent dates or magnitudes. If an "
            "event ended before the horizon, choose return_to_baseline. If a permanent change is "
            "already present in observed history, choose already_in_baseline. Use multiplier, "
            "percentage, or absolute_additive only when the document explicitly quantifies the "
            "effect on the target variable. Otherwise use qualitative_only or, for an explicitly "
            "directional overlapping event, standardized_additive. Do not use tools or inspect "
            "files.\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)
        )
        result = self.client.complete("impact", prompt, IMPACT_SCHEMA)
        if not result:
            return fallback_impacts
        known_sources = {item.document.document_id for item in retrieved}
        impacts: list[EvidenceImpact] = []
        for raw in result.get("impacts", []):
            sources = tuple(
                dict.fromkeys(
                    source
                    for source in raw.get("source_document_ids", [])
                    if source in known_sources
                )
            )
            if not sources:
                continue
            try:
                impacts.append(
                    EvidenceImpact(
                        source_document_ids=sources,
                        event_type=str(raw["event_type"]),
                        start_timestamp=raw.get("start_timestamp"),
                        end_timestamp=raw.get("end_timestamp"),
                        direction=str(raw["direction"]),
                        permanence=str(raw["permanence"]),
                        forecast_relation=str(raw["forecast_relation"]),
                        adjustment_kind=str(raw["adjustment_kind"]),
                        adjustment_value=(
                            float(raw["adjustment_value"])
                            if raw.get("adjustment_value") is not None
                            else None
                        ),
                        confidence=float(raw["confidence"]),
                        rationale="Codex evidence-to-impact: " + str(raw["rationale"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return impacts or fallback_impacts
