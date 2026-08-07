"""Orchestrates agent + forecaster over Dr-CiK tasks, and writes submission-shaped outputs."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .agents.drbench import DRBenchAgent, DRBenchConfig
from .agents.opendr import OpenDRAgent, OpenDRConfig
from .evaluation import cited_document_ids, development_metrics, scrps, smae, srmse
from .forecasters.chronos import ChronosConfig, ChronosForecaster, DEFAULT_CACHE_DIR
from .forecasters.direct_prompt import DirectPromptForecaster
from .llm import GeminiClient, LLMClient
from .local_llm import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
from .local_llm import QwenClient, QwenConfig
from .models import AgentResult, DeepResearchAgent, Forecast, ForecastTask, RunResult


@dataclass(frozen=True)
class RunConfig:
    """All knobs for one dr-cik run."""

    agent: str
    llm_backend: str = "gemini"  # "gemini" | "qwen" (local, for when Gemini is rate-limited)
    gemini_model_id: str = "gemini-3-flash-preview"
    judge_model_id: str = "gemini-3-flash-preview"
    judge_enabled: bool = True
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID
    qwen_device: str | None = None
    chronos_model_id: str = "amazon/chronos-bolt-base"
    chronos_device_map: str = "cpu"
    chronos_cache_dir: str | None = DEFAULT_CACHE_DIR
    num_samples: int = 100
    crps_sample_size: int = 25
    max_react_steps: int = 6
    drbench_top_k: int = 8
    seed: int = 7

    def __post_init__(self) -> None:
        if self.agent not in ("opendr", "drbench"):
            raise ValueError(f"Unknown agent {self.agent!r}, expected 'opendr' or 'drbench'")
        if self.llm_backend not in ("gemini", "qwen"):
            raise ValueError(f"Unknown llm_backend {self.llm_backend!r}, expected 'gemini' or 'qwen'")


def _mean_metrics(all_metrics: Iterable[dict[str, float | None]]) -> dict[str, float]:
    """Average each metric over the tasks where it was computed, skipping None (hidden-test/no-judge)."""
    collected: dict[str, list[float]] = {}
    for metrics in all_metrics:
        for name, value in metrics.items():
            if value is not None:
                collected.setdefault(name, []).append(value)
    return {name: statistics.fmean(values) for name, values in sorted(collected.items())}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _retrieved_document_ids(agent_result: AgentResult) -> set[str]:
    """Every document_id any search/brief step actually touched, for the *_retrieved diagnostic."""
    ids: set[str] = set()
    for step in agent_result.steps:
        if step.kind == "search":
            ids.update(step.payload.get("document_ids", []))
        elif step.kind == "brief":
            document_id = step.payload.get("document_id")
            if document_id:
                ids.add(str(document_id))
    return ids


class DrCikPipeline:
    """Runs one deep-research agent and one forecaster over a set of tasks."""

    def __init__(self, config: RunConfig, agent: DeepResearchAgent, judge: LLMClient | None, forecaster: ChronosForecaster) -> None:
        self.config = config
        self.agent = agent
        self.judge = judge
        self.forecaster = forecaster

    def run(self, task: ForecastTask) -> RunResult:
        view = task.agent_view()
        agent_result = self.agent.run(view)
        forecast = self.forecaster.forecast(view, num_samples=self.config.num_samples)
        used_doc_ids = _retrieved_document_ids(agent_result)
        metrics = development_metrics(
            task, forecast, agent_result.report.evidence, used_doc_ids, self.judge, self.config.crps_sample_size
        )
        return RunResult(
            benchmark_id=task.benchmark_id,
            agent_name=self.config.agent,
            agent_result=agent_result,
            forecast=forecast,
            metrics=metrics,
        )

    def run_many(self, tasks: Iterable[ForecastTask]) -> list[RunResult]:
        return [self.run(task) for task in tasks]


def build_pipeline(
    config: RunConfig,
    *,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    forecaster: ChronosForecaster | None = None,
) -> DrCikPipeline:
    """Wire real Gemini/Qwen + Chronos by default, or accept fakes injected for tests."""
    if llm is not None:
        resolved_llm: LLMClient = llm
    elif config.llm_backend == "qwen":
        resolved_llm = QwenClient(QwenConfig(model_id=config.qwen_model_id, device=config.qwen_device, seed=config.seed))
    else:
        resolved_llm = GeminiClient(model_id=config.gemini_model_id)

    if config.agent == "opendr":
        agent: DeepResearchAgent = OpenDRAgent(resolved_llm, OpenDRConfig(max_steps=config.max_react_steps))
    else:
        agent = DRBenchAgent(resolved_llm, DRBenchConfig(top_k_search=config.drbench_top_k))

    if judge is not None:
        resolved_judge: LLMClient | None = judge
    elif not config.judge_enabled:
        resolved_judge = None
    elif config.llm_backend == "qwen":
        resolved_judge = resolved_llm  # reuse the same loaded weights instead of doubling GPU memory
    else:
        resolved_judge = GeminiClient(model_id=config.judge_model_id)
    resolved_forecaster = forecaster or ChronosForecaster(
        ChronosConfig(
            model_id=config.chronos_model_id,
            device_map=config.chronos_device_map,
            cache_dir=config.chronos_cache_dir,
            num_samples=config.num_samples,
        )
    )
    return DrCikPipeline(config, agent, resolved_judge, resolved_forecaster)


def write_outputs(results: list[RunResult], output_dir: str | Path) -> None:
    """Write forecasts.jsonl, deep_research.jsonl, run_report.jsonl, and summary.json."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    forecasts_rows = [{"benchmark_id": result.benchmark_id, "samples": list(result.forecast.samples)} for result in results]
    deep_research_rows = [
        {
            "benchmark_id": result.benchmark_id,
            "cited_document_ids": sorted(cited_document_ids(result.agent_result.report.evidence)),
            "evidence": [item.claim for item in result.agent_result.report.evidence],
        }
        for result in results
    ]
    run_report_rows = [
        {
            "benchmark_id": result.benchmark_id,
            "agent": result.agent_name,
            "forecast": {"mean": list(result.forecast.mean), "method": result.forecast.method, "num_samples": len(result.forecast.samples)},
            "report_markdown": result.agent_result.report.report_markdown,
            "evidence": [asdict(item) for item in result.agent_result.report.evidence],
            "agent_trace": [asdict(step) for step in result.agent_result.steps],
            "metrics": result.metrics,
            "stop_reason": result.agent_result.stop_reason,
            "llm_call_count": result.agent_result.llm_call_count,
        }
        for result in results
    ]

    _write_jsonl(output / "forecasts.jsonl", forecasts_rows)
    _write_jsonl(output / "deep_research.jsonl", deep_research_rows)
    _write_jsonl(output / "run_report.jsonl", run_report_rows)
    _write_json(
        output / "summary.json",
        {
            "num_tasks": len(results),
            "agent": results[0].agent_name if results else None,
            "mean_metrics": _mean_metrics(result.metrics for result in results),
            "note": (
                "smae/srmse/scrps are local development proxies (S=25 per the paper's formula, not the "
                "S>=100 in forecasts.jsonl); evidence_recall is our own LLM-judge approximation, not "
                "Dr-CiK's private official scorer."
            ),
        },
    )


@dataclass(frozen=True)
class DirectPromptRunResult:
    """One task's Direct-Prompt forecast plus its forecast-only metrics."""

    benchmark_id: str
    forecast: Forecast
    metrics: dict[str, float | None]


def run_direct_prompt(
    tasks: Iterable[ForecastTask],
    forecaster: DirectPromptForecaster,
    context_by_id: dict[str, str],
    crps_sample_size: int = 25,
) -> list[DirectPromptRunResult]:
    """Forecast every task via the Direct-Prompt LLM baseline; score smae/srmse/scrps only (no new evidence to grade)."""
    results: list[DirectPromptRunResult] = []
    for task in tasks:
        view = task.agent_view()
        context_text = context_by_id.get(task.benchmark_id, "")
        forecast = forecaster.forecast(view, context_text)
        if task.future_values is not None:
            crps_samples = forecast.samples[:crps_sample_size] or forecast.samples
            metrics: dict[str, float | None] = {
                "smae": smae(task.future_values, forecast.samples),
                "srmse": srmse(task.future_values, forecast.samples),
                "scrps": scrps(task.future_values, crps_samples),
            }
        else:
            metrics = {"smae": None, "srmse": None, "scrps": None}
        results.append(DirectPromptRunResult(benchmark_id=task.benchmark_id, forecast=forecast, metrics=metrics))
    return results


def write_direct_prompt_outputs(results: list[DirectPromptRunResult], output_dir: str | Path, model_id: str, from_run_dir: str) -> None:
    """Write forecasts.jsonl (SUBMISSION.md-exact) and summary.json for a Direct-Prompt run."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        output / "forecasts.jsonl",
        [{"benchmark_id": result.benchmark_id, "samples": list(result.forecast.samples)} for result in results],
    )
    _write_json(
        output / "summary.json",
        {
            "num_tasks": len(results),
            "model_id": model_id,
            "from_run_dir": from_run_dir,
            "mean_metrics": _mean_metrics(result.metrics for result in results),
            "methods": sorted({result.forecast.method for result in results}),
            "note": (
                "Direct-Prompt baseline: the LLM forecasts numbers directly from history + the "
                "DR-synthesized context loaded from from_run_dir, no numeric foundation model involved. "
                "Check `methods` for any :degraded-fallback or :padded runs, where the model's own "
                "output was incomplete and was substituted or resampled."
            ),
        },
    )
