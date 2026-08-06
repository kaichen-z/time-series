"""Orchestrates an agent + Chronos backbone over Dr-CiK tasks, and writes submission-shaped outputs."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .agents.drbench import DRBenchAgent, DRBenchConfig
from .agents.opendr import OpenDRAgent, OpenDRConfig
from .backbone import ChronosBackboneConfig, ChronosForecastBackbone, DEFAULT_CACHE_DIR
from .evaluation import cited_document_ids, development_metrics
from .llm import GeminiClient, LLMClient
from .local_llm import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
from .local_llm import QwenClient, QwenConfig
from .models import AgentResult, DeepResearchAgent, ForecastTask, RunResult


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
    """Runs one deep-research agent and one forecast backbone over a set of tasks."""

    def __init__(self, config: RunConfig, 
                 agent: DeepResearchAgent, 
                 judge: LLMClient | None, 
                 backbone: ChronosForecastBackbone) -> None:
        self.config = config
        self.agent = agent
        self.judge = judge
        self.backbone = backbone

    def run(self, task: ForecastTask) -> RunResult:
        view = task.agent_view()
        agent_result = self.agent.run(view)
        forecast = self.backbone.forecast(view, num_samples=self.config.num_samples)
        used_doc_ids = _retrieved_document_ids(agent_result)
        metrics = development_metrics(
            task, forecast, agent_result.report.evidence, used_doc_ids, self.judge, self.config.crps_sample_size
        )
        return RunResult(benchmark_id=task.benchmark_id, 
                         agent_name=self.config.agent, 
                         agent_result=agent_result, 
                         forecast=forecast, 
                         metrics=metrics)

    def run_many(self, tasks: Iterable[ForecastTask]) -> list[RunResult]:
        return [self.run(task) for task in tasks]


def build_pipeline(
    config: RunConfig,
    *,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    backbone: ChronosForecastBackbone | None = None,
) -> DrCikPipeline:
    """Wire real Gemini/Qwen + Chronos by default, or accept fakes injected for tests."""
    if llm is not None:
        resolved_llm: LLMClient = llm
    elif config.llm_backend == "qwen":
        resolved_llm = QwenClient(QwenConfig(model_id=config.qwen_model_id, device=config.qwen_device))
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
    resolved_backbone = backbone or ChronosForecastBackbone(
        ChronosBackboneConfig(
            model_id=config.chronos_model_id,
            device_map=config.chronos_device_map,
            cache_dir=config.chronos_cache_dir,
            num_samples=config.num_samples,
        )
    )
    return DrCikPipeline(config, agent, resolved_judge, resolved_backbone)


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

    files = {
        "forecasts.jsonl": forecasts_rows,
        "deep_research.jsonl": deep_research_rows,
        "run_report.jsonl": run_report_rows,
    }
    for filename, rows in files.items():
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    numeric_metrics: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.metrics.items():
            if value is not None:
                numeric_metrics.setdefault(name, []).append(value)
    summary = {
        "num_tasks": len(results),
        "agent": results[0].agent_name if results else None,
        "mean_metrics": {name: statistics.fmean(values) for name, values in sorted(numeric_metrics.items())},
        "note": (
            "smae/srmse/scrps are local development proxies (S=25 per the paper's formula, not the "
            "S>=100 in forecasts.jsonl); evidence_recall is our own Gemini-judge approximation, not "
            "Dr-CiK's private official scorer."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
