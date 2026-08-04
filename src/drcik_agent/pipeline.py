from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .agents import (
    EvidenceSynthesisAgent,
    ProbabilisticForecastAgent,
    RetrievalAgent,
    TimeSeriesDiagnosisAgent,
)
from .metrics import forecast_metrics, retrieval_metrics
from .models import ForecastTask, RunResult


@dataclass(frozen=True)
class SystemConfig:
    top_k: int = 8
    num_samples: int = 100
    context_weight: float = 0.75
    seed: int = 7


class MinimalAgentSystem:
    """Diagnose → retrieve → synthesize evidence → probabilistic forecast."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self.diagnosis_agent = TimeSeriesDiagnosisAgent()
        self.retrieval_agent = RetrievalAgent()
        self.evidence_agent = EvidenceSynthesisAgent()
        self.forecast_agent = ProbabilisticForecastAgent()

    def run(self, task: ForecastTask) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        retrieved = self.retrieval_agent.retrieve(task, diagnosis, self.config.top_k)
        evidence = self.evidence_agent.synthesize(task, diagnosis, retrieved)
        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        forecast = self.forecast_agent.forecast(
            task=task,
            diagnosis=diagnosis,
            retrieved=retrieved,
            num_samples=self.config.num_samples,
            seed=task_seed,
            context_weight=self.config.context_weight,
        )
        metrics = retrieval_metrics(task, retrieved, evidence)
        metrics.update(forecast_metrics(task, forecast))
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=retrieved,
            evidence=evidence,
            forecast=forecast,
            metrics=metrics or None,
        )

    def run_many(self, tasks: Iterable[ForecastTask]) -> list[RunResult]:
        return [self.run(task) for task in tasks]


def write_outputs(results: list[RunResult], output_dir: str | Path) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "forecasts.jsonl": [result.forecast_submission() for result in results],
        "deep_research.jsonl": [result.research_submission() for result in results],
        "run_report.jsonl": [result.report_dict() for result in results],
    }
    if any(result.loop_trace for result in results):
        files["loop_trace.jsonl"] = [result.trace_submission() for result in results]
    for filename, rows in files.items():
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    numeric_metrics: dict[str, list[float]] = {}
    for result in results:
        for name, value in (result.metrics or {}).items():
            numeric_metrics.setdefault(name, []).append(value)
    summary = {
        "num_tasks": len(results),
        "mean_metrics": {
            name: statistics.fmean(values) for name, values in sorted(numeric_metrics.items())
        },
        "note": "sMAE/sRMSE/sCRPS values are transparent development proxies; hidden-test official scores are computed by Dr-CiK maintainers.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
