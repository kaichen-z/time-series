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
from .backbones import ChronosBackboneConfig, TimesFMBackboneConfig, build_forecast_backbone
from .metrics import forecast_metrics, retrieval_metrics
from .models import ForecastTask, RunResult


@dataclass(frozen=True)
class SystemConfig:
    top_k: int = 8
    num_samples: int = 100
    context_weight: float = 0.75
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

    def __post_init__(self) -> None:
        if self.backbone not in {"chronos", "timesfm", "statistical"}:
            raise ValueError("backbone must be 'chronos', 'timesfm', or 'statistical'")


class MinimalAgentSystem:
    """Diagnose → retrieve → synthesize evidence → probabilistic forecast."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self.diagnosis_agent = TimeSeriesDiagnosisAgent()
        self.retrieval_agent = RetrievalAgent()
        self.evidence_agent = EvidenceSynthesisAgent()
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


class NumericalBaselineSystem(MinimalAgentSystem):
    """Numbers-only backbone baseline with no document retrieval or text revision."""

    def run(self, task: ForecastTask) -> RunResult:
        diagnosis = self.diagnosis_agent.diagnose(task)
        task_seed = self.config.seed + sum(ord(character) for character in task.benchmark_id)
        forecast = self.forecast_agent.forecast(
            task=task,
            diagnosis=diagnosis,
            retrieved=[],
            num_samples=self.config.num_samples,
            seed=task_seed,
            context_weight=0.0,
        )
        metrics = forecast_metrics(task, forecast)
        return RunResult(
            benchmark_id=task.benchmark_id,
            diagnosis=diagnosis,
            retrieved=[],
            evidence=[],
            forecast=forecast,
            metrics=metrics or None,
        )


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
        "revision_outcomes": {
            "improved": sum(
                1
                for result in results
                if (result.metrics or {}).get("revision_value_mae", 0.0) > 1e-9
            ),
            "unchanged": sum(
                1
                for result in results
                if abs((result.metrics or {}).get("revision_value_mae", 0.0)) <= 1e-9
            ),
            "harmed": sum(
                1
                for result in results
                if (result.metrics or {}).get("revision_value_mae", 0.0) < -1e-9
            ),
        },
        "note": "sMAE/sRMSE/sCRPS values are transparent development proxies; hidden-test official scores are computed by Dr-CiK maintainers.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
