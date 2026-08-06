"""End-to-end pipeline run against real sample tasks, and CLI argument parsing."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch

from dr_cik.backbone import ChronosBackboneConfig, ChronosForecastBackbone
from dr_cik.cli import build_parser
from dr_cik.llm import FakeLLMClient
from dr_cik.pipeline import RunConfig, build_pipeline, write_outputs

from .conftest import requires_sample


class _FT(Enum):
    SAMPLES = "s"
    QUANTILES = "q"


class _FakePipeline:
    forecast_type = _FT.QUANTILES

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        count = len(quantile_levels)
        quantiles = torch.randn(1, prediction_length, count) * 5 + 400.0
        return quantiles, quantiles[:, :, count // 2]


def _fake_backbone(num_samples: int) -> ChronosForecastBackbone:
    fake_module = SimpleNamespace(BaseChronosPipeline=_FakePipeline, ForecastType=_FT)
    return ChronosForecastBackbone(ChronosBackboneConfig(num_samples=num_samples), runtime_module=fake_module)


def _drbench_responder(system: str, messages: list[dict[str, str]]) -> str:
    content = messages[0]["content"]
    if content.startswith("Document"):
        document_id = content.split("Document ")[1].split(":")[0]
        return json.dumps({"document_id": document_id, "relevant": True, "brief": "b", "key_claims": []})
    return json.dumps({"report": "a report", "evidence": [{"claim": "a claim", "source_doc_ids": []}]})


@requires_sample
def test_pipeline_produces_submission_shaped_files(sample_tasks, tmp_path: Path) -> None:
    config = RunConfig(agent="drbench", num_samples=12, crps_sample_size=6, drbench_top_k=2, judge_enabled=False)
    llm = FakeLLMClient(responses=_drbench_responder)
    pipeline = build_pipeline(config, llm=llm, judge=None, backbone=_fake_backbone(12))
    results = pipeline.run_many(sample_tasks)

    write_outputs(results, tmp_path)

    forecasts = [json.loads(line) for line in (tmp_path / "forecasts.jsonl").read_text().splitlines()]
    deep_research = [json.loads(line) for line in (tmp_path / "deep_research.jsonl").read_text().splitlines()]
    run_report = [json.loads(line) for line in (tmp_path / "run_report.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(forecasts) == len(sample_tasks) == len(deep_research) == len(run_report)
    for row, task in zip(forecasts, sample_tasks):
        assert row["benchmark_id"] == task.benchmark_id
        assert len(row["samples"]) == 12
        assert all(len(sample) == task.prediction_length for sample in row["samples"])
    for row in deep_research:
        assert set(row.keys()) == {"benchmark_id", "cited_document_ids", "evidence"}
    assert summary["num_tasks"] == len(sample_tasks)
    assert summary["agent"] == "drbench"
    assert "smae" in summary["mean_metrics"]
    assert "evidence_recall" not in summary["mean_metrics"]  # judge_enabled=False -> always None -> excluded


def test_cli_parses_run_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--agent", "opendr", "--sample-dir", "/tmp/sample", "--output-dir", "/tmp/out", "--num-samples", "50"]
    )
    assert args.command == "run"
    assert args.agent == "opendr"
    assert args.num_samples == 50


def test_cli_requires_exactly_one_data_source() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["run", "--agent", "drbench", "--output-dir", "/tmp/out"])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_cli_download_data_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["download-data", "--local-dir", "/tmp/data"])
    assert args.command == "download-data"
    assert args.local_dir == "/tmp/data"
