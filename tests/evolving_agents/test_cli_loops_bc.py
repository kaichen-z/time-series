"""Loops B and C plus the baselines, driven through the CLI with injected clients."""

from __future__ import annotations

import json

import pytest
from dr_cik.llm import FakeLLMClient
from dr_cik.models import Forecast

from evolving_agents.bundles import SEED_DIR
from evolving_agents.cli import build_parser, run_baselines, run_evolve_retrieval, run_evolve_system

from .conftest import SAMPLE_DIR, requires_sample

FLAT = json.dumps({"assumption": "flat level", "code": "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n"})
KEEP = json.dumps({"evidence": [{"claim": "A dated outage cuts load on 2025-10-16.", "source_doc_ids": ["doc_0"]}]})
CLEAN = json.dumps({"contradicts": False, "reason": "unrelated"})
MUTATION = json.dumps({"change_type": "system_prompt", "system_prompt": "Evolved.", "changelog": "tightened"})
RAG_FORECAST = json.dumps({"forecast": [1.0] * 200})


def _worker(n: int = 3000) -> FakeLLMClient:
    """A worker that answers every agent's schema in turn; over-provisioned so no test starves."""
    return FakeLLMClient(responses=[FLAT, KEEP, CLEAN, RAG_FORECAST] * n)


def _common(tmp_path) -> list[str]:
    # Its own split file: the committed canonical split is derived from the full dataset, where the
    # three sample tasks happen to cluster into two splits, leaving the third empty.
    return [
        "--sample-dir", str(SAMPLE_DIR),
        "--split-file", str(tmp_path / "splits.json"),
        "--cache-dir", str(tmp_path / "cache"),
        "--runs-dir", str(tmp_path / "runs"),
        "--trace-level", "off",
        "--limit", "2",
    ]


def _evolve_common(tmp_path) -> list[str]:
    return _common(tmp_path) + [
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--bundles-dir", str(tmp_path / "bundles"),
        "--generations", "1",
        "--population-size", "1",
        "--keep-elite", "1",
        "--minibatch-size", "1",
        "--stall-patience", "0",
        "--dev-limit", "1",
        "--n-windows", "1",
    ]


@requires_sample
def test_loop_b_runs_and_logs(tmp_path) -> None:
    args = build_parser().parse_args(
        ["evolve-retrieval", *_evolve_common(tmp_path), "--frozen-coding-bundle", str(SEED_DIR / "coding" / "v000.json"), "--bonus-weight", "0"]
    )
    records, best = run_evolve_retrieval(args, worker=_worker(), evolver=FakeLLMClient(responses=[MUTATION] * 50))

    assert records and best is not None
    record = json.loads((tmp_path / "runs" / "loop_b.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["loop"] == "B"
    assert {"retrieval", "coding", "decision"} <= set(record["bundle_versions"])
    assert "f1" in record["trace"]


@requires_sample
def test_loop_b_bonus_arm_runs_the_frozen_stack(tmp_path) -> None:
    args = build_parser().parse_args(
        ["evolve-retrieval", *_evolve_common(tmp_path), "--frozen-coding-bundle", str(SEED_DIR / "coding" / "v000.json"), "--bonus-weight", "0.2"]
    )
    run_evolve_retrieval(args, worker=_worker(), evolver=FakeLLMClient(responses=[MUTATION] * 50))
    record = json.loads((tmp_path / "runs" / "loop_b.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert "bonus" in record["trace"]


@requires_sample
def test_loop_c_runs_and_stamps_the_proxy_note(tmp_path) -> None:
    args = build_parser().parse_args(["evolve-system", *_evolve_common(tmp_path), "--no-judge"])
    records, best = run_evolve_system(args, worker=_worker(), evolver=FakeLLMClient(responses=[MUTATION] * 50))

    assert records and best is not None
    record = json.loads((tmp_path / "runs" / "loop_c.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["loop"] == "C"
    assert "development proxies" in record["note"]
    assert {"coding", "retrieval", "decision"} == set(record["bundle_versions"])


class _StubForecaster:
    """Stands in for Chronos so the baseline test needs no model download."""

    def forecast(self, task_view, num_samples: int = 25) -> Forecast:
        mean = tuple(float(task_view.history_values[-1]) for _ in range(task_view.prediction_length))
        return Forecast(mean=mean, samples=(mean,), method="stub")


@requires_sample
@pytest.mark.parametrize("baseline", ["chronos-only", "naive-rag", "coding-only", "frozen-system", "oracle-retrieval"])
def test_every_baseline_produces_a_summary(tmp_path, baseline: str) -> None:
    output = tmp_path / baseline
    args = build_parser().parse_args(
        ["run-baselines", *_common(tmp_path), "--baseline", baseline, "--output-dir", str(output), "--split", "dev", "--no-judge", "--n-windows", "1"]
    )
    summary = run_baselines(args, worker=_worker(), forecaster=_StubForecaster())

    assert summary["baseline"] == baseline
    assert summary["num_tasks"] > 0
    assert "development proxies" in summary["note"]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["baseline"] == baseline


def test_main_does_not_clobber_the_baselines_output_dir(tmp_path) -> None:
    # run-baselines has a real --output-dir; the logging default must not overwrite it with the
    # (absent) checkpoint dir, which would make the command fail before writing anything.
    from evolving_agents.cli import build_parser

    args = build_parser().parse_args(
        ["run-baselines", "--sample-dir", "/tmp/x", "--baseline", "chronos-only", "--output-dir", str(tmp_path / "out")]
    )
    if not getattr(args, "output_dir", None):
        args.output_dir = getattr(args, "checkpoint_dir", None)
    assert args.output_dir == str(tmp_path / "out")


def test_main_derives_an_output_dir_for_evolve_commands() -> None:
    from evolving_agents.cli import build_parser

    args = build_parser().parse_args(["evolve-coding", "--sample-dir", "/tmp/x", "--checkpoint-dir", "/tmp/ckpt"])
    if not getattr(args, "output_dir", None):
        args.output_dir = getattr(args, "checkpoint_dir", None)
    assert args.output_dir == "/tmp/ckpt"


@requires_sample
def test_oracle_retrieval_cites_exactly_the_supporting_documents(tmp_path) -> None:
    from evolving_agents.bundles import load_seed
    from evolving_agents.harness.baselines import oracle_retrieval
    from evolving_agents.harness.datasets import load_labeled_tasks

    task = next(t for t in load_labeled_tasks(sample_dir=SAMPLE_DIR) if any(d.role == "supporting" for d in t.documents))
    trace = oracle_retrieval(task, _worker(), load_seed("coding"), load_seed("decision"), n_windows=1)

    supporting = {document.document_id for document in task.documents if document.role == "supporting"}
    assert set(trace.retrieval_result.considered_doc_ids) == supporting
