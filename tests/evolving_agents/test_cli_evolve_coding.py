"""Loop A end to end through the CLI path, with injected clients so no GPU is touched."""

from __future__ import annotations

import json

import pytest
from dr_cik.llm import FakeLLMClient

from evolving_agents.cli import build_parser, run_evolve_coding

from .conftest import SAMPLE_DIR, requires_sample

SEASONAL = json.dumps(
    {
        "assumption": "a repeating cycle dominates",
        "code": "def forecast(history, horizon, frequency):\n    period = min(24, len(history))\n    tail = history[-period:]\n    return [tail[i % period] for i in range(horizon)]\n",
    }
)
FLAT = json.dumps({"assumption": "the level is flat", "code": "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n"})
MUTATION = json.dumps({"change_type": "system_prompt", "system_prompt": "Evolved: prefer seasonal structure.", "changelog": "favor seasonality"})


def _args(tmp_path, **overrides):
    argv = [
        "evolve-coding",
        "--sample-dir", str(SAMPLE_DIR),
        "--split-file", str(tmp_path / "splits.json"),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--bundles-dir", str(tmp_path / "bundles"),
        "--runs-dir", str(tmp_path / "runs"),
        "--generations", str(overrides.pop("generations", 2)),
        "--population-size", str(overrides.pop("population_size", 2)),
        "--keep-elite", "1",
        "--stall-patience", "0",
        "--minibatch-size", str(overrides.pop("minibatch_size", 2)),
        "--dev-limit", str(overrides.pop("dev_limit", 1)),
        "--trace-level", overrides.pop("trace_level", "summary"),
        "--limit", str(overrides.pop("limit", 4)),
        "--cache-dir", str(tmp_path / "cache"),
    ]
    return build_parser().parse_args(argv)


def _clients(n: int = 400):
    """A worker that always returns runnable code, and an evolver that always returns a legal change."""
    return FakeLLMClient(responses=[SEASONAL, FLAT] * n), FakeLLMClient(responses=[MUTATION] * n)


@requires_sample
def test_loop_a_runs_checkpoints_and_logs(tmp_path) -> None:
    worker, evolver = _clients()
    records, best = run_evolve_coding(_args(tmp_path), worker=worker, evolver=evolver)

    assert len(records) == 2
    assert best is not None
    assert (tmp_path / "ckpt" / "gen_000.json").is_file()
    assert (tmp_path / "ckpt" / "gen_001.json").is_file()

    lines = (tmp_path / "runs" / "loop_a.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert record["loop"] == "A"
    assert record["benchmark"] == "dr_cik"
    assert "coding" in record["bundle_versions"]
    assert isinstance(record["score"], float)


@requires_sample
def test_run_records_carry_only_call_hashes_not_prompt_text(tmp_path) -> None:
    worker, evolver = _clients()
    run_evolve_coding(_args(tmp_path, trace_level="full"), worker=worker, evolver=evolver)

    body = (tmp_path / "runs" / "loop_a.jsonl").read_text(encoding="utf-8")
    assert "prompt_hash" in body
    # The compact machine log must never inline prompts or responses, whatever the trace level.
    assert "system_text" not in body
    assert "response_text" not in body


@requires_sample
def test_resume_skips_the_completed_generation(tmp_path, caplog) -> None:
    args = _args(tmp_path)
    worker, evolver = _clients()
    run_evolve_coding(args, worker=worker, evolver=evolver)
    first_calls = len(worker.calls)

    second_worker, second_evolver = _clients()
    records, _ = run_evolve_coding(args, worker=second_worker, evolver=second_evolver)
    assert second_worker.calls == []  # every generation came from the checkpoints
    assert first_calls > 0
    assert len(records) == 2


@requires_sample
def test_bundles_are_written_for_every_individual(tmp_path) -> None:
    worker, evolver = _clients()
    run_evolve_coding(_args(tmp_path), worker=worker, evolver=evolver)
    written = list((tmp_path / "bundles" / "coding").glob("v*.json"))
    assert len(written) >= 2


@requires_sample
def test_empty_evolve_split_is_a_clean_error(tmp_path) -> None:
    from evolving_agents.harness.datasets import load_labeled_tasks

    # The file must name every loaded task, or the loader rightly decides it is stale and recomputes.
    all_ids = [task.benchmark_id for task in load_labeled_tasks(sample_dir=SAMPLE_DIR)]
    args = _args(tmp_path)
    args.split_file = str(tmp_path / "no_evolve_splits.json")
    (tmp_path / "no_evolve_splits.json").write_text(json.dumps({"evolve": [], "dev": all_ids, "test": []}), encoding="utf-8")

    worker, evolver = _clients()
    with pytest.raises(SystemExit, match="evolve split is empty"):
        run_evolve_coding(args, worker=worker, evolver=evolver)


@requires_sample
def test_limit_zero_means_zero_not_unlimited(tmp_path) -> None:
    args = _args(tmp_path)
    args.limit = 0
    worker, evolver = _clients()
    with pytest.raises(SystemExit, match="evolve split is empty"):
        run_evolve_coding(args, worker=worker, evolver=evolver)


def test_parser_defaults_to_the_local_qwen_models() -> None:
    args = build_parser().parse_args(["evolve-coding", "--sample-dir", "/tmp/x", "--checkpoint-dir", "/tmp/c"])
    assert args.worker_model_id.startswith("Qwen/")
    assert args.evolver_model_id.startswith("Qwen/")
    assert args.trace_level == "summary"
