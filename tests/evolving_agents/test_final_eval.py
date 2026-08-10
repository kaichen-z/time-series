"""The frozen final evaluation, and the guard that keeps the test split scored exactly once."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from dr_cik.llm import FakeLLMClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from final_eval import build_parser, run_final_eval  # noqa: E402

from .conftest import SAMPLE_DIR, requires_sample  # noqa: E402

FLAT = json.dumps({"assumption": "flat", "code": "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n"})
KEEP = json.dumps({"evidence": []})
CLEAN = json.dumps({"contradicts": False, "reason": "unrelated"})


def _worker() -> FakeLLMClient:
    return FakeLLMClient(responses=[FLAT, KEEP, CLEAN] * 2000)


def _args(tmp_path, **extra):
    argv = [
        "--sample-dir", str(SAMPLE_DIR),
        "--output-dir", str(tmp_path / "final"),
        "--cache-dir", str(tmp_path / "cache"),
        "--runs-dir", str(tmp_path / "runs"),
        "--trace-level", "off",
        "--n-windows", "1",
        "--limit", "2",
        "--no-judge",
        *extra.pop("extra", []),
    ]
    return build_parser().parse_args(argv)


@requires_sample
def test_final_eval_writes_a_summary(tmp_path) -> None:
    summary = run_final_eval(_args(tmp_path), worker=_worker())
    assert summary["split"] == "test"
    assert summary["num_tasks"] > 0
    assert "development proxies" in summary["note"]
    assert set(summary["bundle_versions"]) == {"coding", "retrieval", "decision"}
    assert (tmp_path / "final" / "summary.json").is_file()
    assert (tmp_path / "runs" / "final_eval.jsonl").is_file()


@requires_sample
def test_a_second_run_against_the_same_output_dir_is_refused(tmp_path) -> None:
    run_final_eval(_args(tmp_path), worker=_worker())
    with pytest.raises(SystemExit, match="exactly once"):
        run_final_eval(_args(tmp_path), worker=_worker())


@requires_sample
def test_force_allows_a_deliberate_rerun(tmp_path) -> None:
    run_final_eval(_args(tmp_path), worker=_worker())
    summary = run_final_eval(_args(tmp_path, extra=["--force"]), worker=_worker())
    assert summary["num_tasks"] > 0


@requires_sample
def test_final_eval_only_touches_the_test_split(tmp_path) -> None:
    from evolving_agents.harness.datasets import load_drcik_splits

    args = _args(tmp_path)
    run_final_eval(args, worker=_worker())
    splits = load_drcik_splits(sample_dir=SAMPLE_DIR, seed=args.seed, split_file=None)
    scored = {json.loads(line)["task_id"] for line in (tmp_path / "runs" / "final_eval.jsonl").read_text().strip().splitlines()}

    test_ids = {task.benchmark_id for task in splits.test}
    assert scored <= test_ids
    assert not scored & {task.benchmark_id for task in splits.evolve}
    assert not scored & {task.benchmark_id for task in splits.dev}
