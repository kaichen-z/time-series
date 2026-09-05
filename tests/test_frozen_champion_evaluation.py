"""Boundary tests for one-shot frozen Champion Public regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numerical_agent.evaluate_frozen_champion import (
    build_parser,
    load_public_partition,
    main,
)


def test_frozen_public_cli_has_no_proposer_or_release_write_option() -> None:
    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert (
        not {"--proposer-model", "--generations", "--checkpoint", "--release-output"}
        & options
    )


def test_frozen_evaluation_refuses_existing_completed_report(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "public_regression_report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already completed"):
        main(
            [
                "--repo",
                str(tmp_path / "repo"),
                "--split-file",
                str(tmp_path / "split.json"),
                "--tasks-file",
                str(tmp_path / "tasks.jsonl"),
                "--release-dir",
                str(tmp_path / "release"),
                "--forecast-store",
                str(tmp_path / "forecasts"),
                "--output-dir",
                str(output),
            ]
        )


def test_frozen_loader_rejects_duplicate_requested_jsonl_rows(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"partitions": {"public_test": {"task_ids": ["public"]}}}),
        encoding="utf-8",
    )
    row = {
        "benchmark_id": "public",
        "series": {"history_values": [1.0, 2.0], "future_values": [3.0]},
        "task_metadata": {"prediction_length": 1, "frequency": "D"},
    }
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n{invalid train body}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate requested task"):
        load_public_partition(split, tasks)


@pytest.mark.parametrize(
    "reason", ("split fingerprint", "runtime identity", "source closure")
)
def test_frozen_authority_rejection_happens_before_public_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    decoded = False

    def forbidden_loader(*args, **kwargs):
        nonlocal decoded
        decoded = True
        raise AssertionError("Public rows must remain unopened")

    def reject(*args, **kwargs):
        raise ValueError(reason)

    monkeypatch.setattr(
        "numerical_agent.evaluate_frozen_champion.load_tasks_by_id", forbidden_loader
    )
    monkeypatch.setattr(
        "numerical_agent.evaluate_frozen_champion._validate_frozen_authority", reject
    )
    with pytest.raises(ValueError, match=reason):
        main(
            [
                "--repo",
                str(tmp_path / "repo"),
                "--split-file",
                str(tmp_path / "split.json"),
                "--tasks-file",
                str(tmp_path / "tasks.jsonl"),
                "--release-dir",
                str(tmp_path / "release"),
                "--forecast-store",
                str(tmp_path / "forecasts"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert decoded is False
