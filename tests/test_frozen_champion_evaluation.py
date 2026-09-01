"""Boundary tests for one-shot frozen Champion Public regression."""

from __future__ import annotations

from pathlib import Path

import pytest

from numerical_agent.evaluate_frozen_champion import build_parser, main


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
