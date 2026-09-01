"""Boundary tests for the formal Champion evolution command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numerical_agent.run_champion_evolution import build_parser, main


def _options(parser):
    return {option for action in parser._actions for option in action.option_strings}


def test_evolution_cli_has_no_public_option() -> None:
    options = _options(build_parser())
    assert "--public" not in options
    assert "--public-output" not in options


def test_provision_only_creates_authority_without_reading_tasks(tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    missing = tmp_path / "must-not-be-read.jsonl"

    assert (
        main(
            [
                "--authority-root",
                str(authority),
                "--authority-identity",
                "a" * 64,
                "--provision-authority",
                "--tasks-file",
                str(missing),
            ]
        )
        == 0
    )

    assert (authority / "authority_identity.json").is_file()


def test_evolution_partition_loader_never_decodes_public_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "partitions": {
                    "train": {"task_ids": ["train"]},
                    "dev": {"task_ids": ["dev"]},
                    "public": {"task_ids": ["public"]},
                }
            }
        ),
        encoding="utf-8",
    )
    seen: list[tuple[str, ...]] = []

    def load_only(path, ids):
        del path
        seen.append(tuple(ids))
        return []

    monkeypatch.setattr(
        "numerical_agent.run_champion_evolution.load_tasks_by_id", load_only
    )
    authority = tmp_path / "authority"
    assert (
        main(
            [
                "--authority-root",
                str(authority),
                "--authority-identity",
                "b" * 64,
                "--provision-authority",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="missing train task"):
        main(
            [
                "--repo",
                str(tmp_path / "repo"),
                "--split-file",
                str(split),
                "--tasks-file",
                str(tmp_path / "tasks.jsonl"),
                "--output-dir",
                str(tmp_path / "out"),
                "--authority-root",
                str(authority),
                "--authority-identity",
                "b" * 64,
            ]
        )
    assert seen == [("train", "dev")]
