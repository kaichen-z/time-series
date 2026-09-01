"""Boundary tests for the formal Champion evolution command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from numerical_agent.run_champion_evolution import build_parser, main
from numerical_agent.run_champion_evolution import (
    _balanced_task_folds,
    _clean_git_source,
    _fold_stratified_screen_task_ids,
    _inventory,
    _load_screening_policy,
    _materialize_rows,
    _screened_candidates,
    load_evolution_partitions,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.screening import (
    ApplicabilityClause,
    ApplicabilityPolicy,
    FeatureTest,
    ScreeningEntry,
    ScreeningPolicy,
    profile_task,
)


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


def test_evolution_loader_rejects_duplicate_requested_jsonl_rows(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {"partitions": {"train": {"task_ids": ["train"]}, "dev": {"task_ids": []}}}
        ),
        encoding="utf-8",
    )
    row = {
        "benchmark_id": "train",
        "series": {"history_values": [1.0, 2.0], "future_values": [3.0]},
        "task_metadata": {"prediction_length": 1, "frequency": "D"},
    }
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n{invalid Public body}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate requested task"):
        load_evolution_partitions(split, tasks)


def test_clean_git_source_accepts_gitfile_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, "true\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        )
    )
    monkeypatch.setattr(
        "numerical_agent.run_champion_evolution.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )
    _clean_git_source(tmp_path)


def _reviewed_screening() -> ScreeningPolicy:
    return ScreeningPolicy(
        (
            ScreeningEntry(
                "broad_stat",
                "statistical",
                "keep",
                ApplicabilityPolicy(),
                "reviewed broad method",
            ),
            ScreeningEntry(
                "long_history_stat",
                "statistical",
                "specialized",
                ApplicabilityPolicy(
                    (
                        ApplicabilityClause(
                            feature_tests=(FeatureTest("history_length", ">", 100),)
                        ),
                    )
                ),
                "reviewed long-history specialist",
            ),
            ScreeningEntry(
                "repair_stat",
                "statistical",
                "repair",
                ApplicabilityPolicy(),
                "must not execute before repair",
            ),
            ScreeningEntry(
                "timesfm_leaf",
                "tsfm",
                "keep",
                ApplicabilityPolicy(),
                "reviewed foundation model",
            ),
            ScreeningEntry(
                "reviewed_combined",
                "combined",
                "keep",
                ApplicabilityPolicy(),
                "reviewed Combined policy",
            ),
        ),
        ("broad_stat", "timesfm_leaf"),
    )


def test_formal_inventory_uses_reviewed_dictionary_statuses_and_namespace() -> None:
    screening = _reviewed_screening()
    module = SimpleNamespace(
        methods=tuple(
            SimpleNamespace(name=name, docstring=f"{name} docs")
            for name in ("broad_stat", "long_history_stat", "repair_stat")
        )
    )
    timesfm = SimpleNamespace(name="timesfm_leaf")
    combined = SimpleNamespace(name="reviewed_combined")
    portfolio = SimpleNamespace(tsfm=(timesfm,), all_policies=(timesfm, combined))

    inventory = _inventory(module, portfolio, screening)

    statuses = {
        record.definition.method_id: record.status for record in inventory.methods
    }
    assert statuses == {
        "broad_stat": "accepted",
        "long_history_stat": "specialized",
        "repair_stat": "quarantined",
        "timesfm_leaf": "accepted",
        "reviewed_combined": "accepted",
    }


def test_formal_dictionary_is_parsed_and_controls_task_applicability(
    tmp_path: Path,
) -> None:
    dictionary = tmp_path / "dictionary.py"
    dictionary.write_text(
        "CANDIDATES = "
        + repr(
            [
                {
                    "name": entry.name,
                    "family": entry.family,
                    "status": entry.status,
                    "any_of": [
                        {
                            "all_tags": list(clause.all_tags),
                            "feature_tests": [
                                {
                                    "field": test.field,
                                    "operator": test.operator,
                                    "value": test.value,
                                }
                                for test in clause.feature_tests
                            ],
                        }
                        for clause in entry.applicability.any_of
                    ],
                    "reason": entry.reason,
                }
                for entry in _reviewed_screening().entries
            ]
        )
        + "\nFALLBACK_NAMES = "
        + repr(_reviewed_screening().fallback_names)
        + "\n",
        encoding="utf-8",
    )
    policy = _load_screening_policy(dictionary)
    profile = profile_task(
        Task("short", tuple(float(i) for i in range(20)), 2, "D", ())
    )
    candidates = (
        ("broad_stat", "statistical"),
        ("long_history_stat", "statistical"),
        ("repair_stat", "statistical"),
        ("timesfm_leaf", "tsfm"),
        ("reviewed_combined", "combined"),
    )

    assert _screened_candidates(policy, profile, candidates) == (
        ("broad_stat", "statistical"),
        ("timesfm_leaf", "tsfm"),
        ("reviewed_combined", "combined"),
    )


def test_formal_five_fold_assignment_is_stable_balanced_and_order_independent() -> None:
    tasks = tuple(
        SimpleNamespace(task_id=f"task_{index:02d}", entity_name=f"entity_{index:02d}")
        for index in range(16)
    )

    first = _balanced_task_folds(tasks, seed=20260901)
    second = _balanced_task_folds(tuple(reversed(tasks)), seed=20260901)

    assert first == second
    assert set(first.values()) == {0, 1, 2, 3, 4}
    counts = tuple(first.values()).count
    assert max(counts(fold) for fold in range(5)) - min(
        counts(fold) for fold in range(5)
    ) <= 1


def test_formal_screen_membership_is_nested_and_fold_stratified() -> None:
    tasks = tuple(
        SimpleNamespace(task_id=f"task_{index:02d}", entity_name=f"entity_{index:02d}")
        for index in range(64)
    )
    folds = _balanced_task_folds(tasks, seed=20260901)

    screens = _fold_stratified_screen_task_ids(tasks, folds, sizes=(8, 32, 64))

    assert tuple(map(len, screens)) == (8, 32, 64)
    assert set(screens[0]).issubset(screens[1])
    assert set(screens[1]).issubset(screens[2])
    assert set(folds[task_id] for task_id in screens[0]) == {0, 1, 2, 3, 4}
    assert set(screens[-1]) == {task.task_id for task in tasks}


def test_formal_specialist_inapplicability_is_a_complete_failed_row() -> None:
    screening = _reviewed_screening()
    tasks = (
        SimpleNamespace(
            task_id="short",
            entity_name="short_entity",
            history_values=tuple(float(index) for index in range(20)),
            future_values=(20.0, 21.0),
            prediction_length=2,
            frequency="D",
        ),
        SimpleNamespace(
            task_id="long",
            entity_name="long_entity",
            history_values=tuple(float(index) for index in range(120)),
            future_values=(120.0, 121.0),
            prediction_length=2,
            frequency="D",
        ),
    )

    class Store:
        identity_hash = "fixture-store"

        @staticmethod
        def forecast(
            _name: str,
            history: tuple[float, ...],
            horizon: int,
            _frequency: str,
        ) -> tuple[float, ...]:
            return (history[-1],) * horizon

    rows = _materialize_rows(
        Store(),  # type: ignore[arg-type]
        tasks,  # type: ignore[arg-type]
        (("broad_stat", "statistical"), ("long_history_stat", "statistical")),
        screening,
        {"short": 0, "long": 1},
        "build",
    )

    by_key = {(row.candidate_name, row.task_id): row for row in rows}
    assert set(by_key) == {
        ("broad_stat", "short"),
        ("broad_stat", "long"),
        ("long_history_stat", "short"),
        ("long_history_stat", "long"),
    }
    inactive = by_key[("long_history_stat", "short")]
    assert inactive.forecast is None
    assert inactive.failure_reason == "NotApplicable: screening_policy"
    assert by_key[("long_history_stat", "long")].forecast is not None
