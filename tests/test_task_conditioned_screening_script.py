from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from numerical_agent.run_task_conditioned_screening import (
    _merge_cache_summaries,
    _report,
    _train_constraints_met,
    _write_policy_artifacts,
    build_parser,
    load_frozen_partitions,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
    ScreeningConstraints,
)


ROOT = Path(__file__).resolve().parents[1]


def test_partition_loader_does_not_open_dev_records_in_train_only_mode(
    tmp_path, monkeypatch
):
    split = tmp_path / "split.json"
    split.write_text(
        '{"partitions":{"train":{"task_ids":["train-1"]},'
        '"dev":{"task_ids":["task_dev_1"]}}}'.replace("train-1", "task_train_1"),
        encoding="utf-8",
    )
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    record = (
        '{{"benchmark_id":"{task_id}","series":{{"history_values":[1,2],'
        '"future_values":[3]}},"task_metadata":{{"prediction_length":1,'
        '"frequency":"D"}},"entity_name":"{entity}"}}'
    )
    (task_dir / "task_train_1.json").write_text(
        record.format(task_id="task_train_1", entity="train-entity"), encoding="utf-8"
    )
    (task_dir / "task_dev_1.json").write_text(
        record.format(task_id="task_dev_1", entity="dev-entity"), encoding="utf-8"
    )
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "task_dev_1.json":
            raise AssertionError("Train-only loading opened a Dev record")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    train, dev = load_frozen_partitions(
        split, task_dir, train_limit=1, dev_limit=0
    )

    assert [task.task_id for task in train] == ["task_train_1"]
    assert dev == ()


def test_train_constraints_do_not_require_dev_and_cache_summaries_are_additive():
    score = SimpleNamespace(
        global_oracle_retention=1.0,
        min_active_candidates=12,
        max_active_candidates=80,
        unique_active_dictionaries=3,
        task_count=80,
        mean_pairwise_jaccard=0.9,
        conditioned_entries_by_family={
            "statistical": 1,
            "tsfm": 1,
            "combined": 1,
        },
    )
    constraints = ScreeningConstraints(max_active_candidates=103)

    assert _train_constraints_met(score, constraints)
    assert _merge_cache_summaries(
        {"statistical_hits": 10, "tsfm_misses": 2},
        {"statistical_hits": 3, "tsfm_misses": 1},
    ) == {"statistical_hits": 13, "tsfm_misses": 3}


def test_screening_cli_has_train_dev_but_no_public_test_option():
    parser = build_parser()
    args = parser.parse_args([
        "--repo", "repo",
        "--tasks-file", "tasks",
        "--outcome-cache-dir", "method-cache",
        "--policy-outcome-cache-dir", "policy-cache",
        "--output-dir", "output",
        "--target-batches-file", "batches.json",
        "--train-limit", "80",
        "--dev-limit", "20",
    ])
    assert args.train_limit == 80
    assert args.dev_limit == 20
    assert args.seed_policy == "all"
    assert args.baseline_method == "toto_2_0"
    assert args.screen_min_candidates == 12
    assert args.screen_max_candidates == 103
    assert args.screen_min_unique_dictionaries == 3
    assert args.screen_min_dev_oracle_retention == 0.9
    assert args.screen_batch_size == 8
    assert args.screen_refinement_generations == 3
    assert args.screen_refinement_batch_size == 24
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo", "repo", "--tasks-file", "tasks",
            "--outcome-cache-dir", "cache", "--policy-outcome-cache-dir", "cache2",
            "--output-dir", "out", "--target-batches-file", "batch",
            "--public-test-limit", "99",
        ])


def test_screening_shell_forwards_formal_configuration():
    script = ROOT / "scripts" / "run_task_conditioned_screening.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    for option in (
        "--repo", "--split-file", "--tasks-file", "--outcome-cache-dir",
        "--policy-outcome-cache-dir", "--train-limit", "--dev-limit",
        "--target-batches-file", "--output-dir", "--codex-model",
            "--screen-refinement-generations", "--screen-refinement-batch-size",
            "--screen-min-dev-oracle-retention",
    ):
        assert option in source
    assert 'MAX_CANDIDATES="${SCREEN_MAX_CANDIDATES:-103}"' in source


def test_report_exposes_task_conditioning_and_family_coverage():
    score = {
        "coverage": 1.0,
        "active_success_rate": 0.8,
        "failure_exposure": 0.1,
        "not_applicable_exposure": 0.1,
        "global_oracle_retention": 1.0,
        "mean_active_oracle_regret": 0.0,
        "compression": 0.4,
        "mean_active_candidates": 32.5,
        "min_active_candidates": 24,
        "max_active_candidates": 40,
        "unique_active_dictionaries": 7,
        "mean_pairwise_jaccard": 0.81,
        "conditioned_entries_by_family": {
            "statistical": 9,
            "tsfm": 2,
            "combined": 3,
        },
    }
    report = _report({
        "candidate_count": 103,
        "train_tasks": 80,
        "dev_tasks": 20,
        "accepted_generations": [1, 2],
        "candidate_screening_policy_sha256": "abc",
        "frozen_screening_policy_sha256": "abc",
        "public_test_accessed": False,
        "final_constraints_met": True,
        "dev_evaluations": 1,
        "final_dev_gate": {"accepted": True, "reason": "safe on held-out Dev"},
        "train": score,
        "dev": score,
    })

    assert "Unique dictionaries" in report
    assert "Pairwise Jaccard" in report
    assert "Statistical / TSFM / Combined" in report
    assert "9 / 2 / 3" in report
    assert "Dev evaluations: 1" in report
    assert "safe on held-out Dev" in report


def test_rejected_candidate_is_not_published_as_frozen(tmp_path):
    policy = ScreeningPolicy(
        (
            ScreeningEntry("naive", "statistical", "keep", ApplicabilityPolicy(()), "safe"),
            ScreeningEntry("timesfm", "tsfm", "keep", ApplicabilityPolicy(()), "safe"),
            ScreeningEntry("blend", "combined", "keep", ApplicabilityPolicy(()), "safe"),
        ),
        ("naive", "timesfm", "blend"),
    )

    rejected = _write_policy_artifacts(tmp_path, policy, accepted=False)

    assert (tmp_path / "candidate_screening_policy.py").is_file()
    assert not (tmp_path / "frozen_screening_policy.py").exists()
    assert rejected["frozen_screening_policy_sha256"] is None

    accepted = _write_policy_artifacts(tmp_path, policy, accepted=True)

    assert (tmp_path / "frozen_screening_policy.py").is_file()
    assert accepted["frozen_screening_policy_sha256"] is not None
