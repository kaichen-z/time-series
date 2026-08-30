from types import SimpleNamespace

from numerical_agent.run_filter_evolution import (
    SCALED_METRIC_POLICY,
    _manifest_fingerprint,
    _markdown,
    _paired_filter_counts,
    build_parser,
)
from common.evolution_core.contracts import METRIC_POLICY


def test_filter_smoke_defaults_to_eight_train_two_dev_and_luna() -> None:
    args = build_parser().parse_args(
        [
            "--repo", "run-repo",
            "--tasks-file", "tasks",
            "--outcome-cache-dir", "method-cache",
            "--policy-outcome-cache-dir", "policy-cache",
        ]
    )

    assert args.train_limit == 8
    assert args.validation_tail == 2
    assert args.codex_model == "gpt-5.6-luna"
    assert args.codex_reasoning_effort == "medium"


def test_filter_report_and_manifest_lead_with_bound_scaled_objective() -> None:
    assert SCALED_METRIC_POLICY == METRIC_POLICY
    score = {
        "mean_smae": 0.8,
        "mean_srmse": 0.9,
        "median_smae": 0.7,
        "median_srmse": 0.8,
        "se_smae": 0.01,
        "se_srmse": 0.02,
        "p90_smae_raw": 6.0,
        "p95_smae_raw": 7.0,
        "p90_srmse_raw": 8.0,
        "p95_srmse_raw": 9.0,
        "smae_clipped_count": 1,
        "smae_clipped_rate": 0.1,
        "srmse_clipped_count": 2,
        "srmse_clipped_rate": 0.2,
        "coverage": 1.0,
        "eligible_crashed": 1,
        "eligible_invalid": 2,
        "eligible_missing": 3,
        "eligible_malformed_success": 4,
    }
    payload = {
        "accepted": True,
        "reason": "scaled improvement",
        "elapsed_seconds": 1.0,
        "changes": [],
        "metric_policy": SCALED_METRIC_POLICY,
        "diagnostic_only_metrics": ["mase", "mae", "smape"],
        "parent": {"train": score, "dev": score},
        "child": {"train": score, "dev": score},
        "paired_joint_wtl": {
            "train": {"wins": 1, "ties": 2, "losses": 3, "missing": 4, "unscored": 5},
            "dev": {"wins": 5, "ties": 4, "losses": 3, "missing": 2, "unscored": 1},
        },
    }

    report = _markdown(payload)

    assert "Parent mean sMAE" in report
    assert "Parent mean sRMSE" in report
    assert "Parent median sMAE" in report
    assert "Parent sMAE SE" in report
    assert "Raw P90/P95 sMAE" in report
    assert "Clipped sMAE/sRMSE" in report
    assert "Wins / Ties / Losses / Missing / Unscored" in report
    assert "Crash / invalid / missing / malformed" in report
    assert "1 / 2 / 3 / 4" in report
    assert "MASE" not in report
    assert _manifest_fingerprint(payload) != _manifest_fingerprint(
        {**payload, "metric_policy": {**SCALED_METRIC_POLICY, "scaled_metric_cap": 4.0}}
    )


def test_filter_paired_counts_conserve_tasks_with_both_missing_unscored() -> None:
    parent = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "left": (1.0, 1.0)})
    child = SimpleNamespace(task_count=4, task_scaled_pairs={"same": (1.0, 1.0), "right": (1.0, 1.0)})

    counts = _paired_filter_counts(parent, child)

    assert counts == {"wins": 0, "ties": 1, "losses": 0, "missing": 2, "unscored": 1}
    assert sum(counts.values()) == 4
