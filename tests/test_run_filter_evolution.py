from numerical_agent.run_filter_evolution import (
    SCALED_METRIC_POLICY,
    _manifest_fingerprint,
    _markdown,
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
    }

    report = _markdown(payload)

    assert "Parent mean sMAE" in report
    assert "Parent mean sRMSE" in report
    assert "Crash / invalid / missing / malformed" in report
    assert "1 / 2 / 3 / 4" in report
    assert "MASE" not in report
    assert _manifest_fingerprint(payload) != _manifest_fingerprint(
        {**payload, "metric_policy": {**SCALED_METRIC_POLICY, "scaled_metric_cap": 4.0}}
    )
