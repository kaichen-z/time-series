from numerical_agent.run_filter_evolution import build_parser


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
