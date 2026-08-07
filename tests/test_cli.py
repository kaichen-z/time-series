"""Argument parsing for every dr-cik subcommand."""

from __future__ import annotations

from dr_cik.cli import build_parser


def test_run_arguments() -> None:
    args = build_parser().parse_args(
        ["run", "--agent", "opendr", "--sample-dir", "/tmp/sample", "--output-dir", "/tmp/out", "--num-samples", "50"]
    )
    assert args.command == "run"
    assert args.agent == "opendr"
    assert args.num_samples == 50


def test_exactly_one_data_source_is_required() -> None:
    """--sample-dir and --data-dir are mutually exclusive, and one is mandatory."""
    parser = build_parser()
    try:
        parser.parse_args(["run", "--agent", "drbench", "--output-dir", "/tmp/out"])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_download_data_arguments() -> None:
    args = build_parser().parse_args(["download-data", "--local-dir", "/tmp/data"])
    assert args.command == "download-data"
    assert args.local_dir == "/tmp/data"


def test_direct_prompt_arguments() -> None:
    args = build_parser().parse_args(
        [
            "direct-prompt",
            "--sample-dir", "/tmp/sample",
            "--from-run-dir", "/tmp/run",
            "--model-id", "Qwen/Qwen3.5-4B",
            "--output-dir", "/tmp/out",
        ]
    )
    assert args.command == "direct-prompt"
    assert args.model_id == "Qwen/Qwen3.5-4B"
    assert args.num_samples == 25
    assert args.temperature == 1.0  # must stay >0, or all S sampled draws come out identical


def test_plot_samples_arguments() -> None:
    args = build_parser().parse_args(
        [
            "plot-samples",
            "--sample-dir", "/tmp/sample",
            "--forecasts", "/tmp/out/forecasts.jsonl",
            "--output-dir", "/tmp/plots",
            "--label", "Qwen/Qwen3.5-4B",
        ]
    )
    assert args.command == "plot-samples"
    assert args.label == "Qwen/Qwen3.5-4B"


def test_plot_compare_arguments() -> None:
    args = build_parser().parse_args(
        [
            "plot-compare",
            "--sample-dir", "/tmp/sample",
            "--series", "Chronos=/tmp/a/forecasts.jsonl",
            "--series", "Qwen3.5-4B=/tmp/b/forecasts.jsonl",
            "--output-dir", "/tmp/compare",
        ]
    )
    assert args.command == "plot-compare"
    assert args.series == ["Chronos=/tmp/a/forecasts.jsonl", "Qwen3.5-4B=/tmp/b/forecasts.jsonl"]
