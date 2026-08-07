"""Command-line entrypoint: dr-cik download-data | download-models | run | direct-prompt | plot-samples | plot-compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .data import DEFAULT_DATA_DIR, download_dataset, load_sample_tasks, load_tasks
from .forecasters.chronos import ChronosConfig, ChronosForecaster, DEFAULT_CACHE_DIR
from .forecasters.direct_prompt import DirectPromptConfig, DirectPromptForecaster, load_prior_context
from .local_llm import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
from .local_llm import QwenClient, QwenConfig
from .pipeline import RunConfig, build_pipeline, run_direct_prompt, write_direct_prompt_outputs, write_outputs
from .plotting import plot_comparison_files, plot_forecasts_file
from .retrieval import RETRIEVERS


def _add_task_source_args(subparser: argparse.ArgumentParser) -> None:
    """--sample-dir / --data-dir + --split + --task-id + --limit, shared by run/direct-prompt/plot-samples."""
    source = subparser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir", help="Path to the official Dr-CiK sample/ directory")
    source.add_argument("--data-dir", help="Path to a downloaded full Dr-CiK dataset")
    subparser.add_argument("--split", choices=("public-dev", "hidden-test", "all"), default="public-dev")
    subparser.add_argument("--task-id", action="append", help="Run only this benchmark_id; repeatable")
    subparser.add_argument("--limit", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dr-cik", description="OpenDR/DRBench + Chronos reproduction for Dr-CiK.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_data = subparsers.add_parser("download-data", help="Download the full ServiceNow/Dr-CiK dataset")
    download_data.add_argument("--local-dir", default=str(DEFAULT_DATA_DIR))
    download_data.add_argument("--revision", default=None)

    download_models = subparsers.add_parser("download-models", help="Download the Chronos checkpoint")
    download_models.add_argument("--chronos-model-id", default="amazon/chronos-bolt-base")
    download_models.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)

    run = subparsers.add_parser("run", help="Run an agent + Chronos over Dr-CiK tasks")
    run.add_argument("--agent", choices=("opendr", "drbench"), required=True)
    _add_task_source_args(run)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--llm-backend", choices=("gemini", "qwen"), default="gemini")
    run.add_argument("--gemini-model-id", default="gemini-3-flash-preview")
    run.add_argument("--judge-model-id", default="gemini-3-flash-preview")
    run.add_argument("--qwen-model-id", default=DEFAULT_QWEN_MODEL_ID)
    run.add_argument("--qwen-device", default=None, help="e.g. cuda:2; default auto-picks the GPU with the most free memory")
    run.add_argument("--no-judge", action="store_true")
    run.add_argument("--chronos-model-id", default="amazon/chronos-bolt-base")
    run.add_argument("--chronos-cache-dir", default=DEFAULT_CACHE_DIR)
    run.add_argument("--chronos-device-map", default="cpu")
    run.add_argument("--num-samples", type=int, default=100)
    run.add_argument("--crps-sample-size", type=int, default=25)
    run.add_argument("--max-react-steps", type=int, default=6)
    run.add_argument("--drbench-top-k", type=int, default=8)
    run.add_argument("--retriever", choices=RETRIEVERS, default="bm25", help="bm25 (lexical, default) or dense (embeddings, as DRBench itself uses)")
    run.add_argument("--seed", type=int, default=7)

    direct_prompt = subparsers.add_parser("direct-prompt", help="Direct-Prompt LLM baseline: forecast directly from history + a prior run's DR-synthesized context")
    _add_task_source_args(direct_prompt)
    direct_prompt.add_argument("--from-run-dir", required=True, help="A completed `run` output dir to load report_markdown/evidence context from")
    direct_prompt.add_argument("--model-id", required=True, help="e.g. Qwen/Qwen3.5-4B or Qwen/Qwen3.5-9B")
    direct_prompt.add_argument("--output-dir", required=True)
    direct_prompt.add_argument("--qwen-device", default=None, help="e.g. cuda:2; default auto-picks the GPU with the most free memory")
    direct_prompt.add_argument("--num-samples", type=int, default=25)
    direct_prompt.add_argument("--temperature", type=float, default=1.0)
    direct_prompt.add_argument("--max-output-tokens", type=int, default=512, help="Floor only; scales up automatically with the task's horizon")
    direct_prompt.add_argument("--crps-sample-size", type=int, default=25)
    direct_prompt.add_argument("--seed", type=int, default=7)

    plot_samples = subparsers.add_parser("plot-samples", help="Plot history + sample trajectories + mean forecast from a forecasts.jsonl")
    _add_task_source_args(plot_samples)
    plot_samples.add_argument("--forecasts", required=True, help="Path to a forecasts.jsonl (from `run` or `direct-prompt`)")
    plot_samples.add_argument("--output-dir", required=True)
    plot_samples.add_argument("--label", required=True, help="Shown in each plot's title, e.g. a model id")

    plot_compare = subparsers.add_parser("plot-compare", help="Overlay multiple forecasts.jsonl runs (e.g. Chronos vs several Direct-Prompt models) per task")
    _add_task_source_args(plot_compare)
    plot_compare.add_argument("--series", action="append", required=True, metavar="LABEL=PATH", help="Repeatable; e.g. --series 'Chronos=outputs/drbench-sample/forecasts.jsonl'")
    plot_compare.add_argument("--output-dir", required=True)
    return parser


def _load_tasks(args: argparse.Namespace):
    """Shared --sample-dir/--data-dir/--split/--task-id/--limit resolution for run/direct-prompt/plot-samples."""
    if args.sample_dir:
        tasks = load_sample_tasks(args.sample_dir)
    else:
        labels_public = {"public-dev": True, "hidden-test": False, "all": None}[args.split]
        tasks = load_tasks(data_dir=args.data_dir, labels_public=labels_public)
    return _select_tasks(tasks, args.task_id, args.limit)


def _select_tasks(tasks, task_ids: list[str] | None, limit: int | None):
    if task_ids:
        requested = set(task_ids)
        tasks = [task for task in tasks if task.benchmark_id in requested]
        missing = requested - {task.benchmark_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task IDs: {', '.join(sorted(missing))}")
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise SystemExit("No tasks selected")
    return tasks


def main(argv: list[str] | None = None) -> None:
    load_dotenv()  # picks up GEMINI_API_KEY etc. from a .env file, walking up from cwd
    args = build_parser().parse_args(argv)

    if args.command == "download-data":
        path = download_dataset(local_dir=args.local_dir, revision=args.revision)
        print(f"Dataset downloaded to {path}")
        return

    if args.command == "download-models":
        forecaster = ChronosForecaster(ChronosConfig(model_id=args.chronos_model_id, cache_dir=args.cache_dir))
        forecaster.warm_up()
        print(f"Chronos checkpoint {args.chronos_model_id} cached under {args.cache_dir}")
        return

    if args.command == "run":
        tasks = _load_tasks(args)
        config = RunConfig(
            agent=args.agent,
            llm_backend=args.llm_backend,
            gemini_model_id=args.gemini_model_id,
            judge_model_id=args.judge_model_id,
            qwen_model_id=args.qwen_model_id,
            qwen_device=args.qwen_device,
            judge_enabled=not args.no_judge,
            chronos_model_id=args.chronos_model_id,
            chronos_device_map=args.chronos_device_map,
            chronos_cache_dir=args.chronos_cache_dir,
            num_samples=args.num_samples,
            crps_sample_size=args.crps_sample_size,
            max_react_steps=args.max_react_steps,
            drbench_top_k=args.drbench_top_k,
            retriever=args.retriever,
            seed=args.seed,
        )
        pipeline = build_pipeline(config)
        results = pipeline.run_many(tasks)
        write_outputs(results, args.output_dir, config=config)

        summary_path = Path(args.output_dir).expanduser().resolve() / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Completed {len(results)} task(s) with {args.agent}.")
        print(f"Outputs: {summary_path.parent}")
        for name, value in summary["mean_metrics"].items():
            print(f"  {name}: {value:.6f}")
        return

    if args.command == "direct-prompt":
        tasks = _load_tasks(args)
        context_by_id = load_prior_context(args.from_run_dir)
        llm = QwenClient(QwenConfig(model_id=args.model_id, device=args.qwen_device, seed=args.seed))
        forecaster = DirectPromptForecaster(
            llm,
            DirectPromptConfig(
                model_id=args.model_id,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                seed=args.seed,
            ),
        )
        results = run_direct_prompt(tasks, forecaster, context_by_id, crps_sample_size=args.crps_sample_size)
        write_direct_prompt_outputs(results, args.output_dir, model_id=args.model_id, from_run_dir=args.from_run_dir)

        summary_path = Path(args.output_dir).expanduser().resolve() / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Completed {len(results)} task(s) with direct-prompt/{args.model_id}.")
        print(f"Outputs: {summary_path.parent}")
        for name, value in summary["mean_metrics"].items():
            print(f"  {name}: {value:.6f}")
        return

    if args.command == "plot-samples":
        tasks = _load_tasks(args)
        written = plot_forecasts_file(tasks, args.forecasts, args.output_dir, label=args.label)
        print(f"Wrote {len(written)} plot(s) to {Path(args.output_dir).expanduser().resolve()}")
        return

    if args.command == "plot-compare":
        tasks = _load_tasks(args)
        series: list[tuple[str, str]] = []
        for item in args.series:
            if "=" not in item:
                raise SystemExit(f"--series must be LABEL=PATH, got {item!r}")
            label, path = item.split("=", 1)
            series.append((label, path))
        written = plot_comparison_files(tasks, series, args.output_dir)
        print(f"Wrote {len(written)} comparison plot(s) to {Path(args.output_dir).expanduser().resolve()}")
        return


if __name__ == "__main__":
    main()
