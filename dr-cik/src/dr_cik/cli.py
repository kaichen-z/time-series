"""Command-line entrypoint: dr-cik download-data | download-models | run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .backbone import ChronosBackboneConfig, ChronosForecastBackbone, DEFAULT_CACHE_DIR
from .data import DEFAULT_DATA_DIR, download_dataset, load_sample_tasks, load_tasks
from .local_llm import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
from .pipeline import RunConfig, build_pipeline, write_outputs


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
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir", help="Path to the official Dr-CiK sample/ directory")
    source.add_argument("--data-dir", help="Path to a downloaded full Dr-CiK dataset")
    run.add_argument("--split", choices=("public-dev", "hidden-test", "all"), default="public-dev")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--task-id", action="append", help="Run only this benchmark_id; repeatable")
    run.add_argument("--limit", type=int, default=None)
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
    run.add_argument("--seed", type=int, default=7)
    return parser


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
        backbone = ChronosForecastBackbone(ChronosBackboneConfig(model_id=args.chronos_model_id, cache_dir=args.cache_dir))
        backbone.warm_up()
        print(f"Chronos checkpoint {args.chronos_model_id} cached under {args.cache_dir}")
        return

    if args.sample_dir:
        tasks = load_sample_tasks(args.sample_dir)
    else:
        labels_public = {"public-dev": True, "hidden-test": False, "all": None}[args.split]
        tasks = load_tasks(data_dir=args.data_dir, labels_public=labels_public)
    tasks = _select_tasks(tasks, args.task_id, args.limit)

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
        seed=args.seed,
    )
    pipeline = build_pipeline(config)
    results = pipeline.run_many(tasks)
    write_outputs(results, args.output_dir)

    summary_path = Path(args.output_dir).expanduser().resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Completed {len(results)} task(s) with {args.agent}.")
    print(f"Outputs: {summary_path.parent}")
    for name, value in summary["mean_metrics"].items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
