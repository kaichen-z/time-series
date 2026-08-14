from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codex_baseline import CodexContractSystem, CodexDirectBaseline, CodexDirectConfig
from .data import load_huggingface_tasks, load_sample_tasks
from .loop import IterativeAgentSystem, LoopConfig
from .pipeline import MinimalAgentSystem, SystemConfig, write_outputs
from .triad import ThreeAgentForecastSystem, TriadConfig


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="outputs/agent-loop")
    parser.add_argument("--task-id", action="append", help="Run only this benchmark_id; repeatable")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--system",
        choices=("iterative", "triad", "one-pass", "codex-direct", "codex-contract"),
        default="iterative",
        help="Run our iterative loop, one-pass ablation, or full-corpus Codex baseline",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Documents retrieved per loop step")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-no-progress", type=int, default=4)
    parser.add_argument("--convergence-tolerance", type=float, default=0.002)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--context-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--memory-file",
        default=None,
        help="Optional JSONL post-hoc memory used to calibrate later tasks",
    )
    parser.add_argument(
        "--feedback-file",
        default=None,
        help="Optional JSONL module-level feedback log for --system triad",
    )
    parser.add_argument(
        "--evolution-file",
        default=None,
        help="Persisted interpretable policy read and updated by --system triad",
    )
    parser.add_argument(
        "--agent-bundle",
        default=None,
        help="Versioned co-evolved Coding/Retrieval/Decision prompt bundle for Codex triad",
    )
    parser.add_argument(
        "--learn-from-public-outcomes",
        action="store_true",
        help="Research-only: write memory after each labeled task resolves; never used for hidden test",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="Retrieve this multiple of top-k before forecast-utility reranking",
    )
    parser.add_argument(
        "--context-character-budget",
        type=int,
        default=12000,
        help="Total importance-aware context budget across accepted documents",
    )
    parser.add_argument(
        "--revision-threshold",
        type=float,
        default=0.60,
        help="Minimum predicted utility required to revise the numerical prior",
    )
    parser.add_argument(
        "--min-information-gain",
        type=float,
        default=0.05,
        help="Stop when the controller expects less marginal value from another retrieval turn",
    )
    parser.add_argument(
        "--backbone",
        choices=("chronos", "timesfm", "statistical"),
        default="chronos",
        help="Numerical forecasting backbone; Chronos-Bolt is the default",
    )
    parser.add_argument(
        "--chronos-model-id",
        default="amazon/chronos-bolt-small",
    )
    parser.add_argument("--chronos-device-map", default="cpu")
    parser.add_argument("--chronos-max-context", type=int, default=2048)
    parser.add_argument("--chronos-max-horizon", type=int, default=1024)
    parser.add_argument("--chronos-cache-dir", default=None)
    parser.add_argument(
        "--chronos-local-files-only",
        action="store_true",
        help="Do not download a Chronos checkpoint; require it in the local Hugging Face cache",
    )
    parser.add_argument(
        "--timesfm-model-id",
        default="google/timesfm-2.5-200m-pytorch",
    )
    parser.add_argument("--timesfm-max-context", type=int, default=4096)
    parser.add_argument("--timesfm-max-horizon", type=int, default=1024)
    parser.add_argument("--timesfm-cache-dir", default=None)
    parser.add_argument(
        "--timesfm-local-files-only",
        action="store_true",
        help="Do not download a checkpoint; require it in the local Hugging Face cache",
    )
    parser.add_argument(
        "--allow-statistical-fallback",
        action="store_true",
        help="Explicitly fall back to the statistical ablation if the selected backbone cannot load",
    )
    parser.add_argument(
        "--oracle-evidence",
        action="store_true",
        help=(
            "PUBLIC DEVELOPMENT DIAGNOSTIC ONLY: bypass retrieval with gt_evidence "
            "to measure the downstream forecasting ceiling"
        ),
    )
    parser.add_argument(
        "--allow-unvalidated-event-revisions",
        action="store_true",
        help=(
            "Unsafe research ablation: allow generic text-derived multiply/add revisions "
            "without a history backtest"
        ),
    )
    parser.add_argument(
        "--reasoning-agent",
        choices=("rules", "codex"),
        default="rules",
        help="Use deterministic text agents (default) or schema-constrained Codex agents",
    )
    parser.add_argument(
        "--codex-stages",
        default="query,verify,impact",
        help="Comma-separated Codex stages: query,verify,impact",
    )
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Optional Codex model override; by default use the Codex CLI default",
    )
    parser.add_argument("--codex-cache-dir", default="outputs/codex-cache")
    parser.add_argument("--codex-timeout", type=int, default=180)
    parser.add_argument("--codex-max-document-characters", type=int, default=12000)
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=None,
        help="Defaults to high for Codex full-corpus systems and low for the hybrid loop",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drcik-agent",
        description="Run the forecast-aware iterative retrieval agent on Dr-CiK.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("run-sample", help="Run the official repository sample")
    sample.add_argument("--sample-dir", required=True, help="Path to Dr-CiK/sample")
    _add_shared_arguments(sample)

    huggingface = subparsers.add_parser("run-hf", help="Run the Hugging Face release")
    split = huggingface.add_mutually_exclusive_group()
    split.add_argument("--public-dev", action="store_true", help="Run public labeled tasks (default)")
    split.add_argument("--hidden-test", action="store_true", help="Run hidden unlabeled tasks")
    _add_shared_arguments(huggingface)
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
    arguments = build_parser().parse_args(argv)
    if getattr(arguments, "hidden_test", False) and arguments.learn_from_public_outcomes:
        raise SystemExit("--learn-from-public-outcomes cannot be used on hidden test tasks")
    if getattr(arguments, "hidden_test", False) and arguments.oracle_evidence:
        raise SystemExit("--oracle-evidence is forbidden on hidden test tasks")
    if arguments.system == "one-pass" and arguments.learn_from_public_outcomes:
        raise SystemExit("--learn-from-public-outcomes requires --system iterative or triad")
    if arguments.system == "one-pass" and arguments.oracle_evidence:
        raise SystemExit("--oracle-evidence requires --system iterative")
    if arguments.system == "one-pass" and arguments.allow_unvalidated_event_revisions:
        raise SystemExit(
            "--allow-unvalidated-event-revisions applies only to --system iterative"
        )
    if arguments.system == "one-pass" and arguments.reasoning_agent == "codex":
        raise SystemExit("--reasoning-agent codex requires --system iterative")
    if arguments.system in {"codex-direct", "codex-contract"} and (
        arguments.learn_from_public_outcomes
        or arguments.oracle_evidence
        or arguments.allow_unvalidated_event_revisions
    ):
        raise SystemExit(
            f"--system {arguments.system} does not use memory, oracle evidence, or hybrid revision flags"
        )
    codex_stages = tuple(
        stage.strip() for stage in arguments.codex_stages.split(",") if stage.strip()
    )
    invalid_codex_stages = set(codex_stages) - {"query", "verify", "impact"}
    if invalid_codex_stages:
        raise SystemExit(
            "Unknown --codex-stages: " + ", ".join(sorted(invalid_codex_stages))
        )
    codex_reasoning_effort = arguments.codex_reasoning_effort or (
        "high"
        if arguments.system in {"codex-direct", "codex-contract"}
        or (arguments.system == "triad" and arguments.reasoning_agent == "codex")
        else "low"
    )
    if arguments.command == "run-sample":
        tasks = load_sample_tasks(arguments.sample_dir)
    else:
        labels_public = not arguments.hidden_test
        tasks = load_huggingface_tasks(labels_public=labels_public)
    tasks = _select_tasks(tasks, arguments.task_id, arguments.limit)

    if arguments.system == "one-pass":
        system = MinimalAgentSystem(
            SystemConfig(
                top_k=arguments.top_k,
                num_samples=arguments.samples,
                context_weight=arguments.context_weight,
                seed=arguments.seed,
                backbone=arguments.backbone,
                chronos_model_id=arguments.chronos_model_id,
                chronos_device_map=arguments.chronos_device_map,
                chronos_max_context=arguments.chronos_max_context,
                chronos_max_horizon=arguments.chronos_max_horizon,
                chronos_cache_dir=arguments.chronos_cache_dir,
                chronos_local_files_only=arguments.chronos_local_files_only,
                timesfm_model_id=arguments.timesfm_model_id,
                timesfm_max_context=arguments.timesfm_max_context,
                timesfm_max_horizon=arguments.timesfm_max_horizon,
                timesfm_cache_dir=arguments.timesfm_cache_dir,
                timesfm_local_files_only=arguments.timesfm_local_files_only,
                allow_statistical_fallback=arguments.allow_statistical_fallback,
            )
        )
    elif arguments.system == "triad":
        system = ThreeAgentForecastSystem(
            TriadConfig(
                max_rounds=arguments.max_steps,
                documents_per_round=arguments.top_k,
                num_samples=arguments.samples,
                seed=arguments.seed,
                feedback_path=arguments.feedback_file,
                evolution_path=arguments.evolution_file,
                agent_bundle_path=arguments.agent_bundle,
                learn_from_public_outcomes=arguments.learn_from_public_outcomes,
                backbone=arguments.backbone,
                chronos_model_id=arguments.chronos_model_id,
                chronos_device_map=arguments.chronos_device_map,
                chronos_max_context=arguments.chronos_max_context,
                chronos_max_horizon=arguments.chronos_max_horizon,
                chronos_cache_dir=arguments.chronos_cache_dir,
                chronos_local_files_only=arguments.chronos_local_files_only,
                timesfm_model_id=arguments.timesfm_model_id,
                timesfm_max_context=arguments.timesfm_max_context,
                timesfm_max_horizon=arguments.timesfm_max_horizon,
                timesfm_cache_dir=arguments.timesfm_cache_dir,
                timesfm_local_files_only=arguments.timesfm_local_files_only,
                allow_statistical_fallback=arguments.allow_statistical_fallback,
                reasoning_agent=arguments.reasoning_agent,
                codex_binary=arguments.codex_binary,
                codex_model=arguments.codex_model,
                codex_cache_dir=arguments.codex_cache_dir,
                codex_timeout_seconds=arguments.codex_timeout,
                codex_max_document_characters=arguments.codex_max_document_characters,
                codex_reasoning_effort=codex_reasoning_effort,
            )
        )
    elif arguments.system == "codex-direct":
        system = CodexDirectBaseline(
            CodexDirectConfig(
                num_samples=arguments.samples,
                seed=arguments.seed,
                backbone=arguments.backbone,
                chronos_model_id=arguments.chronos_model_id,
                chronos_device_map=arguments.chronos_device_map,
                chronos_max_context=arguments.chronos_max_context,
                chronos_max_horizon=arguments.chronos_max_horizon,
                chronos_cache_dir=arguments.chronos_cache_dir,
                chronos_local_files_only=arguments.chronos_local_files_only,
                timesfm_model_id=arguments.timesfm_model_id,
                timesfm_max_context=arguments.timesfm_max_context,
                timesfm_max_horizon=arguments.timesfm_max_horizon,
                timesfm_cache_dir=arguments.timesfm_cache_dir,
                timesfm_local_files_only=arguments.timesfm_local_files_only,
                allow_statistical_fallback=arguments.allow_statistical_fallback,
                codex_binary=arguments.codex_binary,
                codex_model=arguments.codex_model,
                codex_cache_dir=arguments.codex_cache_dir,
                codex_timeout_seconds=arguments.codex_timeout,
                codex_reasoning_effort=codex_reasoning_effort,
            )
        )
    elif arguments.system == "codex-contract":
        system = CodexContractSystem(
            CodexDirectConfig(
                num_samples=arguments.samples,
                seed=arguments.seed,
                backbone=arguments.backbone,
                chronos_model_id=arguments.chronos_model_id,
                chronos_device_map=arguments.chronos_device_map,
                chronos_max_context=arguments.chronos_max_context,
                chronos_max_horizon=arguments.chronos_max_horizon,
                chronos_cache_dir=arguments.chronos_cache_dir,
                chronos_local_files_only=arguments.chronos_local_files_only,
                timesfm_model_id=arguments.timesfm_model_id,
                timesfm_max_context=arguments.timesfm_max_context,
                timesfm_max_horizon=arguments.timesfm_max_horizon,
                timesfm_cache_dir=arguments.timesfm_cache_dir,
                timesfm_local_files_only=arguments.timesfm_local_files_only,
                allow_statistical_fallback=arguments.allow_statistical_fallback,
                codex_binary=arguments.codex_binary,
                codex_model=arguments.codex_model,
                codex_cache_dir=arguments.codex_cache_dir,
                codex_timeout_seconds=arguments.codex_timeout,
                codex_reasoning_effort=codex_reasoning_effort,
            )
        )
    else:
        system = IterativeAgentSystem(
            LoopConfig(
                max_steps=arguments.max_steps,
                documents_per_step=arguments.top_k,
                num_samples=arguments.samples,
                context_weight=arguments.context_weight,
                max_no_progress=arguments.max_no_progress,
                convergence_tolerance=arguments.convergence_tolerance,
                seed=arguments.seed,
                memory_path=arguments.memory_file,
                learn_from_public_outcomes=arguments.learn_from_public_outcomes,
                retrieval_candidate_multiplier=arguments.candidate_multiplier,
                context_character_budget=arguments.context_character_budget,
                revision_utility_threshold=arguments.revision_threshold,
                min_expected_information_gain=arguments.min_information_gain,
                backbone=arguments.backbone,
                chronos_model_id=arguments.chronos_model_id,
                chronos_device_map=arguments.chronos_device_map,
                chronos_max_context=arguments.chronos_max_context,
                chronos_max_horizon=arguments.chronos_max_horizon,
                chronos_cache_dir=arguments.chronos_cache_dir,
                chronos_local_files_only=arguments.chronos_local_files_only,
                timesfm_model_id=arguments.timesfm_model_id,
                timesfm_max_context=arguments.timesfm_max_context,
                timesfm_max_horizon=arguments.timesfm_max_horizon,
                timesfm_cache_dir=arguments.timesfm_cache_dir,
                timesfm_local_files_only=arguments.timesfm_local_files_only,
                allow_statistical_fallback=arguments.allow_statistical_fallback,
                oracle_evidence=arguments.oracle_evidence,
                allow_unvalidated_event_revisions=(
                    arguments.allow_unvalidated_event_revisions
                ),
                reasoning_agent=arguments.reasoning_agent,
                codex_stages=codex_stages,
                codex_binary=arguments.codex_binary,
                codex_model=arguments.codex_model,
                codex_cache_dir=arguments.codex_cache_dir,
                codex_timeout_seconds=arguments.codex_timeout,
                codex_max_document_characters=(
                    arguments.codex_max_document_characters
                ),
                codex_reasoning_effort=codex_reasoning_effort,
            )
        )
    results = system.run_many(tasks)
    write_outputs(results, arguments.output_dir)
    summary_path = Path(arguments.output_dir).expanduser().resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Completed {len(results)} task(s).")
    print(f"Outputs: {summary_path.parent}")
    for name, value in summary["mean_metrics"].items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
