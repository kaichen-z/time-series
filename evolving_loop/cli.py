"""CLI for one-pass harness evaluation and held-out co-evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    CoEvolutionEngine,
    HarnessPolicy,
    evaluation_diagnostics,
)
from evolving_loop.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import (
    ContextTask,
    DEFAULT_TASKS_FILE,
    _to_context_task,
    load_context_tasks,
    load_huggingface_context_tasks,
)
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.frozen_inference import inference_view, run_frozen_inference
from evolving_loop.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_loop.morphology_adapter import MorphologyProvider
from evolving_loop.retrieval_agent.agent import RetrievalAgent
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    RetrievalPolicyError,
    RetrievalRelease,
    _load_retrieval_release_for_operator,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkill,
    RetrievalSkillLibrary,
    _load_verified_checkpoint_for_operator,
    _move_artifact_entry_to_quarantine,
    _rename_artifact_entry_noreplace,
    _restore_quarantined_artifact_entry,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.skill_learning import OutcomeSkillLearner
from evolving_loop.source_evolution import (
    SourceEvaluation,
    SourceEvolutionConfig,
    SourceEvolutionEngine,
    save_source_trace,
)
from evolving_loop.source_evolution.source_inference import run_source_inference
from common.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig, QwenClient
from common.metrics import linear_quantile
from evolving_loop.evaluation import score_after_resolution
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvaluation,
    RetrievalEvolutionConfig,
    RetrievalEvolutionEngine,
    RetrievalEvolutionError,
    RetrievalEvolutionResult,
    RetrievalForecastingFailure,
    RetrievalGenerationTrace,
    _open_checkpoint_parent,
    _open_retrieval_checkpoint_authority_for_operator,
    _revalidate_checkpoint_parent,
    _unique_checkpoint_temporary,
)
from common.tsfm import ChronosConfig, ChronosForecaster

BASELINE_CHOICES = (
    "skill-fresh",
    "skill-library",
    "chronos",
    "timesfm",
    "statistical",
    "one-pass",
    "iterative",
    "iterative-unsafe",
    "oracle-context",
    "rules-triad",
    "codex-triad",
    "codex-direct",
    "codex-contract",
    "evolving-harness",
)
EVOLUTION_CHOICES = ("prompt", "genome", "source", "retrieval")
INFERENCE_CHOICES = EVOLUTION_CHOICES
DRCIK_PUBLIC_80_20_99_SHA256 = (
    "3cc81f45878c1aae93e5ba48dc367df6553698db6661dbe06fbe5efb06afca92"
)
RETRIEVAL_CHECKPOINT_AUTHORITY_KEY_ENV = (
    "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY"
)
RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED_ENV = (
    "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED"
)
_FROZEN_RETRIEVAL_MANIFEST_METADATA = {
    "dataset": "ServiceNow/Dr-CiK",
    "source_split": "public_dev",
    "seed": 20260816,
    "grouping": "entity_disjoint",
    "stratification_features": [
        "frequency",
        "horizon_bin",
        "reasoning_hops",
        "origin",
    ],
}


class _RetrievalDefaultsParser(argparse.ArgumentParser):
    """Apply mode-dependent defaults after root and legacy parsing converge."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        retrieval = any(
            getattr(parsed, field, None) == "retrieval"
            for field in ("evolution", "inference", "evolution_mode")
        )
        defaults = {
            "retrieval_mode": "two-stage" if retrieval else "single-pass",
            "retrieval_release_path": (
                "evolving_loop/retrieval_agent/releases/v000"
                if retrieval
                else None
            ),
            "screen_train_tasks": 8 if retrieval else 6,
            "screen_promote": 2 if retrieval else 1,
            "screen_dev_tasks": 2,
        }
        for field, value in defaults.items():
            if hasattr(parsed, field) and getattr(parsed, field) is None:
                setattr(parsed, field, value)
        return parsed


def _add_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample-dir", help="Path to the Dr-CiK sample directory.")
    source.add_argument(
        "--public-dev",
        action="store_true",
        help="Run the Hugging Face public-development split.",
    )
    source.add_argument(
        "--hidden-test",
        action="store_true",
        help="Run the Hugging Face hidden-test split.",
    )


def _add_unified_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    _add_data_source_arguments(parser)
    parser.add_argument("--task-id", action="append", help="Repeatable benchmark_id filter.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Approved root containing a frozen-inference output directory.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--context-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backbone", choices=("chronos", "timesfm", "statistical"), default="chronos")
    parser.add_argument("--chronos-model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--chronos-device-map", default="cpu")
    parser.add_argument("--chronos-cache-dir", default=None)
    parser.add_argument("--chronos-local-files-only", action="store_true")
    parser.add_argument("--timesfm-model-id", default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--timesfm-cache-dir", default=None)
    parser.add_argument("--timesfm-local-files-only", action="store_true")
    parser.add_argument("--allow-statistical-fallback", action="store_true")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", choices=("none", "low", "medium", "high"), default=None)
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument("--codex-timeout", type=int, default=None)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--claude-cache-dir", default=None)
    parser.add_argument("--claude-timeout", type=int, default=None)


def _add_successive_halving_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--successive-halving",
        action="store_true",
        help="Screen all children on a small Train subset before full evaluation.",
    )
    parser.add_argument("--screen-train-tasks", type=int, default=None)
    parser.add_argument(
        "--screen-dev-tasks",
        type=int,
        default=None,
        help="Deprecated compatibility option; Dev is never used for screening.",
    )
    parser.add_argument("--screen-promote", type=int, default=None)
    parser.add_argument("--screen-tolerance", type=float, default=0.01)


def _add_retrieval_topology_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("single-pass", "two-stage"),
        default=None,
    )
    parser.add_argument("--retrieval-release-path", default=None)


def _add_retrieval_evolution_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-root",
        default=None,
        help="Approved root containing every Retrieval evolution output artifact.",
    )
    parser.add_argument(
        "--split-manifest-sha256",
        default=DRCIK_PUBLIC_80_20_99_SHA256,
    )
    parser.add_argument("--verifier-sha256", default=None)
    parser.add_argument("--evaluator-sha256", default=None)
    parser.add_argument("--metric-sha256", default=None)
    parser.add_argument("--mutation-model-sha256", default=None)
    parser.add_argument("--harness-sha256", default=None)
    parser.add_argument("--metric-cap", type=float, default=5.0)
    parser.add_argument("--train-folds", type=int, default=5)
    parser.add_argument("--evolution-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--checkpoint-authority-path",
        default=None,
        help="Out-of-band trusted digest/epoch record used for checkpoint resume.",
    )
    parser.add_argument(
        "--checkpoint-authority-head-path",
        default=None,
        help="Protected monotonic head for the authenticated checkpoint journal.",
    )
    parser.add_argument(
        "--checkpoint-authority-anchor-path",
        default=None,
        help=(
            "Operator-protected append-only monotonic anchor ledger outside "
            "the Retrieval run tree."
        ),
    )
    parser.add_argument(
        "--checkpoint-authority-key-env",
        default=RETRIEVAL_CHECKPOINT_AUTHORITY_KEY_ENV,
        help="Environment variable holding the operator authority key.",
    )
    parser.add_argument(
        "--checkpoint-authority-expected-env",
        default=RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED_ENV,
        help=(
            "Environment variable holding the independently retained "
            "expected authority epoch:head anchor for resume."
        ),
    )


def _add_unified_evolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--setting", choices=("llm_only", "statistics", "tsfm", "combined"), default="statistics")
    parser.add_argument("--llm-backend", choices=("codex", "qwen", "claude"), default="codex")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--coding-initial-programs", type=int, default=3)
    parser.add_argument("--coding-mutations", type=int, default=1)
    parser.add_argument("--coding-validation-folds", type=int, default=3)
    parser.add_argument(
        "--setting2-knowledge",
        action="store_true",
        help="Add diagnostic-selected source-backed forecasting rules to statistics/combined Coding.",
    )
    parser.add_argument("--seed-policy-path", default=None)
    parser.add_argument("--library-path", default="runs/evolving/skills.json")
    parser.add_argument("--retrieval-library-path", default="runs/evolving/retrieval_skills.json")
    parser.add_argument("--decision-library-path", default="runs/evolving/decision_skills.json")
    _add_retrieval_topology_arguments(parser)
    parser.add_argument("--chronos-device", default="cpu")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--children", type=int, default=2)
    _add_successive_halving_arguments(parser)
    parser.add_argument(
        "--evolve-target",
        choices=("auto", "coding", "retrieval", "decision"),
        default="auto",
        help="Restrict prompt/genome mutations to one role; auto diagnoses the weakest role.",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
        help="Entity-disjoint public holdout reserved from all evolution decisions.",
    )
    parser.add_argument(
        "--split-manifest-path",
        default="runs/evolving/split_manifest.json",
        help="Reproducible train/dev/holdout task manifest written by evolution.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Frozen-inference manifest used with --split-name.",
    )
    parser.add_argument(
        "--split-name",
        choices=("all", "train", "dev", "holdout"),
        default="all",
    )
    parser.add_argument(
        "--score-public",
        action="store_true",
        help="Score a labeled frozen run; forbidden if any selected task is hidden.",
    )
    parser.add_argument("--policy-path", default="runs/evolving/best_policy.json")
    parser.add_argument("--trace-path", default="runs/evolving/evolution_trace.json")
    parser.add_argument("--source-patch-path", default="runs/evolving/best_source.patch")
    parser.add_argument("--seed-source-patch", default=None)
    parser.add_argument("--source-engineer-timeout", type=int, default=1800)
    parser.add_argument("--source-test-timeout", type=int, default=300)
    parser.add_argument("--source-eval-timeout", type=int, default=7200)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing evolution checkpoint and start from the seed artifact.",
    )
    _add_retrieval_evolution_controls(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = _RetrievalDefaultsParser(
        description=(
            "Run contextual forecasting baselines and self-evolution modes from one entrypoint. "
            "Use --baseline NAME or --evolution NAME for the unified interface; legacy run/evolve "
            "subcommands remain supported."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", choices=BASELINE_CHOICES, metavar="NAME")
    mode.add_argument("--evolution", choices=EVOLUTION_CHOICES, metavar="NAME")
    mode.add_argument(
        "--inference",
        choices=INFERENCE_CHOICES,
        metavar="NAME",
        help="Run one frozen prompt/genome/source artifact without learning.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print all unified baseline and evolution names, then exit.",
    )

    # The unified options live on the root parser. argparse still permits the
    # older subcommand grammar because all legacy options remain below.
    _add_unified_baseline_arguments(parser)
    _add_unified_evolution_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "evolve"):
        child = subparsers.add_parser(command)
        child.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
        child.add_argument(
            "--setting",
            choices=("llm_only", "statistics", "tsfm", "combined"),
            default="statistics",
        )
        child.add_argument("--model-id", default=None)
        child.add_argument("--device", default=None)
        child.add_argument(
            "--llm-backend",
            choices=("codex", "qwen", "claude"),
            default="codex",
            help="LLM used by all three agents; defaults to this machine's Codex CLI.",
        )
        child.add_argument("--codex-model", default=None)
        child.add_argument("--codex-reasoning-effort", default="high")
        child.add_argument("--codex-timeout", type=int, default=900)
        child.add_argument("--codex-cache-dir", default="runs/evolving/codex-cache")
        child.add_argument("--claude-model", default=None)
        child.add_argument("--claude-timeout", type=int, default=900)
        child.add_argument("--claude-cache-dir", default="runs/evolving/claude-cache")
        child.add_argument("--coding-initial-programs", type=int, default=3)
        child.add_argument("--coding-mutations", type=int, default=1)
        child.add_argument("--coding-validation-folds", type=int, default=3)
        child.add_argument(
            "--setting2-knowledge",
            action="store_true",
            help="Add diagnostic-selected source-backed forecasting rules to statistics/combined Coding.",
        )
        child.add_argument(
            "--seed-policy-path",
            default=None,
            help="Load a previously accepted Harness Genome and continue from it.",
        )
        child.add_argument("--seed", type=int, default=7)
        child.add_argument("--limit", type=int, default=None)
        child.add_argument("--library-path", default="runs/evolving/skills.json")
        child.add_argument(
            "--retrieval-library-path",
            default="runs/evolving/retrieval_skills.json",
        )
        child.add_argument(
            "--decision-library-path",
            default="runs/evolving/decision_skills.json",
        )
        _add_retrieval_topology_arguments(child)
        child.add_argument("--chronos-model-id", default="amazon/chronos-bolt-small")
        child.add_argument("--chronos-device", default="cpu")
        child.add_argument("--chronos-cache-dir", default=None)
        child.add_argument("--chronos-local-files-only", action="store_true")
    run = subparsers.choices["run"]
    run.add_argument("--results-path", default="runs/evolving/harness_results.jsonl")
    run.add_argument(
        "--learn-from-public-outcomes",
        action="store_true",
        help="After each public label resolves, generate/update all three skill libraries.",
    )
    evolve = subparsers.choices["evolve"]
    evolve.add_argument("--generations", type=int, default=3)
    evolve.add_argument("--children", type=int, default=2)
    _add_successive_halving_arguments(evolve)
    evolve.add_argument(
        "--evolve-target",
        choices=("auto", "coding", "retrieval", "decision"),
        default="auto",
        help="Restrict prompt/genome mutations to one role; auto diagnoses the weakest role.",
    )
    evolve.add_argument(
        "--evolution-mode",
        choices=("prompt", "genome", "source", "retrieval"),
        default="genome",
        help=(
            "prompt changes one role prompt; genome co-evolves prompts/budgets/topology; "
            "source edits agent/orchestration Python in isolated Git worktrees."
        ),
    )
    evolve.add_argument("--dev-fraction", type=float, default=0.25)
    evolve.add_argument("--holdout-fraction", type=float, default=0.20)
    evolve.add_argument(
        "--split-manifest",
        default=None,
        help="Immutable Train/Dev/Public Regression manifest for Retrieval evolution.",
    )
    evolve.add_argument(
        "--split-manifest-path", default="runs/evolving/split_manifest.json"
    )
    evolve.add_argument("--policy-path", default="runs/evolving/best_policy.json")
    evolve.add_argument("--trace-path", default="runs/evolving/evolution_trace.json")
    evolve.add_argument(
        "--source-patch-path",
        default="runs/evolving/best_source.patch",
        help="Accepted cumulative source patch for source evolution mode.",
    )
    evolve.add_argument(
        "--seed-source-patch",
        default=None,
        help="Previously accepted source patch from which source evolution continues.",
    )
    evolve.add_argument("--source-engineer-timeout", type=int, default=1800)
    evolve.add_argument("--source-test-timeout", type=int, default=300)
    evolve.add_argument("--source-eval-timeout", type=int, default=7200)
    evolve.add_argument("--checkpoint-path", default=None)
    evolve.add_argument("--progress-path", default=None)
    evolve.add_argument("--no-resume", action="store_true")
    _add_retrieval_evolution_controls(evolve)
    # Root-mode invocations have no subcommand, while legacy calls require one.
    subparsers.required = False
    return parser


def _task_subset(tasks: list[ContextTask], seed: int, limit: int | None) -> list[ContextTask]:
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    return tasks if limit is None else tasks[:limit]


def _baseline_argv(args: argparse.Namespace) -> list[str]:
    """Translate one friendly baseline name into the established Dr-CiK runner."""
    if args.sample_dir:
        argv = ["run-sample", "--sample-dir", args.sample_dir]
    elif args.public_dev:
        argv = ["run-hf", "--public-dev"]
    elif args.hidden_test:
        argv = ["run-hf", "--hidden-test"]
    else:
        raise SystemExit(
            "A baseline requires one data source: --sample-dir, --public-dev, or --hidden-test"
        )

    system = {
        "chronos": "backbone-only",
        "timesfm": "backbone-only",
        "statistical": "backbone-only",
        "one-pass": "one-pass",
        "iterative": "iterative",
        "iterative-unsafe": "iterative",
        "oracle-context": "iterative",
        "rules-triad": "triad",
        "codex-triad": "triad",
        "codex-direct": "codex-direct",
        "codex-contract": "codex-contract",
    }.get(args.baseline)
    if system is None:
        raise SystemExit(
            "--baseline evolving-harness uses --tasks-file or --sample-dir and is routed internally"
        )
    backbone = {
        "chronos": "chronos",
        "timesfm": "timesfm",
        "statistical": "statistical",
    }.get(args.baseline, args.backbone)
    output_dir = args.output_dir or f"outputs/baselines/{args.baseline}"
    argv.extend(
        [
            "--system", system,
            "--backbone", backbone,
            "--output-dir", output_dir,
            "--top-k", str(args.top_k),
            "--max-steps", str(args.max_steps),
            "--samples", str(args.samples),
            "--context-weight", str(args.context_weight),
            "--seed", str(args.seed),
            "--chronos-model-id", args.chronos_model_id,
            "--chronos-device-map", args.chronos_device_map,
            "--timesfm-model-id", args.timesfm_model_id,
            "--codex-cache-dir", args.codex_cache_dir or "outputs/codex-cache",
            "--codex-timeout", str(args.codex_timeout or 180),
        ]
    )
    for task_id in args.task_id or ():
        argv.extend(("--task-id", task_id))
    if args.limit is not None:
        argv.extend(("--limit", str(args.limit)))
    if args.chronos_cache_dir:
        argv.extend(("--chronos-cache-dir", args.chronos_cache_dir))
    if args.chronos_local_files_only:
        argv.append("--chronos-local-files-only")
    if args.timesfm_cache_dir:
        argv.extend(("--timesfm-cache-dir", args.timesfm_cache_dir))
    if args.timesfm_local_files_only:
        argv.append("--timesfm-local-files-only")
    if args.allow_statistical_fallback:
        argv.append("--allow-statistical-fallback")
    if args.codex_model:
        argv.extend(("--codex-model", args.codex_model))
    if args.codex_reasoning_effort:
        argv.extend(("--codex-reasoning-effort", args.codex_reasoning_effort))
    if args.baseline == "codex-triad":
        argv.extend(("--reasoning-agent", "codex"))
    elif args.baseline == "rules-triad":
        argv.extend(("--reasoning-agent", "rules"))
    if args.baseline == "iterative-unsafe":
        argv.append("--allow-unvalidated-event-revisions")
    if args.baseline == "oracle-context":
        argv.append("--oracle-evidence")
    return argv


def baseline_command(args: argparse.Namespace) -> dict | None:
    """Run one named baseline without duplicating the established implementations."""
    if args.baseline in {"skill-fresh", "skill-library"}:
        if args.sample_dir:
            raise SystemExit(
                f"{args.baseline} consumes a JSONL numeric task file; use --tasks-file, not --sample-dir"
            )
        if args.public_dev or args.hidden_test:
            raise SystemExit(
                f"{args.baseline} consumes --tasks-file and does not directly load a Hugging Face split"
            )
        from evolving_loop.coding_agent.baseline import main as coding_baseline_main

        mode = "fresh" if args.baseline == "skill-fresh" else "library"
        output_dir = Path(args.output_dir or f"outputs/baselines/{args.baseline}")
        argv = [
            "--mode", mode,
            "--tasks-file", args.tasks_file,
            "--results-path", str(output_dir / "results.jsonl"),
            "--log-file", str(output_dir / "run.log"),
            "--seed", str(args.seed),
            "--library-path", args.library_path,
        ]
        if args.limit is not None:
            argv.extend(("--limit", str(args.limit)))
        if args.model_id:
            argv.extend(("--model-id", args.model_id))
        if args.device:
            argv.extend(("--device", args.device))
        coding_baseline_main(argv)
        return None
    if args.baseline == "evolving-harness":
        if args.public_dev or args.hidden_test:
            raise SystemExit(
                "evolving-harness currently requires --tasks-file or --sample-dir, not a Hugging Face split"
            )
        if args.sample_dir:
            sample_path = Path(args.sample_dir)
            task_directory = sample_path / "tasks"
            args.tasks_file = str(task_directory if task_directory.is_dir() else sample_path)
        args.learn_from_public_outcomes = False
        args.codex_reasoning_effort = args.codex_reasoning_effort or "high"
        args.codex_timeout = args.codex_timeout or 900
        args.codex_cache_dir = args.codex_cache_dir or "runs/evolving/codex-cache"
        args.claude_timeout = args.claude_timeout or 900
        args.claude_cache_dir = args.claude_cache_dir or "runs/evolving/claude-cache"
        args.results_path = str(
            Path(args.output_dir or "outputs/baselines/evolving-harness") / "results.jsonl"
        )
        return run_command(args)
    from drcik_agent.cli import main as drcik_main

    drcik_main(_baseline_argv(args))
    return None


def _entity_split(
    tasks: list[ContextTask], seed: int, dev_fraction: float
) -> tuple[list[ContextTask], list[ContextTask]]:
    entities = sorted({task.numeric.entity_name for task in tasks})
    random.Random(seed).shuffle(entities)
    count = max(1, round(len(entities) * dev_fraction))
    dev_entities = set(entities[:count])
    return (
        [task for task in tasks if task.numeric.entity_name not in dev_entities],
        [task for task in tasks if task.numeric.entity_name in dev_entities],
    )


def _three_way_entity_split(
    tasks: list[ContextTask],
    seed: int,
    dev_fraction: float,
    holdout_fraction: float,
) -> tuple[list[ContextTask], list[ContextTask], list[ContextTask]]:
    """Create deterministic entity-disjoint train/dev/holdout partitions."""
    if not 0 < dev_fraction < 1:
        raise ValueError("--dev-fraction must be between 0 and 1")
    if not 0 < holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be between 0 and 1")
    if dev_fraction + holdout_fraction >= 1:
        raise ValueError("dev and holdout fractions must sum to less than 1")
    entities = sorted({task.numeric.entity_name for task in tasks})
    required = 3
    if len(entities) < required:
        raise ValueError(
            f"entity split needs at least {required} distinct entities; got {len(entities)}"
        )
    random.Random(seed).shuffle(entities)
    dev_count = max(1, round(len(entities) * dev_fraction))
    holdout_count = max(1, round(len(entities) * holdout_fraction))
    while dev_count + holdout_count >= len(entities):
        if dev_count >= holdout_count and dev_count > 1:
            dev_count -= 1
        elif holdout_count > 1:
            holdout_count -= 1
        else:
            raise ValueError("entity split cannot keep train/dev/holdout non-empty")
    holdout_entities = set(entities[:holdout_count])
    dev_entities = set(entities[holdout_count : holdout_count + dev_count])
    train, dev, holdout = [], [], []
    for task in tasks:
        entity = task.numeric.entity_name
        if entity in holdout_entities:
            holdout.append(task)
        elif entity in dev_entities:
            dev.append(task)
        else:
            train.append(task)
    return train, dev, holdout


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_string_token(raw: str, start: int) -> tuple[str, int]:
    if start >= len(raw) or raw[start] != '"':
        raise ValueError("Retrieval dataset record has invalid benchmark metadata")
    escaped = False
    for index in range(start + 1, len(raw)):
        character = raw[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            try:
                value = json.loads(raw[start : index + 1])
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Retrieval dataset record has invalid benchmark metadata"
                ) from error
            if not isinstance(value, str):
                raise ValueError(
                    "Retrieval dataset record has invalid benchmark metadata"
                )
            return value, index + 1
    raise ValueError("Retrieval dataset record has invalid benchmark metadata")


def _retrieval_task_record_id(raw: str) -> str:
    """Extract only the top-level identifier without decoding any label value."""
    index = 0
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw) or raw[index] != "{":
        raise ValueError("Retrieval dataset record has invalid benchmark metadata")
    stack: list[str] = []
    matches: list[str] = []
    while index < len(raw):
        character = raw[index]
        if character.isspace():
            index += 1
            continue
        if character in "{[":
            stack.append(character)
            index += 1
            continue
        if character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack.pop() != expected:
                raise ValueError(
                    "Retrieval dataset record has invalid benchmark metadata"
                )
            index += 1
            if not stack:
                if raw[index:].strip():
                    raise ValueError(
                        "Retrieval dataset record has invalid benchmark metadata"
                    )
                break
            continue
        if character != '"':
            index += 1
            continue
        token, end = _json_string_token(raw, index)
        after = end
        while after < len(raw) and raw[after].isspace():
            after += 1
        if stack == ["{"] and after < len(raw) and raw[after] == ":":
            if token == "benchmark_id":
                value_start = after + 1
                while value_start < len(raw) and raw[value_start].isspace():
                    value_start += 1
                task_id, value_end = _json_string_token(raw, value_start)
                matches.append(task_id)
                index = value_end
                continue
        index = end
    if stack or len(matches) != 1 or not matches[0]:
        raise ValueError("Retrieval dataset record has invalid benchmark metadata")
    return matches[0]


def _open_retrieval_task_source(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Retrieval dataset source cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Retrieval dataset source must be a regular file")
        if (
            expected_identity is not None
            and (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            raise ValueError(
                "Retrieval dataset source identity changed after preflight"
            )
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def _open_retrieval_task_member(
    directory_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError("Retrieval dataset member cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            raise ValueError(
                "Retrieval dataset member identity changed after preflight"
            )
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def _snapshot_retrieval_task_source(
    path: Path,
) -> dict[str, object] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError("Retrieval dataset source cannot be inspected safely") from error
    try:
        metadata = os.fstat(descriptor)
        source_identity = (metadata.st_dev, metadata.st_ino)
        if stat.S_ISREG(metadata.st_mode):
            return {
                "kind": "file",
                "source_identity": source_identity,
                "members": (),
            }
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Retrieval dataset source must be a file or directory")
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise ValueError(
                "Retrieval dataset directory cannot be enumerated safely"
            ) from error
        members: list[tuple[str, tuple[int, int]]] = []
        member_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for name in names:
            if name in {"", ".", ".."} or Path(name).name != name:
                raise ValueError("Retrieval dataset member name is invalid")
            try:
                member_descriptor = os.open(
                    name, member_flags, dir_fd=descriptor
                )
            except OSError as error:
                raise ValueError(
                    "Retrieval dataset member cannot be inspected safely"
                ) from error
            try:
                member_metadata = os.fstat(member_descriptor)
                if not stat.S_ISREG(member_metadata.st_mode):
                    raise ValueError(
                        "Retrieval dataset member must be a regular file"
                    )
                members.append(
                    (
                        name,
                        (member_metadata.st_dev, member_metadata.st_ino),
                    )
                )
            finally:
                os.close(member_descriptor)
        return {
            "kind": "directory",
            "source_identity": source_identity,
            "members": tuple(members),
        }
    finally:
        os.close(descriptor)


def _iter_retrieval_task_record_texts(
    tasks_file: str | Path,
    *,
    expected_source_snapshot: Mapping[str, object] | None = None,
):
    """Yield one raw task record at a time while refusing source symlinks."""
    source = Path(tasks_file)
    snapshot = _snapshot_retrieval_task_source(source)
    if snapshot is None:
        raise ValueError("Retrieval dataset source is unavailable")
    if expected_source_snapshot is not None and snapshot != expected_source_snapshot:
        raise ValueError("Retrieval dataset source identity changed after preflight")
    if snapshot["kind"] == "directory":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            directory_descriptor = os.open(source, directory_flags)
        except OSError as error:
            raise ValueError(
                "Retrieval dataset directory cannot be opened safely"
            ) from error
        try:
            metadata = os.fstat(directory_descriptor)
            if (metadata.st_dev, metadata.st_ino) != snapshot["source_identity"]:
                raise ValueError(
                    "Retrieval dataset directory identity changed after preflight"
                )
            for name, identity in snapshot["members"]:
                if Path(name).suffix.lower() != ".json":
                    continue
                with _open_retrieval_task_member(
                    directory_descriptor,
                    name,
                    expected_identity=identity,
                ) as handle:
                    raw = handle.read()
                if raw.strip():
                    yield raw
        finally:
            os.close(directory_descriptor)
        return
    with _open_retrieval_task_source(
        source,
        expected_identity=snapshot["source_identity"],
    ) as handle:
        for line in handle:
            if line.strip():
                yield line


def _load_retrieval_evolution_tasks(
    tasks_file: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    include_public_ids: bool = False,
    expected_source_snapshot: Mapping[str, object] | None = None,
) -> (
    tuple[tuple[ContextTask, ...], tuple[ContextTask, ...], str]
    | tuple[
        tuple[ContextTask, ...],
        tuple[ContextTask, ...],
        str,
        frozenset[str],
    ]
):
    """Authenticate the frozen split before loading exactly its Train and Dev tasks."""
    source = Path(manifest_path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Retrieval split manifest: {source}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Retrieval split manifest must be an object")
    internal_sha256 = manifest.get("manifest_sha256")
    if not isinstance(internal_sha256, str) or len(internal_sha256) != 64:
        raise ValueError("Retrieval split manifest requires manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_sha256 = _canonical_sha256(unsigned)
    if internal_sha256 != actual_sha256:
        raise ValueError("Retrieval split manifest sha256 mismatch")
    if actual_sha256 != DRCIK_PUBLIC_80_20_99_SHA256:
        raise ValueError("Retrieval split manifest does not match the pinned frozen hash")
    if (
        expected_manifest_sha256 is not None
        and expected_manifest_sha256 != DRCIK_PUBLIC_80_20_99_SHA256
    ):
        raise ValueError("Retrieval split manifest does not match the frozen hash")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Retrieval split manifest schema")
    expected_manifest_fields = {
        "schema_version",
        "dataset",
        "source_split",
        "seed",
        "grouping",
        "stratification_features",
        "selection_uses_future_values",
        "selection_uses_gt_evidence",
        "selection_uses_document_labels",
        "target_sizes",
        "actual_sizes",
        "partitions",
        "manifest_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("Retrieval split manifest metadata is not frozen")
    for field, expected in _FROZEN_RETRIEVAL_MANIFEST_METADATA.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Retrieval split manifest has invalid frozen {field}")
    for flag in (
        "selection_uses_future_values",
        "selection_uses_gt_evidence",
        "selection_uses_document_labels",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"Retrieval split manifest has unsafe {flag}")
    expected_sizes = {"train": 80, "dev": 20, "public_test": 99}
    if manifest.get("target_sizes") != expected_sizes:
        raise ValueError("Retrieval split manifest target sizes are not frozen")
    if manifest.get("actual_sizes") != expected_sizes:
        raise ValueError("Retrieval split manifest actual sizes are not frozen")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(expected_sizes):
        raise ValueError("Retrieval split manifest partitions are invalid")
    partition_ids: dict[str, tuple[str, ...]] = {}
    all_ids: set[str] = set()
    for name, size in expected_sizes.items():
        partition = partitions.get(name)
        if not isinstance(partition, dict):
            raise ValueError(f"Retrieval split manifest {name} partition is invalid")
        raw_ids = partition.get("task_ids")
        if (
            not isinstance(raw_ids, list)
            or len(raw_ids) != size
            or any(not isinstance(task_id, str) or not task_id for task_id in raw_ids)
            or len(set(raw_ids)) != size
        ):
            raise ValueError(f"Retrieval split manifest {name} task IDs are invalid")
        ids = tuple(raw_ids)
        if all_ids.intersection(ids):
            raise ValueError("Retrieval split manifest partitions overlap")
        partition_ids[name] = ids
        all_ids.update(ids)

    # Authenticate completeness using only record IDs. Public Regression rows are
    # never decoded, so their future values, GT evidence, and document roles cannot
    # become ContextTask objects or reach the evolution engine.
    selected_ids = set((*partition_ids["train"], *partition_ids["dev"]))
    available_ids: set[str] = set()
    selected_records: dict[str, str] = {}
    for raw in _iter_retrieval_task_record_texts(
        tasks_file,
        expected_source_snapshot=expected_source_snapshot,
    ):
        task_id = _retrieval_task_record_id(raw)
        if task_id in available_ids:
            raise ValueError("Retrieval dataset contains duplicate task metadata")
        available_ids.add(task_id)
        if task_id in selected_ids:
            selected_records[task_id] = raw
    if available_ids != all_ids:
        raise ValueError("Retrieval dataset is incomplete or does not match the frozen manifest")

    available: dict[str, ContextTask] = {}
    for task_id, raw in selected_records.items():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("selected Retrieval dataset record is invalid") from error
        if not isinstance(record, dict) or record.get("benchmark_id") != task_id:
            raise ValueError("selected Retrieval dataset metadata changed during loading")
        task = _to_context_task(record)
        if task.numeric.task_id != task_id:
            raise ValueError("selected Retrieval task identity is invalid")
        available[task_id] = task

    def select(name: str) -> tuple[ContextTask, ...]:
        missing = [task_id for task_id in partition_ids[name] if task_id not in available]
        if missing:
            raise ValueError(f"Retrieval {name} split is incomplete")
        selected = tuple(available[task_id] for task_id in partition_ids[name])
        if any(not task.labels_public or not task.numeric.future_values for task in selected):
            raise ValueError(f"Retrieval {name} evaluation requires trusted public labels")
        return selected

    selected = (select("train"), select("dev"), actual_sha256)
    if include_public_ids:
        return (*selected, frozenset(partition_ids["public_test"]))
    return selected


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _policy_with_retrieval_release(
    policy: HarnessPolicy,
    release: RetrievalRelease,
    *,
    changelog: str,
) -> HarnessPolicy:
    """Embed an authenticated release snapshot without turning it into authority."""
    embedded = {
        "genome": release.genome.to_payload(),
        "round1_prompt": release.round1_prompt,
        "round2_prompt": release.round2_prompt,
        "skills": _plain_json(release.skills),
        "manifest": _plain_json(release.manifest),
    }
    digest = _canonical_sha256(embedded)
    return replace(
        policy,
        version=release.genome.version,
        parent=release.genome.parent,
        retrieval_prompt=release.round1_prompt,
        retrieval_skills=tuple(embedded["skills"]),
        changelog=changelog,
        retrieval_release_payload=embedded,
        retrieval_release_sha256=digest,
    )


def _require_authorized_retrieval_release_state(
    release: RetrievalRelease,
) -> None:
    version = release.genome.version
    state = release.manifest["state"]
    if not (
        (version == "v000" and state == "seed")
        or (version != "v000" and state == "accepted")
    ):
        raise ValueError(
            "Retrieval permits only the v000 seed or an authorized accepted release state"
        )


def _policy_for_retrieval_release(
    policy: HarnessPolicy,
    release: RetrievalRelease,
) -> HarnessPolicy:
    """Bind legacy v000 policies; require accepted policies to carry exact payloads."""
    _require_authorized_retrieval_release_state(release)
    expected = _policy_with_retrieval_release(
        policy, release, changelog=policy.changelog
    )
    if policy.retrieval_release_payload is None:
        if release.genome.version == "v000" and release.manifest["state"] == "seed":
            return expected
        raise ValueError(
            "accepted Retrieval release requires an embedded HarnessPolicy payload"
        )
    if (
        policy.retrieval_release_sha256 != expected.retrieval_release_sha256
        or policy.retrieval_release_payload != expected.retrieval_release_payload
    ):
        raise ValueError(
            "HarnessPolicy Retrieval release does not match the trusted operator release"
        )
    return policy


class _TrustedRetrievalEvaluator:
    """Run label-free harnesses first, then resolve metrics in the trusted host."""

    def evaluate(
        self,
        genome,
        tasks,
        *,
        stage,
        skill_library,
        harness_factory,
        persist,
        writers_enabled,
        evolver_enabled,
        cache_keys,
        metric_cap,
    ) -> RetrievalEvaluation:
        del stage
        if persist or writers_enabled or evolver_enabled:
            raise RetrievalEvolutionError(
                "trusted Retrieval evaluation forbids persistence, writers, and evolvers"
            )
        if metric_cap != 5.0:
            raise RetrievalEvolutionError(
                "trusted Retrieval evaluation uses the frozen Dr-CiK metric cap 5"
            )
        if not callable(harness_factory):
            raise RetrievalEvolutionError("trusted Retrieval evaluation requires a harness factory")
        task_ids = tuple(task.numeric.task_id for task in tasks)
        if tuple(getattr(key, "task_id", None) for key in cache_keys) != task_ids:
            raise RetrievalEvolutionError(
                "trusted Retrieval cache keys do not match exact task order"
            )

        outcomes = []
        traces: list[dict[str, object]] = []
        for task in tasks:
            try:
                harness = harness_factory(genome, skill_library)
                result = harness.run(
                    inference_view(task), allow_skill_writes=False
                )
            except Exception:
                raise RetrievalForecastingFailure(
                    "InferenceRuntimeFailure"
                ) from None
            if getattr(result, "task_id", task.numeric.task_id) != task.numeric.task_id:
                raise RetrievalForecastingFailure("InvalidHarnessResult") from None
            try:
                outcome = score_after_resolution(task, result)
                diagnostics = outcome.retrieval_diagnostics
            except Exception:
                raise RetrievalForecastingFailure(
                    "TrustedScoringFailure"
                ) from None
            if diagnostics is None:
                raise RetrievalForecastingFailure(
                    "MissingRetrievalDiagnostics",
                )
            metric_values = {
                "final_smae": outcome.final_smae,
                "final_srmse": outcome.final_srmse,
                "contextual_oracle_smae": outcome.contextual_oracle_smae,
                "contextual_oracle_srmse": outcome.contextual_oracle_srmse,
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in metric_values.values()
            ):
                raise RetrievalForecastingFailure(
                    "InvalidTrustedMetric"
                )
            outcomes.append((outcome, diagnostics))
            traces.append(
                {
                    "task_id": task.numeric.task_id,
                    "entity_name": task.numeric.entity_name,
                    **{key: float(value) for key, value in metric_values.items()},
                }
            )

        def mean_outcome(field: str) -> float:
            return statistics.fmean(float(getattr(outcome, field)) for outcome, _ in outcomes)

        def mean_diagnostic(field: str) -> float:
            return statistics.fmean(float(getattr(diagnostics, field)) for _, diagnostics in outcomes)

        final_smae = [float(trace["final_smae"]) for trace in traces]
        return RetrievalEvaluation(
            version=genome.version,
            task_count=len(tasks),
            mean_final_smae=mean_outcome("final_smae"),
            mean_final_srmse=mean_outcome("final_srmse"),
            mean_contextual_oracle_smae=mean_outcome("contextual_oracle_smae"),
            mean_contextual_oracle_srmse=mean_outcome("contextual_oracle_srmse"),
            p90_smae=linear_quantile(final_smae, 0.90),
            p95_smae=linear_quantile(final_smae, 0.95),
            supporting_recall=mean_diagnostic("supporting_recall"),
            distractor_avoidance=mean_diagnostic("distractor_avoidance"),
            exact_quote_validity=mean_diagnostic("exact_quote_validity"),
            complete_chain_rate=mean_diagnostic("complete_chain_rate"),
            invalid_count=sum(
                int(diagnostics.invalid_count) for _, diagnostics in outcomes
            ),
            catastrophic_count=sum(
                int(diagnostics.catastrophic_count) for _, diagnostics in outcomes
            ),
            task_traces=tuple(traces),
        )


class _ConservativeMorphologyProvider:
    """Safe deployment bridge until a Numerical Morphology provider is configured."""

    def assumptions(self, task: ContextTask) -> tuple[object, ...]:
        del task
        return ()


def _write_split_manifest(
    path: str | Path,
    *,
    seed: int,
    dev_fraction: float,
    holdout_fraction: float,
    train: list[ContextTask],
    dev: list[ContextTask],
    holdout: list[ContextTask],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def partition(tasks: list[ContextTask]) -> dict:
        return {
            "task_ids": [task.numeric.task_id for task in tasks],
            "entities": sorted({task.numeric.entity_name for task in tasks}),
        }

    payload = {
        "schema_version": 1,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "holdout_fraction": holdout_fraction,
        "train": partition(train),
        "dev": partition(dev),
        "holdout": partition(holdout),
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def _select_manifest_split(
    tasks: list[ContextTask], manifest_path: str | Path, split_name: str
) -> list[ContextTask]:
    if split_name == "all":
        return tasks
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    requested = set(payload[split_name]["task_ids"])
    selected = [task for task in tasks if task.numeric.task_id in requested]
    missing = requested - {task.numeric.task_id for task in selected}
    if missing:
        raise ValueError(
            f"split manifest references {len(missing)} unavailable task(s): "
            + ", ".join(sorted(missing)[:5])
        )
    return selected


class _ChronosNumericForecaster:
    """Adapts common.tsfm.ChronosForecaster to the NumericForecaster(history, horizon, frequency) protocol."""

    def __init__(self, forecaster: ChronosForecaster) -> None:
        self._forecaster = forecaster

    def forecast(self, history, horizon, frequency):
        del frequency  # Chronos consumes the numerical history directly.
        return self._forecaster.forecast(history, horizon)


def _components(
    args,
    *,
    retrieval_library_override: RetrievalSkillLibrary | None = None,
    disable_llm_cache: bool = False,
    llm_subprocess_env: Mapping[str, str] | None = None,
):
    if args.llm_backend == "codex":
        llm = CodexCLIClient(
            CodexCLIConfig(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning_effort,
                timeout_seconds=args.codex_timeout,
                cache_dir=None if disable_llm_cache else args.codex_cache_dir,
                subprocess_env=llm_subprocess_env,
            )
        )
    elif args.llm_backend == "claude":
        llm = ClaudeCLIClient(
            ClaudeCLIConfig(
                model=args.claude_model,
                timeout_seconds=args.claude_timeout,
                cache_dir=None if disable_llm_cache else args.claude_cache_dir,
                subprocess_env=llm_subprocess_env,
            )
        )
    else:
        kwargs = {}
        if args.model_id:
            kwargs["model_id"] = args.model_id
        if args.device:
            kwargs["device"] = args.device
        llm = QwenClient(**kwargs)
    library = SkillLibrary.load(args.library_path)
    retrieval_library = (
        retrieval_library_override
        if retrieval_library_override is not None
        else _load_verified_checkpoint_for_operator(args.retrieval_library_path)
    )
    decision_library = DecisionSkillLibrary.load(args.decision_library_path)
    tsfm = None
    if args.setting in {"tsfm", "combined"}:
        tsfm = _ChronosNumericForecaster(
            ChronosForecaster(
                ChronosConfig(
                    model_id=args.chronos_model_id,
                    device_map=args.chronos_device,
                    cache_dir=args.chronos_cache_dir,
                    local_files_only=args.chronos_local_files_only,
                )
            )
        )
    return llm, library, retrieval_library, decision_library, tsfm


def _seed_policy(args) -> HarnessPolicy:
    if args.seed_policy_path:
        path = Path(args.seed_policy_path)
        if not path.exists():
            raise FileNotFoundError(f"seed Harness Genome does not exist: {path}")
        return HarnessPolicy.load(path)
    return HarnessPolicy(
        coding_initial_programs=args.coding_initial_programs,
        coding_mutations=args.coding_mutations,
        coding_validation_folds=args.coding_validation_folds,
    )


def _factory(
    args,
    llm,
    library,
    retrieval_library,
    decision_library,
    tsfm,
    *,
    isolate_library: bool = False,
    morphology_provider: MorphologyProvider | None = None,
    retrieval_genome=None,
    retrieval_skill_source: RetrievalSkillLibrary | None = None,
):
    retrieval_mode = getattr(args, "retrieval_mode", "single-pass")
    if retrieval_mode not in {"single-pass", "two-stage"}:
        raise ValueError("retrieval_mode must be single-pass or two-stage")
    release = None
    fixed_genome = retrieval_genome
    release_library: RetrievalSkillLibrary | None = None
    if retrieval_mode == "two-stage":
        if not isinstance(morphology_provider, MorphologyProvider) or not callable(
            getattr(morphology_provider, "assumptions", None)
        ):
            raise ValueError("two-stage construction requires MorphologyProvider")
        if fixed_genome is not None:
            if retrieval_skill_source is None:
                raise ValueError(
                    "candidate two-stage construction requires a verified Retrieval Skill source"
                )
            release_library = retrieval_skill_source
        else:
            release_path = getattr(args, "retrieval_release_path", None)
            if not release_path:
                raise ValueError("two-stage construction requires --retrieval-release-path")
            release = _load_retrieval_release_for_operator(release_path)
            fixed_genome = release.genome
            release_library = RetrievalSkillLibrary._from_loaded_release(release)
        available_ids = {item.skill_id for item in release_library.all()}
        missing_ids = set(fixed_genome.active_skill_ids) - available_ids
        if missing_ids:
            raise ValueError(
                "retrieval release references unavailable active skills: "
                + ", ".join(sorted(missing_ids))
            )

    def build(policy: HarnessPolicy) -> EvolvingForecastHarness:
        coding_records = {skill.name: skill for skill in library.all()}
        coding_records.update(
            {record["name"]: Skill(**record) for record in policy.coding_skills}
        )
        runtime_skill_source = None
        if fixed_genome is None and policy.retrieval_skill_source is not None:
            if not isinstance(
                policy.retrieval_skill_source, RetrievalSkillLibrary
            ):
                raise ValueError("invalid runtime Retrieval Skill source")
            source_records = tuple(
                skill.to_payload()
                for skill in policy.retrieval_skill_source.all()
            )
            if tuple(policy.retrieval_skills) != source_records:
                raise ValueError(
                    "policy snapshot changed its verified Retrieval Skill source"
                )
            runtime_skill_source = policy.retrieval_skill_source
            retrieval_records = []
        elif fixed_genome is None:
            verified_by_payload = {
                json.dumps(skill.to_payload(), sort_keys=True): skill
                for skill in retrieval_library.all()
                if skill.is_active
            }
            policy_retrieval_records = []
            for record in policy.retrieval_skills:
                if record.get("status") in {"accepted", "specialized"}:
                    matched = verified_by_payload.get(json.dumps(record, sort_keys=True))
                    if matched is None:
                        raise ValueError(
                            "policy snapshot contains an active Retrieval Skill without verified source provenance"
                        )
                    policy_retrieval_records.append(matched)
                else:
                    policy_retrieval_records.append(RetrievalSkill(**record))
            policy_retrieval_records = tuple(policy_retrieval_records)
            policy_retrieval_ids = {
                skill.skill_id for skill in policy_retrieval_records
            }
            retrieval_records = [
                skill
                for skill in retrieval_library.all()
                if skill.skill_id not in policy_retrieval_ids
            ]
            retrieval_records.extend(policy_retrieval_records)
        else:
            retrieval_records = []
        decision_records = {skill.name: skill for skill in decision_library.all()}
        decision_records.update(
            {record["name"]: DecisionSkill(**record) for record in policy.decision_skills}
        )
        task_library = SkillLibrary(
            library.path,
            list(coding_records.values()),
            persist=not isolate_library,
        )
        if release_library is not None:
            task_retrieval_library = release_library.replay_snapshot(
                release_library.all(), persist=False
            )
        elif runtime_skill_source is not None:
            task_retrieval_library = runtime_skill_source.replay_snapshot(
                runtime_skill_source.all(), persist=False
            )
        else:
            task_retrieval_library = retrieval_library.replay_snapshot(
                retrieval_records, persist=not isolate_library
            )
        task_decision_library = DecisionSkillLibrary(
            decision_library.path,
            list(decision_records.values()),
            persist=not isolate_library,
        )
        coding = CodingEvolutionAgent(
            llm,
            task_library,
            CodingEvolutionConfig(
                setting=args.setting,
                initial_programs=policy.coding_initial_programs,
                mutations=policy.coding_mutations,
                mutation_children=policy.coding_mutation_children,
                validation_folds=policy.coding_validation_folds,
                validation_horizon=policy.coding_validation_horizon,
                use_external_knowledge=getattr(args, "setting2_knowledge", False),
            ),
            tsfm_forecaster=tsfm,
            generation_prompt=policy.coding_generation_prompt,
            revision_prompt=policy.coding_revision_prompt,
        )
        retrieval_agent = (
            TwoStageRetrievalAgent(llm, fixed_genome, task_retrieval_library)
            if fixed_genome is not None
            else RetrievalAgent(
                llm,
                task_retrieval_library,
                prompt=policy.retrieval_prompt,
            )
        )
        return EvolvingForecastHarness(
            coding,
            retrieval_agent,
            DecisionAgent(
                llm,
                task_decision_library,
                prompt=policy.decision_prompt,
            ),
            OutcomeSkillLearner(
                llm,
                task_retrieval_library,
                task_decision_library,
            ),
            HarnessRuntimeConfig(
                workflow=policy.workflow,
                retrieval_mode=retrieval_mode.replace("-", "_"),
                enable_evidence_adjustments=policy.enable_evidence_adjustments,
                max_evidence_adjustments=policy.max_evidence_adjustments,
                decision_aggregation=policy.decision_aggregation,
            ),
            morphology=morphology_provider,
        )
    return build


def run_command(args) -> dict:
    tasks = _task_subset(load_context_tasks(args.tasks_file), args.seed, args.limit)
    llm, library, retrieval_library, decision_library, tsfm = _components(args)
    factory = _factory(
        args,
        llm,
        library,
        retrieval_library,
        decision_library,
        tsfm,
        isolate_library=not args.learn_from_public_outcomes,
    )
    seed_policy = _seed_policy(args)
    harness = factory(seed_policy) if args.learn_from_public_outcomes else None
    destination = Path(args.results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with destination.open("w", encoding="utf-8") as output:
        for task in tasks:
            task_harness = harness or factory(seed_policy)
            result = task_harness.run(task)
            if args.learn_from_public_outcomes:
                outcome, learning = task_harness.record_outcome(task, result)
            else:
                outcome = task_harness.score_after_resolution(task, result)
                learning = None
            outcomes.append(outcome)
            output.write(
                json.dumps(
                    {
                        "outcome": asdict(outcome),
                        "selected_candidate_id": result.decision.selected.candidate_id,
                        "host_default_id": result.decision.host_default_id,
                        "retrieved_document_ids": list(result.retrieval.selected_document_ids),
                        "retrieval_rejections": list(result.retrieval.rejected),
                        "used_retrieval_skills": list(result.retrieval.used_skill_names),
                        "used_decision_skills": list(result.decision.used_skill_names),
                        "learned_skills": asdict(learning) if learning is not None else None,
                        "coding_candidates": [
                            {
                                "name": item.program.name,
                                "assumption": item.program.assumption,
                                "failure_condition": item.program.failure_condition,
                                "hindcast_smae": item.hindcast_smae,
                                "hindcast_srmse": item.hindcast_srmse,
                                "knowledge_ids": list(item.program.knowledge_ids),
                            }
                            for item in result.coding.candidates
                        ],
                        "setting2_knowledge": {
                            "version": result.coding.knowledge_base_version,
                            "retrieved_entry_ids": list(
                                result.coding.retrieved_knowledge_ids
                            ),
                            "selected_entry_ids": list(result.coding.selected_knowledge_ids),
                            "diagnostic_profile": (
                                asdict(result.coding.diagnostic_profile)
                                if result.coding.diagnostic_profile is not None
                                else None
                            ),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
    return {
        "n_tasks": len(outcomes),
        **evaluation_diagnostics(outcomes),
        "results_path": str(destination),
        "skills_saved": len(library),
        "retrieval_skills_saved": len(retrieval_library),
        "decision_skills_saved": len(decision_library),
        "online_skill_learning": args.learn_from_public_outcomes,
    }


def _sha256_sources(*relative_paths: str) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _frozen_sha256(explicit: str | None, label: str, fallback: str) -> str:
    value = fallback if explicit is None else explicit
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase sha256 digest")
    return value


def _checkpoint_authority_path(args, checkpoint_path: Path) -> Path:
    configured = getattr(args, "checkpoint_authority_path", None)
    if not configured:
        raise ValueError(
            "Retrieval checkpoint authority must be independently provisioned"
        )
    authority = Path(configured)
    if authority.resolve(strict=False).parent == checkpoint_path.resolve(
        strict=False
    ).parent:
        raise ValueError("caller-authored adjacent checkpoint authority is forbidden")
    return authority


def _consume_retrieval_checkpoint_authority_environment(
    args,
    *,
    resume_required: bool,
) -> tuple[bytes, tuple[int, str] | None, dict[str, str]]:
    key_environment_name = getattr(
        args,
        "checkpoint_authority_key_env",
        RETRIEVAL_CHECKPOINT_AUTHORITY_KEY_ENV,
    )
    expected_environment_name = getattr(
        args,
        "checkpoint_authority_expected_env",
        RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED_ENV,
    )
    selected_names = (key_environment_name, expected_environment_name)
    names_to_scrub = {
        RETRIEVAL_CHECKPOINT_AUTHORITY_KEY_ENV,
        RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED_ENV,
        *(name for name in selected_names if isinstance(name, str)),
    }
    consumed = {
        name: os.environ.pop(name, None) for name in names_to_scrub
    }
    if (
        type(key_environment_name) is not str
        or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", key_environment_name
        )
    ):
        raise ValueError("Retrieval checkpoint authority key environment is invalid")
    if (
        type(expected_environment_name) is not str
        or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", expected_environment_name
        )
        or expected_environment_name == key_environment_name
    ):
        raise ValueError(
            "Retrieval checkpoint authority expected-anchor environment is invalid"
        )
    supplied_key = consumed.get(key_environment_name)
    if supplied_key is None or len(supplied_key.encode("utf-8")) < 32:
        raise ValueError(
            "Retrieval checkpoint authority key is missing or too short"
        )
    supplied_expected = consumed.get(expected_environment_name)
    if supplied_expected is None:
        if resume_required:
            raise ValueError(
                "Retrieval checkpoint resume requires an external authority anchor"
            )
        expected_anchor = None
    else:
        match = re.fullmatch(
            r"(0|[1-9][0-9]*):([0-9a-f]{64})", supplied_expected
        )
        if match is None:
            raise ValueError(
                "Retrieval checkpoint external authority anchor is invalid"
            )
        expected_anchor = (int(match.group(1)), match.group(2))
    sensitive_values = {
        value for value in consumed.values() if value is not None
    }
    subprocess_environment = {
        name: value
        for name, value in os.environ.items()
        if name not in names_to_scrub and value not in sensitive_values
    }
    return (
        supplied_key.encode("utf-8"),
        expected_anchor,
        subprocess_environment,
    )


def _restore_retrieval_checkpoint_authority(
    checkpoint_path: Path,
    authority_path: Path,
) -> None:
    del checkpoint_path, authority_path
    raise ValueError(
        "caller-authored checkpoint sidecars cannot activate Retrieval resume; "
        "use the protected operator authority transaction"
    )


def _persist_retrieval_checkpoint_authority(
    engine: object,
    checkpoint_path: Path,
    authority_path: Path,
) -> None:
    del engine, checkpoint_path, authority_path
    raise ValueError(
        "checkpoint authority must commit inside each checkpoint transaction"
    )


def _scope_changelogs(result: RetrievalEvolutionResult) -> list[dict[str, object]]:
    known = {result.original_parent.version: result.original_parent.to_payload()}
    generations: list[dict[str, object]] = []
    for generation in result.generations:
        parent = known.get(generation.parent_version, {})
        children = []
        for scope, proposal in zip(
            generation.child_scopes, generation.child_proposals
        ):
            changed = sorted(
                field
                for field, value in proposal.items()
                if field not in {"schema_version", "version", "parent"}
                and parent.get(field) != value
            )
            children.append(
                {
                    "scope": scope,
                    "version": proposal.get("version"),
                    "changed_fields": changed,
                    "changelog": (
                        f"Scope {scope} changed " + ", ".join(changed)
                        if changed
                        else f"Scope {scope} proposed no owned-field change"
                    ),
                }
            )
            version = proposal.get("version")
            if isinstance(version, str):
                known[version] = proposal
        generations.append(
            {"generation": generation.generation, "children": children}
        )
    return generations


def _selected_train_summary(result: RetrievalEvolutionResult) -> dict[str, object]:
    fingerprint = result.train_winner.fingerprint()
    for generation in reversed(result.generations):
        summary = generation.train_summaries.get(fingerprint)
        if summary:
            return dict(summary)
    return {
        "task_count": 80,
        "winner_version": result.train_winner.version,
        "winner_fingerprint": fingerprint,
    }


def _publish_or_resume_accepted_retrieval_release(
    releases_path: str | Path,
    genome: RetrievalGenome,
    *,
    skills: Sequence[object],
    audit: Mapping[str, object],
    parent_release: RetrievalRelease | None = None,
) -> RetrievalRelease:
    """Rebase an internal winner onto contiguous authoritative release history."""
    releases = Path(releases_path)
    if not releases.is_dir():
        raise RetrievalEvolutionError(
            "accepted Retrieval publication requires an authoritative release history"
        )
    version_names = sorted(
        item.name
        for item in releases.iterdir()
        if re.fullmatch(r"v\d{3}", item.name)
    )
    if not version_names or version_names[0] != "v000":
        raise RetrievalEvolutionError("Retrieval release history must begin at v000")
    numbers = [int(name[1:]) for name in version_names]
    if numbers != list(range(numbers[-1] + 1)):
        raise RetrievalEvolutionError(
            "Retrieval release history contains a gap or collision"
        )
    history = tuple(
        _load_retrieval_release_for_operator(releases / name)
        for name in version_names
    )
    for index, release in enumerate(history):
        expected_state = "seed" if index == 0 else "accepted"
        expected_parent = None if index == 0 else f"v{index - 1:03d}"
        if (
            release.genome.version != f"v{index:03d}"
            or release.genome.parent != expected_parent
            or release.manifest["state"] != expected_state
        ):
            raise RetrievalEvolutionError(
                "Retrieval release history is not one authoritative accepted lineage"
            )
    parent_version = (
        parent_release.genome.version
        if parent_release is not None
        else genome.parent
    )
    if parent_version is None or not re.fullmatch(r"v\d{3}", parent_version):
        raise RetrievalEvolutionError("accepted Retrieval winner has no authoritative Parent")
    parent_number = int(parent_version[1:])
    if parent_number >= len(history):
        raise RetrievalEvolutionError("accepted Retrieval Parent is absent from release history")
    authoritative_parent = history[parent_number]
    if (
        parent_release is not None
        and (
            authoritative_parent.genome.fingerprint()
            != parent_release.genome.fingerprint()
            or authoritative_parent.manifest_file_sha256
            != parent_release.manifest_file_sha256
            or authoritative_parent.skills_file_sha256
            != parent_release.skills_file_sha256
        )
    ):
        raise RetrievalEvolutionError(
            "accepted Retrieval Parent differs from authoritative release history"
        )
    next_version = f"v{parent_number + 1:03d}"
    if len(history) not in {parent_number + 1, parent_number + 2}:
        raise RetrievalEvolutionError(
            "accepted Retrieval Parent is stale or release history has advanced"
        )
    rebased = replace(
        genome,
        version=next_version,
        parent=parent_version,
    )
    destination = releases / next_version

    def verify_existing() -> RetrievalRelease:
        release = _load_retrieval_release_for_operator(destination)
        if release.genome.fingerprint() != rebased.fingerprint():
            raise RetrievalEvolutionError(
                "existing accepted Retrieval release Genome differs from resumed result"
            )
        if _plain_json(release.skills) != _plain_json(list(skills)):
            raise RetrievalEvolutionError(
                "existing accepted Retrieval release Skills differ from resumed result"
            )
        for field, expected in audit.items():
            if release.manifest.get(field) != expected:
                raise RetrievalEvolutionError(
                    f"existing accepted Retrieval release differs at {field}"
                )
        return release

    if len(history) == parent_number + 2:
        if history[-1].path.name != next_version:
            raise RetrievalEvolutionError("accepted Retrieval resume history is inconsistent")
        return verify_existing()
    try:
        return _write_accepted_retrieval_release(
            releases,
            rebased,
            skills=skills,
            audit=audit,
        )
    except RetrievalPolicyError:
        # A concurrent no-replace winner is acceptable only if the private
        # operator loader proves it is byte-equivalent in every bound input.
        if not os.path.lexists(destination):
            raise
        return verify_existing()


def _assert_no_public_regression_ids(
    payload: object, public_ids: frozenset[str]
) -> None:
    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for key, item in value.items():
                yield str(key)
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)

    leaked = sorted(
        task_id
        for task_id in public_ids
        if any(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(task_id)}(?![A-Za-z0-9])",
                value,
            )
            for value in strings(payload)
        )
    )
    if leaked:
        raise RetrievalEvolutionError(
            f"Public Regression task ID reached a Retrieval evolution boundary: {leaked[0]}"
        )


def _assert_retrieval_prompt_inputs_clean(
    policy: HarnessPolicy,
    coding_library: SkillLibrary,
    decision_library: DecisionSkillLibrary,
    public_ids: frozenset[str],
) -> None:
    """Reject Public Regression provenance from every prompt-bearing input."""
    _assert_no_public_regression_ids(
        {
            "policy": policy.to_payload(),
            "coding_skills": [asdict(skill) for skill in coding_library.all()],
            "decision_skills": [
                asdict(skill) for skill in decision_library.all()
            ],
        },
        public_ids,
    )


def _published_retrieval_result(
    result: RetrievalEvolutionResult,
    release: RetrievalRelease,
) -> RetrievalEvolutionResult:
    """Bind a validated internal winner to its authoritative published identity."""
    if (
        not result.accepted
        or result.release_genome is None
        or release.manifest.get("state") != "accepted"
        or not result.trace
        or result.trace[-1].get("kind") != "release_accepted"
    ):
        raise RetrievalEvolutionError(
            "only an accepted Retrieval result can bind a published release"
        )
    publication_event = {
        **result.trace[-1],
        "genome": release.genome.version,
        "publication_deferred": False,
    }
    return replace(
        result,
        selected_genome=release.genome,
        release_genome=release.genome,
        release_published=True,
        trace=(*result.trace[:-1], publication_event),
    )


def _validate_complete_retrieval_result(
    result: RetrievalEvolutionResult,
    *,
    parent: RetrievalGenome,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    public_ids: frozenset[str],
) -> None:
    expected_train = tuple(train_ids)
    expected_dev = tuple(dev_ids)
    train_set = frozenset(expected_train)
    if len(train_set) != len(expected_train) or len(set(expected_dev)) != len(
        expected_dev
    ):
        raise RetrievalEvolutionError("frozen Retrieval task provenance is duplicated")
    if not isinstance(result, RetrievalEvolutionResult):
        raise RetrievalEvolutionError("Retrieval evolution returned an untyped result")
    if result.original_parent.fingerprint() != parent.fingerprint():
        raise RetrievalEvolutionError(
            "Retrieval evolution result changed its authoritative Parent"
        )
    reparsed = RetrievalEvolutionResult.from_payload(result.to_payload())
    if reparsed.to_payload() != result.to_payload():
        raise RetrievalEvolutionError("Retrieval evolution result is not canonical")
    for generation in result.generations:
        if (
            len(set(generation.screen_task_ids)) != len(generation.screen_task_ids)
            or not set(generation.screen_task_ids).issubset(train_set)
        ):
            raise RetrievalEvolutionError(
                "Retrieval generation references a task outside frozen Train"
            )
    if result.parent_dev is None:
        raise RetrievalEvolutionError("Parent Dev task provenance is missing")
    parent_dev_ids = tuple(
        trace.get("task_id") for trace in result.parent_dev.task_traces
    )
    if (
        result.parent_dev.task_count != len(expected_dev)
        or parent_dev_ids != expected_dev
        or result.parent_dev.version != parent.version
    ):
        raise RetrievalEvolutionError("Parent Dev trace provenance is incomplete")
    if result.child_dev is not None:
        child_dev_ids = tuple(
            trace.get("task_id") for trace in result.child_dev.task_traces
        )
        if (
            result.child_dev.task_count != len(expected_dev)
            or child_dev_ids != expected_dev
            or result.child_dev.version != result.train_winner.version
        ):
            raise RetrievalEvolutionError("Child Dev trace provenance is incomplete")
    if result.accepted:
        if result.child_dev is None or result.release_genome is None:
            raise RetrievalEvolutionError(
                "accepted Retrieval result has incomplete trusted provenance"
            )
        if (
            result.selected_genome.fingerprint()
            != result.train_winner.fingerprint()
            or result.release_genome.fingerprint()
            != result.selected_genome.fingerprint()
        ):
            raise RetrievalEvolutionError(
                "accepted Retrieval release Genome changed after Dev selection"
            )
    elif (
        result.release_genome is not None
        or result.selected_genome.fingerprint() != parent.fingerprint()
    ):
        raise RetrievalEvolutionError(
            "rejected Retrieval result cannot carry a release Genome or Child selection"
        )
    if result.release_published:
        raise RetrievalEvolutionError(
            "engine result cannot claim release publication before operator validation"
        )
    _assert_no_public_regression_ids(result.to_payload(), public_ids)


def _validate_retrieval_checkpoint_payload(
    payload: object,
    *,
    parent: RetrievalGenome,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    public_ids: frozenset[str],
) -> None:
    """Apply the frozen task/Parent firewall before every checkpoint commit."""
    _assert_no_public_regression_ids(payload, public_ids)
    if not isinstance(payload, Mapping):
        raise RetrievalEvolutionError("Retrieval checkpoint payload must be an object")
    result_fields = {
        "original_parent",
        "train_winner",
        "selected_genome",
        "accepted",
        "acceptance_reasons",
        "rejection_reasons",
        "parent_dev",
        "child_dev",
        "generations",
        "trace",
        "release_genome",
        "release_published",
    }
    result_payload: object | None
    if set(payload) == result_fields:
        result_payload = payload
    else:
        try:
            original = RetrievalGenome.from_payload(payload["original_parent"])
            current = RetrievalGenome.from_payload(payload["current_parent"])
        except (KeyError, TypeError, ValueError, RetrievalPolicyError) as error:
            raise RetrievalEvolutionError(
                "Retrieval checkpoint Parent provenance is invalid"
            ) from error
        if original.fingerprint() != parent.fingerprint():
            raise RetrievalEvolutionError(
                "Retrieval checkpoint changed its authoritative Parent"
            )
        if current.to_payload() != payload["current_parent"]:
            raise RetrievalEvolutionError(
                "Retrieval checkpoint current Parent is not canonical"
            )
        raw_generations = payload.get("generations")
        if not isinstance(raw_generations, list):
            raise RetrievalEvolutionError(
                "Retrieval checkpoint generation provenance is invalid"
            )
        try:
            generations = tuple(
                RetrievalGenerationTrace.from_payload(item)
                for item in raw_generations
            )
        except (TypeError, ValueError) as error:
            raise RetrievalEvolutionError(
                "Retrieval checkpoint proposal provenance is invalid"
            ) from error
        train_set = frozenset(train_ids)
        for generation in generations:
            if (
                len(set(generation.screen_task_ids))
                != len(generation.screen_task_ids)
                or not set(generation.screen_task_ids).issubset(train_set)
            ):
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint task provenance escaped frozen Train"
                )
        pending = payload.get("pending_children")
        if pending is not None:
            children = pending.get("children") if isinstance(pending, Mapping) else None
            if not isinstance(children, list) or any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("proposal"), Mapping)
                for row in children
            ):
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint pending proposal provenance is invalid"
                )

        expected_dev = tuple(dev_ids)
        allowed_train = frozenset(train_ids)

        def validate_execution_record(record: object) -> None:
            if not isinstance(record, Mapping):
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint execution provenance is invalid"
                )
            task_ids_value = record.get("task_ids")
            stage = record.get("stage")
            if (
                not isinstance(task_ids_value, list)
                or any(not isinstance(task_id, str) for task_id in task_ids_value)
                or not isinstance(stage, str)
            ):
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint execution task provenance is invalid"
                )
            task_vector = tuple(task_ids_value)
            if "dev" in stage:
                valid = task_vector == expected_dev
            else:
                valid = (
                    len(set(task_vector)) == len(task_vector)
                    and set(task_vector).issubset(allowed_train)
                )
            if not valid:
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint execution task provenance is invalid"
                )

        completion = payload.get("task_completion")
        evaluation_cache = payload.get("evaluation_cache")
        terminal_outcomes = payload.get("terminal_outcomes")
        if not isinstance(completion, list):
            raise RetrievalEvolutionError(
                "Retrieval checkpoint task completion provenance is invalid"
            )
        for record in completion:
            validate_execution_record(record)
        for records in (evaluation_cache, terminal_outcomes):
            if not isinstance(records, Mapping):
                raise RetrievalEvolutionError(
                    "Retrieval checkpoint execution ledger is invalid"
                )
            for record in records.values():
                validate_execution_record(record)
        result_payload = payload.get("result")

    if result_payload is not None:
        try:
            result = RetrievalEvolutionResult.from_payload(result_payload)
        except (TypeError, ValueError) as error:
            raise RetrievalEvolutionError(
                "Retrieval checkpoint complete result is invalid"
            ) from error
        _validate_complete_retrieval_result(
            result,
            parent=parent,
            train_ids=train_ids,
            dev_ids=dev_ids,
            public_ids=public_ids,
        )


def _cli_entry_snapshot(
    parent_descriptor: int, name: str
) -> tuple[tuple[int, int], bytes] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Retrieval output target is not a regular file")
        return (metadata.st_dev, metadata.st_ino), handle.read()


def _publish_retrieval_output_bytes(
    path: Path,
    encoded: bytes,
    *,
    expected_identity: tuple[int, int] | None,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    safe, parent_descriptor = _open_checkpoint_parent(
        path, create=expected_parent_identity is None
    )
    temporary: str | None = None
    quarantine: str | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if (
            expected_parent_identity is not None
            and (parent_metadata.st_dev, parent_metadata.st_ino)
            != expected_parent_identity
        ):
            raise ValueError(
                "Retrieval output parent directory identity changed after preflight"
            )
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        current = _cli_entry_snapshot(parent_descriptor, safe.name)
        current_identity = current[0] if current is not None else None
        if current_identity != expected_identity:
            raise ValueError(
                "Retrieval output identity changed after path validation"
            )
        temporary = _unique_checkpoint_temporary(
            parent_descriptor, safe.name, encoded
        )
        staged = _cli_entry_snapshot(parent_descriptor, temporary)
        if staged is None or staged[1] != encoded:
            raise ValueError("Retrieval output staging identity is unavailable")
        staged_identity = staged[0]
        if expected_identity is not None:
            quarantine = _move_artifact_entry_to_quarantine(
                parent_descriptor, safe.name
            )
            if quarantine is None:
                raise ValueError(
                    "Retrieval output disappeared during guarded publication"
                )
            moved = _cli_entry_snapshot(parent_descriptor, quarantine)
            if moved is None or moved[0] != expected_identity:
                if moved is not None:
                    _restore_quarantined_artifact_entry(
                        parent_descriptor,
                        quarantine,
                        safe.name,
                        expected_identity=moved[0],
                    )
                raise ValueError(
                    "Retrieval output replacement was quarantined during publication"
                )
        _revalidate_checkpoint_parent(safe, parent_descriptor)
        try:
            _rename_artifact_entry_noreplace(
                parent_descriptor, temporary, safe.name
            )
        except Exception:
            published = _cli_entry_snapshot(parent_descriptor, safe.name)
            if published is None or published[0] != staged_identity:
                raise
        temporary = None
        published = _cli_entry_snapshot(parent_descriptor, safe.name)
        if published != (staged_identity, encoded):
            raise ValueError("Retrieval output publication verification failed")
        os.fsync(parent_descriptor)
        _revalidate_checkpoint_parent(safe, parent_descriptor)
    finally:
        # Unique unpublished and quarantine names are retained on uncertainty;
        # deleting either by name after an ownership check can remove a replacement.
        os.close(parent_descriptor)


def _write_json_artifact(
    path: Path,
    payload: object,
    *,
    expected_identity: tuple[int, int] | None,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_retrieval_output_bytes(
        path,
        encoded,
        expected_identity=expected_identity,
        expected_parent_identity=expected_parent_identity,
    )


def _canonical_cli_path(value: str | Path, label: str) -> Path:
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot canonicalize {label} path") from error


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _existing_cli_path_identity(
    path: Path, label: str
) -> tuple[int, int] | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"cannot inspect {label} path identity") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} path identity is a symlink")
    return metadata.st_dev, metadata.st_ino


def _validate_retrieval_evolution_paths(args) -> dict[str, object]:
    trace = _canonical_cli_path(args.trace_path, "trace")
    run_root = _canonical_cli_path(
        args.run_root if getattr(args, "run_root", None) else trace.parent,
        "approved run root",
    )
    checkpoint = _canonical_cli_path(
        args.checkpoint_path
        if args.checkpoint_path
        else trace.with_name("checkpoint.json"),
        "checkpoint",
    )
    progress = _canonical_cli_path(
        args.progress_path
        if args.progress_path
        else trace.with_name("progress.jsonl"),
        "progress",
    )
    policy = _canonical_cli_path(args.policy_path, "policy")
    configured_authority = getattr(args, "checkpoint_authority_path", None)
    if not configured_authority:
        raise ValueError(
            "Retrieval evolution requires an independently provisioned --checkpoint-authority-path"
        )
    authority = _canonical_cli_path(configured_authority, "checkpoint authority")
    configured_authority_head = getattr(
        args, "checkpoint_authority_head_path", None
    )
    if not configured_authority_head:
        raise ValueError(
            "Retrieval evolution requires an independently provisioned "
            "--checkpoint-authority-head-path"
        )
    authority_head = _canonical_cli_path(
        configured_authority_head, "checkpoint authority head"
    )
    configured_authority_anchor = getattr(
        args, "checkpoint_authority_anchor_path", None
    )
    if not configured_authority_anchor:
        raise ValueError(
            "Retrieval evolution requires an independently provisioned "
            "--checkpoint-authority-anchor-path ledger"
        )
    authority_anchor = _canonical_cli_path(
        configured_authority_anchor,
        "checkpoint monotonic anchor ledger",
    )
    if (
        authority_head == authority
        or authority_head.parent != authority.parent
    ):
        raise ValueError(
            "checkpoint authority journal and head must be distinct protected records"
        )
    if (
        authority_anchor in {authority, authority_head}
        or authority_anchor.parent != authority.parent
    ):
        raise ValueError(
            "checkpoint monotonic anchor ledger must be a distinct path in the pinned operator directory"
        )
    outputs = {
        "checkpoint": checkpoint,
        "progress": progress,
        "trace": trace,
        "policy": policy,
    }
    for label, path in outputs.items():
        if path == run_root or run_root not in path.parents:
            raise ValueError(f"{label} path escapes the approved run root")
    rows = tuple(outputs.items())
    for index, (left_label, left_path) in enumerate(rows):
        for right_label, right_path in rows[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise ValueError(
                    f"Retrieval output paths must be pairwise disjoint: {left_label}/{right_label}"
                )
    if any(
        path == run_root or run_root in path.parents
        for path in (authority, authority_head, authority_anchor)
    ):
        raise ValueError(
            "checkpoint authority records must be outside the approved run root"
        )
    tasks_path = _canonical_cli_path(args.tasks_file, "tasks")
    task_source_snapshot = _snapshot_retrieval_task_source(tasks_path)
    protected_inputs: list[tuple[str, Path]] = [
        ("tasks", tasks_path),
        ("split manifest", _canonical_cli_path(args.split_manifest, "split manifest")),
        (
            "Retrieval release",
            _canonical_cli_path(args.retrieval_release_path, "Retrieval release"),
        ),
        ("coding library", _canonical_cli_path(args.library_path, "coding library")),
        (
            "Retrieval library",
            _canonical_cli_path(args.retrieval_library_path, "Retrieval library"),
        ),
        (
            "Decision library",
            _canonical_cli_path(args.decision_library_path, "Decision library"),
        ),
        ("checkpoint authority", authority),
        ("checkpoint authority head", authority_head),
        ("checkpoint monotonic anchor ledger", authority_anchor),
    ]
    if getattr(args, "seed_policy_path", None):
        protected_inputs.append(
            (
                "seed policy",
                _canonical_cli_path(args.seed_policy_path, "seed policy"),
            )
        )
    identity_overrides: dict[str, tuple[int, int]] = {}
    if task_source_snapshot is not None:
        source_identity = task_source_snapshot["source_identity"]
        assert isinstance(source_identity, tuple)
        identity_overrides["tasks"] = source_identity
        if task_source_snapshot["kind"] == "directory":
            for member_name, member_identity in task_source_snapshot["members"]:
                label = f"Retrieval task member {member_name}"
                protected_inputs.append((label, tasks_path / member_name))
                identity_overrides[label] = member_identity
    authority_anchor_snapshot = _snapshot_retrieval_task_source(
        authority_anchor
    )
    if authority_anchor_snapshot is not None:
        if authority_anchor_snapshot["kind"] != "directory":
            raise ValueError(
                "checkpoint monotonic anchor ledger must be a directory"
            )
        anchor_identity = authority_anchor_snapshot["source_identity"]
        assert isinstance(anchor_identity, tuple)
        identity_overrides["checkpoint monotonic anchor ledger"] = (
            anchor_identity
        )
        for member_name, member_identity in authority_anchor_snapshot[
            "members"
        ]:
            label = f"checkpoint monotonic anchor member {member_name}"
            protected_inputs.append(
                (label, authority_anchor / member_name)
            )
            identity_overrides[label] = member_identity
    release_root = protected_inputs[2][1].parent
    protected_inputs.append(("Retrieval release root", release_root))
    release_path = protected_inputs[2][1]
    for artifact_name in (
        "genome.json",
        "round1_prompt.md",
        "round2_prompt.md",
        "skills.json",
        "manifest.json",
    ):
        protected_inputs.append(
            (
                f"Retrieval release artifact {artifact_name}",
                release_path / artifact_name,
            )
        )
    for input_label, input_path in protected_inputs:
        if input_label not in {
            "checkpoint authority",
            "checkpoint authority head",
        }:
            for record in (authority, authority_head):
                if _paths_overlap(record, input_path):
                    raise ValueError(
                        "checkpoint authority collides with protected "
                        f"{input_label} path"
                    )
    for output_label, output_path in outputs.items():
        for input_label, input_path in protected_inputs:
            if _paths_overlap(output_path, input_path):
                raise ValueError(
                    f"{output_label} path collides with protected {input_label} path"
                )
    identities: dict[tuple[int, int], tuple[str, Path]] = {}
    output_identities: dict[str, tuple[int, int] | None] = {}
    protected_identities: dict[str, tuple[int, int] | None] = {}
    for label, path in (*outputs.items(), *protected_inputs):
        identity = identity_overrides.get(label)
        if identity is None:
            identity = _existing_cli_path_identity(path, label)
        if label in outputs:
            output_identities[label] = identity
        else:
            protected_identities[label] = identity
        if identity is None:
            continue
        prior = identities.get(identity)
        if prior is not None:
            prior_label, prior_path = prior
            raise ValueError(
                "Retrieval paths are inode aliases: "
                f"{prior_label} ({prior_path}) / {label} ({path})"
            )
        identities[identity] = (label, path)
    output_parent_identities: dict[str, tuple[int, int]] = {}
    for label, path in outputs.items():
        try:
            _safe, parent_descriptor = _open_checkpoint_parent(path, create=True)
        except Exception as error:
            raise ValueError(
                f"cannot establish {label} output parent directory"
            ) from error
        try:
            metadata = os.fstat(parent_descriptor)
            output_parent_identities[label] = (
                metadata.st_dev,
                metadata.st_ino,
            )
            _revalidate_checkpoint_parent(path, parent_descriptor)
        finally:
            os.close(parent_descriptor)
    authority_parent_identity = _existing_cli_path_identity(
        authority.parent,
        "checkpoint authority parent",
    )
    return {
        **outputs,
        "run_root": run_root,
        "authority": authority,
        "authority_head": authority_head,
        "authority_anchor": authority_anchor,
        "output_identities": output_identities,
        "output_parent_identities": output_parent_identities,
        "protected_identities": protected_identities,
        "authority_parent_identity": authority_parent_identity,
        "authority_anchor_snapshot": authority_anchor_snapshot,
        "task_source_snapshot": task_source_snapshot,
    }


def _retrieval_evolve_command(args) -> dict:
    """Run the fixed 80/20 Retrieval evolution protocol; Public Regression stays absent."""
    if not args.split_manifest:
        raise ValueError("Retrieval evolution requires --split-manifest")
    if args.retrieval_mode != "two-stage":
        raise ValueError("Retrieval evolution requires --retrieval-mode two-stage")
    validated_paths = _validate_retrieval_evolution_paths(args)
    args.trace_path = str(validated_paths["trace"])
    args.progress_path = str(validated_paths["progress"])
    args.policy_path = str(validated_paths["policy"])
    args.checkpoint_path = str(validated_paths["checkpoint"])
    args.checkpoint_authority_path = str(validated_paths["authority"])
    args.checkpoint_authority_head_path = str(
        validated_paths["authority_head"]
    )
    args.checkpoint_authority_anchor_path = str(
        validated_paths["authority_anchor"]
    )
    checkpoint_path = validated_paths["checkpoint"]
    authority_path = _checkpoint_authority_path(args, checkpoint_path)
    authority_anchor_snapshot = validated_paths["authority_anchor_snapshot"]
    authority_anchor_has_records = bool(
        authority_anchor_snapshot is not None
        and authority_anchor_snapshot["members"]
    )
    resume_required = any(
        identity is not None
        for identity in (
            validated_paths["output_identities"]["checkpoint"],
            validated_paths["protected_identities"]["checkpoint authority"],
            validated_paths["protected_identities"]["checkpoint authority head"],
        )
    ) or authority_anchor_has_records
    (
        authority_key,
        expected_external_anchor,
        llm_subprocess_env,
    ) = _consume_retrieval_checkpoint_authority_environment(
        args,
        resume_required=resume_required,
    )
    preflight_authority = _open_retrieval_checkpoint_authority_for_operator(
        checkpoint_path,
        authority_path,
        validated_paths["authority_head"],
        authentication_key=authority_key,
        expected_authority_anchor=expected_external_anchor,
        expected_checkpoint_identity=validated_paths["output_identities"][
            "checkpoint"
        ],
        expected_authority_identity=validated_paths["protected_identities"][
            "checkpoint authority"
        ],
        expected_head_identity=validated_paths["protected_identities"][
            "checkpoint authority head"
        ],
        expected_checkpoint_parent_identity=validated_paths[
            "output_parent_identities"
        ]["checkpoint"],
        expected_authority_parent_identity=validated_paths[
            "authority_parent_identity"
        ],
        authority_anchor_path=validated_paths["authority_anchor"],
        expected_authority_anchor_identity=(
            None
            if authority_anchor_snapshot is None
            else authority_anchor_snapshot["source_identity"]
        ),
    )
    try:
        active_authority_anchor = preflight_authority.current_anchor
        active_checkpoint_identity = preflight_authority._checkpoint_identity
        active_authority_identity = preflight_authority._record_identities[
            authority_path.name
        ]
        active_head_identity = preflight_authority._record_identities[
            Path(validated_paths["authority_head"]).name
        ]
        active_authority_anchor_identity = (
            preflight_authority._anchor_directory_identity
        )
    finally:
        preflight_authority.close()
    parent_release = _load_retrieval_release_for_operator(
        args.retrieval_release_path
    )
    _require_authorized_retrieval_release_state(parent_release)
    parent_library = RetrievalSkillLibrary._from_loaded_release(parent_release)
    train, dev, split_sha256, public_ids = _load_retrieval_evolution_tasks(
        args.tasks_file,
        args.split_manifest,
        expected_manifest_sha256=args.split_manifest_sha256,
        include_public_ids=True,
        expected_source_snapshot=validated_paths["task_source_snapshot"],
    )
    if any(task.numeric.task_id in public_ids for task in (*train, *dev)):
        raise RetrievalEvolutionError(
            "Public Regression tasks cannot enter Retrieval evolution"
        )

    _assert_no_public_regression_ids(
        {
            "genome": parent_release.genome.to_payload(),
            "skills": _plain_json(parent_release.skills),
            "manifest": _plain_json(parent_release.manifest),
        },
        public_ids,
    )
    base_policy = _seed_policy(args)
    if base_policy.retrieval_release_payload is not None:
        expected_parent = _policy_with_retrieval_release(
            base_policy, parent_release, changelog=base_policy.changelog
        )
        if (
            base_policy.retrieval_release_sha256
            != expected_parent.retrieval_release_sha256
        ):
            raise ValueError(
                "seed HarnessPolicy does not match the trusted parent Retrieval release"
            )
    _assert_no_public_regression_ids(base_policy.to_payload(), public_ids)

    llm, library, _retrieval_library, decision_library, tsfm = _components(
        args,
        retrieval_library_override=parent_library,
        llm_subprocess_env=llm_subprocess_env,
    )
    _assert_retrieval_prompt_inputs_clean(
        base_policy,
        library,
        decision_library,
        public_ids,
    )
    morphology = _ConservativeMorphologyProvider()

    def harness_factory(
        genome: RetrievalGenome,
        skill_library: RetrievalSkillLibrary | None,
    ):
        if skill_library is None:
            raise RetrievalEvolutionError(
                "Retrieval candidate evaluation requires a verified Skill snapshot"
            )
        return _factory(
            args,
            llm,
            library,
            parent_library,
            decision_library,
            tsfm,
            isolate_library=True,
            morphology_provider=morphology,
            retrieval_genome=genome,
            retrieval_skill_source=skill_library,
        )(base_policy)

    verifier_hash = _frozen_sha256(
        args.verifier_sha256,
        "--verifier-sha256",
        _sha256_sources("evolving_loop/retrieval_agent/verifier.py"),
    )
    evaluator_hash = _frozen_sha256(
        args.evaluator_sha256,
        "--evaluator-sha256",
        _sha256_sources(
            "evolving_loop/cli.py",
            "evolving_loop/evaluation.py",
            "evolving_loop/retrieval_agent/credit.py",
        ),
    )
    metric_hash = _frozen_sha256(
        args.metric_sha256,
        "--metric-sha256",
        _sha256_sources("common/metrics.py"),
    )
    mutation_model_hash = _frozen_sha256(
        args.mutation_model_sha256,
        "--mutation-model-sha256",
        _canonical_sha256(
            {
                "backend": args.llm_backend,
                "model": (
                    args.codex_model
                    if args.llm_backend == "codex"
                    else args.claude_model
                    if args.llm_backend == "claude"
                    else args.model_id
                ),
                "reasoning_effort": getattr(args, "codex_reasoning_effort", None),
                "client_source_sha256": _sha256_sources("common/llm.py"),
            }
        ),
    )
    harness_source_hash = _sha256_sources(
        "evolving_loop/harness.py",
        "evolving_loop/frozen_inference.py",
        "evolving_loop/co_evolution.py",
        "evolving_loop/coding_agent/evolution.py",
        "evolving_loop/decision_agent/agent.py",
        "evolving_loop/morphology_adapter.py",
        "evolving_loop/retrieval_agent/policy.py",
        "evolving_loop/retrieval_agent/two_stage_agent.py",
        "evolving_loop/retrieval_agent/schemas.py",
    )
    harness_hash = _frozen_sha256(
        args.harness_sha256,
        "--harness-sha256",
        _canonical_sha256(
            {
                "source_sha256": harness_source_hash,
                "policy": base_policy.to_payload(),
                "coding_skills": [
                    asdict(skill)
                    for skill in sorted(
                        library.all(), key=lambda item: (item.skill_id, item.name)
                    )
                ],
                "decision_skills": [
                    asdict(skill)
                    for skill in sorted(
                        decision_library.all(),
                        key=lambda item: (item.skill_id, item.name),
                    )
                ],
                "runtime": {
                    "setting": args.setting,
                    "setting2_knowledge": args.setting2_knowledge,
                    "llm_backend": args.llm_backend,
                    "model_id": args.model_id,
                    "codex_model": args.codex_model,
                    "codex_reasoning_effort": args.codex_reasoning_effort,
                    "claude_model": args.claude_model,
                    "device": args.device,
                    "chronos_model_id": args.chronos_model_id,
                    "chronos_device": args.chronos_device,
                    "morphology_provider": "conservative_empty_v1",
                },
            }
        ),
    )
    config = RetrievalEvolutionConfig(
        generations=args.generations,
        screen_tasks=args.screen_train_tasks,
        promote=args.screen_promote,
        train_folds=args.train_folds,
        tolerance=args.evolution_tolerance,
        random_seed=args.seed,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        dataset_split_hash=split_sha256,
        verifier_hash=verifier_hash,
        evaluator_hash=evaluator_hash,
        metric_hash=metric_hash,
        mutation_model_hash=mutation_model_hash,
        harness_hash=harness_hash,
        metric_cap=args.metric_cap,
    )
    authority = _open_retrieval_checkpoint_authority_for_operator(
        checkpoint_path,
        authority_path,
        validated_paths["authority_head"],
        authentication_key=authority_key,
        expected_authority_anchor=active_authority_anchor,
        expected_checkpoint_identity=active_checkpoint_identity,
        expected_authority_identity=active_authority_identity,
        expected_head_identity=active_head_identity,
        expected_checkpoint_parent_identity=validated_paths[
            "output_parent_identities"
        ]["checkpoint"],
        expected_authority_parent_identity=validated_paths[
            "authority_parent_identity"
        ],
        authority_anchor_path=validated_paths["authority_anchor"],
        expected_authority_anchor_identity=active_authority_anchor_identity,
    )
    train_task_ids = tuple(task.numeric.task_id for task in train)
    dev_task_ids = tuple(task.numeric.task_id for task in dev)

    def validate_checkpoint_payload(payload: object) -> None:
        _validate_retrieval_checkpoint_payload(
            payload,
            parent=parent_release.genome,
            train_ids=train_task_ids,
            dev_ids=dev_task_ids,
            public_ids=public_ids,
        )

    try:
        engine = RetrievalEvolutionEngine(
            llm,
            _TrustedRetrievalEvaluator(),
            config,
            skill_library=parent_library,
            harness_factory=harness_factory,
            _checkpoint_authority=authority,
            _checkpoint_payload_validator=validate_checkpoint_payload,
            _checkpoint_parent_identity=validated_paths[
                "output_parent_identities"
            ]["checkpoint"],
        )
        result = engine.evolve(parent_release.genome, train, dev)
        committed_authority_anchor = authority.current_anchor
    finally:
        authority.close()

    _validate_complete_retrieval_result(
        result,
        parent=parent_release.genome,
        train_ids=train_task_ids,
        dev_ids=dev_task_ids,
        public_ids=public_ids,
    )

    scope_changelogs = _scope_changelogs(result)
    selected_release = parent_release
    saved_policy_path: str | None = None
    if result.accepted:
        if result.release_genome is None:
            raise RetrievalEvolutionError(
                "accepted Retrieval evolution omitted its release Genome"
            )
        library_for_release = (
            engine._readonly_library(result.release_genome)
            if callable(getattr(engine, "_readonly_library", None))
            else parent_library
        )
        if library_for_release is None:
            raise RetrievalEvolutionError(
                "accepted Retrieval release omitted its verified Skill history"
            )
        release_skills = tuple(
            skill.to_payload() for skill in library_for_release.all()
        )
        release_audit = {
            "state": "accepted",
            "train_dev_split_sha256": split_sha256,
            "verifier_sha256": verifier_hash,
            "evaluator_sha256": evaluator_hash,
            "metric_sha256": metric_hash,
            "metric_cap": config.metric_cap,
            "train_summary": _selected_train_summary(result),
            "dev_summary": (
                result.child_dev.summary()
                if result.child_dev
                else {"task_count": 20}
            ),
            "acceptance_reason": ";".join(result.acceptance_reasons),
        }
        _assert_no_public_regression_ids(
            {
                "result": result.to_payload(),
                "release_skills": release_skills,
                "release_audit": release_audit,
            },
            public_ids,
        )
        selected_release = _publish_or_resume_accepted_retrieval_release(
            parent_release.path.parent,
            result.release_genome,
            skills=release_skills,
            audit=release_audit,
            parent_release=parent_release,
        )
        accepted_policy = _policy_with_retrieval_release(
            base_policy,
            selected_release,
            changelog="; ".join(
                child["changelog"]
                for generation in scope_changelogs
                for child in generation["children"]
            ),
        )
        _write_json_artifact(
            Path(args.policy_path),
            accepted_policy.to_payload(),
            expected_identity=validated_paths["output_identities"]["policy"],
            expected_parent_identity=validated_paths[
                "output_parent_identities"
            ]["policy"],
        )
        saved_policy_path = str(args.policy_path)
        result = _published_retrieval_result(result, selected_release)

    release_binding = _policy_with_retrieval_release(
        base_policy,
        selected_release,
        changelog=base_policy.changelog,
    )
    trace_payload = {
        "schema_version": 1,
        "evolution_mode": "retrieval",
        "accepted": result.accepted,
        "selected_release": selected_release.genome.version,
        "release_sha256": release_binding.retrieval_release_sha256,
        "hashes": {
            "train_dev_split_sha256": split_sha256,
            "verifier_sha256": verifier_hash,
            "evaluator_sha256": evaluator_hash,
            "metric_sha256": metric_hash,
            "mutation_model_sha256": mutation_model_hash,
            "harness_sha256": harness_hash,
        },
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "public_regression_tasks": 0,
        "scope_changelogs": scope_changelogs,
        "rejection_reasons": list(result.rejection_reasons),
        "result": result.to_payload(),
    }
    _assert_no_public_regression_ids(trace_payload, public_ids)
    trace_path = Path(args.trace_path)
    _write_json_artifact(
        trace_path,
        trace_payload,
        expected_identity=validated_paths["output_identities"]["trace"],
        expected_parent_identity=validated_paths["output_parent_identities"][
            "trace"
        ],
    )
    progress_path = Path(args.progress_path) if args.progress_path else trace_path.with_name(
        "progress.jsonl"
    )
    progress_encoded = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in result.trace
        ).encode("utf-8")
    _publish_retrieval_output_bytes(
        progress_path,
        progress_encoded,
        expected_identity=validated_paths["output_identities"]["progress"],
        expected_parent_identity=validated_paths["output_parent_identities"][
            "progress"
        ],
    )
    return {
        "evolution_mode": "retrieval",
        "accepted": result.accepted,
        "release_path": str(selected_release.path),
        "release_sha256": release_binding.retrieval_release_sha256,
        "policy_path": saved_policy_path,
        "trace_path": str(trace_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_authority_path": str(authority_path),
        "checkpoint_authority_head_path": str(
            validated_paths["authority_head"]
        ),
        "checkpoint_authority_anchor_path": str(
            validated_paths["authority_anchor"]
        ),
        "checkpoint_authority_anchor": {
            "epoch": committed_authority_anchor[0],
            "head": committed_authority_anchor[1],
        },
        "progress_path": str(progress_path),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "public_regression_tasks": 0,
        "rejection_reasons": list(result.rejection_reasons),
    }


def evolve_command(args) -> dict:
    if args.evolution_mode == "retrieval":
        return _retrieval_evolve_command(args)
    tasks = _task_subset(load_context_tasks(args.tasks_file), args.seed, args.limit)
    train, dev, holdout = _three_way_entity_split(
        tasks,
        args.seed,
        args.dev_fraction,
        args.holdout_fraction,
    )
    manifest_path = _write_split_manifest(
        args.split_manifest_path,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        holdout_fraction=args.holdout_fraction,
        train=train,
        dev=dev,
        holdout=holdout,
    )
    trace_path = Path(args.trace_path)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else trace_path.with_name("checkpoint.json")
    progress_path = Path(args.progress_path) if args.progress_path else trace_path.with_name("progress.jsonl")
    if args.evolution_mode == "source":
        result = _source_evolve_command(
            args,
            train,
            dev,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
        )
        return {
            **result,
            "holdout_tasks": len(holdout),
            "split_manifest_path": str(manifest_path),
            "holdout_status": "reserved_unscored_run_frozen_inference_to_evaluate",
        }
    llm, library, retrieval_library, decision_library, tsfm = _components(args)
    engine = CoEvolutionEngine(
        llm,
        _factory(
            args,
            llm,
            library,
            retrieval_library,
            decision_library,
            tsfm,
            isolate_library=True,
        ),
        CoEvolutionConfig(
            generations=args.generations,
            children_per_generation=args.children,
            mode=args.evolution_mode,
            target=args.evolve_target,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            resume=not args.no_resume,
            successive_halving=args.successive_halving,
            screening_train_tasks=args.screen_train_tasks,
            screening_dev_tasks=args.screen_dev_tasks,
            screening_promote=args.screen_promote,
            screening_tolerance=args.screen_tolerance,
        ),
    )
    seed_policy = _seed_policy(args)
    best, trace = engine.evolve(seed_policy, train, dev)
    best.save(args.policy_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps([asdict(item) for item in trace], indent=2))
    return {
        "best_policy": best.version,
        "evolution_mode": args.evolution_mode,
        "policy_path": args.policy_path,
        "trace_path": str(trace_path),
        "checkpoint_path": str(checkpoint_path),
        "progress_path": str(progress_path),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "holdout_tasks": len(holdout),
        "split_manifest_path": str(manifest_path),
        "holdout_status": "reserved_unscored_run_frozen_inference_to_evaluate",
    }


def _inference_tasks(args) -> tuple[list[ContextTask], str]:
    if args.hidden_test:
        tasks = load_huggingface_context_tasks(labels_public=False)
        source = "hidden_test"
    elif args.public_dev:
        tasks = load_huggingface_context_tasks(labels_public=True)
        source = "public_dev"
    else:
        tasks = load_context_tasks(args.tasks_file, include_unlabeled=True)
        source = "tasks_file"
    if args.split_name != "all":
        if not args.split_manifest:
            raise ValueError("--split-name requires --split-manifest")
        tasks = _select_manifest_split(tasks, args.split_manifest, args.split_name)
    if args.task_id:
        requested = set(args.task_id)
        tasks = [task for task in tasks if task.numeric.task_id in requested]
        missing = requested - {task.numeric.task_id for task in tasks}
        if missing:
            raise ValueError("unknown task IDs: " + ", ".join(sorted(missing)))
    return _task_subset(tasks, args.seed, args.limit), source


def _validate_frozen_retrieval_runtime(args) -> None:
    """Reject runtimes that cannot prove local, cache-free, write-free execution."""
    if getattr(args, "llm_backend", None) not in {"codex", "claude"}:
        raise ValueError(
            "frozen Retrieval inference rejects backends that may download or write caches"
        )
    if getattr(args, "setting", None) in {"tsfm", "combined"}:
        raise ValueError(
            "frozen Retrieval inference rejects TSFM runtimes that may download or write caches"
        )


def _validate_frozen_retrieval_paths(args, release: RetrievalRelease) -> Path:
    output = _canonical_cli_path(
        getattr(args, "output_dir", None) or "outputs/inference/retrieval",
        "frozen Retrieval output",
    )
    configured_root = getattr(args, "output_root", None)
    output_root = _canonical_cli_path(
        configured_root if configured_root else output,
        "approved frozen Retrieval output root",
    )
    if output != output_root and output_root not in output.parents:
        raise ValueError("frozen Retrieval output escapes the approved output root")

    protected: list[tuple[str, Path]] = [
        ("policy", _canonical_cli_path(args.policy_path, "policy")),
        ("task source", _canonical_cli_path(args.tasks_file, "task source")),
        ("Retrieval release", _canonical_cli_path(release.path, "Retrieval release")),
        (
            "Retrieval release root",
            _canonical_cli_path(release.path.parent, "Retrieval release root"),
        ),
        ("coding library", _canonical_cli_path(args.library_path, "coding library")),
        (
            "Retrieval library",
            _canonical_cli_path(args.retrieval_library_path, "Retrieval library"),
        ),
        (
            "Decision library",
            _canonical_cli_path(args.decision_library_path, "Decision library"),
        ),
    ]
    for label, value in (
        ("split manifest", getattr(args, "split_manifest", None)),
        ("seed policy", getattr(args, "seed_policy_path", None)),
        (
            "checkpoint authority",
            getattr(args, "checkpoint_authority_path", None),
        ),
    ):
        if value:
            protected.append((label, _canonical_cli_path(value, label)))
    for label, path in protected:
        if _paths_overlap(output, path):
            raise ValueError(
                f"frozen Retrieval output must be disjoint from protected {label} path"
            )
    return output


def inference_command(args) -> dict:
    """Run one accepted artifact with every learner and scorer disabled by default."""
    if args.hidden_test and args.score_public:
        raise ValueError("--score-public is forbidden with --hidden-test")
    if args.inference == "retrieval":
        _validate_frozen_retrieval_runtime(args)
        if not args.policy_path or not Path(args.policy_path).exists():
            raise FileNotFoundError(
                "frozen retrieval inference requires an existing --policy-path"
            )
        policy = HarnessPolicy.load(args.policy_path)
        release = _load_retrieval_release_for_operator(
            args.retrieval_release_path
        )
        policy = _policy_for_retrieval_release(policy, release)
        output_dir = _validate_frozen_retrieval_paths(args, release)
        retrieval_library = RetrievalSkillLibrary._from_loaded_release(release)
        tasks, _data_source = _inference_tasks(args)
        llm, library, _unused, decision_library, tsfm = _components(
            args,
            retrieval_library_override=retrieval_library,
            disable_llm_cache=True,
        )
        return run_frozen_inference(
            policy,
            tasks,
            _factory(
                args,
                llm,
                library,
                retrieval_library,
                decision_library,
                tsfm,
                isolate_library=True,
                morphology_provider=_ConservativeMorphologyProvider(),
                retrieval_genome=release.genome,
                retrieval_skill_source=retrieval_library,
            ),
            output_dir=output_dir,
            samples=args.samples,
            score_public=args.score_public,
            artifact_kind="retrieval",
        )
    if args.inference in {"prompt", "genome"}:
        tasks, _data_source = _inference_tasks(args)
        if not args.policy_path or not Path(args.policy_path).exists():
            raise FileNotFoundError(
                f"frozen {args.inference} inference requires an existing --policy-path"
            )
        llm, library, retrieval_library, decision_library, tsfm = _components(args)
        policy = HarnessPolicy.load(args.policy_path)
        return run_frozen_inference(
            policy,
            tasks,
            _factory(
                args,
                llm,
                library,
                retrieval_library,
                decision_library,
                tsfm,
                isolate_library=True,
            ),
            output_dir=args.output_dir or f"outputs/inference/{args.inference}",
            samples=args.samples,
            score_public=args.score_public,
            artifact_kind=args.inference,
        )

    data_source = "hidden_test" if args.hidden_test else "public_dev" if args.public_dev else "tasks_file"
    patch_path = Path(args.source_patch_path)
    if not patch_path.exists():
        raise FileNotFoundError(
            f"frozen source inference requires an existing --source-patch-path: {patch_path}"
        )
    repo_root = Path.cwd().resolve()
    runtime_keys = (
        "setting",
        "llm_backend",
        "model_id",
        "device",
        "codex_model",
        "codex_reasoning_effort",
        "codex_timeout",
        "codex_cache_dir",
        "coding_initial_programs",
        "coding_mutations",
        "coding_validation_folds",
        "library_path",
        "retrieval_library_path",
        "decision_library_path",
        "retrieval_mode",
        "retrieval_release_path",
        "chronos_model_id",
        "chronos_device",
        "chronos_cache_dir",
        "chronos_local_files_only",
        "seed",
        "limit",
    )
    runtime = {key: getattr(args, key) for key in runtime_keys}
    for key in (
        "codex_cache_dir",
        "library_path",
        "retrieval_library_path",
        "decision_library_path",
        "retrieval_release_path",
        "chronos_cache_dir",
    ):
        if runtime.get(key):
            runtime[key] = str(Path(runtime[key]).resolve())
    config = {
        "runtime": runtime,
        "data_source": data_source,
        "tasks_file": str(Path(args.tasks_file).resolve()),
        "manifest_path": (
            str(Path(args.split_manifest).resolve()) if args.split_manifest else None
        ),
        "split_name": args.split_name,
        "task_ids": list(args.task_id or ()),
        "policy_path": None,
        "output_dir": str(
            Path(args.output_dir or "outputs/inference/source").resolve()
        ),
        "samples": args.samples,
        "score_public": args.score_public,
    }
    return run_source_inference(
        repo_root=repo_root,
        patch_path=patch_path,
        config=config,
        timeout_seconds=args.source_eval_timeout,
    )


def _source_evolve_command(
    args,
    train: list[ContextTask],
    dev: list[ContextTask],
    *,
    checkpoint_path: Path,
    progress_path: Path,
) -> dict:
    """Run source candidates in detached worktrees; keep the current checkout immutable."""
    repo_root = Path.cwd().resolve()
    tracked_dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        check=False,
    ).returncode
    if tracked_dirty:
        raise RuntimeError("source evolution requires a clean tracked worktree")

    runtime_keys = (
        "setting",
        "llm_backend",
        "model_id",
        "device",
        "codex_model",
        "codex_reasoning_effort",
        "codex_timeout",
        "codex_cache_dir",
        "claude_model",
        "claude_timeout",
        "claude_cache_dir",
        "coding_initial_programs",
        "coding_mutations",
        "coding_validation_folds",
        "retrieval_mode",
        "retrieval_release_path",
        "chronos_model_id",
        "chronos_device",
        "chronos_cache_dir",
        "chronos_local_files_only",
    )
    runtime = {key: getattr(args, key) for key in runtime_keys}
    runtime["codex_cache_dir"] = str((repo_root / args.codex_cache_dir).resolve())
    runtime["claude_cache_dir"] = str((repo_root / args.claude_cache_dir).resolve())
    if args.retrieval_release_path:
        runtime["retrieval_release_path"] = str(
            (repo_root / args.retrieval_release_path).resolve()
        )
    if args.chronos_cache_dir:
        runtime["chronos_cache_dir"] = str(
            (repo_root / args.chronos_cache_dir).resolve()
        )
    evaluation_config = {
        "tasks_file": str(Path(args.tasks_file).resolve()),
        "train_ids": [task.numeric.task_id for task in train],
        "dev_ids": [task.numeric.task_id for task in dev],
        "runtime": runtime,
    }

    def evaluate(worktree: Path) -> SourceEvaluation:
        with tempfile.TemporaryDirectory(prefix="source-eval-config-") as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(evaluation_config), encoding="utf-8")
            environment = dict(os.environ)
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(worktree) + (os.pathsep + existing if existing else "")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolving_loop.source_evolution.source_eval",
                    "--config",
                    str(config_path),
                ],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=args.source_eval_timeout,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout)[-2000:])
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("source evaluator produced no result")
            return SourceEvaluation.from_dict(json.loads(lines[-1]))

    engine = SourceEvolutionEngine(
        repo_root,
        evaluate,
        SourceEvolutionConfig(
            generations=args.generations,
            children_per_generation=args.children,
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning_effort,
            codex_timeout_seconds=args.source_engineer_timeout,
            test_timeout_seconds=args.source_test_timeout,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            resume=not args.no_resume,
        ),
    )
    seed_patch = ""
    if args.seed_source_patch:
        source = Path(args.seed_source_patch)
        if not source.exists():
            raise FileNotFoundError(f"seed source patch does not exist: {source}")
        seed_patch = source.read_text(encoding="utf-8")
    best_patch, trace = engine.evolve(seed_patch)
    patch_path = Path(args.source_patch_path)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(best_patch, encoding="utf-8")
    save_source_trace(args.trace_path, trace)
    return {
        "evolution_mode": "source",
        "source_patch_path": str(patch_path),
        "trace_path": args.trace_path,
        "checkpoint_path": str(checkpoint_path),
        "progress_path": str(progress_path),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "accepted_generations": sum(item.accepted_candidate is not None for item in trace),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_methods:
        print(
            json.dumps(
                {
                    "baselines": list(BASELINE_CHOICES),
                    "evolutions": list(EVOLUTION_CHOICES),
                    "frozen_inference": list(INFERENCE_CHOICES),
                },
                indent=2,
            )
        )
        return
    if args.baseline:
        result = baseline_command(args)
        if result is not None:
            print(json.dumps(result, indent=2))
        return
    if args.evolution:
        args.evolution_mode = args.evolution
        args.codex_reasoning_effort = args.codex_reasoning_effort or "high"
        args.codex_timeout = args.codex_timeout or 900
        args.codex_cache_dir = args.codex_cache_dir or "runs/evolving/codex-cache"
        args.claude_timeout = args.claude_timeout or 900
        args.claude_cache_dir = args.claude_cache_dir or "runs/evolving/claude-cache"
        result = evolve_command(args)
        print(json.dumps(result, indent=2))
        return
    if args.inference:
        args.codex_reasoning_effort = args.codex_reasoning_effort or "high"
        args.codex_timeout = args.codex_timeout or 900
        args.codex_cache_dir = args.codex_cache_dir or "runs/evolving/codex-cache"
        result = inference_command(args)
        print(json.dumps(result, indent=2))
        return
    if args.command is None:
        raise SystemExit("Choose --baseline NAME, --evolution NAME, or a legacy run/evolve command")
    result = run_command(args) if args.command == "run" else evolve_command(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
