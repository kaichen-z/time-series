"""CLI for one-pass harness evaluation and held-out co-evolution."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from evolving_agent.co_evolution import (
    CoEvolutionConfig,
    CoEvolutionEngine,
    HarnessPolicy,
)
from evolving_agent.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_agent.coding_agent.skill_library import SkillLibrary
from evolving_agent.data import (
    ContextTask,
    DEFAULT_TASKS_FILE,
    load_context_tasks,
    load_huggingface_context_tasks,
)
from evolving_agent.decision_agent.agent import DecisionAgent
from evolving_agent.decision_agent.skill_library import DecisionSkillLibrary
from evolving_agent.frozen_inference import run_frozen_inference
from evolving_agent.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_agent.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig, QwenClient
from evolving_agent.retrieval_agent.agent import RetrievalAgent
from evolving_agent.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_agent.skill_learning import OutcomeSkillLearner
from evolving_agent.source_evolution import (
    SourceEvaluation,
    SourceEvolutionConfig,
    SourceEvolutionEngine,
    save_source_trace,
)
from evolving_agent.source_inference import run_source_inference
from evolving_agent.tsfm import ChronosConfig, ChronosForecaster

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
EVOLUTION_CHOICES = ("prompt", "genome", "source")
INFERENCE_CHOICES = EVOLUTION_CHOICES


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


def _add_unified_evolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--setting", choices=("llm_only", "statistics", "tsfm", "combined"), default="statistics")
    parser.add_argument("--llm-backend", choices=("codex", "qwen", "claude"), default="codex")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--coding-initial-programs", type=int, default=3)
    parser.add_argument("--coding-mutations", type=int, default=1)
    parser.add_argument("--coding-validation-folds", type=int, default=3)
    parser.add_argument("--seed-policy-path", default=None)
    parser.add_argument("--library-path", default="runs/evolving/skills.json")
    parser.add_argument("--retrieval-library-path", default="runs/evolving/retrieval_skills.json")
    parser.add_argument("--decision-library-path", default="runs/evolving/decision_skills.json")
    parser.add_argument("--chronos-device", default="cpu")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--children", type=int, default=2)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
    evolve.add_argument(
        "--evolution-mode",
        choices=("prompt", "genome", "source"),
        default="genome",
        help=(
            "prompt changes one role prompt; genome co-evolves prompts/budgets/topology; "
            "source edits agent/orchestration Python in isolated Git worktrees."
        ),
    )
    evolve.add_argument("--dev-fraction", type=float, default=0.25)
    evolve.add_argument("--holdout-fraction", type=float, default=0.20)
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
        from evolving_agent.coding_agent.baseline import main as coding_baseline_main

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


def _components(args):
    if args.llm_backend == "codex":
        llm = CodexCLIClient(
            CodexCLIConfig(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning_effort,
                timeout_seconds=args.codex_timeout,
                cache_dir=args.codex_cache_dir,
            )
        )
    elif args.llm_backend == "claude":
        llm = ClaudeCLIClient(
            ClaudeCLIConfig(
                model=args.claude_model,
                timeout_seconds=args.claude_timeout,
                cache_dir=args.claude_cache_dir,
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
    retrieval_library = RetrievalSkillLibrary.load(args.retrieval_library_path)
    decision_library = DecisionSkillLibrary.load(args.decision_library_path)
    tsfm = None
    if args.setting in {"tsfm", "combined"}:
        tsfm = ChronosForecaster(
            ChronosConfig(
                model_id=args.chronos_model_id,
                device_map=args.chronos_device,
                cache_dir=args.chronos_cache_dir,
                local_files_only=args.chronos_local_files_only,
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
):
    def build(policy: HarnessPolicy) -> EvolvingForecastHarness:
        task_library = library.clone(persist=False) if isolate_library else library
        task_retrieval_library = (
            retrieval_library.clone(persist=False) if isolate_library else retrieval_library
        )
        task_decision_library = (
            decision_library.clone(persist=False) if isolate_library else decision_library
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
            ),
            tsfm_forecaster=tsfm,
            generation_prompt=policy.coding_generation_prompt,
            revision_prompt=policy.coding_revision_prompt,
        )
        return EvolvingForecastHarness(
            coding,
            RetrievalAgent(
                llm,
                task_retrieval_library,
                prompt=policy.retrieval_prompt,
            ),
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
                enable_evidence_adjustments=policy.enable_evidence_adjustments,
                max_evidence_adjustments=policy.max_evidence_adjustments,
                decision_aggregation=policy.decision_aggregation,
            ),
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
                                "hindcast_smape": item.hindcast_smape,
                            }
                            for item in result.coding.candidates
                        ],
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
    return {
        "n_tasks": len(outcomes),
        "mean_final_smape": sum(item.final_smape for item in outcomes) / len(outcomes),
        "results_path": str(destination),
        "skills_saved": len(library),
        "retrieval_skills_saved": len(retrieval_library),
        "decision_skills_saved": len(decision_library),
        "online_skill_learning": args.learn_from_public_outcomes,
    }


def evolve_command(args) -> dict:
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
    if args.evolution_mode == "source":
        result = _source_evolve_command(args, train, dev)
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
        ),
    )
    seed_policy = _seed_policy(args)
    best, trace = engine.evolve(seed_policy, train, dev)
    best.save(args.policy_path)
    trace_path = Path(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps([asdict(item) for item in trace], indent=2))
    return {
        "best_policy": best.version,
        "evolution_mode": args.evolution_mode,
        "policy_path": args.policy_path,
        "trace_path": str(trace_path),
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


def inference_command(args) -> dict:
    """Run one accepted artifact with every learner and scorer disabled by default."""
    if args.hidden_test and args.score_public:
        raise ValueError("--score-public is forbidden with --hidden-test")
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
        "chronos_model_id",
        "chronos_device",
        "chronos_cache_dir",
        "chronos_local_files_only",
    )
    runtime = {key: getattr(args, key) for key in runtime_keys}
    runtime["codex_cache_dir"] = str((repo_root / args.codex_cache_dir).resolve())
    runtime["claude_cache_dir"] = str((repo_root / args.claude_cache_dir).resolve())
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
                    "evolving_agent.source_eval",
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
