"""CLI for one-pass harness evaluation and held-out co-evolution."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from evolving_agent.co_evolution import CoEvolutionConfig, CoEvolutionEngine, HarnessPolicy
from evolving_agent.coding_agent.evolution import CodingEvolutionAgent, CodingEvolutionConfig
from evolving_agent.coding_agent.skill_library import SkillLibrary
from evolving_agent.data import ContextTask, DEFAULT_TASKS_FILE, load_context_tasks
from evolving_agent.decision_agent.agent import DecisionAgent
from evolving_agent.decision_agent.skill_library import DecisionSkillLibrary
from evolving_agent.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_agent.llm import CodexCLIClient, CodexCLIConfig, QwenClient
from evolving_agent.retrieval_agent.agent import RetrievalAgent
from evolving_agent.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_agent.skill_learning import OutcomeSkillLearner
from evolving_agent.tsfm import ChronosConfig, ChronosForecaster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evolving time-series agent harness.")
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
            choices=("codex", "qwen"),
            default="codex",
            help="LLM used by all three agents; defaults to this machine's Codex CLI.",
        )
        child.add_argument("--codex-model", default=None)
        child.add_argument("--codex-reasoning-effort", default="high")
        child.add_argument("--codex-timeout", type=int, default=900)
        child.add_argument("--codex-cache-dir", default="runs/evolving/codex-cache")
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
    evolve.add_argument("--dev-fraction", type=float, default=0.25)
    evolve.add_argument("--policy-path", default="runs/evolving/best_policy.json")
    evolve.add_argument("--trace-path", default="runs/evolving/evolution_trace.json")
    return parser


def _task_subset(tasks: list[ContextTask], seed: int, limit: int | None) -> list[ContextTask]:
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    return tasks if limit is None else tasks[:limit]


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
    train, dev = _entity_split(tasks, args.seed, args.dev_fraction)
    if not train or not dev:
        raise ValueError("entity split produced an empty train or dev set")
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
        "policy_path": args.policy_path,
        "trace_path": str(trace_path),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_command(args) if args.command == "run" else evolve_command(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
