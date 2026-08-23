from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from drcik_agent.codex_agents import CodexCLIClient, CodexCLIConfig
from drcik_agent.code_evolution import CodeEvolutionCLIClient, CodeEvolutionCLIConfig
from drcik_agent.co_evolution import (
    AgentPromptBundle,
    CoEvolutionConfig,
    PromptCoEvolutionEngine,
    evaluate_bundle,
)
from drcik_agent.data import load_sample_tasks
from drcik_agent.triad import ThreeAgentForecastSystem, TriadConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal failure-attributed co-evolution on resolved Dr-CiK tasks."
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--keep-elite", type=int, default=2)
    parser.add_argument("--dev-tasks", type=int, default=1)
    parser.add_argument("--backbone", choices=("chronos", "statistical"), default="chronos")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--reasoning-effort", default="high")
    arguments = parser.parse_args()

    tasks = [
        task for task in load_sample_tasks(arguments.sample_dir)
        if task.labels_public and task.future_values is not None
    ]
    if len(tasks) < 2:
        raise SystemExit("At least two resolved tasks are required for train/dev evolution.")
    dev_count = max(1, min(arguments.dev_tasks, len(tasks) - 1))
    train_tasks, dev_tasks = tasks[:-dev_count], tasks[-dev_count:]
    output = Path(arguments.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    worker = CodexCLIClient(CodexCLIConfig(
        model=arguments.codex_model,
        reasoning_effort=arguments.reasoning_effort,
        cache_dir=str(output / "worker-cache"),
    ))
    evolver = CodeEvolutionCLIClient(CodeEvolutionCLIConfig(
        model=arguments.codex_model,
        reasoning_effort=arguments.reasoning_effort,
        cache_dir=str(output / "evolver-cache"),
    ))

    def evaluator(bundle, selected_tasks):
        return evaluate_bundle(
            bundle,
            selected_tasks,
            lambda active_bundle: ThreeAgentForecastSystem(
                TriadConfig(
                    reasoning_agent="codex",
                    backbone=arguments.backbone,
                    learn_from_public_outcomes=False,
                    codex_model=arguments.codex_model,
                    codex_reasoning_effort=arguments.reasoning_effort,
                ),
                codex_client=worker,
                agent_bundle=active_bundle,
            ),
        )

    engine = PromptCoEvolutionEngine(
        evolver,
        evaluator,
        CoEvolutionConfig(
            generations=arguments.generations,
            population_size=arguments.population_size,
            keep_elite=arguments.keep_elite,
        ),
    )
    best, generations = engine.evolve(
        AgentPromptBundle(), train_tasks, dev_tasks
    )
    best.save(output / "best-agent-bundle.json")
    (output / "evolution-log.json").write_text(
        json.dumps([asdict(item) for item in generations], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best bundle: {best.version}")
    print(f"Saved: {output / 'best-agent-bundle.json'}")


if __name__ == "__main__":
    main()
