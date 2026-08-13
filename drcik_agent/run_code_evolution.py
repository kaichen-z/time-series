from __future__ import annotations

import argparse
import json
from pathlib import Path

from drcik_agent.code_evolution import (
    CodeEvolutionCLIClient,
    CodeEvolutionCLIConfig,
    CodeEvolutionConfig,
    CodexCodeEvolutionAgent,
)
from drcik_agent.data import load_sample_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one numbers-only Coding Agent evolution.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--task-id", default="task_42")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-programs", type=int, default=3)
    parser.add_argument("--mutations", type=int, default=2)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-cache-dir", default="outputs/codex-cache-code-evolution")
    parser.add_argument("--codex-timeout", type=int, default=600)
    arguments = parser.parse_args()

    task = next(
        item
        for item in load_sample_tasks(arguments.sample_dir)
        if item.benchmark_id == arguments.task_id
    )
    client = CodeEvolutionCLIClient(
        CodeEvolutionCLIConfig(
            model=arguments.codex_model,
            cache_dir=arguments.codex_cache_dir,
            timeout_seconds=arguments.codex_timeout,
            reasoning_effort=arguments.codex_reasoning_effort,
        )
    )
    agent = CodexCodeEvolutionAgent(
        client,
        CodeEvolutionConfig(
            initial_programs=arguments.initial_programs,
            mutations=arguments.mutations,
        ),
    )
    result = agent.run(task)
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")

    print(f"Output: {output}")
    print(f"Initial best: {result.initial_best.program.program_id}")
    print(f"Selected: {result.selected.program.program_id}")
    print(f"Backtest scaled-MAE gain: {result.backtest_improvement:.6f}")
    if result.initial_future_mae is not None:
        print(f"Initial future MAE: {result.initial_future_mae:.6f}")
        print(f"Selected future MAE: {result.selected_future_mae:.6f}")
        print(f"Future MAE gain: {result.future_mae_improvement:.6f}")


if __name__ == "__main__":
    main()
