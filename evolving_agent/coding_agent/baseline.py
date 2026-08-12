"""meta-harness: run tasks through the coding-skill agent, score them, and log results."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evolving_agent.coding_agent.agent import CodingSkillAgent
from evolving_agent.coding_agent.skill_library import SkillLibrary
from evolving_agent.data import DEFAULT_TASKS_FILE, Task, load_tasks
from evolving_agent.llm import LLMClient, QwenClient
from evolving_agent.metrics import score_forecast
from evolving_agent.tracing import TraceEvent, configure, emit


def build_parser() -> argparse.ArgumentParser:
    """CLI flags for a baseline run: task source, mode, and where to write everything."""
    parser = argparse.ArgumentParser(description="Run the coding-skill baseline over numeric forecasting tasks.")
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--mode", choices=("library", "fresh"), required=True, default="library")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library-path", default=None)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    return parser


def select_tasks(tasks: list[Task], seed: int, limit: int | None) -> list[Task]:
    """Shuffle deterministically and optionally truncate; same seed -> same task order every run."""
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    return shuffled if limit is None else shuffled[:limit]


def run_baseline(
    tasks: list[Task],
    mode: str,
    llm: LLMClient,
    library: SkillLibrary | None,
    results_path: str | Path,
) -> dict:
    """Run every task through the agent, score it, log it, and return a first-half/second-half summary."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    agent = CodingSkillAgent(llm, library, mode=mode)

    scores: list[float] = []
    with open(results_path, "w", encoding="utf-8") as results_file:

        for task in tasks:

            emit(TraceEvent(task_id=task.task_id, mode=mode, event_type="task_start"))
            result = agent.run_task(task)
            score = score_forecast(list(task.future_values), list(result.forecast))

            if library is not None and result.action != "fallback":
                library.record_use(result.skill_name, score["primary"])

            results_file.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "action": result.action,
                        "skill_name": result.skill_name,
                        "smape": score["smape"],
                        "mae": score["mae"],
                        "error": result.error,
                    }
                )
                + "\n"
            )
            results_file.flush()
            scores.append(score["primary"])

            emit(
                TraceEvent(
                    task_id=task.task_id,
                    mode=mode,
                    event_type="task_end",
                    detail={
                        "score": score["primary"],
                        "action": result.action,
                        "skill_name": result.skill_name,
                        "error": result.error,
                    },
                )
            )

    return _summarize(mode, scores, library, results_path)


def _summarize(mode: str, scores: list[float], library: SkillLibrary | None, results_path: Path) -> dict:
    """First-half vs second-half mean sMAPE: the direct answer to 'did it get better as the library grew'."""
    half = len(scores) // 2
    first_half = scores[:half]
    second_half = scores[half:]
    return {
        "mode": mode,
        "n_tasks": len(scores),
        "mean_smape": sum(scores) / len(scores) if scores else None,
        "mean_smape_first_half": sum(first_half) / len(first_half) if first_half else None,
        "mean_smape_second_half": sum(second_half) / len(second_half) if second_half else None,
        "skills_saved": len(library) if library is not None else 0,
        "results_path": str(results_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    library_path = Path(args.library_path or Path("runs") / "skill_library.json")
    results_path = Path(args.results_path or Path("runs") / f"{args.mode}_results.jsonl")
    log_file = Path(args.log_file or Path("runs") / f"{args.mode}.log")
    configure(log_file, console_level="INFO")

    tasks = select_tasks(load_tasks(args.tasks_file), seed=args.seed, limit=args.limit)

    llm_kwargs = {}
    
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id
    if args.device:
        llm_kwargs["device"] = args.device

    llm = QwenClient(**llm_kwargs)

    library = SkillLibrary.load(library_path) if args.mode == "library" else None

    summary = run_baseline(tasks, args.mode, llm, library, results_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
