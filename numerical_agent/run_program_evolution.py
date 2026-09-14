"""Run and inspect compact whole-program forecasting evolution."""
from __future__ import annotations

import argparse
import json
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.data import DEFAULT_TASKS_FILE

from .program_evolution import (
    ClaudeProgramAdapter,
    CodexProgramAdapter,
    ConcurrentEvolutionController,
    EvolutionRun,
    ProgramEvolutionAgent,
    ProgramGrader,
    QwenProgramAdapter,
    VLLMQwenProgramAdapter,
    load_training_tasks,
)


DEFAULT_SPLIT_FILE = Path(__file__).parents[1] / "splits" / "dr_cik_train_100_test_99.jsonl"
DEFAULT_SEED_FILE = Path(__file__).with_name("program_seed.py")
_EVENT_LOCK = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    """Build the six-command program-evolution CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="create or resume an evolution run")
    _run_directory(start)
    start.add_argument(
        "--backend", choices=("qwen", "qwen-vllm", "codex", "claude"), required=True
    )
    start.add_argument("--model", required=True)
    start.add_argument("--qwen-base-url", default=None)
    start.add_argument("--evaluation-budget", type=int, required=True)
    start.add_argument("--agents", type=int, default=2)
    start.add_argument("--grader-workers", type=int, default=None)
    start.add_argument("--temperature", type=float, default=0.5)
    start.add_argument("--heartbeat-actions", type=int, default=24)
    start.add_argument("--heartbeat-idle-seconds", type=float, default=300.0)
    start.add_argument("--min-actions-before-yield", type=int, default=4)
    start.add_argument("--submit-after-heartbeats", type=int, default=4)
    start.add_argument("--curator-model", default=None)
    start.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    start.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    start.add_argument("--reasoning-effort", default="high")

    status = commands.add_parser("status", help="show compact run progress")
    _run_directory(status)

    log = commands.add_parser("log", help="show recent aggregate evaluation events")
    _run_directory(log)
    log.add_argument("--limit", type=int, default=20)

    show = commands.add_parser("show", help="print a recorded forecasting program")
    _run_directory(show)
    show.add_argument("--commit", default=None)

    stop = commands.add_parser("stop", help="persistently stop new work")
    _run_directory(stop)

    export = commands.add_parser("export", help="create a local branch in the run repository")
    _run_directory(export)
    export.add_argument("--branch", required=True)
    export.add_argument("--commit", default=None)
    return parser


def _run_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command."""
    args = build_parser().parse_args(argv)
    if args.command == "start":
        _start(args)
        return 0
    run = EvolutionRun(args.run_dir)
    if args.command == "status":
        print(json.dumps(run.status(), indent=2, sort_keys=True))
    elif args.command == "log":
        _print_log(run, args.limit)
    elif args.command == "show":
        print(run.show_source(args.commit), end="")
    elif args.command == "stop":
        run.stop()
        print("stopped")
    elif args.command == "export":
        commit_hash = run.export_branch(args.branch, args.commit)
        print(f"created local branch {args.branch} at {commit_hash}")
    return 0


def _start(args: argparse.Namespace) -> None:
    if args.agents <= 0:
        raise ValueError("agents must be positive")
    if args.evaluation_budget <= 0:
        raise ValueError("evaluation budget must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    if args.grader_workers is not None and args.grader_workers <= 0:
        raise ValueError("grader workers must be positive")
    if args.heartbeat_actions <= 0:
        raise ValueError("heartbeat actions must be positive")
    if not math.isfinite(args.heartbeat_idle_seconds) or args.heartbeat_idle_seconds <= 0:
        raise ValueError("heartbeat idle seconds must be positive and finite")
    if args.min_actions_before_yield <= 0:
        raise ValueError("min actions before yield must be positive")
    if args.submit_after_heartbeats <= 0:
        raise ValueError("submit after heartbeats must be positive")
    if args.backend != "qwen-vllm" and args.grader_workers not in {None, 1}:
        raise ValueError("multiple grader workers require qwen-vllm")
    adapter = _adapter(args)
    sessions = args.run_dir / ".evolution" / "public" / "sessions.json"
    resuming = sessions.is_file()
    if resuming:
        run = EvolutionRun(args.run_dir)
        state = run.sessions()
        if int(state["evaluation_budget"]) != args.evaluation_budget:
            raise ValueError("evaluation budget does not match the existing run")
        run.configure_backend(args.backend, args.model)
        run.resume()
    else:
        agent_ids = tuple(f"agent-{index}" for index in range(1, args.agents + 1))
        source = args.seed_file.read_text(encoding="utf-8")
        run = EvolutionRun.create(args.run_dir, source, agent_ids, args.evaluation_budget)
        run.configure_backend(args.backend, args.model)

    curator_model = args.curator_model or args.model
    run.configure_heartbeat(
        args.heartbeat_actions,
        args.heartbeat_idle_seconds,
        curator_model,
    )

    tasks = load_training_tasks(DEFAULT_SPLIT_FILE, args.tasks_file)
    grader = ProgramGrader(tasks)
    agent = ProgramEvolutionAgent(run, adapter)
    agent_ids = tuple(sorted(run.sessions()["agents"]))
    grader_workers = args.grader_workers
    if grader_workers is None:
        grader_workers = min(len(agent_ids), 4) if args.backend == "qwen-vllm" else 1
    if grader_workers > len(agent_ids):
        raise ValueError("grader workers cannot exceed the number of agents")
    if resuming:
        _record_runtime_config(
            run,
            args.temperature,
            grader_workers,
            args.heartbeat_actions,
            args.heartbeat_idle_seconds,
            curator_model,
        )
    else:
        _write_run_config(run, args, grader_workers)
    seed = run.agent_head(agent_ids[0])
    if not run.attempts.records():
        source = (run.repo / run.PROGRAM_FILE).read_text(encoding="utf-8")
        if hasattr(grader, "grade_with_diagnostics"):
            evaluation = grader.grade_with_diagnostics(source)
            metrics = evaluation.metrics
            run.save_diagnostics(seed, None, evaluation)
        else:
            metrics = grader.grade(source)
        run.attempts.record(
            commit_hash=seed,
            parent_hash=None,
            agent_id=agent_ids[0],
            backend="host",
            hypothesis="curated seed",
            metrics=metrics,
        )
        _log_event(run, {"event": "seed", "commit_hash": seed, "score": metrics.score})

    if args.backend == "qwen-vllm":
        if resuming:
            _backfill_reflections(run, adapter, max_workers=min(grader_workers, 4))
        ConcurrentEvolutionController(
            run,
            adapter,
            grader,
            lambda payload: _log_event(run, dict(payload)),
            grader_workers=grader_workers,
        ).run_until_complete()
        print(json.dumps(run.status(), indent=2, sort_keys=True))
        return

    while True:
        progress = run.status()
        if progress["status"] != "running":
            break
        if int(progress["evaluation_count"]) >= int(progress["evaluation_budget"]):
            break
        record = run.grade_next(grader)
        if record is None:
            agent_id = agent_ids[int(progress["evaluation_count"]) % len(agent_ids)]
            proposal = agent.propose(agent_id)
            _log_event(run, {
                "event": "proposed",
                "agent_id": agent_id,
                "commit_hash": proposal.commit_hash,
                "parent_hash": proposal.parent_hash,
            })
            continue
        metrics = record["metrics"]
        _log_event(run, {
            "event": "graded",
            "agent_id": record["agent_id"],
            "commit_hash": record["commit_hash"],
            "score": metrics["score"],
            "status": record["status"],
        })
        if hasattr(adapter, "reflect"):
            try:
                reflected = bool(adapter.reflect(run, str(record["agent_id"]), record))
            except Exception as exc:
                reflected = False
                path = run.public / "logs" / f"{record['agent_id']}-reflection.log"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(f"incomplete reflection: {type(exc).__name__}: {exc}\n")
            _log_event(run, {
                "event": "reflected" if reflected else "reflection_incomplete",
                "agent_id": record["agent_id"],
                "commit_hash": record["commit_hash"],
            })
        else:
            run.write_note(
                str(record["agent_id"]),
                str(record["commit_hash"]),
                f"Evaluation {record['status']}",
                (
                    f"Composite score {float(metrics['score']):.6f}; "
                    f"mean sMAE {float(metrics['mean_smae']):.6f}; "
                    f"mean sRMSE {float(metrics['mean_srmse']):.6f}; "
                    f"coverage {float(metrics['coverage']):.3f}."
                ),
            )
    print(json.dumps(run.status(), indent=2, sort_keys=True))


def _adapter(args: argparse.Namespace):
    if args.backend == "qwen":
        return QwenProgramAdapter(
            args.model,
            temperature=args.temperature,
            max_actions=args.heartbeat_actions,
            heartbeat_idle_seconds=args.heartbeat_idle_seconds,
            min_actions_before_yield=args.min_actions_before_yield,
            submit_after_heartbeats=args.submit_after_heartbeats,
            curator_model=args.curator_model,
        )
    if args.backend == "qwen-vllm":
        if not args.qwen_base_url:
            raise ValueError("--qwen-base-url is required for qwen-vllm")
        return VLLMQwenProgramAdapter(
            args.model,
            args.qwen_base_url,
            temperature=args.temperature,
            max_actions=args.heartbeat_actions,
            heartbeat_idle_seconds=args.heartbeat_idle_seconds,
            min_actions_before_yield=args.min_actions_before_yield,
            submit_after_heartbeats=args.submit_after_heartbeats,
            curator_model=args.curator_model,
        )
    if args.backend == "codex":
        return CodexProgramAdapter(args.model, reasoning_effort=args.reasoning_effort)
    return ClaudeProgramAdapter(args.model)


def _write_run_config(
    run: EvolutionRun, args: argparse.Namespace, grader_workers: int
) -> None:
    payload = {
        "schema_version": 1,
        "split_file": str(DEFAULT_SPLIT_FILE.resolve()),
        "tasks_file": str(args.tasks_file.resolve()),
        "seed_file": str(args.seed_file.resolve()),
        "temperature": args.temperature,
        "grader_workers": grader_workers,
        "heartbeat_actions": args.heartbeat_actions,
        "heartbeat_idle_seconds": args.heartbeat_idle_seconds,
        "curator_model": args.curator_model or args.model,
    }
    path = run.private / "taskdata" / "config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_runtime_config(
    run: EvolutionRun,
    temperature: float,
    grader_workers: int,
    heartbeat_actions: int,
    heartbeat_idle_seconds: float,
    curator_model: str,
) -> None:
    """Record model and grader settings when resuming an older run."""
    path = run.private / "taskdata" / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["temperature"] = temperature
    payload["grader_workers"] = grader_workers
    payload["heartbeat_actions"] = heartbeat_actions
    payload["heartbeat_idle_seconds"] = heartbeat_idle_seconds
    payload["curator_model"] = curator_model
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _backfill_reflections(
    run: EvolutionRun,
    adapter: VLLMQwenProgramAdapter,
    *,
    max_workers: int = 4,
) -> dict[str, int]:
    """Retry incomplete child memory concurrently across persistent agents."""
    if isinstance(max_workers, bool) or max_workers <= 0:
        raise ValueError("backfill workers must be positive")
    incomplete = [
        record
        for record in run.attempts.records()
        if record["parent_hash"] is not None and not run.reflection_complete(record)
    ]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in incomplete:
        groups[str(record["agent_id"])].append(record)

    def backfill_agent(records: list[dict[str, object]]) -> int:
        completed = 0
        for record in records:
            commit_hash = str(record["commit_hash"])
            try:
                adapter.reflect(run, str(record["agent_id"]), record)
            except Exception as exc:
                path = run.public / "logs" / f"{record['agent_id']}-reflection.log"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(f"incomplete backfill: {type(exc).__name__}: {exc}\n")
            reflected = run.reflection_complete(record)
            completed += int(reflected)
            _log_event(run, {
                "event": "reflection_backfilled" if reflected else "reflection_incomplete",
                "agent_id": record["agent_id"],
                "commit_hash": commit_hash,
            })
        return completed

    completed = 0
    if groups:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(groups))) as executor:
            completed = sum(executor.map(backfill_agent, groups.values()))
    remaining = sum(
        record["parent_hash"] is not None and not run.reflection_complete(record)
        for record in run.attempts.records()
    )
    summary = {
        "attempted": len(incomplete),
        "completed": completed,
        "failed": len(incomplete) - completed,
        "remaining": remaining,
    }
    _log_event(run, {"event": "reflection_backfill_summary", **summary})
    return summary


def _log_event(run: EvolutionRun, payload: dict[str, object]) -> None:
    path = run.public / "logs" / "events.log"
    with _EVENT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _print_log(run: EvolutionRun, limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    path = run.public / "logs" / "events.log"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for line in lines[-limit:]:
        payload = json.loads(line)
        print(" ".join(f"{key}={value}" for key, value in sorted(payload.items())))


if __name__ == "__main__":
    raise SystemExit(main())
