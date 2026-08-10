"""Command-line entrypoint: evolving-agents evolve-coding | evolve-retrieval | evolve-system | run-baselines."""

from __future__ import annotations

import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from dr_cik.cli import _configure_logging
from dr_cik.llm import LLMClient
from dr_cik.local_llm import QwenClient, QwenConfig

from .bundles import load_bundle
from .cli_common import add_bundle_args, add_evolve_args, add_llm_args, add_task_source_args, resolve_split_file
from .evolve.evaluate import TaskResult
from .evolve.loop import EvolveConfig, evolve, make_task_sampler, select_best
from .evolve.loops import loop_a_score_fn, loop_b_score_fn, loop_c_score_fn
from .evolve.mutate import mutate, mutate_triple
from .harness.baselines import BASELINES, chronos_only, coding_only, frozen_system, mean_metrics, naive_rag, oracle_retrieval
from .harness.context import CallContext
from .harness.datasets import load_drcik_splits
from .harness.run_log import PROXY_NOTE, append_run_record, build_record
from .harness.trace import configure_tracing
from .llm_cache import DEFAULT_CACHE_DIR, CachingLLMClient
from .models import Bundle, BundleTriple

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree, mirroring dr_cik/cli.py's subcommand and logging conventions."""
    parser = argparse.ArgumentParser(prog="evolving-agents", description="Self-evolving three-agent forecasting over Dr-CiK.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-file", default=None, help="Also write logs here; defaults to ./logs/<checkpoint-dir-name>.log")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evolve_coding = subparsers.add_parser("evolve-coding", help="Loop A: evolve the Coding Agent against hindcast error")
    add_task_source_args(evolve_coding)
    add_llm_args(evolve_coding)
    add_evolve_args(evolve_coding)
    evolve_coding.add_argument("--coding-bundle", default=None, help="Seed bundle; defaults to the committed coding/v000.json")
    evolve_coding.add_argument("--n-windows", type=int, default=3, help="Hindcast windows carved per task")
    evolve_coding.add_argument("--dev-limit", type=int, default=10, help="Dev tasks scored per generation for early stopping")

    evolve_retrieval = subparsers.add_parser("evolve-retrieval", help="Loop B: evolve the Retrieval Agent against document labels")
    add_task_source_args(evolve_retrieval)
    add_llm_args(evolve_retrieval)
    add_evolve_args(evolve_retrieval)
    evolve_retrieval.add_argument("--retrieval-bundle", default=None, help="Seed bundle; defaults to retrieval/v000.json")
    evolve_retrieval.add_argument("--frozen-coding-bundle", required=True, help="Loop A's selected coding bundle, held fixed here")
    evolve_retrieval.add_argument("--frozen-decision-bundle", default=None, help="Defaults to the seed decision/v000.json")
    evolve_retrieval.add_argument("--bonus-weight", type=float, default=0.2, help="Weight on downstream forecast improvement; 0 scores label F1 alone")
    evolve_retrieval.add_argument("--n-windows", type=int, default=2)
    evolve_retrieval.add_argument("--dev-limit", type=int, default=10)

    evolve_system = subparsers.add_parser("evolve-system", help="Loop C: evolve all three bundles against end-to-end error")
    add_task_source_args(evolve_system)
    add_llm_args(evolve_system)
    add_evolve_args(evolve_system)
    add_bundle_args(evolve_system)
    evolve_system.add_argument("--n-windows", type=int, default=2)
    evolve_system.add_argument("--dev-limit", type=int, default=10)
    evolve_system.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge evidence_recall term")

    run_baselines = subparsers.add_parser("run-baselines", help="Run one reference system over a split")
    add_task_source_args(run_baselines)
    add_llm_args(run_baselines)
    add_bundle_args(run_baselines)
    run_baselines.add_argument("--baseline", choices=BASELINES, required=True)
    run_baselines.add_argument("--output-dir", required=True)
    run_baselines.add_argument("--split", choices=("evolve", "dev", "test"), default="dev")
    run_baselines.add_argument("--runs-dir", default="runs")
    run_baselines.add_argument("--trace-level", choices=("off", "summary", "full"), default="summary")
    run_baselines.add_argument("--n-windows", type=int, default=2)
    run_baselines.add_argument("--seed", type=int, default=7)
    run_baselines.add_argument("--no-judge", action="store_true")
    return parser


def run_evolve_coding(
    args: argparse.Namespace, worker: LLMClient | None = None, evolver: LLMClient | None = None
) -> tuple[list, str | None]:
    """Run Loop A end to end; inner clients are injectable so the whole path is testable without a GPU."""
    _splits, evolve_tasks, dev_tasks, context, worker_client, evolver_client = _prepare(args, worker, evolver, "coding")
    seed_bundle = load_bundle(args.coding_bundle) if args.coding_bundle else _seed("coding")

    def score_fn(bundle: Bundle, task) -> TaskResult:
        """Score one bundle on one task and append its run record."""
        context.start(task.benchmark_id, agent="coding")
        result = loop_a_score_fn(bundle, task, worker_client, n_windows=args.n_windows)
        append_run_record(
            args.runs_dir,
            "loop_a.jsonl",
            build_record(
                task_id=task.benchmark_id, loop="A", bundle_versions={"coding": bundle.version},
                score=result.score, llm_calls=context.drain(), extra={"trace": result.trace},
            ),
        )
        return result

    records = evolve(
        [seed_bundle],
        make_task_sampler(evolve_tasks, args.minibatch_size, args.seed),
        score_fn,
        partial(_mutate_with, evolver=evolver_client, bundles_dir=args.bundles_dir),
        _evolve_config(args),
        checkpoint_dir=args.checkpoint_dir,
        bundles_dir=args.bundles_dir,
        dev_tasks=dev_tasks,
    )
    return records, select_best(records)


def _mutate_with(parent: Bundle, worst: list[TaskResult], evolver: LLMClient, bundles_dir: str) -> Bundle:
    """Adapt mutate() to the loop's two-argument mutate_fn signature."""
    return mutate(parent, worst, evolver, bundles_dir)


def _wrap(inner, model_id: str, cache_dir: str | None, enable_thinking: bool, device: str | None, on_call=None) -> CachingLLMClient:
    """Build (or wrap an injected) client so the cache -- and therefore on_call -- is always present."""
    return CachingLLMClient(
        inner or QwenClient(QwenConfig(model_id=model_id, device=device, enable_thinking=enable_thinking)),
        model_id=model_id,
        cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        enable_thinking=enable_thinking,
        on_call=on_call,
    )


def _prepare(args: argparse.Namespace, worker, evolver, agent: str):
    """Shared setup for every evolve subcommand: tracing, splits, clients, and the call context."""
    configure_tracing(args.trace_level, runs_dir=args.runs_dir)
    splits = load_drcik_splits(
        data_dir=args.data_dir, sample_dir=args.sample_dir, seed=args.seed, split_file=resolve_split_file(args)
    )
    evolve_tasks = splits.evolve if args.limit is None else splits.evolve[: args.limit]
    dev_tasks = splits.dev if args.dev_limit is None else splits.dev[: args.dev_limit]
    if not evolve_tasks:
        raise SystemExit("the evolve split is empty; check --sample-dir/--data-dir and --split-file")

    context = CallContext(agent=agent)
    worker_client = _wrap(worker, args.worker_model_id, args.cache_dir, True, args.worker_device, on_call=context.record)
    evolver_client = _wrap(evolver, args.evolver_model_id, args.cache_dir, True, args.evolver_device)
    return splits, evolve_tasks, dev_tasks, context, worker_client, evolver_client


def _evolve_config(args: argparse.Namespace) -> EvolveConfig:
    """Build the loop configuration from parsed arguments."""
    return EvolveConfig(
        generations=args.generations, population_size=args.population_size, keep_elite=args.keep_elite,
        stall_patience=args.stall_patience, minibatch_size=args.minibatch_size, seed=args.seed,
    )


def run_evolve_retrieval(args: argparse.Namespace, worker=None, evolver=None) -> tuple[list, str | None]:
    """Run Loop B: evolve the Retrieval Agent while the coding/decision stack stays frozen."""
    _splits, evolve_tasks, dev_tasks, context, worker_client, evolver_client = _prepare(args, worker, evolver, "retrieval")
    seed_bundle = load_bundle(args.retrieval_bundle) if args.retrieval_bundle else _seed("retrieval")
    frozen_coding = load_bundle(args.frozen_coding_bundle)
    frozen_decision = load_bundle(args.frozen_decision_bundle) if args.frozen_decision_bundle else _seed("decision")

    def score_fn(bundle: Bundle, task) -> TaskResult:
        """Score one retrieval bundle on one task and append its run record."""
        context.start(task.benchmark_id, agent="retrieval")
        result = loop_b_score_fn(
            bundle, task, worker_client, frozen_coding, frozen_decision,
            bonus_weight=args.bonus_weight, n_windows=args.n_windows,
        )
        append_run_record(
            args.runs_dir, "loop_b.jsonl",
            build_record(
                task_id=task.benchmark_id, loop="B",
                bundle_versions={"retrieval": bundle.version, "coding": frozen_coding.version, "decision": frozen_decision.version},
                score=result.score, llm_calls=context.drain(), extra={"trace": result.trace},
            ),
        )
        return result

    records = evolve(
        [seed_bundle], make_task_sampler(evolve_tasks, args.minibatch_size, args.seed), score_fn,
        partial(_mutate_with, evolver=evolver_client, bundles_dir=args.bundles_dir),
        _evolve_config(args), checkpoint_dir=args.checkpoint_dir, bundles_dir=args.bundles_dir, dev_tasks=dev_tasks,
    )
    return records, select_best(records)


def run_evolve_system(args: argparse.Namespace, worker=None, evolver=None) -> tuple[list, str | None]:
    """Run Loop C: evolve all three bundles jointly against the end-to-end proxy score."""
    _splits, evolve_tasks, dev_tasks, context, worker_client, evolver_client = _prepare(args, worker, evolver, "system")
    triple = BundleTriple(
        coding=load_bundle(args.coding_bundle), retrieval=load_bundle(args.retrieval_bundle), decision=load_bundle(args.decision_bundle)
    )
    judge = None if args.no_judge else worker_client

    def score_fn(individual: BundleTriple, task) -> TaskResult:
        """Score one triple on one task and append its run record."""
        context.start(task.benchmark_id, agent="system")
        result = loop_c_score_fn(individual, task, worker_client, judge=judge, n_windows=args.n_windows)
        append_run_record(
            args.runs_dir, "loop_c.jsonl",
            build_record(
                task_id=task.benchmark_id, loop="C",
                bundle_versions={
                    "coding": individual.coding.version, "retrieval": individual.retrieval.version, "decision": individual.decision.version,
                },
                score=result.score, llm_calls=context.drain(), extra={"trace": result.trace, "note": PROXY_NOTE},
            ),
        )
        return result

    rng = random.Random(args.seed)
    records = evolve(
        [triple], make_task_sampler(evolve_tasks, args.minibatch_size, args.seed), score_fn,
        lambda parent, worst: mutate_triple(parent, worst, evolver_client, args.bundles_dir, rng=rng),
        _evolve_config(args), checkpoint_dir=args.checkpoint_dir, bundles_dir=args.bundles_dir, dev_tasks=dev_tasks,
    )
    return records, select_best(records)


def run_baselines(args: argparse.Namespace, worker=None, forecaster=None) -> dict:
    """Run one reference system over a split and write summary.json plus per-task records."""
    configure_tracing(args.trace_level, runs_dir=args.runs_dir)
    splits = load_drcik_splits(
        data_dir=args.data_dir, sample_dir=args.sample_dir, seed=args.seed, split_file=resolve_split_file(args)
    )
    tasks = splits.named(args.split)
    tasks = tasks if args.limit is None else tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"the {args.split} split is empty")

    context = CallContext(agent="baseline")
    worker_client = _wrap(worker, args.worker_model_id, args.cache_dir, False, args.worker_device, on_call=context.record)
    judge = None if args.no_judge else worker_client
    coding, retrieval, decision = (load_bundle(getattr(args, f"{name}_bundle")) for name in ("coding", "retrieval", "decision"))

    results = []
    for task in tasks:
        context.start(task.benchmark_id, agent="baseline")
        results.append(_one_baseline(args, task, worker_client, forecaster, coding, retrieval, decision, judge))
        append_run_record(
            args.runs_dir, f"baseline_{args.baseline}.jsonl",
            build_record(
                task_id=task.benchmark_id, loop="baseline", bundle_versions={"coding": coding.version},
                score=-(results[-1].metrics.get("smae") or 0.0), llm_calls=context.drain(),
                extra={"baseline": args.baseline, "metrics": results[-1].metrics},
            ),
        )

    summary = {
        "baseline": args.baseline,
        "split": args.split,
        "num_tasks": len(results),
        "mean_metrics": mean_metrics(results),
        "note": PROXY_NOTE,
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _one_baseline(args, task, llm, forecaster, coding: Bundle, retrieval: Bundle, decision: Bundle, judge):
    """Dispatch a single task to the requested baseline."""
    if args.baseline == "chronos-only":
        from dr_cik.forecasters.chronos import ChronosConfig, ChronosForecaster

        return chronos_only(task, forecaster or ChronosForecaster(ChronosConfig()))
    if args.baseline == "naive-rag":
        return naive_rag(task, llm, judge=judge)
    if args.baseline == "coding-only":
        return coding_only(task, llm, coding, n_windows=args.n_windows)
    if args.baseline == "frozen-system":
        return frozen_system(task, llm, coding, retrieval, decision, judge=judge, n_windows=args.n_windows)
    return oracle_retrieval(task, llm, coding, decision, judge=judge, n_windows=args.n_windows)


def _seed(agent: str) -> Bundle:
    """Load a committed seed bundle by agent name."""
    from .bundles import load_seed

    return load_seed(agent)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the requested subcommand."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    # _configure_logging derives a default path from args.output_dir; the evolve commands have no
    # such flag, so point it at the checkpoint dir without clobbering run-baselines' real one.
    if not getattr(args, "output_dir", None):
        args.output_dir = getattr(args, "checkpoint_dir", None)
    _configure_logging(args)

    runners = {"evolve-coding": ("A", run_evolve_coding), "evolve-retrieval": ("B", run_evolve_retrieval), "evolve-system": ("C", run_evolve_system)}
    if args.command in runners:
        loop, runner = runners[args.command]
        records, best = runner(args)
        print(
            json.dumps(
                {
                    "loop": loop,
                    "generations_run": len(records),
                    "best_individual": best,
                    "dev_scores": [record.dev_score for record in records],
                    "best_evolve_score": max((item.mean_score for record in records for item in record.eval_results), default=None),
                },
                indent=2,
            )
        )
        return

    if args.command == "run-baselines":
        print(json.dumps(run_baselines(args), indent=2))
        return


if __name__ == "__main__":
    main()
