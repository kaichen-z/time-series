"""The generic evaluate -> select -> mutate loop, reused by Loops A, B, and C."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dr_cik.llm import LLMClient
from dr_cik.models import ForecastTask

from ..bundles import save_bundle
from ..models import Bundle, BundleTriple
from .checkpoint import GenerationRecord, latest_generation, load_generation, save_generation
from .evaluate import EvalResult, ScoreFn, TaskResult, evaluate, evaluate_population, individual_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolveConfig:
    """Population size, budget, and the early-stop rule for one evolve run."""

    generations: int = 10
    population_size: int = 6
    keep_elite: int = 2
    stall_patience: int = 3
    minibatch_size: int = 20
    worst_n: int = 5
    seed: int = 7


MutateFn = Callable[[Any, list[TaskResult]], Any]


def make_task_sampler(tasks: list[ForecastTask], minibatch_size: int, seed: int) -> Callable[[int], list[ForecastTask]]:
    """Return a sampler drawing a fresh, deterministic minibatch per generation."""

    def sample(generation: int) -> list[ForecastTask]:
        """Draw this generation's minibatch."""
        if minibatch_size <= 0 or minibatch_size >= len(tasks):
            return list(tasks)
        return random.Random(seed + generation).sample(tasks, minibatch_size)

    return sample


def _save_all(population: list[Any], bundles_dir: str | Path) -> dict[str, str]:
    """Persist every individual's bundles and return their paths by individual id."""
    paths: dict[str, str] = {}
    for individual in population:
        parts = [individual] if isinstance(individual, Bundle) else [individual.coding, individual.retrieval, individual.decision]
        paths[individual_id(individual)] = ",".join(str(save_bundle(part, bundles_dir)) for part in parts)
    return paths


def evolve(
    seed_individuals: list[Any],
    task_sampler: Callable[[int], list[ForecastTask]],
    score_fn: ScoreFn,
    mutate_fn: MutateFn,
    config: EvolveConfig,
    checkpoint_dir: str | Path,
    bundles_dir: str | Path,
    dev_tasks: list[ForecastTask] | None = None,
    on_task: Callable[[TaskResult], None] | None = None,
) -> list[GenerationRecord]:
    """Run the loop, checkpointing each generation and resuming from the last completed one.

    Selection uses only the evolve-split minibatch; the dev score is computed for early stopping
    and final model choice, never to decide which individuals survive or get mutated.
    """
    population = list(seed_individuals)
    if not population:
        raise ValueError("at least one seed individual is required")

    records: list[GenerationRecord] = []
    resume_from = latest_generation(checkpoint_dir)
    dev_history: list[float] = []

    for generation in range(config.generations):
        if generation <= resume_from:
            existing = load_generation(checkpoint_dir, generation)
            if existing is not None:
                logger.info("generation %d already checkpointed, skipping", generation)
                records.append(existing)
                if existing.dev_score is not None:
                    dev_history.append(existing.dev_score)
                continue

        # Fill out generation 0 by mutating the seeds, so the first evaluation sees real variety.
        while len(population) < config.population_size:
            population.append(mutate_fn(population[len(population) % len(seed_individuals)], []))

        tasks = task_sampler(generation)
        logger.info("generation %d: evaluating %d individual(s) on %d task(s)", generation, len(population), len(tasks))
        results = evaluate_population(population, tasks, score_fn, worst_n=config.worst_n, on_task=on_task)

        order = sorted(range(len(population)), key=lambda index: results[index].mean_score, reverse=True)
        elite = [population[index] for index in order[: config.keep_elite]]
        elite_results = [results[index] for index in order[: config.keep_elite]]

        dev_score = None
        if dev_tasks:
            dev_score = evaluate(elite[0], dev_tasks, score_fn, worst_n=config.worst_n).mean_score
            dev_history.append(dev_score)
        stalled = _is_stalled(dev_history, config.stall_patience)

        record = GenerationRecord(
            generation=generation,
            population=tuple(individual_id(individual) for individual in population),
            eval_results=tuple(results),
            elite=tuple(individual_id(individual) for individual in elite),
            dev_score=dev_score,
            stalled=stalled,
            bundle_paths=_save_all(population, bundles_dir),
        )
        save_generation(checkpoint_dir, record)
        records.append(record)
        logger.info(
            "generation %d: best=%.4f (%s) dev=%s", generation, elite_results[0].mean_score, record.elite[0],
            f"{dev_score:.4f}" if dev_score is not None else "n/a",
        )

        if stalled:
            logger.info("dev score stalled for %d generation(s), stopping early", config.stall_patience)
            break
        if generation == config.generations - 1:
            break

        children = []
        while len(elite) + len(children) < config.population_size:
            parent = elite[len(children) % len(elite)]
            parent_result = elite_results[len(children) % len(elite)]
            children.append(mutate_fn(parent, list(parent_result.worst)))
        population = elite + children

    return records


def _is_stalled(dev_history: list[float], patience: int) -> bool:
    """Report whether the dev score has failed to improve for `patience` consecutive generations."""
    if patience <= 0 or len(dev_history) < patience + 1:
        return False
    best_before = max(dev_history[: -patience])
    return all(score <= best_before for score in dev_history[-patience:])


def select_best(records: list[GenerationRecord]) -> str | None:
    """Return the elite individual id from the best-scoring generation, preferring dev score."""
    scored = [record for record in records if record.dev_score is not None]
    if scored:
        return max(scored, key=lambda record: record.dev_score).elite[0]
    ranked = [record for record in records if record.eval_results]
    if not ranked:
        return None
    return max(ranked, key=lambda record: max(item.mean_score for item in record.eval_results)).elite[0]
