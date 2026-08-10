"""Loop mechanics: selection, elitism, minibatching, early stop, and checkpoint resume."""

from __future__ import annotations

import dataclasses

import pytest
from dr_cik.models import ForecastTask

from evolving_agents.bundles import load_seed
from evolving_agents.evolve.checkpoint import latest_generation, load_generation, save_generation
from evolving_agents.evolve.evaluate import EvalResult, TaskResult, evaluate, evaluate_population
from evolving_agents.evolve.loop import EvolveConfig, evolve, make_task_sampler, select_best

SEED = load_seed("coding")


def _task(index: int) -> ForecastTask:
    return ForecastTask(
        benchmark_id=f"task_{index}",
        entity_name=f"Entity{index % 3}",
        target_name="t",
        target_description="d",
        frequency="H",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=("a", "b"),
        history_values=(1.0, 2.0),
        future_timestamps=("c", "d"),
        future_values=(3.0, 4.0),
        documents=(),
        gt_evidence=(),
        labels_public=True,
    )


TASKS = [_task(index) for index in range(10)]


def _quality(bundle) -> float:
    """Score a bundle by a number embedded in its system prompt, so tests can steer evolution."""
    try:
        return float(bundle.system_prompt.split("quality=")[1].split()[0])
    except (IndexError, ValueError):
        return 0.0


def _score_fn(bundle, task) -> TaskResult:
    return TaskResult(task_id=task.benchmark_id, score=_quality(bundle), trace={"quality": _quality(bundle)})


def _improving_mutate(counter: list[int]):
    """Return a mutate_fn that makes each child strictly better than the last."""

    def mutate_fn(parent, worst):
        counter[0] += 1
        return dataclasses.replace(
            parent,
            bundle_id=f"coding/v{counter[0]:03d}",
            version=f"v{counter[0]:03d}",
            parent=parent.version,
            system_prompt=f"quality={counter[0]}",
        )

    return mutate_fn


def _flat_mutate(counter: list[int]):
    """Return a mutate_fn whose children never improve, to exercise the stall rule."""

    def mutate_fn(parent, worst):
        counter[0] += 1
        return dataclasses.replace(
            parent, bundle_id=f"coding/v{counter[0]:03d}", version=f"v{counter[0]:03d}", parent=parent.version, system_prompt="quality=1"
        )

    return mutate_fn


def test_evaluate_reports_mean_and_worst() -> None:
    result = evaluate(SEED, TASKS[:4], lambda bundle, task: TaskResult(task_id=task.benchmark_id, score=-int(task.benchmark_id[-1])))
    assert result.mean_score == pytest.approx(-1.5)
    assert [item.task_id for item in result.worst][:2] == ["task_3", "task_2"]


def test_evaluate_population_scores_everyone() -> None:
    results = evaluate_population([SEED, SEED], TASKS[:2], _score_fn)
    assert len(results) == 2
    assert all(isinstance(item, EvalResult) for item in results)


def test_task_sampler_is_deterministic_and_varies_by_generation() -> None:
    sampler = make_task_sampler(TASKS, minibatch_size=4, seed=7)
    assert [task.benchmark_id for task in sampler(0)] == [task.benchmark_id for task in sampler(0)]
    assert [task.benchmark_id for task in sampler(0)] != [task.benchmark_id for task in sampler(1)]
    assert len(sampler(0)) == 4


def test_sampler_returns_everything_when_the_minibatch_is_large() -> None:
    assert len(make_task_sampler(TASKS, minibatch_size=999, seed=7)(0)) == len(TASKS)


def test_evolution_improves_and_checkpoints_every_generation(tmp_path) -> None:
    records = evolve(
        [SEED],
        make_task_sampler(TASKS, 4, 7),
        _score_fn,
        _improving_mutate([0]),
        EvolveConfig(generations=3, population_size=3, keep_elite=1, stall_patience=0),
        checkpoint_dir=tmp_path / "ckpt",
        bundles_dir=tmp_path / "bundles",
    )
    assert len(records) == 3
    best_per_generation = [max(item.mean_score for item in record.eval_results) for record in records]
    assert best_per_generation == sorted(best_per_generation)
    assert latest_generation(tmp_path / "ckpt") == 2
    assert load_generation(tmp_path / "ckpt", 1).generation == 1


def test_population_is_filled_to_size_before_the_first_evaluation(tmp_path) -> None:
    records = evolve(
        [SEED],
        make_task_sampler(TASKS, 2, 7),
        _score_fn,
        _improving_mutate([0]),
        EvolveConfig(generations=1, population_size=4, keep_elite=2, stall_patience=0),
        checkpoint_dir=tmp_path / "ckpt",
        bundles_dir=tmp_path / "bundles",
    )
    assert len(records[0].population) == 4


def test_resume_skips_completed_generations(tmp_path) -> None:
    config = EvolveConfig(generations=2, population_size=2, keep_elite=1, stall_patience=0)
    first_counter = [0]
    evolve([SEED], make_task_sampler(TASKS, 2, 7), _score_fn, _improving_mutate(first_counter), config,
           checkpoint_dir=tmp_path / "ckpt", bundles_dir=tmp_path / "bundles")
    calls_first_run = first_counter[0]

    second_counter = [0]
    records = evolve([SEED], make_task_sampler(TASKS, 2, 7), _score_fn, _improving_mutate(second_counter), config,
                     checkpoint_dir=tmp_path / "ckpt", bundles_dir=tmp_path / "bundles")
    assert second_counter[0] == 0  # nothing was re-mutated; both generations came from disk
    assert calls_first_run > 0
    assert len(records) == 2


def test_dev_score_is_recorded_without_driving_selection(tmp_path) -> None:
    records = evolve(
        [SEED],
        make_task_sampler(TASKS, 2, 7),
        _score_fn,
        _improving_mutate([0]),
        EvolveConfig(generations=2, population_size=2, keep_elite=1, stall_patience=0),
        checkpoint_dir=tmp_path / "ckpt",
        bundles_dir=tmp_path / "bundles",
        dev_tasks=TASKS[:3],
    )
    assert all(record.dev_score is not None for record in records)


def test_early_stop_fires_when_the_dev_score_stalls(tmp_path) -> None:
    records = evolve(
        [SEED],
        make_task_sampler(TASKS, 2, 7),
        _score_fn,
        _flat_mutate([0]),
        EvolveConfig(generations=10, population_size=2, keep_elite=1, stall_patience=2),
        checkpoint_dir=tmp_path / "ckpt",
        bundles_dir=tmp_path / "bundles",
        dev_tasks=TASKS[:3],
    )
    assert len(records) < 10
    assert records[-1].stalled


def test_select_best_prefers_the_best_dev_generation() -> None:
    from evolving_agents.evolve.checkpoint import GenerationRecord

    records = [
        GenerationRecord(generation=0, population=("a",), eval_results=(EvalResult("a", 1.0, ()),), elite=("a",), dev_score=0.1),
        GenerationRecord(generation=1, population=("b",), eval_results=(EvalResult("b", 9.0, ()),), elite=("b",), dev_score=0.9),
        GenerationRecord(generation=2, population=("c",), eval_results=(EvalResult("c", 99.0, ()),), elite=("c",), dev_score=0.2),
    ]
    assert select_best(records) == "b"  # not "c", whose evolve-split score was highest


def test_evolve_requires_a_seed(tmp_path) -> None:
    with pytest.raises(ValueError):
        evolve([], make_task_sampler(TASKS, 2, 7), _score_fn, _improving_mutate([0]), EvolveConfig(),
               checkpoint_dir=tmp_path, bundles_dir=tmp_path)


def test_on_task_callback_sees_every_scored_task(tmp_path) -> None:
    seen: list[TaskResult] = []
    evolve([SEED], make_task_sampler(TASKS, 3, 7), _score_fn, _improving_mutate([0]),
           EvolveConfig(generations=1, population_size=2, keep_elite=1, stall_patience=0),
           checkpoint_dir=tmp_path / "ckpt", bundles_dir=tmp_path / "bundles", on_task=seen.append)
    assert len(seen) == 6  # 2 individuals x 3 tasks


def test_corrupt_checkpoint_is_treated_as_incomplete(tmp_path) -> None:
    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "gen_000.json").write_text("{ not json", encoding="utf-8")
    assert load_generation(directory, 0) is None
