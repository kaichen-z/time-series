from __future__ import annotations

from dataclasses import replace

import pytest

from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.v2.source.contracts import SourceVariantV2
from tests.build_evolution_v2_source_fixture import build_source_case


def test_source_fitness_uses_real_pipeline_and_train_only_requests(tmp_path):
    case = build_source_case(tmp_path)
    result = case.evaluator.train(case.improving_source)
    assert result.feasible and result.mean_gain > 1e-12
    assert len(result.fold_gains) == 2
    assert {entry["agent"] for entry in case.pipeline_trace} >= {"numerical", "retrieval", "decision"}
    for request in case.policy_requests:
        assert set(request) == {"schema_version", "enabled_arms", "step", "seed", "train_reward_by_arm"}


def test_meta_validation_stays_host_only(tmp_path):
    case = build_source_case(tmp_path)
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    assert evidence.passed
    assert all("dev" not in str(request).lower() for request in case.policy_requests)


def test_failed_held_out_validation_rejects_without_changing_train(tmp_path):
    case = build_source_case(tmp_path)
    original_factory = case.pipeline_factory

    def failing_factory(catalog):
        pipeline = original_factory(catalog)
        original_evaluate = pipeline.evaluate

        def evaluate(bundle, tasks, stage):
            result = original_evaluate(bundle, tasks, stage)
            if stage == "dev" and bundle.numerical_release_sha256 != case.seed_bundle.numerical_release_sha256:
                return PackageEvaluation.from_rows(
                    bundle.fingerprint(),
                    tuple(replace(row, final_smae=5.0, final_srmse=5.0,
                                  final_smae_raw=5.0, final_srmse_raw=5.0)
                          for row in result.task_rows),
                    result.expected_task_ids, result.secondary_diagnostics,
                )
            return result

        pipeline.evaluate = evaluate
        return pipeline

    case.pipeline_factory = failing_factory
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    assert not evidence.passed
    assert evidence.reason == "held_out_not_strictly_better"


@pytest.mark.parametrize("field", ("task_id", "entity_name"))
def test_rejects_public_or_cross_split_overlap_before_source_execution(tmp_path, field):
    case = build_source_case(tmp_path)
    task = case.dev_tasks[0]
    numeric = replace(task.numeric, **{field: "public-overlap"})
    case.dev_tasks = (replace(task, numeric=numeric),)
    case.evaluator = None
    from evolving_loop.v2.source.meta import SourceMetaEvaluatorV2
    with pytest.raises(ValueError, match="Public|entity"):
        SourceMetaEvaluatorV2(case)
    assert case.policy_requests == []


def test_replay_and_canary_bind_deterministic_execution(tmp_path):
    case = build_source_case(tmp_path)
    train = case.evaluator.train(case.improving_source)
    assert case.evaluator.replay(case.improving_source, train)
    canary = case.evaluator.canary(case.seed_source, case.improving_source, epoch_seed=17)
    assert canary.passed
    assert canary.commitment_sha256


def test_finalist_bundle_is_selected_by_all_train_evaluations_before_dev(tmp_path):
    """Removing full-Train comparison of either fold child must fail this test."""
    case = build_source_case(tmp_path)
    selected_by_step = SourceVariantV2.child(
        case.seed_source,
        "def choose_arm(request):\n"
        "    return 'numerical' if request['step'] == 0 else 'decision'\n",
        "split_fold_arms",
    )
    original_factory = case.pipeline_factory
    full_train: list[tuple[str, float]] = []

    def recording_factory(catalog):
        pipeline = original_factory(catalog)
        original_evaluate = pipeline.evaluate

        def evaluate(bundle, tasks, stage):
            result = original_evaluate(bundle, tasks, stage)
            if stage == "train" and tuple(tasks) == case.train_tasks:
                full_train.append((bundle.fingerprint(), result.mean_joint))
            return result

        pipeline.evaluate = evaluate
        return pipeline

    case.pipeline_factory = recording_factory
    case.evaluator.train(selected_by_step)

    assert len(full_train) == 2
    expected = min(full_train, key=lambda item: (item[1], item[0]))[0]
    episode = case.evaluator._episodes[(selected_by_step.fingerprint(), 0)]
    assert episode.selected_bundle.fingerprint() == expected
