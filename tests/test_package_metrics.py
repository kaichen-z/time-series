from __future__ import annotations

from dataclasses import replace

import pytest

from common.metrics import linear_quantile
from evolving_loop.package_metrics import (
    PackageEvaluation,
    PackageGateConfig,
    PackageTaskScore,
    package_full_gate_failures,
    package_rank_key,
    package_screen_failures,
)


def _score(
    task_id: str,
    *,
    smae: float = 1.0,
    srmse: float = 1.0,
    smae_raw: float | None = None,
    srmse_raw: float | None = None,
    invalid_count: int = 0,
    fallback_count: int = 0,
) -> PackageTaskScore:
    raw_smae = smae if smae_raw is None else smae_raw
    raw_srmse = srmse if srmse_raw is None else srmse_raw
    return PackageTaskScore(
        task_id=task_id,
        entity_name=f"entity_{task_id}",
        final_smae=smae,
        final_srmse=srmse,
        final_smae_raw=raw_smae,
        final_srmse_raw=raw_srmse,
        smae_clipped=raw_smae != smae,
        srmse_clipped=raw_srmse != srmse,
        invalid_count=invalid_count,
        catastrophic_count=int(raw_smae > 10.0 or raw_srmse > 10.0),
        fallback_count=fallback_count,
        selected_candidate_id="candidate",
        numerical_package_sha256="1" * 64,
        final_retrieval_sha256="2" * 64,
        final_decision_sha256="3" * 64,
    )


def _score_rows() -> tuple[PackageTaskScore, ...]:
    return (
        _score("task_1", smae=0.5, srmse=0.6),
        _score("task_2", smae=1.0, srmse=1.1, invalid_count=1),
        _score("task_3", smae=1.5, srmse=1.7, fallback_count=1),
        _score("task_4", smae=5.0, srmse=5.0, smae_raw=12.0),
    )


def _evaluation(rows: tuple[PackageTaskScore, ...]) -> PackageEvaluation:
    return PackageEvaluation.from_rows(
        "a" * 64,
        rows,
        expected_task_ids=tuple(row.task_id for row in rows),
    )


def _uniform_evaluation(value: float) -> PackageEvaluation:
    return _evaluation(
        tuple(
            _score(f"task_{index}", smae=value, srmse=value)
            for index in range(5)
        )
    )


def test_package_evaluation_aggregates_both_scaled_metrics_and_failure_burden():
    rows = _score_rows()
    evaluation = PackageEvaluation.from_rows(
        "a" * 64,
        rows,
        expected_task_ids=tuple(row.task_id for row in rows),
    )
    assert evaluation.task_count == 4
    assert evaluation.coverage == 1.0
    assert evaluation.mean_joint == pytest.approx(
        (evaluation.mean_smae + evaluation.mean_srmse) / 2.0
    )
    assert evaluation.p90_srmse == pytest.approx(
        linear_quantile([row.final_srmse for row in rows], 0.90)
    )
    assert evaluation.p95_srmse == pytest.approx(
        linear_quantile([row.final_srmse for row in rows], 0.95)
    )
    assert evaluation.invalid_count == 1
    assert evaluation.fallback_count == 1
    assert evaluation.clipped_count == 1


def test_package_evaluation_records_missing_membership_and_freezes_diagnostics():
    evaluation = PackageEvaluation.from_rows(
        "a" * 64,
        (_score("task_1"),),
        expected_task_ids=("task_1", "task_2"),
        secondary_diagnostics={"retrieval_supporting_recall": 0.5},
    )

    assert evaluation.coverage == 0.5
    assert evaluation.missing_task_ids == ("task_2",)
    with pytest.raises(TypeError):
        evaluation.secondary_diagnostics["retrieval_supporting_recall"] = 1.0


def test_result_bytes_exclude_only_candidate_publication_identity():
    evaluation = _evaluation((_score("task_1"),))
    changed_candidate = replace(evaluation, candidate_sha256="b" * 64)
    changed_result = replace(
        evaluation,
        task_rows=(replace(evaluation.task_rows[0], final_retrieval_sha256="9" * 64),),
    )

    assert evaluation.result_bytes() == changed_candidate.result_bytes()
    assert evaluation.result_bytes() != changed_result.result_bytes()


def test_full_gate_rejects_srmse_tail_regression_even_when_means_improve():
    parent = _uniform_evaluation(1.0)
    child_rows = tuple(
        _score(f"task_{index}", smae=0.90, srmse=value)
        for index, value in enumerate((0.80, 0.80, 0.80, 0.80, 1.30))
    )
    child = _evaluation(child_rows)

    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="calibration",
    )

    assert child.mean_srmse < parent.mean_srmse
    assert "p95_srmse" in failures


def test_screen_requires_pareto_safety_but_not_half_percent_gain():
    parent = _uniform_evaluation(1.0)
    child = _uniform_evaluation(0.999)
    assert package_screen_failures(child, parent, PackageGateConfig()) == ()


def test_full_gate_requires_half_percent_joint_gain():
    parent = _uniform_evaluation(1.0)
    child = _uniform_evaluation(0.996)
    assert "minimum_relative_joint_gain" in package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="calibration",
    )


class _FiveFoldManifest:
    task_fold_map = {f"task_{index}": index for index in range(5)}


def test_build_gate_requires_four_safe_and_three_improving_folds():
    parent = _uniform_evaluation(1.0)
    child = _evaluation(
        tuple(
            _score(f"task_{index}", smae=value, srmse=value)
            for index, value in enumerate((0.8, 0.8, 1.0, 1.1, 1.1))
        )
    )
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="build",
        fold_manifest=_FiveFoldManifest(),
    )
    assert "nonregressing_folds" in failures
    assert "improving_folds" in failures


def test_task_185_shaped_overlay_is_rejected_by_joint_regret():
    parent = _evaluation(
        (_score("task_185", smae=0.3147676772, srmse=0.5355035986),)
    )
    child = _evaluation(
        (_score("task_185", smae=1.1801735667, srmse=1.4480493045),)
    )
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="calibration",
    )
    assert "maximum_task_joint_regret" in failures


def test_child_must_remain_pareto_safe_against_initial_toto():
    initial_toto = _uniform_evaluation(1.0)
    parent = _uniform_evaluation(1.2)
    child = _uniform_evaluation(1.1)
    failures = package_full_gate_failures(
        child,
        parent,
        PackageGateConfig(),
        stage="dev",
        initial=initial_toto,
    )
    assert "initial_toto_mean_smae" in failures


def test_package_rank_key_prefers_joint_error_then_srmse_then_smae():
    evaluation = _evaluation((_score("task_1", smae=0.8, srmse=1.0),))

    assert package_rank_key(evaluation) == (0.9, 1.0, 0.8)
