"""Retrieval F1 and the composite Loop-C score, including their degenerate cases."""

from __future__ import annotations

import pytest
from dr_cik.models import Document, EvidenceItem, ForecastTask

from evolving_agents.harness.metrics import loop_c_score, retrieval_f1, retrieval_scores


def _task(*roles: str) -> ForecastTask:
    return ForecastTask(
        benchmark_id="t",
        entity_name="e",
        target_name="t",
        target_description="d",
        frequency="H",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=("a", "b"),
        history_values=(1.0, 2.0),
        future_timestamps=("c", "d"),
        future_values=(3.0, 4.0),
        documents=tuple(Document(f"doc_{index}", "x", role=role, subtype=None) for index, role in enumerate(roles)),
        gt_evidence=(),
        labels_public=True,
    )


def _kept(*doc_ids: str) -> tuple[EvidenceItem, ...]:
    return tuple(EvidenceItem(claim=f"claim about {doc_id}", source_doc_ids=(doc_id,)) for doc_id in doc_ids)


def test_perfect_retrieval_scores_one() -> None:
    task = _task("supporting", "distractor")
    assert retrieval_f1(task, _kept("doc_0")) == pytest.approx(1.0)


def test_citing_a_distractor_lowers_precision() -> None:
    task = _task("supporting", "distractor")
    scores = retrieval_scores(task, _kept("doc_0", "doc_1"))
    assert scores.recall == pytest.approx(1.0)
    assert scores.precision == pytest.approx(0.5)
    assert scores.f1 == pytest.approx(2 / 3)


def test_missing_a_supporting_document_lowers_recall() -> None:
    task = _task("supporting", "supporting")
    scores = retrieval_scores(task, _kept("doc_0"))
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(0.5)


def test_keeping_nothing_is_unscoreable_not_zero() -> None:
    # Mirrors dr_cik.evaluation's discipline: a degenerate case is None so it is skipped in a mean,
    # never a score that flatters or punishes an agent that simply did not answer.
    assert retrieval_f1(_task("supporting", "distractor"), ()) is None


def test_a_task_with_no_supporting_documents_is_unscoreable() -> None:
    scores = retrieval_scores(_task("distractor", "distractor"), _kept("doc_0"))
    assert scores.f1 is None
    assert scores.supporting_total == 0


def test_citing_only_distractors_scores_zero_not_none() -> None:
    assert retrieval_f1(_task("supporting", "distractor"), _kept("doc_1")) == 0.0


def test_loop_c_score_is_higher_is_better() -> None:
    good = loop_c_score({"smae": 0.2, "srmse": 0.3, "scrps": 0.1, "evidence_recall": 0.9})
    bad = loop_c_score({"smae": 2.0, "srmse": 2.5, "scrps": 1.8, "evidence_recall": 0.1})
    assert good > bad


def test_loop_c_score_rewards_evidence_recall() -> None:
    base = {"smae": 0.5, "srmse": 0.5, "scrps": 0.5}
    assert loop_c_score({**base, "evidence_recall": 1.0}) > loop_c_score({**base, "evidence_recall": 0.0})


def test_loop_c_score_treats_missing_metrics_as_absent() -> None:
    assert loop_c_score({"smae": None, "srmse": None, "scrps": None, "evidence_recall": None}) == 0.0
