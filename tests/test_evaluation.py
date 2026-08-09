"""Metric formulas against hand-computed toy vectors."""

from __future__ import annotations

import json

from dr_cik.evaluation import (
    distractor_avoidance,
    evidence_recall,
    scale_normalizer,
    scrps,
    smae,
    srmse,
    subsample,
    supp_doc_recall,
    winsorize,
)
from dr_cik.llm import FakeLLMClient
from dr_cik.models import Document, EvidenceItem, ForecastTask


def _task(documents: tuple[Document, ...]) -> ForecastTask:
    return ForecastTask(
        benchmark_id="t",
        entity_name="e",
        target_name="tgt",
        target_description="d",
        frequency="D",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=("a", "b"),
        history_values=(1.0, 2.0),
        future_timestamps=("c", "d"),
        future_values=(10.0, 20.0),
        documents=documents,
        gt_evidence=(),
        labels_public=True,
    )


def test_scale_normalizer_is_inverse_mean_absolute() -> None:
    assert scale_normalizer((10.0, 20.0)) == 1.0 / 15.0


def test_scale_normalizer_falls_back_on_all_zero_horizon() -> None:
    assert scale_normalizer((0.0, 0.0)) == 1.0


def test_perfect_forecast_has_zero_error() -> None:
    future = (10.0, 20.0)
    samples = ((10.0, 20.0), (10.0, 20.0), (10.0, 20.0))
    assert smae(future, samples) == 0.0
    assert srmse(future, samples) == 0.0
    assert abs(scrps(future, samples)) < 1e-9


def test_large_error_is_winsorized_at_five() -> None:
    assert smae((1.0, 1.0), ((1000.0, 1000.0),)) == 5.0
    assert winsorize(9.3) == 5.0
    assert winsorize(2.1) == 2.1


def test_supp_doc_recall_counts_only_supporting_docs() -> None:
    documents = (Document("d1", "x", role="supporting"), Document("d2", "x", role="supporting"), Document("d3", "x", role="distractor"))
    task = _task(documents)
    assert supp_doc_recall(task, {"d1"}) == 0.5
    assert supp_doc_recall(task, {"d1", "d2"}) == 1.0


def test_supp_doc_recall_is_none_when_no_supporting_docs_exist() -> None:
    task = _task((Document("d1", "x", role="distractor"),))
    assert supp_doc_recall(task, {"d1"}) is None


def test_distractor_avoidance() -> None:
    documents = (Document("d1", "x", role="supporting"), Document("d3", "x", role="distractor"))
    task = _task(documents)
    assert distractor_avoidance(task, {"d1", "d3"}) == 0.5
    assert distractor_avoidance(task, {"d1"}) == 1.0


def test_distractor_avoidance_is_none_when_nothing_was_cited() -> None:
    task = _task((Document("d1", "x", role="supporting"), Document("d3", "x", role="distractor")))
    assert distractor_avoidance(task, set()) is None


def test_evidence_recall_uses_judge_matches() -> None:
    gt = ({"id": "E1", "evidence": "fact one"}, {"id": "E2", "evidence": "fact two"})
    evidence = (EvidenceItem(claim="fact one restated", source_doc_ids=("d1",)),)
    judge = FakeLLMClient(responses=[json.dumps({"matches": [{"gt_id": "E1", "matched": True}, {"gt_id": "E2", "matched": False}]})])
    result = evidence_recall(gt, evidence, judge)
    assert result.recall == 0.5
    assert result.matched_gt_ids == ("E1",)


def test_evidence_recall_is_none_with_no_ground_truth() -> None:
    judge = FakeLLMClient(responses=[])
    result = evidence_recall((), (), judge)
    assert result.recall is None


def test_evidence_recall_ignores_duplicate_and_unknown_gt_ids() -> None:
    gt = ({"id": "E1", "evidence": "fact one"},)
    matches = [{"gt_id": "E1", "matched": True}, {"gt_id": "E1", "matched": True}, {"gt_id": "E9", "matched": True}]
    judge = FakeLLMClient(responses=[json.dumps({"matches": matches})])
    result = evidence_recall(gt, (), judge)
    assert result.recall == 1.0  # not 3.0
    assert result.matched_gt_ids == ("E1",)


def test_evidence_recall_survives_malformed_match_entries() -> None:
    gt = ({"id": "E1", "evidence": "fact one"},)
    judge = FakeLLMClient(responses=[json.dumps({"matches": ["E1", 7, None]})])
    assert evidence_recall(gt, (), judge).recall == 0.0


def test_subsample_spans_the_full_matrix() -> None:
    matrix = tuple((float(i),) for i in range(100))
    assert subsample(matrix, 25)[0] == (0.0,)
    assert subsample(matrix, 25)[-1] == (99.0,)  # a head slice would end at 24.0
    assert len(subsample(matrix, 25)) == 25


def test_subsample_returns_everything_when_it_cannot_shrink() -> None:
    matrix = tuple((float(i),) for i in range(4))
    assert subsample(matrix, 10) == matrix
    assert subsample(matrix, 0) == matrix
    assert subsample(matrix, 1) == ((2.0,),)
