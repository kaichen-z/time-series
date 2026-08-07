"""Dr-CiK's official metrics: sMAE/sRMSE/sCRPS, SuppDocRecall, DistractorAvoidance, EvidenceRecall."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .llm import JsonExtractionError, LLMClient, parse_json_object
from .models import EvidenceItem, Forecast, ForecastTask

WINSORIZE_CAP = 5.0


def winsorize(value: float, cap: float = WINSORIZE_CAP) -> float:
    """Clip a per-task score to cap, per SUBMISSION.md's winsorization rule."""
    return min(value, cap)


def scale_normalizer(future_values: tuple[float, ...]) -> float:
    """a = (1/T * sum|y_t|)^-1 over the ground-truth horizon, robust to an all-zero horizon."""
    mean_abs = statistics.fmean(abs(value) for value in future_values)
    if mean_abs <= 1e-8:
        return 1.0
    return 1.0 / mean_abs


def _sample_mean(sample_matrix: tuple[tuple[float, ...], ...], horizon: int) -> tuple[float, ...]:
    return tuple(statistics.fmean(sample[step] for sample in sample_matrix) for step in range(horizon))


def smae(future_values: tuple[float, ...], sample_matrix: tuple[tuple[float, ...], ...]) -> float:
    """Scaled mean absolute error of the sample-mean forecast against ground truth."""
    scale = scale_normalizer(future_values)
    mean_forecast = _sample_mean(sample_matrix, len(future_values))
    return winsorize(scale * statistics.fmean(abs(m - y) for m, y in zip(mean_forecast, future_values)))


def srmse(future_values: tuple[float, ...], sample_matrix: tuple[tuple[float, ...], ...]) -> float:
    """Scaled root mean squared error of the sample-mean forecast against ground truth."""
    scale = scale_normalizer(future_values)
    mean_forecast = _sample_mean(sample_matrix, len(future_values))
    mse = statistics.fmean((m - y) ** 2 for m, y in zip(mean_forecast, future_values))
    return winsorize(scale * math.sqrt(mse))


def _crps_at_step(truth: float, samples_at_step: list[float]) -> float:
    """CRPS_t in O(S log S), via the closed-form pairwise-distance identity."""
    ordered = sorted(samples_at_step)
    count = len(ordered)
    accuracy = statistics.fmean(abs(value - truth) for value in ordered)
    pairwise_half = sum((2 * index - count + 1) * value for index, value in enumerate(ordered)) / (count * count)
    return accuracy - pairwise_half


def scrps(future_values: tuple[float, ...], sample_matrix: tuple[tuple[float, ...], ...]) -> float:
    """Scaled CRPS averaged over the horizon."""
    if not sample_matrix:
        raise ValueError("sample_matrix must not be empty")
    scale = scale_normalizer(future_values)
    horizon = len(future_values)
    per_step = [_crps_at_step(future_values[step], [sample[step] for sample in sample_matrix]) for step in range(horizon)]
    return winsorize(scale * statistics.fmean(per_step))


def cited_document_ids(evidence: tuple[EvidenceItem, ...]) -> set[str]:
    """Every document_id cited by any evidence item."""
    ids: set[str] = set()
    for item in evidence:
        ids.update(item.source_doc_ids)
    return ids


def supp_doc_recall(task: ForecastTask, used_doc_ids: set[str]) -> float | None:
    """|used cited docs ∩ supporting docs| / |supporting docs|, or None if there are none."""
    supporting = {document.document_id for document in task.documents if document.role == "supporting"}
    if not supporting:
        return None
    return len(used_doc_ids & supporting) / len(supporting)


def distractor_avoidance(task: ForecastTask, used_doc_ids: set[str]) -> float:
    """1 - |used cited docs ∩ distractors| / |used cited docs| (1.0 if nothing was cited)."""
    if not used_doc_ids:
        return 1.0
    distractors = {document.document_id for document in task.documents if document.role == "distractor"}
    return 1.0 - len(used_doc_ids & distractors) / len(used_doc_ids)


EVIDENCE_JUDGE_INSTRUCTIONS = (
    "You are grading whether a forecasting agent's synthesized evidence covers each piece of "
    "required ground-truth evidence. For each ground-truth item, decide whether ANY predicted "
    'claim conveys the same fact. Respond with exactly one JSON object: {"matches": '
    '[{"gt_id": "...", "matched": true|false}]}'
)


@dataclass(frozen=True)
class EvidenceRecallResult:
    """The outcome of the LLM-judge evidence-recall proxy, with the raw judge text kept for audit."""

    recall: float | None
    matched_gt_ids: tuple[str, ...]
    judge_raw_text: str


def evidence_recall(
    gt_evidence: tuple[dict[str, str], ...],
    evidence: tuple[EvidenceItem, ...],
    judge: LLMClient,
    judge_model_id: str = "gemini-3-flash-preview",
) -> EvidenceRecallResult:
    """Approximate Dr-CiK's evidence-recall metric with our own LLM-judge prompt (a proxy, not the official scorer)."""
    if not gt_evidence:
        return EvidenceRecallResult(recall=None, matched_gt_ids=(), judge_raw_text="")
    gt_block = "\n".join(f"{item['id']}: {item['evidence']}" for item in gt_evidence)
    predicted_block = "\n".join(f"- {item.claim}" for item in evidence) or "(no evidence submitted)"
    prompt = f"Ground-truth evidence:\n{gt_block}\n\nPredicted claims:\n{predicted_block}"
    response = judge.complete(system=EVIDENCE_JUDGE_INSTRUCTIONS, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    try:
        parsed = parse_json_object(response.text)
        matched_ids = tuple(str(entry["gt_id"]) for entry in parsed.get("matches", []) if entry.get("matched"))
    except (JsonExtractionError, KeyError, TypeError):
        matched_ids = ()
    return EvidenceRecallResult(recall=len(matched_ids) / len(gt_evidence), matched_gt_ids=matched_ids, judge_raw_text=response.text)


def development_metrics(
    task: ForecastTask,
    forecast: Forecast,
    evidence: tuple[EvidenceItem, ...],
    used_doc_ids: set[str],
    judge: LLMClient | None,
    crps_sample_size: int = 25,
) -> dict[str, float | None]:
    """Compose all metrics for one task; skips ground-truth-dependent ones when labels are hidden."""
    metrics: dict[str, float | None] = {}
    cited = cited_document_ids(evidence)
    metrics["supp_doc_recall_cited"] = supp_doc_recall(task, cited)
    metrics["distractor_avoidance_cited"] = distractor_avoidance(task, cited)
    metrics["supp_doc_recall_retrieved"] = supp_doc_recall(task, used_doc_ids)
    metrics["distractor_avoidance_retrieved"] = distractor_avoidance(task, used_doc_ids)

    if task.future_values is not None:
        crps_samples = forecast.samples[:crps_sample_size] or forecast.samples
        metrics["smae"] = smae(task.future_values, forecast.samples)
        metrics["srmse"] = srmse(task.future_values, forecast.samples)
        metrics["scrps"] = scrps(task.future_values, crps_samples)
    else:
        metrics["smae"] = metrics["srmse"] = metrics["scrps"] = None

    if judge is not None and task.labels_public and task.gt_evidence:
        metrics["evidence_recall"] = evidence_recall(task.gt_evidence, evidence, judge).recall
    else:
        metrics["evidence_recall"] = None
    return metrics
