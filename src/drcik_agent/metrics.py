from __future__ import annotations

import math
import statistics

from .agents import tokenize
from .models import Evidence, Forecast, ForecastTask, RetrievedDocument


def _mean_absolute_error(truth: tuple[float, ...], prediction: tuple[float, ...]) -> float:
    return statistics.fmean(abs(actual - predicted) for actual, predicted in zip(truth, prediction))


def _root_mean_squared_error(truth: tuple[float, ...], prediction: tuple[float, ...]) -> float:
    return math.sqrt(statistics.fmean((actual - predicted) ** 2 for actual, predicted in zip(truth, prediction)))


def _crps_at_point(truth: float, samples: list[float]) -> float:
    """Empirical CRPS in O(n log n) time."""
    ordered = sorted(samples)
    count = len(ordered)
    accuracy = statistics.fmean(abs(value - truth) for value in ordered)
    pairwise_half = sum((2 * index - count + 1) * value for index, value in enumerate(ordered)) / (count * count)
    return accuracy - pairwise_half


def crps_ensemble(truth: tuple[float, ...], samples: tuple[tuple[float, ...], ...]) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    horizon = len(truth)
    if any(len(sample) != horizon for sample in samples):
        raise ValueError("every sample must match the truth horizon")
    return statistics.fmean(
        _crps_at_point(truth[step], [sample[step] for sample in samples])
        for step in range(horizon)
    )


def development_scale(task: ForecastTask) -> float:
    """A transparent MASE-like scale for local development metrics.

    The hidden-test leaderboard uses the maintainers' private official scorer.
    """
    values = task.history_values
    period = task.seasonal_period
    if period and 0 < period < len(values):
        errors = [abs(values[index] - values[index - period]) for index in range(period, len(values))]
    else:
        errors = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
    nonzero_mean = statistics.fmean(errors) if errors else 0.0
    fallback = max(statistics.pstdev(values), max(values) - min(values), abs(values[-1]) * 0.1, 1.0)
    return nonzero_mean if nonzero_mean > 1e-8 else fallback


def forecast_metrics(task: ForecastTask, forecast: Forecast) -> dict[str, float]:
    if task.future_values is None:
        return {}
    truth = task.future_values
    scale = development_scale(task)
    mae = _mean_absolute_error(truth, forecast.mean)
    rmse = _root_mean_squared_error(truth, forecast.mean)
    crps = crps_ensemble(truth, forecast.samples)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "crps": crps,
        "development_scale": scale,
        "smae_proxy": min(5.0, mae / scale),
        "srmse_proxy": min(5.0, rmse / scale),
        "scrps_proxy": min(5.0, crps / scale),
    }
    if forecast.baseline_mean:
        baseline_mae = _mean_absolute_error(truth, forecast.baseline_mean)
        baseline_rmse = _root_mean_squared_error(truth, forecast.baseline_mean)
        metrics.update(
            {
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "revision_value_mae": baseline_mae - mae,
                "relative_revision_gain": (
                    (baseline_mae - mae) / baseline_mae if baseline_mae > 1e-12 else 0.0
                ),
                "harmful_revision": float(mae > baseline_mae),
            }
        )
    return metrics


def retrieval_metrics(
    task: ForecastTask,
    retrieved: list[RetrievedDocument],
    evidence: list[Evidence],
) -> dict[str, float]:
    retrieved_ids = {item.document.document_id for item in retrieved}
    supporting_ids = {document.document_id for document in task.documents if document.role == "supporting"}
    distractor_ids = {document.document_id for document in task.documents if document.role == "distractor"}
    support_hits = len(retrieved_ids & supporting_ids)
    distractor_hits = len(retrieved_ids & distractor_ids)
    metrics = {
        "supporting_document_recall": support_hits / len(supporting_ids) if supporting_ids else 0.0,
        "retrieval_precision": support_hits / len(retrieved_ids) if retrieved_ids else 0.0,
        "distractor_avoidance": 1 - distractor_hits / len(retrieved_ids) if retrieved_ids else 1.0,
    }

    if task.gt_evidence:
        predicted_tokens = set(tokenize(" ".join(item.claim for item in evidence)))
        gold_tokens = set(tokenize(" ".join(task.gt_evidence)))
        metrics["evidence_token_recall_proxy"] = (
            len(predicted_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0.0
        )
    return metrics
