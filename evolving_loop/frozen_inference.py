"""Label-free inference and Dr-CiK submission export for frozen harnesses."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Sequence

from common.metrics import aggregate_drcik_point_metrics
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, Document
from evolving_loop.evaluation import score_after_resolution
from evolving_loop.harness import EvolvingForecastHarness

HarnessFactory = Callable[[HarnessPolicy], EvolvingForecastHarness]


def _public_score_payload(outcome: object | None) -> dict[str, object] | None:
    """Expose point-score fields without evaluator GT/document-label diagnostics."""
    if outcome is None:
        return None
    fields = (
        "task_id",
        "final_smae",
        "final_srmse",
        "coding_oracle_smae",
        "coding_oracle_srmse",
        "contextual_oracle_smae",
        "contextual_oracle_srmse",
        "decision_selection_smae_regret",
        "decision_selection_srmse_regret",
        "candidate_count",
    )
    return {field: getattr(outcome, field, None) for field in fields}


def inference_view(task: ContextTask) -> ContextTask:
    """Strip every evaluator-only field before any mutable harness code runs."""
    return replace(
        task,
        numeric=replace(task.numeric, future_values=()),
        documents=tuple(
            Document(document.document_id, document.content)
            for document in task.documents
        ),
        gt_evidence=(),
        labels_public=False,
    )


def run_frozen_inference(
    policy: HarnessPolicy,
    tasks: Sequence[ContextTask],
    harness_factory: HarnessFactory,
    *,
    output_dir: str | Path,
    samples: int = 100,
    score_public: bool = False,
    artifact_kind: str = "genome",
) -> dict:
    """Run an immutable policy without outcome learning and export submissions."""
    if not tasks:
        raise ValueError("frozen inference needs at least one task")
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if score_public and any(not task.labels_public for task in tasks):
        raise ValueError("--score-public is forbidden when any selected task is hidden")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    forecast_path = destination / "forecasts.jsonl"
    research_path = destination / "deep_research.jsonl"
    report_path = destination / "run_report.jsonl"
    seen_ids: set[str] = set()
    outcomes = []

    with (
        forecast_path.open("w", encoding="utf-8") as forecasts,
        research_path.open("w", encoding="utf-8") as research,
        report_path.open("w", encoding="utf-8") as reports,
    ):
        for task in tasks:
            task_id = task.numeric.task_id
            if task_id in seen_ids:
                raise ValueError(f"duplicate benchmark_id in inference input: {task_id}")
            seen_ids.add(task_id)

            # One isolated harness per task prevents accidental online skill writes
            # from changing later hidden predictions.
            result = harness_factory(policy).run(
                inference_view(task), allow_skill_writes=False
            )
            values = tuple(float(value) for value in result.forecast)
            if len(values) != task.numeric.prediction_length:
                raise ValueError(
                    f"{task_id}: forecast length {len(values)} != "
                    f"prediction_length {task.numeric.prediction_length}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{task_id}: forecast contains non-finite values")

            # The current harness is point-valued. Repeated trajectories are a valid,
            # deliberately degenerate empirical distribution and make that limitation explicit.
            forecasts.write(
                json.dumps(
                    {
                        "benchmark_id": task_id,
                        "samples": [list(values) for _ in range(samples)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            research.write(
                json.dumps(
                    {
                        "benchmark_id": task_id,
                        "cited_document_ids": list(result.retrieval.selected_document_ids),
                        "evidence": [item.claim for item in result.retrieval.evidence],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            outcome = score_after_resolution(task, result) if score_public else None
            if outcome is not None:
                outcomes.append(outcome)
            card = getattr(result, "retrieval_card", None)
            if card is not None:
                retrieval_payload = card.to_payload()
                assumption_stances = [
                    {
                        "chain_id": chain.chain_id,
                        "assumption_ids": list(chain.addressed_assumption_ids),
                        "stance": chain.stance,
                    }
                    for chain in card.chains
                ]
                used_skill_ids = list(
                    dict.fromkeys(
                        skill_id
                        for chain in card.chains
                        for skill_id in chain.used_skill_ids
                    )
                )
            else:
                # Keep the report schema stable for legacy single-pass artifacts.
                retrieval_payload = {
                    "round1": None,
                    "round2": None,
                    "chains": [],
                    "selected_document_ids": list(
                        result.retrieval.selected_document_ids
                    ),
                    "rejected": list(result.retrieval.rejected),
                    "unresolved_contradictions": [],
                    "complete": bool(result.retrieval.sufficient),
                    "gaps": [],
                }
                assumption_stances = []
                used_skill_ids = []
            reports.write(
                json.dumps(
                    {
                        "benchmark_id": task_id,
                        "artifact_kind": artifact_kind,
                        "policy_version": policy.version,
                        "release_sha256": policy.retrieval_release_sha256,
                        "labels_accessed": score_public,
                        "selected_candidate_id": result.decision.selected.candidate_id,
                        "host_default_id": result.decision.host_default_id,
                        "forecast": list(values),
                        "retrieved_document_ids": list(result.retrieval.selected_document_ids),
                        "evidence": [asdict(item) for item in result.retrieval.evidence],
                        "impacts": [asdict(item) for item in result.retrieval.impacts],
                        "retrieval_rejections": list(result.retrieval.rejected),
                        "retrieval": retrieval_payload,
                        "assumption_stances": assumption_stances,
                        "used_skill_ids": used_skill_ids,
                        "decision": {
                            "rationale": result.decision.rationale,
                            "supporting_document_ids": list(
                                result.decision.supporting_document_ids
                            ),
                            "rejection_reason": result.decision.rejection_reason,
                            "llm_override_accepted": result.decision.llm_override_accepted,
                        },
                        "coding_candidates": [
                            {
                                "candidate_id": item.candidate_id,
                                "assumption": item.assumption,
                                "failure_condition": item.failure_condition,
                                "hindcast_smae": item.hindcast_smae,
                                "hindcast_srmse": item.hindcast_srmse,
                            }
                            for item in result.candidates
                        ],
                        "outcome": _public_score_payload(outcome),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    aggregate = (
        aggregate_drcik_point_metrics(
            [
                {"smae": item.final_smae, "srmse": item.final_srmse}
                for item in outcomes
            ]
        )
        if outcomes
        else None
    )
    summary = {
        "artifact_kind": artifact_kind,
        "policy_version": policy.version,
        "release_sha256": policy.retrieval_release_sha256,
        "num_tasks": len(tasks),
        "labels_accessed": score_public,
        "samples_per_task": samples,
        "probabilistic_note": (
            "Point forecasts are exported as repeated trajectories; uncertainty is not yet modeled."
        ),
        "forecasts_path": str(forecast_path),
        "deep_research_path": str(research_path),
        "run_report_path": str(report_path),
        "mean_smae": aggregate["smae"] if aggregate else None,
        "mean_srmse": aggregate["srmse"] if aggregate else None,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
