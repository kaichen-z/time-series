from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta

from drcik_agent.agents import TimeSeriesDiagnosisAgent
from drcik_agent.models import Document, ForecastTask
from drcik_agent.triad import (
    CodingForecastAgent,
    DecisionForecastAgent,
    ThreeAgentForecastSystem,
    TriadConfig,
    TriadEvolutionPolicy,
)


def _task(with_event: bool = True) -> ForecastTask:
    documents = (
        Document(
            document_id="doc_future_event",
            text=(
                "Alpha Station will run a temporary promotion from "
                "2024-01-03 00:00:00 to 2024-01-03 01:00:00. "
                "The promotion will increase energy demand by 50 percent throughout the event."
            ),
            role="supporting",
        ),
    ) if with_event else ()
    return ForecastTask(
        benchmark_id="task_triad",
        entity_name="Alpha Station",
        target_name="energy demand",
        target_description="Hourly energy demand at Alpha Station",
        frequency="1 hour",
        prediction_length=2,
        seasonal_period=2,
        history_timestamps=(
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ),
        history_values=(10.0, 20.0, 11.0, 21.0),
        future_timestamps=("2024-01-03 00:00:00", "2024-01-03 01:00:00"),
        future_values=(18.0, 33.0) if with_event else (12.0, 22.0),
        documents=documents,
        gt_evidence=("A future promotion increases energy demand by 50 percent.",) if with_event else (),
    )


def _system(**overrides) -> ThreeAgentForecastSystem:
    config = TriadConfig(
        backbone="statistical",
        max_rounds=overrides.pop("max_rounds", 2),
        documents_per_round=overrides.pop("documents_per_round", 2),
        num_samples=20,
        **overrides,
    )
    return ThreeAgentForecastSystem(config)


def test_triad_converts_future_event_into_candidate_and_selects_it() -> None:
    result = _system().run(_task())

    candidates = result.loop_trace[-1]["coding_candidates"]
    selected = set(result.loop_trace[-1]["decision"]["selected_candidate_ids"])
    adjusted = [item for item in candidates if "evidence_adjusted" in item["tags"]]

    assert adjusted
    assert selected & {item["candidate_id"] for item in adjusted}
    assert result.forecast.mean == (17.625, 32.625)


def test_triad_preserves_backbone_without_documents() -> None:
    task = replace(_task(with_event=False), labels_public=False, future_values=None)
    result = _system(max_rounds=1).run(task)

    assert result.forecast.mean == result.forecast.baseline_mean
    assert result.loop_trace[-1]["decision"]["selected_candidate_ids"] == ("c_backbone",)


def test_delayed_feedback_attributes_three_modules() -> None:
    system = _system()
    task = _task()
    feedback = system.record_outcome(task, system.run(task))

    assert {item.agent_name for item in feedback} == {
        "coding_agent",
        "retrieval_agent",
        "decision_agent",
    }


def test_delayed_feedback_updates_and_reloads_policy(tmp_path) -> None:
    policy_path = tmp_path / "triad-policy.json"
    system = _system(evolution_path=str(policy_path))
    task = _task()
    system.record_outcome(task, system.run(task))

    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["tasks_seen"] == 1
    assert payload["coding_tag_bias"]
    assert payload["retrieval_term_weights"]
    assert payload["decision_tag_bias"]
    assert _system(evolution_path=str(policy_path)).policy.tasks_seen == 1


def test_decision_uses_rolling_backtest_instead_of_model_family_prior() -> None:
    class BadBackbone:
        def forecast(self, task, diagnosis):
            del diagnosis
            return tuple(0.0 for _ in task.future_timestamps), "bad_backbone"

    start = datetime(2024, 1, 1)
    timestamps = tuple(
        (start + timedelta(hours=index)).isoformat(sep=" ") for index in range(52)
    )
    history = tuple(10.0 if index % 2 == 0 else 20.0 for index in range(48))
    task = ForecastTask(
        benchmark_id="rolling_validation",
        entity_name="Alpha Station",
        target_name="energy demand",
        target_description="Synthetic two-step seasonal demand",
        frequency="1 hour",
        prediction_length=4,
        seasonal_period=2,
        history_timestamps=timestamps[:48],
        history_values=history,
        future_timestamps=timestamps[48:],
        future_values=None,
        documents=(),
        labels_public=False,
    )
    policy = TriadEvolutionPolicy()
    coding = CodingForecastAgent(
        BadBackbone(),
        policy,
        validation_folds=3,
        validation_horizon=4,
        minimum_validation_history=12,
    )
    diagnosis = TimeSeriesDiagnosisAgent().diagnose(task)
    candidates, _method = coding.initial_candidates(task, diagnosis)
    decision = DecisionForecastAgent(0.12, policy).decide(
        candidates, [], 1, 1, False
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    assert by_id["c_statistical"].validation_mae == 0.0
    assert (
        by_id["c_statistical"].historical_score
        > by_id["c_backbone"].historical_score
    )
    assert decision.selected_candidate_ids == ("c_statistical",)
