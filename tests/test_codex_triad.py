from __future__ import annotations

import json
from types import SimpleNamespace

from drcik_agent.models import Document, ForecastTask
from drcik_agent.triad import ThreeAgentForecastSystem, TriadConfig


class FakeTriadCodexClient:
    def __init__(self) -> None:
        self.calls = 0
        self.config = SimpleNamespace(max_document_characters=12000)

    def stats(self):
        return {
            "calls": self.calls,
            "cache_hits": 0,
            "failures": 0,
            "latency_seconds": 0.01,
            "last_error": None,
        }

    def complete(self, stage, _prompt, _schema, workspace_files=None):
        self.calls += 1
        if stage.startswith("triad_coding_plan"):
            return {
                "candidate_families": ["backbone", "statistical"],
                "assumptions": [
                    {"family": "backbone", "assumption": "The seasonal pattern persists."},
                    {"family": "statistical", "assumption": "Trend and seasonality persist."},
                ],
                "information_needs": ["Look for a quantified event in the horizon."],
            }
        if stage.startswith("triad_retrieval"):
            return {
                "query": "Alpha Station energy demand promotion",
                "rationale": "A future promotion distinguishes the candidates.",
                "selected_document_ids": ["doc_event"],
                "evidence": [
                    {
                        "document_id": "doc_event",
                        "claim": "A promotion raises demand by 50 percent.",
                        "exact_quote": "The promotion will increase energy demand by 50 percent throughout the event.",
                        "confidence": 0.99,
                    }
                ],
                "impacts": [
                    {
                        "source_document_ids": ["doc_event"],
                        "event_type": "promotion",
                        "start_timestamp": "2024-01-03 00:00:00",
                        "end_timestamp": "2024-01-03 01:00:00",
                        "direction": "up",
                        "permanence": "temporary",
                        "forecast_relation": "overlaps_forecast",
                        "adjustment_kind": "percentage",
                        "adjustment_value": 0.5,
                        "confidence": 0.99,
                        "rationale": "The source explicitly quantifies the target effect.",
                    }
                ],
                "sufficient": True,
            }
        if stage.startswith("triad_coding_revision"):
            payload = json.loads(workspace_files["candidate_workspace.json"])
            adjusted = next(
                item["candidate_id"]
                for item in payload["candidates"]
                if "evidence_adjusted" in item["tags"]
            )
            return {
                "selected_candidate_ids": ["c_backbone", adjusted],
                "rationale": "Keep the baseline and one grounded intervention hypothesis.",
            }
        if stage.startswith("triad_decision"):
            payload = json.loads(workspace_files["candidates.json"])
            adjusted = next(
                (
                    item["candidate_id"]
                    for item in payload["candidates"]
                    if "evidence_adjusted" in item["tags"]
                ),
                "c_backbone",
            )
            return {
                "selected_candidate_id": adjusted,
                "request_more_retrieval": False,
                "request_new_candidates": False,
                "rationale": "The quantified event supports the executable adjustment.",
                "supporting_document_ids": ["doc_event"],
            }
        raise AssertionError(stage)


def _task() -> ForecastTask:
    return ForecastTask(
        benchmark_id="task_codex_triad",
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
        future_values=(18.0, 33.0),
        documents=(
            Document(
                "doc_event",
                "Alpha Station will run a promotion. "
                "The promotion will increase energy demand by 50 percent throughout the event.",
                role="supporting",
            ),
        ),
        gt_evidence=("A promotion raises demand by 50 percent.",),
    )


def test_all_three_triad_roles_use_codex_but_host_executes_forecast() -> None:
    client = FakeTriadCodexClient()
    system = ThreeAgentForecastSystem(
        TriadConfig(
            backbone="statistical",
            max_rounds=1,
            documents_per_round=2,
            num_samples=20,
            reasoning_agent="codex",
        ),
        codex_client=client,
    )
    result = system.run(_task())

    assert client.calls == 4  # coding plan, retrieval, coding revision, decision
    assert result.loop_trace[0]["agent_backend"] == "codex"
    assert result.loop_trace[0]["accepted_document_ids"] == ["doc_event"]
    assert "evidence_adjusted" in next(
        item["tags"]
        for item in result.loop_trace[0]["coding_candidates"]
        if item["candidate_id"]
        == result.loop_trace[0]["decision"]["selected_candidate_ids"][0]
    )
    assert result.forecast.mean == (17.625, 32.625)


def test_ungrounded_codex_quote_is_rejected() -> None:
    client = FakeTriadCodexClient()
    original_complete = client.complete

    def complete(stage, prompt, schema, workspace_files=None):
        result = original_complete(stage, prompt, schema, workspace_files)
        if stage.startswith("triad_retrieval"):
            result["evidence"][0]["exact_quote"] = "This sentence is fabricated."
        return result

    client.complete = complete
    system = ThreeAgentForecastSystem(
        TriadConfig(
            backbone="statistical",
            max_rounds=1,
            documents_per_round=2,
            num_samples=20,
            reasoning_agent="codex",
        ),
        codex_client=client,
    )
    result = system.run(_task())

    assert result.retrieved == []
    assert result.evidence == []
    assert result.forecast.mean == result.forecast.baseline_mean
def test_codex_decision_rejects_adjusted_candidate_without_matching_citation() -> None:
    client = FakeTriadCodexClient()
    original_complete = client.complete

    def complete(stage, prompt, schema, workspace_files=None):
        result = original_complete(stage, prompt, schema, workspace_files)
        if stage.startswith("triad_decision"):
            result["supporting_document_ids"] = []
        return result

    client.complete = complete
    system = ThreeAgentForecastSystem(
        TriadConfig(
            backbone="statistical",
            max_rounds=1,
            documents_per_round=2,
            num_samples=20,
            reasoning_agent="codex",
        ),
        codex_client=client,
    )
    result = system.run(_task())

    # The host's deterministic decision remains authoritative when Codex does
    # not satisfy the stricter provenance contract.
    assert result.loop_trace[-1]["decision"]["rationale"].startswith(
        "Select the highest historically validated compatible hypothesis"
    )
