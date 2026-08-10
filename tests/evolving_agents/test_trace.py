"""Trace levels, reasoning sidecars, and the sandbox's tool_call/tool_result pairing."""

from __future__ import annotations

import logging

import pytest

from evolving_agents.harness.sandbox import run_forecast_code
from evolving_agents.harness.trace import (
    TraceEvent,
    collected_events,
    configure_tracing,
    current_level,
    emit,
    emit_llm_call,
    emit_llm_response,
)

LLM_RECORD = {
    "model_id": "Qwen/Qwen2.5-14B-Instruct",
    "prompt_hash": "abc123",
    "temperature": 0.8,
    "draw_index": 2,
    "enable_thinking": True,
    "user_text": "forecast this series",
    "answer": '{"assumption": "19-step cycle"}',
    "reasoning": "The last 40 points dip every 19 steps, so seasonality dominates." * 5,
}


@pytest.fixture(autouse=True)
def _reset_tracing():
    yield
    configure_tracing("off")


def test_off_level_records_nothing() -> None:
    configure_tracing("off", collect=True)
    emit(TraceEvent(task_id="task_1", agent="coding", event_type="llm_call", detail=LLM_RECORD))
    assert collected_events() == ()
    assert current_level() == "off"


def _emit_pair() -> None:
    """Emit the request half from the cache hook and the response half from the parsing agent."""
    emit_llm_call("task_1", "coding.generate", LLM_RECORD)
    emit_llm_response(
        "task_1", "coding.generate", LLM_RECORD["answer"], LLM_RECORD["reasoning"],
        model_id=LLM_RECORD["model_id"], prompt_hash=LLM_RECORD["prompt_hash"],
    )


def test_a_call_and_its_response_are_emitted_exactly_once_each(caplog) -> None:
    configure_tracing("summary", collect=True)
    with caplog.at_level(logging.INFO):
        _emit_pair()

    events = collected_events()
    assert [event.event_type for event in events] == ["llm_call", "llm_response"]
    assert all(event.timestamp for event in events)
    assert caplog.text.count("LLM_RESPONSE") == 1  # the cache hook must not duplicate the agent's response
    assert LLM_RECORD["reasoning"] not in caplog.text


def test_summary_truncates_a_long_answer(caplog) -> None:
    configure_tracing("summary")
    with caplog.at_level(logging.INFO):
        emit_llm_response("task_1", "coding.generate", "x" * 500)
    assert "..." in caplog.text
    assert "x" * 500 not in caplog.text


def test_full_level_prints_reasoning_inline(caplog) -> None:
    configure_tracing("full", collect=True)
    with caplog.at_level(logging.INFO):
        _emit_pair()
    assert "reasoning: The last 40 points dip every 19 steps" in caplog.text


def test_summary_writes_a_reasoning_sidecar(tmp_path, caplog) -> None:
    configure_tracing("summary", runs_dir=tmp_path, collect=True)
    with caplog.at_level(logging.INFO):
        _emit_pair()

    sidecars = list((tmp_path / "reasoning").rglob("*.txt"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text(encoding="utf-8") == LLM_RECORD["reasoning"]
    assert sidecars[0].name == "abc123.txt"
    assert "chars ->" in caplog.text  # the log points at the sidecar instead of inlining it


def test_response_without_reasoning_writes_no_sidecar(tmp_path) -> None:
    configure_tracing("summary", runs_dir=tmp_path, collect=True)
    emit_llm_response("task_1", "retrieval", "{}", None, prompt_hash="d1")
    assert not (tmp_path / "reasoning").exists()


def test_sandbox_emits_a_tool_call_and_result_pair() -> None:
    configure_tracing("summary", collect=True)
    run_forecast_code("def forecast(h, z, f):\n    return [1.0] * z\n", (1.0,), 2, "H", task_id="task_9")

    events = [event for event in collected_events() if event.agent == "coding.sandbox"]
    assert [event.event_type for event in events] == ["tool_call", "tool_result"]
    assert events[0].detail["tool"] == "sandbox.execute"
    assert events[1].detail["ok"] is True
    assert events[1].detail["forecast_len"] == 2


def test_sandbox_failure_still_emits_a_result() -> None:
    configure_tracing("summary", collect=True)
    run_forecast_code("import os\ndef forecast(h, z, f):\n    return [1.0] * z\n", (1.0,), 2, "H", task_id="task_9")

    result = [event for event in collected_events() if event.event_type == "tool_result"][0]
    assert result.detail["ok"] is False
    assert "not allowed" in result.detail["error"]


def test_generation_appears_in_the_rendered_line(caplog) -> None:
    configure_tracing("summary")
    with caplog.at_level(logging.INFO):
        emit(TraceEvent(task_id="task_3", agent="coding", event_type="agent_start", detail={}, generation=3))
    assert "gen03/coding" in caplog.text


def test_unknown_trace_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        configure_tracing("verbose")
