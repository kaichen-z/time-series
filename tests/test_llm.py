"""FakeLLMClient and JSON-extraction behavior."""

from __future__ import annotations

import pytest

from dr_cik.llm import FakeLLMClient, JsonExtractionError, parse_json_object


def test_fake_client_returns_scripted_responses_in_order() -> None:
    client = FakeLLMClient(responses=["one", "two"])
    assert client.complete(system="s", messages=[{"role": "user", "content": "a"}]).text == "one"
    assert client.complete(system="s", messages=[{"role": "user", "content": "b"}]).text == "two"
    assert len(client.calls) == 2


def test_fake_client_raises_when_exhausted() -> None:
    client = FakeLLMClient(responses=["only"])
    client.complete(system="s", messages=[{"role": "user", "content": "a"}])
    with pytest.raises(AssertionError):
        client.complete(system="s", messages=[{"role": "user", "content": "b"}])


def test_fake_client_supports_callable_responses() -> None:
    client = FakeLLMClient(responses=lambda system, messages: f"echo:{messages[0]['content']}")
    response = client.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert response.text == "echo:hi"


def test_parse_json_object_strips_code_fence() -> None:
    parsed = parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_parse_json_object_raises_on_non_json() -> None:
    with pytest.raises(JsonExtractionError):
        parse_json_object("not json at all")


def test_parse_json_object_raises_on_non_dict() -> None:
    with pytest.raises(JsonExtractionError):
        parse_json_object("[1, 2, 3]")


def test_parse_json_object_strips_closed_think_block() -> None:
    parsed = parse_json_object('<think>let me work through this...</think>{"a": 1}')
    assert parsed == {"a": 1}


def test_parse_json_object_raises_on_unclosed_think_block() -> None:
    """An unclosed <think> (truncated by the token budget) must fail, not silently parse garbage."""
    with pytest.raises(JsonExtractionError):
        parse_json_object('<think>reasoning that never finished because tokens ran out')


def test_parse_json_object_recovers_json_after_untagged_reasoning_prose() -> None:
    """Not every reasoning model uses <think> tags; Qwen3.5 was observed writing plain 'Thinking Process:' prose."""
    text = 'Thinking Process:\n1. Analyze the request.\n2. Compute values.\n\n{"forecast": [1.0, 2.0]}'
    parsed = parse_json_object(text)
    assert parsed == {"forecast": [1.0, 2.0]}


def test_parse_json_object_raises_when_reasoning_never_reaches_json() -> None:
    """If the budget ran out mid-reasoning and no JSON object ever appears, this must fail, not guess."""
    with pytest.raises(JsonExtractionError):
        parse_json_object("Thinking Process:\n1. Analyze the request.\n2. Still reasoning when tokens ran out")
