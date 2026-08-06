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
