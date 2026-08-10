"""Cache key derivation, hit/miss accounting, and the on_call hook."""

from __future__ import annotations

from dr_cik.llm import FakeLLMClient, LLMResponse

from evolving_agents.llm_cache import CachingLLMClient

MESSAGES = [{"role": "user", "content": "forecast this"}]


def _client(tmp_path, responses, **kwargs) -> CachingLLMClient:
    return CachingLLMClient(FakeLLMClient(responses=responses), model_id="test-model", cache_dir=tmp_path, **kwargs)


def test_identical_call_hits_cache_and_skips_inner(tmp_path) -> None:
    inner = FakeLLMClient(responses=["first"])
    client = CachingLLMClient(inner, model_id="test-model", cache_dir=tmp_path)

    assert client.complete(system="s", messages=MESSAGES).text == "first"
    assert client.complete(system="s", messages=MESSAGES).text == "first"
    assert len(inner.calls) == 1  # the second call never reached the wrapped client
    assert (client.hits, client.misses) == (1, 1)


def test_different_draw_index_is_a_different_entry(tmp_path) -> None:
    client = _client(tmp_path, ["draw0", "draw1"])
    assert client.complete(system="s", messages=MESSAGES, temperature=0.8, draw_index=0).text == "draw0"
    assert client.complete(system="s", messages=MESSAGES, temperature=0.8, draw_index=1).text == "draw1"
    assert (client.hits, client.misses) == (0, 2)


def test_enable_thinking_is_part_of_the_key(tmp_path) -> None:
    plain = _client(tmp_path, ["no-think"], enable_thinking=False)
    thinking = _client(tmp_path, ["with-think"], enable_thinking=True)
    assert plain.complete(system="s", messages=MESSAGES).text == "no-think"
    assert thinking.complete(system="s", messages=MESSAGES).text == "with-think"
    assert (thinking.hits, thinking.misses) == (0, 1)


def test_temperature_and_prompt_changes_are_different_entries(tmp_path) -> None:
    client = _client(tmp_path, ["a", "b", "c"])
    client.complete(system="s", messages=MESSAGES, temperature=0.0)
    client.complete(system="s", messages=MESSAGES, temperature=0.8)
    client.complete(system="different", messages=MESSAGES, temperature=0.0)
    assert (client.hits, client.misses) == (0, 3)


def test_cache_survives_a_new_client_instance(tmp_path) -> None:
    first = _client(tmp_path, ["persisted"])
    first.complete(system="s", messages=MESSAGES)

    second_inner = FakeLLMClient(responses=[])
    second = CachingLLMClient(second_inner, model_id="test-model", cache_dir=tmp_path)
    assert second.complete(system="s", messages=MESSAGES).text == "persisted"
    assert second_inner.calls == []


def test_corrupt_cache_entry_falls_back_to_the_model(tmp_path) -> None:
    client = _client(tmp_path, ["fresh", "regenerated"])
    client.complete(system="s", messages=MESSAGES)
    for path in tmp_path.rglob("*.json"):
        path.write_text("{ not json", encoding="utf-8")
    assert client.complete(system="s", messages=MESSAGES).text == "regenerated"


def test_on_call_fires_for_both_hit_and_miss(tmp_path) -> None:
    seen: list[dict] = []
    client = _client(tmp_path, ["only"], on_call=seen.append)
    client.complete(system="s", messages=MESSAGES)
    client.complete(system="s", messages=MESSAGES)

    assert [record["cache_hit"] for record in seen] == [False, True]
    assert seen[0]["response_text"] == "only"
    assert seen[0]["user_text"] == "forecast this"
    assert seen[0]["model_id"] == "test-model"


class _BatchingClient:
    """A minimal complete_many-capable client, standing in for QwenClient."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def complete(self, *, system, messages, temperature=0.0, max_output_tokens=None) -> LLMResponse:
        return LLMResponse(text="single")

    def complete_many(self, *, system, messages, count, temperature=1.0, max_output_tokens=None) -> list[LLMResponse]:
        self.batch_sizes.append(count)
        return [LLMResponse(text=f"draw{index}") for index in range(count)]


def test_complete_many_only_regenerates_missing_draws(tmp_path) -> None:
    inner = _BatchingClient()
    client = CachingLLMClient(inner, model_id="test-model", cache_dir=tmp_path)

    first = client.complete_many(system="s", messages=MESSAGES, count=3, temperature=0.8)
    assert [item.text for item in first] == ["draw0", "draw1", "draw2"]
    assert inner.batch_sizes == [3]

    second = client.complete_many(system="s", messages=MESSAGES, count=3, temperature=0.8)
    assert [item.text for item in second] == ["draw0", "draw1", "draw2"]
    assert inner.batch_sizes == [3]  # fully cached, no second batch
    assert client.hits == 3


def test_complete_many_partial_cache_only_fills_the_gap(tmp_path) -> None:
    inner = _BatchingClient()
    client = CachingLLMClient(inner, model_id="test-model", cache_dir=tmp_path)
    client.complete_many(system="s", messages=MESSAGES, count=2, temperature=0.8)
    client.complete_many(system="s", messages=MESSAGES, count=5, temperature=0.8)
    assert inner.batch_sizes == [2, 3]


def test_complete_many_without_batching_support_loops(tmp_path) -> None:
    client = _client(tmp_path, ["a", "b"])
    assert [item.text for item in client.complete_many(system="s", messages=MESSAGES, count=2, temperature=0.8)] == ["a", "b"]
