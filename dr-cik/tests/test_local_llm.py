"""QwenClient against a fake transformers module: complete(), complete_many() batched sampling, and device auto-pick."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dr_cik.local_llm import LocalModelUnavailableError, QwenClient, QwenConfig


class _FakeBatchEncoding(dict):
    """Mimics HF's BatchEncoding: dict-like (for **inputs) but also has .to()."""

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def apply_chat_template(self, chat, add_generation_prompt, tokenize, enable_thinking=False):
        return "PROMPT:" + "|".join(f"{m['role']}={m['content']}" for m in chat)

    def __call__(self, prompt, return_tensors):
        import torch

        return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, token_ids, skip_special_tokens):
        return f"generated reply {token_ids[-1].item()}"


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch

        count = kwargs.get("num_return_sequences", 1)
        # prompt is always the fixed [1, 2, 3] from _FakeTokenizer.__call__; each row's generated
        # tail is distinct (100+i, 200+i) so complete_many's responses are distinguishable per row.
        rows = [[1, 2, 3, 100 + i, 200 + i] for i in range(count)]
        return torch.tensor(rows)


def _fake_transformers() -> SimpleNamespace:
    return SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModelForCausalLM=_FakeModel)


def test_qwen_client_completes_with_fake_runtime() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    response = client.complete(system="You are terse.", messages=[{"role": "user", "content": "hi"}])
    assert response.text == "generated reply 200"


def test_qwen_client_loads_model_only_once() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    client.complete(system="s", messages=[{"role": "user", "content": "a"}])
    first_model = client._model
    client.complete(system="s", messages=[{"role": "user", "content": "b"}])
    assert client._model is first_model


def test_qwen_client_complete_many_returns_one_response_per_count() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    responses = client.complete_many(system="s", messages=[{"role": "user", "content": "hi"}], count=4, temperature=1.0)
    assert len(responses) == 4
    assert len({response.text for response in responses}) == 4  # each row decoded independently -> distinct text


def test_qwen_client_complete_many_samples_with_num_return_sequences() -> None:
    captured: dict = {}

    class _RecordingModel(_FakeModel):
        def generate(self, **kwargs):
            captured.update(kwargs)
            return super().generate(**kwargs)

    runtime = SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModelForCausalLM=_RecordingModel)
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=runtime)

    client.complete_many(system="s", messages=[{"role": "user", "content": "hi"}], count=7, temperature=0.8)

    assert captured["num_return_sequences"] == 7
    assert captured["do_sample"] is True
    assert captured["temperature"] == 0.8


def test_qwen_client_complete_many_chunks_above_max_batch_size() -> None:
    """count=20 with max_batch_size=8 must issue 3 generate() calls (8+8+4), not one 20-way batch."""
    batch_sizes: list[int] = []

    class _RecordingModel(_FakeModel):
        def generate(self, **kwargs):
            batch_sizes.append(kwargs["num_return_sequences"])
            return super().generate(**kwargs)

    runtime = SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModelForCausalLM=_RecordingModel)
    client = QwenClient(QwenConfig(device="cpu", max_batch_size=8), runtime_module=runtime)

    responses = client.complete_many(system="s", messages=[{"role": "user", "content": "hi"}], count=20)

    assert batch_sizes == [8, 8, 4]
    assert len(responses) == 20


def test_qwen_client_complete_many_loads_model_only_once() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    client.complete_many(system="s", messages=[{"role": "user", "content": "a"}], count=3)
    first_model = client._model
    client.complete_many(system="s", messages=[{"role": "user", "content": "b"}], count=3)
    assert client._model is first_model


def test_qwen_client_seeds_torch_once_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """--seed must reach torch, or sampled runs are silently irreproducible."""
    import torch

    seeded: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda value: seeded.append(value))
    client = QwenClient(QwenConfig(device="cpu", seed=1234), runtime_module=_fake_transformers())

    client.complete(system="s", messages=[{"role": "user", "content": "a"}])
    client.complete(system="s", messages=[{"role": "user", "content": "b"}])

    assert seeded == [1234]  # once at load, never per call (per-call would collapse all draws to one)


def test_qwen_client_without_seed_does_not_touch_torch_global_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    seeded: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda value: seeded.append(value))
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())

    client.complete(system="s", messages=[{"role": "user", "content": "a"}])

    assert seeded == []


def test_missing_transformers_raises_local_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)  # forces ImportError regardless of what's actually installed
    client = QwenClient(QwenConfig())
    with pytest.raises(LocalModelUnavailableError):
        client._load_runtime()
