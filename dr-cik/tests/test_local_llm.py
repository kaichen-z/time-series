"""QwenClient against a fake transformers module, and its device auto-pick logic."""

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

    def apply_chat_template(self, chat, add_generation_prompt, tokenize):
        return "PROMPT:" + "|".join(f"{m['role']}={m['content']}" for m in chat)

    def __call__(self, prompt, return_tensors):
        import torch

        return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, token_ids, skip_special_tokens):
        return "generated reply"


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

        return torch.tensor([[1, 2, 3, 4, 5]])


def _fake_transformers() -> SimpleNamespace:
    return SimpleNamespace(AutoTokenizer=_FakeTokenizer, AutoModelForCausalLM=_FakeModel)


def test_qwen_client_completes_with_fake_runtime() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    response = client.complete(system="You are terse.", messages=[{"role": "user", "content": "hi"}])
    assert response.text == "generated reply"


def test_qwen_client_loads_model_only_once() -> None:
    client = QwenClient(QwenConfig(device="cpu"), runtime_module=_fake_transformers())
    client.complete(system="s", messages=[{"role": "user", "content": "a"}])
    first_model = client._model
    client.complete(system="s", messages=[{"role": "user", "content": "b"}])
    assert client._model is first_model


def test_missing_transformers_raises_local_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)  # forces ImportError regardless of what's actually installed
    client = QwenClient(QwenConfig())
    with pytest.raises(LocalModelUnavailableError):
        client._load_runtime()
