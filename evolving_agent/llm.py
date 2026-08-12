"""LLM access: a small client protocol, a local Qwen implementation, and a fake for tests."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

# The default model will be qwen, specifically the Qwen2.5 family as to not have leakage and this is what most papers use.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_CACHE_DIR = "/raid/home/air/khoutaibi/models"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class LLMResponse:
    """The full text a model produced for one call."""

    text: str


class LLMClient(Protocol):
    """Anything that can turn a system prompt + messages into a response."""

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse: ...


class JsonExtractionError(ValueError):
    """Raised when a model's response contains no parseable JSON object."""


def parse_json_object(text: str) -> dict:
    """Strip a <think> block and any ```json fence, then parse the remaining text as JSON."""
    stripped = _THINK_RE.sub("", text).strip()
    fence_match = _FENCE_RE.search(stripped)
    candidate = fence_match.group(1) if fence_match else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JsonExtractionError(f"no valid JSON object in response: {exc}") from exc


class FakeLLMClient:
    """Scriptable client for tests: returns canned responses in call order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        if not self._responses:
            raise RuntimeError("FakeLLMClient has no more scripted responses")
        return LLMResponse(text=self._responses.pop(0))


def pick_free_gpu() -> str:
    """Return the cuda device with the most free memory, or 'cpu' if nvidia-smi is unavailable."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "cpu"

    best_index, best_free = None, -1
    for line in output.strip().splitlines():
        index_s, used_s, total_s = (part.strip() for part in line.split(","))
        free = int(total_s) - int(used_s)
        if free > best_free:
            best_index, best_free = int(index_s), free
    return f"cuda:{best_index}" if best_index is not None else "cpu"


class QwenClient:
    """Local Qwen2.5-14B-Instruct via transformers, lazily loaded on first use."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.device = device or pick_free_gpu()
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
        ).to(self.device)
        self._model.eval()

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        self._ensure_loaded()
        import torch

        chat = [{"role": "system", "content": system}, *messages]
        prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return LLMResponse(text=text)
