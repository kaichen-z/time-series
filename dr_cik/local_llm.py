"""A local Qwen client (via transformers), for when the Gemini API is rate-limited/unavailable."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from .llm import LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/raid/home/air/khoutaibi/models"
# Qwen2.5 (released Sept 2024): a well-documented, non-bleeding-edge checkpoint, chosen
# specifically so its pretraining cutoff sits safely before Dr-CiK's more recent task windows.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


class LocalModelUnavailableError(RuntimeError):
    """Raised when transformers/torch, the checkpoint, or a GPU can't be loaded."""


def _pick_device() -> str:
    """Return the CUDA device with the most free memory right now, or 'cpu' if none."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    best_index, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        if free_bytes > best_free:
            best_index, best_free = index, free_bytes
    return f"cuda:{best_index}"


@dataclass(frozen=True)
class QwenConfig:
    """Local Qwen checkpoint and generation configuration."""

    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None  # None -> auto-pick the freest GPU at load time
    dtype: str = "bfloat16"
    cache_dir: str | None = DEFAULT_CACHE_DIR
    max_new_tokens: int = 1024
    enable_thinking: bool = False  # Qwen3+ reasoning models emit a <think> block by default, burning the token budget
    max_batch_size: int = 8  # cap on num_return_sequences per generate() call; complete_many() chunks above this to bound peak GPU memory on a shared cluster
    seed: int | None = None  # seeds torch once at load, making a fixed sequence of sampled calls reproducible


class QwenClient:
    """Runs a local Qwen chat model via transformers; no API key or network calls per-request."""

    def __init__(self, config: QwenConfig | None = None, runtime_module: Any | None = None) -> None:
        self.config = config or QwenConfig()
        self._runtime_module = runtime_module
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._resolved_device: str | None = None

    def _load_runtime(self) -> Any:
        if self._runtime_module is not None:
            return self._runtime_module
        try:
            import transformers
        except ImportError as exc:
            raise LocalModelUnavailableError("transformers is not installed; pip install 'dr-cik[qwen]'") from exc
        return transformers

    def _ensure_model(self) -> tuple[Any, Any, str]:
        if self._model is not None:
            return self._tokenizer, self._model, self._resolved_device
        import torch

        transformers = self._load_runtime()
        device = self.config.device or _pick_device()
        logger.info("loading %s onto %s (dtype=%s)", self.config.model_id, device, self.config.dtype)
        if self.config.seed is not None:
            # Seed once here, not per call: re-seeding before each generate() would make every
            # sampled draw identical, collapsing the S trajectories the forecaster needs.
            torch.manual_seed(self.config.seed)
        if self.config.cache_dir:
            os.makedirs(os.path.expanduser(self.config.cache_dir), exist_ok=True)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_id, cache_dir=self.config.cache_dir)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.cache_dir,
                dtype=getattr(torch, self.config.dtype),
            ).to(device)
            model.eval()
        except Exception as exc:
            raise LocalModelUnavailableError(f"Failed to load {self.config.model_id} on {device}: {exc}") from exc
        logger.info("loaded %s onto %s", self.config.model_id, device)
        self._tokenizer, self._model, self._resolved_device = tokenizer, model, device
        return tokenizer, model, device

    def _prepare(self, system: str, messages: list[dict[str, str]]):
        tokenizer, model, device = self._ensure_model()
        chat = [{"role": "system", "content": system}] + [
            {"role": "user" if m.get("role", "user") != "assistant" else "assistant", "content": m["content"]} for m in messages
        ]
        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=self.config.enable_thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        return tokenizer, model, inputs

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        import torch

        tokenizer, model, inputs = self._prepare(system, messages)
        budget = max_output_tokens or self.config.max_new_tokens
        prompt_tokens = inputs["input_ids"].shape[1]
        logger.debug("generate: prompt_tokens=%d max_new_tokens=%d temperature=%.2f", prompt_tokens, budget, temperature)
        start = time.monotonic()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info("generate: produced %d token(s) in %.1fs", len(generated), time.monotonic() - start)
        logger.debug("generate: response text=%r", text[:2000])
        return LLMResponse(text=text, raw=output_ids)

    def complete_many(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        count: int,
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """Sample `count` independent completions, batched in generate() calls of at most max_batch_size at a time.

        num_return_sequences replicates activations/KV-cache across the batch dimension, so an
        unbounded count risks CUDA OOM on a shared cluster (seen live: 25-way batch OOM'd on a 9B
        model with a 90-step horizon). Chunking bounds peak memory while keeping most of the speedup.
        """
        import torch

        tokenizer, model, inputs = self._prepare(system, messages)
        budget = max_output_tokens or self.config.max_new_tokens
        prompt_len = inputs["input_ids"].shape[1]
        logger.info(
            "generate_many: sampling %d completion(s) in batches of <=%d (prompt_tokens=%d max_new_tokens=%d temperature=%.2f)",
            count, self.config.max_batch_size, prompt_len, budget, temperature,
        )
        responses: list[LLMResponse] = []
        remaining = count
        batch_num = 0
        while remaining > 0:
            batch_num += 1
            chunk = min(remaining, self.config.max_batch_size)
            start = time.monotonic()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=True,
                    temperature=max(temperature, 1e-4),
                    num_return_sequences=chunk,
                    pad_token_id=tokenizer.eos_token_id,
                )
            logger.debug("generate_many: batch %d (%d completions) done in %.1fs", batch_num, chunk, time.monotonic() - start)
            responses.extend(LLMResponse(text=tokenizer.decode(row[prompt_len:], skip_special_tokens=True), raw=row) for row in output_ids)
            remaining -= chunk
        logger.info("generate_many: collected %d/%d completion(s)", len(responses), count)
        return responses
