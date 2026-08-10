"""A content-addressed cache wrapping any LLMClient, so reruns replay instead of re-generating."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from dr_cik.llm import LLMClient, LLMResponse

DEFAULT_CACHE_DIR = os.environ.get("EVOLVING_AGENTS_CACHE_DIR", ".cache/llm")


@dataclass(frozen=True)
class CacheKey:
    """Everything that can change a completion; two calls agreeing on all of it may share a result."""

    model_id: str
    system_hash: str
    messages_hash: str
    temperature: float
    max_output_tokens: int | None
    draw_index: int
    enable_thinking: bool


def _hash_text(text: str) -> str:
    """Return a short stable hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _key_digest(key: CacheKey) -> str:
    """Return the full hex digest identifying a cache entry on disk."""
    return hashlib.sha256(json.dumps(asdict(key), sort_keys=True).encode("utf-8")).hexdigest()


def _slug(model_id: str) -> str:
    """Turn a model id into a filesystem-safe directory name."""
    return model_id.replace("/", "__").replace(":", "_")


class CachingLLMClient:
    """Wraps an LLMClient and memoizes completions on disk, keyed by prompt and draw index.

    Composition, not subclassing: this satisfies the same LLMClient Protocol, so it is a
    drop-in anywhere a plain client is accepted. `draw_index` is part of the key because the
    Coding Agent deliberately samples K diverse hypotheses from one prompt -- a key without it
    would collapse all K draws onto a single cached entry.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        model_id: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        enable_thinking: bool | None = None,
        on_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        inner_config = getattr(inner, "config", None)
        self.inner = inner
        self.model_id = model_id or getattr(inner_config, "model_id", None) or getattr(inner, "model_id", "unknown")
        self.enable_thinking = (
            enable_thinking if enable_thinking is not None else bool(getattr(inner_config, "enable_thinking", False))
        )
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.on_call = on_call
        self.hits = 0
        self.misses = 0

    def _path_for(self, key: CacheKey) -> Path:
        """Return the sharded on-disk path for a cache key."""
        digest = _key_digest(key)
        return self.cache_dir / _slug(key.model_id) / digest[:2] / f"{digest}.json"

    def _build_key(
        self, system: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int | None, draw_index: int
    ) -> CacheKey:
        """Derive the cache key for one completion request."""
        return CacheKey(
            model_id=self.model_id,
            system_hash=_hash_text(system),
            messages_hash=_hash_text(json.dumps(messages, sort_keys=True)),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            draw_index=draw_index,
            enable_thinking=self.enable_thinking,
        )

    def _read(self, path: Path) -> str | None:
        """Return a cached response text, or None if absent or unreadable."""
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response_text"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _write(self, path: Path, key: CacheKey, text: str) -> None:
        """Atomically persist one completion so a partial write can never be read back."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": asdict(key), "response_text": text, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _record(self, key: CacheKey, *, cache_hit: bool, latency_s: float, system: str, messages: list[dict[str, str]], text: str) -> None:
        """Notify the on_call hook about one completion, hit or miss."""
        if self.on_call is None:
            return
        self.on_call(
            {
                "model_id": key.model_id,
                "prompt_hash": _key_digest(key),
                "cache_hit": cache_hit,
                "latency_s": latency_s,
                "temperature": key.temperature,
                "draw_index": key.draw_index,
                "enable_thinking": key.enable_thinking,
                "system_text": system,
                "user_text": messages[-1]["content"] if messages else "",
                "response_text": text,
            }
        )

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        draw_index: int = 0,
    ) -> LLMResponse:
        """Return a cached completion if one exists, else call the wrapped client and store it."""
        key = self._build_key(system, messages, temperature, max_output_tokens, draw_index)
        path = self._path_for(key)
        cached = self._read(path)
        if cached is not None:
            self.hits += 1
            self._record(key, cache_hit=True, latency_s=0.0, system=system, messages=messages, text=cached)
            return LLMResponse(text=cached)

        start = time.monotonic()
        response = self.inner.complete(
            system=system, messages=messages, temperature=temperature, max_output_tokens=max_output_tokens
        )
        latency = time.monotonic() - start
        self.misses += 1
        self._write(path, key, response.text)
        self._record(key, cache_hit=False, latency_s=latency, system=system, messages=messages, text=response.text)
        return response

    def complete_many(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        count: int,
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """Return `count` completions, serving cached draws and generating only the missing ones."""
        keys = [self._build_key(system, messages, temperature, max_output_tokens, index) for index in range(count)]
        results: list[LLMResponse | None] = []
        missing: list[int] = []
        for index, key in enumerate(keys):
            cached = self._read(self._path_for(key))
            if cached is None:
                results.append(None)
                missing.append(index)
            else:
                self.hits += 1
                self._record(key, cache_hit=True, latency_s=0.0, system=system, messages=messages, text=cached)
                results.append(LLMResponse(text=cached))

        if missing:
            start = time.monotonic()
            batched = getattr(self.inner, "complete_many", None)
            if batched is not None:
                fresh = batched(
                    system=system, messages=messages, count=len(missing), temperature=temperature, max_output_tokens=max_output_tokens
                )
            else:
                fresh = [
                    self.inner.complete(
                        system=system, messages=messages, temperature=temperature, max_output_tokens=max_output_tokens
                    )
                    for _ in missing
                ]
            latency = (time.monotonic() - start) / max(len(missing), 1)
            for index, response in zip(missing, fresh):
                self.misses += 1
                self._write(self._path_for(keys[index]), keys[index], response.text)
                self._record(keys[index], cache_hit=False, latency_s=latency, system=system, messages=messages, text=response.text)
                results[index] = response

        return [item for item in results if item is not None]
