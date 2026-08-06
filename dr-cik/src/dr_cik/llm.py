"""LLM client abstraction: a real Gemini client and a scriptable fake for tests."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """The text returned by an LLM call, plus the raw provider object if useful."""

    text: str
    raw: Any = None


class LLMClient(Protocol):
    """The only interface agent code is allowed to call an LLM through."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


class JsonExtractionError(ValueError):
    """Raised when a model response doesn't contain a parseable JSON object."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Strip a ```json fence if present and parse the remaining text as a JSON object."""
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JsonExtractionError(text) from exc
    if not isinstance(parsed, dict):
        raise JsonExtractionError(text)
    return parsed


@dataclass
class FakeLLMClient:
    """A scriptable LLMClient for offline tests: a fixed response list or a callable."""

    responses: list[str] | Callable[[str, list[dict[str, str]]], str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        if callable(self.responses):
            text = self.responses(system, messages)
        else:
            index = len(self.calls) - 1
            if index >= len(self.responses):
                raise AssertionError(f"FakeLLMClient exhausted after {len(self.responses)} scripted responses")
            text = self.responses[index]
        return LLMResponse(text=text)


class GeminiUnavailableError(RuntimeError):
    """Raised when the google-genai package or an API key is missing."""


class GeminiClient:
    """A thin wrapper over google-genai, lazily imported so it's optional at install time."""

    def __init__(self, model_id: str = "gemini-3-flash-preview", api_key: str | None = None, sdk_module: Any | None = None) -> None:
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._sdk_module = sdk_module
        self._client: Any | None = None

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise GeminiUnavailableError("GEMINI_API_KEY is not set")
        module = self._sdk_module
        if module is None:
            try:
                from google import genai as module  # type: ignore[no-redef]
            except ImportError as exc:
                raise GeminiUnavailableError("google-genai is not installed; pip install 'dr-cik[gemini]'") from exc
        self._client = module.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._load_client()
        contents = [message["content"] for message in messages]
        config = {"system_instruction": system, "temperature": temperature}
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        response = client.models.generate_content(model=self.model_id, contents=contents, config=config)
        return LLMResponse(text=response.text, raw=response)
