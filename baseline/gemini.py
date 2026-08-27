"""Gemini access and the Direct Prompt forecaster the paper's LLM rows use."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from common.llm import JsonExtractionError, LLMResponse, TransientLLMError, parse_json_object

DEFAULT_MODEL = "gemini-3.1-flash-lite"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

FORECAST_SYSTEM = """You are a time series forecaster. You are given the recent history of a \
series as (timestamp, value) lines and must continue it.

Return only a JSON object: {"forecast": [v1, v2, ...]} with exactly the requested number of \
numeric values, in chronological order. No prose, no code, no explanation."""


@dataclass(frozen=True)
class GeminiConfig:
    model: str = DEFAULT_MODEL
    timeout_seconds: int = 120
    transport_retries: int = 5
    retry_delay_seconds: float = 2.0
    max_output_tokens: int = 32768
    # Free-tier gemini-3.1-flash-lite allows 15 requests/minute. Exceeding it returns 429 and,
    # without pacing, most of a sweep's calls are simply thrown away.
    requests_per_minute: int = 15


class RateLimiter:
    """Blocks callers so no more than `per_minute` calls start in any rolling 60s window."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._started: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._started = [t for t in self._started if now - t < 60.0]
                if len(self._started) < self.per_minute:
                    self._started.append(now)
                    return
                wait = 60.0 - (now - self._started[0]) + 0.05
            time.sleep(max(wait, 0.05))


class GeminiClient:
    """Gemini over google.genai, conforming to the repo's LLMClient protocol.

    The key is read from the environment (or .env) and never logged, echoed, or written into
    any artifact this package produces.
    """

    def __init__(self, config: GeminiConfig | None = None, api_key: str | None = None) -> None:
        self.config = config or GeminiConfig()
        self._api_key = api_key or _api_key()
        self._client: Any | None = None
        self._limiter = RateLimiter(self.config.requests_per_minute)
        self._lock = threading.Lock()
        self.calls = 0
        self.rate_limited = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError(
                    "google-genai is not installed; install it with: pip install google-genai"
                ) from error
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self, *, system: str, messages: list[dict], temperature: float = 0.0
    ) -> LLMResponse:
        client = self._ensure_client()
        from google.genai import types

        prompt = "\n\n".join(str(message.get("content", "")) for message in messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        last: Exception | None = None
        for attempt in range(self.config.transport_retries):
            self._limiter.acquire()
            try:
                response = client.models.generate_content(
                    model=self.config.model, contents=prompt, config=config
                )
                with self._lock:
                    self.calls += 1
                return LLMResponse(text=response.text or "")
            except Exception as error:  # the SDK raises many transport-shaped errors
                last = error
                if attempt + 1 >= self.config.transport_retries:
                    break
                # A 429 tells us how long to wait; obeying it beats guessing a backoff.
                delay = _retry_delay(error)
                if delay is not None:
                    with self._lock:
                        self.rate_limited += 1
                time.sleep(delay if delay is not None
                           else self.config.retry_delay_seconds * (attempt + 1))
        raise TransientLLMError(f"Gemini call failed after {self.config.transport_retries} tries: {last}")


def _retry_delay(error: Exception) -> float | None:
    """Seconds the API asked us to wait, or None when this was not a rate-limit refusal."""
    text = str(error)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return None
    match = re.search(r"[Rr]etry in ([0-9.]+)s", text) or re.search(r"retryDelay'?: '?(\d+)s", text)
    return min(float(match.group(1)) + 0.5, 90.0) if match else 15.0


def _api_key() -> str:
    """Read GEMINI_API_KEY from the environment, falling back to the repo .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key and ENV_FILE.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(ENV_FILE)
        except ImportError as error:
            raise RuntimeError("python-dotenv is not installed and GEMINI_API_KEY is unset") from error
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(f"GEMINI_API_KEY is not set in the environment or {ENV_FILE}")
    return key


def build_prompt(
    history: Sequence[float], timestamps: Sequence[str], horizon: int, frequency: str
) -> str:
    """Serialize one task under the No Context condition: the series, its cadence, and the horizon."""
    lines = [
        f"Sampling frequency: {frequency}",
        f"Forecast the next {horizon} values.",
        "",
        "History as (timestamp, value):",
    ]
    stamps = list(timestamps) or [str(index) for index in range(len(history))]
    for stamp, value in zip(stamps[-len(history):], history):
        lines.append(f"({stamp}, {value:.6g})")
    lines.append("")
    lines.append(f'Return {{"forecast": [...]}} with exactly {horizon} numbers.')
    return "\n".join(lines)


class DirectPromptForecaster:
    """Draws trajectories by sampling the LLM repeatedly at a non-zero temperature.

    One call is one path, which is how the paper builds its S trajectories for an LLM
    forecaster. Responses are cached on disk so an interrupted sweep does not re-buy them.
    """

    name = "gemini_dp"

    def __init__(
        self,
        client: GeminiClient | None = None,
        cache_dir: str | Path | None = "runs/baselines/gemini-cache",
        temperature: float = 1.0,
        max_history: int = 512,
        workers: int = 8,
    ) -> None:
        self.client = client or GeminiClient()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.temperature = temperature
        self.max_history = max_history
        # Trajectories are independent calls, so they are drawn concurrently; serially a full
        # sweep is hours of pure network wait.
        self.workers = workers
        self.timestamps: tuple[str, ...] = ()
        self.frequency = ""

    def forecast_samples(
        self, history: Sequence[float], horizon: int, samples: int
    ) -> tuple[tuple[float, ...], ...]:
        context = [float(value) for value in history[-self.max_history :]]
        prompt = build_prompt(context, self.timestamps[-len(context):], horizon, self.frequency)
        with ThreadPoolExecutor(max_workers=min(self.workers, samples)) as pool:
            drawn = pool.map(lambda index: self._safe_path(prompt, horizon, index), range(samples))
        paths = [path for path in drawn if path is not None]
        if not paths:
            raise ValueError(f"no usable forecast in {samples} attempts")
        return tuple(paths)

    def _safe_path(self, prompt: str, horizon: int, index: int) -> tuple[float, ...] | None:
        """A draw that exhausts its retries costs one trajectory, not the whole task."""
        try:
            return self._one_path(prompt, horizon, index)
        except TransientLLMError:
            return None

    def _one_path(self, prompt: str, horizon: int, index: int) -> tuple[float, ...] | None:
        """One sampled trajectory, or None when the model's answer is unusable.

        A malformed path is dropped rather than raised: the ensemble tolerates losing a few
        draws, and one bad response should not cost the whole task.
        """
        cached = self._read_cache(prompt, index)
        if cached is not None:
            return _parse_forecast(cached, horizon)
        response = self.client.complete(
            system=FORECAST_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        self._write_cache(prompt, index, response.text)
        return _parse_forecast(response.text, horizon)

    def _key(self, prompt: str, index: int) -> str:
        digest = hashlib.sha256(f"{self.client.config.model}\n{self.temperature}\n{prompt}".encode())
        return f"{digest.hexdigest()}-{index}"

    def _read_cache(self, prompt: str, index: int) -> str | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{self._key(prompt, index)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except (ValueError, KeyError):
            return None

    def _write_cache(self, prompt: str, index: int, text: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._key(prompt, index)}.json"
        path.write_text(json.dumps({"text": text}), encoding="utf-8")


def _parse_forecast(text: str, horizon: int) -> tuple[float, ...] | None:
    """Read {"forecast": [...]} of the right length, or None if the answer is unusable."""
    try:
        payload = parse_json_object(text)
    except JsonExtractionError:
        return None
    raw = payload.get("forecast")
    if not isinstance(raw, list) or len(raw) != horizon:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        return None
    return tuple(values)
