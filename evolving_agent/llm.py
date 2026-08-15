"""LLM access for tests, local Qwen, and the installed Codex CLI."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Qwen remains available as an explicit local ablation; the harness CLI defaults to Codex.
DEFAULT_MODEL_ID = "Qwen/Qwen3-32B"
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


class TransientLLMError(RuntimeError):
    """Raised when temporary infrastructure prevents an LLM call from completing."""


def parse_json_object(text: str) -> dict:
    """Extract the first valid JSON object from common model response wrappers."""
    stripped = _THINK_RE.sub("", text).strip()
    candidates = [match.group(1).strip() for match in _FENCE_RE.finditer(stripped)]
    candidates.append(stripped)
    last_error: json.JSONDecodeError | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
        # Models occasionally wrap an otherwise valid object in one sentence.
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
    detail = str(last_error) if last_error is not None else "response contained no JSON object"
    raise JsonExtractionError(f"no valid JSON object in response: {detail}")


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


@dataclass(frozen=True)
class CodexCLIConfig:
    """Configuration for one isolated, non-interactive Codex call."""

    binary: str = "codex"
    model: str | None = None
    reasoning_effort: str = "high"
    timeout_seconds: int = 900
    cache_dir: str | Path | None = "runs/evolving/codex-cache"
    json_repair_attempts: int = 2
    transport_retries: int = 2
    transport_retry_delay_seconds: float = 1.0


class CodexCLIClient:
    """Use this machine's authenticated Codex CLI as an ``LLMClient``.

    Each completion runs in a temporary read-only workspace. The model receives no
    repository files or agent rules; it only receives the supplied role prompt and
    messages. Successful JSON responses can be cached for deterministic reruns.
    """

    def __init__(self, config: CodexCLIConfig | None = None) -> None:
        self.config = config or CodexCLIConfig()
        self.calls = 0
        self.cache_hits = 0

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        del temperature  # Codex CLI does not expose sampling temperature.
        prompt = self._prompt(system, messages)
        cache_path = self._cache_path(prompt)
        if cache_path is not None and cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            try:
                parse_json_object(cached)
            except JsonExtractionError:
                # A partial write or an older buggy run must never poison retries.
                pass
            else:
                self.cache_hits += 1
                return LLMResponse(text=cached)

        text = self._execute(prompt)
        try:
            parsed = parse_json_object(text)
        except JsonExtractionError as initial_error:
            parsed = self._repair_json(text, initial_error)
        normalized = json.dumps(parsed, ensure_ascii=False)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(normalized, encoding="utf-8")
        return LLMResponse(text=normalized)

    def _execute(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="evolving-agent-codex-") as directory:
            workdir = Path(directory)
            result_path = workdir / "response.json"
            command = [
                self.config.binary,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--output-last-message",
                str(result_path),
                "--cd",
                str(workdir),
                "-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"',
            ]
            if self.config.model:
                command.extend(["--model", self.config.model])
            command.append("-")
            retries = max(int(self.config.transport_retries), 0)
            for attempt in range(retries + 1):
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        timeout=self.config.timeout_seconds,
                        check=False,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(f"Codex CLI was not found: {self.config.binary}") from exc
                except subprocess.TimeoutExpired as exc:
                    self.calls += 1
                    if attempt < retries:
                        self._transport_backoff(attempt)
                        continue
                    raise TransientLLMError(
                        f"Codex CLI timed out after {self.config.timeout_seconds} seconds"
                    ) from exc
                self.calls += 1
                if completed.returncode == 0:
                    break
                details = (completed.stderr or completed.stdout).strip()[-2000:]
                if self._is_transient_transport_error(details):
                    if attempt < retries:
                        self._transport_backoff(attempt)
                        continue
                    raise TransientLLMError(
                        f"Codex CLI transport failed after {retries + 1} attempts: {details}"
                    )
                raise RuntimeError(
                    f"Codex CLI failed with exit code {completed.returncode}: {details}"
                )
            if not result_path.exists():
                raise RuntimeError("Codex CLI completed without writing its final response")
            text = result_path.read_text(encoding="utf-8").strip()
        return text

    def _transport_backoff(self, attempt: int) -> None:
        delay = max(float(self.config.transport_retry_delay_seconds), 0.0) * (2**attempt)
        if delay:
            time.sleep(delay)

    @staticmethod
    def _is_transient_transport_error(details: str) -> bool:
        lowered = details.lower()
        return any(
            marker in lowered
            for marker in (
                "connection reset",
                "connection refused",
                "failed to connect",
                "error sending request",
                "stream disconnected",
                "temporary failure in name resolution",
                "network is unreachable",
                "service unavailable",
                "selected model is at capacity",
                "rate limit",
                "status code 429",
                "status code 502",
                "status code 503",
                "status code 504",
            )
        )

    def _repair_json(self, broken: str, initial_error: Exception) -> dict:
        error: Exception = initial_error
        candidate = broken
        for attempt in range(1, max(self.config.json_repair_attempts, 0) + 1):
            repair_prompt = (
                "Repair only the JSON syntax in the text below. Preserve every key and semantic "
                "value. Do not add analysis, markdown, or new claims. Return exactly one valid "
                f"JSON object. Repair attempt {attempt}.\n\nBROKEN JSON:\n{candidate}"
            )
            candidate = self._execute(repair_prompt)
            try:
                return parse_json_object(candidate)
            except JsonExtractionError as exc:
                error = exc
        raise JsonExtractionError(
            f"JSON remained invalid after {self.config.json_repair_attempts} repair attempts: {error}"
        ) from error

    def _cache_path(self, prompt: str) -> Path | None:
        if self.config.cache_dir is None:
            return None
        identity = json.dumps(
            {
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "prompt": prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return Path(self.config.cache_dir) / f"{digest}.json"

    @staticmethod
    def _prompt(system: str, messages: list[dict]) -> str:
        conversation = "\n\n".join(
            f"[{str(message.get('role', 'user')).upper()}]\n{message.get('content', '')}"
            for message in messages
        )
        return (
            "Follow the role instructions below. Do not use tools or inspect files. "
            "Return only the exact JSON object requested by the role instructions.\n\n"
            f"[SYSTEM / ROLE INSTRUCTIONS]\n{system}\n\n{conversation}"
        )

# Claude Code, use the subscriiption and not API as to not to pay
@dataclass(frozen=True)
class ClaudeCLIConfig:
    """Configuration for one isolated, non-interactive Claude Code call."""

    binary: str = "claude"
    model: str | None = None
    timeout_seconds: int = 900
    cache_dir: str | Path | None = "runs/evolving/claude-cache"


class ClaudeCLIClient:
    """Use this machine's authenticated Claude Code CLI (subscription login) as an ``LLMClient``.

    Never sets ``--permission-mode plan``: that mode forces Claude Code's own built-in
    planning persona on top of any custom ``--system-prompt``, which was verified to make
    the model treat the supplied role instructions as untrusted injected text rather than
    real system configuration. ``--allowedTools ""`` alone is the safety boundary instead --
    zero tools registered means zero possible side effects, regardless of permission mode.
    """

    def __init__(self, config: ClaudeCLIConfig | None = None) -> None:
        self.config = config or ClaudeCLIConfig()
        self.calls = 0
        self.cache_hits = 0

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        del temperature  # Claude Code CLI print mode does not expose sampling temperature.
        prompt = self._prompt(messages)
        cache_key = self._cache_key(system, prompt)
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.exists():
            self.cache_hits += 1
            return LLMResponse(text=cache_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory(prefix="evolving-agent-claude-") as directory:
            workdir = Path(directory)
            command = [
                self.config.binary,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                system,
                "--allowedTools",
                "",
                "--setting-sources",
                "",
            ]
            if self.config.model:
                command.extend(["--model", self.config.model])
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                    cwd=workdir,
                    env=self._subprocess_env(),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(f"Claude Code CLI was not found: {self.config.binary}") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Claude Code CLI timed out after {self.config.timeout_seconds} seconds"
                ) from exc
            self.calls += 1
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()[-2000:]
                raise RuntimeError(
                    f"Claude Code CLI failed with exit code {completed.returncode}: {details}"
                )
            try:
                envelope = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Claude Code CLI produced non-JSON output: {exc}") from exc
            if envelope.get("is_error"):
                raise RuntimeError(f"Claude Code CLI reported an error result: {envelope}")
            text = str(envelope.get("result", "")).strip()

        # Every evolving-agent prompt has a JSON contract. Fail immediately instead
        # of silently feeding prose or an error message to a downstream agent.
        parse_json_object(text)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return LLMResponse(text=text)

    def _cache_key(self, system: str, prompt: str) -> str:
        identity = json.dumps(
            {"model": self.config.model, "system": system, "prompt": prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _cache_path(self, digest: str) -> Path | None:
        if self.config.cache_dir is None:
            return None
        return Path(self.config.cache_dir) / f"{digest}.json"

    @staticmethod
    def _prompt(messages: list[dict]) -> str:
        return "\n\n".join(
            f"[{str(message.get('role', 'user')).upper()}]\n{message.get('content', '')}"
            for message in messages
        )

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        # Strip every CLAUDE*-prefixed variable: when this process is itself run inside a
        # Claude Code session, those leak into the child and make it behave as a "child
        # session" (inheriting the parent's mode/state) instead of a clean, isolated call --
        # verified directly: the same call produced a suspicious-injection refusal with them
        # present, and a clean JSON answer with them stripped.
        return {key: value for key, value in os.environ.items() if not key.startswith("CLAUDE")}


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
        max_new_tokens: int = 8192,
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
