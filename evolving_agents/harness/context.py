"""Threads the current task/generation into LLM-call records, which the cache itself cannot know."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace import emit_llm_call


@dataclass
class CallContext:
    """Buffers one task's LLM calls and tags them with the task/generation for tracing and logging."""

    agent: str = "worker"
    task_id: str = "-"
    generation: int | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def start(self, task_id: str, generation: int | None = None, agent: str | None = None) -> None:
        """Begin buffering calls for a new task, discarding any left over from the previous one."""
        self.task_id = task_id
        self.generation = generation
        if agent is not None:
            self.agent = agent
        self.calls = []

    def record(self, call: dict[str, Any]) -> None:
        """Buffer one completion and emit its trace events; wire this in as CachingLLMClient's on_call."""
        self.calls.append(call)
        emit_llm_call(self.task_id, self.agent, call, generation=self.generation)

    def drain(self) -> list[dict[str, Any]]:
        """Return the buffered calls and clear the buffer."""
        buffered, self.calls = self.calls, []
        return buffered
