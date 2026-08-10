"""An MCP-inspector-style event stream: every LLM turn, tool call, and agent boundary."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACE_LEVELS = ("off", "summary", "full")
_SUMMARY_TRUNCATE = 200

_GLYPHS = {
    "agent_start": "┌",
    "agent_end": "└",
    "llm_call": "▶",
    "llm_response": "◀",
    "tool_call": "⚙",
    "tool_result": "✓",
}


@dataclass(frozen=True)
class TraceEvent:
    """One observable thing an agent did, rendered to the log and optionally to a sidecar."""

    task_id: str
    agent: str
    event_type: str
    detail: dict[str, Any] = field(default_factory=dict)
    generation: int | None = None
    timestamp: str = ""


@dataclass
class _TraceState:
    """Process-wide tracing configuration, set once from the CLI."""

    level: str = "off"
    runs_dir: Path | None = None
    events: list[TraceEvent] = field(default_factory=list)
    collect: bool = False


_STATE = _TraceState()


def configure_tracing(level: str = "off", runs_dir: str | Path | None = None, collect: bool = False) -> None:
    """Set the global trace level, sidecar location, and whether events are kept in memory."""
    if level not in TRACE_LEVELS:
        raise ValueError(f"Unknown trace level {level!r}, expected one of {TRACE_LEVELS}")
    _STATE.level = level
    _STATE.runs_dir = Path(runs_dir).expanduser().resolve() if runs_dir else None
    _STATE.collect = collect
    _STATE.events = []


def collected_events() -> tuple[TraceEvent, ...]:
    """Return the events captured since configure_tracing(collect=True); for tests and analysis."""
    return tuple(_STATE.events)


def current_level() -> str:
    """Return the active trace level."""
    return _STATE.level


def _truncate(text: str, limit: int = _SUMMARY_TRUNCATE) -> str:
    """Shorten a string for one-line summary rendering."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


def _write_reasoning_sidecar(event: TraceEvent) -> str | None:
    """Persist a reasoning block next to the run and return its path, so summary logs stay readable."""
    reasoning = event.detail.get("reasoning")
    if not reasoning or _STATE.runs_dir is None:
        return None
    model_slug = str(event.detail.get("model_id", "unknown")).replace("/", "__")
    digest = event.detail.get("prompt_hash") or hashlib.sha256(str(reasoning).encode("utf-8")).hexdigest()
    path = _STATE.runs_dir / "reasoning" / model_slug / f"{digest}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(reasoning), encoding="utf-8")
    return str(path)


def _format(event: TraceEvent) -> list[str]:
    """Render one event as the console/log lines shown at the active trace level."""
    glyph = _GLYPHS.get(event.event_type, "·")
    where = f"{event.agent}" if event.generation is None else f"gen{event.generation:02d}/{event.agent}"
    head = f"{glyph} [{event.task_id}] {where:<24} {event.event_type.upper():<13}"
    detail = event.detail
    full = _STATE.level == "full"

    if event.event_type == "llm_call":
        body = (
            f"({detail.get('model_id', '?')}, temp={detail.get('temperature', 0.0)}, "
            f"draw={detail.get('draw_index', 0)}, thinking={detail.get('enable_thinking', False)})"
        )
        lines = [f"{head} {body}"]
        if full and detail.get("user_text"):
            lines.append(f"    prompt: {detail['user_text']}")
        return lines

    if event.event_type == "llm_response":
        answer = detail.get("answer", detail.get("response_text", ""))
        lines = [f"{head} {answer if full else _truncate(answer)}"]
        reasoning = detail.get("reasoning")
        if reasoning:
            if full:
                lines.append(f"    reasoning: {reasoning}")
            else:
                sidecar = _write_reasoning_sidecar(event)
                lines.append(f"    reasoning: {len(str(reasoning))} chars -> {sidecar}" if sidecar else f"    reasoning: {len(str(reasoning))} chars")
        return lines

    if event.event_type == "tool_call":
        args = detail.get("args", {})
        rendered = args if full else _truncate(str(args))
        return [f"{head} {detail.get('tool', '?')}({rendered})"]

    if event.event_type == "tool_result":
        return [f"{head} " + " ".join(f"{name}={value}" for name, value in detail.items() if name != "reasoning")]

    return [f"{head} " + _truncate(str(detail))]


def emit(event: TraceEvent) -> None:
    """Record one trace event, rendering it at the active level and keeping it if collecting."""
    if _STATE.level == "off":
        return
    stamped = event if event.timestamp else TraceEvent(
        task_id=event.task_id,
        agent=event.agent,
        event_type=event.event_type,
        detail=event.detail,
        generation=event.generation,
        timestamp=time.strftime("%H:%M:%S"),
    )
    if _STATE.collect:
        _STATE.events.append(stamped)
    for line in _format(stamped):
        logger.info("%s", line)


def emit_llm_call(task_id: str, agent: str, record: dict[str, Any], generation: int | None = None) -> None:
    """Emit the request half of one completion, from a CachingLLMClient on_call record.

    The response half is emitted by whichever agent parses it, so the rendered answer is the
    structured one and a raw duplicate never lands beside it.
    """
    emit(TraceEvent(task_id=task_id, agent=agent, event_type="llm_call", detail=record, generation=generation))


def emit_llm_response(
    task_id: str,
    agent: str,
    answer: str,
    reasoning: str | None = None,
    model_id: str = "?",
    generation: int | None = None,
    prompt_hash: str | None = None,
) -> None:
    """Emit the response half of one completion, after the caller has separated reasoning from answer."""
    emit(
        TraceEvent(
            task_id=task_id,
            agent=agent,
            event_type="llm_response",
            generation=generation,
            detail={"answer": answer, "reasoning": reasoning, "model_id": model_id, "prompt_hash": prompt_hash},
        )
    )
