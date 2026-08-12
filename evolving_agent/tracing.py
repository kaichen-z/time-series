"""Structured run logging: the log file gets every event as JSON, the console gets one line per task."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_file_logger = logging.getLogger("evolving_agent.trace_file")
_console_logger = logging.getLogger("evolving_agent.trace_console")


@dataclass(frozen=True)
class TraceEvent:
    """One structured record: a task-level milestone, an LLM call, or a sandbox run."""

    task_id: str
    mode: str
    event_type: str
    detail: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def configure(log_file: str | Path, console_level: str = "INFO") -> None:
    """Wire up the file logger (every event, JSON-lines) and the console logger (task summaries only)."""
    for logger in (_file_logger, _console_logger):
        logger.handlers.clear()
        logger.propagate = False

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    _file_logger.addHandler(file_handler)
    _file_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _console_logger.addHandler(console_handler)
    _console_logger.setLevel(getattr(logging, console_level.upper()))


def emit(event: TraceEvent) -> None:
    """Write the full event to the file always; print a short summary for task-level events only."""
    _file_logger.debug(json.dumps(asdict(event)))
    if event.event_type in ("task_start", "task_end"):
        _console_logger.info(_summary_line(event))


def _summary_line(event: TraceEvent) -> str:
    if event.event_type == "task_end":
        score = event.detail.get("score")
        action = event.detail.get("action")
        skill = event.detail.get("skill_name")
        line = f"[{event.mode}] {event.task_id}: action={action} skill={skill} smape={score}"
        error = event.detail.get("error")
        if error:
            line += f" error={error!r}"
        return line
    return f"[{event.mode}] {event.task_id}: starting"
