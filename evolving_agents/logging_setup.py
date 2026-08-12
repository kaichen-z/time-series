"""Two-stream logging: everything to a file, a readable subset to the terminal."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

# The per-call trace is the bulk of the output and is meant to be read after the fact,
# so it goes to the file only unless the user explicitly asks to watch it live.
TRACE_LOGGER = "evolving_agents.harness.trace"

FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


class _DropTrace(logging.Filter):
    """Keeps the verbose per-call trace out of one handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(TRACE_LOGGER)


def configure(
    log_file: str | Path,
    log_level: str = "INFO",
    console_level: str = "INFO",
    trace_to_console: bool = False,
) -> Path:
    """Send full output to log_file and a filtered view to stderr; returns the resolved path."""
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(file_handler)

    # stderr, not stdout: stdout carries the summary JSON that callers redirect into a file.
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(getattr(logging, console_level))
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    if not trace_to_console:
        console.addFilter(_DropTrace())
    root.addHandler(console)

    logging.getLogger(__name__).info("full log -> %s", path)
    if not trace_to_console:
        logging.getLogger(__name__).info("per-call trace is in the log file only (--trace-console to watch it live)")
    return path


def log_exception(exc: BaseException) -> None:
    """Write an uncaught traceback into the log file, not just onto a scrolling terminal."""
    logging.getLogger("evolving_agents").error("run failed: %s\n%s", exc, "".join(traceback.format_exception(exc)))
