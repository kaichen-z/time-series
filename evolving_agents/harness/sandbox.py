"""Runs agent-written forecast code behind a static import allow-list and a subprocess timeout."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import SandboxResult
from .trace import TraceEvent, emit

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).parent / "_sandbox_runner.py"

ALLOWED_IMPORTS = frozenset(
    {"numpy", "pandas", "scipy", "statsmodels", "math", "statistics", "itertools", "functools", "collections"}
)
# Names that would let sandboxed code escape the allow-list by resolving imports or files at runtime.
FORBIDDEN_NAMES = frozenset({"__import__", "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "vars"})
FORBIDDEN_ATTRIBUTES = frozenset({"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__", "__code__"})
_PASSTHROUGH_ENV = frozenset({"PATH", "HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV"})
_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class UnsafeCodeError(ValueError):
    """Raised when generated code fails the static check before it is ever executed."""


@dataclass(frozen=True)
class SandboxConfig:
    """Limits applied to every sandboxed execution."""

    timeout_s: float = 10.0
    memory_mb: int = 2048
    allowed_imports: frozenset[str] = field(default_factory=lambda: ALLOWED_IMPORTS)


def code_hash(code: str) -> str:
    """Return a short stable digest identifying a piece of generated code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def _root_module(name: str) -> str:
    """Return the top-level package of a dotted module path."""
    return name.split(".")[0]


def _sandbox_env() -> dict[str, str]:
    """Build the child env: keep what the interpreter needs to import, neutralize outbound proxies.

    HOME/PYTHONPATH must survive or user site-packages (numpy, scipy) stop resolving; the import
    allow-list, not the environment, is what actually keeps generated code from reaching the network.
    Pinning the BLAS thread counts is load-bearing, not tuning: OpenBLAS reserves a large virtual
    arena per thread, and under RLIMIT_AS that reservation fails and `import numpy` spins forever.
    """
    env = {name: value for name, value in os.environ.items() if name in _PASSTHROUGH_ENV}
    env.update({"PYTHONHASHSEED": "0", "no_proxy": "*", "NO_PROXY": "*", "http_proxy": "", "https_proxy": ""})
    env.update(dict.fromkeys(_THREAD_ENV, "1"))
    return env


def check_code(code: str, allowed_imports: frozenset[str] = ALLOWED_IMPORTS) -> None:
    """Reject code that fails to parse or reaches outside the allow-list; raises UnsafeCodeError."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"code does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_module(alias.name) not in allowed_imports:
                    raise UnsafeCodeError(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and _root_module(node.module) not in allowed_imports:
                raise UnsafeCodeError(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            raise UnsafeCodeError(f"access to {node.attr!r} is not allowed")


def run_forecast_code(
    code: str,
    history: tuple[float, ...],
    horizon: int,
    frequency: str,
    config: SandboxConfig | None = None,
    backbone: tuple[float, ...] | None = None,
    task_id: str = "-",
    generation: int | None = None,
) -> SandboxResult:
    """Execute one forecast() in a subprocess and return its output or the reason it failed."""
    settings = config or SandboxConfig()
    digest = code_hash(code)
    emit(
        TraceEvent(
            task_id=task_id,
            agent="coding.sandbox",
            event_type="tool_call",
            generation=generation,
            detail={"tool": "sandbox.execute", "args": {"code_hash": digest, "timeout_s": settings.timeout_s, "horizon": horizon}},
        )
    )

    start = time.monotonic()

    def _result(ok: bool, forecast: tuple[float, ...] | None, error: str | None) -> SandboxResult:
        """Build the result, emit its trace event, and hand it back."""
        duration_ms = (time.monotonic() - start) * 1000
        emit(
            TraceEvent(
                task_id=task_id,
                agent="coding.sandbox",
                event_type="tool_result",
                generation=generation,
                detail={"ok": ok, "duration_ms": round(duration_ms, 1), "error": error} if not ok
                else {"ok": ok, "duration_ms": round(duration_ms, 1), "forecast_len": len(forecast or ())},
            )
        )
        return SandboxResult(ok=ok, forecast=forecast, error=error, duration_ms=duration_ms, code_hash=digest)

    try:
        check_code(code, settings.allowed_imports)
    except UnsafeCodeError as exc:
        logger.warning("sandbox[%s]: rejected %s before execution: %s", task_id, digest, exc)
        return _result(False, None, f"rejected by static check: {exc}")

    request = json.dumps(
        {
            "code": code,
            "history": list(history),
            "horizon": horizon,
            "frequency": frequency,
            "memory_mb": settings.memory_mb,
            "backbone": list(backbone) if backbone is not None else None,
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(RUNNER)],
            input=request,
            capture_output=True,
            text=True,
            timeout=settings.timeout_s,
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        return _result(False, None, f"timed out after {settings.timeout_s}s")
    except OSError as exc:
        return _result(False, None, f"failed to start sandbox: {exc}")

    if completed.returncode != 0:
        return _result(False, None, f"sandbox exited {completed.returncode}: {(completed.stderr or '').strip()[:500]}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return _result(False, None, f"sandbox produced unreadable output: {completed.stdout[:500]!r}")

    if not payload.get("ok"):
        return _result(False, None, str(payload.get("error", "unknown sandbox failure")))
    return _result(True, tuple(float(value) for value in payload["forecast"]), None)
