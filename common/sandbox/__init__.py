"""Safe, isolated execution of LLM-written forecast() code."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ALLOWED_IMPORTS = frozenset({"numpy", "math", "statistics", "itertools", "functools", "collections", "statsmodels"})
FORBIDDEN_NAMES = frozenset({"__import__", "eval", "exec","compile", "open", "input", "breakpoint"})

_RUNNER = Path(__file__).parent / "_sandbox_runner.py"

_PASSTHROUGH_ENV = ("PATH", "HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL")
_SINGLE_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class UnsafeCodeError(ValueError):
    """Raised when submitted code fails the static safety check."""


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of running one candidate forecast() implementation."""

    ok: bool
    forecast: tuple[float, ...] | None
    error: str | None
    duration_ms: float


def check_code(
    code: str,
    allowed: frozenset[str] | None = None,
    allowed_dunders: frozenset[str] = frozenset(),
) -> None:
    """Statically reject disallowed imports and dangerous builtins before anything runs.

    Callers may widen the import allow-list for their own runtime; the default set stays
    narrow because it also gates evolving_loop's published-results code.
    """
    permitted = ALLOWED_IMPORTS if allowed is None else allowed
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, permitted)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import leaves node.module as None, which used to slip past the
                # root check entirely; there is no package context here in any case.
                raise UnsafeCodeError("relative imports are not allowed")
            _check_module(node.module or "", permitted)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"use of forbidden name: {node.id}")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr not in allowed_dunders
        ):
            raise UnsafeCodeError(f"use of forbidden dunder attribute: {node.attr}")


def _check_module(module_name: str, permitted: frozenset[str]) -> None:
    """Permit a module by its root package, or by its exact dotted name.

    Exact names let a caller allow one specific module without opening its whole package.
    """
    if module_name in permitted:
        return
    root = module_name.split(".")[0]
    if root and root not in permitted:
        raise UnsafeCodeError(f"disallowed import: {module_name}")


def _sandbox_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _PASSTHROUGH_ENV if name in os.environ}
    env.update({name: "1" for name in _SINGLE_THREAD_ENV})
    return env


def run_forecast_code(
    code: str,
    history: list[float],
    horizon: int,
    frequency: str,
    timeout_s: float = 10.0,
    memory_mb: int = 1024,
) -> SandboxResult:
    """Run code's forecast(history, horizon, frequency) in an isolated subprocess."""
    try:
        check_code(code)
    except UnsafeCodeError as exc:
        return SandboxResult(ok=False, forecast=None, error=str(exc), duration_ms=0.0)

    request = {
        "code": code,
        "history": list(history),
        "horizon": horizon,
        "frequency": frequency,
        "memory_mb": memory_mb,
    }
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(_RUNNER)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - start) * 1000
        return SandboxResult(ok=False, forecast=None, error=f"timed out after {timeout_s}s", duration_ms=duration_ms)

    duration_ms = (time.monotonic() - start) * 1000
    if completed.returncode != 0 or not completed.stdout.strip():
        error = completed.stderr.strip() or f"sandbox process exited with code {completed.returncode}"
        return SandboxResult(ok=False, forecast=None, error=error, duration_ms=duration_ms)

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload["ok"]:
        return SandboxResult(ok=True, forecast=tuple(payload["forecast"]), error=None, duration_ms=duration_ms)
    return SandboxResult(ok=False, forecast=None, error=payload["error"], duration_ms=duration_ms)
