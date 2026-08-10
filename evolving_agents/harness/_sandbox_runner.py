"""Executes one agent-written forecast() in an isolated subprocess; reads JSON stdin, writes JSON stdout."""

from __future__ import annotations

import inspect
import json
import sys


def _fail(message: str) -> None:
    """Report a failure to the parent process and exit cleanly."""
    sys.stdout.write(json.dumps({"ok": False, "error": message}))
    sys.exit(0)


def main() -> None:
    """Run the supplied code's forecast() and emit its output as JSON."""
    request = json.loads(sys.stdin.read())
    memory_mb = request.get("memory_mb")
    if memory_mb:
        import resource

        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    namespace: dict = {}
    try:
        exec(compile(request["code"], "<agent_code>", "exec"), namespace)  # noqa: S102 - the whole point of this process
    except Exception as exc:
        _fail(f"code failed to define: {type(exc).__name__}: {exc}")

    function = namespace.get("forecast")
    if not callable(function):
        _fail("code did not define a callable named 'forecast'")

    horizon = int(request["horizon"])
    kwargs = {}
    if "backbone" in inspect.signature(function).parameters:
        kwargs["backbone"] = request.get("backbone")

    try:
        raw = function(list(request["history"]), horizon, request["frequency"], **kwargs)
    except Exception as exc:
        _fail(f"forecast() raised: {type(exc).__name__}: {exc}")

    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        _fail(f"forecast() did not return a sequence of numbers: {exc}")

    if len(values) != horizon:
        _fail(f"forecast() returned {len(values)} values, expected {horizon}")
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        _fail("forecast() returned a non-finite value")

    sys.stdout.write(json.dumps({"ok": True, "forecast": values}))


if __name__ == "__main__":
    main()
