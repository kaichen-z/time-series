"""Subprocess entry point: exec() untrusted forecast code and report a JSON result on stdout."""
from __future__ import annotations

import json
import math
import resource
import sys


def _apply_memory_limit(memory_mb: int) -> None:
    """Cap this process's address space so a runaway allocation gets killed, not the host."""
    limit_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def main() -> None:
    request = json.loads(sys.stdin.read())
    _apply_memory_limit(request["memory_mb"])

    namespace: dict = {}
    try:

        exec(request["code"], namespace)  # sandbox's execution 
        forecast_fn = namespace.get("forecast")

        if forecast_fn is None:
            raise ValueError("code did not define a forecast(history, horizon, frequency) function")

        result = forecast_fn(request["history"], request["horizon"], request["frequency"])
        result = [float(v) for v in result]

        if len(result) != request["horizon"]:
            raise ValueError(f"forecast() returned {len(result)} values, expected {request['horizon']}")

        if not all(math.isfinite(v) for v in result):
            raise ValueError("forecast() returned a non-finite value")
        
        print(json.dumps({"ok": True, "forecast": result, "error": None}))

    except BaseException as exc:  # Failure must become a reported error, not a crash
        print(json.dumps({"ok": False, "forecast": None, "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
