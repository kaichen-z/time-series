"""Static rejection, execution limits, and output validation for agent-written code."""

from __future__ import annotations

import pytest

from evolving_agents.harness.sandbox import (
    SandboxConfig,
    UnsafeCodeError,
    check_code,
    code_hash,
    run_forecast_code,
)

PERSISTENCE = "def forecast(history, horizon, frequency):\n    return [history[-1]] * horizon\n"


def test_happy_path_returns_the_forecast() -> None:
    result = run_forecast_code(PERSISTENCE, (1.0, 2.0, 3.0), 4, "H")
    assert result.ok
    assert result.forecast == (3.0, 3.0, 3.0, 3.0)
    assert result.error is None


def test_allowed_import_runs() -> None:
    code = "import numpy as np\ndef forecast(history, horizon, frequency):\n    return list(np.full(horizon, float(np.mean(history))))\n"
    result = run_forecast_code(code, (2.0, 4.0), 3, "H")
    assert result.ok
    assert result.forecast == (3.0, 3.0, 3.0)


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef forecast(h, z, f):\n    return [0.0] * z\n",
        "import socket\ndef forecast(h, z, f):\n    return [0.0] * z\n",
        "from subprocess import run\ndef forecast(h, z, f):\n    return [0.0] * z\n",
        "def forecast(h, z, f):\n    __import__('os').system('echo hi')\n    return [0.0] * z\n",
        "def forecast(h, z, f):\n    eval('1+1')\n    return [0.0] * z\n",
        "def forecast(h, z, f):\n    open('/etc/passwd').read()\n    return [0.0] * z\n",
        "def forecast(h, z, f):\n    return (0.0).__class__.__bases__\n",
    ],
)
def test_escape_attempts_are_rejected_before_execution(code: str) -> None:
    with pytest.raises(UnsafeCodeError):
        check_code(code)
    result = run_forecast_code(code, (1.0,), 2, "H")
    assert not result.ok
    assert "rejected by static check" in result.error


def test_syntax_error_is_rejected() -> None:
    result = run_forecast_code("def forecast(:\n", (1.0,), 2, "H")
    assert not result.ok
    assert "does not parse" in result.error


def test_timeout_is_enforced() -> None:
    result = run_forecast_code("def forecast(h, z, f):\n    while True:\n        pass\n", (1.0,), 2, "H", SandboxConfig(timeout_s=2.0))
    assert not result.ok
    assert "timed out" in result.error


def test_memory_cap_is_enforced() -> None:
    code = "def forecast(h, z, f):\n    blob = bytearray(4 * 1024 * 1024 * 1024)\n    return [float(len(blob))] * z\n"
    result = run_forecast_code(code, (1.0,), 2, "H", SandboxConfig(timeout_s=30.0, memory_mb=256))
    assert not result.ok
    assert "MemoryError" in result.error or "exited" in result.error


def test_numpy_still_imports_under_the_memory_cap() -> None:
    code = "import numpy as np\ndef forecast(h, z, f):\n    return list(np.zeros(z))\n"
    result = run_forecast_code(code, (1.0,), 3, "H", SandboxConfig(timeout_s=30.0, memory_mb=2048))
    assert result.ok, result.error


def test_wrong_length_is_rejected() -> None:
    result = run_forecast_code("def forecast(h, z, f):\n    return [1.0, 2.0]\n", (1.0,), 5, "H")
    assert not result.ok
    assert "expected 5" in result.error


def test_non_finite_output_is_rejected() -> None:
    result = run_forecast_code("def forecast(h, z, f):\n    return [float('nan')] * z\n", (1.0,), 2, "H")
    assert not result.ok
    assert "non-finite" in result.error


def test_non_numeric_output_is_rejected() -> None:
    result = run_forecast_code("def forecast(h, z, f):\n    return ['a', 'b']\n", (1.0,), 2, "H")
    assert not result.ok
    assert "sequence of numbers" in result.error


def test_missing_forecast_function_is_reported() -> None:
    result = run_forecast_code("def predict(h, z, f):\n    return [0.0] * z\n", (1.0,), 2, "H")
    assert not result.ok
    assert "callable named 'forecast'" in result.error


def test_runtime_exception_is_reported_not_raised() -> None:
    result = run_forecast_code("def forecast(h, z, f):\n    return [1.0 / 0] * z\n", (1.0,), 2, "H")
    assert not result.ok
    assert "ZeroDivisionError" in result.error


def test_backbone_is_passed_only_when_requested() -> None:
    code = "def forecast(history, horizon, frequency, backbone):\n    return list(backbone)\n"
    result = run_forecast_code(code, (1.0,), 3, "H", backbone=(7.0, 8.0, 9.0))
    assert result.ok
    assert result.forecast == (7.0, 8.0, 9.0)


def test_three_arg_signature_still_works_with_a_backbone_available() -> None:
    result = run_forecast_code(PERSISTENCE, (5.0,), 2, "H", backbone=(1.0, 2.0))
    assert result.ok
    assert result.forecast == (5.0, 5.0)


def test_code_hash_is_stable_and_differs_per_code() -> None:
    assert code_hash(PERSISTENCE) == code_hash(PERSISTENCE)
    assert code_hash(PERSISTENCE) != code_hash(PERSISTENCE + "\n")
