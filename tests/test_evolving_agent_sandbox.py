from __future__ import annotations

import unittest

from evolving_loop.sandbox import UnsafeCodeError, check_code, run_forecast_code

VALID_CODE = """
def forecast(history, horizon, frequency):
    last = history[-1]
    return [last for _ in range(horizon)]
"""

DISALLOWED_IMPORT_CODE = """
import os
def forecast(history, horizon, frequency):
    os.system("echo hi")
    return [0.0] * horizon
"""

FORBIDDEN_BUILTIN_CODE = """
def forecast(history, horizon, frequency):
    eval("1 + 1")
    return [0.0] * horizon
"""

WRONG_LENGTH_CODE = """
def forecast(history, horizon, frequency):
    return [0.0]
"""

NO_FORECAST_FN_CODE = """
x = 1
"""

TIMEOUT_CODE = """
def forecast(history, horizon, frequency):
    total = 0
    for i in range(10**9):
        total += i
    return [float(total)] * horizon
"""

NUMPY_CODE = """
import numpy as np
def forecast(history, horizon, frequency):
    mean = float(np.mean(history))
    return [mean for _ in range(horizon)]
"""


class CheckCodeTests(unittest.TestCase):
    def test_valid_code_passes(self):
        check_code(VALID_CODE)

    def test_disallowed_import_is_rejected(self):
        with self.assertRaises(UnsafeCodeError):
            check_code(DISALLOWED_IMPORT_CODE)

    def test_forbidden_builtin_is_rejected(self):
        with self.assertRaises(UnsafeCodeError):
            check_code(FORBIDDEN_BUILTIN_CODE)

    def test_syntax_error_is_rejected(self):
        with self.assertRaises(UnsafeCodeError):
            check_code("def forecast(:")

    def test_numpy_import_is_allowed(self):
        check_code(NUMPY_CODE)


class RunForecastCodeTests(unittest.TestCase):
    def test_valid_code_runs_and_returns_forecast(self):
        result = run_forecast_code(VALID_CODE, history=[1.0, 2.0, 3.0], horizon=3, frequency="1 day")
        self.assertTrue(result.ok)
        self.assertEqual(result.forecast, (3.0, 3.0, 3.0))
        self.assertIsNone(result.error)

    def test_unsafe_code_never_executes(self):
        result = run_forecast_code(DISALLOWED_IMPORT_CODE, history=[1.0], horizon=1, frequency="1 day")
        self.assertFalse(result.ok)
        self.assertIn("disallowed import", result.error)

    def test_wrong_length_output_is_an_error(self):
        result = run_forecast_code(WRONG_LENGTH_CODE, history=[1.0], horizon=5, frequency="1 day")
        self.assertFalse(result.ok)
        self.assertIn("expected 5", result.error)

    def test_missing_forecast_function_is_an_error(self):
        result = run_forecast_code(NO_FORECAST_FN_CODE, history=[1.0], horizon=1, frequency="1 day")
        self.assertFalse(result.ok)
        self.assertIn("forecast", result.error)

    def test_timeout_is_reported_not_hung(self):
        result = run_forecast_code(TIMEOUT_CODE, history=[1.0], horizon=1, frequency="1 day", timeout_s=1.0)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)

    def test_numpy_code_runs_under_the_memory_cap(self):
        result = run_forecast_code(NUMPY_CODE, history=[1.0, 2.0, 3.0], horizon=2, frequency="1 day")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.forecast, (2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
