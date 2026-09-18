from __future__ import annotations

import pytest

from evolving_loop.adjustment.llm_design import (
    audit, design_loop, make_regime_estimator, safe_compile,
)

_GOOD = '''
def estimate(history_values, history_timestamps, future_timestamps):
    wk = [v for v, t in zip(history_values, history_timestamps) if weekday(t) < 5]
    we = [v for v, t in zip(history_values, history_timestamps) if weekday(t) >= 5]
    if not wk or not we or mean(wk) == 0:
        return None
    ratio = mean(we) / mean(wk)
    return [ratio for _ in future_timestamps]
'''


def test_safe_good_function_compiles_and_runs():
    fn = safe_compile(_GOOD)
    hts = [f"2024-06-{d:02d}T00:00:00" for d in range(3, 17)]
    hv = [10.0 if __import__("datetime").date(2024, 6, d).weekday() < 5 else 6.0
          for d in range(3, 17)]
    out = fn(hv, hts, ["2024-06-17T00:00:00", "2024-06-18T00:00:00"])
    assert len(out) == 2 and abs(out[0] - 0.6) < 1e-9


@pytest.mark.parametrize("code", [
    "import os\ndef estimate(a,b,c):\n    return None",
    "def estimate(a,b,c):\n    return open('/etc/passwd').read()",
    "def estimate(a,b,c):\n    return __import__('os').listdir('.')",
    "def estimate(a,b,c):\n    return ().__class__.__bases__",
    "def estimate(a,b,c):\n    while True:\n        pass",
    "def estimate(a,b,c):\n    return eval('1+1')",
])
def test_dangerous_code_is_rejected(code):
    with pytest.raises(ValueError):
        safe_compile(code)


def test_wrapped_estimator_is_defensive_on_bad_output():
    # returns wrong length -> None (dropped), never crashes the pipeline
    fn = safe_compile("def estimate(a,b,c):\n    return [1.0]")
    est = make_regime_estimator(fn)
    assert est([1, 2], ["t1", "t2"], None, ["f1", "f2", "f3"], (True, True, True)) is None
    # a crashing function -> None
    fn2 = safe_compile("def estimate(a,b,c):\n    return c[999]")
    est2 = make_regime_estimator(fn2)
    assert est2([1], ["t"], None, ["f"], (True,)) is None


def test_design_loop_selects_the_best_proposal():
    # three proposals; the loop should keep the one the evaluator scores highest.
    proposals = [
        "def estimate(a,b,c):\n    return [0.9 for _ in c]",   # scored 0.9
        "import os\ndef estimate(a,b,c):\n    return [0.0]",   # rejected by sandbox
        "def estimate(a,b,c):\n    return [0.5 for _ in c]",   # scored 0.5
    ]
    it = iter(proposals)
    proposer = lambda archive: next(it)

    # evaluator: reward the estimator whose factor (on a dummy call) is closest to 0.9
    def evaluate(est):
        out = est([1.0], ["t"], None, ["f"], (True,))
        return -abs(out[0] - 0.9) if out else -99.0

    archive, log = design_loop(proposer, evaluate, rounds=3, keep=2)
    assert any(r["status"] == "rejected" for r in log)      # the import was caught
    assert archive and abs(archive[0][0]) < 1e-9            # best == the 0.9 proposal
    assert "0.9" in archive[0][1]
