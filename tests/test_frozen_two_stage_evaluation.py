from __future__ import annotations

import hashlib
import json

import pytest

from numerical_agent.evaluate_frozen_two_stage import (
    ForecastResult,
    build_parser,
    score_forecast_results,
    verify_frozen_policies,
)
from numerical_agent.evolution.execution import Task


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def test_freeze_verifier_rejects_hash_mismatch_and_existing_completion(tmp_path):
    screen = tmp_path / "screen"
    selector = tmp_path / "selector"
    output = tmp_path / "out"
    screen.mkdir(); selector.mkdir(); output.mkdir()
    screen_hash = _write(screen / "frozen_screening_policy.py", "screen")
    decision_hash = _write(selector / "frozen_decision_policy.py", "decision")
    (screen / "screening_manifest.json").write_text(json.dumps({
        "frozen_screening_policy_sha256": screen_hash,
        "public_test_accessed": False,
    }))
    (selector / "selector_manifest.json").write_text(json.dumps({
        "screening_policy_sha256": screen_hash,
        "frozen_decision_policy_sha256": decision_hash,
        "public_test_accessed": False,
    }))
    assert verify_frozen_policies(screen, selector, output) == (screen_hash, decision_hash)
    (selector / "frozen_decision_policy.py").write_text("changed")
    with pytest.raises(ValueError, match="decision"):
        verify_frozen_policies(screen, selector, output)
    (selector / "frozen_decision_policy.py").write_text("decision")
    (output / "evaluation_complete.json").write_text("{}")
    with pytest.raises(ValueError, match="already"):
        verify_frozen_policies(screen, selector, output)


def test_evaluation_cli_has_no_llm_or_mutation_options():
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    assert "codex_model" not in options
    assert "generations" not in options
    assert "llm_backend" not in options


def test_score_reports_mean_median_rmsse_and_diversity():
    task = Task("t", (1.0, 2.0, 3.0), 2, "D", (4.0, 5.0))
    perfect = ForecastResult("t", (4.0, 5.0), ("a",), ("statistical",), "single")
    score = score_forecast_results((task,), (perfect,))
    assert score["coverage"] == 1.0
    assert score["mean_mase"] == 0.0
    assert score["median_mase"] == 0.0
    assert score["mean_rmsse"] == 0.0
    assert score["method_diversity"] == 1
    assert score["family_diversity"] == 1

