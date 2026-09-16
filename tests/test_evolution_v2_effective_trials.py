"""Effective trials measure executed Train behavior, never source/name novelty."""
from types import SimpleNamespace as NS

import pytest

from evolving_loop.v2.numerical_qd import runner


def registry(rows):
    return NS(package_for=lambda task: NS(
        ranked_alternatives=tuple(NS(name=name, forecast=values)
                                  for name, values in rows[task.numeric.task_id].items()),
        protected_baseline=NS(name="anchor", forecast=(0.0,)),
        final_forecast=(0.0,),
    ))


def evidence(rows, *, parent=None, seen=(), feasible=True, applicable=("a", "b")):
    tasks = tuple(NS(numeric=NS(task_id=name, prediction_length=1)) for name in ("a", "b"))
    child = NS(candidate=NS(registry=registry(rows)), fit=NS(recipe=NS(name="new")))
    evaluation = NS(constraints=NS(feasible=feasible), task_ids=("a", "b"),
                    task_statuses={"a": "passed", "b": "passed"})
    return runner._effective_forecast_evidence(
        child, evaluation, tasks, {"a": "a" * 64, "b": "b" * 64},
        registry(parent or {"a": {"old": (1.0,)}, "b": {"old": (2.0,)}}), set(seen),
        applicable_task_ids=set(applicable),
    )


def test_renamed_dictionary_forecast_is_ineffective_even_when_anchor_differs():
    assert evidence({"a": {"new": (1.0,)}, "b": {"new": (2.0,)}})["status"] == "unchanged"


def test_distinct_recipe_counts_even_when_final_forecast_remains_anchor():
    row = evidence({"a": {"new": (3.0,)}, "b": {"new": (4.0,)}})
    assert row["status"] == "effective"
    assert evidence({"a": {"new": (3.0,)}, "b": {"new": (4.0,)}},
                    seen=(row["forecast_sha256"],))["status"] == "duplicate"


def test_legal_specialist_coverage_records_inactive_mask_without_anchor_forecast():
    row = evidence({"a": {"new": (3.0,)}, "b": {}}, applicable=("a",))
    assert row["status"] == "effective"
    assert row["covered_task_ids"] == ["a"]
    assert evidence({"a": {"new": (1.0,)}, "b": {}}, applicable=("a",))["status"] == "unchanged"


@pytest.mark.parametrize("rows,feasible,status", [
    ({"a": {"new": (3.0,)}, "b": {}}, True, "missing_forecast"),
    ({"a": {"new": (3.0,)}, "b": {"new": (4.0,)}}, False, "infeasible"),
    ({"a": {"new": (float("nan"),)}, "b": {"new": (4.0,)}}, True, "invalid_forecast"),
])
def test_failed_train_or_anchor_fallback_never_counts(rows, feasible, status):
    assert evidence(rows, feasible=feasible)["status"] == status


def test_effective_target_requires_finite_cap_before_materialization(tmp_path):
    from tests.test_evolution_v2_numerical_runner import fixture
    config, supply, folds, adapter = fixture(task_budget=1)
    with pytest.raises(ValueError, match="finite.*finalize_after"):
        runner.run_numerical_qd(tmp_path / "run", config, supply, folds, adapter,
                                min_effective_candidates=2)


def test_effective_cap_exhaustion_resume_and_identity(tmp_path):
    import json
    import time
    from tests.test_evolution_v2_numerical_runner import fixture
    config, supply, folds, adapter = fixture(task_budget=1, clock=time.monotonic)
    args = (tmp_path / "run", config, supply, folds, adapter)
    result = runner.run_numerical_qd(*args, finalize_after=1, min_effective_candidates=2)
    summary = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())["summary"]
    assert summary["effective_trials"] == {
        "attempted_generations": 1, "effective_count": 0, "minimum_effective_candidates": 2,
        "generation_cap": 1, "status": "attempt_cap_exhausted",
    }
    assert result.accepted_steps == 0
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "run").rglob("*") if p.is_file()}
    repeated = runner.run_numerical_qd(*args, finalize_after=1, min_effective_candidates=2, resume=True)
    assert repeated.effective_trials == summary["effective_trials"]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(ValueError, match="mismatch"):
        runner.run_numerical_qd(*args, finalize_after=1, min_effective_candidates=1, resume=True)


def test_ineffective_then_effective_execution_resumes_and_stops_at_target(tmp_path, monkeypatch):
    import json
    import time
    from common.llm import LLMResponse
    from tests.test_evolution_v2_numerical_runner import fixture, Client

    class Trials(Client):
        def complete(self, **kwargs):
            payload = json.loads(super().complete(**kwargs).text)
            # First proposal copies the Anchor; second changes executable output.
            self.count = getattr(self, "count", 0) + 1
            generation = self.count
            payload["source_candidates"][0]["code"] = payload["source_candidates"][0]["code"].replace(
                "seasonal_naive", f"novel_trial_{generation}").replace(
                "+ 1.0", "+ 0.0" if generation == 1 else "+ 3.0")
            return LLMResponse(json.dumps(payload))

    config, supply, folds, adapter = fixture(task_budget=4000, provider="hybrid", clock=time.monotonic)
    from evolving_loop.v2.numerical_qd.config import NumericalQDConfigV2
    payload = config.to_payload()
    payload["adapter"]["task_timeout_seconds"] = 1.0
    payload["budget"]["hard_limit_seconds"] = 0
    config = NumericalQDConfigV2.from_payload(payload)
    # Keep real sandbox execution, Train80 evaluation and persisted evidence.
    # Export/promotion is independent of whether a feasible trial is effective.
    args = (tmp_path / "run", config, supply, folds, adapter, Trials("legal"))
    paused = runner.run_numerical_qd(*args, finalize_after=3, min_effective_candidates=1, stop_after=1)
    assert paused.status == "numerical_qd_paused"
    assert paused.effective_trials["effective_count"] == 0
    assert paused.effective_trials["status"] == "paused"
    result = runner.run_numerical_qd(*args, finalize_after=3, min_effective_candidates=1, resume=True)
    assert result.effective_trials["status"] == "target_reached"
    assert result.effective_trials["attempted_generations"] == 2
    assert result.effective_trials["effective_count"] == 1
    assert result.accepted_steps == 0
    completed = json.loads((tmp_path / "run/evaluation_complete.json").read_bytes())
    assert completed["summary"]["effective_trials"] == result.effective_trials
    repeated = runner.run_numerical_qd(*args, finalize_after=3, min_effective_candidates=1, resume=True)
    assert repeated.effective_trials == result.effective_trials
