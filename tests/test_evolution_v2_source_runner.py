from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from evolving_loop.v2 import ResourceUse
from evolving_loop.v2.source import SourceConfigV2, run_source_evolution
from evolving_loop.v2.source import runner as source_runner
from tests.build_evolution_v2_source_fixture import build_source_case


def test_closed_validation_resume_matches_full_result(tmp_path):
    config = SourceConfigV2.smoke(seed=17)
    full = run_source_evolution(
        tmp_path / "full", config, build_source_case(tmp_path / "full-inputs")
    )
    run_source_evolution(
        tmp_path / "resumed", config,
        build_source_case(tmp_path / "partial-inputs"), stop_after="validation",
    )
    validation_checkpoint = json.loads(
        (tmp_path / "resumed" / "checkpoint.json").read_bytes()
    )
    cached_memberships = [
        entry["evaluation"]["expected_task_ids"]
        for entry in validation_checkpoint["evaluator_cache"]["aggregates"]
    ]
    cached_memberships.extend(
        evaluation["expected_task_ids"]
        for episode in validation_checkpoint["evaluator_cache"]["episodes"]
        for evaluation in episode["selected_evaluations"]
    )
    assert cached_memberships
    assert all("cooperative-04" not in membership for membership in cached_memberships)
    resumed = run_source_evolution(
        tmp_path / "resumed", config,
        build_source_case(tmp_path / "fresh-resume-inputs"), resume=True,
    )
    assert resumed.canonical_bytes() == full.canonical_bytes()
    assert ((tmp_path / "resumed" / "authority" / "active_source.json").read_bytes()
            == (tmp_path / "full" / "authority" / "active_source.json").read_bytes())
    assert ((tmp_path / "resumed" / "source_archive" / "events.jsonl").read_bytes()
            == (tmp_path / "full" / "source_archive" / "events.jsonl").read_bytes())


def test_closed_candidate_resume_matches_full_result(tmp_path):
    config = dataclasses.replace(
        SourceConfigV2.smoke(seed=17),
        max_candidates=1,
        resource_ceilings=ResourceUse(
            task_executions=144, subprocesses=8, wall_seconds=120.0,
        ),
    )
    full = run_source_evolution(
        tmp_path / "full", config, build_source_case(tmp_path / "full-inputs")
    )
    run_source_evolution(
        tmp_path / "resumed", config,
        build_source_case(tmp_path / "partial-inputs"), stop_after="candidate:1",
    )
    resume_case = build_source_case(tmp_path / "fresh-resume-inputs")
    resumed = run_source_evolution(
        tmp_path / "resumed", config, resume_case, resume=True,
    )
    assert resumed.canonical_bytes() == full.canonical_bytes()
    assert ((tmp_path / "resumed" / "authority" / "active_source.json").read_bytes()
            == (tmp_path / "full" / "authority" / "active_source.json").read_bytes())
    assert ((tmp_path / "resumed" / "source_archive" / "events.jsonl").read_bytes()
            == (tmp_path / "full" / "source_archive" / "events.jsonl").read_bytes())
    full_checkpoint = json.loads((tmp_path / "full" / "checkpoint.json").read_bytes())
    resumed_checkpoint = json.loads((tmp_path / "resumed" / "checkpoint.json").read_bytes())
    assert full_checkpoint["budget_checkpoint"]["charged_use"] == resumed_checkpoint["budget_checkpoint"]["charged_use"]
    assert resumed_checkpoint["budget_checkpoint"]["charged_use"]["task_executions"] == 92
    assert resumed_checkpoint["budget_checkpoint"]["charged_use"]["subprocesses"] == 8
    assert [request["seed"] for request in resume_case.policy_requests] == [
        0, 1, 18, 19, 18, 19,
    ]


def test_failed_validation_preserves_parent(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    original = case.evaluator.validate
    monkeypatch.setattr(case.evaluator, "validate", lambda *args: dataclasses.replace(original(*args), passed=False, reason="rejected"))
    result = run_source_evolution(tmp_path / "run", SourceConfigV2.smoke(seed=17), case)
    assert result.activated == 0
    assert (tmp_path / "run" / "authority" / "active_source.json").read_bytes() == case.seed_source.canonical_bytes()


def test_completed_resume_performs_no_evaluator_calls_or_writes(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    config = SourceConfigV2.smoke(seed=17)
    run_source_evolution(tmp_path / "run", config, case)
    before = {
        path.relative_to(tmp_path / "run"): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (tmp_path / "run").rglob("*") if path.is_file()
    }
    monkeypatch.setattr(case.evaluator, "train", lambda *_args: pytest.fail("unexpected evaluation"))
    monkeypatch.setattr(case.evaluator, "validate", lambda *_args: pytest.fail("unexpected evaluation"))
    monkeypatch.setattr(case.evaluator, "canary", lambda *_args, **_kwargs: pytest.fail("unexpected evaluation"))
    resumed = run_source_evolution(tmp_path / "run", config, case, resume=True)
    after = {
        path.relative_to(tmp_path / "run"): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (tmp_path / "run").rglob("*") if path.is_file()
    }
    assert resumed.canonical_bytes() == before[Path("evaluation_complete.json")][0]
    assert after == before


def test_changed_source_identity_rejects_resume_before_evaluation(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    config = SourceConfigV2.smoke(seed=17)
    run_source_evolution(tmp_path / "run", config, case, stop_after="candidate:1")
    changed = dataclasses.replace(case, seed_source=case.neutral_source)
    monkeypatch.setattr(changed.evaluator, "train", lambda *_args: pytest.fail("unexpected evaluation"))
    with pytest.raises(ValueError, match="resume configuration/input mismatch"):
        run_source_evolution(tmp_path / "run", config, changed, resume=True)


def test_changed_task_content_rejects_resume_before_evaluation(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    config = SourceConfigV2.smoke(seed=17)
    run_source_evolution(tmp_path / "run", config, case, stop_after="candidate:1")
    changed_task = dataclasses.replace(case.train_tasks[0], target_description="changed frozen input")
    changed = dataclasses.replace(case, train_tasks=(changed_task, *case.train_tasks[1:]))
    monkeypatch.setattr(changed.evaluator, "train", lambda *_args: pytest.fail("unexpected evaluation"))
    with pytest.raises(ValueError, match="resume configuration/input mismatch"):
        run_source_evolution(tmp_path / "run", config, changed, resume=True)


def test_closed_candidate_checkpoint_has_charged_closed_budget(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    run_source_evolution(
        tmp_path / "run", SourceConfigV2.smoke(seed=17), case,
        stop_after="candidate:1",
    )
    checkpoint = json.loads((tmp_path / "run" / "checkpoint.json").read_bytes())
    budget = checkpoint["budget_checkpoint"]
    assert budget["open_reservations"] == []
    assert budget["closed_stage_ids"] == ["candidate:1"]
    assert budget["charged_use"]["task_executions"] == 32
    assert budget["charged_use"]["subprocesses"] == 2


def test_resource_ceiling_denies_candidate_before_evaluation(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    config = dataclasses.replace(
        SourceConfigV2.smoke(seed=17),
        resource_ceilings=ResourceUse(task_executions=35, subprocesses=100, wall_seconds=120.0),
    )
    monkeypatch.setattr(case.evaluator, "train", lambda *_args: pytest.fail("unexpected evaluation"))

    result = run_source_evolution(tmp_path / "run", config, case)

    checkpoint = json.loads((tmp_path / "run" / "checkpoint.json").read_bytes())
    assert result.status == "source_evolution_checkpointed"
    assert result.proposed == 0
    assert checkpoint["budget_checkpoint"]["charged_use"]["task_executions"] == 0


def test_pre_canary_wall_boundary_resumes_with_fresh_time_and_retained_cost(tmp_path, monkeypatch):
    case = build_source_case(tmp_path / "inputs")
    config = dataclasses.replace(
        SourceConfigV2.smoke(seed=17),
        resource_ceilings=ResourceUse(task_executions=1000, subprocesses=1000, wall_seconds=120.0),
    )
    now = [0.0]
    monkeypatch.setattr(source_runner.time, "monotonic", lambda: now[0])
    original_validate = case.evaluator.validate
    calls = {"train": 0, "validate": 0, "canary": 0}
    original_train = case.evaluator.train
    original_canary = case.evaluator.canary

    def train(*args):
        calls["train"] += 1
        return original_train(*args)

    def validate(*args):
        calls["validate"] += 1
        result = original_validate(*args)
        now[0] = 119.0
        return result

    def canary(*args, **kwargs):
        calls["canary"] += 1
        return original_canary(*args, **kwargs)

    monkeypatch.setattr(case.evaluator, "train", train)
    monkeypatch.setattr(case.evaluator, "validate", validate)
    monkeypatch.setattr(case.evaluator, "canary", canary)
    checkpointed = run_source_evolution(tmp_path / "run", config, case)
    assert checkpointed.status == "source_evolution_checkpointed"
    assert calls == {"train": 2, "validate": 1, "canary": 0}
    before = json.loads((tmp_path / "run" / "checkpoint.json").read_bytes())
    charged_before = before["budget_checkpoint"]["charged_use"]

    completed = run_source_evolution(tmp_path / "run", config, case, resume=True)
    assert completed.status == "source_evolution_complete"
    assert calls == {"train": 2, "validate": 1, "canary": 1}
    after = json.loads((tmp_path / "run" / "checkpoint.json").read_bytes())
    assert after["budget_checkpoint"]["charged_use"]["task_executions"] > charged_before["task_executions"]
