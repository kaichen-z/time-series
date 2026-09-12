"""Observable isolation guarantees for the bounded cooperative prototype."""

from __future__ import annotations

import json

import pytest

from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2 import cooperative as cooperative_api
from evolving_loop.v2.cooperative import runner as cooperative_runner
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from tests.test_evolution_v2_cooperative_runner import (
    _Config,
    _Pipeline,
    _control,
    _pair,
)


def _case(tmp_path):
    seed_pair = _pair(alternate=False)
    alternate_pair = _pair(alternate=True)
    retrieval = RetrievalModuleV2(1, "a" * 64, RetrievalGenome.seed().to_payload(), ())
    decision = DecisionModuleV2(1, "seed prompt", (), True, 2, "last")
    seed_artifacts = {
        "numerical": seed_pair,
        "retrieval": retrieval,
        "decision": decision,
    }
    tasks = {
        "train": tuple(f"train-{index}" for index in range(4)),
        "dev": ("dev-0",),
    }

    def run(*, tasks_override=None, directory="run"):
        pipeline = _Pipeline(alternate_pair.release.fingerprint, lambda: 0.0)
        pipeline.seed_retrieval_sha = retrieval.fingerprint()
        pipeline.seed_decision_sha = decision.fingerprint()
        adapters = {
            "numerical": cooperative_api.NumericalCoordinateAdapter((alternate_pair,)),
            "retrieval": cooperative_api.RetrievalCoordinateAdapter(),
            "decision": cooperative_api.DecisionCoordinateAdapter(("changed prompt",)),
            "pipeline": pipeline,
            "monotonic": lambda: 0.0,
        }
        return cooperative_api.run_cooperative_evolution(
            tmp_path / directory,
            _Config(_control("ucb")),
            seed_artifacts,
            tasks if tasks_override is None else tasks_override,
            adapters,
        )

    return tasks, run


def test_public_membership_is_rejected_before_proposal(tmp_path, monkeypatch):
    tasks, run = _case(tmp_path)
    proposals = []
    original = cooperative_runner.propose_bundle_candidate

    def record_proposal(*args, **kwargs):
        proposals.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(cooperative_runner, "propose_bundle_candidate", record_proposal)

    with pytest.raises(ValueError, match="Public"):
        run(tasks_override=tasks | {"public": ("public-0",)})

    assert proposals == []


def test_dev_values_never_enter_scheduler_or_proposal_feedback(tmp_path, monkeypatch):
    _, run = _case(tmp_path)
    feedback_payloads = []
    original = cooperative_runner.propose_bundle_candidate

    def capture_feedback(parent, arm, catalog, adapters, feedback, step):
        feedback_payloads.append(canonical_v2_bytes(feedback.to_payload()))
        return original(parent, arm, catalog, adapters, feedback, step)

    monkeypatch.setattr(
        cooperative_runner, "propose_bundle_candidate", capture_feedback
    )
    run()

    scheduler_payloads = []
    for path in (tmp_path / "run" / "objects").glob("*.json"):
        payload = json.loads(path.read_bytes())
        if {"mode", "draw_counter", "completed_step", "arms"} <= set(payload):
            scheduler_payloads.append(path.read_bytes())
    payloads = (*feedback_payloads, *scheduler_payloads)
    forbidden = (b"future_values", b"dev_metrics", b"task_id", b"forecast")

    assert len(feedback_payloads) == 4
    assert scheduler_payloads
    assert all(
        token not in payload.lower() for payload in payloads for token in forbidden
    )


def test_runner_persists_no_per_forecast_artifacts(tmp_path):
    _, run = _case(tmp_path)
    result = run()
    files = [
        str(path.relative_to(tmp_path / "run"))
        for path in (tmp_path / "run").rglob("*")
        if path.is_file()
    ]
    aggregate_objects = [
        path
        for path in (tmp_path / "run" / "objects").glob("*.json")
        if json.loads(path.read_bytes()).get("kind") == "cooperative_aggregate"
    ]

    assert result.status == "cooperative_complete"
    assert not any("forecast" in path or "task_result" in path for path in files)
    assert len(aggregate_objects) <= 8
