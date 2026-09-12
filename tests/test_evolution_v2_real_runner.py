from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.runner import (
    RealStagePorts,
    SealedStageV2,
    run_real_evolution,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class SimulatedCrash(RuntimeError):
    pass


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += seconds


class RunnerCase:
    def __init__(self, tmp_path: Path) -> None:
        self.output = tmp_path / "real"
        self.clock = FakeClock()
        self.stage_durations = {"p2": 840, "p3": 360, "p4": 120, "p5": 120}
        self.stage_statuses = {stage: "complete" for stage in self.stage_durations}
        self.public_flags = {stage: False for stage in self.stage_durations}
        self.grants: dict[str, int] = {}
        self.calls: Counter[str] = Counter()
        self.order: list[str] = []
        self.crashing_stage: str | None = None
        self.finalization_started_at: float | None = None
        self.manifest = RealEvolutionManifestV2.from_payload(
            {
                "schema_version": 1,
                "profile": "real-30m",
                "model": {
                    "schema_version": 1,
                    "name": "gpt-5.6-luna",
                    "reasoning_effort": "medium",
                },
                "files": [
                    {"role": "split", "relative_path": "inputs/split.json", "sha256": digest("split")},
                    {"role": "tasks", "relative_path": "inputs/tasks.json", "sha256": digest("tasks")},
                ],
                "runtime_locations": [
                    {"role": "python", "relative_path": "runtime/python", "identity_sha256": digest("python")}
                ],
                "l0_fingerprints": {"metric": digest("metric")},
            }
        )

    def crash_during(self, stage: str) -> None:
        self.crashing_stage = stage

    def _run(self, stage: str, context):
        self.calls[stage] += 1
        self.order.append(f"run_{stage}")
        self.grants[stage] = context.grant_seconds
        self.clock.advance(self.stage_durations[stage])
        if self.crashing_stage == stage:
            raise SimulatedCrash(stage)
        payload = {"stage": stage, "status": self.stage_statuses[stage]}
        (context.output_dir / "evaluation_complete.json").write_bytes(canonical_v2_bytes(payload))
        return payload

    def _seal(self, stage: str, context, result):
        self.order.append(f"seal_{stage}")
        payload = {"stage": stage, "completion": fingerprint_payload(result)}
        return SealedStageV2(
            completion_sha256=fingerprint_payload(result),
            handoff_payload=payload,
            summary={"status": result["status"]},
            public_test_accessed=self.public_flags[stage],
        )

    @property
    def ports(self) -> RealStagePorts:
        return RealStagePorts(
            run_p2=lambda context: self._run("p2", context),
            seal_p2=lambda context, result: self._seal("p2", context, result),
            run_p3=lambda context: self._run("p3", context),
            seal_p3=lambda context, result: self._seal("p3", context, result),
            run_p4=lambda context: self._run("p4", context),
            seal_p4=lambda context, result: self._seal("p4", context, result),
            run_p5=lambda context: self._run("p5", context),
            seal_p5=lambda context, result: self._seal("p5", context, result),
            begin_finalization=lambda: setattr(self, "finalization_started_at", self.clock()),
        )

    def run(self):
        return run_real_evolution(self.output, self.manifest, self.ports, monotonic=self.clock)

    def resume(self):
        return self.run()

    def grant(self, stage: str) -> int:
        return self.grants[stage]

    def closed_charge(self, stage: str) -> int:
        import json

        checkpoint = json.loads((self.output / "checkpoint.json").read_text())
        return next(row["charged_seconds"] for row in checkpoint["stage_records"] if row["stage"] == stage)


@pytest.fixture
def case(tmp_path: Path) -> RunnerCase:
    return RunnerCase(tmp_path)


def test_base_grants_and_call_order(case: RunnerCase):
    result = case.run()

    assert result.status == "complete"
    assert case.grants == {"p2": 840, "p3": 360, "p4": 120, "p5": 120}
    assert case.order == [
        "run_p2", "seal_p2", "run_p3", "seal_p3", "run_p4", "seal_p4", "run_p5", "seal_p5",
    ]


def test_unused_time_rolls_forward_without_touching_reserve(case: RunnerCase):
    case.stage_durations.update(p2=600, p3=400, p4=100, p5=100)

    result = case.run()

    assert case.grants == {"p2": 840, "p3": 600, "p4": 320, "p5": 240}
    assert result.status == "complete"
    assert case.finalization_started_at is not None
    assert case.finalization_started_at <= 1440


def test_finalization_reserve_is_not_borrowed(case: RunnerCase):
    case.stage_durations["p2"] = 840
    case.stage_durations["p3"] = 360
    case.stage_durations["p4"] = 120
    case.stage_durations["p5"] = 121

    result = case.run()

    assert result.status == "failed"
    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1, "p5": 1})
    assert case.finalization_started_at is None


def test_overrun_stops_later_stages(case: RunnerCase):
    case.stage_durations["p2"] = 841

    result = case.run()

    assert result.status == "failed"
    assert case.calls == Counter({"p2": 1})


def test_stopped_child_returns_incomplete(case: RunnerCase):
    case.stage_statuses["p3"] = "incomplete"

    result = case.run()

    assert result.status == "incomplete"
    assert case.calls == Counter({"p2": 1, "p3": 1})


def test_public_evidence_blocks_completion(case: RunnerCase):
    case.public_flags["p4"] = True

    result = case.run()

    assert result.status == "failed"
    assert not (case.output / "evaluation_complete.json").exists()
    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1})


def test_unknown_inflight_work_is_fully_charged_and_not_replayed(case: RunnerCase):
    case.crash_during("p3")
    with pytest.raises(SimulatedCrash):
        case.run()

    resumed = case.resume()

    assert resumed.status == "incomplete"
    assert case.calls["p3"] == 1
    assert case.closed_charge("p3") == case.grant("p3")


def test_completed_resume_is_read_only_and_returns_identical_result(case: RunnerCase):
    first = case.run()
    before = {path: path.read_bytes() for path in case.output.rglob("*") if path.is_file()}

    second = case.resume()
    after = {path: path.read_bytes() for path in case.output.rglob("*") if path.is_file()}

    assert second.canonical_bytes() == first.canonical_bytes()
    assert after == before
    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1, "p5": 1})
