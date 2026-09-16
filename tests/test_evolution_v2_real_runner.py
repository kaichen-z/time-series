from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.runner import (
    RealStageContextV2,
    RealStagePorts,
    SealedStageV2,
    run_real_evolution,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class SimulatedCrash(RuntimeError):
    pass


class SimulatedSealFailure(RuntimeError):
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
        self.completion_public_flags = {stage: False for stage in self.stage_durations}
        self.omit_completion_public_flag: set[str] = set()
        self.seal_failure_stage: str | None = None
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
        if stage not in self.omit_completion_public_flag:
            payload["public_test_accessed"] = self.completion_public_flags[stage]
        (context.output_dir / "evaluation_complete.json").write_bytes(canonical_v2_bytes(payload))
        return payload

    def _seal(self, stage: str, context, result):
        self.order.append(f"seal_{stage}")
        if self.seal_failure_stage == stage:
            raise SimulatedSealFailure(stage)
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
        "seal_p2", "seal_p3", "seal_p4", "seal_p5",
    ]


def test_explicit_generation_target_expands_the_p2_grant(case: RunnerCase):
    result = run_real_evolution(
        case.output,
        case.manifest,
        case.ports,
        monotonic=case.clock,
        p2_generations=10,
    )

    assert result.status == "complete"
    assert case.grant("p2") == 4200


def test_effective_target_is_bound_to_root_resume_identity(case: RunnerCase):
    from evolving_loop.v2.real.runner import RealRunnerError
    run_real_evolution(case.output, case.manifest, case.ports, monotonic=case.clock,
                       p2_generations=3, p2_min_effective_candidates=1)
    with pytest.raises(RealRunnerError, match="root manifest"):
        run_real_evolution(case.output, case.manifest, case.ports, monotonic=case.clock,
                           p2_generations=3, p2_min_effective_candidates=2)


def test_unused_time_rolls_forward_without_touching_reserve(case: RunnerCase):
    case.stage_durations.update(p2=600, p3=400, p4=100, p5=100)

    result = case.run()

    assert case.grants == {"p2": 840, "p3": 600, "p4": 320, "p5": 340}
    assert result.status == "complete"
    assert case.finalization_started_at is not None
    assert case.finalization_started_at <= 1440


def test_carry_accumulates_across_p2_and_p3(case: RunnerCase):
    case.stage_durations.update(p2=600, p3=100, p4=100, p5=100)

    result = case.run()

    assert result.status == "complete"
    assert case.grants == {"p2": 840, "p3": 600, "p4": 620, "p5": 640}


def test_elapsed_search_time_caps_carried_p5_grant(case: RunnerCase):
    """Uncharged orchestration time must shrink, not deny, the final stage."""
    case.stage_durations.update(p2=600, p3=100, p4=138, p5=491)
    original_seal = case._seal
    p4_overhead_applied = False

    def seal_with_r7_overhead(stage, context, result):
        nonlocal p4_overhead_applied
        sealed = original_seal(stage, context, result)
        if stage == "p4" and not p4_overhead_applied:
            case.clock.advance(110.8)
            p4_overhead_applied = True
        return sealed

    case._seal = seal_with_r7_overhead

    result = case.run()

    assert result.status == "complete"
    assert case.grants == {"p2": 840, "p3": 600, "p4": 620, "p5": 491}
    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1, "p5": 1})
    assert case.finalization_started_at == pytest.approx(1439.8)
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
    case.completion_public_flags["p4"] = True

    result = case.run()

    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert result.status == "failed"
    assert checkpoint["phase"] == "FAILED"
    assert not (case.output / "evaluation_complete.json").exists()
    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1})


@pytest.mark.parametrize(
    "completion_public, sealed_public, omit_completion_public",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_child_completion_public_evidence_must_be_false_and_match_seal(
    case: RunnerCase,
    completion_public: bool,
    sealed_public: bool,
    omit_completion_public: bool,
):
    case.completion_public_flags["p2"] = completion_public
    case.public_flags["p2"] = sealed_public
    if omit_completion_public:
        case.omit_completion_public_flag.add("p2")

    with pytest.raises(ValueError):
        case.run()

    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "FAILED"
    assert checkpoint["active_stage"] is None
    assert checkpoint["stage_records"][0]["status"] == "failed"
    assert case.calls == Counter({"p2": 1})


def test_seal_failure_closes_the_active_stage_as_failed(case: RunnerCase):
    case.seal_failure_stage = "p2"

    with pytest.raises(SimulatedSealFailure):
        case.run()

    checkpoint = json.loads((case.output / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "FAILED"
    assert checkpoint["active_stage"] is None
    assert checkpoint["stage_records"][0]["status"] == "failed"
    assert case.calls == Counter({"p2": 1})


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


def test_completed_resume_reauthenticates_native_stage_seals(case: RunnerCase):
    case.run()
    case.seal_failure_stage = "p3"

    with pytest.raises(SimulatedSealFailure):
        case.resume()

    assert case.calls == Counter({"p2": 1, "p3": 1, "p4": 1, "p5": 1})


def test_stage_context_reports_only_unconsumed_grant(case: RunnerCase):
    context = RealStageContextV2(
        "p2",
        case.output / "p2",
        840,
        case.manifest,
        case.manifest.fingerprint(),
        case.manifest.model.fingerprint(),
        {},
        deadline_monotonic=840.0,
        monotonic=case.clock,
    )

    case.clock.advance(311)

    assert context.remaining_seconds() == 529
    case.clock.advance(530)
    assert context.remaining_seconds() == 0


def test_completed_resume_rejects_a_result_with_tampered_bound_fields(case: RunnerCase):
    case.run()
    completion_path = case.output / "evaluation_complete.json"
    payload = json.loads(completion_path.read_text())
    payload["model_binding_sha256"] = digest("tampered-model")
    completion_path.write_bytes(canonical_v2_bytes(payload))

    with pytest.raises(ValueError, match="completed root result"):
        case.resume()


def test_real_host_never_exposes_raw_p2_numerical_alternatives():
    from tests.test_evolution_v2_real_cooperative import _host, _tasks_100

    host = _host(_tasks_100(), object())

    assert not hasattr(host, "numerical_alternatives")
