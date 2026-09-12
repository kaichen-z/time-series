from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_numerical_supply import build_package_registry
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.budget import ResourceUse
from evolving_loop.v2.contracts import (
    EvolutionV2Config,
    KernelProtocolCommitment,
    canonical_v2_bytes,
)
from evolving_loop.v2.cooperative import (
    DecisionCoordinateAdapter,
    DecisionModuleV2,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
    RetrievalModuleV2,
    run_cooperative_evolution,
)
from evolving_loop.v2.fakes import FakeClock
from evolving_loop.v2.kernel import KernelAuthorityError
from evolving_loop.v2.numerical_qd.adapters import (
    FrozenNumericalArtifactsV2,
    import_numerical_seed,
)
from tests.test_package_numerical_supply import (
    _package_for_task,
    _registry_tasks,
    _supply_release,
)
from tests.test_package_stage_runner import _evaluation


def _pair(*, alternate: bool) -> FrozenNumericalArtifactsV2:
    release = _supply_release() if alternate else _supply_release(alternatives=())
    tasks = _registry_tasks()
    registry = build_package_registry(
        tasks,
        release,
        lambda task, supplied: _package_for_task(task, supplied),
    )
    envelope = import_numerical_seed(release, registry, tasks=tasks).envelope
    return FrozenNumericalArtifactsV2(release, registry, envelope, ())


def _control(scheduler: str) -> EvolutionV2Config:
    protocol = KernelProtocolCommitment(*[str(index) * 64 for index in range(1, 8)])
    return EvolutionV2Config(
        1,
        "smoke",
        23,
        scheduler,
        ("numerical", "retrieval", "decision", "joint"),
        {"cooperative": 4},
        {"resource_levels": [1]},
        {"python": "9" * 64},
        protocol,
        100,
        0.2,
        "production",
    )


@dataclass(frozen=True)
class _Config:
    control: EvolutionV2Config
    max_steps: int = 4
    children_per_step: int = 1
    discount: float = 0.9
    task_cost_weight: float = 0.05
    metric_cap: float = 5.0
    acceptance_tolerance: float = 1e-12
    resource_ceilings: ResourceUse = ResourceUse(
        wall_seconds=100.0,
        task_executions=100,
        artifact_bytes=1_000_000,
    )

    def to_payload(self):
        return {
            "schema_version": 1,
            "control": self.control.to_payload(),
            "max_steps": self.max_steps,
            "children_per_step": self.children_per_step,
            "discount": self.discount,
            "task_cost_weight": self.task_cost_weight,
            "metric_cap": self.metric_cap,
            "acceptance_tolerance": self.acceptance_tolerance,
            "resource_ceilings": self.resource_ceilings.to_payload(),
        }


class _Pipeline:
    def __init__(
        self, alternate_release: str, clock, *, reject_dev: bool = False
    ):
        self.alternate_release = alternate_release
        self.clock = clock
        self.reject_dev = reject_dev
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, bundle, tasks, stage) -> PackageEvaluation:
        self.calls.append((bundle.fingerprint(), stage))
        advance = getattr(self.clock, "advance", None)
        if advance is not None:
            advance(1.0)
        changed_module = (
            bundle.retrieval_release_sha256 != self.seed_retrieval_sha
            or bundle.decision_policy_sha256 != self.seed_decision_sha
        )
        if changed_module:
            error = 1.2
        elif bundle.numerical_release_sha256 == self.alternate_release:
            error = 1.0
        else:
            error = 2.0
        if stage == "dev" and self.reject_dev and error < 2.0:
            error = 3.0
        return _evaluation(
            bundle.fingerprint(),
            tuple(str(task) for task in tasks),
            error,
        )


@pytest.fixture
def run_case(tmp_path):
    seed_pair = _pair(alternate=False)
    alternate_pair = _pair(alternate=True)
    retrieval = RetrievalModuleV2(
        1, "a" * 64, RetrievalGenome.seed().to_payload(), ()
    )
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

    class RunCase:
        pipelines: dict[str, _Pipeline] = {}

        def run(
            self,
            *,
            scheduler: str,
            directory: str = "run",
            resume: bool = False,
            stop_after: int | None = None,
            decision_prompt: str = "changed prompt",
            reject_dev: bool = False,
            real_clock: bool = False,
            resource_reporter=None,
        ):
            clock = time.monotonic if real_clock else FakeClock()
            pipeline = _Pipeline(
                alternate_pair.release.fingerprint, clock, reject_dev=reject_dev
            )
            pipeline.seed_retrieval_sha = retrieval.fingerprint()
            pipeline.seed_decision_sha = decision.fingerprint()
            self.pipelines[directory] = pipeline
            adapters = {
                "numerical": NumericalCoordinateAdapter((alternate_pair,)),
                "retrieval": RetrievalCoordinateAdapter(),
                "decision": DecisionCoordinateAdapter((decision_prompt,)),
                "pipeline": pipeline,
                "monotonic": clock,
            }
            config = _Config(_control(scheduler))
            if resource_reporter is not None:
                adapters["resource_reporter"] = resource_reporter
                config = _Config(
                    _control(scheduler),
                    resource_ceilings=ResourceUse(
                        wall_seconds=100.0,
                        task_executions=100,
                        llm_calls=100,
                        input_tokens=10000,
                        output_tokens=10000,
                        subprocesses=100,
                        artifact_bytes=1_000_000,
                    ),
                )
            return run_cooperative_evolution(
                tmp_path / directory,
                config,
                seed_artifacts,
                tasks,
                adapters,
                resume=resume,
                stop_after=stop_after,
            )

        def progress(self, directory: str = "run"):
            return [
                json.loads(line)
                for line in (tmp_path / directory / "cooperative_progress.jsonl")
                .read_text()
                .splitlines()
            ]

        def read(self, relative: str) -> bytes:
            return (tmp_path / relative).read_bytes()

        def snapshot(self, directory: str):
            root = tmp_path / directory
            return {
                str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in root.rglob("*")
                if path.is_file()
            }

    return RunCase()


def test_runner_executes_four_arms_and_preserves_rejected_parent(run_case):
    result = run_case.run(scheduler="ucb")

    assert result.attempted_arms == (
        "numerical",
        "retrieval",
        "decision",
        "joint",
    )
    assert result.accepted_steps == 1
    assert result.rejected_steps == 3
    rejected = [row for row in run_case.progress() if row["decision"] == "reject"]
    assert len(rejected) == 3
    assert all(row["active_before"] == row["active_after"] for row in rejected)


def test_runner_charges_cumulative_host_llm_usage(run_case):
    def report():
        calls = len(run_case.pipelines["accounted"].calls)
        return ResourceUse(
            llm_calls=calls,
            input_tokens=100 * calls,
            output_tokens=20 * calls,
            subprocesses=calls,
        )

    run_case.run(
        scheduler="ucb",
        directory="accounted",
        stop_after=1,
        resource_reporter=report,
    )

    checkpoint = json.loads(run_case.read("accounted/checkpoint.json"))
    charged = checkpoint["budget"]["charged_use"]
    assert charged["llm_calls"] == 4
    assert charged["input_tokens"] == 400
    assert charged["output_tokens"] == 80
    assert charged["subprocesses"] == 4


def test_runner_uses_train_aggregate_cache_and_opens_dev_only_after_train_gate(
    run_case,
):
    run_case.run(scheduler="ucb", directory="cache")

    calls = run_case.pipelines["cache"].calls
    assert [stage for _, stage in calls].count("train") == 5
    assert [stage for _, stage in calls].count("dev") == 2
    assert len(list(run_case.read("cache/cooperative_progress.jsonl").splitlines())) == 4


@pytest.mark.parametrize("scheduler", ["ucb", "thompson"])
def test_closed_step_resume_matches_uninterrupted_bytes(run_case, scheduler):
    full = run_case.run(scheduler=scheduler, directory=f"full-{scheduler}")
    run_case.run(
        scheduler=scheduler,
        directory=f"resumed-{scheduler}",
        stop_after=2,
    )
    resumed = run_case.run(
        scheduler=scheduler,
        directory=f"resumed-{scheduler}",
        resume=True,
    )

    assert resumed.to_payload() == full.to_payload()
    assert run_case.read(
        f"resumed-{scheduler}/evaluation_complete.json"
    ) == run_case.read(f"full-{scheduler}/evaluation_complete.json")


def test_complete_resume_is_a_read_only_noop(run_case):
    first = run_case.run(scheduler="ucb", directory="complete")
    before = run_case.snapshot("complete")

    second = run_case.run(scheduler="ucb", directory="complete", resume=True)

    assert second == first
    assert run_case.snapshot("complete") == before


def test_resume_rejects_progress_that_disagrees_with_the_closed_checkpoint(
    run_case, tmp_path
):
    run_case.run(scheduler="ucb", directory="damaged", stop_after=2)
    path = tmp_path / "damaged" / "cooperative_progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["candidate_sha256"] = "f" * 64
    path.write_bytes(b"".join(canonical_v2_bytes(row) for row in rows))

    with pytest.raises(KernelAuthorityError, match="progress/checkpoint"):
        run_case.run(scheduler="ucb", directory="damaged", resume=True)


def test_resume_rejects_changed_decision_operator_inputs(run_case):
    run_case.run(scheduler="ucb", directory="changed-input", stop_after=2)

    with pytest.raises(KernelAuthorityError, match="configuration/input"):
        run_case.run(
            scheduler="ucb",
            directory="changed-input",
            resume=True,
            decision_prompt="different remaining proposal",
        )


def test_resume_rejects_rewritten_aggregate_object(run_case, tmp_path):
    run_case.run(scheduler="ucb", directory="rewritten-object", stop_after=2)
    objects = tmp_path / "rewritten-object" / "objects"
    path = next(
        path
        for path in objects.glob("*.json")
        if json.loads(path.read_text()).get("kind") == "cooperative_aggregate"
    )
    payload = json.loads(path.read_text())
    payload["mean_smae"] += 0.125
    path.write_bytes(canonical_v2_bytes(payload))

    with pytest.raises(KernelAuthorityError, match="digest"):
        run_case.run(scheduler="ucb", directory="rewritten-object", resume=True)


def test_dev_rejection_does_not_update_train_only_scheduler_outcomes(
    run_case, tmp_path
):
    result = run_case.run(
        scheduler="ucb", directory="dev-reject", reject_dev=True
    )
    checkpoint = json.loads(
        (tmp_path / "dev-reject" / "cooperative_checkpoint.json").read_text()
    )

    assert result.accepted_steps == 0
    assert result.rejected_steps == 4
    assert [
        arm["acceptances"]
        for arm in checkpoint["scheduler_state"]["arms"].values()
    ] == [1, 1, 1, 1]


def test_real_clock_keeps_margin_for_the_first_stage_reservation(run_case):
    result = run_case.run(
        scheduler="ucb", directory="real-clock", stop_after=1, real_clock=True
    )

    assert result.attempted_arms == ("numerical",)
    assert len(run_case.pipelines["real-clock"].calls) == 4


def test_stop_after_zero_executes_no_scheduled_step(run_case):
    result = run_case.run(scheduler="ucb", directory="stop-zero", stop_after=0)

    assert result.attempted_arms == ()
    assert result.accepted_steps == result.rejected_steps == 0
    assert run_case.pipelines["stop-zero"].calls == []


def test_resume_at_stop_boundary_executes_no_additional_step(run_case):
    partial = run_case.run(
        scheduler="ucb", directory="same-stop", stop_after=2
    )
    before = run_case.snapshot("same-stop")

    resumed = run_case.run(
        scheduler="ucb", directory="same-stop", resume=True, stop_after=2
    )

    assert resumed == partial
    assert run_case.pipelines["same-stop"].calls == []
    assert run_case.snapshot("same-stop") == before
