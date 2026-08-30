from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import evolving_loop.cli as cli_module
import common.evolution_core.contracts as evolution_contracts
import numerical_agent.run_evolution as numerical_evolution_runner

from evolving_loop.cli import (
    BASELINE_CHOICES,
    EVOLUTION_CHOICES,
    _baseline_argv,
    _factory,
    _source_evolve_command,
    _three_way_entity_split,
    build_parser,
    inference_command,
)
from evolving_loop.co_evolution import HarnessPolicy, snapshot_policy_skills
from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import ContextTask, Document, Task
from evolving_loop.decision_agent.skill_library import DecisionSkill, DecisionSkillLibrary
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkill,
    RetrievalSkillLibrary,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
    write_retrieval_release,
)
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCheckpointError,
    RetrievalEvaluation,
    RetrievalEvolutionError,
    RetrievalForecastingFailure,
    RetrievalEvolutionResult,
    RetrievalGenerationTrace,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from common.llm import FakeLLMClient


def test_active_evolution_defaults_bind_the_joint_scaled_metric_policy() -> None:
    config = evolution_contracts.EvolutionConfig()

    assert config.metric.name == "smae"
    assert config.metric_policy == evolution_contracts.METRIC_POLICY
    assert config.metric_policy_fingerprint == evolution_contracts.METRIC_POLICY_FINGERPRINT


def test_numerical_evolution_run_manifest_rejects_legacy_metric_policy(tmp_path) -> None:
    manifest = numerical_evolution_runner._load_or_create_run_manifest(tmp_path)
    assert manifest["schema_version"] == 2
    assert manifest["metric_policy_fingerprint"] == evolution_contracts.METRIC_POLICY_FINGERPRINT

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric_policy": {
                    "schema_version": 1,
                    "primary": ["mase"],
                    "ordering": "median_mase",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy metric policy"):
        numerical_evolution_runner._load_or_create_run_manifest(tmp_path)


def test_numerical_evolution_refuses_to_label_an_existing_legacy_run_as_active(
    tmp_path,
) -> None:
    (tmp_path / "generation_001_result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing metric policy"):
        numerical_evolution_runner._load_or_create_run_manifest(tmp_path)


def test_evolve_cli_exposes_three_evolution_modes() -> None:
    parser = build_parser()
    for mode in ("prompt", "genome", "source"):
        args = parser.parse_args(["evolve", "--evolution-mode", mode])
        assert args.evolution_mode == mode


def test_genome_remains_the_default_evolution_mode() -> None:
    assert build_parser().parse_args(["evolve"]).evolution_mode == "genome"


def test_retrieval_topology_controls_are_explicit_for_both_interfaces() -> None:
    parser = build_parser()
    for prefix in (["evolve"], ["--evolution", "genome", "--tasks-file", "tasks"]):
        legacy = parser.parse_args(prefix)
        assert legacy.retrieval_mode == "single-pass"
        assert legacy.retrieval_release_path is None
        two_stage = parser.parse_args(
            [*prefix, "--retrieval-mode", "two-stage", "--retrieval-release-path", "release"]
        )
        assert two_stage.retrieval_mode == "two-stage"
        assert two_stage.retrieval_release_path == "release"


def test_coordinate_phase_controls_are_explicit_for_root_and_legacy_genome() -> None:
    """Catches a coordinate selector that exists on only one CLI grammar."""
    parser = build_parser()
    for prefix in (["evolve"], ["--evolution", "genome", "--tasks-file", "tasks"]):
        assert parser.parse_args(prefix).coordinate_phase is None
        for phase in ("retrieval", "decision", "alternate"):
            args = parser.parse_args([*prefix, "--coordinate-phase", phase])
            assert args.coordinate_phase == phase


@pytest.mark.parametrize("phase", ("retrieval", "decision", "alternate"))
def test_coordinate_phases_inherit_frozen_two_stage_retrieval_defaults(
    phase: str,
) -> None:
    """Catches Genome coordinate selection retaining unusable single-pass defaults."""
    args = build_parser().parse_args(
        ["--evolution", "genome", "--coordinate-phase", phase]
    )

    assert args.retrieval_mode == "two-stage"
    assert args.screen_train_tasks == 8
    assert args.screen_promote == 2
    assert args.retrieval_release_path.endswith("/v000")


@pytest.mark.parametrize("phase", ("decision", "alternate"))
def test_coordinate_decision_phases_require_nonseed_accepted_embedded_retrieval_before_tasks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Catches task/component work beginning from v000 or an unembedded accepted release."""
    seed_release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    args = build_parser().parse_args(
        [
            "--evolution",
            "genome",
            "--coordinate-phase",
            phase,
            "--retrieval-mode",
            "two-stage",
            "--retrieval-release-path",
            str(seed_release.path),
            "--tasks-file",
            str(tmp_path / "must-not-load.jsonl"),
        ]
    )
    args.evolution_mode = args.evolution
    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("task loading must occur after the coordinate release gate")
        ),
    )

    with pytest.raises(ValueError, match="accepted|v000|seed"):
        cli_module.evolve_command(args)


@pytest.mark.parametrize("phase", ("decision", "alternate"))
def test_coordinate_decision_phases_reject_unembedded_accepted_release_before_tasks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Catches treating an operator path as a substitute for bundle provenance."""
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        replace(RetrievalGenome.seed(), version="v001", parent="v000"),
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )
    args = build_parser().parse_args(
        [
            "--evolution",
            "genome",
            "--coordinate-phase",
            phase,
            "--retrieval-release-path",
            str(release.path),
            "--tasks-file",
            str(tmp_path / "must-not-load.jsonl"),
        ]
    )
    args.evolution_mode = args.evolution
    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("task loading must occur after embedded release validation")
        ),
    )

    with pytest.raises(ValueError, match="embedded accepted Retrieval release"):
        cli_module.evolve_command(args)


def test_coordinate_authority_environment_is_consumed_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches coordinate model subprocesses inheriting operator authority."""
    args = SimpleNamespace(
        checkpoint_authority_key_env="CUSTOM_COORDINATE_KEY",
        checkpoint_authority_expected_env="CUSTOM_COORDINATE_ANCHOR",
    )
    monkeypatch.setenv("CUSTOM_COORDINATE_KEY", "k" * 64)
    monkeypatch.setenv("CUSTOM_COORDINATE_ANCHOR", "0:" + "a" * 64)
    monkeypatch.setenv("RETRIEVAL_CHECKPOINT_AUTHORITY_KEY", "default-key-secret")
    monkeypatch.setenv(
        "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED", "default-anchor-secret"
    )

    key, anchor, subprocess_environment = (
        cli_module._consume_coordinate_authority_environment(args)
    )

    assert key == ("k" * 64).encode("utf-8")
    assert anchor == (0, "a" * 64)
    for name in (
        "CUSTOM_COORDINATE_KEY",
        "CUSTOM_COORDINATE_ANCHOR",
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY",
        "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED",
    ):
        assert name not in os.environ
        assert name not in subprocess_environment
    assert not {
        "k" * 64,
        "0:" + "a" * 64,
        "default-key-secret",
        "default-anchor-secret",
    }.intersection(subprocess_environment.values())


def test_evolve_cli_exposes_targeted_agent_evolution() -> None:
    parser = build_parser()
    for target in ("auto", "coding", "retrieval", "decision"):
        args = parser.parse_args(["evolve", "--evolve-target", target])
        assert args.evolve_target == target
    assert parser.parse_args(["evolve"]).evolve_target == "auto"


def test_setting2_knowledge_is_explicitly_opt_in_for_both_interfaces() -> None:
    parser = build_parser()

    assert parser.parse_args(["evolve"]).setting2_knowledge is False
    assert parser.parse_args(["evolve", "--setting2-knowledge"]).setting2_knowledge is True
    unified = parser.parse_args(
        ["--evolution", "genome", "--tasks-file", "tasks.jsonl", "--setting2-knowledge"]
    )
    assert unified.setting2_knowledge is True


def test_evolve_cli_exposes_successive_halving_controls() -> None:
    parser = build_parser()
    values = [
        "--successive-halving",
        "--screen-train-tasks",
        "6",
        "--screen-dev-tasks",
        "2",
        "--screen-promote",
        "1",
        "--screen-tolerance",
        "0.01",
    ]
    for prefix in (["evolve"], ["--evolution", "genome", "--tasks-file", "tasks"]):
        args = parser.parse_args([*prefix, *values])
        assert args.successive_halving is True
        assert args.screen_train_tasks == 6
        assert args.screen_dev_tasks == 2
        assert args.screen_promote == 1
        assert args.screen_tolerance == pytest.approx(0.01)


def test_unified_cli_exposes_every_named_method() -> None:
    parser = build_parser()
    for name in BASELINE_CHOICES:
        args = parser.parse_args(["--baseline", name, "--sample-dir", "sample"])
        assert args.baseline == name
    for name in EVOLUTION_CHOICES:
        args = parser.parse_args(["--evolution", name, "--tasks-file", "tasks.jsonl"])
        assert args.evolution == name
        frozen = parser.parse_args(["--inference", name, "--hidden-test"])
        assert frozen.inference == name


def test_retrieval_cli_has_frozen_defaults_in_root_and_legacy_forms() -> None:
    parser = build_parser()
    root = parser.parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            "external/Dr-CiK/sample/tasks.jsonl",
            "--split-manifest",
            "splits/drcik_public_80_20_99_v1.json",
        ]
    )
    legacy = parser.parse_args(
        [
            "evolve",
            "--evolution-mode",
            "retrieval",
            "--tasks-file",
            "external/Dr-CiK/sample/tasks.jsonl",
            "--split-manifest",
            "splits/drcik_public_80_20_99_v1.json",
        ]
    )
    frozen = parser.parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
        ]
    )

    for args in (root, legacy, frozen):
        assert args.retrieval_mode == "two-stage"
        assert args.screen_train_tasks == 8
        assert args.screen_promote == 2
    assert root.evolution == "retrieval"
    assert root.split_manifest_sha256 == (
        "3cc81f45878c1aae93e5ba48dc367df6553698db6661dbe06fbe5efb06afca92"
    )
    assert legacy.evolution_mode == "retrieval"
    assert frozen.inference == "retrieval"


def _manifest_task(task_id: str) -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id=task_id,
            history_values=(1.0, 2.0),
            future_values=(3.0,),
            prediction_length=1,
            frequency="1 day",
            seasonal_period=None,
            entity_name=f"entity_{task_id}",
        ),
        target_name="target",
        target_description="",
        history_timestamps=("1", "2"),
        future_timestamps=("3",),
        documents=(),
        labels_public=True,
    )


def _manifest_record(task_id: str) -> dict[str, object]:
    return {
        "benchmark_id": task_id,
        "labels_public": True,
        "series": {
            "history_values": [1.0, 2.0],
            "future_values": [3.0],
            "history_timestamps": ["1", "2"],
            "future_timestamps": ["3"],
        },
        "task_metadata": {
            "prediction_length": 1,
            "frequency": "1 day",
            "target_description": "public target",
        },
        "showcase": {
            "entity": {"name": f"entity_{task_id}"},
            "time_series_variable": {"name": "target"},
        },
        "documents": [],
    }


def test_retrieval_manifest_loader_selects_exact_train_and_dev_only(
    tmp_path,
) -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "splits"
        / "drcik_public_80_20_99_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest["partitions"]
    train_ids = tuple(partitions["train"]["task_ids"])
    dev_ids = tuple(partitions["dev"]["task_ids"])
    public_ids = tuple(partitions["public_test"]["task_ids"])
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(_manifest_record(task_id)) + "\n"
            for task_id in (*public_ids, *dev_ids, *train_ids)
        ),
        encoding="utf-8",
    )

    train, dev, manifest_sha256 = (
        cli_module._load_retrieval_evolution_tasks(
            dataset,
            manifest_path,
            expected_manifest_sha256=manifest["manifest_sha256"],
        )
    )

    assert tuple(task.numeric.task_id for task in train) == train_ids
    assert tuple(task.numeric.task_id for task in dev) == dev_ids
    assert len(train) == 80
    assert len(dev) == 20
    assert manifest_sha256 == manifest["manifest_sha256"]
    assert not set(public_ids).intersection(
        task.numeric.task_id for task in (*train, *dev)
    )


def test_retrieval_manifest_loader_never_instantiates_public_regression_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest["partitions"]
    all_ids = tuple(
        task_id
        for name in ("public_test", "dev", "train")
        for task_id in partitions[name]["task_ids"]
    )
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "".join(json.dumps(_manifest_record(task_id)) + "\n" for task_id in all_ids),
        encoding="utf-8",
    )
    public_ids = frozenset(partitions["public_test"]["task_ids"])
    selected_ids = frozenset(
        (*partitions["train"]["task_ids"], *partitions["dev"]["task_ids"])
    )
    original_convert = getattr(cli_module, "_to_context_task", None)
    instantiated: list[str] = []

    def track_selected_only(record: dict[str, object]) -> ContextTask:
        task_id = str(record["benchmark_id"])
        if task_id in public_ids:
            raise AssertionError("Public Regression labels were instantiated")
        instantiated.append(task_id)
        assert original_convert is not None
        return original_convert(record)

    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bulk task loading materializes Public Regression labels")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_to_context_task",
        track_selected_only,
        raising=False,
    )

    train, dev, _digest = cli_module._load_retrieval_evolution_tasks(
        dataset,
        manifest_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )

    assert len(train) == 80
    assert len(dev) == 20
    assert frozenset(instantiated) == selected_ids


def test_retrieval_manifest_loader_rejects_incomplete_public_source_without_parsing_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest["partitions"]
    omitted_public = partitions["public_test"]["task_ids"][0]
    ids = tuple(
        task_id
        for name in ("train", "dev", "public_test")
        for task_id in partitions[name]["task_ids"]
        if task_id != omitted_public
    )
    dataset = tmp_path / "incomplete.jsonl"
    dataset.write_text(
        "".join(json.dumps(_manifest_record(task_id)) + "\n" for task_id in ids),
        encoding="utf-8",
    )
    original_convert = getattr(cli_module, "_to_context_task", None)

    def reject_public_conversion(record: dict[str, object]) -> ContextTask:
        assert str(record["benchmark_id"]) not in set(
            partitions["public_test"]["task_ids"]
        )
        assert original_convert is not None
        return original_convert(record)

    monkeypatch.setattr(
        cli_module,
        "_to_context_task",
        reject_public_conversion,
        raising=False,
    )

    with pytest.raises(ValueError, match="incomplete|missing|dataset"):
        cli_module._load_retrieval_evolution_tasks(
            dataset,
            manifest_path,
            expected_manifest_sha256=manifest["manifest_sha256"],
        )


def test_retrieval_manifest_loader_rejects_nested_public_id_spoof(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest["partitions"]
    public_ids = set(partitions["public_test"]["task_ids"])
    spoofed_id = partitions["public_test"]["task_ids"][0]
    records = []
    for name in ("train", "dev", "public_test"):
        for task_id in partitions[name]["task_ids"]:
            record = _manifest_record(task_id)
            if task_id == spoofed_id:
                del record["benchmark_id"]
                record["nested_metadata"] = {"benchmark_id": task_id}
            records.append(record)
    dataset = tmp_path / "spoofed.jsonl"
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    original_convert = cli_module._to_context_task

    def selected_only(record):
        assert record.get("benchmark_id") not in public_ids
        return original_convert(record)

    monkeypatch.setattr(cli_module, "_to_context_task", selected_only)

    with pytest.raises(ValueError, match="metadata|dataset|record|incomplete"):
        cli_module._load_retrieval_evolution_tasks(
            dataset,
            manifest_path,
            expected_manifest_sha256=manifest["manifest_sha256"],
        )


@pytest.mark.parametrize("tamper", ("manifest", "expected_hash"))
def test_retrieval_manifest_loader_rejects_every_hash_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "splits"
        / "drcik_public_80_20_99_v1.json"
    )
    manifest = json.loads(source.read_text(encoding="utf-8"))
    expected = manifest["manifest_sha256"]
    if tamper == "manifest":
        manifest["partitions"]["train"]["task_ids"][0] = "forged_task"
    else:
        expected = "f" * 64
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("tasks must not load before manifest authentication")
        ),
    )

    with pytest.raises(ValueError, match="manifest|hash|sha256"):
        cli_module._load_retrieval_evolution_tasks(
            "ignored.jsonl",
            path,
            expected_manifest_sha256=expected,
        )


def test_retrieval_manifest_loader_rejects_a_self_consistent_nonfrozen_split(
    tmp_path,
) -> None:
    source = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v1.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    train_ids = manifest["partitions"]["train"]["task_ids"]
    public_ids = manifest["partitions"]["public_test"]["task_ids"]
    train_ids[0], public_ids[0] = public_ids[0], train_ids[0]
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    forged_hash = cli_module._canonical_sha256(unsigned)
    manifest["manifest_sha256"] = forged_hash
    manifest_path = tmp_path / "forged-split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen|hash|manifest"):
        cli_module._load_retrieval_evolution_tasks(
            tmp_path / "tasks.jsonl",
            manifest_path,
            expected_manifest_sha256=forged_hash,
        )


def test_retrieval_evolution_dispatch_bypasses_generic_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            "tasks.jsonl",
            "--split-manifest",
            "split.json",
        ]
    )
    args.evolution_mode = args.evolution
    captured = []

    def retrieval_dispatch(received):
        captured.append(received)
        return {"evolution_mode": "retrieval"}

    monkeypatch.setattr(
        cli_module,
        "_retrieval_evolve_command",
        retrieval_dispatch,
        raising=False,
    )
    monkeypatch.setattr(
        cli_module,
        "_three_way_entity_split",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic splitter must not run")
        ),
    )

    assert cli_module.evolve_command(args) == {"evolution_mode": "retrieval"}
    assert captured == [args]


def test_coordinate_retrieval_dispatch_preserves_the_task8_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a second, weaker Retrieval path under the Genome selector."""
    args = build_parser().parse_args(
        ["--evolution", "genome", "--coordinate-phase", "retrieval"]
    )
    args.evolution_mode = args.evolution
    captured = []

    def retrieval_dispatch(received):
        captured.append(received)
        return {"accepted": False, "evolution_mode": "retrieval"}

    monkeypatch.setattr(cli_module, "_retrieval_evolve_command", retrieval_dispatch)
    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate Retrieval must use the frozen Task 8 loader")
        ),
    )

    result = cli_module.evolve_command(args)

    assert result == {
        "accepted": False,
        "evolution_mode": "genome",
        "coordinate_phase": "retrieval",
    }
    assert captured == [args]


@pytest.mark.parametrize("mode", ("prompt", "source", "retrieval"))
def test_coordinate_retrieval_is_rejected_outside_genome_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Catches the Retrieval coordinate bypassing its Genome-only mode gate."""
    args = build_parser().parse_args(
        ["--evolution", mode, "--coordinate-phase", "retrieval"]
    )
    args.evolution_mode = args.evolution
    monkeypatch.setattr(
        cli_module,
        "_retrieval_evolve_command",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid coordinate mode must fail before dispatch")
        ),
    )

    with pytest.raises(ValueError, match="only for Genome evolution"):
        cli_module.evolve_command(args)


@pytest.mark.parametrize("phase", ("decision", "alternate"))
def test_coordinate_dispatch_uses_authenticated_preflight_and_scoped_runner(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Catches a coordinate phase falling through to generic three-way evolution."""
    args = build_parser().parse_args(
        ["--evolution", "genome", "--coordinate-phase", phase]
    )
    args.evolution_mode = args.evolution
    release = SimpleNamespace(marker="trusted-release")
    policy = HarnessPolicy()
    calls = []
    monkeypatch.setattr(
        cli_module,
        "_coordinate_retrieval_preflight",
        lambda received: (release, policy),
    )
    helper_name = f"_{phase}_coordinate_evolve_command"

    def run(received, *, release: object, seed_policy: HarnessPolicy):
        calls.append((received, release, seed_policy))
        return {"coordinate_phase": phase}

    monkeypatch.setattr(cli_module, helper_name, run)
    monkeypatch.setattr(
        cli_module,
        "load_context_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate phase must not use the generic splitter")
        ),
    )

    assert cli_module.evolve_command(args) == {"coordinate_phase": phase}
    assert calls == [(args, release, policy)]


def test_harness_policy_embeds_canonical_release_payload_without_a_path(
    tmp_path,
) -> None:
    genome = replace(RetrievalGenome.seed(), version="v001", parent="v000")
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        genome,
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )

    policy = cli_module._policy_with_retrieval_release(
        HarnessPolicy(),
        release,
        changelog="Accepted Retrieval v001.",
    )
    payload = policy.to_payload()
    embedded = payload["retrieval_release_payload"]
    expected_sha256 = hashlib.sha256(
        json.dumps(
            embedded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert set(embedded) == {
        "genome",
        "round1_prompt",
        "round2_prompt",
        "skills",
        "manifest",
    }
    assert "path" not in embedded
    assert payload["retrieval_release_sha256"] == expected_sha256
    destination = tmp_path / "policy.json"
    policy.save(destination)
    assert HarnessPolicy.load(destination) == policy

    mutated_in_memory = HarnessPolicy.load(destination)
    with pytest.raises(TypeError):
        mutated_in_memory.retrieval_release_payload["genome"]["round1_strategy"] = (
            "entity_first"
        )

    tampered = json.loads(destination.read_text(encoding="utf-8"))
    tampered["retrieval_release_payload"]["genome"]["round1_strategy"] = (
        "entity_first"
    )
    destination.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrieval release|retrieval release|fingerprint"):
        HarnessPolicy.load(destination)


def test_retrieval_cli_exposes_every_frozen_hash_control() -> None:
    hashes = [f"{index:x}" * 64 for index in range(1, 6)]
    args = build_parser().parse_args(
        [
            "--evolution",
            "retrieval",
            "--split-manifest",
            "split.json",
            "--split-manifest-sha256",
            hashes[0],
            "--verifier-sha256",
            hashes[1],
            "--evaluator-sha256",
            hashes[2],
            "--metric-sha256",
            hashes[3],
            "--harness-sha256",
            hashes[4],
            "--metric-cap",
            "5",
            "--evolution-tolerance",
            "0.0001",
        ]
    )
    assert args.split_manifest_sha256 == hashes[0]
    assert args.verifier_sha256 == hashes[1]
    assert args.evaluator_sha256 == hashes[2]
    assert args.metric_sha256 == hashes[3]
    assert args.harness_sha256 == hashes[4]
    assert args.metric_cap == 5.0
    assert args.evolution_tolerance == pytest.approx(0.0001)


def test_trusted_retrieval_evaluator_sanitizes_before_inference_and_scores_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = replace(
        _manifest_task("train_1"),
        documents=(Document("doc_1", "public", "SECRET_ROLE", "SECRET_SUBTYPE"),),
        gt_evidence=("SECRET_GT",),
    )
    observed: dict[str, object] = {}

    class Harness:
        def run(self, received, *, allow_skill_writes=True):
            observed["inference_task"] = received
            observed["allow_skill_writes"] = allow_skill_writes
            return SimpleNamespace(task_id=received.numeric.task_id)

    def factory(genome, library):
        observed["genome"] = genome
        observed["library"] = library
        return Harness()

    diagnostics = SimpleNamespace(
        supporting_recall=0.8,
        distractor_avoidance=0.9,
        exact_quote_validity=1.0,
        complete_chain_rate=0.7,
        invalid_count=0,
        catastrophic_count=0,
    )

    def score(original, result):
        observed["scoring_task"] = original
        observed["scoring_result"] = result
        return SimpleNamespace(
            final_smae=0.4,
            final_srmse=0.5,
            contextual_oracle_smae=0.3,
            contextual_oracle_srmse=0.35,
            retrieval_diagnostics=diagnostics,
        )

    monkeypatch.setattr(cli_module, "score_after_resolution", score, raising=False)
    cache_key = SimpleNamespace(task_id="train_1")
    library = object()
    result = cli_module._TrustedRetrievalEvaluator().evaluate(
        RetrievalGenome.seed(),
        (task,),
        stage="screen_train_parent",
        skill_library=library,
        harness_factory=factory,
        persist=False,
        writers_enabled=False,
        evolver_enabled=False,
        cache_keys=(cache_key,),
        metric_cap=5.0,
    )

    inference_task = observed["inference_task"]
    assert inference_task.numeric.future_values == ()
    assert inference_task.gt_evidence == ()
    assert inference_task.labels_public is False
    assert inference_task.documents[0].role is None
    assert inference_task.documents[0].subtype is None
    assert observed["allow_skill_writes"] is False
    assert observed["scoring_task"] is task
    assert result.task_count == 1
    assert result.mean_final_smae == pytest.approx(0.4)
    assert result.task_traces == (
        {
            "task_id": "train_1",
            "entity_name": "entity_train_1",
            "final_smae": 0.4,
            "final_srmse": 0.5,
            "contextual_oracle_smae": 0.3,
            "contextual_oracle_srmse": 0.35,
        },
    )


def test_trusted_retrieval_evaluator_never_serializes_scorer_exception_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _manifest_task("train_1")

    class Harness:
        def run(self, received, *, allow_skill_writes=True):
            assert allow_skill_writes is False
            return SimpleNamespace(task_id=received.numeric.task_id)

    monkeypatch.setattr(
        cli_module,
        "score_after_resolution",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("SECRET_LABEL_VALUE_FROM_TRUSTED_SCORER")
        ),
    )

    with pytest.raises(RetrievalForecastingFailure) as captured:
        cli_module._TrustedRetrievalEvaluator().evaluate(
            RetrievalGenome.seed(),
            (task,),
            stage="screen_train_parent",
            skill_library=object(),
            harness_factory=lambda *_args: Harness(),
            persist=False,
            writers_enabled=False,
            evolver_enabled=False,
            cache_keys=(SimpleNamespace(task_id="train_1"),),
            metric_cap=5.0,
        )

    failure = captured.value
    encoded = repr((str(failure), repr(failure), failure.args))
    assert "SECRET_LABEL_VALUE_FROM_TRUSTED_SCORER" not in encoded
    assert failure.error_type == "TrustedScoringFailure"
    assert failure.__cause__ is None


def _retrieval_evaluation(
    version: str, tasks: tuple[ContextTask, ...], error: float
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        version=version,
        task_count=len(tasks),
        mean_final_smae=error,
        mean_final_srmse=error,
        mean_contextual_oracle_smae=error,
        mean_contextual_oracle_srmse=error,
        p90_smae=error,
        p95_smae=error,
        supporting_recall=0.9,
        distractor_avoidance=0.9,
        exact_quote_validity=1.0,
        complete_chain_rate=0.8,
        invalid_count=0,
        catastrophic_count=0,
        task_traces=tuple(
            {
                "task_id": task.numeric.task_id,
                "entity_name": task.numeric.entity_name,
                "final_smae": error,
                "final_srmse": error,
                "contextual_oracle_smae": error,
                "contextual_oracle_srmse": error,
            }
            for task in tasks
        ),
    )


def _retrieval_result(
    parent: RetrievalGenome,
    train: tuple[ContextTask, ...],
    dev: tuple[ContextTask, ...],
    *,
    accepted: bool,
) -> RetrievalEvolutionResult:
    children = (
        replace(
            parent,
            version="v001",
            parent="v000",
            round1_prompt=parent.round1_prompt + "\nChild A.",
        ),
        replace(
            parent,
            version="v002",
            parent="v000",
            max_evidence_chains=parent.max_evidence_chains + 1,
        ),
        replace(
            parent,
            version="v003",
            parent="v000",
            round2_prompt=parent.round2_prompt + "\nChild C.",
        ),
    )
    winner = children[0]
    generation = RetrievalGenerationTrace(
        generation=0,
        parent_version="v000",
        parent_fingerprint=parent.fingerprint(),
        child_versions=tuple(child.version for child in children),
        child_fingerprints=tuple(child.fingerprint() for child in children),
        child_scopes=("A", "B", "C"),
        child_proposals=tuple(child.to_payload() for child in children),
        screen_task_ids=tuple(task.numeric.task_id for task in train[:8]),
        fold_entities=(tuple(task.numeric.entity_name for task in train[8:]),),
        promoted_fingerprints=(winner.fingerprint(),),
        train_winner_version=winner.version,
        train_winner_fingerprint=winner.fingerprint(),
        rejection_reasons={children[1].fingerprint(): "dominated"},
        screen_summaries={},
        train_summaries={winner.fingerprint(): {"task_count": 80, "mean_final_smae": 0.9}},
    )
    parent_dev = _retrieval_evaluation(parent.version, dev, 1.0)
    child_dev = _retrieval_evaluation(winner.version, dev, 0.9 if accepted else 1.1)
    return RetrievalEvolutionResult(
        original_parent=parent,
        train_winner=winner,
        selected_genome=winner if accepted else parent,
        accepted=accepted,
        acceptance_reasons=("all_dev_gates_passed",) if accepted else (),
        rejection_reasons=() if accepted else ("mean_final_smae",),
        parent_dev=parent_dev,
        child_dev=child_dev,
        generations=(generation,),
        trace=(
            {"kind": "generation_started", "generation": 0},
            {"kind": "release_accepted" if accepted else "release_rejected"},
        ),
        release_genome=winner if accepted else None,
    )


def test_coordinate_retrieval_revalidates_complete_dev_provenance_before_publication() -> None:
    """Catches alternate mode publishing an accepted result without Child Dev."""
    parent = RetrievalGenome.seed()
    train = tuple(_manifest_task(f"train_{index:03d}") for index in range(80))
    dev = tuple(_manifest_task(f"dev_{index:03d}") for index in range(20))
    incomplete = replace(
        _retrieval_result(parent, train, dev, accepted=True),
        child_dev=None,
    )

    with pytest.raises(RetrievalEvolutionError, match="incomplete trusted provenance"):
        cli_module._validate_coordinate_retrieval_result_before_publication(
            incomplete,
            parent=parent,
            train_tasks=train,
            dev_tasks=dev,
            public_ids=frozenset({"public_forbidden"}),
        )


def _install_fake_retrieval_engine(
    monkeypatch: pytest.MonkeyPatch,
    result: RetrievalEvolutionResult,
    captured: dict[str, object],
) -> None:
    class Engine:
        def __init__(self, mutation_llm, evaluator, config, **kwargs):
            captured["mutation_llm"] = mutation_llm
            captured["evaluator"] = evaluator
            captured["config"] = config
            captured["engine_kwargs"] = kwargs
            self.config = config
            self._checkpoint_file_sha256 = None
            self._checkpoint_authority_epoch = None

        def evolve(self, parent, train, dev):
            captured["parent"] = parent
            captured["train"] = train
            captured["dev"] = dev
            checkpoint = Path(self.config.checkpoint_path)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_file_sha256 = hashlib.sha256(b"trusted-checkpoint").hexdigest()
            authority = captured["engine_kwargs"]["_checkpoint_authority"]
            staged = checkpoint.with_name("checkpoint.json.fake-staged")
            staged.write_bytes(b"trusted-checkpoint")
            metadata = staged.stat()
            token = authority.prepare(
                self._checkpoint_file_sha256,
                checkpoint_identity=(metadata.st_dev, metadata.st_ino),
            )
            staged.rename(checkpoint)
            self._checkpoint_authority_epoch = authority.commit(token)
            validator = captured["engine_kwargs"]["_checkpoint_payload_validator"]
            validator(result.to_payload())
            return result

    monkeypatch.setattr(cli_module, "RetrievalEvolutionEngine", Engine, raising=False)


@pytest.mark.parametrize("accepted", (True, False))
def test_retrieval_evolution_publishes_only_accepted_release_and_keeps_traces_in_runs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
) -> None:
    releases = tmp_path / "releases"
    parent_release = write_retrieval_release(releases, RetrievalGenome.seed())
    train = tuple(_manifest_task(f"train_{index:03d}") for index in range(80))
    dev = tuple(_manifest_task(f"dev_{index:03d}") for index in range(20))
    result = _retrieval_result(parent_release.genome, train, dev, accepted=accepted)
    captured: dict[str, object] = {}
    _install_fake_retrieval_engine(monkeypatch, result, captured)
    manifest_path = Path(__file__).parents[1] / "splits" / "drcik_public_80_20_99_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (
            train,
            dev,
            manifest["manifest_sha256"],
            frozenset(manifest["partitions"]["public_test"]["task_ids"]),
        ),
    )
    def sanitized_components(*_args, **kwargs):
        assert "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY" not in os.environ
        subprocess_env = kwargs["llm_subprocess_env"]
        assert "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY" not in subprocess_env
        captured["llm_subprocess_env"] = subprocess_env
        return (
            object(),
            SimpleNamespace(all=lambda: ()),
            kwargs.get("retrieval_library_override"),
            SimpleNamespace(all=lambda: ()),
            None,
        )

    monkeypatch.setattr(cli_module, "_components", sanitized_components)
    monkeypatch.setattr(
        cli_module,
        "_factory",
        lambda *_args, **_kwargs: (lambda _policy: object()),
    )
    runs = tmp_path / "runs"
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "checkpoint.json"
    authority_head_path = authority_directory / "checkpoint.head.json"
    authority_anchor_path = authority_directory / "checkpoint.anchors"
    authority_anchor_path.mkdir(mode=0o700)
    # A crash before the bootstrap no-replace rename may leave only this
    # private staged file; it is not a durable operator anchor or a resume.
    (authority_anchor_path / ".anchor-ledger.interrupted.tmp").write_bytes(
        b"incomplete bootstrap staging\n"
    )
    operator_key = "task-8-cli-operator-authority-key-32-bytes"
    monkeypatch.setenv("RETRIEVAL_CHECKPOINT_AUTHORITY_KEY", operator_key)
    policy_path = runs / "best_policy.json"
    if not accepted:
        policy_path.parent.mkdir(parents=True)
        policy_path.write_text("parent-policy-must-remain", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            "tasks.jsonl",
            "--split-manifest",
            str(manifest_path),
            "--split-manifest-sha256",
            manifest["manifest_sha256"],
            "--retrieval-release-path",
            str(parent_release.path),
            "--checkpoint-path",
            str(runs / "checkpoint.json"),
            "--progress-path",
            str(runs / "progress.jsonl"),
            "--trace-path",
            str(runs / "evolution_trace.json"),
            "--policy-path",
            str(policy_path),
            "--run-root",
            str(runs),
            "--checkpoint-authority-path",
            str(authority_path),
            "--checkpoint-authority-head-path",
            str(authority_head_path),
            "--checkpoint-authority-anchor-path",
            str(authority_anchor_path),
        ]
    )
    args.evolution_mode = "retrieval"

    output = cli_module._retrieval_evolve_command(args)

    assert len(captured["train"]) == 80
    assert len(captured["dev"]) == 20
    config = captured["config"]
    assert config.screen_tasks == 8
    assert config.promote == 2
    assert config.dataset_split_hash == manifest["manifest_sha256"]
    assert all(
        len(getattr(config, field)) == 64
        for field in (
            "verifier_hash",
            "evaluator_hash",
            "metric_hash",
            "mutation_model_hash",
            "harness_hash",
        )
    )
    authority = json.loads(
        authority_path.read_text(encoding="utf-8")
    )
    assert authority["checkpoint_sha256"] == hashlib.sha256(
        b"trusted-checkpoint"
    ).hexdigest()
    assert authority["authority_epoch"] == 1
    assert authority["pending"] is None
    assert authority["schema_version"] == 5
    assert authority_head_path.is_file()
    authority_anchor_path = Path(output["checkpoint_authority_anchor_path"])
    assert authority_anchor_path.is_dir()
    assert len(tuple(authority_anchor_path.glob("anchor-*.json"))) == 1
    persisted_authority = (
        authority_path.read_text(encoding="utf-8")
        + authority_head_path.read_text(encoding="utf-8")
    )
    assert operator_key not in persisted_authority
    assert operator_key not in json.dumps(output)
    assert operator_key not in captured["llm_subprocess_env"].values()
    assert output["checkpoint_authority_anchor"] == {
        "epoch": authority["authority_epoch"],
        "head": authority["authority_head"],
    }
    trace = json.loads((runs / "evolution_trace.json").read_text(encoding="utf-8"))
    public_ids = set(manifest["partitions"]["public_test"]["task_ids"])
    encoded_trace = json.dumps(trace)
    assert all(f'"{task_id}"' not in encoded_trace for task_id in public_ids)
    assert [item["scope"] for item in trace["scope_changelogs"][0]["children"]] == [
        "A",
        "B",
        "C",
    ]
    assert trace["rejection_reasons"] == list(result.rejection_reasons)

    if accepted:
        release = releases / "v001"
        assert release.is_dir()
        assert output["release_path"] == str(release)
        saved = HarnessPolicy.load(policy_path)
        assert saved.retrieval_release_payload["genome"]["version"] == "v001"
        assert saved.retrieval_release_sha256 == output["release_sha256"]
    else:
        assert sorted(path.name for path in releases.iterdir()) == ["v000"]
        assert policy_path.read_text(encoding="utf-8") == "parent-policy-must-remain"
        assert output["release_path"] == str(parent_release.path)
        assert output["policy_path"] is None


def test_retrieval_inference_uses_private_release_binding_and_frozen_runner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = write_retrieval_release(tmp_path / "releases", RetrievalGenome.seed())
    policy = cli_module._policy_with_retrieval_release(
        HarnessPolicy(), release, changelog="Seed Retrieval release."
    )
    policy_path = tmp_path / "policy.json"
    policy.save(policy_path)
    hidden = replace(
        _manifest_task("hidden_1"),
        numeric=replace(_manifest_task("hidden_1").numeric, future_values=()),
        labels_public=False,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli_module, "_inference_tasks", lambda _args: ([hidden], "hidden_test"))
    monkeypatch.setattr(
        cli_module,
        "_components",
        lambda *_args, **kwargs: (
            object(),
            object(),
            kwargs["retrieval_library_override"],
            object(),
            None,
        ),
    )

    def factory(*_args, **kwargs):
        observed["factory_kwargs"] = kwargs
        return lambda _policy: object()

    def frozen(policy_arg, tasks, harness_factory, **kwargs):
        observed["policy"] = policy_arg
        observed["tasks"] = tasks
        observed["harness_factory"] = harness_factory
        observed["frozen_kwargs"] = kwargs
        return {"labels_accessed": False}

    monkeypatch.setattr(cli_module, "_factory", factory)
    monkeypatch.setattr(cli_module, "run_frozen_inference", frozen)
    args = build_parser().parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
            "--policy-path",
            str(policy_path),
            "--retrieval-release-path",
            str(release.path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert inference_command(args) == {"labels_accessed": False}
    assert observed["policy"] == policy
    assert observed["tasks"] == [hidden]
    assert observed["factory_kwargs"]["retrieval_genome"] == release.genome
    assert observed["factory_kwargs"]["retrieval_skill_source"].persist is False
    assert observed["frozen_kwargs"]["artifact_kind"] == "retrieval"
    assert observed["frozen_kwargs"]["score_public"] is False


def test_retrieval_inference_rejects_policy_release_mismatch_before_running(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = tmp_path / "releases"
    seed = write_retrieval_release(releases, RetrievalGenome.seed())
    candidate = write_retrieval_release(
        releases,
        replace(RetrievalGenome.seed(), version="v001", parent="v000"),
    )
    policy_path = tmp_path / "policy.json"
    cli_module._policy_with_retrieval_release(
        HarnessPolicy(), seed, changelog="Seed."
    ).save(policy_path)
    monkeypatch.setattr(
        cli_module,
        "_inference_tasks",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("mismatched policy must fail before loading inference tasks")
        ),
    )
    args = build_parser().parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
            "--policy-path",
            str(policy_path),
            "--retrieval-release-path",
            str(candidate.path),
        ]
    )

    with pytest.raises(ValueError, match="policy|Policy|release"):
        inference_command(args)


def test_frozen_retrieval_rejects_matching_candidate_release_state(
    tmp_path,
) -> None:
    candidate = write_retrieval_release(
        tmp_path / "releases",
        replace(RetrievalGenome.seed(), version="v001", parent="v000"),
    )
    with pytest.raises(ValueError, match="accepted|seed|candidate|state"):
        cli_module._policy_with_retrieval_release(
            HarnessPolicy(), candidate, changelog="Candidate only."
        )


@pytest.mark.parametrize(
    ("backend", "setting"),
    (("qwen", "statistics"), ("codex", "tsfm")),
)
def test_frozen_retrieval_rejects_cache_writing_runtime_before_task_loading(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    setting: str,
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    policy_path = tmp_path / "policy.json"
    cli_module._policy_with_retrieval_release(
        HarnessPolicy(), release, changelog="Seed."
    ).save(policy_path)
    monkeypatch.setattr(
        cli_module,
        "_inference_tasks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unsafe runtime must fail before task loading")
        ),
    )
    args = build_parser().parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
            "--policy-path",
            str(policy_path),
            "--retrieval-release-path",
            str(release.path),
            "--llm-backend",
            backend,
            "--setting",
            setting,
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    with pytest.raises(ValueError, match="frozen|cache|local|write|backend"):
        inference_command(args)


def test_frozen_retrieval_output_cannot_target_release_artifacts_before_task_loading(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    policy_path = tmp_path / "policy.json"
    cli_module._policy_with_retrieval_release(
        HarnessPolicy(), release, changelog="Seed."
    ).save(policy_path)
    monkeypatch.setattr(
        cli_module,
        "_inference_tasks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("colliding output must fail before task loading")
        ),
    )
    args = build_parser().parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
            "--policy-path",
            str(policy_path),
            "--retrieval-release-path",
            str(release.path),
            "--output-dir",
            str(release.path),
        ]
    )

    with pytest.raises(ValueError, match="output|release|collision|disjoint"):
        inference_command(args)


@pytest.mark.parametrize("protected_kind", ("head", "anchor"))
def test_frozen_retrieval_output_cannot_target_complete_checkpoint_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    policy_path = tmp_path / "policy.json"
    cli_module._policy_with_retrieval_release(
        HarnessPolicy(), release, changelog="Seed."
    ).save(policy_path)
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "checkpoint.json"
    authority_head_path = authority_directory / "checkpoint.head.json"
    authority_anchor_path = authority_directory / "checkpoint.anchors"
    authority_head_path.write_bytes(b"protected authority head\n")
    authority_anchor_path.mkdir(mode=0o700)
    output = (
        authority_head_path
        if protected_kind == "head"
        else authority_anchor_path
    )
    monkeypatch.setattr(
        cli_module,
        "_inference_tasks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("authority collision must fail before task loading")
        ),
    )
    args = build_parser().parse_args(
        [
            "--inference",
            "retrieval",
            "--hidden-test",
            "--policy-path",
            str(policy_path),
            "--retrieval-release-path",
            str(release.path),
            "--checkpoint-authority-path",
            str(authority_path),
            "--checkpoint-authority-head-path",
            str(authority_head_path),
            "--checkpoint-authority-anchor-path",
            str(authority_anchor_path),
            "--output-dir",
            str(output),
        ]
    )

    with pytest.raises(
        ValueError,
        match="output|authority|head|anchor|ledger|protected|collision|disjoint",
    ):
        inference_command(args)


@pytest.mark.parametrize(
    "protected_member",
    ("bootstrap", "anchor_record", "release_manifest"),
)
def test_frozen_retrieval_output_member_cannot_alias_authority_or_release(
    tmp_path: Path,
    protected_member: str,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    output = tmp_path / "frozen-output"
    output.mkdir()
    args.output_dir = str(output)
    args.output_root = str(output)
    authority_anchor = Path(args.checkpoint_authority_anchor_path)
    authority_anchor.mkdir(parents=True, mode=0o700)
    if protected_member == "release_manifest":
        protected = release.path / "manifest.json"
    else:
        protected = authority_anchor / (
            "bootstrap.json"
            if protected_member == "bootstrap"
            else "anchor-00000000000000000001-" + "a" * 64 + ".json"
        )
        protected.write_bytes(b"protected operator anchor member\n")
    output_member = output / "forecasts.jsonl"
    os.link(protected, output_member)
    original = protected.read_bytes()

    with pytest.raises(
        ValueError,
        match="alias|inode|identity|hard|authority|anchor|release|protected",
    ):
        cli_module._validate_frozen_retrieval_paths(args, release)

    assert protected.read_bytes() == original
    assert output_member.read_bytes() == original


def test_v000_operator_release_can_bind_an_old_unembedded_policy(tmp_path) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )

    bound = cli_module._policy_for_retrieval_release(
        HarnessPolicy(), release
    )

    assert bound.retrieval_release_payload["genome"]["version"] == "v000"
    assert bound.retrieval_release_sha256 is not None


def test_accepted_release_publication_is_idempotent_for_checkpoint_resume(
    tmp_path,
) -> None:
    genome = replace(RetrievalGenome.seed(), version="v001", parent="v000")
    audit = {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all gates passed",
    }

    releases = tmp_path / "releases"
    seed = write_retrieval_release(releases, RetrievalGenome.seed())
    first = cli_module._publish_or_resume_accepted_retrieval_release(
        releases, genome, skills=(), audit=audit, parent_release=seed
    )
    second = cli_module._publish_or_resume_accepted_retrieval_release(
        releases, genome, skills=(), audit=audit, parent_release=seed
    )

    assert first.path == second.path
    assert first.manifest == second.manifest
    assert tuple(sorted(path.name for path in releases.iterdir())) == ("v000", "v001")


def test_accepted_internal_winner_is_rebased_to_next_contiguous_release(
    tmp_path,
) -> None:
    releases = tmp_path / "releases"
    parent = write_retrieval_release(releases, RetrievalGenome.seed())
    audit = {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all gates passed",
    }
    for index in range(1, 7):
        parent = _write_accepted_retrieval_release(
            releases,
            replace(
                parent.genome,
                version=f"v{index:03d}",
                parent=parent.genome.version,
            ),
            audit=audit,
        )
    internal_winner = replace(
        parent.genome,
        version="v009",
        parent="v006",
        round1_prompt=parent.genome.round1_prompt + "\nAccepted internal winner.",
    )

    published = cli_module._publish_or_resume_accepted_retrieval_release(
        releases,
        internal_winner,
        skills=(),
        audit=audit,
    )

    assert published.path.name == "v007"
    assert published.genome.version == "v007"
    assert published.genome.parent == "v006"
    assert not (releases / "v008").exists()
    assert not (releases / "v009").exists()


def test_published_result_records_the_authoritative_rebased_release(tmp_path) -> None:
    releases = tmp_path / "releases"
    parent = write_retrieval_release(releases, RetrievalGenome.seed())
    internal_winner = replace(
        parent.genome,
        version="v009",
        parent="v000",
        round1_prompt=parent.genome.round1_prompt + "\nAccepted internal winner.",
    )
    result = replace(
        _retrieval_result(
            parent.genome,
            (_manifest_task("train_000"),),
            (_manifest_task("dev_000"),),
            accepted=True,
        ),
        train_winner=internal_winner,
        selected_genome=internal_winner,
        release_genome=internal_winner,
        trace=(
            {
                "kind": "release_accepted",
                "genome": "v009",
                "publication_deferred": True,
            },
        ),
    )
    audit = {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all gates passed",
    }
    published = cli_module._publish_or_resume_accepted_retrieval_release(
        releases,
        internal_winner,
        skills=(),
        audit=audit,
        parent_release=parent,
    )

    traced = cli_module._published_retrieval_result(result, published)

    assert traced.train_winner.version == "v009"
    assert traced.selected_genome == published.genome
    assert traced.release_genome == published.genome
    assert traced.release_published is True
    assert traced.trace[-1] == {
        "kind": "release_accepted",
        "genome": "v001",
        "publication_deferred": False,
    }


def test_accepted_winner_rejects_replaced_parent_skill_lineage(tmp_path) -> None:
    releases = tmp_path / "releases"
    parent = write_retrieval_release(releases, RetrievalGenome.seed())
    audit = {
        "state": "accepted",
        "train_dev_split_sha256": "1" * 64,
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "metric_sha256": "4" * 64,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "all gates passed",
    }
    for index in range(1, 7):
        parent = _write_accepted_retrieval_release(
            releases,
            replace(
                parent.genome,
                version=f"v{index:03d}",
                parent=parent.genome.version,
            ),
            skills=({"lineage": "original"},),
            audit=audit,
        )
    replacement = _write_accepted_retrieval_release(
        tmp_path / "replacement",
        parent.genome,
        skills=({"lineage": "foreign replacement"},),
        audit=audit,
    )
    (releases / "v006").rename(tmp_path / "displaced-v006")
    replacement.path.rename(releases / "v006")
    internal_winner = replace(
        parent.genome,
        version="v009",
        parent="v006",
        round1_prompt=parent.genome.round1_prompt + "\nInternal winner.",
    )

    with pytest.raises(Exception, match="Parent|Skill|history|authoritative"):
        cli_module._publish_or_resume_accepted_retrieval_release(
            releases,
            internal_winner,
            skills=({"lineage": "original"},),
            audit=audit,
            parent_release=parent,
        )

    assert not (releases / "v007").exists()


def test_public_id_firewall_detects_ids_adjacent_to_underscores() -> None:
    with pytest.raises(Exception, match="Public Regression"):
        cli_module._assert_no_public_regression_ids(
            {"message": "prefix_task_100_suffix"},
            frozenset({"task_100"}),
        )


def test_public_ids_are_rejected_from_every_prompt_bearing_input(tmp_path) -> None:
    public_ids = frozenset({"task_100"})
    clean_policy = HarnessPolicy()
    clean_coding = SkillLibrary(tmp_path / "coding.json")
    clean_decision = DecisionSkillLibrary(tmp_path / "decision.json")
    tainted_policy = replace(
        clean_policy,
        decision_prompt="Apply the rule learned from task_100.",
    )
    tainted_coding = SkillLibrary(
        tmp_path / "tainted-coding.json",
        [
            Skill(
                skill_id="coding-public",
                name="public_example",
                description="Derived from task_100.",
                code="def forecast(history, horizon, frequency): return [0] * horizon",
                created_from_task="task_100",
            )
        ],
    )
    tainted_decision = DecisionSkillLibrary(
        tmp_path / "tainted-decision.json",
        [
            DecisionSkill(
                skill_id="decision-public",
                name="public_example",
                description="Derived from task_100.",
                applicability="Always.",
                decision_rule="Keep the leader.",
                failure_condition="Never.",
                created_from_task="task_100",
            )
        ],
    )

    for policy, coding, decision in (
        (tainted_policy, clean_coding, clean_decision),
        (clean_policy, tainted_coding, clean_decision),
        (clean_policy, clean_coding, tainted_decision),
    ):
        with pytest.raises(Exception, match="Public Regression"):
            cli_module._assert_retrieval_prompt_inputs_clean(
                policy,
                coding,
                decision,
                public_ids,
            )


def test_complete_result_task_provenance_is_rejected_by_checkpoint_firewall() -> None:
    parent = RetrievalGenome.seed()
    train = tuple(_manifest_task(f"train_{index:03d}") for index in range(80))
    dev = tuple(_manifest_task(f"dev_{index:03d}") for index in range(20))
    result = _retrieval_result(parent, train, dev, accepted=True)
    assert result.parent_dev is not None
    tainted = replace(
        result,
        parent_dev=replace(
            result.parent_dev,
            task_traces=(
                *result.parent_dev.task_traces,
                dict(result.parent_dev.task_traces[0]),
            ),
        ),
    )

    with pytest.raises(Exception, match="provenance|Dev|task"):
        cli_module._validate_retrieval_checkpoint_payload(
            tainted.to_payload(),
            parent=parent,
            train_ids=tuple(task.numeric.task_id for task in train),
            dev_ids=tuple(task.numeric.task_id for task in dev),
            public_ids=frozenset({"task_100"}),
        )


def test_retrieval_path_collision_is_rejected_before_task_loading(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs" / "formal"
    authority = tmp_path / "authority" / "checkpoint.json"
    collided = run_root / "artifact.json"
    args = build_parser().parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            str(tmp_path / "tasks.jsonl"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--retrieval-release-path",
            str(tmp_path / "releases" / "v000"),
            "--trace-path",
            str(collided),
            "--policy-path",
            str(collided),
            "--checkpoint-path",
            str(run_root / "checkpoint.json"),
            "--checkpoint-authority-path",
            str(authority),
            "--checkpoint-authority-head-path",
            str(authority.with_name("checkpoint.head.json")),
            "--checkpoint-authority-anchor-path",
            str(authority.with_name("checkpoint.anchors")),
        ]
    )
    args.run_root = str(run_root)
    args.evolution_mode = "retrieval"
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("colliding paths must fail before task loading")
        ),
    )

    with pytest.raises(ValueError, match="path|disjoint|collision"):
        cli_module._retrieval_evolve_command(args)


def _retrieval_security_paths(tmp_path: Path) -> SimpleNamespace:
    run_root = tmp_path / "runs" / "formal"
    authority_root = tmp_path / "authority"
    return SimpleNamespace(
        trace_path=str(run_root / "trace.json"),
        run_root=str(run_root),
        checkpoint_path=str(run_root / "checkpoint.json"),
        progress_path=str(run_root / "progress.jsonl"),
        policy_path=str(run_root / "policy.json"),
        checkpoint_authority_path=str(authority_root / "checkpoint.json"),
        checkpoint_authority_head_path=str(
            authority_root / "checkpoint.head.json"
        ),
        checkpoint_authority_anchor_path=str(
            authority_root / "checkpoint.anchors"
        ),
        tasks_file=str(tmp_path / "inputs" / "tasks.jsonl"),
        split_manifest=str(tmp_path / "inputs" / "split.json"),
        retrieval_release_path=str(tmp_path / "releases" / "v000"),
        library_path=str(tmp_path / "libraries" / "coding.json"),
        retrieval_library_path=str(
            tmp_path / "libraries" / "retrieval.json"
        ),
        decision_library_path=str(tmp_path / "libraries" / "decision.json"),
        seed_policy_path=None,
    )


@pytest.mark.parametrize(
    ("left_name", "right_name"),
    (("trace_path", "policy_path"), ("checkpoint_path", "progress_path")),
)
def test_retrieval_output_hard_link_aliases_are_rejected_before_writes(
    tmp_path: Path, left_name: str, right_name: str
) -> None:
    args = _retrieval_security_paths(tmp_path)
    left = Path(getattr(args, left_name))
    right = Path(getattr(args, right_name))
    left.parent.mkdir(parents=True)
    left.write_bytes(b"shared output inode\n")
    os.link(left, right)

    with pytest.raises(ValueError, match="alias|inode|identity|hard"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_rejects_hard_link_to_symlinked_protected_input(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    real_tasks = tmp_path / "protected" / "real-tasks.jsonl"
    real_tasks.parent.mkdir(parents=True)
    real_tasks.write_bytes(b"protected task bytes\n")
    tasks_alias = Path(args.tasks_file)
    tasks_alias.parent.mkdir(parents=True)
    tasks_alias.symlink_to(real_tasks)
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    os.link(real_tasks, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_rejects_hard_link_to_release_artifact(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    release_manifest = Path(args.retrieval_release_path) / "manifest.json"
    release_manifest.parent.mkdir(parents=True)
    release_manifest.write_bytes(b"protected release manifest\n")
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    os.link(release_manifest, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard|release"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_rejects_hard_link_to_task_directory_member(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    task_directory = Path(args.tasks_file)
    task_directory.mkdir(parents=True)
    task_member = task_directory / "partition-000.json"
    task_member.write_bytes(b'{"benchmark_id":"protected_member"}\n')
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    os.link(task_member, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard|task"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_rejects_hard_link_to_non_json_task_directory_member(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    task_directory = Path(args.tasks_file)
    task_directory.mkdir(parents=True)
    readme = task_directory / "README.txt"
    readme.write_bytes(b"protected task-source documentation\n")
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    os.link(readme, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard|task"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_rejects_hard_link_to_monotonic_anchor_member(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    anchor_directory = Path(args.checkpoint_authority_anchor_path)
    anchor_directory.mkdir(parents=True, mode=0o700)
    anchor_member = anchor_directory / ("anchor-" + "0" * 20 + "-" + "a" * 64 + ".json")
    anchor_member.write_bytes(b"protected monotonic anchor record\n")
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    os.link(anchor_member, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard|anchor"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_task_directory_replacement_fails_against_preflight_snapshot(
    tmp_path: Path,
) -> None:
    task_directory = tmp_path / "task-source"
    task_directory.mkdir()
    member = task_directory / "partition-000.json"
    encoded = b'{"benchmark_id":"protected_member"}\n'
    member.write_bytes(encoded)
    snapshot = cli_module._snapshot_retrieval_task_source(task_directory)
    displaced = tmp_path / "displaced-task-source"
    task_directory.rename(displaced)
    task_directory.mkdir()
    (task_directory / member.name).write_bytes(encoded)

    with pytest.raises(ValueError, match="source|directory|identity|preflight"):
        tuple(
            cli_module._iter_retrieval_task_record_texts(
                task_directory,
                expected_source_snapshot=snapshot,
            )
        )


@pytest.mark.parametrize(
    "protected_name",
    (
        "split_manifest",
        "library_path",
        "retrieval_library_path",
        "decision_library_path",
        "checkpoint_authority_path",
        "checkpoint_authority_head_path",
    ),
)
def test_retrieval_output_rejects_hard_link_to_every_protected_record(
    tmp_path: Path, protected_name: str
) -> None:
    args = _retrieval_security_paths(tmp_path)
    protected = Path(getattr(args, protected_name))
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(b"protected artifact inode\n")
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True, exist_ok=True)
    os.link(protected, trace)

    with pytest.raises(ValueError, match="alias|inode|identity|hard"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_output_writer_rejects_replacement_after_path_validation(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    trace = Path(args.trace_path)
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"owned prior trace\n")
    validated = cli_module._validate_retrieval_evolution_paths(args)
    displaced = trace.with_name("displaced-owned-trace.json")
    trace.rename(displaced)
    foreign = b"foreign replacement must survive\n"
    trace.write_bytes(foreign)

    with pytest.raises(ValueError, match="changed|identity|replacement"):
        cli_module._write_json_artifact(
            trace,
            {"safe": True},
            expected_identity=validated["output_identities"]["trace"],
        )

    assert trace.read_bytes() == foreign
    assert displaced.read_bytes() == b"owned prior trace\n"


def test_retrieval_output_writer_rejects_parent_replacement_after_preflight(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    authority_parent = Path(args.checkpoint_authority_path).parent
    authority_parent.mkdir(mode=0o700)
    validated = cli_module._validate_retrieval_evolution_paths(args)
    run_root = Path(args.run_root)
    displaced = run_root.with_name("displaced-formal-run")
    run_root.rename(displaced)
    run_root.mkdir()
    trace = Path(args.trace_path)

    with pytest.raises(ValueError, match="parent|directory|identity|replacement"):
        cli_module._write_json_artifact(
            trace,
            {"safe": True},
            expected_identity=validated["output_identities"]["trace"],
            expected_parent_identity=validated["output_parent_identities"]["trace"],
        )

    assert not trace.exists()
    assert not (displaced / trace.name).exists()


def test_retrieval_output_symlink_cannot_escape_approved_run_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs" / "formal"
    outside = tmp_path / "outside"
    run_root.mkdir(parents=True)
    outside.mkdir()
    (run_root / "escape").symlink_to(outside, target_is_directory=True)
    args = build_parser().parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            str(tmp_path / "tasks.jsonl"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--retrieval-release-path",
            str(tmp_path / "releases" / "v000"),
            "--trace-path",
            str(run_root / "escape" / "trace.json"),
            "--policy-path",
            str(run_root / "policy.json"),
            "--checkpoint-path",
            str(run_root / "checkpoint.json"),
            "--checkpoint-authority-path",
            str(tmp_path / "authority" / "checkpoint.json"),
            "--checkpoint-authority-head-path",
            str(tmp_path / "authority" / "checkpoint.head.json"),
            "--checkpoint-authority-anchor-path",
            str(tmp_path / "authority" / "checkpoint.anchors"),
        ]
    )
    args.run_root = str(run_root)
    args.evolution_mode = "retrieval"
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("escaping paths must fail before task loading")
        ),
    )

    with pytest.raises(ValueError, match="root|escape|path"):
        cli_module._retrieval_evolve_command(args)


def test_retrieval_authority_cannot_live_inside_a_release_or_library(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs" / "formal"
    release = tmp_path / "releases" / "v000"
    args = build_parser().parse_args(
        [
            "--evolution",
            "retrieval",
            "--tasks-file",
            str(tmp_path / "tasks.jsonl"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--retrieval-release-path",
            str(release),
            "--trace-path",
            str(run_root / "trace.json"),
            "--policy-path",
            str(run_root / "policy.json"),
            "--checkpoint-path",
            str(run_root / "checkpoint.json"),
            "--checkpoint-authority-path",
            str(release / "authority.json"),
            "--checkpoint-authority-head-path",
            str(release / "authority.head.json"),
            "--checkpoint-authority-anchor-path",
            str(release / "authority.anchors"),
        ]
    )
    args.run_root = str(run_root)
    args.evolution_mode = "retrieval"
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority collision must fail before task loading")
        ),
    )

    with pytest.raises(ValueError, match="authority|release|protected|collision"):
        cli_module._retrieval_evolve_command(args)


def test_retrieval_monotonic_anchor_cannot_live_inside_a_release(
    tmp_path: Path,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    args.checkpoint_authority_anchor_path = str(
        Path(args.retrieval_release_path) / "checkpoint.anchors"
    )

    with pytest.raises(ValueError, match="anchor|authority|release|protected|collision"):
        cli_module._validate_retrieval_evolution_paths(args)


def test_retrieval_evolution_requires_operator_head_and_secret_before_task_loading(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs" / "formal"
    authority = tmp_path / "authority" / "checkpoint.json"
    base_argv = [
        "--evolution",
        "retrieval",
        "--tasks-file",
        str(tmp_path / "tasks.jsonl"),
        "--split-manifest",
        str(tmp_path / "split.json"),
        "--retrieval-release-path",
        str(tmp_path / "releases" / "v000"),
        "--trace-path",
        str(run_root / "trace.json"),
        "--policy-path",
        str(run_root / "policy.json"),
        "--checkpoint-path",
        str(run_root / "checkpoint.json"),
        "--checkpoint-authority-path",
        str(authority),
        "--checkpoint-authority-anchor-path",
        str(authority.with_name("checkpoint.anchors")),
    ]
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority prerequisites must fail before task loading")
        ),
    )
    monkeypatch.delenv("RETRIEVAL_CHECKPOINT_AUTHORITY_KEY", raising=False)

    missing_head = build_parser().parse_args(base_argv)
    missing_head.run_root = str(run_root)
    missing_head.evolution_mode = "retrieval"
    with pytest.raises(ValueError, match="authority.*head|head.*authority"):
        cli_module._retrieval_evolve_command(missing_head)

    missing_anchor = build_parser().parse_args(
        [
            *base_argv[:-2],
            "--checkpoint-authority-head-path",
            str(authority.with_name("checkpoint.head.json")),
        ]
    )
    missing_anchor.run_root = str(run_root)
    missing_anchor.evolution_mode = "retrieval"
    with pytest.raises(ValueError, match="anchor|ledger"):
        cli_module._retrieval_evolve_command(missing_anchor)

    missing_key = build_parser().parse_args(
        [
            *base_argv,
            "--checkpoint-authority-head-path",
            str(authority.with_name("checkpoint.head.json")),
        ]
    )
    missing_key.run_root = str(run_root)
    missing_key.evolution_mode = "retrieval"
    with pytest.raises(ValueError, match="authority.*key|key.*authority"):
        cli_module._retrieval_evolve_command(missing_key)


def test_retrieval_authority_environment_is_consumed_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_key_name = "CUSTOM_RETRIEVAL_AUTHORITY_KEY"
    custom_expected_name = "CUSTOM_RETRIEVAL_AUTHORITY_EXPECTED"
    custom_key = "custom-private-authority-key-at-least-32-bytes"
    default_key = "default-private-authority-key-at-least-32-bytes"
    custom_expected = "7:" + "7" * 64
    default_expected = "8:" + "8" * 64
    args = SimpleNamespace(
        checkpoint_authority_key_env=custom_key_name,
        checkpoint_authority_expected_env=custom_expected_name,
    )
    monkeypatch.setenv(custom_key_name, custom_key)
    monkeypatch.setenv(custom_expected_name, custom_expected)
    monkeypatch.setenv("RETRIEVAL_CHECKPOINT_AUTHORITY_KEY", default_key)
    monkeypatch.setenv(
        "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED", default_expected
    )

    key, expected, subprocess_env = (
        cli_module._consume_retrieval_checkpoint_authority_environment(
            args,
            resume_required=True,
        )
    )

    assert key == custom_key.encode("utf-8")
    assert expected == (7, "7" * 64)
    for name in (
        custom_key_name,
        custom_expected_name,
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY",
        "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED",
    ):
        assert name not in os.environ
        assert name not in subprocess_env
    for value in (custom_key, default_key, custom_expected, default_expected):
        assert value not in subprocess_env.values()


@pytest.mark.parametrize(
    "expected_anchor",
    (None, "malformed", "-1:" + "0" * 64, "1:" + "A" * 64),
)
def test_retrieval_resume_requires_well_formed_external_anchor_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_anchor: str | None,
) -> None:
    args = _retrieval_security_paths(tmp_path)
    args.retrieval_mode = "two-stage"
    args.evolution_mode = "retrieval"
    args.checkpoint_authority_key_env = "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY"
    args.checkpoint_authority_expected_env = (
        "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED"
    )
    checkpoint = Path(args.checkpoint_path)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resume requires external monotonic authority\n")
    Path(args.checkpoint_authority_path).parent.mkdir(mode=0o700)
    monkeypatch.setenv(
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY",
        "task-8-anchor-prerequisite-key-32-bytes",
    )
    if expected_anchor is None:
        monkeypatch.delenv(
            "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED", raising=False
        )
    else:
        monkeypatch.setenv(
            "RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED", expected_anchor
        )
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_release_for_operator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("anchor gate must precede Parent loading")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("anchor gate must precede task loading")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_components",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("anchor gate must precede component loading")
        ),
    )

    with pytest.raises(
        (ValueError, RetrievalCheckpointError),
        match="anchor|authority|epoch|head",
    ):
        cli_module._retrieval_evolve_command(args)


def test_retrieval_evolution_rejects_candidate_parent_before_tasks_or_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        round1_prompt=RetrievalGenome.seed().round1_prompt + "\nCandidate only.",
    )
    candidate_release = write_retrieval_release(
        tmp_path / "releases", candidate
    )
    args = _retrieval_security_paths(tmp_path)
    args.retrieval_release_path = str(candidate_release.path)
    args.split_manifest_sha256 = cli_module.DRCIK_PUBLIC_80_20_99_SHA256
    args.retrieval_mode = "two-stage"
    args.evolution_mode = "retrieval"
    args.checkpoint_authority_key_env = (
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY"
    )
    Path(args.checkpoint_authority_path).parent.mkdir(mode=0o700)
    monkeypatch.setenv(
        "RETRIEVAL_CHECKPOINT_AUTHORITY_KEY",
        "task-8-parent-gate-operator-key-32-bytes",
    )
    monkeypatch.setattr(
        cli_module,
        "_load_retrieval_evolution_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate parent must fail before task loading")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_components",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate parent must fail before component loading")
        ),
    )

    with pytest.raises(ValueError, match="seed|accepted|candidate|parent"):
        cli_module._retrieval_evolve_command(args)


def test_caller_authored_checkpoint_sidecar_never_confers_resume_authority(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    authority = tmp_path / "checkpoint.authority.json"
    checkpoint.write_bytes(b"checkpoint-one")

    with pytest.raises(ValueError, match="sidecar|authority"):
        cli_module._restore_retrieval_checkpoint_authority(checkpoint, authority)

    digest = hashlib.sha256(b"checkpoint-one").hexdigest()
    authority.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "authority_epoch": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar|authority"):
        cli_module._restore_retrieval_checkpoint_authority(checkpoint, authority)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("chronos", ("--system", "backbone-only", "--backbone", "chronos")),
        ("timesfm", ("--system", "backbone-only", "--backbone", "timesfm")),
        ("statistical", ("--system", "backbone-only", "--backbone", "statistical")),
        ("codex-contract", ("--system", "codex-contract")),
        ("codex-direct", ("--system", "codex-direct")),
    ],
)
def test_baseline_names_route_to_existing_drcik_systems(name, expected) -> None:
    args = build_parser().parse_args(["--baseline", name, "--sample-dir", "sample"])
    translated = _baseline_argv(args)
    for index in range(len(expected) - 1):
        if expected[index].startswith("--"):
            position = translated.index(expected[index])
            assert translated[position + 1] == expected[index + 1]


def test_codex_and_rules_triad_have_distinct_reasoning_backends() -> None:
    parser = build_parser()
    codex = _baseline_argv(
        parser.parse_args(["--baseline", "codex-triad", "--sample-dir", "sample"])
    )
    rules = _baseline_argv(
        parser.parse_args(["--baseline", "rules-triad", "--sample-dir", "sample"])
    )
    assert codex[codex.index("--reasoning-agent") + 1] == "codex"
    assert rules[rules.index("--reasoning-agent") + 1] == "rules"


def test_baseline_requires_a_data_source() -> None:
    args = build_parser().parse_args(["--baseline", "chronos"])
    with pytest.raises(SystemExit, match="requires one data source"):
        _baseline_argv(args)


def test_three_way_split_is_entity_disjoint_and_reproducible() -> None:
    tasks = [
        ContextTask(
            numeric=Task(
                task_id=f"task_{index}",
                history_values=(1.0, 2.0),
                future_values=(3.0,),
                prediction_length=1,
                frequency="1 day",
                seasonal_period=None,
                entity_name=f"entity_{index // 2}",
            ),
            target_name="target",
            target_description="",
            history_timestamps=("1", "2"),
            future_timestamps=("3",),
            documents=(),
        )
        for index in range(12)
    ]
    first = _three_way_entity_split(tasks, 7, 0.25, 0.20)
    second = _three_way_entity_split(tasks, 7, 0.25, 0.20)
    assert [[item.numeric.task_id for item in split] for split in first] == [
        [item.numeric.task_id for item in split] for split in second
    ]
    entity_sets = [
        {item.numeric.entity_name for item in split}
        for split in first
    ]
    assert all(first)
    assert entity_sets[0].isdisjoint(entity_sets[1])
    assert entity_sets[0].isdisjoint(entity_sets[2])
    assert entity_sets[1].isdisjoint(entity_sets[2])


def test_harness_factory_hydrates_policy_embedded_skills(tmp_path) -> None:
    policy = HarnessPolicy(
        coding_skills=(
            {
                "skill_id": "coding-1",
                "name": "embedded_numeric",
                "description": "Embedded numeric strategy.",
                "code": "def forecast(history, horizon, frequency): return [history[-1]] * horizon",
                "created_from_task": "task_train",
                "assumption": "The level persists.",
                "failure_condition": "The regime changes.",
                "validation_score": 1.0,
                "uses": 0,
                "avg_score": None,
            },
        ),
        retrieval_skills=(
            {
                "skill_id": "retrieval-1",
                "name": "embedded_retrieval",
                "description": "Embedded retrieval strategy.",
                "applicability": "Scheduled events.",
                "query_strategy": "Search exact dates.",
                "verification_rule": "Require an exact quote.",
                "created_from_task": "task_train",
                "validation_score": 0.8,
                "uses": 0,
                "avg_score": None,
            },
        ),
        decision_skills=(
            {
                "skill_id": "decision-1",
                "name": "embedded_decision",
                "description": "Embedded selection strategy.",
                "applicability": "Multiple candidates.",
                "decision_rule": "Preserve the validated leader.",
                "failure_condition": "Evidence falsifies it.",
                "created_from_task": "task_train",
                "validation_score": 0.9,
                "uses": 0,
                "avg_score": None,
            },
        ),
    )
    harness = _factory(
        SimpleNamespace(setting="llm_only"),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
    )(policy)

    assert harness.coding.library.get("embedded_numeric") is not None
    assert harness.retrieval.library.get("embedded_retrieval") is not None
    assert harness.decision.library.get("embedded_decision") is not None


def test_factory_loads_retrieval_release_only_for_two_stage(tmp_path) -> None:
    single_pass = _factory(
        SimpleNamespace(
            setting="llm_only",
            retrieval_mode="single-pass",
            retrieval_release_path=tmp_path / "does-not-exist",
        ),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
    )(HarnessPolicy())
    assert not isinstance(single_pass.retrieval, TwoStageRetrievalAgent)

    release = write_retrieval_release(tmp_path / "releases", RetrievalGenome.seed())

    class Morphology:
        def assumptions(self, task):
            del task
            return ()

    two_stage = _factory(
        SimpleNamespace(
            setting="llm_only",
            retrieval_mode="two-stage",
            retrieval_release_path=release.path,
        ),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
        morphology_provider=Morphology(),
    )(HarnessPolicy())
    assert isinstance(two_stage.retrieval, TwoStageRetrievalAgent)
    assert two_stage.runtime.retrieval_mode == "two_stage"


def test_decision_coordinate_factory_freezes_coding_and_retrieval_learning(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches actual Decision Train evaluation mutating a frozen coordinate."""
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    release_library = RetrievalSkillLibrary._from_loaded_release(release)

    class Morphology:
        def assumptions(self, task):
            del task
            return ()

    harness = _factory(
        SimpleNamespace(
            setting="llm_only",
            retrieval_mode="two-stage",
            retrieval_release_path=release.path,
        ),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        release_library,
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
        morphology_provider=Morphology(),
        retrieval_genome=release.genome,
        retrieval_skill_source=release_library,
        coordinate_target="decision",
    )(
        cli_module._policy_with_retrieval_release(
            HarnessPolicy(), release, changelog="Seed Retrieval."
        )
    )
    observed: list[bool] = []
    monkeypatch.setattr(
        harness.coding._delegate,
        "run_task",
        lambda _task, *, allow_skill_writes=True: observed.append(
            allow_skill_writes
        ),
    )

    harness.coding.run_task(object(), allow_skill_writes=True)

    assert observed == [False]
    assert harness.coding.config is harness.coding._delegate.config
    assert harness.coding._folds.__self__ is harness.coding._delegate
    assert harness.outcome_learner.retrieval_library is not harness.retrieval.skills
    assert harness.outcome_learner.retrieval_library.all() == ()
    assert harness.outcome_learner.decision_library is harness.decision.library


def test_factory_rejects_two_stage_without_morphology_provider(tmp_path) -> None:
    release = write_retrieval_release(tmp_path / "releases", RetrievalGenome.seed())

    with pytest.raises(ValueError, match="MorphologyProvider"):
        _factory(
            SimpleNamespace(
                setting="llm_only",
                retrieval_mode="two-stage",
                retrieval_release_path=release.path,
            ),
            FakeLLMClient([]),
            SkillLibrary(tmp_path / "coding.json"),
            RetrievalSkillLibrary(tmp_path / "retrieval.json"),
            DecisionSkillLibrary(tmp_path / "decision.json"),
            None,
            isolate_library=True,
        )


def test_two_stage_factory_ignores_legacy_policy_retrieval_rows(tmp_path) -> None:
    release = write_retrieval_release(tmp_path / "releases", RetrievalGenome.seed())

    class Morphology:
        def assumptions(self, task):
            del task
            return ()

    factory = _factory(
        SimpleNamespace(
            setting="llm_only",
            retrieval_mode="two-stage",
            retrieval_release_path=release.path,
        ),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
        morphology_provider=Morphology(),
    )

    harness = factory(HarnessPolicy(retrieval_skills=({"legacy_free_form": True},)))

    assert isinstance(harness.retrieval, TwoStageRetrievalAgent)
    assert harness.retrieval.skills.all() == ()


def test_source_inference_propagates_two_stage_controls_and_worker_fails_closed(
    tmp_path, monkeypatch
) -> None:
    patch_path = tmp_path / "accepted.patch"
    patch_path.write_text("")
    release_path = tmp_path / "retrieval-release"
    captured = {}

    def fake_source_inference(**kwargs):
        captured.update(kwargs["config"]["runtime"])
        worker_args = SimpleNamespace(**kwargs["config"]["runtime"])
        with pytest.raises(ValueError, match="MorphologyProvider"):
            _factory(
                worker_args,
                FakeLLMClient([]),
                SkillLibrary(tmp_path / "worker-coding.json"),
                RetrievalSkillLibrary(tmp_path / "worker-retrieval.json"),
                DecisionSkillLibrary(tmp_path / "worker-decision.json"),
                None,
                isolate_library=True,
            )
        return {"status": "fail_closed"}

    monkeypatch.setattr(cli_module, "run_source_inference", fake_source_inference)
    args = build_parser().parse_args(
        [
            "--inference",
            "source",
            "--source-patch-path",
            str(patch_path),
            "--retrieval-mode",
            "two-stage",
            "--retrieval-release-path",
            str(release_path),
        ]
    )

    result = inference_command(args)

    assert result == {"status": "fail_closed"}
    assert captured["retrieval_mode"] == "two-stage"
    assert captured["retrieval_release_path"] == str(release_path.resolve())


def test_source_evolution_worker_receives_two_stage_controls_and_fails_closed(
    tmp_path, monkeypatch
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("")
    release_path = tmp_path / "retrieval-release"
    real_run = subprocess.run

    def allow_dirty_worktree_check(command, *args, **kwargs):
        if list(command[:3]) == ["git", "diff", "--quiet"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    class EvaluatingEngine:
        def __init__(self, repo_root, evaluate, config):
            del repo_root, config
            self.evaluate = evaluate

        def evolve(self, seed_patch=""):
            del seed_patch
            self.evaluate(cli_module.Path.cwd())
            raise AssertionError("two-stage worker must fail before evaluation")

    monkeypatch.setattr(cli_module.subprocess, "run", allow_dirty_worktree_check)
    monkeypatch.setattr(cli_module, "SourceEvolutionEngine", EvaluatingEngine)
    args = build_parser().parse_args(
        [
            "evolve",
            "--evolution-mode",
            "source",
            "--tasks-file",
            str(tasks_path),
            "--retrieval-mode",
            "two-stage",
            "--retrieval-release-path",
            str(release_path),
            "--trace-path",
            str(tmp_path / "trace.json"),
            "--source-patch-path",
            str(tmp_path / "source.patch"),
        ]
    )

    with pytest.raises(RuntimeError, match="MorphologyProvider"):
        _source_evolve_command(
            args,
            [],
            [],
            checkpoint_path=tmp_path / "checkpoint.json",
            progress_path=tmp_path / "progress.jsonl",
        )


def test_harness_factory_replays_complete_versioned_retrieval_skill_history(tmp_path) -> None:
    base = {
        "skill_id": "window_skill",
        "version": 1,
        "parent_version": None,
        "stage": "both",
        "status": "accepted",
        "name": "window_skill",
        "description": "Find windows.",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": [],
            "temporal_relations": [],
        },
        "query_steps": ["Find event boundaries."],
        "required_chain_fields": ["entity", "target"],
        "counterevidence_rule": "Search for cancellation.",
        "failure_conditions": ["The event is outside the forecast window."],
        "validated_task_ids": ["train_1", "train_2", "train_3"],
        "validated_entities": ["north", "south"],
        "validation_smae_gain": 0.1,
        "validation_srmse_gain": 0.1,
        "merged_from_skill_ids": [],
        "quarantine_reason": None,
    }
    version2 = {
        **base,
        "version": 2,
        "parent_version": 1,
        "description": "Find inclusive event boundaries.",
    }
    version3 = {
        **version2,
        "version": 3,
        "parent_version": 2,
        "stage": "round2",
        "status": "specialized",
        "applicability": {
            "assumption_kinds": [],
            "gap_types": ["missing_window"],
            "temporal_relations": [],
        },
    }
    version4 = {
        **version3,
        "version": 4,
        "parent_version": 3,
        "status": "quarantined",
        "quarantine_reason": "unsafe on ambiguous dates",
    }
    genome = replace(RetrievalGenome.seed(), version="v001", parent="v000")
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        genome,
        skills=(base, version2, version3, version4),
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )
    source = RetrievalSkillLibrary.from_release(release.path)
    policy = snapshot_policy_skills(
        HarnessPolicy(),
        SimpleNamespace(retrieval=SimpleNamespace(library=source)),
    )
    policy_payload = policy.to_payload()
    assert "retrieval_skill_source" not in policy_payload
    assert isinstance(policy.retrieval_skill_source, RetrievalSkillLibrary)
    json.dumps(policy_payload)

    harness = _factory(
        SimpleNamespace(setting="llm_only"),
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
    )(policy)

    history = harness.retrieval.library.history("window_skill")
    assert [item.version for item in history] == [1, 2, 3, 4]
    assert history[-1].status == "quarantined"
