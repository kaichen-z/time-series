from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
import evolving_loop.cli as cli_module

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
from evolving_loop.coding_agent.skill_library import SkillLibrary
from evolving_loop.data import ContextTask, Task
from evolving_loop.decision_agent.skill_library import DecisionSkillLibrary
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkill,
    RetrievalSkillLibrary,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
    write_retrieval_release,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from common.llm import FakeLLMClient


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
