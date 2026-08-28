from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
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
    RetrievalEvaluation,
    RetrievalForecastingFailure,
    RetrievalEvolutionResult,
    RetrievalGenerationTrace,
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
    mutated_in_memory.retrieval_release_payload["genome"]["round1_strategy"] = (
        "entity_first"
    )
    with pytest.raises(ValueError, match="Retrieval release|retrieval release|fingerprint"):
        mutated_in_memory.save(tmp_path / "mutated-policy.json")

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
            token = authority.prepare(self._checkpoint_file_sha256)
            checkpoint.write_bytes(b"trusted-checkpoint")
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
    monkeypatch.setattr(
        cli_module,
        "_components",
        lambda *_args, **kwargs: (
            object(),
            SimpleNamespace(all=lambda: ()),
            kwargs.get("retrieval_library_override"),
            SimpleNamespace(all=lambda: ()),
            None,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_factory",
        lambda *_args, **_kwargs: (lambda _policy: object()),
    )
    runs = tmp_path / "runs"
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "checkpoint.json"
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
    policy = cli_module._policy_with_retrieval_release(
        HarnessPolicy(), candidate, changelog="Candidate only."
    )

    with pytest.raises(ValueError, match="accepted|seed|candidate|state"):
        cli_module._policy_for_retrieval_release(policy, candidate)


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
