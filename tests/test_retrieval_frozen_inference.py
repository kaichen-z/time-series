from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import evolving_loop.cli as cli_module
import evolving_loop.frozen_inference as frozen_module
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, Document, Task
from evolving_loop.decision_agent.agent import DecisionCandidate, DecisionResult
from evolving_loop.frozen_inference import run_frozen_inference
from evolving_loop.evaluation import ResolvedOutcome
from evolving_loop.retrieval_agent.credit import RetrievalTaskDiagnostics
from evolving_loop.retrieval_agent.schemas import FinalRetrievalCard
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    write_retrieval_release,
)


def _hidden_task() -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="hidden_1",
            history_values=(1.0, 2.0),
            future_values=(),
            prediction_length=1,
            frequency="1 day",
            seasonal_period=None,
            entity_name="Entity",
        ),
        target_name="volume",
        target_description="public description",
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03",),
        documents=(Document("doc_1", "Demand rose by 10%.", "SECRET_ROLE", "SECRET_SUBTYPE"),),
        gt_evidence=("SECRET_GT",),
        labels_public=False,
    )


def _chain(chain_id: str, *, stance: str, skill_id: str) -> dict[str, object]:
    return {
        "chain_id": chain_id,
        "claim": "Demand rose.",
        "entity_match": True,
        "target_match": True,
        "temporal_relation": "overlaps_future",
        "mechanism": "future_driver",
        "direction": "up",
        "magnitude_kind": "relative",
        "magnitude_value": 0.1,
        "start_timestamp": "2026-01-01",
        "end_timestamp": "2026-01-03",
        "citations": [
            {"document_id": "doc_1", "exact_quote": "Demand rose by 10%."}
        ],
        "missing_links": [],
        "used_skill_ids": [skill_id],
        "addressed_assumption_ids": ["assumption_1"],
        "stance": stance,
        "numeric_eligible": True,
    }


def _card() -> FinalRetrievalCard:
    first = _chain("round1_chain", stance="supports", skill_id="skill_round1")
    second = _chain("round2_chain", stance="challenges", skill_id="skill_round2")
    return FinalRetrievalCard.from_payload(
        {
            "round1": {
                "evidence_chains": [first],
                "counterevidence": [],
                "missing_information": ["Need counterevidence"],
                "sufficient": False,
                "gaps": [
                    {
                        "assumption_id": "assumption_1",
                        "gap_type": "counterevidence",
                        "missing_information": "Could demand reverse?",
                        "priority": "high",
                    }
                ],
                "rejected": ["round1_bad_citation"],
            },
            "round2": {
                "evidence_chains": [second],
                "counterevidence": [],
                "missing_information": [],
                "sufficient": True,
                "rejected": ["round2_bad_citation"],
            },
            "chains": [first, second],
            "selected_document_ids": ["doc_1"],
            "rejected": ["round1_bad_citation", "round2_bad_citation"],
            "unresolved_contradictions": ["direction_conflict"],
            "complete": True,
            "gaps": [
                {
                    "assumption_id": "assumption_1",
                    "gap_type": "counterevidence",
                    "missing_information": "Could demand reverse?",
                    "priority": "high",
                }
            ],
        }
    )


def _result() -> SimpleNamespace:
    candidate = DecisionCandidate(
        candidate_id="level",
        forecast=(3.0,),
        assumption="level persists",
        failure_condition="regime changes",
        hindcast_smape=1.0,
    )
    card = _card()
    return SimpleNamespace(
        forecast=(3.0,),
        retrieval=card.to_legacy_result(),
        retrieval_card=card,
        decision=DecisionResult(
            selected=candidate,
            host_default_id="level",
            requested_more_retrieval=False,
            rationale="verified",
            supporting_document_ids=("doc_1",),
            llm_override_accepted=False,
        ),
        candidates=(candidate,),
    )


def _frozen_path_args(
    root: Path,
    *,
    release_path: Path,
    output: Path,
) -> SimpleNamespace:
    authority = root / "authority"
    return SimpleNamespace(
        output_dir=str(output),
        output_root=str(output),
        policy_path=str(root / "policy.json"),
        tasks_file=str(root / "tasks.jsonl"),
        retrieval_release_path=str(release_path),
        library_path=str(root / "coding-library.json"),
        retrieval_library_path=str(root / "retrieval-library.json"),
        decision_library_path=str(root / "decision-library.json"),
        split_manifest=None,
        seed_policy_path=None,
        checkpoint_authority_path=str(authority / "checkpoint.json"),
        checkpoint_authority_head_path=str(authority / "checkpoint.head.json"),
        checkpoint_authority_anchor_path=str(authority / "checkpoint.anchors"),
    )


def _release_files(path: Path) -> dict[str, bytes]:
    return {
        child.name: child.read_bytes()
        for child in path.iterdir()
        if child.is_file()
    }


def test_hidden_retrieval_inference_is_write_free_and_reports_both_rounds(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "library.json"
    sentinel.write_text("unchanged", encoding="utf-8")
    calls: list[tuple[ContextTask, bool]] = []

    class Harness:
        def run(self, task: ContextTask, *, allow_skill_writes: bool = True):
            calls.append((task, allow_skill_writes))
            return _result()

    release = write_retrieval_release(
        outside / "release-root", RetrievalGenome.seed()
    )
    policy = cli_module._policy_with_retrieval_release(
        HarnessPolicy(), release, changelog="Seed Retrieval."
    )
    summary = run_frozen_inference(
        policy,
        [_hidden_task()],
        lambda _policy: Harness(),
        output_dir=tmp_path / "output",
        score_public=False,
        artifact_kind="retrieval",
    )

    assert len(calls) == 1
    inference_task, allow_skill_writes = calls[0]
    assert allow_skill_writes is False
    assert inference_task.numeric.future_values == ()
    assert inference_task.gt_evidence == ()
    assert inference_task.labels_public is False
    assert inference_task.documents[0].role is None
    assert inference_task.documents[0].subtype is None
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert {path.name for path in tmp_path.iterdir()} == {"outside", "output"}

    report = json.loads(
        (tmp_path / "output" / "run_report.jsonl").read_text(encoding="utf-8")
    )
    assert report["labels_accessed"] is False
    assert report["release_sha256"] == policy.retrieval_release_sha256
    assert report["retrieval"]["round1"]["evidence_chains"][0]["chain_id"] == "round1_chain"
    assert report["retrieval"]["round2"]["evidence_chains"][0]["chain_id"] == "round2_chain"
    assert report["retrieval"]["rejected"] == [
        "round1_bad_citation",
        "round2_bad_citation",
    ]
    assert report["retrieval"]["round1"]["gaps"][0]["assumption_id"] == "assumption_1"
    assert report["retrieval"]["gaps"][0]["assumption_id"] == "assumption_1"
    assert report["assumption_stances"] == [
        {
            "chain_id": "round1_chain",
            "assumption_ids": ["assumption_1"],
            "stance": "supports",
        },
        {
            "chain_id": "round2_chain",
            "assumption_ids": ["assumption_1"],
            "stance": "challenges",
        },
    ]
    assert report["used_skill_ids"] == ["skill_round1", "skill_round2"]
    encoded = json.dumps(report).lower()
    assert "secret_gt" not in encoded
    assert "secret_role" not in encoded
    assert "secret_subtype" not in encoded
    assert "gt_evidence" not in encoded
    assert summary["labels_accessed"] is False


def test_explicit_public_scoring_never_serializes_gt_or_document_role_fields(
    tmp_path, monkeypatch
) -> None:
    task = replace(
        _hidden_task(),
        numeric=replace(_hidden_task().numeric, future_values=(3.0,)),
        gt_evidence=("SECRET_GT",),
        labels_public=True,
    )

    class Harness:
        def run(self, _task, *, allow_skill_writes=True):
            assert allow_skill_writes is False
            return _result()

    diagnostics = RetrievalTaskDiagnostics(
        supporting_recall=1.0,
        gt_evidence_recall=0.5,
        distractor_avoidance=1.0,
        exact_quote_validity=1.0,
        complete_chain_rate=1.0,
        contextual_oracle_smae_gain=0.0,
        contextual_oracle_srmse_gain=0.0,
        invalid_count=0,
        catastrophic_count=0,
        chain_credit=(),
    )
    monkeypatch.setattr(
        frozen_module,
        "score_after_resolution",
        lambda original, _result: ResolvedOutcome(
            task_id=original.numeric.task_id,
            final_smae=0.1,
            final_srmse=0.2,
            contextual_oracle_smae=0.1,
            contextual_oracle_srmse=0.2,
            retrieval_diagnostics=diagnostics,
        ),
    )

    run_frozen_inference(
        HarnessPolicy(),
        [task],
        lambda _policy: Harness(),
        output_dir=tmp_path,
        score_public=True,
        artifact_kind="retrieval",
    )

    encoded = (tmp_path / "run_report.jsonl").read_text(encoding="utf-8").lower()
    assert "gt_evidence" not in encoded
    assert "secret_gt" not in encoded
    assert "secret_role" not in encoded
    assert "secret_subtype" not in encoded


def test_frozen_publication_rejects_post_validation_output_symlink_to_release(
    tmp_path: Path,
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    output = tmp_path / "frozen-output"
    output.mkdir()
    args = _frozen_path_args(
        tmp_path,
        release_path=release.path,
        output=output,
    )
    validated_output = cli_module._validate_frozen_retrieval_paths(
        args, release
    )
    release_before = _release_files(release.path)
    displaced = tmp_path / "displaced-frozen-output"
    output.rename(displaced)
    output.symlink_to(release.path, target_is_directory=True)

    class Harness:
        def run(self, _task, *, allow_skill_writes=True):
            assert allow_skill_writes is False
            return _result()

    with pytest.raises(
        ValueError,
        match="output|directory|identity|symlink|changed|replacement|safe",
    ):
        run_frozen_inference(
            HarnessPolicy(),
            [_hidden_task()],
            lambda _policy: Harness(),
            output_dir=validated_output,
            artifact_kind="retrieval",
        )

    assert output.is_symlink()
    assert _release_files(release.path) == release_before
    assert {child.name for child in release.path.iterdir()} == set(release_before)


def test_frozen_publication_retains_replaced_output_member_after_validation(
    tmp_path: Path,
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    output = tmp_path / "frozen-output"
    output.mkdir()
    forecast = output / "forecasts.jsonl"
    forecast.write_bytes(b"owned preflight output\n")
    args = _frozen_path_args(
        tmp_path,
        release_path=release.path,
        output=output,
    )
    validated_output = cli_module._validate_frozen_retrieval_paths(
        args, release
    )
    displaced = output / "retained-owned-forecast.jsonl"
    forecast.rename(displaced)
    foreign = b"foreign output replacement must survive\n"
    forecast.write_bytes(foreign)

    class Harness:
        def run(self, _task, *, allow_skill_writes=True):
            return _result()

    with pytest.raises(
        ValueError,
        match="output|identity|changed|replacement|publication",
    ):
        run_frozen_inference(
            HarnessPolicy(),
            [_hidden_task()],
            lambda _policy: Harness(),
            output_dir=validated_output,
            artifact_kind="retrieval",
        )

    assert forecast.read_bytes() == foreign
    assert displaced.read_bytes() == b"owned preflight output\n"
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("swap_after", (1, 4))
def test_frozen_publication_mid_bundle_directory_swap_never_writes_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_after: int,
) -> None:
    release = write_retrieval_release(
        tmp_path / "releases", RetrievalGenome.seed()
    )
    output = tmp_path / "frozen-output"
    output.mkdir()
    (output / "summary.json").write_text(
        '{"status":"prior successful run"}\n', encoding="utf-8"
    )
    args = _frozen_path_args(
        tmp_path,
        release_path=release.path,
        output=output,
    )
    validated_output = cli_module._validate_frozen_retrieval_paths(
        args, release
    )
    release_before = _release_files(release.path)
    real_publish = getattr(
        frozen_module, "_publish_frozen_output_entry", None
    )
    publication_count = 0
    displaced = tmp_path / "displaced-mid-bundle-output"

    def publish_then_swap(*publish_args, **publish_kwargs):
        nonlocal publication_count
        assert real_publish is not None
        result = real_publish(*publish_args, **publish_kwargs)
        publication_count += 1
        if publication_count == swap_after:
            output.rename(displaced)
            output.symlink_to(release.path, target_is_directory=True)
        return result

    monkeypatch.setattr(
        frozen_module,
        "_publish_frozen_output_entry",
        publish_then_swap,
        raising=False,
    )

    class Harness:
        def run(self, _task, *, allow_skill_writes=True):
            return _result()

    with pytest.raises(
        ValueError,
        match="output|directory|identity|symlink|changed|replacement",
    ):
        run_frozen_inference(
            HarnessPolicy(),
            [_hidden_task()],
            lambda _policy: Harness(),
            output_dir=validated_output,
            artifact_kind="retrieval",
        )

    assert publication_count == swap_after
    assert output.is_symlink()
    assert _release_files(release.path) == release_before
    assert not (release.path / "summary.json").exists()
    assert not (displaced / "summary.json").exists()


def test_frozen_publication_retires_summary_when_noreplace_commits_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frozen-output"
    real_rename = frozen_module._rename_artifact_entry_noreplace

    def rename_then_raise(
        directory_descriptor: int, source: str, destination: str
    ) -> None:
        real_rename(directory_descriptor, source, destination)
        if destination == "summary.json":
            raise OSError("simulated uncertainty after durable summary rename")

    monkeypatch.setattr(
        frozen_module,
        "_rename_artifact_entry_noreplace",
        rename_then_raise,
    )

    class Harness:
        def run(self, _task, *, allow_skill_writes=True):
            return _result()

    with pytest.raises(ValueError, match="output|summary|publication|replacement"):
        run_frozen_inference(
            HarnessPolicy(),
            [_hidden_task()],
            lambda _policy: Harness(),
            output_dir=output,
            artifact_kind="retrieval",
        )

    assert not (output / "summary.json").exists()
