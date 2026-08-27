from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from common.data import Task
from evolving_loop.data import ContextTask, Document
from evolving_loop.harness import (
    CandidatePoolEntry,
    CandidatePoolSnapshot,
    SkillLeaveOneOutSnapshot,
    _candidate_pool_snapshots,
)
from evolving_loop.decision_agent.agent import DecisionCandidate
from evolving_loop.retrieval_agent.credit import (
    RetrievalSkillTaskEvidence,
    assign_chain_credit,
    promote_retrieval_skills,
    validate_skill_necessity,
)
from evolving_loop.retrieval_agent.schemas import (
    EvidenceChain,
    EvidenceCitation,
    FinalRetrievalCard,
    RetrievalRoundResult,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillLibrary,
)


def _task() -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="credit_task",
            history_values=(8.0, 9.0, 10.0),
            future_values=(12.0, 12.0),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Alpha",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=("2026-01-18", "2026-01-19", "2026-01-20"),
        future_timestamps=("2026-01-21", "2026-01-22"),
        documents=(
            Document(
                "support",
                "Alpha sales will increase by 20 percent from 2026-01-21 through 2026-01-22.",
                role="supporting",
            ),
            Document("noise", "The office carpet is blue.", role="distractor"),
        ),
        gt_evidence=(
            "Alpha sales will increase by 20 percent from 2026-01-21 through 2026-01-22.",
        ),
    )


def _chain(
    chain_id: str,
    *,
    numeric_eligible: bool,
    used_skill_ids: tuple[str, ...] = (),
    document_id: str = "support",
) -> EvidenceChain:
    quote = (
        "Alpha sales will increase by 20 percent from 2026-01-21 through 2026-01-22."
        if document_id == "support"
        else "The office carpet is blue."
    )
    return EvidenceChain(
        chain_id=chain_id,
        claim=quote,
        entity_match=document_id == "support",
        target_match=document_id == "support",
        temporal_relation="overlaps_future" if numeric_eligible else "unknown",
        mechanism="future_driver" if numeric_eligible else "irrelevant",
        direction="up" if numeric_eligible else "unknown",
        magnitude_kind="relative" if numeric_eligible else "unknown",
        magnitude_value=0.2 if numeric_eligible else None,
        start_timestamp="2026-01-21" if numeric_eligible else None,
        end_timestamp="2026-01-22" if numeric_eligible else None,
        citations=(EvidenceCitation(document_id, quote),),
        missing_links=() if numeric_eligible else ("magnitude",),
        used_skill_ids=used_skill_ids,
        addressed_assumption_ids=(),
        stance="supports",
        numeric_eligible=numeric_eligible,
    )


def _snapshot(
    after_chain_id: str | None,
    *candidates: tuple[str, tuple[float, ...]],
) -> CandidatePoolSnapshot:
    return CandidatePoolSnapshot(
        after_chain_id=after_chain_id,
        candidates=tuple(CandidatePoolEntry(candidate_id, forecast) for candidate_id, forecast in candidates),
    )


def _result(
    *chains: EvidenceChain,
    snapshots: tuple[CandidatePoolSnapshot, ...],
    leave_one_out: tuple[SkillLeaveOneOutSnapshot, ...] = (),
    forecast: tuple[float, ...] = (10.0, 10.0),
):
    round1 = RetrievalRoundResult(chains, (), (), True)
    card = FinalRetrievalCard(
        round1=round1,
        round2=None,
        chains=chains,
        selected_document_ids=tuple(
            dict.fromkeys(citation.document_id for chain in chains for citation in chain.citations)
        ),
        rejected=(),
        unresolved_contradictions=(),
        complete=all(chain.numeric_eligible for chain in chains),
    )
    return SimpleNamespace(
        retrieval_card=card,
        candidate_pool_snapshots=snapshots,
        skill_leave_one_out_snapshots=leave_one_out,
        forecast=forecast,
    )


def test_candidate_snapshots_are_immutable_and_contain_no_evaluator_labels() -> None:
    snapshot = _snapshot(None, ("numeric", (10.0, 10.0)))

    with pytest.raises(FrozenInstanceError):
        snapshot.after_chain_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.candidates[0].forecast = (12.0, 12.0)  # type: ignore[misc]
    assert set(vars(snapshot)) == {"after_chain_id", "candidates"}
    assert set(vars(snapshot.candidates[0])) == {"candidate_id", "forecast"}


def test_legacy_result_fallback_canonicalizes_duplicate_candidate_ids() -> None:
    duplicate = DecisionCandidate("numeric", (10.0, 10.0), "numeric", "none")
    result = SimpleNamespace(
        candidate_pool_snapshots=(),
        coding=SimpleNamespace(
            candidates=(
                SimpleNamespace(program=SimpleNamespace(name="numeric")),
                SimpleNamespace(program=SimpleNamespace(name="numeric")),
            )
        ),
        candidates=(duplicate, duplicate),
        retrieval=SimpleNamespace(selected_document_ids=(), rejected=()),
        retrieval_card=None,
        forecast=(10.0, 10.0),
    )

    report = assign_chain_credit(_task(), result)

    assert report.coding_oracle_smae == pytest.approx(1.0 / 6.0)


def test_credit_interfaces_are_exported_from_the_retrieval_package() -> None:
    from evolving_loop.retrieval_agent import (
        EvidenceChainCredit,
        RetrievalTaskDiagnostics,
        assign_chain_credit,
    )

    assert EvidenceChainCredit.__module__.endswith(".credit")
    assert RetrievalTaskDiagnostics.__module__.endswith(".credit")
    assert callable(assign_chain_credit)


def test_chain_credit_uses_candidate_pool_gain_not_final_decision() -> None:
    complete = _chain("complete", numeric_eligible=True)
    incomplete = _chain("qualitative", numeric_eligible=False, document_id="noise")
    result = _result(
        complete,
        incomplete,
        snapshots=(
            _snapshot(None, ("numeric", (10.0, 10.0))),
            _snapshot(
                "complete",
                ("numeric", (10.0, 10.0)),
                ("numeric__evidence_0", (12.0, 12.0)),
            ),
        ),
    )

    report = assign_chain_credit(_task(), result)

    assert report.coding_oracle_smae - report.contextual_oracle_smae == pytest.approx(
        sum(item.marginal_smae_gain for item in report.chains)
    )
    assert report.decision_smae_regret == pytest.approx(
        report.final_smae - report.contextual_oracle_smae
    )
    assert report.chains[1].chain_id == "qualitative"
    assert report.chains[1].marginal_smae_gain == 0.0
    assert report.chains[1].marginal_srmse_gain == 0.0
    assert report.diagnostics.supporting_recall == 1.0
    assert report.diagnostics.gt_evidence_recall == 1.0
    assert report.diagnostics.distractor_avoidance == 0.0
    assert report.diagnostics.exact_quote_validity == 1.0
    assert report.diagnostics.complete_chain_rate == 0.5


def test_chain_credit_follows_verified_order_without_double_attribution() -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    result = _result(
        first,
        second,
        snapshots=(
            _snapshot(None, ("numeric", (6.0, 6.0))),
            _snapshot(
                "first",
                ("numeric", (6.0, 6.0)),
                ("first_candidate", (10.0, 10.0)),
            ),
            _snapshot(
                "second",
                ("numeric", (6.0, 6.0)),
                ("first_candidate", (10.0, 10.0)),
                ("second_candidate", (12.0, 12.0)),
            ),
        ),
    )

    report = assign_chain_credit(_task(), result)

    assert [item.chain_id for item in report.chains] == ["first", "second"]
    assert sum(item.marginal_smae_gain for item in report.chains) == pytest.approx(
        report.coding_oracle_smae - report.contextual_oracle_smae
    )


def test_skipped_chain_keeps_an_unchanged_pool_without_stealing_later_credit() -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    round1 = RetrievalRoundResult((first, second), (), (), True)
    card = FinalRetrievalCard(
        round1=round1,
        round2=None,
        chains=(first, second),
        selected_document_ids=("support",),
        rejected=(),
        unresolved_contradictions=(),
        complete=True,
    )
    candidates = (
        DecisionCandidate("numeric", (10.0, 10.0), "numeric", "none"),
        DecisionCandidate(
            "numeric__evidence_1",
            (12.0, 12.0),
            "second chain only",
            "none",
            tags=("evidence_adjusted", "future_driver"),
        ),
    )

    snapshots = _candidate_pool_snapshots(candidates, card)

    assert [snapshot.after_chain_id for snapshot in snapshots] == [None, "first", "second"]
    assert [len(snapshot.candidates) for snapshot in snapshots] == [1, 1, 2]


def test_legacy_history_clean_candidate_enters_contextual_not_numeric_pool() -> None:
    candidates = (
        DecisionCandidate("numeric", (10.0, 10.0), "numeric", "none"),
        DecisionCandidate(
            "history_clean__numeric",
            (12.0, 12.0),
            "verified observation repair",
            "none",
            tags=("history_cleaned", "observation", "single"),
        ),
    )

    snapshots = _candidate_pool_snapshots(candidates, None)

    assert [snapshot.after_chain_id for snapshot in snapshots] == [
        None,
        "legacy_history_clean_0",
    ]
    assert [len(snapshot.candidates) for snapshot in snapshots] == [1, 2]


def test_joint_skill_credit_requires_leave_one_out_replay() -> None:
    chain = _chain(
        "joint",
        numeric_eligible=True,
        used_skill_ids=("window_search", "quote_check"),
    )
    baseline = _snapshot(None, ("numeric", (10.0, 10.0)))
    full = _snapshot(
        "joint",
        ("numeric", (10.0, 10.0)),
        ("joint_candidate", (12.0, 12.0)),
    )
    result = _result(
        chain,
        snapshots=(baseline, full),
        leave_one_out=(
            SkillLeaveOneOutSnapshot("joint", "window_search", baseline),
            SkillLeaveOneOutSnapshot("joint", "quote_check", full),
        ),
    )

    report = assign_chain_credit(_task(), result)
    assert report.chains[0].skill_credit == ()

    validated = validate_skill_necessity(_task(), result, "joint")
    assert {item.skill_id for item in validated if item.necessary} == {"window_search"}


def _candidate_skill() -> RetrievalSkill:
    return RetrievalSkill(
        skill_id="window_search",
        version=1,
        parent_version=None,
        stage="round1",
        status="candidate",
        name="window_search",
        description="Find an exact future event window.",
        applicability=RetrievalApplicability(),
        query_steps=("Find both inclusive endpoints.",),
        required_chain_fields=("start_timestamp", "end_timestamp"),
        counterevidence_rule="Search for cancellation.",
        failure_conditions=("The event does not overlap the horizon.",),
    )


def _promotion_rows(**overrides: object) -> tuple[RetrievalSkillTaskEvidence, ...]:
    rows = []
    for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1):
        values = {
            "skill_id": "window_search",
            "task_id": f"train_{index}",
            "entity_name": entity,
            "split": "train",
            "exact_quote_validity": 1.0,
            "without_skill_smae": 0.5,
            "without_skill_srmse": 0.6,
            "with_skill_smae": 0.4 if index == 1 else 0.5,
            "with_skill_srmse": 0.5 if index == 1 else 0.6,
            "added_catastrophic_count": 0,
            "necessary": True,
        }
        values.update(overrides)
        rows.append(RetrievalSkillTaskEvidence(**values))
    return tuple(rows)


def test_skill_promotion_requires_every_cross_train_gate(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))

    promoted = promote_retrieval_skills(library, _promotion_rows())

    assert promoted == ("window_search",)
    accepted = library.get_by_id("window_search")
    assert accepted is not None
    assert accepted.status == "accepted"
    assert accepted.version == 2
    assert accepted.validated_task_ids == ("train_1", "train_2", "train_3")
    assert accepted.validated_entities == ("Alpha", "Beta")


@pytest.mark.parametrize(
    "overrides",
    (
        {"task_id": "same_task"},
        {"entity_name": "Alpha"},
        {"exact_quote_validity": 0.99},
        {"with_skill_smae": 0.7},
        {"with_skill_smae": 0.5, "with_skill_srmse": 0.6},
        {"added_catastrophic_count": 1},
        {"necessary": False},
    ),
)
def test_skill_promotion_fails_closed_when_any_train_gate_fails(tmp_path, overrides) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))

    assert promote_retrieval_skills(library, _promotion_rows(**overrides)) == ()
    assert library.get_by_id("window_search").status == "candidate"


def test_dev_evidence_is_read_only_and_cannot_promote_skills(tmp_path) -> None:
    path = tmp_path / "skills.json"
    library = RetrievalSkillLibrary(path, (_candidate_skill(),))
    library.save()
    before = path.read_bytes()
    dev_rows = tuple(replace(row, split="dev") for row in _promotion_rows())

    assert promote_retrieval_skills(library, dev_rows) == ()
    assert library.get_by_id("window_search").status == "candidate"
    assert path.read_bytes() == before
