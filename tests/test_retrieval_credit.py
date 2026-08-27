from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import pytest
import evolving_loop.retrieval_agent.skill_library as skill_library_module

from common.data import Task
from evolving_loop.data import ContextTask, Document
from evolving_loop.harness import (
    CandidatePoolEntry,
    CandidatePoolSnapshot,
    SkillLeaveOneOutSnapshot,
    _candidate_pool_snapshots,
)
from evolving_loop.decision_agent.agent import DecisionCandidate
from evolving_loop.evaluation import score_after_resolution
from evolving_loop.retrieval_agent.credit import (
    RetrievalSkillTaskEvidence,
    assign_chain_credit,
    derive_retrieval_skill_evidence,
    evaluate_and_promote_retrieval_skills,
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
    RetrievalSkillError,
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
    executed = tuple(
        DecisionCandidate(
            candidate.candidate_id,
            candidate.forecast,
            candidate.candidate_id,
            "none",
            tags=("evidence_adjusted",)
            if "__evidence_" in candidate.candidate_id
            else (),
        )
        for candidate in snapshots[-1].candidates
    )
    coding = tuple(
        SimpleNamespace(program=SimpleNamespace(name=candidate.candidate_id), forecast=candidate.forecast)
        for candidate in snapshots[0].candidates
    )
    selected = next((candidate for candidate in executed if candidate.forecast == forecast), executed[0])
    return SimpleNamespace(
        retrieval_card=card,
        candidate_pool_snapshots=snapshots,
        skill_leave_one_out_snapshots=leave_one_out,
        forecast=forecast,
        candidates=executed,
        coding=SimpleNamespace(candidates=coding),
        retrieval=SimpleNamespace(selected_document_ids=card.selected_document_ids, rejected=()),
        decision=SimpleNamespace(selected=selected),
    )


def test_candidate_snapshots_are_immutable_and_contain_no_evaluator_labels() -> None:
    snapshot = _snapshot(None, ("numeric", (10.0, 10.0)))

    with pytest.raises(FrozenInstanceError):
        snapshot.after_chain_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.candidates[0].forecast = (12.0, 12.0)  # type: ignore[misc]
    assert set(vars(snapshot)) == {"after_chain_id", "candidates"}
    assert set(vars(snapshot.candidates[0])) == {"candidate_id", "forecast"}
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        CandidatePoolSnapshot(
            None,
            (
                CandidatePoolEntry("duplicate", (10.0, 10.0)),
                CandidatePoolEntry("duplicate", (12.0, 12.0)),
            ),
        )


def test_credit_rejects_missing_snapshots_instead_of_reconstructing_them() -> None:
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

    with pytest.raises(ValueError, match="snapshots"):
        assign_chain_credit(_task(), result)


def test_resolved_scoring_returns_invalid_diagnostics_for_horizon_mismatch() -> None:
    snapshots = (_snapshot(None, ("numeric", (10.0,))),)
    result = _result(snapshots=snapshots, forecast=(10.0,))

    outcome = score_after_resolution(_task(), result)

    assert outcome.final_smae == 5.0
    assert outcome.final_srmse == 5.0
    assert outcome.retrieval_diagnostics.invalid_count == 1


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
                ("numeric__evidence_0", (10.0, 10.0)),
            ),
            _snapshot(
                "second",
                ("numeric", (6.0, 6.0)),
                ("numeric__evidence_0", (10.0, 10.0)),
                ("numeric__evidence_1", (12.0, 12.0)),
            ),
        ),
    )

    report = assign_chain_credit(_task(), result)

    assert [item.chain_id for item in report.chains] == ["first", "second"]
    assert sum(item.marginal_smae_gain for item in report.chains) == pytest.approx(
        report.coding_oracle_smae - report.contextual_oracle_smae
    )


@pytest.mark.parametrize(
    "snapshots, match",
    (
        (
            (
                _snapshot(None, ("numeric", (6.0, 6.0))),
                _snapshot("first", ("numeric", (7.0, 7.0)), ("numeric__evidence_0", (10.0, 10.0))),
                _snapshot("second", ("numeric", (7.0, 7.0)), ("numeric__evidence_0", (10.0, 10.0)), ("numeric__evidence_1", (12.0, 12.0))),
            ),
            "changed",
        ),
        (
            (
                _snapshot(None, ("numeric", (6.0, 6.0))),
                _snapshot("first", ("numeric__evidence_0", (10.0, 10.0)), ("numeric", (6.0, 6.0))),
                _snapshot("second", ("numeric__evidence_0", (10.0, 10.0)), ("numeric", (6.0, 6.0)), ("numeric__evidence_1", (12.0, 12.0))),
            ),
            "prefix",
        ),
        (
            (
                _snapshot(None, ("numeric", (6.0, 6.0))),
                _snapshot("first", ("numeric", (6.0, 6.0)), ("phantom", (10.0, 10.0))),
                _snapshot("second", ("numeric", (6.0, 6.0)), ("phantom", (10.0, 10.0)), ("second_candidate", (12.0, 12.0))),
            ),
            "executed",
        ),
    ),
)
def test_credit_rejects_snapshots_not_bound_to_executed_cumulative_pool(
    snapshots, match
) -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    result = _result(first, second, snapshots=snapshots)
    if match == "executed":
        result.candidates = tuple(
            candidate for candidate in result.candidates if candidate.candidate_id != "phantom"
        )

    with pytest.raises(ValueError, match=match):
        assign_chain_credit(_task(), result)


@pytest.mark.parametrize("card_backed", (False, True))
def test_snapshot_stage_rejects_swapped_executed_evidence_candidates(card_backed) -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    stage_ids = ("first", "second") if card_backed else (
        "legacy_evidence_0",
        "legacy_evidence_1",
    )
    snapshots = (
        _snapshot(None, ("numeric", (6.0, 6.0))),
        _snapshot(
            stage_ids[0],
            ("numeric", (6.0, 6.0)),
            ("numeric__evidence_1", (12.0, 12.0)),
        ),
        _snapshot(
            stage_ids[1],
            ("numeric", (6.0, 6.0)),
            ("numeric__evidence_1", (12.0, 12.0)),
            ("numeric__evidence_0", (10.0, 10.0)),
        ),
    )
    if card_backed:
        result = _result(first, second, snapshots=snapshots)
    else:
        candidates = tuple(
            DecisionCandidate(
                item.candidate_id,
                item.forecast,
                item.candidate_id,
                "none",
                tags=("evidence_adjusted",)
                if "__evidence_" in item.candidate_id
                else (),
            )
            for item in snapshots[-1].candidates
        )
        result = _legacy_result(candidates, snapshots)

    with pytest.raises(ValueError, match="candidate|stage|addition"):
        assign_chain_credit(_task(), result)


def test_card_snapshot_allows_unchanged_stage_only_when_candidate_was_not_created() -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    candidates = (
        DecisionCandidate("numeric", (10.0, 10.0), "numeric", "none"),
        DecisionCandidate(
            "numeric__evidence_1",
            (12.0, 12.0),
            "second only",
            "none",
            tags=("evidence_adjusted",),
        ),
    )
    snapshots = (
        _snapshot(None, ("numeric", (10.0, 10.0))),
        _snapshot("first", ("numeric", (10.0, 10.0))),
        _snapshot(
            "second",
            ("numeric", (10.0, 10.0)),
            ("numeric__evidence_1", (12.0, 12.0)),
        ),
    )
    result = _result(first, second, snapshots=snapshots)
    result.candidates = candidates

    report = assign_chain_credit(_task(), result)

    assert report.chains[0].marginal_smae_gain == 0.0
    assert report.chains[1].marginal_smae_gain > 0.0


def test_card_snapshot_rejects_delaying_an_existing_candidate_to_a_later_chain() -> None:
    first = _chain("first", numeric_eligible=True)
    second = _chain("second", numeric_eligible=True)
    snapshots = (
        _snapshot(None, ("numeric", (10.0, 10.0))),
        _snapshot("first", ("numeric", (10.0, 10.0))),
        _snapshot(
            "second",
            ("numeric", (10.0, 10.0)),
            ("numeric__evidence_0", (12.0, 12.0)),
        ),
    )

    with pytest.raises(ValueError, match="exact executed candidate"):
        assign_chain_credit(_task(), _result(first, second, snapshots=snapshots))


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


def _legacy_result(
    candidates: tuple[DecisionCandidate, ...],
    snapshots: tuple[CandidatePoolSnapshot, ...],
):
    baseline = snapshots[0].candidates
    return SimpleNamespace(
        retrieval_card=None,
        candidate_pool_snapshots=snapshots,
        skill_leave_one_out_snapshots=(),
        forecast=candidates[0].forecast,
        candidates=candidates,
        coding=SimpleNamespace(
            candidates=tuple(
                SimpleNamespace(
                    program=SimpleNamespace(name=item.candidate_id),
                    forecast=item.forecast,
                )
                for item in baseline
            )
        ),
        retrieval=SimpleNamespace(selected_document_ids=(), rejected=()),
        decision=SimpleNamespace(selected=candidates[0]),
    )


@pytest.mark.parametrize("mutation", ("extra", "reordered"))
def test_legacy_snapshot_order_is_derived_from_executed_candidate_artifacts(
    mutation,
) -> None:
    numeric = DecisionCandidate("numeric", (10.0, 10.0), "numeric", "none")
    evidence = DecisionCandidate(
        "numeric__evidence_0",
        (11.0, 11.0),
        "evidence",
        "none",
        tags=("evidence_adjusted",),
    )
    history = DecisionCandidate(
        "history_clean__numeric",
        (12.0, 12.0),
        "history",
        "none",
        tags=("history_cleaned",),
    )
    candidates = (numeric, evidence) if mutation == "extra" else (numeric, evidence, history)
    if mutation == "extra":
        snapshots = (
            _snapshot(None, ("numeric", (10.0, 10.0))),
            _snapshot("legacy_evidence_0", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (11.0, 11.0))),
            _snapshot("legacy_evidence_1", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (11.0, 11.0))),
        )
    else:
        snapshots = (
            _snapshot(None, ("numeric", (10.0, 10.0))),
            _snapshot("legacy_history_clean_0", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (11.0, 11.0))),
            _snapshot("legacy_evidence_0", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (11.0, 11.0)), ("history_clean__numeric", (12.0, 12.0))),
        )

    with pytest.raises(ValueError, match="legacy|order|extra"):
        assign_chain_credit(_task(), _legacy_result(candidates, snapshots))


@pytest.mark.parametrize("mutation", ("outside_pool", "changed_forecast"))
def test_selected_candidate_must_exactly_match_the_frozen_executed_pool(mutation) -> None:
    baseline = _snapshot(None, ("numeric", (10.0, 10.0)))
    result = _result(snapshots=(baseline,), forecast=(10.0, 10.0))
    selected_id = "rogue" if mutation == "outside_pool" else "numeric"
    result.decision.selected = DecisionCandidate(
        selected_id, (12.0, 12.0), "mutated selection", "none"
    )
    result.forecast = (12.0, 12.0)

    outcome = score_after_resolution(_task(), result)

    assert outcome.final_smae == 5.0
    assert outcome.final_srmse == 5.0
    assert outcome.retrieval_diagnostics.invalid_count == 1


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
        ("numeric__evidence_0", (12.0, 12.0)),
    )
    result = _result(
        chain,
        snapshots=(baseline, full),
        leave_one_out=(
            SkillLeaveOneOutSnapshot(
                "joint", "window_search", replace(baseline, after_chain_id="joint")
            ),
            SkillLeaveOneOutSnapshot("joint", "quote_check", full),
        ),
    )

    report = assign_chain_credit(_task(), result)
    assert report.chains[0].skill_credit == ()

    validated = validate_skill_necessity(_task(), result, "joint")
    assert {item.skill_id for item in validated if item.necessary} == {"window_search"}


@pytest.mark.parametrize("mutation", ("duplicate", "extra", "wrong_chain", "unrelated_change"))
def test_leave_one_out_replay_provenance_fails_closed(mutation) -> None:
    chain = _chain("joint", numeric_eligible=True, used_skill_ids=("window_search",))
    baseline = _snapshot(None, ("numeric", (10.0, 10.0)))
    full = _snapshot("joint", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (12.0, 12.0)))
    valid = SkillLeaveOneOutSnapshot(
        "joint", "window_search", _snapshot("joint", ("numeric", (10.0, 10.0)))
    )
    replays = [valid]
    if mutation == "duplicate":
        replays.append(valid)
    elif mutation == "extra":
        replays.append(SkillLeaveOneOutSnapshot("joint", "extra_skill", full))
    elif mutation == "wrong_chain":
        replays[0] = SkillLeaveOneOutSnapshot("other", "window_search", valid.snapshot)
    else:
        replays[0] = SkillLeaveOneOutSnapshot(
            "joint", "window_search", _snapshot("joint", ("numeric", (9.0, 9.0)))
        )
    result = _result(chain, snapshots=(baseline, full), leave_one_out=tuple(replays))

    with pytest.raises(ValueError, match="leave-one-out|replay"):
        validate_skill_necessity(_task(), result, "joint")


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


def _promotion_task_result(
    index: int,
    entity: str,
    *,
    include_replay: bool = True,
    replay_keeps_full_pool: bool = False,
    valid_quote: bool = True,
):
    task = _task()
    task = replace(
        task,
        numeric=replace(task.numeric, task_id=f"train_{index}", entity_name=entity),
    )
    chain = _chain("gain", numeric_eligible=True, used_skill_ids=("window_search",))
    if not valid_quote:
        chain = replace(
            chain,
            citations=(EvidenceCitation("support", "invented quote"),),
        )
    baseline = _snapshot(None, ("numeric", (10.0, 10.0)))
    full = _snapshot("gain", ("numeric", (10.0, 10.0)), ("numeric__evidence_0", (12.0, 12.0)))
    replays = (
        SkillLeaveOneOutSnapshot(
            "gain",
            "window_search",
            full
            if replay_keeps_full_pool
            else _snapshot("gain", ("numeric", (10.0, 10.0))),
        ),
    ) if include_replay else ()
    return task, _result(chain, snapshots=(baseline, full), leave_one_out=replays, forecast=(12.0, 12.0))


def test_skill_promotion_requires_trusted_evaluator_derived_evidence(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )

    promoted = evaluate_and_promote_retrieval_skills(
        library, task_results, split="train"
    )

    assert promoted == ("window_search",)
    accepted = library.get_by_id("window_search")
    assert accepted is not None
    assert accepted.status == "accepted"
    assert accepted.version == 2
    assert accepted.validated_task_ids == ("train_1", "train_2", "train_3")
    assert accepted.validated_entities == ("Alpha", "Beta")
    reloaded = RetrievalSkillLibrary.load_verified_checkpoint(library.path)
    assert reloaded.get_by_id("window_search").status == "accepted"
    assert reloaded.get_by_id("window_search").version == 2


def test_evaluator_checkpoint_cannot_be_copied_to_claim_file_provenance(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )
    assert evaluate_and_promote_retrieval_skills(
        library, task_results, split="train"
    ) == ("window_search",)
    copied = tmp_path / "copied.json"
    copied.write_bytes(library.path.read_bytes())

    with pytest.raises(RetrievalSkillError, match="checkpoint|provenance|witness"):
        RetrievalSkillLibrary.load_verified_checkpoint(copied)


def test_evaluator_promotion_checkpoint_rejects_direct_record_tampering(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )
    assert evaluate_and_promote_retrieval_skills(
        library, task_results, split="train"
    ) == ("window_search",)
    payload = __import__("json").loads(library.path.read_text(encoding="utf-8"))
    payload["skills"][-1]["validation_smae_gain"] = 4.9
    library.path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalSkillError, match="provenance|integrity|hash|tamper"):
        RetrievalSkillLibrary.load_verified_checkpoint(library.path)


def test_loaded_library_refuses_to_overwrite_a_replaced_checkpoint_path(tmp_path) -> None:
    path = tmp_path / "skills.json"
    library = RetrievalSkillLibrary(path, (_candidate_skill(),))
    library.save()
    loaded = RetrievalSkillLibrary.load(path)
    replacement = b'{"schema_version": 1, "skills": []}\n'
    path.write_bytes(replacement)

    with pytest.raises(RetrievalSkillError, match="replaced|changed|tamper"):
        loaded.save()

    assert path.read_bytes() == replacement


def test_evaluator_promotion_write_failure_is_atomic(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "skills.json"
    library = RetrievalSkillLibrary(path, (_candidate_skill(),))
    library.save()
    before = path.read_bytes()
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )

    def fail_replace(source, destination):
        del source, destination
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(skill_library_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        evaluate_and_promote_retrieval_skills(library, task_results, split="train")

    assert library.get_by_id("window_search").status == "candidate"
    assert library.get_by_id("window_search").version == 1
    assert path.read_bytes() == before
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert not (path.parent / f".{path.name}.provenance").exists()


def test_no_public_evidence_row_api_can_authorize_promotion(tmp_path) -> None:
    for name in ("promote_retrieval_skills", "_TRUSTED_EVIDENCE_PROVENANCE"):
        with pytest.raises(ImportError):
            exec(
                f"from evolving_loop.retrieval_agent.credit import {name}",
                {},
            )
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )
    derived = derive_retrieval_skill_evidence(task_results, split="train")
    forged_batches = (
        _promotion_rows(),
        tuple(copy.copy(row) for row in derived),
        tuple(replace(row) for row in derived),
        pickle.loads(pickle.dumps(derived)),
        tuple(asdict(row) for row in derived),
    )
    for index, forged in enumerate(forged_batches):
        library = RetrievalSkillLibrary(
            tmp_path / f"skills_{index}.json", (_candidate_skill(),)
        )

        assert evaluate_and_promote_retrieval_skills(
            library, forged, split="train"  # type: ignore[arg-type]
        ) == ()
        assert library.get_by_id("window_search").status == "candidate"


def test_zero_leave_one_out_replays_cannot_promote(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))
    task_results = tuple(
        _promotion_task_result(index, entity, include_replay=False)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )

    rows = derive_retrieval_skill_evidence(task_results, split="train")

    assert rows == ()
    assert evaluate_and_promote_retrieval_skills(
        library, rows, split="train"  # type: ignore[arg-type]
    ) == ()


@pytest.mark.parametrize(
    "task_results",
    (
        tuple(_promotion_task_result(index, "Alpha") for index in range(1, 4)),
        tuple(_promotion_task_result(index, entity) for index, entity in enumerate(("Alpha", "Beta"), 1)),
        tuple(
            _promotion_task_result(index, entity, valid_quote=index != 2)
            for index, entity in enumerate(("Alpha", "Alpha", "Beta"), 1)
        ),
        tuple(
            _promotion_task_result(index, entity, replay_keeps_full_pool=index == 2)
            for index, entity in enumerate(("Alpha", "Alpha", "Beta"), 1)
        ),
    ),
)
def test_trusted_evidence_still_requires_every_cross_train_gate(
    tmp_path, task_results
) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))

    assert evaluate_and_promote_retrieval_skills(
        library, task_results, split="train"
    ) == ()
    assert library.get_by_id("window_search").status == "candidate"


def test_skill_evidence_metrics_are_capped_to_formal_report_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        replace(_promotion_rows()[0], with_skill_smae=5.01)


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
def test_caller_created_rows_never_bypass_train_gates(tmp_path, overrides) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", (_candidate_skill(),))

    assert evaluate_and_promote_retrieval_skills(
        library, _promotion_rows(**overrides), split="train"  # type: ignore[arg-type]
    ) == ()
    assert library.get_by_id("window_search").status == "candidate"


def test_dev_evidence_is_read_only_and_cannot_promote_skills(tmp_path) -> None:
    path = tmp_path / "skills.json"
    library = RetrievalSkillLibrary(path, (_candidate_skill(),))
    library.save()
    before = path.read_bytes()
    task_results = tuple(
        _promotion_task_result(index, entity)
        for index, entity in enumerate(("Alpha", "Alpha", "Beta"), start=1)
    )

    assert evaluate_and_promote_retrieval_skills(
        library, task_results, split="dev"
    ) == ()
    assert library.get_by_id("window_search").status == "candidate"
    assert path.read_bytes() == before
