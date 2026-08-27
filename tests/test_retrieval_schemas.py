from __future__ import annotations

import json

import pytest

from evolving_loop.data import ContextTask, Document
from common.data import Task
from evolving_loop.retrieval_agent.schemas import (
    EvidenceChain,
    FinalRetrievalCard,
    RetrievalAssumption,
    RetrievalContractError,
    RetrievalGap,
    RetrievalRoundResult,
    build_round1_payload,
    build_round2_payload,
)


@pytest.fixture
def context_task() -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="task_1",
            history_values=(1.0, 2.0, 3.0),
            future_values=(),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="entity_a",
        ),
        target_name="target_a",
        target_description="a target",
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03", "2026-01-04"),
        documents=(
            Document(
                document_id="doc_1",
                content="An event affects target_a from 2026-01-03 through 2026-01-04.",
                role="relevant",
                subtype="event",
            ),
        ),
        gt_evidence=("must not be exposed",),
        labels_public=True,
    )


def _citation() -> dict[str, object]:
    return {"document_id": "doc_1", "exact_quote": "An event affects target_a"}


def _chain(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "chain_id": "chain_1",
        "claim": "An event affects the target.",
        "entity_match": True,
        "target_match": True,
        "temporal_relation": "overlaps_future",
        "mechanism": "future_driver",
        "direction": "up",
        "magnitude_kind": "unknown",
        "magnitude_value": None,
        "start_timestamp": "2026-01-03",
        "end_timestamp": "2026-01-04",
        "citations": [_citation()],
        "missing_links": ["explicit_magnitude"],
        "used_skill_ids": [],
        "addressed_assumption_ids": [],
        "stance": "unresolved",
        "numeric_eligible": False,
    }
    value.update(overrides)
    return value


def _round(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_chains": [_chain()],
        "counterevidence": [],
        "missing_information": [],
        "sufficient": False,
    }
    value.update(overrides)
    return value


def test_round1_payload_is_assumption_blind(context_task: ContextTask):
    payload = build_round1_payload(context_task, skills=())
    encoded = json.dumps(payload, sort_keys=True)
    assert "documents" in payload
    for forbidden in (
        "coding_hypotheses", "assumptions", "future_values", "gt_evidence",
        "role", "subtype", "hindcast_smae", "hindcast_srmse",
    ):
        assert forbidden not in encoded


def test_round2_rejects_candidate_identity_and_scores():
    with pytest.raises(RetrievalContractError, match="forbidden round-two field"):
        RetrievalAssumption.from_payload({
            "assumption_id": "a_trend",
            "kind": "trend_persistence",
            "claim": "The recent trend persists.",
            "failure_condition": "A regime reversal occurs.",
            "candidate_id": "linear_trend",
            "hindcast_smae": 0.8,
        })


def test_schema_rejects_unknown_keys_duplicate_ids_and_invalid_enums():
    with pytest.raises(RetrievalContractError):
        EvidenceChain.from_payload({**_chain(), "unexpected": True})
    with pytest.raises(RetrievalContractError, match="duplicate"):
        RetrievalRoundResult.from_payload(_round(evidence_chains=[_chain(), _chain()]))
    with pytest.raises(RetrievalContractError, match="invalid temporal"):
        EvidenceChain.from_payload(_chain(temporal_relation="maybe"))


def test_schema_rejects_nonfinite_magnitudes_and_malformed_timestamps():
    with pytest.raises(RetrievalContractError, match="finite"):
        EvidenceChain.from_payload(_chain(magnitude_kind="absolute", magnitude_value=float("nan")))
    with pytest.raises(RetrievalContractError, match="timestamp"):
        EvidenceChain.from_payload(_chain(start_timestamp="not-a-timestamp"))


def test_round2_payload_contains_only_sanitized_fields(context_task: ContextTask):
    round1 = RetrievalRoundResult.from_payload(_round())
    assumption = RetrievalAssumption.from_payload({
        "assumption_id": "a_trend",
        "kind": "trend_persistence",
        "claim": "The recent trend persists.",
        "failure_condition": "A reversal occurs.",
    })
    payload = build_round2_payload(
        context_task,
        round1,
        gaps=({
            "assumption_id": "a_trend",
            "gap_type": "continuation_or_reversal",
            "missing_information": "Evidence of continuation or reversal",
            "priority": "high",
        },),
        assumptions=(assumption,),
        skills=(),
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert set(payload["assumptions"][0]) == {
        "assumption_id", "kind", "claim", "failure_condition"
    }
    for forbidden in ("candidate_id", "forecast_values", "hindcast_smae", "hindcast_srmse", "code"):
        assert forbidden not in encoded


def test_round2_revalidates_directly_constructed_invalid_assumption(context_task: ContextTask):
    forged = RetrievalAssumption(
        assumption_id="a_trend",
        kind="candidate_score_leak",
        claim="candidate_id=linear_trend hindcast_smae=0.8",
        failure_condition="A reversal occurs.",
    )
    with pytest.raises(RetrievalContractError, match="invalid assumption kind"):
        build_round2_payload(context_task, RetrievalRoundResult.from_payload(_round()), (), (forged,))


def test_round2_revalidates_directly_constructed_invalid_gap(context_task: ContextTask):
    assumption = RetrievalAssumption(
        assumption_id="a_trend",
        kind="trend_persistence",
        claim="The recent trend persists.",
        failure_condition="A reversal occurs.",
    )
    forged = RetrievalGap(
        assumption_id="a_trend",
        gap_type="candidate_score_leak",
        missing_information="candidate_id=linear_trend hindcast_srmse=0.4",
        priority="high",
    )
    with pytest.raises(RetrievalContractError, match="forbidden round-two field"):
        build_round2_payload(
            context_task,
            RetrievalRoundResult.from_payload(_round()),
            (forged,),
            (assumption,),
        )


def test_final_card_requires_both_stages_and_rejects_duplicate_merged_ids():
    raw = {
        "round1": _round(),
        "round2": _round(),
        "chains": [_chain()],
        "selected_document_ids": ["doc_1"],
        "rejected": [],
        "unresolved_contradictions": [],
        "complete": False,
    }
    card = FinalRetrievalCard.from_payload(raw)
    assert card.round1.chains[0].chain_id == "chain_1"
    with pytest.raises(RetrievalContractError, match="duplicate"):
        FinalRetrievalCard.from_payload({**raw, "chains": [_chain(), _chain(chain_id="chain_1")]})
