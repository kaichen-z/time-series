from __future__ import annotations

from dataclasses import replace

import pytest

from common.data import Task
from evolving_loop.data import ContextTask, Document
from evolving_loop.retrieval_agent.schemas import FinalRetrievalCard
from evolving_loop.retrieval_agent.verifier import (
    merge_verified_rounds,
    stable_chain_id,
    verify_round_result,
)


@pytest.fixture
def context_task() -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="task_1",
            history_values=(1.0, 2.0),
            future_values=(),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Entity A",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=("2026-01-01", "2026-01-02"),
        future_timestamps=("2026-01-03", "2026-01-04"),
        documents=(
            Document(
                "doc_1",
                "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04. "
                "The scheduled promotion is a documented future driver for Entity A sales.",
            ),
            Document(
                "doc_other",
                "Entity B sales will increase by 20 percent from 2026-01-03 through 2026-01-04.",
            ),
            Document(
                "doc_wrongtarget",
                "Entity A inventory will increase by 20 percent from 2026-01-03 through 2026-01-04.",
            ),
            Document(
                "doc_no_magnitude",
                "Entity A sales will increase from 2026-01-03 through 2026-01-04.",
            ),
        ),
        labels_public=False,
    )


def _chain(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "chain_id": "untrusted_chain",
        "claim": "The scheduled promotion increases sales.",
        "entity_match": True,
        "target_match": True,
        "temporal_relation": "overlaps_future",
        "mechanism": "future_driver",
        "direction": "up",
        "magnitude_kind": "relative",
        "magnitude_value": 0.2,
        "start_timestamp": "2026-01-03",
        "end_timestamp": "2026-01-04",
        "citations": [{
            "document_id": "doc_1",
            "exact_quote": "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04.",
        }],
        "missing_links": [],
        "used_skill_ids": ["s_window"],
        "addressed_assumption_ids": [],
        "stance": "supports",
        "numeric_eligible": True,
    }
    value.update(overrides)
    return value


def _payload(*chains: dict[str, object]) -> dict[str, object]:
    return {
        "evidence_chains": list(chains),
        "counterevidence": [],
        "missing_information": [],
        "sufficient": True,
    }


def _verified(context_task: ContextTask, raw: dict[str, object]):
    return verify_round_result(
        context_task,
        raw,
        stage="round1",
        allowed_skill_ids=("s_window",),
        allowed_assumption_ids=(),
    )


def test_verifier_canonicalizes_a_complete_chain_and_legacy_adapter_projects_only_it(context_task):
    verified = _verified(context_task, _payload(_chain()))

    chain = verified.chains[0]
    assert chain.numeric_eligible is True
    assert chain.chain_id == stable_chain_id(chain)
    card = FinalRetrievalCard(
        round1=verified,
        round2=None,
        chains=verified.chains,
        selected_document_ids=("doc_1",),
        rejected=verified.rejected,
        unresolved_contradictions=(),
        complete=True,
    )
    legacy = card.to_legacy_result()
    assert legacy.selected_document_ids == ("doc_1",)
    assert len(legacy.evidence) == 1
    assert legacy.impacts[0].adjustment_kind == "multiply"
    assert legacy.impacts[0].adjustment_value == 0.2


def test_verifier_rejects_fabricated_spans_and_retains_independently_exact_split_spans(context_task):
    fabricated = _verified(
        context_task,
        _payload(_chain(citations=[{"document_id": "doc_1", "exact_quote": "Made up 20 percent claim."}])),
    )
    assert fabricated.chains[0].numeric_eligible is False
    assert "citation" in fabricated.chains[0].missing_links

    split = _verified(
        context_task,
        _payload(_chain(citations=[{
            "document_id": "doc_1",
                "exact_quote": (
                "The scheduled promotion is a documented future driver for Entity A sales. "
                "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04."
                ),
        }])),
    )
    assert tuple(item.exact_quote for item in split.chains[0].citations) == (
        "The scheduled promotion is a documented future driver for Entity A sales.",
        "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04.",
    )


@pytest.mark.parametrize(
    ("overrides", "missing_link"),
    [
        ({"citations": [{"document_id": "doc_other", "exact_quote": "Entity B sales will increase by 20 percent from 2026-01-03 through 2026-01-04."}]}, "entity"),
        ({"citations": [{"document_id": "doc_wrongtarget", "exact_quote": "Entity A inventory will increase by 20 percent from 2026-01-03 through 2026-01-04."}]}, "target"),
        ({"start_timestamp": "2027-01-03", "end_timestamp": "2027-01-04"}, "forecast_window"),
        ({"magnitude_value": 0.2, "citations": [{"document_id": "doc_no_magnitude", "exact_quote": "Entity A sales will increase from 2026-01-03 through 2026-01-04."}]}, "magnitude"),
        ({"magnitude_kind": "multiplier", "magnitude_value": 0.0}, "multiplier"),
        ({"used_skill_ids": ["unknown_skill"]}, "skill"),
        ({"addressed_assumption_ids": ["unknown_assumption"]}, "assumption"),
    ],
)
def test_invalid_entity_target_window_magnitude_skill_or_assumption_never_becomes_numeric_eligible(
    context_task, overrides, missing_link
):
    verified = _verified(context_task, _payload(_chain(**overrides)))

    chain = verified.chains[0]
    assert chain.numeric_eligible is False
    assert missing_link in chain.missing_links


def test_duplicate_citations_are_deduplicated_and_incomplete_chains_remain_qualitative(context_task):
    raw = _chain(citations=[
        {"document_id": "doc_1", "exact_quote": "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04."},
        {"document_id": "doc_1", "exact_quote": "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04."},
    ], magnitude_kind="unknown", magnitude_value=None, missing_links=["explicit_magnitude"])
    verified = _verified(context_task, _payload(raw))

    assert len(verified.chains[0].citations) == 1
    assert verified.chains[0].numeric_eligible is False
    assert "magnitude" in verified.chains[0].missing_links


def test_round2_cannot_erase_or_replace_round1_chain(context_task):
    round1 = _verified(context_task, _payload(_chain()))
    raw = _payload(_chain(claim="Contradictory replacement text"))
    round2 = verify_round_result(
        context_task,
        raw,
        stage="round2",
        allowed_skill_ids=("s_window",),
        allowed_assumption_ids=("a_trend",),
    )
    conflicting = replace(round2.chains[0], chain_id=round1.chains[0].chain_id)
    round2 = replace(round2, chains=(conflicting,))

    merged = merge_verified_rounds(round1, round2)
    assert merged.chains[0] == round1.chains[0]
    assert "round2_chain_identity_conflict" in merged.rejected
