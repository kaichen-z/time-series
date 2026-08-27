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
            Document(
                "doc_counter",
                "Entity A sales will decrease by 10 percent from 2026-01-03 through 2026-01-04.",
            ),
            Document(
                "doc_stale",
                "Entity A sales will increase by 20 percent from 2025-01-03 through 2025-01-04.",
            ),
            Document(
                "doc_negated",
                "Entity A sales will not increase by 20 percent from 2026-01-03 through 2026-01-04.",
            ),
            Document(
                "doc_multiplier",
                "Entity A sales will increase 2 times from 2026-01-03 through 2026-01-04.",
            ),
            Document(
                "doc_absolute_down",
                "Entity A sales will decrease by 5 units from 2026-01-03 through 2026-01-04.",
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


def test_numeric_fields_must_be_bound_to_the_same_entity_target_evidence(context_task):
    verified = _verified(context_task, _payload(_chain(citations=[
        {
            "document_id": "doc_no_magnitude",
            "exact_quote": "Entity A sales will increase from 2026-01-03 through 2026-01-04.",
        },
        {
            "document_id": "doc_other",
            "exact_quote": "Entity B sales will increase by 20 percent from 2026-01-03 through 2026-01-04.",
        },
    ])))

    assert verified.chains[0].numeric_eligible is False
    assert "magnitude" in verified.chains[0].missing_links


def test_declared_magnitude_must_match_a_local_quote_and_stay_within_domain(context_task):
    mismatched = _verified(context_task, _payload(_chain(magnitude_value=0.3)))
    assert mismatched.chains[0].numeric_eligible is False
    assert "magnitude" in mismatched.chains[0].missing_links

    huge_document = Document(
        "doc_huge",
        "Entity A sales will increase by 1000000 units from 2026-01-03 through 2026-01-04.",
    )
    huge_task = replace(context_task, documents=(*context_task.documents, huge_document))
    huge = _verified(huge_task, _payload(_chain(
        magnitude_kind="absolute",
        magnitude_value=1_000_000.0,
        citations=[{"document_id": "doc_huge", "exact_quote": huge_document.content}],
    )))
    assert huge.chains[0].numeric_eligible is False
    assert "magnitude_value" in huge.chains[0].missing_links


def test_legacy_adapter_defense_in_depth_excludes_out_of_range_numeric_impacts(context_task):
    verified = _verified(context_task, _payload(_chain()))
    forged = replace(verified.chains[0], magnitude_value=21.0, numeric_eligible=True)
    card = FinalRetrievalCard(
        round1=verified,
        round2=None,
        chains=(forged,),
        selected_document_ids=("doc_1",),
        rejected=(),
        unresolved_contradictions=(),
        complete=True,
    )

    assert card.to_legacy_result().impacts == ()


def test_round2_submitted_round1_identity_is_rejected_before_canonicalization(context_task):
    round1 = _verified(context_task, _payload(_chain()))
    replacement = _payload(_chain(
        chain_id=round1.chains[0].chain_id,
        claim="Contradictory replacement text",
    ))

    round2 = verify_round_result(
        context_task,
        replacement,
        stage="round2",
        allowed_skill_ids=("s_window",),
        allowed_assumption_ids=("a_trend",),
        prior_round1=round1,
    )

    assert round2.chains == ()
    assert "round2_chain_identity_conflict" in round2.rejected
    assert merge_verified_rounds(round1, round2).chains == round1.chains


def test_malformed_citation_payload_is_rejected_without_raising(context_task):
    raw = _payload(_chain(citations=[{"document_id": [], "exact_quote": "anything"}]))

    verified = _verified(context_task, raw)

    assert verified.chains == ()
    assert verified.rejected


def test_round2_counterevidence_is_append_only_in_the_final_ledger(context_task):
    round1 = _verified(context_task, _payload(_chain()))
    counter = _chain(
        chain_id="counter_chain",
        claim="Counterevidence says the event decreases sales.",
        direction="down",
        magnitude_value=0.1,
        citations=[{
            "document_id": "doc_counter",
            "exact_quote": "Entity A sales will decrease by 10 percent from 2026-01-03 through 2026-01-04.",
        }],
        stance="challenges",
    )
    round2 = verify_round_result(
        context_task,
        {**_payload(), "counterevidence": [counter]},
        stage="round2",
        allowed_skill_ids=("s_window",),
        allowed_assumption_ids=("a_trend",),
    )

    merged = merge_verified_rounds(round1, round2)
    assert round2.counterevidence[0].chain_id in {item.chain_id for item in merged.chains}
    assert "doc_counter" in merged.selected_document_ids
    assert next(item for item in merged.chains if item.chain_id == round2.counterevidence[0].chain_id).numeric_eligible is False


def test_stable_chain_identity_includes_host_canonical_entity_and_target(context_task):
    document = Document(
        "doc_dual",
        "Entity A and Entity B sales will increase by 20 percent from 2026-01-03 through 2026-01-04.",
    )
    dual_documents = (document,)
    task_a = replace(context_task, documents=dual_documents)
    task_b = replace(
        context_task,
        numeric=replace(context_task.numeric, entity_name="Entity B"),
        documents=dual_documents,
    )
    raw = _payload(_chain(citations=[{"document_id": "doc_dual", "exact_quote": document.content}]))

    chain_a = _verified(task_a, raw).chains[0]
    chain_b = _verified(task_b, raw).chains[0]

    assert chain_a.numeric_eligible and chain_b.numeric_eligible
    assert chain_a.chain_id != chain_b.chain_id


def _legacy_impact(verified):
    card = FinalRetrievalCard(
        round1=verified,
        round2=None,
        chains=verified.chains,
        selected_document_ids=tuple(
            citation.document_id for chain in verified.chains for citation in chain.citations
        ),
        rejected=verified.rejected,
        unresolved_contradictions=(),
        complete=True,
    )
    return card.to_legacy_result().impacts


def test_declared_window_must_belong_to_the_task_future_timestamps(context_task):
    stale = _verified(context_task, _payload(_chain(
        start_timestamp="2025-01-03",
        end_timestamp="2025-01-04",
        citations=[{
            "document_id": "doc_stale",
            "exact_quote": "Entity A sales will increase by 20 percent from 2025-01-03 through 2025-01-04.",
        }],
    )))

    assert stale.chains[0].numeric_eligible is False
    assert "forecast_window" in stale.chains[0].missing_links


def test_relative_percentages_require_fractional_input_and_project_the_fraction(context_task):
    ambiguous = _verified(context_task, _payload(_chain(magnitude_value=20.0)))
    assert ambiguous.chains[0].numeric_eligible is False
    assert "magnitude_value" in ambiguous.chains[0].missing_links

    verified = _verified(context_task, _payload(_chain(magnitude_value=0.2)))
    assert _legacy_impact(verified)[0].adjustment_value == 0.2


def test_direction_canonicalizes_unsigned_relative_and_absolute_quantities_once(context_task):
    down_relative = _verified(context_task, _payload(_chain(
        direction="down",
        magnitude_value=0.1,
        citations=[{
            "document_id": "doc_counter",
            "exact_quote": "Entity A sales will decrease by 10 percent from 2026-01-03 through 2026-01-04.",
        }],
    )))
    assert down_relative.chains[0].numeric_eligible is True
    assert _legacy_impact(down_relative)[0].adjustment_value == -0.1

    down_absolute = _verified(context_task, _payload(_chain(
        direction="down",
        magnitude_kind="absolute",
        magnitude_value=5.0,
        citations=[{
            "document_id": "doc_absolute_down",
            "exact_quote": "Entity A sales will decrease by 5 units from 2026-01-03 through 2026-01-04.",
        }],
    )))
    assert down_absolute.chains[0].numeric_eligible is True
    impact = _legacy_impact(down_absolute)[0]
    assert impact.adjustment_kind == "add"
    assert impact.adjustment_value == -5.0


def test_negated_or_zero_effect_language_never_becomes_a_numeric_adjustment(context_task):
    negated = _verified(context_task, _payload(_chain(citations=[{
        "document_id": "doc_negated",
        "exact_quote": "Entity A sales will not increase by 20 percent from 2026-01-03 through 2026-01-04.",
    }])))
    assert negated.chains[0].numeric_eligible is False
    assert "direction" in negated.chains[0].missing_links

    zero = _verified(context_task, _payload(_chain(magnitude_value=0.0)))
    assert zero.chains[0].numeric_eligible is False
    assert "magnitude_value" in zero.chains[0].missing_links


def test_multiplier_uses_factor_input_and_projects_factor_minus_one(context_task):
    multiplier = _verified(context_task, _payload(_chain(
        magnitude_kind="multiplier",
        magnitude_value=2.0,
        citations=[{
            "document_id": "doc_multiplier",
            "exact_quote": "Entity A sales will increase 2 times from 2026-01-03 through 2026-01-04.",
        }],
    )))

    assert multiplier.chains[0].numeric_eligible is True
    impact = _legacy_impact(multiplier)[0]
    assert impact.adjustment_kind == "multiply"
    assert impact.adjustment_value == 1.0


def test_declared_window_must_exactly_equal_one_cited_inclusive_interval(context_task):
    document = Document(
        "doc_range",
        "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-05.",
    )
    ranged_task = replace(
        context_task,
        future_timestamps=("2026-01-03", "2026-01-04", "2026-01-05"),
        documents=(document,),
    )
    citation = [{"document_id": "doc_range", "exact_quote": document.content}]

    exact = _verified(ranged_task, _payload(_chain(
        end_timestamp="2026-01-05", citations=citation,
    )))
    shortened = _verified(ranged_task, _payload(_chain(
        end_timestamp="2026-01-03", citations=citation,
    )))
    swapped = _verified(ranged_task, _payload(_chain(
        start_timestamp="2026-01-05", end_timestamp="2026-01-03", citations=citation,
    )))

    assert exact.chains[0].numeric_eligible is True
    assert shortened.chains[0].numeric_eligible is False
    assert "forecast_window" in shortened.chains[0].missing_links
    assert swapped.chains == ()


def test_multiple_cited_ranges_are_ambiguous_even_when_one_matches(context_task):
    document = Document(
        "doc_ranges",
        "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-05, "
        "then recover from 2026-01-06 through 2026-01-07.",
    )
    ranged_task = replace(
        context_task,
        future_timestamps=("2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"),
        documents=(document,),
    )
    verified = _verified(ranged_task, _payload(_chain(
        end_timestamp="2026-01-05",
        citations=[{"document_id": "doc_ranges", "exact_quote": document.content}],
    )))

    assert verified.chains[0].numeric_eligible is False
    assert "forecast_window" in verified.chains[0].missing_links


@pytest.mark.parametrize("text", [
    "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04, but the promotion will not occur.",
    "Entity A sales will increase by 20 percent from 2026-01-03 through 2026-01-04; the event was cancelled.",
])
def test_cancellation_or_nonoccurrence_in_an_anchored_span_blocks_numeric_eligibility(context_task, text):
    task = replace(context_task, documents=(Document("doc_cancelled", text),))
    verified = _verified(task, _payload(_chain(citations=[{
        "document_id": "doc_cancelled", "exact_quote": text,
    }])))

    assert verified.chains[0].numeric_eligible is False
    assert "causal_status" in verified.chains[0].missing_links


@pytest.mark.parametrize("token", ["2x", "2×"])
def test_compact_multiplier_tokens_are_normalized_as_factors(context_task, token):
    text = f"Entity A sales will increase {token} from 2026-01-03 through 2026-01-04."
    task = replace(context_task, documents=(Document("doc_multiplier_compact", text),))
    verified = _verified(task, _payload(_chain(
        magnitude_kind="multiplier",
        magnitude_value=2.0,
        citations=[{"document_id": "doc_multiplier_compact", "exact_quote": text}],
    )))

    assert verified.chains[0].numeric_eligible is True
    assert _legacy_impact(verified)[0].adjustment_value == 1.0


def test_malformed_compact_multiplier_token_is_not_numeric_evidence(context_task):
    text = "Entity A sales will increase 2xboost from 2026-01-03 through 2026-01-04."
    task = replace(context_task, documents=(Document("doc_multiplier_bad", text),))
    verified = _verified(task, _payload(_chain(
        magnitude_kind="multiplier",
        magnitude_value=2.0,
        citations=[{"document_id": "doc_multiplier_bad", "exact_quote": text}],
    )))

    assert verified.chains[0].numeric_eligible is False
    assert "magnitude" in verified.chains[0].missing_links
