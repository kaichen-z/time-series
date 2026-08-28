from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from common.data import Task
from common.llm import FakeLLMClient, TransientLLMError
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import DecisionAgent, DecisionCandidate
from evolving_loop.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_loop.morphology_adapter import MorphologyAdapter
from evolving_loop.retrieval_agent.policy import (
    ROUND1_STRATEGIES,
    ROUND2_STRATEGIES,
    RetrievalGenome,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.schemas import (
    RetrievalAssumption,
    RetrievalContractError,
    RetrievalGap,
)
from evolving_loop.retrieval_agent.agent import RetrievalResult
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent


def _task() -> ContextTask:
    return ContextTask(
        numeric=Task(
            task_id="two_stage",
            history_values=tuple(float(value) for value in range(1, 21)),
            future_values=(26.0, 27.0),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Entity A",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=tuple(f"2026-01-{index:02d}" for index in range(1, 21)),
        future_timestamps=("2026-01-21", "2026-01-22"),
        documents=(
            Document(
                "doc_1",
                "Entity A sales will increase by 20 percent from 2026-01-21 through 2026-01-22. "
                "The scheduled promotion is a documented future driver for Entity A sales.",
                role="supporting",
                subtype="future_event",
            ),
            Document(
                "doc_2",
                "Entity A sales will decrease by 10 percent from 2026-01-21 through 2026-01-22. "
                "The scheduled supply outage is a documented future driver for Entity A sales.",
                role="supporting",
                subtype="counterevidence",
            ),
            Document("doc_noise", "The office carpet is blue.", role="distractor"),
        ),
        gt_evidence=("A future promotion overlaps the horizon.",),
    )


def _chain(
    *,
    chain_id: str = "round1_chain",
    document_id: str = "doc_1",
    direction: str = "up",
    magnitude_value: float = 0.2,
    addressed_assumption_ids: list[str] | None = None,
    numeric_eligible: bool = True,
    missing_links: list[str] | None = None,
    citations: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    quote = (
        "Entity A sales will increase by 20 percent from 2026-01-21 through 2026-01-22."
        if document_id == "doc_1"
        else "Entity A sales will decrease by 10 percent from 2026-01-21 through 2026-01-22."
    )
    return {
        "chain_id": chain_id,
        "claim": "A scheduled event changes sales during the forecast window.",
        "entity_match": True,
        "target_match": True,
        "temporal_relation": "overlaps_future",
        "mechanism": "future_driver",
        "direction": direction,
        "magnitude_kind": "relative",
        "magnitude_value": magnitude_value,
        "start_timestamp": "2026-01-21",
        "end_timestamp": "2026-01-22",
        "citations": citations or [{"document_id": document_id, "exact_quote": quote}],
        "missing_links": missing_links or [],
        "used_skill_ids": [],
        "addressed_assumption_ids": addressed_assumption_ids or [],
        "stance": "challenges" if addressed_assumption_ids else "supports",
        "numeric_eligible": numeric_eligible,
    }


def _round(*chains: dict[str, object], sufficient: bool = True) -> str:
    return json.dumps(
        {
            "evidence_chains": list(chains),
            "counterevidence": [],
            "missing_information": [] if sufficient else ["missing_magnitude"],
            "sufficient": sufficient,
        }
    )


def _decision(*, gaps: list[dict[str, object]] | None = None, request: bool = False) -> str:
    return json.dumps(
        {
            "selected_candidate_id": "numeric",
            "supporting_document_ids": [],
            "rationale": "Preserve the numeric host while checking a named assumption gap.",
            "request_more_retrieval": request,
            "gaps": gaps or [],
            "used_skill_names": [],
        }
    )


def _gap(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "assumption_id": "a_trend",
        "gap_type": "continuation_or_reversal",
        "missing_information": "Evidence of continuation or reversal",
        "priority": "high",
    }
    payload.update(overrides)
    return payload


class _Coding:
    def __init__(self) -> None:
        program = SimpleNamespace(
            name="numeric",
            assumption="The local trend persists.",
            failure_condition="A new future regime begins.",
            source="generated",
        )
        self.candidate = SimpleNamespace(
            program=program,
            forecast=(21.0, 22.0),
            hindcast_smae=0.1,
            hindcast_srmse=0.2,
            fold_smae=(0.1, 0.1),
            fold_srmse=(0.2, 0.2),
            fold_errors=(),
        )

    def run_task(self, task: Task, *, allow_skill_writes: bool):
        assert task.future_values == ()
        del allow_skill_writes
        return SimpleNamespace(candidates=(self.candidate,), selected=self.candidate)


class _NumericalMorphology:
    def run(self, task: Task):
        assert task.future_values == ()
        return SimpleNamespace(
            assumptions=(
                SimpleNamespace(
                    assumption_id="a_trend",
                    kind="trend_persistence",
                    claim="The historical trend continues into the horizon.",
                    failure_condition="A future event reverses the trend.",
                    candidate_id="must_not_cross_boundary",
                    forecast=(999.0,),
                    hindcast_srmse=0.0,
                ),
            )
        )


class _FailingMorphologyProvider:
    def assumptions(self, task: ContextTask):
        del task
        raise RuntimeError("numerical morphology unavailable")


class _FailingThenRespondingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0):
        from common.llm import LLMResponse

        self.calls.append(
            {"system": system, "messages": messages, "temperature": temperature}
        )
        if len(self.calls) == 1:
            raise OSError("candidate_id forecast hindcast_srmse must not cross stages")
        return LLMResponse(self.response)


class _TransientRetrievalLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **_kwargs):
        self.calls += 1
        raise TransientLLMError("temporary inner retrieval outage")


def _agent(
    responses: list[str],
    *,
    genome: RetrievalGenome | None = None,
) -> TwoStageRetrievalAgent:
    return TwoStageRetrievalAgent(
        FakeLLMClient(responses),
        genome or RetrievalGenome.seed(),
        RetrievalSkillLibrary("unused-two-stage-skills.json", persist=False),
    )


def _harness(
    retrieval: TwoStageRetrievalAgent,
    decision_responses: list[str],
    *,
    morphology=None,
) -> EvolvingForecastHarness:
    return EvolvingForecastHarness(
        _Coding(),
        retrieval,
        DecisionAgent(FakeLLMClient(decision_responses)),
        runtime=HarnessRuntimeConfig(retrieval_mode="two_stage"),
        morphology=morphology or MorphologyAdapter(_NumericalMorphology()),
    )


def test_two_stage_prompt_boundaries_and_decision_gap_projection() -> None:
    retrieval = _agent(
        [
            _round(_chain()),
            _round(
                _chain(
                    chain_id="round2_chain",
                    document_id="doc_2",
                    direction="down",
                    magnitude_value=0.1,
                    addressed_assumption_ids=["a_trend"],
                )
            ),
        ]
    )
    gap = _gap()
    harness = _harness(
        retrieval,
        [_decision(gaps=[gap], request=True), _decision()],
    )

    result = harness.run(_task())

    first = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
    second = json.loads(retrieval.llm.calls[1]["messages"][0]["content"])
    provisional = json.loads(harness.decision.llm.calls[0]["messages"][0]["content"])
    assert "assumptions" not in first
    assert "round1" not in first
    assert "forecast" in provisional["candidates"][0]
    assert "hindcast_srmse" in provisional["candidates"][0]
    assert set(second["assumptions"][0]) == {
        "assumption_id", "kind", "claim", "failure_condition"
    }
    assert second["gaps"] == [gap]
    encoded_first = json.dumps(first, sort_keys=True)
    encoded_second = json.dumps(second, sort_keys=True)
    assert result.retrieval.evidence
    assert result.retrieval_card is not None
    assert result.retrieval_card.round2 is not None
    assert [item.to_payload() for item in result.retrieval_card.gaps] == [gap]
    assert result.retrieval_card.to_payload()["gaps"] == [gap]
    for forbidden in (
        "candidate_id", "forecast", "hindcast_smae", "hindcast_srmse", "code",
        "future_values", "gt_evidence", "role", "subtype",
    ):
        assert f'"{forbidden}":' not in encoded_first
        assert f'"{forbidden}":' not in encoded_second


def test_morphology_adapter_maps_only_the_sanitized_card_fields() -> None:
    assumptions = MorphologyAdapter(_NumericalMorphology()).assumptions(_task())

    assert [item.to_payload() for item in assumptions] == [
        {
            "assumption_id": "a_trend",
            "kind": "trend_persistence",
            "claim": "The historical trend continues into the horizon.",
            "failure_condition": "A future event reverses the trend.",
        }
    ]


def test_two_stage_construction_fails_before_run_without_a_valid_provider() -> None:
    retrieval = _agent([_round(_chain())])

    with pytest.raises(ValueError, match="MorphologyProvider"):
        EvolvingForecastHarness(
            _Coding(),
            retrieval,
            DecisionAgent(FakeLLMClient([])),
            runtime=HarnessRuntimeConfig(retrieval_mode="two_stage"),
        )
    with pytest.raises(ValueError, match="MorphologyProvider"):
        EvolvingForecastHarness(
            _Coding(),
            retrieval,
            DecisionAgent(FakeLLMClient([])),
            runtime=HarnessRuntimeConfig(retrieval_mode="two_stage"),
            morphology=SimpleNamespace(assumptions=()),
        )
    with pytest.raises(ValueError, match="retrieval_mode"):
        HarnessRuntimeConfig(retrieval_mode="repeated_free_form")
    with pytest.raises(RetrievalContractError, match=r"run\(task\)"):
        MorphologyAdapter(SimpleNamespace())


def test_round1_parse_failure_falls_back_to_numeric_candidates() -> None:
    genome = replace(RetrievalGenome.seed(), second_round_trigger="on_incomplete_chain")
    retrieval = _agent(["not json", _round(_chain())], genome=genome)
    harness = _harness(retrieval, [_decision(), _decision()])

    result = harness.run(_task())

    assert len(retrieval.llm.calls) == 1
    assert [item.candidate_id for item in result.candidates] == ["numeric"]
    assert result.retrieval.evidence == ()
    assert result.forecast == (21.0, 22.0)
    assert "invalid_round1_response" in result.retrieval.rejected


def test_named_gap_mode_skips_round2_without_a_valid_gap() -> None:
    retrieval = _agent([_round(_chain())])
    harness = _harness(retrieval, [_decision(request=True), _decision()])

    result = harness.run(_task())

    assert len(retrieval.llm.calls) == 1
    assert result.decision.requested_more_retrieval is False
    assert result.retrieval_card is not None
    assert result.retrieval_card.round2 is None


def test_invalid_gap_does_not_invalidate_the_provisional_selection() -> None:
    retrieval = _agent([_round(_chain())])
    harness = _harness(
        retrieval,
        [_decision(gaps=[_gap(candidate_id="numeric")], request=True), _decision()],
    )

    result = harness.run(_task())

    provisional = json.loads(harness.decision.llm.calls[0]["messages"][0]["content"])
    assert provisional["candidates"][0]["candidate_id"] == "numeric"
    assert len(retrieval.llm.calls) == 1
    assert result.forecast == (21.0, 22.0)


def test_round2_failure_preserves_verified_round1_and_numeric_fallback() -> None:
    retrieval = _agent([_round(_chain()), "not json"])
    harness = _harness(
        retrieval,
        [_decision(gaps=[_gap()], request=True), _decision()],
    )

    result = harness.run(_task())

    assert len(retrieval.llm.calls) == 2
    assert {item.candidate_id for item in result.candidates} >= {"numeric"}
    assert {item.document_id for item in result.retrieval.evidence} == {"doc_1"}
    assert "invalid_round2_response" in result.retrieval.rejected
    assert result.retrieval_card is not None
    assert result.retrieval_card.round1.chains


def test_morphology_runtime_failure_is_recorded_and_skips_round2() -> None:
    retrieval = _agent([_round(_chain())])
    harness = _harness(
        retrieval,
        [_decision(gaps=[_gap()], request=True), _decision()],
        morphology=_FailingMorphologyProvider(),
    )

    result = harness.run(_task())

    assert len(retrieval.llm.calls) == 1
    assert {item.candidate_id for item in result.candidates} >= {"numeric"}
    assert result.retrieval.evidence
    assert "morphology_provider_failed:RuntimeError" in result.retrieval.rejected


def test_non_runtime_round1_failure_skips_incomplete_round2() -> None:
    genome = replace(RetrievalGenome.seed(), second_round_trigger="on_incomplete_chain")
    llm = _FailingThenRespondingLLM(_round())
    retrieval = TwoStageRetrievalAgent(
        llm,
        genome,
        RetrievalSkillLibrary("unused-two-stage-skills.json", persist=False),
    )
    harness = _harness(retrieval, [_decision(), _decision()])

    result = harness.run(_task())

    assert len(llm.calls) == 1
    assert [item.candidate_id for item in result.candidates] == ["numeric"]
    assert "invalid_round1_response" in result.retrieval.rejected


def test_two_stage_agent_reraises_transient_inner_llm_failures() -> None:
    llm = _TransientRetrievalLLM()
    retrieval = TwoStageRetrievalAgent(
        llm,
        RetrievalGenome.seed(),
        RetrievalSkillLibrary("unused-two-stage-skills.json", persist=False),
    )

    with pytest.raises(TransientLLMError, match="temporary inner retrieval outage"):
        retrieval.run_round1(_task())

    assert llm.calls == 1


def test_two_stage_harness_reraises_transient_morphology_failures() -> None:
    class Provider:
        def assumptions(self, task):
            del task
            raise TransientLLMError("temporary morphology outage")

    retrieval = _agent([_round(_chain())])
    harness = _harness(retrieval, [_decision(), _decision()], morphology=Provider())

    with pytest.raises(TransientLLMError, match="temporary morphology outage"):
        harness.run(_task())

    assert retrieval.llm.calls == []


def test_non_runtime_morphology_failure_is_recorded_and_skips_round2() -> None:
    class Provider:
        def assumptions(self, task):
            del task
            raise OSError("morphology card could not be read")

    retrieval = _agent([_round(_chain())])
    harness = _harness(
        retrieval,
        [_decision(gaps=[_gap()], request=True), _decision()],
        morphology=Provider(),
    )

    result = harness.run(_task())

    assert len(retrieval.llm.calls) == 1
    assert "morphology_provider_failed:OSError" in result.retrieval.rejected


def test_incomplete_chain_trigger_runs_round2_without_candidate_derived_gaps() -> None:
    genome = replace(RetrievalGenome.seed(), second_round_trigger="on_incomplete_chain")
    incomplete = _chain(
        magnitude_value=0.2,
        numeric_eligible=False,
        missing_links=["missing_mechanism"],
    )
    retrieval = _agent(
        [
            _round(incomplete, sufficient=False),
            _round(
                _chain(
                    chain_id="round2_chain",
                    document_id="doc_2",
                    direction="down",
                    magnitude_value=0.1,
                    addressed_assumption_ids=["a_trend"],
                )
            ),
        ],
        genome=genome,
    )
    harness = _harness(retrieval, [_decision(), _decision()])

    harness.run(_task())

    second = json.loads(retrieval.llm.calls[1]["messages"][0]["content"])
    assert second["gaps"] == []
    assert "candidates" not in second


def test_two_stage_agent_enforces_fixed_document_chain_and_citation_budgets() -> None:
    genome = replace(
        RetrievalGenome.seed(),
        max_selected_documents=1,
        max_evidence_chains=1,
        max_citations_per_chain=1,
    )
    citations = [
        {
            "document_id": "doc_1",
            "exact_quote": "Entity A sales will increase by 20 percent from 2026-01-21 through 2026-01-22.",
        },
        {
            "document_id": "doc_1",
            "exact_quote": "The scheduled promotion is a documented future driver for Entity A sales.",
        },
    ]
    retrieval = _agent(
        [
            _round(
                _chain(citations=citations),
                _chain(chain_id="over_budget"),
            )
        ],
        genome=genome,
    )
    task = _task()
    budget_task = replace(
        task,
        documents=(task.documents[0], task.documents[2]),
    )

    result = retrieval.run_round1(budget_task)

    prompt = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
    assert len(prompt["documents"]) == 1
    assert len(result.chains) + len(result.counterevidence) == 1
    assert len(result.chains[0].citations) == 1


def test_document_budget_selection_is_content_ranked_and_permutation_invariant() -> None:
    genome = replace(RetrievalGenome.seed(), max_selected_documents=1)
    task = _task()
    permutations = (
        task.documents,
        tuple(reversed(task.documents)),
        (*task.documents[2:], *task.documents[:2]),
    )
    selected_payloads = []

    for documents in permutations:
        retrieval = _agent(["not json"], genome=genome)
        retrieval.run_round1(replace(task, documents=documents))
        selected_payloads.append(
            json.loads(retrieval.llm.calls[0]["messages"][0]["content"])["documents"]
        )

    assert selected_payloads[0] == selected_payloads[1] == selected_payloads[2]
    assert len(selected_payloads[0]) == 1
    assert "Entity A sales" in selected_payloads[0][0]["content"]


def test_document_ids_do_not_change_content_selection() -> None:
    genome = replace(RetrievalGenome.seed(), max_selected_documents=1)
    relevant = (
        "Entity A sales will increase by 20 percent from 2026-01-21 through "
        "2026-01-22 because a scheduled promotion begins."
    )
    distractor = "The office carpet is blue."
    selected = []

    for relevant_id, distractor_id in (("zzz", "aaa"), ("aaa", "zzz")):
        task = replace(
            _task(),
            documents=(
                Document(distractor_id, distractor),
                Document(relevant_id, relevant),
            ),
        )
        retrieval = _agent(["not json"], genome=genome)
        retrieval.run_round1(task)
        selected.append(
            json.loads(retrieval.llm.calls[0]["messages"][0]["content"])["documents"]
        )

    assert [row[0]["content"] for row in selected] == [relevant, relevant]
    assert [row[0]["document_id"] for row in selected] == ["zzz", "aaa"]


def test_duplicate_content_groups_are_selected_all_or_none_within_budget() -> None:
    genome = replace(RetrievalGenome.seed(), max_selected_documents=2)
    duplicate = (
        "Entity A sales will increase by 20 percent from 2026-01-21 through "
        "2026-01-22 because a scheduled promotion begins."
    )
    unique = (
        "Entity A sales will decrease by 10 percent from 2026-01-21 through "
        "2026-01-22 because a supply outage begins."
    )
    task = replace(
        _task(),
        documents=(
            Document("duplicate_c", duplicate),
            Document("duplicate_a", duplicate),
            Document("duplicate_b", duplicate),
            Document("unique", unique),
        ),
    )
    retrieval = _agent(["not json"], genome=genome)

    retrieval.run_round1(task)

    documents = json.loads(
        retrieval.llm.calls[0]["messages"][0]["content"]
    )["documents"]
    assert documents == [{"document_id": "unique", "content": unique}]


def test_unselected_documents_cannot_authorize_numeric_citations() -> None:
    task = _task()
    genome = replace(RetrievalGenome.seed(), max_selected_documents=1)

    class UnselectedCitationLLM:
        def __init__(self) -> None:
            self.selected_ids: tuple[str, ...] = ()
            self.cited_id = ""

        def complete(self, *, messages, **_kwargs):
            from common.llm import LLMResponse

            payload = json.loads(messages[0]["content"])
            self.selected_ids = tuple(
                item["document_id"] for item in payload["documents"]
            )
            numeric_documents = {
                document.document_id: document
                for document in task.documents[:2]
            }
            self.cited_id = next(
                document_id
                for document_id in numeric_documents
                if document_id not in self.selected_ids
            )
            document = numeric_documents[self.cited_id]
            chain = _chain(
                document_id=self.cited_id,
                direction="up" if self.cited_id == "doc_1" else "down",
                magnitude_value=0.2 if self.cited_id == "doc_1" else 0.1,
                citations=[
                    {
                        "document_id": self.cited_id,
                        "exact_quote": document.content.split(". ", 1)[0] + ".",
                    }
                ],
            )
            return LLMResponse(_round(chain))

    llm = UnselectedCitationLLM()
    retrieval = TwoStageRetrievalAgent(
        llm,
        genome,
        RetrievalSkillLibrary("unused-two-stage-skills.json", persist=False),
    )

    result = retrieval.run_round1(task)

    assert len(llm.selected_ids) == genome.max_selected_documents
    assert llm.cited_id not in llm.selected_ids
    assert result.chains[0].numeric_eligible is False
    assert result.chains[0].citations == ()
    assert result.quote_attempt_count == 1
    assert result.valid_quote_count == 0
    assert f"unselected_document:{llm.cited_id}" in result.rejected


_ROUND1_QUERY_PLANS = {
    "timeline_first": {
        "ordered_objectives": [
            "Anchor evidence to the forecast window before composing causal claims.",
            "Order event evidence chronologically and flag ended events.",
            "Search explicitly for cancellations, postponements, and reversals.",
        ],
        "selection_features": [
            "forecast_window_overlap",
            "explicit_event_dates",
            "entity_target_phrase",
        ],
    },
    "entity_first": {
        "ordered_objectives": [
            "Resolve the exact entity and target phrase before considering an event.",
            "Reject evidence about neighboring entities or similarly named targets.",
            "Then verify mechanism, magnitude, and forecast-window coverage.",
        ],
        "selection_features": [
            "entity_target_phrase",
            "canonical_name_boundaries",
            "forecast_window_overlap",
        ],
    },
    "contrastive": {
        "ordered_objectives": [
            "Build separate support and challenge hypotheses for each material event.",
            "Seek matched counterevidence before declaring the ledger sufficient.",
            "Retain unresolved contradictions rather than averaging them away.",
        ],
        "selection_features": [
            "support_challenge_pairing",
            "cancellation_or_reversal",
            "entity_target_phrase",
        ],
    },
}

_ROUND2_QUERY_PLANS = {
    "counterevidence_first": {
        "ordered_objectives": [
            "Search first for evidence that invalidates or limits each named assumption.",
            "Prioritize cancellation, postponement, reversal, containment, or recovery.",
            "Report unresolved assumptions when no exact counterevidence exists.",
        ],
        "selection_features": [
            "assumption_failure_condition",
            "counterevidence",
            "named_gap_priority",
        ],
    },
    "gap_first": {
        "ordered_objectives": [
            "Process named gaps in host-provided priority order.",
            "Fill only the missing link stated for each gap.",
            "Keep evidence attached to its addressed assumption ID.",
        ],
        "selection_features": [
            "named_gap_priority",
            "missing_information",
            "assumption_id",
        ],
    },
    "causal_chain_first": {
        "ordered_objectives": [
            "Complete entity, target, mechanism, window, and magnitude links in that order.",
            "Prefer exact evidence that closes an incomplete verified Round 1 chain.",
            "Preserve contradictions and do not overwrite Round 1 evidence.",
        ],
        "selection_features": [
            "incomplete_chain_fields",
            "verified_round1",
            "causal_link_completeness",
        ],
    },
}


@pytest.mark.parametrize("strategy", sorted(ROUND1_STRATEGIES))
def test_every_round1_strategy_projects_concrete_host_query_objectives(
    strategy: str,
) -> None:
    genome = replace(RetrievalGenome.seed(), round1_strategy=strategy)
    retrieval = _agent(["not json"], genome=genome)

    retrieval.run_round1(_task())

    payload = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
    assert payload["query_plan"] == _ROUND1_QUERY_PLANS[strategy]


@pytest.mark.parametrize("strategy", sorted(ROUND2_STRATEGIES))
def test_every_round2_strategy_projects_concrete_host_query_objectives(
    strategy: str,
) -> None:
    round1 = _agent([_round(_chain())]).run_round1(_task())
    genome = replace(RetrievalGenome.seed(), round2_strategy=strategy)
    retrieval = _agent(["not json"], genome=genome)

    retrieval.run_round2(
        _task(),
        round1,
        (RetrievalGap.from_payload(_gap()),),
        (
            RetrievalAssumption(
                "a_trend",
                "trend_persistence",
                "The historical trend continues.",
                "A future event reverses the trend.",
            ),
        ),
    )

    payload = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
    assert payload["query_plan"] == _ROUND2_QUERY_PLANS[strategy]


def test_strategy_only_genomes_change_only_the_owned_stage_wire_behavior() -> None:
    gap = (RetrievalGap.from_payload(_gap()),)
    assumptions = (
        RetrievalAssumption(
            "a_trend",
            "trend_persistence",
            "The historical trend continues.",
            "A future event reverses the trend.",
        ),
    )

    def calls(genome: RetrievalGenome) -> tuple[dict[str, object], dict[str, object]]:
        retrieval = _agent([_round(_chain()), "not json"], genome=genome)
        round1 = retrieval.run_round1(_task())
        retrieval.run_round2(_task(), round1, gap, assumptions)
        return retrieval.llm.calls[0], retrieval.llm.calls[1]

    parent = RetrievalGenome.seed()
    parent_round1, parent_round2 = calls(parent)
    round1_child = replace(parent, round1_strategy="contrastive")
    child_round1, child_round2 = calls(round1_child)
    round2_child = replace(parent, round2_strategy="gap_first")
    other_round1, other_round2 = calls(round2_child)

    assert child_round1 != parent_round1
    assert child_round2 == parent_round2
    assert other_round1 == parent_round1
    assert other_round2 != parent_round2


def test_all_matching_stage_skills_see_materialized_generator_selectors(tmp_path) -> None:
    def skill(skill_id: str) -> dict[str, object]:
        return {
            "skill_id": skill_id,
            "version": 1,
            "parent_version": None,
            "stage": "round2",
            "status": "accepted",
            "name": skill_id,
            "description": "Investigate the named trend gap.",
            "applicability": {
                "assumption_kinds": ["trend_persistence"],
                "gap_types": ["continuation_or_reversal"],
                "temporal_relations": [],
            },
            "query_steps": ["Search for reversal evidence."],
            "required_chain_fields": ["entity", "target"],
            "counterevidence_rule": "Search for continuation evidence.",
            "failure_conditions": ["The evidence concerns another entity."],
            "validated_task_ids": ["train_1", "train_2", "train_3"],
            "validated_entities": ["north", "south"],
            "validation_smae_gain": 0.1,
            "validation_srmse_gain": 0.1,
            "merged_from_skill_ids": [],
            "quarantine_reason": None,
        }

    genome = replace(
        RetrievalGenome.seed(),
        version="v001",
        parent="v000",
        active_skill_ids=("gap_alpha", "gap_beta"),
    )
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        genome,
        skills=(skill("gap_alpha"), skill("gap_beta")),
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
    agent = TwoStageRetrievalAgent(
        FakeLLMClient([]),
        genome,
        RetrievalSkillLibrary.from_release(release.path),
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    gap = RetrievalGap.from_payload(_gap())

    selected = agent._skills(
        "round2",
        assumptions=(item for item in (assumption,)),
        gaps=(item for item in (gap,)),
    )

    assert tuple(item.skill_id for item in selected) == ("gap_alpha", "gap_beta")


@pytest.mark.parametrize(
    "gap",
    [
        _gap(assumption_id="unknown"),
        _gap(candidate_id="numeric"),
    ],
)
def test_invalid_gap_preserves_provisional_selection_but_disables_request(gap) -> None:
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumptions = (
        RetrievalAssumption(
            "a_trend",
            "trend_persistence",
            "The historical trend continues.",
            "A future event reverses the trend.",
        ),
    )
    agent = DecisionAgent(FakeLLMClient([_decision(gaps=[gap], request=True)]))

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=assumptions,
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is False
    assert result.gaps == ()
    assert result.rejection_reason is not None
    assert "invalid_retrieval_gaps" in result.rejection_reason


@pytest.mark.parametrize("raw_request", ["false", 1, None])
def test_decision_requires_exact_boolean_retrieval_request(raw_request) -> None:
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    agent = DecisionAgent(
        FakeLLMClient([_decision(gaps=[_gap()], request=raw_request)])
    )

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=(assumption,),
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is False
    assert result.rejection_reason is not None
    assert "invalid_retrieval_request" in result.rejection_reason


@pytest.mark.parametrize("field", ["forecast", "source_code"])
def test_decision_rejects_top_level_schema_drift_without_losing_selection(field) -> None:
    payload = json.loads(_decision(gaps=[_gap()], request=True))
    payload[field] = [999.0] if field == "forecast" else "unsafe candidate code"
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    agent = DecisionAgent(FakeLLMClient([json.dumps(payload)]))

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=(assumption,),
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is False
    assert result.rejection_reason is not None
    assert "forbidden_decision_fields" in result.rejection_reason


@pytest.mark.parametrize(
    ("field", "nested_value"),
    [
        ("selected_candidate_id", {"forecast": [999.0]}),
        ("selected_candidate_id", ["numeric"]),
        ("rationale", {"source_code": "unsafe candidate code"}),
        ("rationale", ["unsafe nested rationale"]),
        ("supporting_document_ids", [{"forecast": [999.0]}]),
        ("supporting_document_ids", [["doc_1"]]),
        ("used_skill_names", [{"source_code": "unsafe candidate code"}]),
        ("used_skill_names", [["unsafe_skill"]]),
    ],
)
def test_decision_rejects_nested_schema_drift_in_text_and_list_fields(
    field: str,
    nested_value: object,
) -> None:
    payload = json.loads(_decision(gaps=[_gap()], request=True))
    payload[field] = nested_value
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    agent = DecisionAgent(FakeLLMClient([json.dumps(payload)]))

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=(assumption,),
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is False
    assert result.gaps == ()
    assert result.rejection_reason is not None
    assert "invalid_decision_response_schema" in result.rejection_reason


@pytest.mark.parametrize(
    "malformed_gaps",
    [
        {"a_trend": _gap()},
        [[_gap()]],
        [_gap(missing_information={"forecast": [999.0]})],
        [_gap(priority=["high"])],
    ],
)
def test_decision_rejects_malformed_nested_gap_schema(malformed_gaps: object) -> None:
    payload = json.loads(_decision(request=True))
    payload["gaps"] = malformed_gaps
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    agent = DecisionAgent(FakeLLMClient([json.dumps(payload)]))

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=(assumption,),
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is False
    assert result.gaps == ()
    assert result.rejection_reason is not None
    assert "invalid_retrieval_gaps" in result.rejection_reason


def test_decision_accepts_complete_valid_gap_interface_response() -> None:
    candidate = DecisionCandidate(
        candidate_id="numeric",
        forecast=(21.0, 22.0),
        assumption="The local trend persists.",
        failure_condition="A future event reverses the trend.",
        hindcast_smae=0.1,
        hindcast_srmse=0.2,
    )
    assumption = RetrievalAssumption(
        "a_trend",
        "trend_persistence",
        "The historical trend continues.",
        "A future event reverses the trend.",
    )
    agent = DecisionAgent(
        FakeLLMClient([_decision(gaps=[_gap()], request=True)])
    )

    result = agent.run(
        (candidate,),
        RetrievalResult("", (), (), (), False, ()),
        assumptions=(assumption,),
    )

    assert result.selected == candidate
    assert result.requested_more_retrieval is True
    assert tuple(gap.assumption_id for gap in result.gaps) == ("a_trend",)
    assert result.rationale == (
        "Preserve the numeric host while checking a named assumption gap."
    )
    assert result.rejection_reason is None


def test_retrieval_gap_contract_is_strict_and_typed() -> None:
    assert RetrievalGap.from_payload(_gap()).priority == "high"
    with pytest.raises(ValueError, match="gap type"):
        RetrievalGap.from_payload(_gap(gap_type="invented"))
