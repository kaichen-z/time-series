from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from common.data import Task
from common.llm import FakeLLMClient
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import DecisionAgent, DecisionCandidate
from evolving_loop.harness import EvolvingForecastHarness, HarnessRuntimeConfig
from evolving_loop.morphology_adapter import MorphologyAdapter
from evolving_loop.retrieval_agent.policy import RetrievalGenome
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
    retrieval = _agent(["not json"])
    harness = _harness(retrieval, [_decision(), _decision()])

    result = harness.run(_task())

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


def test_non_runtime_llm_failure_is_sanitized_before_incomplete_round2() -> None:
    genome = replace(RetrievalGenome.seed(), second_round_trigger="on_incomplete_chain")
    llm = _FailingThenRespondingLLM(_round())
    retrieval = TwoStageRetrievalAgent(
        llm,
        genome,
        RetrievalSkillLibrary("unused-two-stage-skills.json", persist=False),
    )
    harness = _harness(retrieval, [_decision(), _decision()])

    result = harness.run(_task())

    assert len(llm.calls) == 2
    second = json.loads(llm.calls[1]["messages"][0]["content"])
    encoded = json.dumps(second, sort_keys=True)
    assert '"candidate_id":' not in encoded
    assert '"forecast":' not in encoded
    assert '"hindcast_srmse":' not in encoded
    assert "invalid_round1_response" in result.retrieval.rejected


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

    result = retrieval.run_round1(_task())

    prompt = json.loads(retrieval.llm.calls[0]["messages"][0]["content"])
    assert len(prompt["documents"]) == 1
    assert len(result.chains) + len(result.counterevidence) == 1
    assert len(result.chains[0].citations) == 1


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


def test_retrieval_gap_contract_is_strict_and_typed() -> None:
    assert RetrievalGap.from_payload(_gap()).priority == "high"
    with pytest.raises(ValueError, match="gap type"):
        RetrievalGap.from_payload(_gap(gap_type="invented"))
