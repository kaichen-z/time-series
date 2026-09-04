from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from common.data import Task as ContextNumericTask
from common.llm import FakeLLMClient
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_retrieval_evolution import PackageRetrievalEvaluator
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_registry import (
    FrozenNumericalPackageRegistry,
    PackageRegistryError,
)
from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.retrieval_agent.quality import score_retrieval_card_quality
from evolving_loop.retrieval_agent.evolution import build_inference_cache_key
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.retrieval_agent.schemas import (
    EvidenceChain,
    EvidenceCitation,
    FinalRetrievalCard,
    RetrievalRoundResult,
)
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyObservation,
    MorphologyToolCall,
)
from numerical_agent.evolution.numerical_handoff import safe_retrieval_projection
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)


_HISTORY = (1.0, 2.0, 3.0) * 12


def _task() -> ContextTask:
    return ContextTask(
        numeric=ContextNumericTask(
            task_id="champion-runtime",
            history_values=_HISTORY,
            future_values=(1.0, 2.0),
            prediction_length=2,
            frequency="D",
            seasonal_period=None,
            entity_name="Entity A",
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=tuple(f"2026-01-{index + 1:02d}" for index in range(36)),
        future_timestamps=("2026-02-06", "2026-02-07"),
        documents=(
            Document(
                "support",
                "Entity A sales remain at one and two units during the forecast window.",
                role="supporting",
            ),
            Document(
                "distractor",
                "Entity B inventory changes during an unrelated window.",
                role="distractor",
            ),
        ),
        gt_evidence=("Entity A sales remain at one and two units",),
        labels_public=True,
    )


def _release(*, lineage: tuple[str, ...] = ("champion_a",)) -> ChampionRelease:
    assumption = EvolutionAssumption(
        assumption_id="history_ready",
        candidate_name="specialist",
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="The reviewed specialist applies to this history length.",
        failure_condition="The history length leaves the reviewed range.",
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="select_specialist",
                kind="select",
                parents=("specialist",),
                fallback_parent="specialist",
                assumptions=(assumption,),
            ),
            thresholds=(("history_ready", 0.0),),
        ),
        source_hashes=(("dictionary", hashlib.sha256(b"reviewed").hexdigest()),),
        metric_policy_fingerprint="1" * 64,
        lineage=lineage,
    )


def _seed_supply_release(
    *, lineage: tuple[str, ...] = ("champion_a",)
) -> NumericalSupplyRelease:
    return NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_release(lineage=lineage).to_payload(),
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": "4" * 64},
        runtime_fingerprints={"materializer": "5" * 64},
    )


def _frozen_registry(entries, *, release: NumericalSupplyRelease | None = None):
    release = release or _seed_supply_release()
    return FrozenNumericalPackageRegistry(
        entries,
        release_sha256=release.fingerprint,
        expected_task_ids=tuple(sorted(task.numeric.task_id for task, _package in entries)),
    )


def _diagnostic(
    name: str,
    family: str,
    error: float,
    forecast: tuple[float, ...],
) -> CandidateDiagnostics:
    return CandidateDiagnostics.synthetic(
        name=name,
        family=family,
        median_mase=error,
        fold_forecasts=(forecast,) * 3,
        fold_truths=((1.0, 2.0),) * 3,
        median_smae=error,
        recent_smae=error,
        worst_smae=error,
        median_srmse=error,
        recent_srmse=error,
        worst_srmse=error,
        worst_smae_raw=error,
        worst_srmse_raw=error,
    )


def _package(
    *,
    specialist: tuple[float, float] = (1.0, 2.0),
    lineage: tuple[str, ...] = ("champion_a",),
):
    forecasts = {"safe_anchor": (7.0, 7.0), "specialist": specialist}
    diagnostics = {
        "safe_anchor": _diagnostic("safe_anchor", "tsfm", 0.1, forecasts["safe_anchor"]),
        "specialist": _diagnostic(
            "specialist", "statistical", 0.2, forecasts["specialist"]
        ),
    }
    entries = tuple(
        ScreeningEntry(
            name,
            diagnostics[name].family,
            "keep",
            ApplicabilityPolicy(),
            "fixture",
        )
        for name in forecasts
    )
    package = run_numerical_loop(
        Task("champion-runtime", _HISTORY, 2, "D", ()),
        screening_policy=ScreeningPolicy(entries, ("safe_anchor",)),
        candidate_runner=lambda name, _history, _horizon, _frequency: forecasts[name],
        diagnostics=diagnostics,
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        champion_release=_release(lineage=lineage),
    )
    broad = MorphologyToolCall("broad", "detect_periodicity", 0, len(_HISTORY))
    recent = MorphologyToolCall(
        "recent", "detect_periodicity", len(_HISTORY) // 2, len(_HISTORY)
    )
    assumption = AssumptionGrounding(
        assumption_id="cycle",
        kind="seasonality",
        claim="The observed cycle persists through the forecast window.",
        failure_condition="The cycle changes phase or disappears.",
        supporting_call_ids=("broad", "recent"),
        candidate_names=("specialist",),
        prior_confidence=0.9,
    )
    card = MorphologyCard(
        short_term="The recent segment is periodic.",
        long_term="The broad history is periodic.",
        tool_calls=(broad, recent),
        observations=(
            MorphologyObservation(broad, {"strength": 1.0}),
            MorphologyObservation(recent, {"strength": 1.0}),
        ),
        assumptions=(assumption,),
    )
    accepted, rejected, handoff = safe_retrieval_projection(card.assumptions, {})
    return replace(
        package,
        morphology_card=card,
        accepted_assumptions=accepted,
        rejected_assumptions=rejected,
        retrieval_handoff=handoff,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "morphology_card": card.fingerprint,
            "numerical_supply_release": _seed_supply_release(
                lineage=lineage
            ).fingerprint,
        },
    )


def _retrieval_card(*, quote_attempts: int = 1, valid_quotes: int = 1):
    task = _task()
    support = task.documents[0]
    chain = EvidenceChain(
        chain_id="support_chain",
        claim=support.content,
        entity_match=True,
        target_match=True,
        temporal_relation="overlaps_future",
        mechanism="future_driver",
        direction="stable",
        magnitude_kind="none",
        magnitude_value=None,
        start_timestamp=task.future_timestamps[0],
        end_timestamp=task.future_timestamps[-1],
        citations=(EvidenceCitation(support.document_id, support.content),),
        missing_links=(),
        used_skill_ids=(),
        addressed_assumption_ids=(),
        stance="supports",
        numeric_eligible=True,
    )
    round1 = RetrievalRoundResult(
        chains=(chain,),
        counterevidence=(),
        missing_information=(),
        sufficient=True,
        quote_attempt_count=quote_attempts,
        valid_quote_count=valid_quotes,
    )
    return FinalRetrievalCard(
        round1=round1,
        round2=None,
        chains=(chain,),
        selected_document_ids=(support.document_id,),
        rejected=(),
        unresolved_contradictions=(),
        complete=True,
    )


def _round_response() -> str:
    card = _retrieval_card()
    return json.dumps(
        {
            "evidence_chains": [item.to_payload() for item in card.round1.chains],
            "counterevidence": [],
            "missing_information": [],
            "sufficient": True,
        }
    )


def _decision_response(candidate_id: str = "specialist") -> str:
    return json.dumps(
        {
            "selected_candidate_id": candidate_id,
            "supporting_document_ids": ["support"],
            "rationale": "Verified same-entity evidence supports the specialist.",
            "request_more_retrieval": False,
            "gaps": [],
            "used_skill_names": [],
        }
    )


def _package_evaluator(tmp_path):
    task = _task()
    registry = _frozen_registry(((task, _package()),))
    library = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False).clone(
        read_only=True
    )
    genome = RetrievalGenome.seed()
    evaluator = PackageRetrievalEvaluator(
        registry,
        lambda candidate, skills: TwoStageRetrievalAgent(
            FakeLLMClient([_round_response()]), candidate, skills
        ),
        lambda: DecisionAgent(
            FakeLLMClient([_decision_response(), _decision_response()])
        ),
        dependency_fingerprints={
            "retrieval_factory": "1" * 64,
            "decision_factory": "2" * 64,
            "bridge_runtime": "3" * 64,
        },
    )
    cache_key = build_inference_cache_key(
        task,
        genome,
        library,
        verifier_hash="verifier-v1",
        evaluator_hash=evaluator.evaluator_hash,
        metric_hash="metric-v1",
        metric_cap=5.0,
        harness_hash="package-registry-v1",
        scientific_inputs_hash="4" * 64,
    )
    return task, genome, library, evaluator, cache_key


def test_registry_rejects_a_rebound_task() -> None:
    task = _task()
    package = _package()
    registry = _frozen_registry(((task, package),))

    assert registry.package_for(task) is package
    with pytest.raises(PackageRegistryError, match="task binding"):
        registry.package_for(replace(task, target_name="different"))


def test_registry_identity_changes_with_a_materialized_forecast() -> None:
    task = _task()
    left = _frozen_registry(((task, _package()),))
    right = _frozen_registry(
        ((task, _package(specialist=(2.0, 3.0))),)
    )

    assert left.fingerprint != right.fingerprint


def test_registry_identity_changes_with_the_champion_release() -> None:
    task = _task()
    left = _frozen_registry(((task, _package()),))
    right = _frozen_registry(
        ((task, _package(lineage=("champion_b",))),),
        release=_seed_supply_release(lineage=("champion_b",)),
    )

    assert left.fingerprint != right.fingerprint


def test_registry_manifest_is_transitively_immutable() -> None:
    registry = _frozen_registry(((_task(), _package()),))
    entries = registry.manifest["entries"]

    with pytest.raises(TypeError):
        entries[0]["task_id"] = "rebound"


def test_typed_card_quality_counts_support_distractors_and_quotes() -> None:
    quality = score_retrieval_card_quality(_task(), _retrieval_card())

    assert quality.supporting_recall == 1.0
    assert quality.gt_evidence_recall == 1.0
    assert quality.distractor_avoidance == 1.0
    assert quality.exact_quote_validity == 1.0
    assert quality.complete_chain_rate == 1.0
    assert quality.rejection_count == 0


def test_typed_card_quality_uses_raw_quote_audit_denominator() -> None:
    quality = score_retrieval_card_quality(
        _task(),
        _retrieval_card(quote_attempts=2, valid_quotes=1),
    )

    assert quality.exact_quote_validity == 0.5


def test_package_retrieval_evaluator_scores_final_choice_over_frozen_pool(
    tmp_path,
) -> None:
    task, genome, library, evaluator, cache_key = _package_evaluator(tmp_path)

    evaluation = evaluator.evaluate(
        genome,
        (task,),
        stage="g0_parent_screen_train",
        skill_library=library,
        harness_factory=None,
        persist=False,
        writers_enabled=False,
        evolver_enabled=False,
        cache_keys=(cache_key,),
        metric_cap=5.0,
    )

    assert evaluation.mean_contextual_oracle_smae == 0.0
    assert evaluation.mean_contextual_oracle_srmse == 0.0
    assert evaluation.mean_final_smae == 0.0
    assert evaluation.mean_final_srmse == 0.0
    assert evaluation.p90_smae == 0.0
    assert evaluation.p95_smae == 0.0
    assert evaluation.supporting_recall == 1.0
    assert evaluation.distractor_avoidance == 1.0
    assert evaluation.exact_quote_validity == 1.0
    assert evaluation.invalid_count == 0
    assert evaluation.catastrophic_count == 0
    assert evaluation.promotion_evidence == ()
    assert evaluation.promotion_replays == ()


def test_package_retrieval_evaluator_exposes_shared_package_evaluation(
    tmp_path,
) -> None:
    task, genome, library, evaluator, _cache_key = _package_evaluator(tmp_path)

    evaluation = evaluator.evaluate_package(
        genome,
        (task,),
        stage="g0_parent_screen_train",
        skill_library=library,
    )

    assert isinstance(evaluation, PackageEvaluation)
    assert evaluation.candidate_sha256 == genome.fingerprint()
    assert evaluation.coverage == 1.0
    assert evaluation.task_rows[0].selected_candidate_id == "specialist"
    assert evaluation.secondary_diagnostics["retrieval_supporting_recall"] == 1.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"persist": True},
        {"writers_enabled": True},
        {"evolver_enabled": True},
        {"harness_factory": object()},
    ),
)
def test_package_retrieval_evaluator_rejects_mutating_or_legacy_modes(
    tmp_path,
    overrides,
) -> None:
    task, genome, library, evaluator, cache_key = _package_evaluator(tmp_path)
    kwargs = {
        "stage": "g0_parent_screen_train",
        "skill_library": library,
        "harness_factory": None,
        "persist": False,
        "writers_enabled": False,
        "evolver_enabled": False,
        "cache_keys": (cache_key,),
        "metric_cap": 5.0,
        **overrides,
    }

    with pytest.raises(ValueError, match="package|read-only|legacy|writer"):
        evaluator.evaluate(genome, (task,), **kwargs)


def test_package_retrieval_evaluator_requires_resolved_non_public_tasks_only(
    tmp_path,
) -> None:
    task, genome, library, evaluator, cache_key = _package_evaluator(tmp_path)
    hidden = replace(task, labels_public=False)

    with pytest.raises(ValueError, match="resolved|label"):
        evaluator.evaluate(
            genome,
            (hidden,),
            stage="g0_parent_screen_train",
            skill_library=library,
            harness_factory=None,
            persist=False,
            writers_enabled=False,
            evolver_enabled=False,
            cache_keys=(cache_key,),
            metric_cap=5.0,
        )


def test_package_retrieval_evaluator_requires_canonical_dependency_hashes(
    tmp_path,
) -> None:
    task = _task()
    registry = _frozen_registry(((task, _package()),))

    with pytest.raises(ValueError, match="fingerprint"):
        PackageRetrievalEvaluator(
            registry,
            lambda candidate, skills: TwoStageRetrievalAgent(
                FakeLLMClient([_round_response()]), candidate, skills
            ),
            lambda: DecisionAgent(
                FakeLLMClient([_decision_response(), _decision_response()])
            ),
            dependency_fingerprints={
                "retrieval_factory": "not-a-hash",
                "decision_factory": "2" * 64,
                "bridge_runtime": "3" * 64,
            },
        )
