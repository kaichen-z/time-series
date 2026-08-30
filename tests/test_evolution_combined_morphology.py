"""Tests for sanitized Train-only morphology evidence in Combined proposals."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from common.llm import FakeLLMClient
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.portfolio import PolicyPortfolio
from numerical_agent.evolution.screening import profile_task


def _periodic_profiles(count: int = 3):
    return tuple(
        profile_task(
            Task(
                f"secret-task-{index}",
                tuple(float((step % 3) + 1) for step in range(36)),
                3,
                "1 day",
                (99.0, 98.0, 97.0),
            )
        )
        for index in range(count)
    )


def _trusted_group_inputs(count: int = 3) -> dict[str, object]:
    return {
        "entity_ids": tuple(chr(ord("a") + index) for index in range(count)),
        "split": "train",
        "group_id": "periodic_high_confidence",
        "eligible_leaves": ("timesfm_2_5", "seasonal_naive"),
        "baseline": "toto_2_0",
        "reviewed_leaf_names": ("timesfm_2_5", "seasonal_naive", "toto_2_0"),
        "truths": ((1.0, 1.0),) * count,
        "candidate_forecasts": ((1.5, 1.5),) * count,
        "baseline_forecasts": ((2.0, 2.0),) * count,
        "forecast_disagreements": (0.25,) * count,
    }


def test_trusted_group_builder_exposes_no_precomputed_metric_authority() -> None:
    """A module caller must have no record or token that grants scorer-free authority."""
    import numerical_agent.evolution.combined_evolution as combined_evolution

    signature = inspect.signature(
        combined_evolution.summarize_morphology_group_evidence
    )

    assert "scaled_deltas" not in signature.parameters
    assert not hasattr(combined_evolution, "CanonicalScaledDelta")
    assert not hasattr(combined_evolution, "winsorized_scaled_metric_delta")
    assert not hasattr(combined_evolution, "_CANONICAL_EVIDENCE_TOKEN")


def test_only_aligned_truth_and_forecast_arrays_build_sanitized_group_evidence() -> None:
    """Removing internal scoring or retaining its arrays would break the trust boundary."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        summarize_morphology_group_evidence,
    )

    evidence = summarize_morphology_group_evidence(
        _periodic_profiles(), **_trusted_group_inputs()
    )

    assert evidence.winsorized_smae_delta == pytest.approx(-0.5)
    assert evidence.winsorized_srmse_delta == pytest.approx(-0.5)
    forbidden = {"truth", "truths", "forecast", "forecasts"}
    assert forbidden.isdisjoint(vars(evidence))
    assert forbidden.isdisjoint(evidence.to_payload())
    with pytest.raises(CombinedEvolutionError, match="align|complete"):
        summarize_morphology_group_evidence(
            _periodic_profiles(),
            **{
                **_trusted_group_inputs(),
                "candidate_forecasts": ((1.5, 1.5), (1.5, 1.5)),
            },
        )


def test_zero_scale_group_evidence_serializes_infinite_raw_tail_canonically() -> None:
    """Rejecting or emitting JSON Infinity would erase catastrophic zero-scale evidence."""
    from numerical_agent.evolution.combined_evolution import (
        summarize_morphology_group_evidence,
    )

    inputs = _trusted_group_inputs()
    inputs.update(
        truths=((0.0, 0.0),) * 3,
        candidate_forecasts=((1.0, 1.0),) * 3,
        baseline_forecasts=((0.0, 0.0),) * 3,
    )
    evidence = summarize_morphology_group_evidence(_periodic_profiles(), **inputs)
    payload = evidence.to_payload()

    assert evidence.candidate_worst_smae_raw == float("inf")
    assert evidence.candidate_worst_srmse_raw == float("inf")
    assert evidence.candidate_smae_clipped_count == 3
    assert evidence.candidate_srmse_clipped_count == 3
    assert payload["candidate_worst_smae_raw"] == "positive_infinity"
    assert payload["candidate_worst_srmse_raw"] == "positive_infinity"
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert "Infinity" not in serialized


def _evidence(**overrides: object):
    from numerical_agent.evolution.combined_evolution import (
        summarize_morphology_group_evidence,
    )

    group_id = str(overrides.pop("group_id", "periodic_high_confidence"))
    eligible_leaves = overrides.pop(
        "eligible_leaves", ("timesfm_2_5", "seasonal_naive")
    )
    baseline = overrides.pop("baseline", "toto_2_0")
    profiles = tuple(
        profile_task(
            Task(
                f"task-{index}",
                tuple(float((step % 3) + 1) for step in range(36)),
                3,
                "1 day",
                (0.0, 0.0, 0.0),
            )
        )
        for index in range(3)
    )
    if group_id == "intermittent":
        profiles = tuple(replace(profile, intermittency_adi=2.0) for profile in profiles)
    evidence = summarize_morphology_group_evidence(
        profiles,
        entity_ids=("a", "b", "c"),
        split="train",
        group_id=group_id,
        eligible_leaves=eligible_leaves,  # type: ignore[arg-type]
        baseline=baseline,  # type: ignore[arg-type]
        reviewed_leaf_names=(
            "timesfm_2_5",
            "seasonal_naive",
            "holt_damped_trend",
            "toto_2_0",
            "unknown_leaf",
        ),
        truths=((1.0, 1.0),) * 3,
        candidate_forecasts=((1.5, 1.5),) * 3,
        baseline_forecasts=((2.0, 2.0),) * 3,
        forecast_disagreements=(0.25, 0.25, 0.25),
    )
    expected_predicates = {
        "periodic_high_confidence": ("periodicity_strength", "at_least", 0.6),
        "intermittent": ("intermittency_adi", "at_least", 1.32),
    }
    expected = expected_predicates.get(group_id)
    for name, index in (("feature", 0), ("operator", 1), ("threshold", 2)):
        if expected is not None and overrides.get(name) == expected[index]:
            overrides.pop(name)
    if overrides:
        for name, value in overrides.items():
            object.__setattr__(evidence, name, value)
    return evidence


def test_morphology_group_evidence_is_immutable_and_projects_only_canonical_aggregates() -> None:
    """Adding raw identities or mutable payloads would leak Train examples to the model."""
    evidence = _evidence()

    assert evidence.to_payload() == {
        "baseline": "toto_2_0",
        "coverage": 1.0,
        "eligible_leaves": ["seasonal_naive", "timesfm_2_5"],
        "entity_count": 3,
        "failure_rate": 0.0,
        "feature": "periodicity_strength",
        "forecast_disagreement": 0.25,
        "group_id": "periodic_high_confidence",
        "operator": "at_least",
        "task_count": 3,
        "threshold": 0.6,
        "winsorized_smae_delta": -0.5,
        "winsorized_srmse_delta": -0.5,
        "candidate_worst_smae_raw": 0.5,
        "candidate_worst_srmse_raw": 0.5,
        "baseline_worst_smae_raw": 1.0,
        "baseline_worst_srmse_raw": 1.0,
        "candidate_smae_clipped_count": 0,
        "candidate_srmse_clipped_count": 0,
        "baseline_smae_clipped_count": 0,
        "baseline_srmse_clipped_count": 0,
        "candidate_smae_clipped_rate": 0.0,
        "candidate_srmse_clipped_rate": 0.0,
        "baseline_smae_clipped_rate": 0.0,
        "baseline_srmse_clipped_rate": 0.0,
    }
    with pytest.raises(FrozenInstanceError):
        evidence.task_count = 9  # type: ignore[misc]
    assert not {
        "task_id", "timestamp", "future", "document", "split", "dev", "public", "hidden"
    } & evidence.to_payload().keys()


@pytest.mark.parametrize(
    "overrides",
    (
        {"group_id": "raw_task_17"},
        {"feature": "future_mean"},
        {"operator": "arbitrary"},
        {"threshold": 0.61},
        {"entity_count": 2},
        {"task_count": 2, "entity_count": 3},
        {"eligible_leaves": ["timesfm_2_5", "seasonal_naive"]},
        {"eligible_leaves": ("timesfm_2_5", "timesfm_2_5")},
        {"winsorized_smae_delta": float("nan")},
        {"winsorized_srmse_delta": float("inf")},
        {"coverage": -0.1},
        {"failure_rate": 1.1},
        {"coverage": 0.8, "failure_rate": 0.8},
        {"forecast_disagreement": float("nan")},
    ),
)
def test_morphology_group_evidence_rejects_unsupported_or_hostile_values(
    overrides: dict[str, object],
) -> None:
    """Weak aggregate validation could admit unsupported predicates or poisoned metrics."""
    from numerical_agent.evolution.combined_evolution import CombinedEvolutionError

    with pytest.raises(CombinedEvolutionError):
        _evidence(**overrides).to_payload()


def test_train_profiles_are_summarized_by_the_fixed_predicate_without_exposing_ids() -> None:
    """A wrong predicate or row alignment would corrupt support and aggregate deltas."""
    from numerical_agent.evolution.combined_evolution import (
        summarize_morphology_group_evidence,
    )

    profiles = tuple(
        profile_task(
            Task(
                f"secret-task-{index}",
                tuple(float((step % 3) + 1) for step in range(36)),
                3,
                "1 day",
                (99.0, 98.0, 97.0),
            )
        )
        for index in range(4)
    )
    evidence = summarize_morphology_group_evidence(
        profiles,
        entity_ids=("entity-a", "entity-b", "entity-c", "entity-a"),
        split="train",
        group_id="periodic_high_confidence",
        eligible_leaves=("timesfm_2_5", "seasonal_naive"),
        baseline="toto_2_0",
        reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
        truths=((1.0, 1.0), (1.0, 1.0), None, (1.0, 1.0)),
        candidate_forecasts=((1.5, 1.5), (1.5, 1.5), None, (1.5, 1.5)),
        baseline_forecasts=((2.0, 2.0), (2.0, 2.0), None, (2.0, 2.0)),
        forecast_disagreements=(0.2, 0.4, None, 0.6),
    )

    assert evidence.task_count == 4
    assert evidence.entity_count == 3
    assert evidence.coverage == pytest.approx(0.75)
    assert evidence.failure_rate == pytest.approx(0.25)
    assert evidence.winsorized_smae_delta == pytest.approx(-0.5)
    assert evidence.winsorized_srmse_delta == pytest.approx(-0.5)
    assert evidence.forecast_disagreement == pytest.approx(0.4)
    assert "secret-task" not in json.dumps(evidence.to_payload(), sort_keys=True)
    assert "99.0" not in json.dumps(evidence.to_payload(), sort_keys=True)


def test_trusted_group_builder_uses_canonical_cap_and_rejects_incomplete_pairs() -> None:
    """A hand-rolled or single-metric producer could leak uncapped proposal evidence."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        summarize_morphology_group_evidence,
    )

    evidence = summarize_morphology_group_evidence(
        _periodic_profiles(),
        **{
            **_trusted_group_inputs(),
            "truths": ((1.0, 1.0, 1.0, 1.0),) * 3,
            "candidate_forecasts": ((1.0, 1.0, 1.0, 25.0),) * 3,
            "baseline_forecasts": ((1.0, 1.0, 1.0, 1.0),) * 3,
        },
    )

    assert evidence.winsorized_smae_delta == 5.0
    assert evidence.winsorized_srmse_delta == 5.0
    assert evidence.candidate_worst_smae_raw == 6.0
    assert evidence.candidate_worst_srmse_raw == 12.0
    assert evidence.baseline_worst_smae_raw == 0.0
    assert evidence.baseline_worst_srmse_raw == 0.0
    assert evidence.candidate_smae_clipped_count == 3
    assert evidence.candidate_srmse_clipped_count == 3
    assert evidence.baseline_smae_clipped_count == 0
    assert evidence.baseline_srmse_clipped_count == 0

    with pytest.raises(CombinedEvolutionError, match="complete"):
        summarize_morphology_group_evidence(
            _periodic_profiles(),
            **{
                **_trusted_group_inputs(),
                "candidate_forecasts": ((1.0,),) * 3,
            },
        )


def test_precomputed_delta_floats_cannot_enter_morphology_group_evidence() -> None:
    """A caller-supplied metric pair must not acquire canonical evidence authority."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        MorphologyGroupEvidence,
        summarize_morphology_group_evidence,
    )

    with pytest.raises(CombinedEvolutionError, match="trusted group builder"):
        MorphologyGroupEvidence(
            group_id="periodic_high_confidence",
            feature="periodicity_strength",
            operator="at_least",
            threshold=0.6,
            task_count=3,
            entity_count=3,
            eligible_leaves=("timesfm_2_5", "seasonal_naive"),
            baseline="toto_2_0",
            winsorized_smae_delta=-0.5,
            winsorized_srmse_delta=-0.5,
            coverage=1.0,
            failure_rate=0.0,
        )

    profiles = tuple(
        profile_task(Task(str(index), (1.0, 2.0, 1.0, 2.0), 1, "D", (0.0,)))
        for index in range(3)
    )
    with pytest.raises(TypeError):
        summarize_morphology_group_evidence(
            profiles,
            entity_ids=("a", "b", "c"),
            split="train",
            group_id="periodic_high_confidence",
            eligible_leaves=("timesfm_2_5", "seasonal_naive"),
            baseline="toto_2_0",
            reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
            scaled_deltas=((-0.5, -0.5),) * 3,
            forecast_disagreements=(0.0, 0.0, 0.0),
        )


def test_canonical_group_snapshot_rejects_post_construction_metric_forgery() -> None:
    from numerical_agent.evolution.combined_evolution import CombinedEvolutionError

    evidence = _evidence()
    object.__setattr__(evidence, "winsorized_smae_delta", 0.0)

    with pytest.raises(CombinedEvolutionError, match="trusted group builder"):
        evidence.to_payload()


def test_group_builder_projects_sanitized_raw_tail_and_clipping_aggregates() -> None:
    from numerical_agent.evolution.combined_evolution import (
        summarize_morphology_group_evidence,
    )

    profiles = tuple(
        profile_task(
            Task(
                f"secret-{index}",
                tuple(float((step % 3) + 1) for step in range(36)),
                3,
                "1 day",
                (99.0, 98.0, 97.0),
            )
        )
        for index in range(3)
    )
    evidence = summarize_morphology_group_evidence(
        profiles,
        entity_ids=("a", "b", "c"),
        split="train",
        group_id="periodic_high_confidence",
        eligible_leaves=("timesfm_2_5", "seasonal_naive"),
        baseline="toto_2_0",
        reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
        truths=(
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
        ),
        candidate_forecasts=(
            (1.0, 1.0, 1.0, 25.0),
            (1.5, 1.5),
            (1.5, 1.5),
        ),
        baseline_forecasts=(
            (1.0, 1.0, 1.0, 1.0),
            (2.0, 2.0),
            (2.0, 2.0),
        ),
        forecast_disagreements=(0.2, 0.3, 0.4),
    )
    payload = evidence.to_payload()

    assert payload["candidate_worst_smae_raw"] == 6.0
    assert payload["candidate_worst_srmse_raw"] == 12.0
    assert payload["baseline_worst_smae_raw"] == 1.0
    assert payload["baseline_worst_srmse_raw"] == 1.0
    assert payload["candidate_smae_clipped_count"] == 1
    assert payload["candidate_srmse_clipped_count"] == 1
    assert payload["candidate_smae_clipped_rate"] == pytest.approx(1 / 3)
    assert payload["candidate_srmse_clipped_rate"] == pytest.approx(1 / 3)
    assert payload["baseline_smae_clipped_count"] == 0
    assert payload["baseline_srmse_clipped_count"] == 0
    serialized = json.dumps(payload, sort_keys=True)
    assert "secret-" not in serialized
    assert "99.0" not in serialized


@pytest.mark.parametrize("measurement", (True, 1, type("FloatSubclass", (float,), {})(1.0)))
def test_morphology_summarizer_rejects_nonexact_profile_measurements(
    measurement: object,
) -> None:
    """Numeric coercion must not turn bools, ints, or subclasses into trusted evidence."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        summarize_morphology_group_evidence,
    )

    base_profiles = tuple(
        profile_task(
            Task(
                f"task-{index}",
                tuple(float((step % 3) + 1) for step in range(36)),
                3,
                "1 day",
                (0.0, 0.0, 0.0),
            )
        )
        for index in range(3)
    )
    profiles = tuple(
        replace(profile, periodicity_strength=measurement)  # type: ignore[arg-type]
        for profile in base_profiles
    )

    with pytest.raises(CombinedEvolutionError):
        summarize_morphology_group_evidence(
            profiles,
            entity_ids=("a", "b", "c"),
            split="train",
            group_id="periodic_high_confidence",
            eligible_leaves=("timesfm_2_5", "seasonal_naive"),
            baseline="toto_2_0",
            reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
            truths=((1.0, 1.0),) * 3,
            candidate_forecasts=((1.5, 1.5),) * 3,
            baseline_forecasts=((2.0, 2.0),) * 3,
            forecast_disagreements=(0.0, 0.0, 0.0),
        )


def test_morphology_summarizer_rejects_non_train_and_hostile_containers() -> None:
    """Dev/Public data and polymorphic sequences must fail before aggregate projection."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedEvolutionError,
        summarize_morphology_group_evidence,
    )

    with pytest.raises(CombinedEvolutionError, match="Train"):
        summarize_morphology_group_evidence(
            object(),  # type: ignore[arg-type]
            entity_ids=object(),  # type: ignore[arg-type]
            split="dev",
            group_id="periodic_high_confidence",
            eligible_leaves=("timesfm_2_5", "seasonal_naive"),
            baseline="toto_2_0",
            reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
            truths=object(),  # type: ignore[arg-type]
            candidate_forecasts=object(),  # type: ignore[arg-type]
            baseline_forecasts=object(),  # type: ignore[arg-type]
            forecast_disagreements=object(),  # type: ignore[arg-type]
        )

    profiles = tuple(
        profile_task(Task(str(index), (1.0, 2.0, 1.0, 2.0), 1, "D", (0.0,)))
        for index in range(3)
    )
    with pytest.raises(CombinedEvolutionError):
        summarize_morphology_group_evidence(
            profiles,
            entity_ids=["a", "b", "c"],  # type: ignore[arg-type]
            split="train",
            group_id="periodic_high_confidence",
            eligible_leaves=("timesfm_2_5", "seasonal_naive"),
            baseline="toto_2_0",
            reviewed_leaf_names=("timesfm_2_5", "seasonal_naive", "toto_2_0"),
            truths=((1.0, 1.0),) * 3,
            candidate_forecasts=((1.5, 1.5),) * 3,
            baseline_forecasts=((2.0, 2.0),) * 3,
            forecast_disagreements=(0.0, 0.0, 0.0),
        )


def test_proposal_prompt_receives_only_reviewed_morphology_group_evidence() -> None:
    """Dropping identity checks could expose unknown leaves as executable parents."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedProposalDiagnostics,
        propose_combined_child,
    )

    parent = PolicyPortfolio.flagship5()
    diagnostics = CombinedProposalDiagnostics(128, 0.25, 4, 1, (_evidence(),))
    agent = FakeLLMClient(['{"operations": []}'])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=diagnostics,
        agent=agent,
    )

    assert result.child is parent
    prompt = json.loads(agent.calls[0]["messages"][0]["content"])
    assert prompt["diagnostics"]["morphology_groups"] == [_evidence().to_payload()]
    assert set(prompt["allowed_operations"]["policy"]["signals"]) == {
        "history_length",
        "horizon",
        "horizon_ratio",
        "intermittency_adi",
        "noise_relative_scale",
        "outlier_fraction",
        "periodicity_strength",
        "recent_regime_confidence",
        "trend_strength",
        "zero_fraction",
    }


def test_statistical_only_evidence_reaches_one_llm_call() -> None:
    """Evidence leaves need not include a TSFM; only Combined policies have that rule."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedProposalDiagnostics,
        propose_combined_child,
    )

    parent = PolicyPortfolio.flagship5()
    diagnostics = CombinedProposalDiagnostics(
        128,
        0.25,
        4,
        1,
        (
            _evidence(
                eligible_leaves=("seasonal_naive", "holt_damped_trend")
            ),
        ),
    )
    agent = FakeLLMClient(['{"operations": []}'])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive", "holt_damped_trend"),
        diagnostics=diagnostics,
        agent=agent,
    )

    assert result.child is parent
    assert len(agent.calls) == 1


def test_evidence_and_group_permutations_have_one_canonical_prompt_and_fingerprint() -> None:
    """Caller ordering must not change semantically equivalent prompt bytes or hashes."""
    from numerical_agent.evolution.combined_evolution import (
        CombinedProposalDiagnostics,
        propose_combined_child,
    )

    parent = PolicyPortfolio.flagship5()
    periodic_left = _evidence(
        eligible_leaves=("timesfm_2_5", "seasonal_naive")
    )
    periodic_right = _evidence(
        eligible_leaves=("seasonal_naive", "timesfm_2_5")
    )
    intermittent_left = _evidence(
        group_id="intermittent",
        feature="intermittency_adi",
        threshold=1.32,
        eligible_leaves=("seasonal_naive", "timesfm_2_5"),
    )
    intermittent_right = _evidence(
        group_id="intermittent",
        feature="intermittency_adi",
        threshold=1.32,
        eligible_leaves=("timesfm_2_5", "seasonal_naive"),
    )
    left_agent = FakeLLMClient(['{"operations": []}'])
    right_agent = FakeLLMClient(['{"operations": []}'])

    propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=CombinedProposalDiagnostics(
            128, 0.25, 4, 1, (periodic_left, intermittent_left)
        ),
        agent=left_agent,
    )
    propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=CombinedProposalDiagnostics(
            128, 0.25, 4, 1, (intermittent_right, periodic_right)
        ),
        agent=right_agent,
    )

    left_prompt = left_agent.calls[0]["messages"][0]["content"]
    right_prompt = right_agent.calls[0]["messages"][0]["content"]
    assert left_prompt == right_prompt
    left_fingerprint = hashlib.sha256(
        json.dumps(periodic_left.to_payload(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    right_fingerprint = hashlib.sha256(
        json.dumps(periodic_right.to_payload(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert left_fingerprint == right_fingerprint


def test_proposal_rejects_unknown_evidence_leaf_before_calling_llm() -> None:
    """An unreviewed leaf name must never enter a prompt or executable policy."""
    from numerical_agent.evolution.combined_evolution import CombinedProposalDiagnostics, propose_combined_child

    parent = PolicyPortfolio.flagship5()
    diagnostics = CombinedProposalDiagnostics(
        128,
        0.25,
        4,
        1,
        (_evidence(eligible_leaves=("timesfm_2_5", "unknown_leaf")),),
    )
    agent = FakeLLMClient(['{"operations": []}'])

    result = propose_combined_child(
        parent,
        statistical_names=("seasonal_naive",),
        diagnostics=diagnostics,
        agent=agent,
    )

    assert result.child is parent
    assert result.changed is False
    assert agent.calls == []
