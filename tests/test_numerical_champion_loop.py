"""Runtime integration contract for a frozen Numerical Champion."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace

import pytest
import numerical_agent.evolution.numerical_loop as loop_module
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.champion_runtime import (
    ChampionExecution,
    execute_champion as real_execute,
)
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_package import NumericalForecastPackage
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
)
from numerical_agent.evolution.screening import (
    ApplicabilityPolicy,
    ScreeningEntry,
    ScreeningPolicy,
)


def _task() -> Task:
    return Task("champion-runtime", (1.0, 2.0, 3.0) * 12, 2, "D", ())


def _screening() -> ScreeningPolicy:
    entries = (
        ScreeningEntry("safe_anchor", "tsfm", "keep", ApplicabilityPolicy(), "fixture"),
        ScreeningEntry(
            "specialist", "statistical", "keep", ApplicabilityPolicy(), "fixture"
        ),
    )
    return ScreeningPolicy(entries, ("safe_anchor",))


def _diagnostics() -> dict[str, CandidateDiagnostics]:
    truth = (1.0, 2.0)
    return {
        name: CandidateDiagnostics.synthetic(
            name=name,
            family=family,
            median_mase=error,
            fold_forecasts=(forecast,) * 3,
            fold_truths=(truth,) * 3,
            median_smae=error,
            recent_smae=error,
            worst_smae=error,
            median_srmse=error,
            recent_srmse=error,
            worst_srmse=error,
            worst_smae_raw=error,
            worst_srmse_raw=error,
        )
        for name, family, error, forecast in (
            ("safe_anchor", "tsfm", 0.1, (7.0, 7.0)),
            ("specialist", "statistical", 0.2, (1.0, 2.0)),
        )
    }


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


def _run(*, release: ChampionRelease | None = None, calls: Counter[str] | None = None):
    runner_calls = calls if calls is not None else Counter()

    def runner(name: str, _history: tuple[float, ...], horizon: int, _frequency: str):
        runner_calls[name] += 1
        return (
            (7.0,) * horizon
            if name == "safe_anchor"
            else tuple(float(i + 1) for i in range(horizon))
        )

    return run_numerical_loop(
        _task(),
        screening_policy=_screening(),
        candidate_runner=runner,
        diagnostics=_diagnostics(),
        decision_policy=DecisionPolicy(ensemble_enabled=False),
        champion_release=release,
    )


def test_champion_selects_only_an_already_materialized_ranked_forecast(
    monkeypatch,
) -> None:
    # This fails if Champion execution receives hindcast truth or if it reruns a leaf.
    observed: list[object] = []

    def checked_execute(policy, forecasts, diagnostics, profile, history, horizon):
        observed.append((forecasts, diagnostics, profile, history, horizon))
        assert all(item.folds == () for item in diagnostics.values())
        assert all(item.fold_forecasts == () for item in diagnostics.values())
        assert all(item.fold_truths == () for item in diagnostics.values())
        assert all(item.long_horizon_fold is None for item in diagnostics.values())
        return real_execute(policy, forecasts, diagnostics, profile, history, horizon)

    monkeypatch.setattr(loop_module, "execute_champion", checked_execute)
    calls: Counter[str] = Counter()

    package = _run(release=_release(), calls=calls)

    assert package.selection_decision.selected == ("specialist",)
    assert package.selection_decision.selected[0] in {
        item.name for item in package.ranked_alternatives
    }
    assert package.final_forecast == (1.0, 2.0)
    assert calls == Counter({"safe_anchor": 1, "specialist": 1})
    assert len(observed) == 1
    assert package.selection_decision.assumption_ids == ("history_ready",)
    assert set(package.component_fingerprints) >= {
        "champion_release",
        "champion_recipe",
        "champion_assumptions",
    }
    assert not hasattr(package, "champion_release")
    assert not hasattr(package, "champion_thresholds")


def test_champion_release_fingerprint_changes_package_identity() -> None:
    first = _run(release=_release(lineage=("champion_a",)))
    second = _run(release=_release(lineage=("champion_b",)))

    assert (
        first.component_fingerprints["champion_release"]
        != second.component_fingerprints["champion_release"]
    )
    assert (
        first.component_fingerprints["champion_recipe"]
        == second.component_fingerprints["champion_recipe"]
    )
    assert (
        first.component_fingerprints["champion_assumptions"]
        == second.component_fingerprints["champion_assumptions"]
    )


def test_package_rejects_partial_or_non_hash_champion_fingerprints() -> None:
    package = _run(release=_release())

    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            package,
            component_fingerprints={
                **dict(package.component_fingerprints),
                "champion_release": "not-a-sha256",
            },
        )
    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            package,
            component_fingerprints={
                key: value
                for key, value in package.component_fingerprints.items()
                if key != "champion_assumptions"
            },
        )


def test_legacy_package_keeps_legacy_named_component_fingerprints() -> None:
    package = _run()

    legacy = replace(
        package,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "champion_release": "legacy-tag",
        },
    )

    assert legacy.component_fingerprints["champion_release"] == "legacy-tag"
    assert legacy == replace(
        package,
        component_fingerprints={
            **dict(package.component_fingerprints),
            "champion_release": "legacy-tag",
        },
    )


def test_legacy_package_never_infers_champion_provenance_from_contents() -> None:
    legacy = _run()
    champion = _run(release=_release())

    collision = replace(
        legacy,
        selection_decision=replace(
            legacy.selection_decision,
            reason_codes=("frozen_champion", "legacy_reason"),
        ),
        component_fingerprints={
            **dict(legacy.component_fingerprints),
            **{
                key: champion.component_fingerprints[key]
                for key in (
                    "champion_release",
                    "champion_recipe",
                    "champion_assumptions",
                )
            },
        },
    )

    assert type(collision) is NumericalForecastPackage
    assert collision.selection_decision.reason_codes == (
        "frozen_champion",
        "legacy_reason",
    )


def test_champion_package_subtype_preserves_its_provenance_invariant() -> None:
    champion = _run(release=_release())

    assert type(champion) is not NumericalForecastPackage
    assert type(replace(champion)) is type(champion)
    with pytest.raises(ValueError, match="fixed provenance marker"):
        replace(
            champion,
            selection_decision=replace(
                champion.selection_decision,
                reason_codes=("legacy_reason",),
            ),
        )
    with pytest.raises(ValueError, match="Champion component fingerprints"):
        replace(
            champion,
            component_fingerprints={
                key: value
                for key, value in champion.component_fingerprints.items()
                if key != "champion_assumptions"
            },
        )


class _FloatSubclass(float):
    pass


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "malformed",
    (
        pytest.param(
            ChampionExecution((10**10000, 2.0), ("specialist",), (), None),
            id="huge_int",
        ),
        ChampionExecution((_FloatSubclass(1.0), 2.0), ("specialist",), (), None),
        ChampionExecution(("1.0", 2.0), ("specialist",), (), None),
        ChampionExecution((object(), 2.0), ("specialist",), (), None),
        ChampionExecution((1.0, 2.0), (_StringSubclass("specialist"),), (), None),
    ),
)
def test_hostile_champion_execution_values_never_escape_fallback(
    monkeypatch, malformed: ChampionExecution
) -> None:
    monkeypatch.setattr(loop_module, "execute_champion", lambda *_args: malformed)

    package = _run(release=_release())

    assert package.selection_decision.selected == ("specialist",)
    assert package.fallback_reason == "champion_execution_failed:invalid_result"


@pytest.mark.parametrize(
    "malformed",
    (
        ChampionExecution((1, 2), None, (), None),  # type: ignore[arg-type]
        ChampionExecution((1.0,), ("specialist",), (), None),
        ChampionExecution((1.0, 2.0), ("unknown",), (), None),
        ChampionExecution((1.0, 2.0), ("specialist",), ("",), None),
        ChampionExecution((1.0, 2.0), ("specialist",), ("not an id",), None),
        ChampionExecution((1.0, 2.0), ("specialist",), (), ""),
    ),
)
def test_malformed_champion_execution_fields_use_fixed_materialized_fallback(
    monkeypatch, malformed: ChampionExecution
) -> None:
    monkeypatch.setattr(loop_module, "execute_champion", lambda *_args: malformed)

    package = _run(release=_release())

    assert package.selection_decision.selected == ("specialist",)
    assert package.final_forecast == (1.0, 2.0)
    assert package.fallback_reason == "champion_execution_failed:invalid_result"


def test_public_champion_package_hides_recipe_operator_and_threshold_text() -> None:
    release = _release()

    package = _run(release=release)

    assert package.selection_decision.reason_codes == (
        "frozen_champion",
        "champion_materialized_selection",
    )
    assert package.selection_decision.assumption_ids == ("history_ready",)
    assert all(
        value != release.policy.recipe.kind
        for value in package.selection_decision.reason_codes
    )
    assert all(
        "threshold" not in value for value in package.selection_decision.reason_codes
    )
    assert all(
        len(package.component_fingerprints[key]) == 64
        for key in (
            "champion_release",
            "champion_recipe",
            "champion_assumptions",
        )
    )
    assert not hasattr(package, "champion_thresholds")


def test_champion_non_materialized_arithmetic_falls_back_to_declared_parent() -> None:
    release = _release()
    weighted = replace(
        release,
        policy=FittedChampionPolicy(
            recipe=ChampionRecipe(
                name="weighted_runtime_vector",
                kind="weighted",
                parents=("safe_anchor", "specialist"),
                fallback_parent="safe_anchor",
                assumptions=(
                    replace(
                        release.policy.recipe.assumptions[0],
                        candidate_name="specialist",
                        operator="weighted",
                    ),
                ),
            ),
            thresholds=(("history_ready", 0.0),),
            weights=(0.5, 0.5),
        ),
    )

    package = _run(release=weighted)

    assert package.selection_decision.selected == ("safe_anchor",)
    assert package.final_forecast == (7.0, 7.0)
    assert package.fallback_reason == "champion_non_materialized_selection"
