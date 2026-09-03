"""Trusted proposal, fitting, and materialization of Numerical package states."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from common.payload import canonical_json_bytes
from evolving_loop.data import ContextTask
from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyRelease,
    bound_numerical_package,
    build_package_registry,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    FittedChampionPolicy,
    champion_fingerprint,
    parse_champion_release,
)
from numerical_agent.evolution.champion_controller import (
    ChampionProposerAdapter,
    fit_champion_recipe,
)
from numerical_agent.evolution.champion_evidence import (
    ChampionTaskRow,
    ProposerEvidence,
)
from numerical_agent.evolution.champion_runtime import execute_champion
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_package import (
    RankedNumericalForecast,
    valid_forecast,
)
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastConfig,
)
from numerical_agent.evolution.portfolio import CombinedPolicy
from numerical_agent.evolution.screening import ScreeningPolicy
from numerical_agent.evolution.specialist_atlas import (
    AtlasMaterializedCandidate,
    AtlasRelease,
    AtlasTaskCase,
    atlas_feature,
    route_atlas_task,
)
from numerical_agent.evolution.task_local_evolution import (
    GroupFoldManifest,
    TaskLocalTaskRow,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"n[0-9]{3}\Z")


class NumericalPackageEvolutionError(ValueError):
    """A Numerical proposal crossed its fitting or materialization boundary."""


def _fail(message: str) -> None:
    raise NumericalPackageEvolutionError(message)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()


def _fingerprints(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{label} must be a nonempty fingerprint mapping")
    result = dict(value)
    if any(
        type(key) is not str
        or not key
        or type(item) is not str
        or _SHA256.fullmatch(item) is None
        for key, item in result.items()
    ):
        _fail(f"{label} contains a noncanonical fingerprint")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class NumericalRecipeFit:
    recipe: ChampionRecipe
    full_build_policy: FittedChampionPolicy
    build_fold_policies: tuple[tuple[int, FittedChampionPolicy], ...]
    full_build_task_ids: tuple[str, ...]
    fold_training_task_ids: tuple[tuple[int, tuple[str, ...]], ...]
    numerical_score_sha256: str

    def __post_init__(self) -> None:
        if type(self.recipe) is not ChampionRecipe:
            _fail("Numerical fit requires an exact ChampionRecipe")
        if (
            type(self.full_build_policy) is not FittedChampionPolicy
            or self.full_build_policy.recipe != self.recipe
        ):
            _fail("Numerical full-Build policy does not bind its recipe")
        if type(self.build_fold_policies) is not tuple or tuple(
            fold for fold, _policy in self.build_fold_policies
        ) != (0, 1, 2, 3, 4):
            _fail("Numerical fit requires policies for Build folds zero through four")
        if any(
            type(policy) is not FittedChampionPolicy or policy.recipe != self.recipe
            for _fold, policy in self.build_fold_policies
        ):
            _fail("Numerical fold policy does not bind its recipe")
        if (
            type(self.full_build_task_ids) is not tuple
            or len(self.full_build_task_ids) != 64
            or len(self.full_build_task_ids) != len(set(self.full_build_task_ids))
            or any(
                type(task_id) is not str or not task_id
                for task_id in self.full_build_task_ids
            )
        ):
            _fail("Numerical fit requires the exact 64-task Build universe")
        if type(self.fold_training_task_ids) is not tuple or tuple(
            fold for fold, _task_ids in self.fold_training_task_ids
        ) != (0, 1, 2, 3, 4):
            _fail("Numerical fit requires training membership for every Build fold")
        universe = set(self.full_build_task_ids)
        for _fold, task_ids in self.fold_training_task_ids:
            if (
                type(task_ids) is not tuple
                or not task_ids
                or len(task_ids) != len(set(task_ids))
                or not set(task_ids) < universe
            ):
                _fail("Numerical fold training membership is incomplete")
        if (
            type(self.numerical_score_sha256) is not str
            or len(self.numerical_score_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.numerical_score_sha256
            )
        ):
            _fail("Numerical fit score must be a lowercase SHA-256 value")


def fit_numerical_recipe(
    recipe: ChampionRecipe,
    build_rows: Sequence[ChampionTaskRow],
    fold_manifest: GroupFoldManifest,
    parent: ChampionRelease,
) -> NumericalRecipeFit:
    """Fit one full-Build policy and five held-out-group policies."""
    if type(recipe) is not ChampionRecipe:
        _fail("Numerical fitting requires an exact ChampionRecipe")
    if type(build_rows) not in {tuple, list} or not build_rows:
        _fail("Numerical fitting requires a nonempty exact Build row sequence")
    supplied_rows = tuple(build_rows)
    if any(type(row) is not ChampionTaskRow for row in supplied_rows):
        _fail("Numerical fitting requires exact ChampionTaskRow values")
    rows = tuple(
        sorted(supplied_rows, key=lambda row: (row.task_id, row.candidate_name))
    )
    if any(row.split != "build" for row in rows):
        _fail("Numerical fitting accepts Build rows only")
    if type(fold_manifest) is not GroupFoldManifest or fold_manifest.fold_count != 5:
        _fail("Numerical fitting requires an exact five-fold group manifest")
    if type(parent) is not ChampionRelease:
        _fail("Numerical fitting requires an exact ChampionRelease Parent")

    task_ids = tuple(sorted({row.task_id for row in rows}))
    if len(task_ids) != 64 or set(task_ids) != set(fold_manifest.task_fold_map):
        _fail("Numerical fitting requires the exact registered 64-task Build universe")
    task_folds = dict(fold_manifest.task_fold_map)
    if any(row.fold != task_folds[row.task_id] for row in rows):
        _fail("Build row folds do not match the registered group manifest")

    try:
        full_build_policy = fit_champion_recipe(recipe, rows, parent)
        fold_policies: list[tuple[int, FittedChampionPolicy]] = []
        memberships: list[tuple[int, tuple[str, ...]]] = []
        for fold in range(5):
            held_out = {
                task_id
                for _group_sha, group_tasks, group_fold in fold_manifest.groups
                if group_fold == fold
                for task_id in group_tasks
            }
            training_ids = tuple(
                task_id for task_id in task_ids if task_id not in held_out
            )
            if not training_ids or set(training_ids) & held_out:
                _fail("Numerical fold training membership overlaps held-out groups")
            training_set = set(training_ids)
            training_rows = tuple(row for row in rows if row.task_id in training_set)
            fold_policies.append(
                (fold, fit_champion_recipe(recipe, training_rows, parent))
            )
            memberships.append((fold, training_ids))
    except NumericalPackageEvolutionError:
        raise
    except Exception as error:
        raise NumericalPackageEvolutionError("Numerical host fitting failed") from error

    score_sha256 = champion_fingerprint(
        {
            "recipe": recipe,
            "parent": parent,
            "manifest": fold_manifest.to_payload(),
            "full_build_policy": full_build_policy,
            "build_fold_policies": fold_policies,
            "full_build_task_ids": task_ids,
            "fold_training_task_ids": memberships,
        }
    )
    return NumericalRecipeFit(
        recipe=recipe,
        full_build_policy=full_build_policy,
        build_fold_policies=tuple(fold_policies),
        full_build_task_ids=task_ids,
        fold_training_task_ids=tuple(memberships),
        numerical_score_sha256=score_sha256,
    )


@dataclass(frozen=True)
class NumericalCoordinateCandidate:
    release: NumericalSupplyRelease
    registry: FrozenNumericalPackageRegistry
    proposal_sha256: str
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.release) is not NumericalSupplyRelease:
            _fail("candidate release must be an exact NumericalSupplyRelease")
        if type(self.registry) is not FrozenNumericalPackageRegistry:
            _fail("candidate registry must be an exact frozen registry")
        if self.registry.release_sha256 != self.release.fingerprint:
            _fail("candidate registry release mismatch")
        if (
            type(self.proposal_sha256) is not str
            or _SHA256.fullmatch(self.proposal_sha256) is None
        ):
            _fail("candidate proposal fingerprint must be canonical")
        if self.invalid_reason is not None and (
            type(self.invalid_reason) is not str or not self.invalid_reason
        ):
            _fail("candidate invalid reason must be a nonempty string or None")


def _history_diagnostic(diagnostic: CandidateDiagnostics) -> CandidateDiagnostics:
    return replace(
        diagnostic,
        folds=(),
        fold_forecasts=(),
        fold_truths=(),
        cache_key="",
        long_horizon_fold=None,
    )


def _sanitized_context_task(task: ContextTask) -> ContextTask:
    if type(task) is not ContextTask:
        _fail("Numerical materialization tasks must be exact ContextTask records")
    return replace(
        task,
        numeric=task.numeric_view(),
        documents=tuple(
            replace(document, role=None, subtype=None) for document in task.documents
        ),
        gt_evidence=(),
        labels_public=False,
    )


def _is_label_free(task: ContextTask) -> bool:
    return bool(
        type(task) is ContextTask
        and not task.numeric.future_values
        and not task.gt_evidence
        and task.labels_public is False
        and all(
            document.role is None and document.subtype is None
            for document in task.documents
        )
    )


class NumericalPackageMaterializer:
    """Build one complete registry from frozen fitted Numerical policies."""

    def __init__(
        self,
        *,
        forecast_store: ForecastStore,
        screening_policy: ScreeningPolicy,
        fold_manifest: GroupFoldManifest,
        original_tasks: Sequence[ContextTask],
        source_fingerprints: Mapping[str, str],
        runtime_fingerprints: Mapping[str, str],
        combined_policies: Sequence[CombinedPolicy] = (),
        atlas_release: AtlasRelease | None = None,
        decision_policy: DecisionPolicy = DecisionPolicy(),
        hindcast_config: HindcastConfig = HindcastConfig(),
        diagnostics_by_task: (
            Mapping[str, Mapping[str, CandidateDiagnostics]] | None
        ) = None,
    ) -> None:
        if not hasattr(forecast_store, "forecast") or not callable(
            forecast_store.forecast
        ):
            _fail("Numerical materializer requires a ForecastStore-compatible boundary")
        if type(screening_policy) is not ScreeningPolicy:
            _fail("Numerical materializer requires an exact ScreeningPolicy")
        if (
            type(fold_manifest) is not GroupFoldManifest
            or fold_manifest.fold_count != 5
        ):
            _fail("Numerical materializer requires an exact five-fold manifest")
        tasks = tuple(original_tasks)
        if not tasks or any(type(task) is not ContextTask for task in tasks):
            _fail("Numerical materializer requires exact host-held ContextTask records")
        task_ids = tuple(task.numeric.task_id for task in tasks)
        if len(task_ids) != len(set(task_ids)):
            _fail("Numerical materializer host task identities must be unique")
        if not set(fold_manifest.task_fold_map).issubset(task_ids):
            _fail("Numerical materializer is missing registered Build tasks")
        policies = tuple(combined_policies)
        if any(type(policy) is not CombinedPolicy for policy in policies):
            _fail("Numerical materializer Combined policies must be exact")
        if atlas_release is not None and type(atlas_release) is not AtlasRelease:
            _fail("Numerical materializer Atlas release must be exact or None")
        if (
            type(decision_policy) is not DecisionPolicy
            or type(hindcast_config) is not HindcastConfig
        ):
            _fail("Numerical materializer runtime policies must be exact")
        diagnostics = dict(diagnostics_by_task or {})
        if set(diagnostics) - set(task_ids) or any(
            not isinstance(value, Mapping)
            or any(type(item) is not CandidateDiagnostics for item in value.values())
            for value in diagnostics.values()
        ):
            _fail("Numerical materializer diagnostics are not task-bound")
        self.forecast_store = forecast_store
        self.screening_policy = screening_policy
        self.fold_manifest = fold_manifest
        self.original_tasks = tasks
        self.source_fingerprints = _fingerprints(
            source_fingerprints, "source_fingerprints"
        )
        self.runtime_fingerprints = _fingerprints(
            runtime_fingerprints, "runtime_fingerprints"
        )
        self.combined_policies = policies
        self.atlas_release = atlas_release
        self.decision_policy = decision_policy
        self.hindcast_config = hindcast_config
        self.diagnostics_by_task = diagnostics

    def _family(self, recipe: ChampionRecipe) -> str:
        if recipe.kind != "select":
            return "combined"
        entry = self.screening_policy.get(recipe.parents[0])
        if entry is None or entry.status not in {"keep", "specialized"}:
            _fail(
                "fitted recipe references a candidate outside the reviewed Dictionary"
            )
        return entry.family

    def _release(
        self,
        parent: NumericalSupplyRelease,
        fit: NumericalRecipeFit,
        *,
        version: str,
    ) -> NumericalSupplyRelease:
        family = self._family(fit.recipe)
        alternative = NumericalAlternativeSpec(
            candidate_id=fit.recipe.name,
            family=family,  # type: ignore[arg-type]
            materializer_kind="champion",
            recipe_payload=fit.recipe.to_payload(),
            full_build_policy_payload=fit.full_build_policy.to_payload(),
            build_fold_policy_payloads=tuple(
                (fold, policy.to_payload()) for fold, policy in fit.build_fold_policies
            ),
            assumption_ids=tuple(item.assumption_id for item in fit.recipe.assumptions),
            failure_conditions=tuple(
                item.failure_condition for item in fit.recipe.assumptions
            ),
        )
        retained = [
            item
            for item in parent.alternatives
            if item.family != family and item.candidate_id != alternative.candidate_id
        ]
        retained.append(alternative)
        if self.atlas_release is not None:
            retained = [item for item in retained if item.family != "atlas_overlay"]
            retained.append(
                NumericalAlternativeSpec(
                    candidate_id="atlas_70_30",
                    family="atlas_overlay",
                    materializer_kind="atlas",
                    recipe_payload=fit.recipe.to_payload(),
                    full_build_policy_payload=fit.full_build_policy.to_payload(),
                    build_fold_policy_payloads=tuple(
                        (fold, policy.to_payload())
                        for fold, policy in fit.build_fold_policies
                    ),
                    assumption_ids=tuple(
                        item.assumption_id for item in fit.recipe.assumptions
                    ),
                    failure_conditions=tuple(
                        item.failure_condition for item in fit.recipe.assumptions
                    ),
                )
            )
        order = {"statistical": 0, "tsfm": 1, "combined": 2, "atlas_overlay": 3}
        alternatives = tuple(sorted(retained, key=lambda item: order[item.family]))
        try:
            return NumericalSupplyRelease(
                schema_version=1,
                version=version,
                parent_sha256=parent.fingerprint,
                anchor_release_payload=cast(
                    dict[str, object], _plain(parent.anchor_release_payload)
                ),
                alternatives=alternatives,
                atlas_release_sha256=(
                    self.atlas_release.fingerprint
                    if self.atlas_release is not None
                    else parent.atlas_release_sha256
                ),
                source_fingerprints={
                    **self.source_fingerprints,
                    "numerical_fit": fit.numerical_score_sha256,
                },
                runtime_fingerprints=self.runtime_fingerprints,
            )
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical materializer could not bind the fitted release"
            ) from error

    def _policy_for_task(
        self, fit: NumericalRecipeFit, task_id: str
    ) -> FittedChampionPolicy:
        fold = self.fold_manifest.task_fold_map.get(task_id)
        return (
            fit.full_build_policy
            if fold is None
            else dict(fit.build_fold_policies)[fold]
        )

    def _proposal_forecast(
        self,
        source,
        fit: NumericalRecipeFit,
        task: ContextTask,
        family: str,
    ) -> RankedNumericalForecast | None:
        forecasts = {item.name: item.forecast for item in source.ranked_alternatives}
        diagnostics = {
            item.name: _history_diagnostic(item.diagnostics)
            for item in source.ranked_alternatives
        }
        try:
            execution = execute_champion(
                self._policy_for_task(fit, task.numeric.task_id),
                forecasts,
                diagnostics,
                source.task_profile,
                task.numeric.history_values,
                task.numeric.prediction_length,
            )
        except Exception:
            return None
        if not valid_forecast(execution.forecast, task.numeric.prediction_length):
            return None
        proxy = diagnostics.get(fit.recipe.fallback_parent)
        if proxy is None:
            return None
        diagnostic = replace(
            proxy,
            name=fit.recipe.name,
            family=family,
            folds=(),
            successful_folds=0,
            eligible=False,
            reason_code="frozen_package_recipe",
            fold_forecasts=(),
            fold_truths=(),
            cache_key="",
            long_horizon_fold=None,
        )
        return RankedNumericalForecast(
            rank=len(source.ranked_alternatives) + 1,
            name=fit.recipe.name,
            family=family,
            forecast=tuple(execution.forecast),
            diagnostics=diagnostic,
        )

    def _atlas_forecast(
        self,
        source,
        task: ContextTask,
    ) -> RankedNumericalForecast | None:
        if self.atlas_release is None:
            return None
        by_name = {item.name: item for item in source.ranked_alternatives}
        anchor_name = self.atlas_release.full_build_model.selected_pool[0]
        anchor = by_name.get(anchor_name)
        if anchor is None:
            return None
        dummy_truth = (0.0,) * task.numeric.prediction_length

        def row(item: RankedNumericalForecast) -> TaskLocalTaskRow:
            return TaskLocalTaskRow(
                task_id=task.numeric.task_id,
                candidate_name=item.name,
                family=item.family,
                profile=source.task_profile,
                history=task.numeric.history_values,
                truth=dummy_truth,
                forecast=item.forecast,
                diagnostic=item.diagnostics,
                split="train",
            )

        anchor_row = row(anchor)
        candidates: list[AtlasMaterializedCandidate] = []
        for name in self.atlas_release.full_build_model.selected_pool[1:]:
            item = by_name.get(name)
            if item is None:
                continue
            candidate_row = row(item)
            try:
                feature = atlas_feature(candidate_row, anchor_row)
            except ValueError:
                continue
            candidates.append(
                AtlasMaterializedCandidate(name, item.family, feature, item.forecast)
            )
        fold = self.fold_manifest.task_fold_map.get(task.numeric.task_id)
        group_sha256 = next(
            (
                group_sha
                for group_sha, task_ids, _fold in self.fold_manifest.groups
                if task.numeric.task_id in task_ids
            ),
            _digest({"history": list(task.numeric.history_values)}),
        )
        routed = route_atlas_task(
            AtlasTaskCase(
                group_sha256=group_sha256,
                anchor=AtlasMaterializedCandidate(
                    anchor.name, anchor.family, None, anchor.forecast
                ),
                candidates=tuple(candidates),
            ),
            self.atlas_release,
            fold=fold,
        )
        if not routed.activated or not valid_forecast(
            routed.forecast, task.numeric.prediction_length
        ):
            return None
        diagnostic = replace(
            _history_diagnostic(anchor.diagnostics),
            name="atlas_70_30",
            family="atlas_overlay",
            successful_folds=0,
            eligible=False,
            reason_code="frozen_atlas_route",
        )
        return RankedNumericalForecast(
            rank=len(source.ranked_alternatives) + 2,
            name="atlas_70_30",
            family="atlas_overlay",
            forecast=routed.forecast,
            diagnostics=diagnostic,
        )

    def materialize(
        self,
        parent_release: NumericalSupplyRelease,
        fit: NumericalRecipeFit,
        tasks: Sequence[ContextTask],
        *,
        version: str,
        generation: int,
    ) -> NumericalCoordinateCandidate:
        if type(parent_release) is not NumericalSupplyRelease:
            _fail("Numerical materialization requires an exact Parent release")
        if type(fit) is not NumericalRecipeFit:
            _fail("Numerical materialization requires an exact fitted recipe")
        NumericalRecipeFit.__post_init__(fit)
        if type(version) is not str or _VERSION.fullmatch(version) is None:
            _fail("Numerical materialization version must use nNNN format")
        if type(generation) is not int or generation < 0:
            _fail("Numerical materialization generation must be nonnegative")
        safe_tasks = tuple(tasks)
        if not safe_tasks or any(not _is_label_free(task) for task in safe_tasks):
            _fail("Numerical materialization requires exact label-free task inputs")
        safe_by_id = {task.numeric.task_id: task for task in safe_tasks}
        if len(safe_by_id) != len(safe_tasks):
            _fail("Numerical materialization task identities must be unique")
        originals = {task.numeric.task_id: task for task in self.original_tasks}
        if set(safe_by_id) != set(originals):
            _fail("Numerical materialization task universe drifted")
        for task_id, safe in safe_by_id.items():
            original = originals[task_id]
            if (
                safe.numeric != original.numeric_view()
                or safe.target_name != original.target_name
                or safe.target_description != original.target_description
                or safe.history_timestamps != original.history_timestamps
                or safe.future_timestamps != original.future_timestamps
                or tuple((item.document_id, item.content) for item in safe.documents)
                != tuple(
                    (item.document_id, item.content) for item in original.documents
                )
            ):
                _fail("Numerical materialization label-free task content drifted")

        release = self._release(parent_release, fit, version=version)
        try:
            anchor_release = parse_champion_release(
                cast(dict[str, object], _plain(parent_release.anchor_release_payload))
            )
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical Parent anchor is not a valid Champion release"
            ) from error
        family = self._family(fit.recipe)

        def package_builder(
            original: ContextTask,
            supplied_release: NumericalSupplyRelease,
        ):
            safe = safe_by_id[original.numeric.task_id]
            diagnostics = self.diagnostics_by_task.get(original.numeric.task_id)
            source = run_numerical_loop(
                RuntimeTask(
                    safe.numeric.task_id,
                    safe.numeric.history_values,
                    safe.numeric.prediction_length,
                    safe.numeric.frequency,
                    (),
                ),
                screening_policy=self.screening_policy,
                candidate_runner=self.forecast_store.forecast,
                combined_policies=self.combined_policies,
                decision_policy=self.decision_policy,
                hindcast_config=self.hindcast_config,
                diagnostics=(
                    dict(diagnostics)
                    if diagnostics is not None
                    else None
                ),
                component_fingerprints={
                    **self.source_fingerprints,
                    **self.runtime_fingerprints,
                },
                champion_release=anchor_release,
            )
            materialized = {item.name: item for item in source.ranked_alternatives}
            proposal = self._proposal_forecast(source, fit, safe, family)
            if proposal is not None:
                materialized[proposal.name] = proposal
            atlas = self._atlas_forecast(source, safe)
            if atlas is not None:
                materialized[atlas.name] = atlas
            return bound_numerical_package(source, supplied_release, materialized)

        try:
            registry = build_package_registry(
                self.original_tasks,
                release,
                package_builder,
            )
        except NumericalPackageEvolutionError:
            raise
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical package registry materialization failed"
            ) from error
        return NumericalCoordinateCandidate(
            release=release,
            registry=registry,
            proposal_sha256=_digest(
                {
                    "parent": parent_release.fingerprint,
                    "fit": fit.numerical_score_sha256,
                    "version": version,
                    "generation": generation,
                    "registry": registry.fingerprint,
                }
            ),
        )


def _structural_fingerprint(recipe: ChampionRecipe) -> str:
    return _digest(
        {
            "kind": recipe.kind,
            "parents": list(recipe.parents),
            "fallback_parent": recipe.fallback_parent,
            "assumptions": [
                {
                    "candidate_name": item.candidate_name,
                    "feature": item.feature,
                    "direction": item.direction,
                    "horizon_region": item.horizon_region,
                    "operator": item.operator,
                }
                for item in recipe.assumptions
            ],
        }
    )


class NumericalPackageProposer:
    """Propose, fit, and materialize exactly bounded direct Numerical Children."""

    def __init__(
        self,
        *,
        proposer: ChampionProposerAdapter,
        materializer: object,
        build_rows: Sequence[ChampionTaskRow],
        fold_manifest: GroupFoldManifest,
        tasks: Sequence[ContextTask],
    ) -> None:
        if type(proposer) is not ChampionProposerAdapter:
            _fail("Numerical proposer requires an exact registered adapter")
        proposer.verify()
        if not hasattr(materializer, "materialize") or not callable(
            materializer.materialize
        ):
            _fail("Numerical proposer requires a materializer boundary")
        rows = tuple(build_rows)
        if not rows or any(type(row) is not ChampionTaskRow for row in rows):
            _fail("Numerical proposer requires exact Build rows")
        evolution_tasks = tuple(tasks)
        if not evolution_tasks or any(
            type(task) is not ContextTask for task in evolution_tasks
        ):
            _fail("Numerical proposer requires exact evolution tasks")
        if len({task.numeric.task_id for task in evolution_tasks}) != len(
            evolution_tasks
        ):
            _fail("Numerical proposer evolution task identities must be unique")
        if not set(fold_manifest.task_fold_map).issubset(
            task.numeric.task_id for task in evolution_tasks
        ):
            _fail("Numerical proposer evolution tasks omit registered Build rows")
        self.proposer = proposer
        self.materializer = materializer
        self.build_rows = rows
        self.fold_manifest = fold_manifest
        self.tasks = evolution_tasks

    def propose(
        self,
        parent_release: NumericalSupplyRelease,
        parent_registry: FrozenNumericalPackageRegistry,
        feedback: ProposerEvidence,
        *,
        generation: int,
        child_count: int = 3,
    ) -> tuple[NumericalCoordinateCandidate, ...]:
        if type(parent_release) is not NumericalSupplyRelease:
            _fail("Numerical proposal requires an exact Parent release")
        if type(parent_registry) is not FrozenNumericalPackageRegistry:
            _fail("Numerical proposal requires an exact Parent registry")
        if parent_registry.release_sha256 != parent_release.fingerprint:
            _fail("Numerical proposal Parent registry release mismatch")
        if type(feedback) is not ProposerEvidence:
            _fail("Numerical proposal feedback must be exact sanitized evidence")
        if type(generation) is not int or generation < 0:
            _fail("Numerical proposal generation must be nonnegative")
        if type(child_count) is not int or not 1 <= child_count <= 3:
            _fail("Numerical proposal child count must be within one through three")
        try:
            sanitized_feedback = ProposerEvidence(
                label=feedback.label,
                independent_generalization_claim=False,
                morphology=tuple(feedback.morphology),
                comparisons=tuple(feedback.comparisons),
                invalid_attempts=tuple(feedback.invalid_attempts),
                structures=tuple(feedback.structures),
            )
            sanitized_feedback.to_payload()
            parent_anchor = parse_champion_release(
                cast(dict[str, object], _plain(parent_release.anchor_release_payload))
            )
            recipes = self.proposer.propose(
                parent_anchor,
                sanitized_feedback,
                generation=generation + 1,
            )
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical structural proposal failed"
            ) from error

        reviewed_names = {row.candidate_name for row in self.build_rows}
        unique: list[ChampionRecipe] = []
        seen_structures: set[str] = set()
        for recipe in sorted(recipes, key=champion_fingerprint):
            if type(recipe) is not ChampionRecipe or any(
                parent not in reviewed_names for parent in recipe.parents
            ):
                continue
            structure = _structural_fingerprint(recipe)
            if structure in seen_structures:
                continue
            seen_structures.add(structure)
            unique.append(recipe)

        safe_tasks = tuple(_sanitized_context_task(task) for task in self.tasks)
        children: list[NumericalCoordinateCandidate] = []
        attempted: list[str] = []
        for recipe in unique:
            if len(children) == child_count:
                break
            proposal_sha256 = _digest(
                {
                    "parent": parent_release.fingerprint,
                    "generation": generation,
                    "structure": _structural_fingerprint(recipe),
                }
            )
            attempted.append(proposal_sha256)
            try:
                fit = fit_numerical_recipe(
                    recipe,
                    self.build_rows,
                    self.fold_manifest,
                    parent_anchor,
                )
                slot = len(children)
                candidate = self.materializer.materialize(
                    parent_release,
                    fit,
                    safe_tasks,
                    version=f"n{generation * 3 + slot + 1:03d}",
                    generation=generation,
                )
                if (
                    type(candidate) is not NumericalCoordinateCandidate
                    or candidate.invalid_reason is not None
                    or candidate.release.parent_sha256 != parent_release.fingerprint
                    or candidate.registry.release_sha256
                    != candidate.release.fingerprint
                ):
                    continue
                children.append(replace(candidate, proposal_sha256=proposal_sha256))
            except Exception:
                continue

        while len(children) < child_count:
            slot = len(children)
            invalid_sha256 = _digest(
                {
                    "parent": parent_release.fingerprint,
                    "generation": generation,
                    "slot": slot,
                    "attempted": attempted,
                    "invalid": "insufficient_valid_proposals",
                }
            )
            children.append(
                NumericalCoordinateCandidate(
                    release=parent_release,
                    registry=parent_registry,
                    proposal_sha256=invalid_sha256,
                    invalid_reason="insufficient_valid_proposals",
                )
            )
        return tuple(children)


__all__ = [
    "NumericalCoordinateCandidate",
    "NumericalPackageMaterializer",
    "NumericalPackageProposer",
    "NumericalPackageEvolutionError",
    "NumericalRecipeFit",
    "fit_numerical_recipe",
]
