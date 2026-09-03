"""Trusted proposal, fitting, and materialization of Numerical package states."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
    _parse_fitted_policy,
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
    validate_proposer_evidence,
)
from numerical_agent.evolution.champion_runtime import execute_champion
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_package import (
    RankedNumericalForecast,
    snapshot_diagnostics,
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
    parent_sha256: str
    fold_manifest_sha256: str
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
            or self.full_build_task_ids != tuple(sorted(self.full_build_task_ids))
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
                or task_ids != tuple(sorted(task_ids))
                or not set(task_ids) < universe
            ):
                _fail("Numerical fold training membership is incomplete")
        omitted = [
            universe - set(task_ids) for _fold, task_ids in self.fold_training_task_ids
        ]
        if set().union(*omitted) != universe or sum(
            len(task_ids) for task_ids in omitted
        ) != len(universe):
            _fail("Numerical fold training complements are not disjoint and complete")
        for value, label in (
            (self.parent_sha256, "Parent"),
            (self.fold_manifest_sha256, "fold manifest"),
            (self.numerical_score_sha256, "score"),
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                _fail(f"Numerical fit {label} must be a lowercase SHA-256 value")


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
        parent_sha256=champion_fingerprint(parent),
        fold_manifest_sha256=_digest(fold_manifest.to_payload()),
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


def _diagnostic_task_sha256(task: ContextTask) -> str:
    return _digest(
        {
            "task_id": task.numeric.task_id,
            "history_values": list(task.numeric.history_values),
            "prediction_length": task.numeric.prediction_length,
            "frequency": task.numeric.frequency,
        }
    )


def _hindcast_policy_sha256(policy: HindcastConfig) -> str:
    return _digest({"hindcast_config": asdict(policy)})


def _diagnostic_sha256(diagnostic: CandidateDiagnostics) -> str:
    return _digest({"diagnostic": asdict(diagnostic)})


@dataclass(frozen=True)
class _NumericalDiagnosticsRecord:
    task_id: str
    task_history_sha256: str
    candidate_name: str
    diagnostic: CandidateDiagnostics
    diagnostic_sha256: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            _fail("diagnostics record requires a task identity")
        if type(self.candidate_name) is not str or not self.candidate_name:
            _fail("diagnostics record requires a candidate identity")
        if (
            type(self.diagnostic) is not CandidateDiagnostics
            or self.diagnostic.name != self.candidate_name
        ):
            _fail("diagnostics record candidate binding mismatch")
        if (
            type(self.task_history_sha256) is not str
            or _SHA256.fullmatch(self.task_history_sha256) is None
        ):
            _fail("diagnostics record history fingerprint is invalid")
        if self.diagnostic_sha256 != _diagnostic_sha256(self.diagnostic):
            _fail("diagnostics record content fingerprint mismatch")


@dataclass(frozen=True)
class FrozenNumericalDiagnosticsRegistry:
    """Content-addressed history-only hindcast diagnostics authority."""

    records: tuple[_NumericalDiagnosticsRecord, ...]
    hindcast_policy_sha256: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(
            type(record) is not _NumericalDiagnosticsRecord for record in self.records
        ):
            _fail("frozen diagnostics registry requires exact records")
        keys: list[tuple[str, str]] = []
        for record in self.records:
            _NumericalDiagnosticsRecord.__post_init__(record)
            keys.append((record.task_id, record.candidate_name))
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            _fail("frozen diagnostics records must be unique and sorted")
        if (
            type(self.hindcast_policy_sha256) is not str
            or _SHA256.fullmatch(self.hindcast_policy_sha256) is None
        ):
            _fail("frozen diagnostics hindcast fingerprint is invalid")
        object.__setattr__(
            self,
            "fingerprint",
            _digest(
                {
                    "hindcast_policy_sha256": self.hindcast_policy_sha256,
                    "records": [
                        {
                            "task_id": record.task_id,
                            "task_history_sha256": record.task_history_sha256,
                            "candidate_name": record.candidate_name,
                            "diagnostic_sha256": record.diagnostic_sha256,
                        }
                        for record in self.records
                    ],
                }
            ),
        )

    @classmethod
    def build(
        cls,
        tasks: Sequence[ContextTask],
        diagnostics_by_task: Mapping[str, Mapping[str, CandidateDiagnostics]],
        hindcast_policy: HindcastConfig,
    ) -> FrozenNumericalDiagnosticsRegistry:
        if type(hindcast_policy) is not HindcastConfig:
            _fail("frozen diagnostics require an exact hindcast policy")
        supplied_tasks = tuple(tasks)
        if not supplied_tasks or any(
            type(task) is not ContextTask for task in supplied_tasks
        ):
            _fail("frozen diagnostics require exact host task records")
        by_id = {task.numeric.task_id: task for task in supplied_tasks}
        if len(by_id) != len(supplied_tasks):
            _fail("frozen diagnostics task identities must be unique")
        if not isinstance(diagnostics_by_task, Mapping):
            _fail("frozen diagnostics source must be a task mapping")
        if set(diagnostics_by_task) - set(by_id):
            _fail("frozen diagnostics contain an unregistered task")
        records: list[_NumericalDiagnosticsRecord] = []
        for task_id in sorted(diagnostics_by_task):
            raw = diagnostics_by_task[task_id]
            if not isinstance(raw, Mapping):
                _fail("frozen diagnostics task values must be candidate mappings")
            active: list[tuple[str, str]] = []
            for name in sorted(raw):
                diagnostic = raw[name]
                if (
                    type(name) is not str
                    or type(diagnostic) is not CandidateDiagnostics
                ):
                    _fail("frozen diagnostics contain an invalid candidate record")
                active.append((name, diagnostic.family))
            frozen = snapshot_diagnostics(raw, tuple(active))
            task_sha256 = _diagnostic_task_sha256(by_id[task_id])
            for name in sorted(frozen):
                diagnostic = frozen[name]
                records.append(
                    _NumericalDiagnosticsRecord(
                        task_id=task_id,
                        task_history_sha256=task_sha256,
                        candidate_name=name,
                        diagnostic=diagnostic,
                        diagnostic_sha256=_diagnostic_sha256(diagnostic),
                    )
                )
        return cls(
            records=tuple(records),
            hindcast_policy_sha256=_hindcast_policy_sha256(hindcast_policy),
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.task_id for record in self.records}))

    def diagnostics_for(
        self,
        task: ContextTask,
        hindcast_policy: HindcastConfig,
    ) -> dict[str, CandidateDiagnostics]:
        FrozenNumericalDiagnosticsRegistry.__post_init__(self)
        if type(task) is not ContextTask:
            _fail("frozen diagnostics lookup requires an exact task")
        if (
            type(hindcast_policy) is not HindcastConfig
            or _hindcast_policy_sha256(hindcast_policy) != self.hindcast_policy_sha256
        ):
            _fail("frozen diagnostics hindcast policy mismatch")
        records = tuple(
            record for record in self.records if record.task_id == task.numeric.task_id
        )
        if not records:
            return {}
        task_sha256 = _diagnostic_task_sha256(task)
        if any(record.task_history_sha256 != task_sha256 for record in records):
            _fail("frozen diagnostics task history mismatch")
        return {record.candidate_name: record.diagnostic for record in records}


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
        diagnostics_registry: object | None = None,
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
            or len(fold_manifest.task_fold_map) != 64
        ):
            _fail(
                "Numerical materializer requires the exact 64-task five-fold manifest"
            )
        tasks = tuple(original_tasks)
        if len(tasks) != 100 or any(type(task) is not ContextTask for task in tasks):
            _fail(
                "Numerical materializer requires the exact 100 host-held Train+Dev tasks"
            )
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
        if atlas_release is not None:
            try:
                atlas_release.validate_manifest(fold_manifest)
            except ValueError as error:
                raise NumericalPackageEvolutionError(
                    "Numerical materializer Atlas fold manifest mismatch"
                ) from error
        if (
            type(decision_policy) is not DecisionPolicy
            or type(hindcast_config) is not HindcastConfig
        ):
            _fail("Numerical materializer runtime policies must be exact")
        if (
            diagnostics_registry is not None
            and type(diagnostics_registry) is not FrozenNumericalDiagnosticsRegistry
        ):
            _fail("Numerical materializer requires a frozen diagnostics registry")
        if type(diagnostics_registry) is FrozenNumericalDiagnosticsRegistry:
            FrozenNumericalDiagnosticsRegistry.__post_init__(diagnostics_registry)
            if set(diagnostics_registry.task_ids) - set(task_ids):
                _fail("Numerical materializer diagnostics are not task-bound")
            if diagnostics_registry.hindcast_policy_sha256 != _hindcast_policy_sha256(
                hindcast_config
            ):
                _fail("Numerical materializer diagnostics hindcast policy mismatch")
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
        self.diagnostics_registry = diagnostics_registry

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
                    **(
                        {
                            "diagnostics_registry": (
                                self.diagnostics_registry.fingerprint
                            )
                        }
                        if type(self.diagnostics_registry)
                        is FrozenNumericalDiagnosticsRegistry
                        else {}
                    ),
                },
                runtime_fingerprints=self.runtime_fingerprints,
            )
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical materializer could not bind the fitted release"
            ) from error

    def _validated_fit_binding(
        self,
        parent_release: NumericalSupplyRelease,
        fit: NumericalRecipeFit,
    ) -> ChampionRelease:
        try:
            anchor_release = parse_champion_release(
                cast(dict[str, object], _plain(parent_release.anchor_release_payload))
            )
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical Parent anchor is not a valid Champion release"
            ) from error
        if fit.parent_sha256 != champion_fingerprint(anchor_release):
            _fail("Numerical recipe fit does not bind the supplied Parent")
        manifest_sha256 = _digest(self.fold_manifest.to_payload())
        if fit.fold_manifest_sha256 != manifest_sha256:
            _fail("Numerical recipe fit fold manifest mismatch")
        manifest_ids = set(self.fold_manifest.task_fold_map)
        if set(fit.full_build_task_ids) != manifest_ids:
            _fail("Numerical recipe fit Build membership mismatch")
        memberships = dict(fit.fold_training_task_ids)
        for fold in range(5):
            held_out = {
                task_id
                for task_id, assigned_fold in self.fold_manifest.task_fold_map.items()
                if assigned_fold == fold
            }
            if set(memberships[fold]) != manifest_ids - held_out:
                _fail("Numerical recipe fit fold training membership mismatch")
        expected_score = champion_fingerprint(
            {
                "recipe": fit.recipe,
                "parent": anchor_release,
                "manifest": self.fold_manifest.to_payload(),
                "full_build_policy": fit.full_build_policy,
                "build_fold_policies": list(fit.build_fold_policies),
                "full_build_task_ids": fit.full_build_task_ids,
                "fold_training_task_ids": list(fit.fold_training_task_ids),
            }
        )
        if fit.numerical_score_sha256 != expected_score:
            _fail("Numerical recipe fit content fingerprint mismatch")
        return anchor_release

    def _proposal_forecast(
        self,
        source,
        policy: FittedChampionPolicy,
        task: ContextTask,
        candidate_id: str,
        family: str,
    ) -> RankedNumericalForecast | None:
        forecasts = {item.name: item.forecast for item in source.ranked_alternatives}
        diagnostics = {
            item.name: _history_diagnostic(item.diagnostics)
            for item in source.ranked_alternatives
        }
        try:
            execution = execute_champion(
                policy,
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
        proxy = diagnostics.get(policy.recipe.fallback_parent)
        if proxy is None:
            return None
        diagnostic = replace(
            proxy,
            name=candidate_id,
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
            name=candidate_id,
            family=family,
            forecast=tuple(execution.forecast),
            diagnostics=diagnostic,
        )

    def _atlas_forecast(
        self,
        source,
        task: ContextTask,
        *,
        candidate_id: str,
    ) -> RankedNumericalForecast | None:
        if self.atlas_release is None:
            return None
        by_name = {item.name: item for item in source.ranked_alternatives}
        fold = self.fold_manifest.task_fold_map.get(task.numeric.task_id)
        model = (
            self.atlas_release.full_build_model
            if fold is None
            else dict(self.atlas_release.build_fold_models)[fold]
        )
        anchor_name = model.selected_pool[0]
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
        for name in model.selected_pool[1:]:
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
            name=candidate_id,
            family="atlas_overlay",
            successful_folds=0,
            eligible=False,
            reason_code="frozen_atlas_route",
        )
        return RankedNumericalForecast(
            rank=len(source.ranked_alternatives) + 2,
            name=candidate_id,
            family="atlas_overlay",
            forecast=routed.forecast,
            diagnostics=diagnostic,
        )

    def _stored_policy_for_task(
        self,
        specification: NumericalAlternativeSpec,
        task_id: str,
    ) -> FittedChampionPolicy:
        fold = self.fold_manifest.task_fold_map.get(task_id)
        payload = (
            specification.full_build_policy_payload
            if fold is None
            else dict(specification.build_fold_policy_payloads)[fold]
        )
        return _parse_fitted_policy(_plain(payload))

    def _materialize_alternative(
        self,
        source,
        task: ContextTask,
        specification: NumericalAlternativeSpec,
    ) -> RankedNumericalForecast | None:
        if specification.materializer_kind == "dictionary":
            return next(
                (
                    item
                    for item in source.ranked_alternatives
                    if item.name == specification.candidate_id
                    and item.family == specification.family
                ),
                None,
            )
        if specification.materializer_kind == "atlas":
            return self._atlas_forecast(
                source,
                task,
                candidate_id=specification.candidate_id,
            )
        if specification.materializer_kind in {"champion", "bounded_overlay"}:
            try:
                policy = self._stored_policy_for_task(
                    specification, task.numeric.task_id
                )
            except Exception:
                return None
            return self._proposal_forecast(
                source,
                policy,
                task,
                specification.candidate_id,
                specification.family,
            )
        return None

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

        anchor_release = self._validated_fit_binding(parent_release, fit)
        release = self._release(parent_release, fit, version=version)

        def package_builder(
            original: ContextTask,
            supplied_release: NumericalSupplyRelease,
        ):
            safe = safe_by_id[original.numeric.task_id]
            diagnostics = (
                self.diagnostics_registry.diagnostics_for(safe, self.hindcast_config)
                if type(self.diagnostics_registry) is FrozenNumericalDiagnosticsRegistry
                else {}
            )
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
                diagnostics=(diagnostics if diagnostics else None),
                component_fingerprints={
                    **self.source_fingerprints,
                    **self.runtime_fingerprints,
                },
                champion_release=anchor_release,
            )
            materialized = {item.name: item for item in source.ranked_alternatives}
            for specification in supplied_release.alternatives:
                alternative = self._materialize_alternative(
                    source,
                    safe,
                    specification,
                )
                if alternative is not None:
                    materialized[alternative.name] = alternative
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
        if len(evolution_tasks) != 100 or any(
            type(task) is not ContextTask for task in evolution_tasks
        ):
            _fail("Numerical proposer requires the exact 100 Train+Dev tasks")
        if len({task.numeric.task_id for task in evolution_tasks}) != len(
            evolution_tasks
        ):
            _fail("Numerical proposer evolution task identities must be unique")
        if (
            type(fold_manifest) is not GroupFoldManifest
            or fold_manifest.fold_count != 5
            or len(fold_manifest.task_fold_map) != 64
        ):
            _fail("Numerical proposer requires the exact 64-task five-fold manifest")
        if {row.task_id for row in rows} != set(fold_manifest.task_fold_map):
            _fail("Numerical proposer Build row universe does not match the manifest")
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
        if type(child_count) is not int or child_count != 3:
            _fail("formal Numerical proposal requires exactly three Child slots")
        expected_task_ids = tuple(sorted(task.numeric.task_id for task in self.tasks))
        if parent_registry.task_ids != expected_task_ids:
            _fail("Numerical proposal tasks do not match the Parent registry")
        build_task_ids = tuple(sorted(self.fold_manifest.task_fold_map))
        try:
            sanitized_feedback = ProposerEvidence(
                label=feedback.label,
                independent_generalization_claim=False,
                morphology=tuple(feedback.morphology),
                comparisons=tuple(feedback.comparisons),
                invalid_attempts=tuple(feedback.invalid_attempts),
                structures=tuple(feedback.structures),
            )
            validate_proposer_evidence(sanitized_feedback, build_task_ids)
        except Exception as error:
            raise NumericalPackageEvolutionError(
                "Numerical proposal feedback identity validation failed"
            ) from error
        try:
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
        accepted_release_sha256s: set[str] = set()
        accepted_proposal_sha256s: set[str] = set()
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
                expected_version = f"n{generation * 3 + slot + 1:03d}"
                candidate = self.materializer.materialize(
                    parent_release,
                    fit,
                    safe_tasks,
                    version=expected_version,
                    generation=generation,
                )
                if type(candidate) is NumericalCoordinateCandidate:
                    NumericalCoordinateCandidate.__post_init__(candidate)
                release_sha256 = (
                    candidate.release.fingerprint
                    if type(candidate) is NumericalCoordinateCandidate
                    else ""
                )
                if (
                    type(candidate) is not NumericalCoordinateCandidate
                    or candidate.invalid_reason is not None
                    or candidate.release.version != expected_version
                    or candidate.release.parent_sha256 != parent_release.fingerprint
                    or candidate.registry.release_sha256
                    != candidate.release.fingerprint
                    or release_sha256 in accepted_release_sha256s
                    or proposal_sha256 in accepted_proposal_sha256s
                ):
                    continue
                children.append(replace(candidate, proposal_sha256=proposal_sha256))
                accepted_release_sha256s.add(release_sha256)
                accepted_proposal_sha256s.add(proposal_sha256)
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
    "FrozenNumericalDiagnosticsRegistry",
    "NumericalCoordinateCandidate",
    "NumericalPackageMaterializer",
    "NumericalPackageProposer",
    "NumericalPackageEvolutionError",
    "NumericalRecipeFit",
    "fit_numerical_recipe",
]
