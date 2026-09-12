"""Train-only grouping and evolution helpers for task-local Numerical ensembles."""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from common.data import Task as DataTask
from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile
from common.payload import canonical_json_bytes

from .execution import Task as RuntimeTask
from .numerical_selector import CandidateDiagnostics
from .screening import TaskProfile, profile_task
from .task_local_confidence import (
    ConfidenceEvidenceRecord,
    ConfidencePolicy,
    HierarchicalEvidenceBank,
    WeightRecipe,
    beta_win_probability,
    coarse_morphology_key,
    exact_morphology_key,
    robust_effect_margin,
)
from .task_local_ensemble import (
    GroupCandidateSupply,
    TaskLocalEnsembleRelease,
    TaskLocalEnsembleReleaseV3,
    TaskLocalTournamentPolicy,
    execute_confidence_task_local_ensemble,
    execute_task_local_ensemble,
    task_local_fingerprint,
)
from .task_shortlist import CandidatePriorV1, fit_candidate_priors


_GROUPING_SCHEMA = 1
_GROUPING_IMPLEMENTATION = hashlib.sha256(
    b"task-local-group-folds-v1:entity+history+source:balanced-morphology"
).hexdigest()


def _canonical_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class GroupFoldManifest:
    """Immutable assignment of indivisible history-only task groups to folds."""

    schema_version: int
    seed: int
    fold_count: int
    groups: tuple[tuple[str, tuple[str, ...], int], ...]
    grouping_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _GROUPING_SCHEMA:
            raise ValueError("group fold manifest schema must be exactly one")
        if type(self.seed) is not int:
            raise ValueError("group fold seed must be an exact integer")
        if type(self.fold_count) is not int or self.fold_count < 2:
            raise ValueError("group fold count must be at least two")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("group fold manifest requires groups")
        seen_tasks: set[str] = set()
        seen_groups: set[str] = set()
        previous = ""
        for item in self.groups:
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("group rows must be exact triples")
            group_sha, task_ids, fold = item
            if (
                type(group_sha) is not str
                or len(group_sha) != 64
                or group_sha <= previous
                or group_sha in seen_groups
            ):
                raise ValueError("group identities must be unique sorted SHA-256 values")
            if (
                type(task_ids) is not tuple
                or not task_ids
                or tuple(sorted(task_ids)) != task_ids
                or any(type(task_id) is not str or not task_id for task_id in task_ids)
                or seen_tasks.intersection(task_ids)
            ):
                raise ValueError("group task memberships must be unique sorted identifiers")
            if type(fold) is not int or not 0 <= fold < self.fold_count:
                raise ValueError("group fold is outside the registered fold universe")
            previous = group_sha
            seen_groups.add(group_sha)
            seen_tasks.update(task_ids)
        if set(range(self.fold_count)) != {fold for _sha, _tasks, fold in self.groups}:
            raise ValueError("every registered fold must contain at least one group")
        if self.grouping_fingerprint != _GROUPING_IMPLEMENTATION:
            raise ValueError("grouping implementation fingerprint mismatch")

    @property
    def task_fold_map(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                task_id: fold
                for _group_sha, task_ids, fold in self.groups
                for task_id in task_ids
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "fold_count": self.fold_count,
            "groups": [
                {"group_sha256": group_sha, "task_ids": list(task_ids), "fold": fold}
                for group_sha, task_ids, fold in self.groups
            ],
            "grouping_fingerprint": self.grouping_fingerprint,
        }


class _DisjointSet:
    def __init__(self, task_ids: Sequence[str]) -> None:
        self.parent = {task_id: task_id for task_id in task_ids}

    def find(self, task_id: str) -> str:
        parent = self.parent[task_id]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while task_id != parent:
            next_id = self.parent[task_id]
            self.parent[task_id] = parent
            task_id = next_id
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _history_fingerprint(task: DataTask) -> str:
    return _sha256({"history": list(task.history_values)})


def _task_profile(task: DataTask) -> TaskProfile:
    return profile_task(
        RuntimeTask(
            task.task_id,
            tuple(task.history_values),
            task.prediction_length,
            task.frequency,
            (),
        )
    )


def task_morphology_key(profile: TaskProfile) -> str:
    """Return a stable task-identity-free morphology bucket."""
    if type(profile) is not TaskProfile:
        raise TypeError("morphology grouping requires an exact TaskProfile")
    history_bucket = (
        "short" if profile.history_length < 64 else
        "medium" if profile.history_length < 256 else
        "long"
    )
    ratio = profile.horizon / profile.history_length
    horizon_bucket = "short" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "long"
    trend = (
        profile.trend_direction
        if profile.trend_strength >= 0.35 and profile.trend_direction != "flat"
        else "flat"
    )
    payload = {
        "frequency": _canonical_identity(profile.frequency),
        "history": history_bucket,
        "horizon": horizon_bucket,
        "trend": trend,
        "periodic": bool(
            profile.periodicity_periods and profile.periodicity_confidence >= 0.5
        ),
        "intermittent": bool(
            profile.zero_fraction >= 0.5 or profile.intermittency_adi >= 1.32
        ),
        "recent_regime": bool(
            profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.5
        ),
        "signed": profile.signed,
    }
    return _sha256(payload)


def build_group_fold_manifest(
    tasks: Sequence[DataTask],
    *,
    seed: int,
    fold_count: int = 5,
    source_series_ids: Mapping[str, str] | None = None,
) -> GroupFoldManifest:
    """Assign history-only transitive task groups to deterministic balanced folds."""
    if type(seed) is not int:
        raise ValueError("group fold seed must be an exact integer")
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("group fold count must be at least two")
    supplied = tuple(tasks)
    if not supplied or any(type(task) is not DataTask for task in supplied):
        raise ValueError("group folds require exact numeric DataTask records")
    task_ids = tuple(task.task_id for task in supplied)
    if (
        any(type(task_id) is not str or not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("group folds require unique nonempty task IDs")
    for task in supplied:
        if (
            not task.history_values
            or any(not math.isfinite(float(value)) for value in task.history_values)
            or type(task.entity_name) is not str
            or not _canonical_identity(task.entity_name)
        ):
            raise ValueError("group folds require finite history and nonempty entity")
    source_map = dict(source_series_ids or {})
    if set(source_map) - set(task_ids) or any(
        type(value) is not str or not _canonical_identity(value)
        for value in source_map.values()
    ):
        raise ValueError("source-series identities must bind known tasks to nonempty strings")

    ordered = tuple(sorted(supplied, key=lambda task: task.task_id))
    disjoint = _DisjointSet(task_ids)
    indexes: tuple[dict[str, str], ...] = ({}, {}, {})
    for task in ordered:
        keys = (
            _canonical_identity(task.entity_name),
            _history_fingerprint(task),
            _canonical_identity(source_map[task.task_id]) if task.task_id in source_map else "",
        )
        for index, key in zip(indexes, keys, strict=True):
            if not key:
                continue
            prior = index.get(key)
            if prior is None:
                index[key] = task.task_id
            else:
                disjoint.union(prior, task.task_id)

    members: dict[str, list[str]] = defaultdict(list)
    by_id = {task.task_id: task for task in ordered}
    for task_id in task_ids:
        members[disjoint.find(task_id)].append(task_id)
    grouped: list[tuple[str, tuple[str, ...], str]] = []
    for task_group in members.values():
        member_ids = tuple(sorted(task_group))
        group_sha = _sha256({"task_ids": list(member_ids)})
        strata = sorted({task_morphology_key(_task_profile(by_id[item])) for item in member_ids})
        grouped.append((group_sha, member_ids, strata[0]))
    if len(grouped) < fold_count:
        raise ValueError("indivisible task groups cannot populate every requested fold")

    fold_sizes = [0] * fold_count
    stratum_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(fold_count)]
    assignments: list[tuple[str, tuple[str, ...], int]] = []
    ranked_groups = sorted(
        grouped,
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(f"{seed}\0{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    for group_sha, member_ids, stratum in ranked_groups:
        fold = min(
            range(fold_count),
            key=lambda value: (
                fold_sizes[value],
                stratum_counts[value][stratum],
                value,
            ),
        )
        fold_sizes[fold] += len(member_ids)
        stratum_counts[fold][stratum] += len(member_ids)
        assignments.append((group_sha, member_ids, fold))
    assignments.sort(key=lambda item: item[0])
    return GroupFoldManifest(
        schema_version=_GROUPING_SCHEMA,
        seed=seed,
        fold_count=fold_count,
        groups=tuple(assignments),
        grouping_fingerprint=_GROUPING_IMPLEMENTATION,
    )


@dataclass(frozen=True)
class TaskLocalTaskRow:
    """Trusted labeled task row retaining history-only folds for local scoring."""

    task_id: str
    candidate_name: str
    family: str
    profile: TaskProfile
    history: tuple[float, ...]
    truth: tuple[float, ...]
    forecast: tuple[float, ...] | None
    diagnostic: CandidateDiagnostics | None
    split: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise ValueError("task-local row requires a task identifier")
        if type(self.candidate_name) is not str or not self.candidate_name.isidentifier():
            raise ValueError("task-local row requires a candidate identifier")
        if type(self.family) is not str or self.family not in {"statistical", "tsfm", "combined"}:
            raise ValueError("task-local row requires a reviewed candidate family")
        if type(self.profile) is not TaskProfile or self.profile.task_id != self.task_id:
            raise ValueError("task-local row profile identity mismatch")
        if (
            type(self.history) is not tuple
            or len(self.history) != self.profile.history_length
            or any(type(value) is not float or not math.isfinite(value) for value in self.history)
        ):
            raise ValueError("task-local row requires a finite exact history")
        if (
            type(self.truth) is not tuple
            or len(self.truth) != self.profile.horizon
            or any(type(value) is not float or not math.isfinite(value) for value in self.truth)
        ):
            raise ValueError("task-local row requires a finite exact truth")
        if self.forecast is None:
            if type(self.failure_reason) is not str or not self.failure_reason:
                raise ValueError("failed task-local rows require a failure reason")
        elif (
            type(self.forecast) is not tuple
            or len(self.forecast) != self.profile.horizon
            or any(type(value) is not float or not math.isfinite(value) for value in self.forecast)
            or self.failure_reason is not None
        ):
            raise ValueError("successful task-local rows require one finite forecast")
        if self.diagnostic is not None and (
            type(self.diagnostic) is not CandidateDiagnostics
            or self.diagnostic.name != self.candidate_name
            or self.diagnostic.family != self.family
        ):
            raise ValueError("task-local diagnostic identity mismatch")
        if type(self.split) is not str or self.split not in {"train", "dev", "public"}:
            raise ValueError("task-local row split is unsupported")


def fit_oof_shortlist_priors(
    rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest, *,
    candidate_names: Sequence[str],
) -> tuple[Mapping[int, tuple[CandidatePriorV1, ...]], tuple[CandidatePriorV1, ...]]:
    """Fit fold priors from complementary Train IDs and final priors from all Train IDs."""
    if type(manifest) is not GroupFoldManifest:
        raise ValueError("OOF prior fitting requires an exact group fold manifest")
    task_ids = tuple(sorted(manifest.task_fold_map))
    morphology: dict[str, str] = {}
    for row in rows:
        if type(row) is not TaskLocalTaskRow or row.split != "train" or row.task_id not in manifest.task_fold_map:
            continue
        key = task_morphology_key(row.profile)
        if row.task_id in morphology and morphology[row.task_id] != key:
            raise ValueError("morphology key is inconsistent for task")
        morphology[row.task_id] = key
    if set(morphology) != set(task_ids):
        raise ValueError("OOF prior fitting requires every manifest task in Train rows")
    folds: dict[int, tuple[CandidatePriorV1, ...]] = {}
    for fold in range(manifest.fold_count):
        fit_ids = tuple(task_id for task_id in task_ids if manifest.task_fold_map[task_id] != fold)
        folds[fold] = fit_candidate_priors(rows, task_ids=fit_ids, morphology_keys={task_id: morphology[task_id] for task_id in fit_ids}, candidate_names=candidate_names)
    final = fit_candidate_priors(rows, task_ids=task_ids, morphology_keys=morphology, candidate_names=candidate_names)
    return folds, final


def build_candidate_priors_bundle(
    rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest, *,
    dictionary_sha256: str, candidate_names: Sequence[str],
) -> dict[str, object]:
    """Canonical Task 3 handoff consumed by formal task-local V3 execution."""
    if type(dictionary_sha256) is not str or len(dictionary_sha256) != 64:
        raise ValueError("candidate-priors bundle requires a dictionary SHA-256")
    folds, final = fit_oof_shortlist_priors(rows, manifest, candidate_names=candidate_names)
    payload: dict[str, object] = {
        "schema_version": 1,
        "grouping_fingerprint": manifest.grouping_fingerprint,
        "dictionary_sha256": dictionary_sha256,
        "fold_priors": {str(fold): [item.to_payload() for item in folds[fold]] for fold in sorted(folds)},
        "final_priors": [item.to_payload() for item in final],
    }
    payload["payload_fingerprint"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def write_candidate_priors_bundle(
    path: Path, rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest, *,
    dictionary_sha256: str, candidate_names: Sequence[str],
) -> dict[str, object]:
    payload = build_candidate_priors_bundle(
        rows, manifest, dictionary_sha256=dictionary_sha256, candidate_names=candidate_names,
    )
    path.write_bytes(canonical_json_bytes(payload))
    return payload


@dataclass(frozen=True)
class RecipeTaskScore:
    """Anonymous trusted score for one recipe on one Train task group."""

    task_group_sha256: str
    improvement_smae: float
    improvement_srmse: float
    improvement_joint: float
    regret_smae_raw: float
    regret_srmse_raw: float
    child_smae_clipped: bool
    child_srmse_clipped: bool

    def __post_init__(self) -> None:
        if (
            type(self.task_group_sha256) is not str
            or len(self.task_group_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.task_group_sha256
            )
        ):
            raise ValueError("recipe score requires an opaque group SHA-256")
        for name in (
            "improvement_smae",
            "improvement_srmse",
            "improvement_joint",
            "regret_smae_raw",
            "regret_srmse_raw",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("recipe scores must be finite exact floats")
        if self.regret_smae_raw < 0.0 or self.regret_srmse_raw < 0.0:
            raise ValueError("recipe score regrets must be nonnegative")
        if type(self.child_smae_clipped) is not bool or type(self.child_srmse_clipped) is not bool:
            raise ValueError("recipe clipping flags must be exact booleans")


def _recipe_bounds(region: str, horizon: int) -> tuple[int, int] | None:
    if region == "full":
        return (0, horizon)
    if horizon < 4:
        return None
    midpoint = (horizon + 1) // 2
    return (0, midpoint) if region == "early" else (midpoint, horizon)


def score_recipe_on_task(
    recipe: WeightRecipe,
    anchor: TaskLocalTaskRow,
    task_rows: Mapping[str, TaskLocalTaskRow],
    *,
    task_group_sha256: str,
) -> RecipeTaskScore | None:
    """Score a closed recipe using only already-materialized trusted rows."""
    if type(recipe) is not WeightRecipe or type(anchor) is not TaskLocalTaskRow:
        raise TypeError("recipe scoring requires exact recipe and task rows")
    if recipe.names[0] != anchor.candidate_name or anchor.forecast is None:
        raise ValueError("recipe score anchor identity is inconsistent")
    bounds = _recipe_bounds(recipe.region, len(anchor.truth))
    if bounds is None:
        return None
    start, stop = bounds
    selected: list[TaskLocalTaskRow] = []
    for name in recipe.names:
        row = task_rows.get(name)
        if (
            type(row) is not TaskLocalTaskRow
            or row.task_id != anchor.task_id
            or row.truth != anchor.truth
            or row.forecast is None
        ):
            return None
        selected.append(row)
    truth = anchor.truth[start:stop]
    parent_forecast = anchor.forecast[start:stop]
    child_forecast = tuple(
        math.fsum(
            weight * row.forecast[index]
            for weight, row in zip(recipe.weights, selected, strict=True)
        )
        for index in range(start, stop)
    )
    if not truth or any(not math.isfinite(value) for value in child_forecast):
        return None
    parent = drcik_point_metrics(truth, parent_forecast)
    child = drcik_point_metrics(truth, child_forecast)
    parent_smae = float(parent["smae"])
    parent_srmse = float(parent["srmse"])
    child_smae = float(child["smae"])
    child_srmse = float(child["srmse"])
    return RecipeTaskScore(
        task_group_sha256=task_group_sha256,
        improvement_smae=parent_smae - child_smae,
        improvement_srmse=parent_srmse - child_srmse,
        improvement_joint=joint_scaled_error(parent_smae, parent_srmse)
        - joint_scaled_error(child_smae, child_srmse),
        regret_smae_raw=max(
            0.0, float(child["smae_raw"]) - float(parent["smae_raw"])
        ),
        regret_srmse_raw=max(
            0.0, float(child["srmse_raw"]) - float(parent["srmse_raw"])
        ),
        child_smae_clipped=bool(child["smae_clipped"]),
        child_srmse_clipped=bool(child["srmse_clipped"]),
    )


def _confidence_recipes(
    anchor_name: str, candidate_names: Sequence[str]
) -> tuple[WeightRecipe, ...]:
    supplied = tuple(candidate_names)
    if (
        not supplied
        or supplied[0] != anchor_name
        or len(supplied) > 8
        or len(set(supplied)) != len(supplied)
        or any(type(name) is not str or not name.isidentifier() for name in supplied)
    ):
        raise ValueError("confidence recipe supply must be unique, bounded, and anchored")
    specialists = tuple(sorted(supplied[1:], key=_canonical_identity))
    recipes: list[WeightRecipe] = []
    for region in ("full", "early", "late"):
        for count in range(1, min(2, len(specialists)) + 1):
            for chosen in itertools.combinations(specialists, count):
                for anchor_units in range(5, 10):
                    remainder = 10 - anchor_units
                    allocations = (
                        ((remainder,),)
                        if count == 1
                        else tuple(
                            (left, remainder - left)
                            for left in range(1, remainder)
                        )
                    )
                    for specialist_units in allocations:
                        if any(unit <= 0 for unit in specialist_units):
                            continue
                        recipes.append(
                            WeightRecipe(
                                region,
                                (anchor_name, *chosen),
                                (anchor_units, *specialist_units),
                            )
                        )
    return tuple(sorted(recipes, key=lambda recipe: recipe.fingerprint))


def _evidence_record(
    *,
    level: str,
    group_key: str,
    recipe: WeightRecipe,
    task_support: int,
    scores: Sequence[RecipeTaskScore],
    policy: ConfidencePolicy,
) -> ConfidenceEvidenceRecord:
    wins = sum(score.improvement_joint > 1e-12 for score in scores)
    losses = sum(score.improvement_joint < -1e-12 for score in scores)
    ties = len(scores) - wins - losses
    return ConfidenceEvidenceRecord(
        level=level,
        group_key=group_key,
        recipe=recipe,
        independent_groups=len({score.task_group_sha256 for score in scores}),
        task_support=task_support,
        wins=wins,
        ties=ties,
        losses=losses,
        posterior_win_probability=beta_win_probability(wins, losses),
        robust_margin_joint=robust_effect_margin(
            tuple(score.improvement_joint for score in scores),
            multiplier=policy.robust_mad_multiplier,
        ),
        robust_margin_smae=robust_effect_margin(
            tuple(score.improvement_smae for score in scores),
            multiplier=policy.robust_mad_multiplier,
        ),
        robust_margin_srmse=robust_effect_margin(
            tuple(score.improvement_srmse for score in scores),
            multiplier=policy.robust_mad_multiplier,
        ),
        p90_regret_smae_raw=float(
            linear_quantile([score.regret_smae_raw for score in scores], 0.9)
        ),
        p90_regret_srmse_raw=float(
            linear_quantile([score.regret_srmse_raw for score in scores], 0.9)
        ),
        failure_count=task_support - len(scores),
        clipped_smae_count=sum(score.child_smae_clipped for score in scores),
        clipped_srmse_count=sum(score.child_srmse_clipped for score in scores),
    )


def build_hierarchical_evidence(
    rows: Sequence[TaskLocalTaskRow],
    *,
    task_ids: Sequence[str],
    group_ids: Mapping[str, str],
    anchor_name: str,
    candidate_names: Sequence[str],
    policy: ConfidencePolicy,
) -> HierarchicalEvidenceBank:
    """Build anonymous exact/coarse/global recipe evidence from Train rows."""
    requested = tuple(task_ids)
    if type(policy) is not ConfidencePolicy:
        raise TypeError("hierarchical evidence requires an exact confidence policy")
    ConfidencePolicy.__post_init__(policy)
    if (
        not requested
        or len(requested) != len(set(requested))
        or set(requested) != set(group_ids)
    ):
        raise ValueError("hierarchical evidence requires exact task/group coverage")
    for digest in group_ids.values():
        RecipeTaskScore(digest, 0.0, 0.0, 0.0, 0.0, 0.0, False, False)
    rows_by_task = _rows_by_task(rows, requested)
    profiles: dict[str, TaskProfile] = {}
    for task_id in requested:
        task_rows = rows_by_task[task_id]
        anchor = task_rows.get(anchor_name)
        if anchor is None:
            raise ValueError("hierarchical evidence is missing an anchor row")
        profiles[task_id] = anchor.profile

    recipes = _confidence_recipes(anchor_name, candidate_names)
    scored: dict[str, dict[str, RecipeTaskScore]] = {}
    for recipe in recipes:
        per_task: dict[str, RecipeTaskScore] = {}
        for task_id in requested:
            anchor = rows_by_task[task_id][anchor_name]
            result = score_recipe_on_task(
                recipe,
                anchor,
                rows_by_task[task_id],
                task_group_sha256=group_ids[task_id],
            )
            if result is not None:
                per_task[task_id] = result
        scored[recipe.fingerprint] = per_task

    exact_buckets: dict[str, list[str]] = defaultdict(list)
    coarse_buckets: dict[str, list[str]] = defaultdict(list)
    for task_id in requested:
        exact_buckets[exact_morphology_key(profiles[task_id])].append(task_id)
        coarse_buckets[coarse_morphology_key(profiles[task_id])].append(task_id)
    levels: tuple[tuple[str, Mapping[str, Sequence[str]], int], ...] = (
        ("exact", exact_buckets, policy.exact_minimum_support),
        ("coarse", coarse_buckets, policy.coarse_minimum_support),
        (
            "global",
            {HierarchicalEvidenceBank.global_group_key(): requested},
            policy.global_minimum_support,
        ),
    )
    records: list[ConfidenceEvidenceRecord] = []
    for recipe in recipes:
        per_task = scored[recipe.fingerprint]
        for level, buckets, minimum_support in levels:
            for group_key, members in sorted(buckets.items()):
                member_ids = tuple(members)
                scores = tuple(
                    per_task[task_id]
                    for task_id in member_ids
                    if task_id in per_task
                )
                if (
                    len(scores) < minimum_support
                    or len({score.task_group_sha256 for score in scores}) < 2
                ):
                    continue
                records.append(
                    _evidence_record(
                        level=level,
                        group_key=group_key,
                        recipe=recipe,
                        task_support=len(member_ids),
                        scores=scores,
                        policy=policy,
                    )
                )
    return HierarchicalEvidenceBank.build(
        policy,
        records=records,
        fit_group_ids=tuple(sorted(set(group_ids.values()))),
    )


def build_cross_fitted_evidence(
    rows: Sequence[TaskLocalTaskRow],
    manifest: GroupFoldManifest,
    *,
    anchor_name: str,
    supplies: Mapping[int, Sequence[str]],
    policy: ConfidencePolicy,
) -> Mapping[int, HierarchicalEvidenceBank]:
    """Fit one evidence bank per outer fold without its connected groups."""
    if type(manifest) is not GroupFoldManifest:
        raise TypeError("cross-fitted evidence requires an exact group manifest")
    if set(supplies) != set(range(manifest.fold_count)):
        raise ValueError("cross-fitted evidence requires one supply per fold")
    task_folds = dict(manifest.task_fold_map)
    all_ids = tuple(sorted(task_folds))
    group_ids = {
        task_id: group_sha
        for group_sha, group_tasks, _fold in manifest.groups
        for task_id in group_tasks
    }
    result: dict[int, HierarchicalEvidenceBank] = {}
    for fold in range(manifest.fold_count):
        fit_ids = tuple(task_id for task_id in all_ids if task_folds[task_id] != fold)
        fit_groups = {task_id: group_ids[task_id] for task_id in fit_ids}
        result[fold] = build_hierarchical_evidence(
            rows,
            task_ids=fit_ids,
            group_ids=fit_groups,
            anchor_name=anchor_name,
            candidate_names=supplies[fold],
            policy=policy,
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class _TaskOutcome:
    group_key: str
    activated: bool
    fallback: bool
    failed: bool
    parent_smae: float
    parent_srmse: float
    parent_smae_raw: float
    parent_srmse_raw: float
    child_smae: float
    child_srmse: float
    child_smae_raw: float
    child_srmse_raw: float
    max_fold_regret: float


@dataclass(frozen=True)
class ConditionalUpliftReport:
    """Aggregate label-bearing gate authority with a sanitized proposer projection."""

    schema_version: int
    split: str
    task_count: int
    oof_task_count: int
    group_count: int
    fit_leakage_count: int
    activation_count: int
    activation_rate: float
    fallback_count: int
    failure_count: int
    activated_wins: int
    activated_ties: int
    activated_losses: int
    parent_mean_smae: float
    child_mean_smae: float
    parent_mean_srmse: float
    child_mean_srmse: float
    parent_median_smae: float
    child_median_smae: float
    parent_median_srmse: float
    child_median_srmse: float
    parent_p90_smae: float
    child_p90_smae: float
    parent_p95_smae: float
    child_p95_smae: float
    parent_p90_srmse: float
    child_p90_srmse: float
    parent_p95_srmse: float
    child_p95_srmse: float
    parent_p90_smae_raw: float
    child_p90_smae_raw: float
    parent_p95_smae_raw: float
    child_p95_smae_raw: float
    parent_p90_srmse_raw: float
    child_p90_srmse_raw: float
    parent_p95_srmse_raw: float
    child_p95_srmse_raw: float
    parent_smae_clipped_count: int
    child_smae_clipped_count: int
    parent_srmse_clipped_count: int
    child_srmse_clipped_count: int
    mean_delta_smae: float
    mean_delta_srmse: float
    median_delta_smae: float
    median_delta_srmse: float
    maximum_task_regret_smae: float
    maximum_task_regret_srmse: float
    maximum_fold_regret: float
    activated_group_count: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    report_fingerprint: str

    def to_proposer_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_count": self.task_count,
            "group_count": self.group_count,
            "activation_count": self.activation_count,
            "activation_rate": self.activation_rate,
            "fallback_count": self.fallback_count,
            "failure_count": self.failure_count,
            "activated_group_count": self.activated_group_count,
            "activated_wtl": {
                "wins": self.activated_wins,
                "ties": self.activated_ties,
                "losses": self.activated_losses,
            },
            "paired_delta": {
                "mean_smae": self.mean_delta_smae,
                "mean_srmse": self.mean_delta_srmse,
                "median_smae": self.median_delta_smae,
                "median_srmse": self.median_delta_srmse,
            },
            "tail": {
                "maximum_task_regret_smae": self.maximum_task_regret_smae,
                "maximum_task_regret_srmse": self.maximum_task_regret_srmse,
                "maximum_fold_regret": self.maximum_fold_regret,
            },
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _rows_by_task(
    rows: Sequence[TaskLocalTaskRow], task_ids: Sequence[str]
) -> dict[str, dict[str, TaskLocalTaskRow]]:
    requested = tuple(task_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("task-local evaluation requires unique task IDs")
    result = {task_id: {} for task_id in requested}
    for row in rows:
        if type(row) is not TaskLocalTaskRow:
            raise ValueError("task-local evaluation requires exact rows")
        if row.task_id not in result:
            continue
        if row.candidate_name in result[row.task_id]:
            raise ValueError("task-local rows contain duplicate candidate/task keys")
        result[row.task_id][row.candidate_name] = row
    if any(not result[task_id] for task_id in requested):
        raise ValueError("task-local evaluation is missing a requested task")
    return result


def _candidate_rank(
    rows_by_task: Mapping[str, Mapping[str, TaskLocalTaskRow]],
    task_ids: Sequence[str],
    candidate_name: str,
) -> tuple[object, ...]:
    smae: list[float] = []
    srmse: list[float] = []
    successes = 0
    for task_id in task_ids:
        row = rows_by_task[task_id].get(candidate_name)
        if row is None or row.forecast is None:
            smae.append(5.0)
            srmse.append(5.0)
            continue
        point = drcik_point_metrics(row.truth, row.forecast)
        successes += 1
        smae.append(float(point["smae"]))
        srmse.append(float(point["srmse"]))
    joints = [joint_scaled_error(left, right) for left, right in zip(smae, srmse, strict=True)]
    return (
        -successes,
        statistics.fmean(joints),
        linear_quantile(joints, 0.9),
        statistics.fmean(smae),
        statistics.fmean(srmse),
        _canonical_identity(candidate_name),
    )


def _bounded_names(
    rows_by_task: Mapping[str, Mapping[str, TaskLocalTaskRow]],
    task_ids: Sequence[str],
    *,
    anchor_name: str,
    maximum_candidates: int,
) -> tuple[str, ...]:
    names = sorted(
        {name for task_id in task_ids for name in rows_by_task[task_id]},
        key=lambda name: _candidate_rank(rows_by_task, task_ids, name),
    )
    if anchor_name not in names:
        raise ValueError("task-local fitting requires its anchor on every universe")
    ranked = [name for name in names if name != anchor_name]
    family_by_name: dict[str, str] = {}
    for task_id in task_ids:
        for name, row in rows_by_task[task_id].items():
            family_by_name.setdefault(name, row.family)
    selected: list[str] = []
    for family in ("statistical", "tsfm", "combined"):
        match = next((name for name in ranked if family_by_name.get(name) == family), None)
        if match is not None and match not in selected:
            selected.append(match)
    for name in ranked:
        if name not in selected:
            selected.append(name)
    return (anchor_name, *selected[: maximum_candidates - 1])


def fit_group_candidate_supply(
    rows: Sequence[TaskLocalTaskRow],
    *,
    task_ids: Sequence[str],
    group_keys: Mapping[str, str],
    anchor_name: str,
    maximum_candidates: int = 8,
) -> tuple[tuple[str, ...], tuple[GroupCandidateSupply, ...]]:
    """Fit bounded global/group supplies from the exact supplied Train IDs."""
    rows_by_task = _rows_by_task(rows, task_ids)
    if set(task_ids) - set(group_keys):
        raise ValueError("candidate supply requires a group key for every fit task")
    default = _bounded_names(
        rows_by_task,
        task_ids,
        anchor_name=anchor_name,
        maximum_candidates=maximum_candidates,
    )
    supplies: list[GroupCandidateSupply] = []
    for group_key in sorted({group_keys[task_id] for task_id in task_ids}):
        members = tuple(task_id for task_id in task_ids if group_keys[task_id] == group_key)
        names = (
            _bounded_names(
                rows_by_task,
                members,
                anchor_name=anchor_name,
                maximum_candidates=maximum_candidates,
            )
            if len(members) >= 4
            else default
        )
        supplies.append(GroupCandidateSupply(group_key, names))
    return default, tuple(supplies)


def _evaluate_rows(
    release: TaskLocalEnsembleRelease | TaskLocalEnsembleReleaseV3,
    rows: Sequence[TaskLocalTaskRow],
    task_ids: Sequence[str],
    *,
    report_group_keys: Mapping[str, str] | None = None,
) -> tuple[_TaskOutcome, ...]:
    rows_by_task = _rows_by_task(rows, task_ids)
    outcomes: list[_TaskOutcome] = []
    for task_id in task_ids:
        task_rows = rows_by_task[task_id]
        anchor = task_rows.get(release.anchor_name)
        if anchor is None or anchor.forecast is None:
            raise ValueError("task-local anchor outcome is missing or failed")
        morphology_key = task_morphology_key(anchor.profile)
        names = (
            release.candidate_names(morphology_key)
            if type(release) is TaskLocalEnsembleRelease
            else tuple(task_rows)
        )
        forecasts = {
            name: row.forecast
            for name in names
            if (row := task_rows.get(name)) is not None and row.forecast is not None
        }
        diagnostics = {
            name: row.diagnostic
            for name in names
            if (row := task_rows.get(name)) is not None and row.diagnostic is not None
        }
        if type(release) is TaskLocalEnsembleRelease and release.schema_version == 2:
            assert release.confidence_evidence is not None
            result = execute_confidence_task_local_ensemble(
                release.policy,
                release.confidence_evidence.policy,
                candidate_names=names,
                forecasts=forecasts,
                diagnostics=diagnostics,
                horizon=anchor.profile.horizon,
                profile=anchor.profile,
                confidence_evidence=release.confidence_evidence,
            )
            maximum_fold_regret = max(
                region.maximum_fold_regret for region in result.regions
            )
        else:
            result = execute_task_local_ensemble(
                release.policy if type(release) is TaskLocalEnsembleRelease else release.tournament_policy,
                candidate_names=names,
                forecasts=forecasts,
                diagnostics=diagnostics,
                horizon=anchor.profile.horizon,
            )
            maximum_fold_regret = result.maximum_fold_regret
        parent = drcik_point_metrics(anchor.truth, anchor.forecast)
        child = drcik_point_metrics(anchor.truth, result.forecast)
        outcomes.append(
            _TaskOutcome(
                group_key=(report_group_keys or {}).get(task_id, morphology_key),
                activated=result.activated,
                fallback=not result.activated,
                failed=False,
                parent_smae=float(parent["smae"]),
                parent_srmse=float(parent["srmse"]),
                parent_smae_raw=float(parent["smae_raw"]),
                parent_srmse_raw=float(parent["srmse_raw"]),
                child_smae=float(child["smae"]),
                child_srmse=float(child["srmse"]),
                child_smae_raw=float(child["smae_raw"]),
                child_srmse_raw=float(child["srmse_raw"]),
                max_fold_regret=maximum_fold_regret,
            )
        )
    return tuple(outcomes)


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(linear_quantile(list(values), probability))


def _report(
    outcomes: Sequence[_TaskOutcome],
    *,
    split: str,
    oof_task_count: int,
    fit_leakage_count: int,
    policy: TaskLocalTournamentPolicy,
) -> ConditionalUpliftReport:
    if not outcomes:
        raise ValueError("conditional uplift requires task outcomes")
    parent_smae = [item.parent_smae for item in outcomes]
    parent_srmse = [item.parent_srmse for item in outcomes]
    child_smae = [item.child_smae for item in outcomes]
    child_srmse = [item.child_srmse for item in outcomes]
    parent_smae_raw = [item.parent_smae_raw for item in outcomes]
    parent_srmse_raw = [item.parent_srmse_raw for item in outcomes]
    child_smae_raw = [item.child_smae_raw for item in outcomes]
    child_srmse_raw = [item.child_srmse_raw for item in outcomes]
    delta_smae = [child - parent for child, parent in zip(child_smae, parent_smae, strict=True)]
    delta_srmse = [child - parent for child, parent in zip(child_srmse, parent_srmse, strict=True)]
    activated = [item for item in outcomes if item.activated]
    wins = ties = losses = 0
    for item in activated:
        child_joint = joint_scaled_error(item.child_smae, item.child_srmse)
        parent_joint = joint_scaled_error(item.parent_smae, item.parent_srmse)
        if child_joint < parent_joint - 1e-12:
            wins += 1
        elif child_joint > parent_joint + 1e-12:
            losses += 1
        else:
            ties += 1
    rejections: list[str] = []
    if not (
        statistics.fmean(child_smae) <= statistics.fmean(parent_smae) + 1e-12
        and statistics.fmean(child_srmse) <= statistics.fmean(parent_srmse) + 1e-12
        and (
            statistics.fmean(child_smae) < statistics.fmean(parent_smae) - 1e-12
            or statistics.fmean(child_srmse) < statistics.fmean(parent_srmse) - 1e-12
        )
    ):
        rejections.append("mean_pareto")
    for label, parent, child in (
        ("p90_smae", parent_smae, child_smae),
        ("p95_smae", parent_smae, child_smae),
        ("p90_srmse", parent_srmse, child_srmse),
        ("p95_srmse", parent_srmse, child_srmse),
        ("p90_smae_raw", parent_smae_raw, child_smae_raw),
        ("p95_smae_raw", parent_smae_raw, child_smae_raw),
        ("p90_srmse_raw", parent_srmse_raw, child_srmse_raw),
        ("p95_srmse_raw", parent_srmse_raw, child_srmse_raw),
    ):
        probability = 0.9 if label.startswith("p90") else 0.95
        if _quantile(child, probability) > _quantile(parent, probability) + 1e-12:
            rejections.append(label)
    if len(activated) < policy.minimum_activation_support:
        rejections.append("activation_support")
    activated_groups = len({item.group_key for item in activated})
    if activated_groups < policy.minimum_activation_groups:
        rejections.append("activation_groups")
    if (
        wins + losses == 0
        or wins / (wins + losses) < policy.minimum_activation_precision
    ):
        rejections.append("activation_precision")
    max_regret_smae = max(max(0.0, value) for value in delta_smae)
    max_regret_srmse = max(max(0.0, value) for value in delta_srmse)
    if max_regret_smae > policy.maximum_worst_joint_regret:
        rejections.append("task_regret_smae")
    if max_regret_srmse > policy.maximum_worst_joint_regret:
        rejections.append("task_regret_srmse")
    payload = {
        "split": split,
        "task_count": len(outcomes),
        "oof_task_count": oof_task_count,
        "groups": sorted({item.group_key for item in outcomes}),
        "activation_count": len(activated),
        "parent_smae": parent_smae,
        "child_smae": child_smae,
        "parent_srmse": parent_srmse,
        "child_srmse": child_srmse,
        "rejections": rejections,
    }
    fingerprint = _sha256(payload)
    return ConditionalUpliftReport(
        schema_version=1,
        split=split,
        task_count=len(outcomes),
        oof_task_count=oof_task_count,
        group_count=len({item.group_key for item in outcomes}),
        fit_leakage_count=fit_leakage_count,
        activation_count=len(activated),
        activation_rate=len(activated) / len(outcomes),
        fallback_count=sum(item.fallback for item in outcomes),
        failure_count=sum(item.failed for item in outcomes),
        activated_wins=wins,
        activated_ties=ties,
        activated_losses=losses,
        parent_mean_smae=statistics.fmean(parent_smae),
        child_mean_smae=statistics.fmean(child_smae),
        parent_mean_srmse=statistics.fmean(parent_srmse),
        child_mean_srmse=statistics.fmean(child_srmse),
        parent_median_smae=float(statistics.median(parent_smae)),
        child_median_smae=float(statistics.median(child_smae)),
        parent_median_srmse=float(statistics.median(parent_srmse)),
        child_median_srmse=float(statistics.median(child_srmse)),
        parent_p90_smae=_quantile(parent_smae, 0.9),
        child_p90_smae=_quantile(child_smae, 0.9),
        parent_p95_smae=_quantile(parent_smae, 0.95),
        child_p95_smae=_quantile(child_smae, 0.95),
        parent_p90_srmse=_quantile(parent_srmse, 0.9),
        child_p90_srmse=_quantile(child_srmse, 0.9),
        parent_p95_srmse=_quantile(parent_srmse, 0.95),
        child_p95_srmse=_quantile(child_srmse, 0.95),
        parent_p90_smae_raw=_quantile(parent_smae_raw, 0.9),
        child_p90_smae_raw=_quantile(child_smae_raw, 0.9),
        parent_p95_smae_raw=_quantile(parent_smae_raw, 0.95),
        child_p95_smae_raw=_quantile(child_smae_raw, 0.95),
        parent_p90_srmse_raw=_quantile(parent_srmse_raw, 0.9),
        child_p90_srmse_raw=_quantile(child_srmse_raw, 0.9),
        parent_p95_srmse_raw=_quantile(parent_srmse_raw, 0.95),
        child_p95_srmse_raw=_quantile(child_srmse_raw, 0.95),
        parent_smae_clipped_count=sum(value > 5.0 for value in parent_smae_raw),
        child_smae_clipped_count=sum(value > 5.0 for value in child_smae_raw),
        parent_srmse_clipped_count=sum(value > 5.0 for value in parent_srmse_raw),
        child_srmse_clipped_count=sum(value > 5.0 for value in child_srmse_raw),
        mean_delta_smae=statistics.fmean(delta_smae),
        mean_delta_srmse=statistics.fmean(delta_srmse),
        median_delta_smae=float(statistics.median(delta_smae)),
        median_delta_srmse=float(statistics.median(delta_srmse)),
        maximum_task_regret_smae=max_regret_smae,
        maximum_task_regret_srmse=max_regret_srmse,
        maximum_fold_regret=max(item.max_fold_regret for item in outcomes),
        activated_group_count=activated_groups,
        accepted=not rejections,
        rejection_reasons=tuple(rejections),
        report_fingerprint=fingerprint,
    )


def evaluate_task_local_release(
    release: TaskLocalEnsembleRelease | TaskLocalEnsembleReleaseV3,
    rows: Sequence[TaskLocalTaskRow],
    *,
    task_ids: Sequence[str],
    split: str,
) -> ConditionalUpliftReport:
    if type(release) not in {TaskLocalEnsembleRelease, TaskLocalEnsembleReleaseV3}:
        raise TypeError("task-local evaluation requires an exact recognized release")
    outcomes = _evaluate_rows(release, rows, task_ids)
    return _report(
        outcomes,
        split=split,
        oof_task_count=len(outcomes) if split == "oof" else 0,
        fit_leakage_count=0,
        policy=release.policy if type(release) is TaskLocalEnsembleRelease else release.tournament_policy,
    )


def fit_oof_release(
    rows: Sequence[TaskLocalTaskRow],
    manifest: GroupFoldManifest,
    *,
    anchor_release_sha256: str,
    anchor_name: str,
    source_hashes: tuple[tuple[str, str], ...],
    policy: TaskLocalTournamentPolicy,
    confidence_policy: ConfidencePolicy | None = None,
) -> tuple[TaskLocalEnsembleRelease, ConditionalUpliftReport]:
    """Fit held-out supplies for every Train group, then reconstruct on all Train."""
    if type(manifest) is not GroupFoldManifest or type(policy) is not TaskLocalTournamentPolicy:
        raise TypeError("OOF fitting requires exact manifest and policy objects")
    if confidence_policy is not None and type(confidence_policy) is not ConfidencePolicy:
        raise TypeError("OOF confidence policy must be exact or None")
    task_folds = dict(manifest.task_fold_map)
    all_ids = tuple(sorted(task_folds))
    report_group_keys = {
        task_id: group_sha
        for group_sha, task_ids, _fold in manifest.groups
        for task_id in task_ids
    }
    rows_by_task = _rows_by_task(rows, all_ids)
    group_keys = {
        task_id: task_morphology_key(next(iter(rows_by_task[task_id].values())).profile)
        for task_id in all_ids
    }
    oof_outcomes: list[_TaskOutcome] = []
    for fold in range(manifest.fold_count):
        fit_ids = tuple(task_id for task_id in all_ids if task_folds[task_id] != fold)
        held_out = tuple(task_id for task_id in all_ids if task_folds[task_id] == fold)
        if not fit_ids or not held_out or set(fit_ids) & set(held_out):
            raise ValueError("OOF fold is empty or overlapping")
        default, supplies = fit_group_candidate_supply(
            rows,
            task_ids=fit_ids,
            group_keys=group_keys,
            anchor_name=anchor_name,
            maximum_candidates=policy.maximum_candidates,
        )
        evidence = (
            build_hierarchical_evidence(
                rows,
                task_ids=fit_ids,
                group_ids={task_id: report_group_keys[task_id] for task_id in fit_ids},
                anchor_name=anchor_name,
                candidate_names=default,
                policy=confidence_policy,
            )
            if confidence_policy is not None
            else None
        )
        temporary = TaskLocalEnsembleRelease(
            schema_version=2 if confidence_policy is not None else 1,
            anchor_release_sha256=anchor_release_sha256,
            anchor_name=anchor_name,
            policy=policy,
            default_candidate_names=default,
            group_supplies=supplies,
            grouping_fingerprint=manifest.grouping_fingerprint,
            oof_report_sha256="0" * 64,
            source_hashes=source_hashes,
            metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
            lineage=(
                ("task_local_oof", "task_local_confidence_v2")
                if confidence_policy is not None
                else ("task_local_oof",)
            ),
            confidence_evidence=evidence,
        )
        oof_outcomes.extend(
            _evaluate_rows(
                temporary,
                rows,
                held_out,
                report_group_keys=report_group_keys,
            )
        )
    oof_report = _report(
        tuple(oof_outcomes),
        split="oof",
        oof_task_count=len(oof_outcomes),
        fit_leakage_count=0,
        policy=policy,
    )
    default, supplies = fit_group_candidate_supply(
        rows,
        task_ids=all_ids,
        group_keys=group_keys,
        anchor_name=anchor_name,
        maximum_candidates=policy.maximum_candidates,
    )
    evidence = (
        build_hierarchical_evidence(
            rows,
            task_ids=all_ids,
            group_ids=report_group_keys,
            anchor_name=anchor_name,
            candidate_names=default,
            policy=confidence_policy,
        )
        if confidence_policy is not None
        else None
    )
    release = TaskLocalEnsembleRelease(
        schema_version=2 if confidence_policy is not None else 1,
        anchor_release_sha256=anchor_release_sha256,
        anchor_name=anchor_name,
        policy=policy,
        default_candidate_names=default,
        group_supplies=supplies,
        grouping_fingerprint=manifest.grouping_fingerprint,
        oof_report_sha256=oof_report.report_fingerprint,
        source_hashes=source_hashes,
        metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
        lineage=(
            ("task_local_confidence_v2",)
            if confidence_policy is not None
            else ("task_local_v1",)
        ),
        confidence_evidence=evidence,
    )
    return release, oof_report
