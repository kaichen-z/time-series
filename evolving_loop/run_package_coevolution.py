"""Run package-native Numerical, Retrieval, and Decision co-evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.llm import CodexCLIClient, CodexCLIConfig
from common.payload import canonical_json_bytes, read_json_object
from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    HarnessPolicy,
    embed_retrieval_release,
)
from evolving_loop.data import ContextTask, load_context_tasks_by_ids
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_artifacts import (
    PackageArtifactError,
    PackageArtifactStore,
    PackageCacheKey,
    PackageCheckpoint,
    PackageInferenceCache,
)
from evolving_loop.package_candidate_proposal import (
    PackageCandidate,
    PackageProposalFeedback,
    embed_retrieval_candidate,
    proposal_fingerprint,
)
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateController,
    PackageCoordinateState,
    PackageCoordinateStep,
    _skip_step,
    package_principal_fingerprints,
)
from evolving_loop.package_decision_evolution import (
    DecisionCandidateProposer,
    PackageDecisionEvaluator,
    PackageDecisionEvolutionEngine,
    _proposal_evaluation,
)
from evolving_loop.package_metrics import (
    PackageEvaluation,
    PackageGateConfig,
    PackageTaskScore,
    package_full_gate_failures,
)
from evolving_loop.package_numerical_evolution import (
    NumericalCandidateProposer,
    NumericalPackageMaterializer,
    NumericalPackageProposer,
    _sanitized_context_task,
)
from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyRelease,
    bound_numerical_package,
    build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import (
    FrozenNumericalPackageRegistry,
    task_registry_fingerprint,
)
from evolving_loop.package_stage_runner import (
    PackageCoordinatePhaseOutcome,
    PackageCoordinatePhaseRunner,
    PackageStageEvidence,
    PackageStageSchedule,
)
from evolving_loop.retrieval_agent.evolution import (
    CHILD_SCOPES,
    RetrievalCandidateProposer,
    RetrievalEvolutionConfig,
    RetrievalGenomeProposer,
    retrieval_behavior_fingerprint,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalRelease,
    _load_retrieval_release_for_operator,
)
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    champion_fingerprint,
    parse_champion_release,
)
from numerical_agent.evolution.champion_controller import (
    ChampionProposerAdapter,
    fit_champion_recipe,
)
from numerical_agent.evolution.champion_evidence import ProposerEvidence
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.module import read_module
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.numerical_selector import DecisionPolicy, HindcastConfig
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.specialist_atlas import (
    AtlasPolicy,
    AtlasRelease,
    fit_atlas_release,
    parse_atlas_release,
)
from numerical_agent.evolution.task_local_evolution import (
    GroupFoldManifest,
    build_group_fold_manifest,
)
from numerical_agent.main import _add_tsfm_runtime_options, _runtime_registry
from numerical_agent.run_champion_evolution import (
    _clean_git_source,
    _inventory,
    _load_screening_policy,
    _materialize_rows as _materialize_champion_rows,
    _source_files,
)
from numerical_agent.run_selector_evolution import _forecast_runtime_identity
from numerical_agent.run_task_local_ensemble_evolution import (
    _materialize_rows as _materialize_atlas_rows,
    _reviewed_candidates,
)


_FORMAL_SEED = 20260903
_FORMAL_COUNTS = {"train": 80, "dev": 20, "public_test": 99}


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _split_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_from_payload(payload: object) -> PackageEvaluation:
    if not isinstance(payload, Mapping):
        raise ValueError("cached package evaluation must be an object")
    try:
        rows = tuple(
            PackageTaskScore(**dict(row))
            for row in payload["task_rows"]
            if isinstance(row, Mapping)
        )
        evaluation = PackageEvaluation.from_rows(
            str(payload["candidate_sha256"]),
            rows,
            tuple(payload["expected_task_ids"]),
            dict(payload["secondary_diagnostics"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cached package evaluation is malformed") from error
    if _split_digest(evaluation.to_payload()) != _split_digest(dict(payload)):
        raise ValueError("cached package evaluation fields do not recompute")
    return evaluation


class PackageCacheBackedEvaluator:
    """Bind full-pipeline evaluations to bundle, stage, tasks, and runtime."""

    def __init__(
        self,
        evaluator: object,
        cache_root: str | Path,
        *,
        runtime_fingerprints: Mapping[str, str],
    ) -> None:
        if not callable(getattr(evaluator, "evaluate", None)):
            raise ValueError("cache-backed evaluator requires evaluate(...)")
        runtime = dict(runtime_fingerprints)
        if not runtime or any(
            type(name) is not str
            or not name
            or type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for name, value in runtime.items()
        ):
            raise ValueError("cache-backed evaluator runtime must be canonical")
        self.evaluator = evaluator
        self.cache_root = Path(cache_root)
        self.runtime_fingerprints = dict(sorted(runtime.items()))

    def _key(
        self,
        bundle: PackageCoordinateBundle,
        registry: FrozenNumericalPackageRegistry,
        tasks: tuple[ContextTask, ...],
        stage: str,
    ) -> PackageCacheKey:
        return PackageCacheKey(
            layer="decision",
            task_sha256=_digest(
                [task_registry_fingerprint(task) for task in tasks]
            ),
            candidate_sha256=bundle.fingerprint(),
            dependency_fingerprints={
                **self.runtime_fingerprints,
                "registry": registry.fingerprint,
                "stage": _digest({"stage": stage}),
            },
        )

    def evaluate(
        self,
        bundle: PackageCoordinateBundle,
        registry: FrozenNumericalPackageRegistry,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
        cache_only: bool = False,
    ) -> PackageEvaluation:
        if not isinstance(bundle, PackageCoordinateBundle):
            raise TypeError("cache-backed evaluation requires a package bundle")
        if not isinstance(registry, FrozenNumericalPackageRegistry):
            raise TypeError("cache-backed evaluation requires a frozen registry")
        resolved = tuple(tasks)
        if not resolved or any(not isinstance(task, ContextTask) for task in resolved):
            raise ValueError("cache-backed evaluation requires ContextTask records")
        if type(stage) is not str or not stage:
            raise ValueError("cache-backed evaluation requires a stage")
        if type(cache_only) is not bool:
            raise ValueError("cache_only must be an exact boolean")
        key = self._key(bundle, registry, resolved, stage)
        cache = PackageInferenceCache(self.cache_root, cache_only=cache_only)

        def compute() -> object:
            evaluation = self.evaluator.evaluate(
                bundle,
                registry,
                resolved,
                stage=stage,
                cache_only=False,
            )
            if not isinstance(evaluation, PackageEvaluation):
                raise TypeError("package evaluator returned an invalid result")
            return evaluation.to_payload()

        return _evaluation_from_payload(cache.get_or_compute(key, compute))

    def promote_cached_bundle(
        self,
        source: PackageCoordinateBundle,
        published: PackageCoordinateBundle,
        registry: FrozenNumericalPackageRegistry,
        tasks: Sequence[ContextTask],
        *,
        stage: str,
    ) -> None:
        """Copy behavior-identical cached bytes onto a published lineage key."""
        resolved = tuple(tasks)
        source_evaluation = self.evaluate(
            source,
            registry,
            resolved,
            stage=stage,
            cache_only=True,
        )
        payload = source_evaluation.to_payload()
        payload["candidate_sha256"] = published.fingerprint()
        promoted = _evaluation_from_payload(payload)
        PackageInferenceCache(self.cache_root, cache_only=False).get_or_compute(
            self._key(published, registry, resolved, stage),
            promoted.to_payload,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--numerical-champion-release", required=True)
    parser.add_argument("--forecast-store", required=True)
    parser.add_argument("--atlas-release")
    parser.add_argument("--retrieval-seed-release", required=True)
    parser.add_argument("--retrieval-skills", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--authority-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--children-per-coordinate", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    _add_tsfm_runtime_options(parser)
    return parser


def _validated_split(path: str | Path) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    payload = read_json_object(path)
    claimed = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if claimed != _split_digest(unsigned):
        raise ValueError("split manifest digest mismatch")
    try:
        target_sizes = payload["target_sizes"]
        actual_sizes = payload["actual_sizes"]
        partitions = payload["partitions"]
        train = tuple(partitions["train"]["task_ids"])
        dev = tuple(partitions["dev"]["task_ids"])
        public = tuple(partitions["public_test"]["task_ids"])
    except (KeyError, TypeError) as error:
        raise ValueError("split manifest is missing Train/Dev/Public membership") from error
    if target_sizes != _FORMAL_COUNTS or actual_sizes != _FORMAL_COUNTS:
        raise ValueError("split manifest must register exactly 80/20/99 tasks")
    memberships = (train, dev, public)
    if any(
        len(ids) != expected
        or len(ids) != len(set(ids))
        or any(type(task_id) is not str or not task_id for task_id in ids)
        for ids, expected in zip(memberships, (80, 20, 99), strict=True)
    ):
        raise ValueError("split manifest membership does not match 80/20/99")
    if any(
        set(left) & set(right)
        for index, left in enumerate(memberships)
        for right in memberships[index + 1 :]
    ):
        raise ValueError("split manifest partitions must be task-disjoint")
    if any(
        payload.get(name) is not False
        for name in (
            "selection_uses_future_values",
            "selection_uses_gt_evidence",
            "selection_uses_document_labels",
        )
    ):
        raise ValueError("split manifest selection must be label-free")
    return payload, train, dev


def _validate_mode(args: argparse.Namespace) -> None:
    expected = (1, 1) if args.smoke else (2, 3)
    if (args.cycles, args.children_per_coordinate) != expected:
        label = "non-formal smoke" if args.smoke else "formal"
        raise ValueError(
            f"{label} evolution requires cycles={expected[0]} and "
            f"children-per-coordinate={expected[1]}"
        )
    if args.seed != _FORMAL_SEED:
        raise ValueError(f"package evolution seed must be {_FORMAL_SEED}")
    for name in ("tasks_file", "forecast_store"):
        if not Path(getattr(args, name)).is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path")


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configuration_identity(args: argparse.Namespace) -> str:
    return _digest(
        {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "cycles": args.cycles,
            "children_per_coordinate": args.children_per_coordinate,
            "seed": args.seed,
            "smoke": args.smoke,
            "forecast_runtime": _forecast_runtime_identity(args),
        }
    )


def _early_resume_guard(args: argparse.Namespace) -> bool:
    """Reject a changed runtime before constructing any proposal client."""
    output = Path(args.output_dir)
    manifest_path = output / "run_manifest.json"
    if not args.resume:
        if manifest_path.exists():
            raise PackageArtifactError("existing run artifacts require --resume")
        return False
    if not manifest_path.is_file():
        raise PackageArtifactError("no package run manifest to resume")
    manifest = read_json_object(manifest_path)
    if manifest.get("configuration_sha256") != _configuration_identity(args):
        raise PackageArtifactError("resume runtime fingerprint changed")
    completion_path = output / "evaluation_complete.json"
    if not completion_path.exists():
        return False
    completion = read_json_object(completion_path)
    if completion.get("status") != "complete" or not (output / "final_bundle.json").is_file():
        raise PackageArtifactError("resume completion marker is malformed")
    return True


@dataclass(frozen=True)
class _SmokeSchedule:
    """Explicit non-formal 8/2/2 schedule; never accepted by the formal runner."""

    seed: int
    build8_ids: tuple[str, ...]
    calibration2_ids: tuple[str, ...]
    dev2_ids: tuple[str, ...]
    fold_manifest: GroupFoldManifest

    @classmethod
    def build(
        cls,
        formal: PackageStageSchedule,
        task_map: Mapping[str, ContextTask],
    ) -> "_SmokeSchedule":
        build_ids = formal.screen8_ids
        fold = build_group_fold_manifest(
            tuple(task_map[task_id].numeric for task_id in build_ids),
            seed=formal.seed,
            fold_count=5,
        )
        return cls(
            seed=formal.seed,
            build8_ids=build_ids,
            calibration2_ids=formal.calibration16_ids[:2],
            dev2_ids=formal.dev20_ids[:2],
            fold_manifest=fold,
        )

    def stage_ids(self, stage: str) -> tuple[str, ...]:
        try:
            return {
                "screen8": self.build8_ids,
                "build64": self.build8_ids,
                "calibration16": self.calibration2_ids,
                "dev20": self.dev2_ids,
            }[stage]
        except KeyError as error:
            raise ValueError(f"unknown smoke stage {stage!r}") from error

    def tasks_for(
        self, stage: str, task_map: Mapping[str, ContextTask]
    ) -> tuple[ContextTask, ...]:
        return tuple(task_map[task_id] for task_id in self.stage_ids(stage))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "formal_run": False,
            "seed": self.seed,
            "counts": {"build": 8, "calibration": 2, "dev": 2},
            "build8_ids": list(self.build8_ids),
            "calibration2_ids": list(self.calibration2_ids),
            "dev2_ids": list(self.dev2_ids),
            "fold_manifest": self.fold_manifest.to_payload(),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_payload())


def _runtime_fingerprints(
    args: argparse.Namespace,
    *,
    source_fingerprints: Mapping[str, str],
    forecast_store: ForecastStore,
    seed_release: RetrievalRelease,
) -> dict[str, str]:
    root = Path(__file__).parent
    model = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    return {
        "bridge_runtime": _file_sha256(root / "numerical_two_stage.py"),
        "numerical_runtime": _digest(
            {
                "source": dict(source_fingerprints),
                "forecast_store": forecast_store.identity_hash,
            }
        ),
        "retrieval_runtime": _digest(
            {
                "implementation": _file_sha256(root / "retrieval_agent" / "two_stage_agent.py"),
                "seed_release": seed_release.manifest_file_sha256,
            }
        ),
        "decision_runtime": _file_sha256(root / "decision_agent" / "agent.py"),
        "retrieval_verifier": _digest(
            {
                "verifier": _file_sha256(root / "retrieval_agent" / "verifier.py"),
                "quality": _file_sha256(root / "retrieval_agent" / "quality.py"),
            }
        ),
        "metric_policy": METRIC_POLICY_FINGERPRINT,
        "model_runtime": _digest(model),
        "llm_runtime": _digest(
            {**model, "client": _file_sha256(Path(__file__).parents[1] / "common" / "llm.py")}
        ),
    }


def _initial_supply_release(
    champion: object,
    candidates: Sequence[tuple[str, str]],
    *,
    source_fingerprints: Mapping[str, str],
    runtime_fingerprints: Mapping[str, str],
    atlas: AtlasRelease | None,
) -> NumericalSupplyRelease:
    policy = champion.policy
    protected_names = set(policy.recipe.parents)
    selected: list[NumericalAlternativeSpec] = []
    seen_families: set[str] = set()
    for candidate_id, family in candidates:
        if family not in {"statistical", "tsfm", "combined"}:
            continue
        if candidate_id in protected_names or family in seen_families:
            continue
        selected.append(
            NumericalAlternativeSpec(
                candidate_id=candidate_id,
                family=cast(str, family),
                materializer_kind="dictionary",
                recipe_payload=policy.recipe.to_payload(),
                full_build_policy_payload=policy.to_payload(),
                build_fold_policy_payloads=tuple(
                    (fold, policy.to_payload()) for fold in range(5)
                ),
                assumption_ids=tuple(
                    assumption.assumption_id for assumption in policy.recipe.assumptions
                ),
                failure_conditions=tuple(
                    assumption.failure_condition for assumption in policy.recipe.assumptions
                ),
            )
        )
        seen_families.add(family)
    if atlas is not None:
        selected.append(
            NumericalAlternativeSpec(
                candidate_id="atlas_70_30",
                family="atlas_overlay",
                materializer_kind="atlas",
                recipe_payload=policy.recipe.to_payload(),
                full_build_policy_payload=policy.to_payload(),
                build_fold_policy_payloads=tuple(
                    (fold, policy.to_payload()) for fold in range(5)
                ),
                assumption_ids=tuple(
                    assumption.assumption_id for assumption in policy.recipe.assumptions
                ),
                failure_conditions=tuple(
                    assumption.failure_condition for assumption in policy.recipe.assumptions
                ),
            )
        )
    return NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=champion.to_payload(),
        alternatives=tuple(selected),
        atlas_release_sha256=atlas.fingerprint if atlas is not None else None,
        source_fingerprints=source_fingerprints,
        runtime_fingerprints=runtime_fingerprints,
    )


def _build_registry(
    tasks: Sequence[ContextTask],
    release: NumericalSupplyRelease,
    materializer: NumericalPackageMaterializer,
) -> FrozenNumericalPackageRegistry:
    anchor = parse_champion_release(
        cast(dict[str, object], release.to_payload()["anchor_release_payload"])
    )

    def build(original: ContextTask, supplied: NumericalSupplyRelease):
        safe = _sanitized_context_task(original)
        source = run_numerical_loop(
            RuntimeTask(
                safe.numeric.task_id,
                safe.numeric.history_values,
                safe.numeric.prediction_length,
                safe.numeric.frequency,
                (),
            ),
            screening_policy=materializer.screening_policy,
            candidate_runner=materializer.forecast_store.forecast,
            combined_policies=materializer.combined_policies,
            decision_policy=materializer.decision_policy,
            hindcast_config=materializer.hindcast_config,
            component_fingerprints={
                **materializer.source_fingerprints,
                **materializer.runtime_fingerprints,
            },
            champion_release=anchor,
        )
        alternatives = {item.name: item for item in source.ranked_alternatives}
        for specification in supplied.alternatives:
            item = materializer._materialize_alternative(source, safe, specification)
            if item is not None:
                alternatives[item.name] = item
        return bound_numerical_package(source, supplied, alternatives)

    return build_package_registry(tasks, release, build)


class _RetrievalPublisher:
    def __init__(
        self,
        releases: Path,
        library: RetrievalSkillLibrary,
        cache: PackageCacheBackedEvaluator,
        schedule: object,
        task_map: Mapping[str, ContextTask],
        *,
        split_sha256: str,
        runtime_fingerprints: Mapping[str, str],
        train_count: int,
        dev_count: int,
    ) -> None:
        self.releases = releases
        self.library = library.clone(persist=False, read_only=True)
        self.cache = cache
        self.schedule = schedule
        self.task_map = dict(task_map)
        self.audit = {
            "state": "accepted",
            "train_dev_split_sha256": split_sha256,
            "verifier_sha256": runtime_fingerprints["retrieval_verifier"],
            "evaluator_sha256": runtime_fingerprints["llm_runtime"],
            "metric_sha256": runtime_fingerprints["metric_policy"],
            "metric_cap": 5.0,
            "train_summary": {"task_count": train_count},
            "dev_summary": {"task_count": dev_count},
            "acceptance_reason": "package_dev_gates_passed",
        }

    def publish(
        self, finalist: PackageCandidate, phase_parent: PackageCoordinateState
    ) -> PackageCoordinateState:
        from evolving_loop.cli import _publish_or_resume_accepted_retrieval_release

        genome = finalist.state.bundle.policy.retrieval_genome
        parent_genome = phase_parent.bundle.policy.retrieval_genome
        if genome is None or parent_genome is None:
            raise ValueError("Retrieval publication requires bound Genomes")
        parent_release = _load_retrieval_release_for_operator(
            self.releases / parent_genome.version
        )
        release = _publish_or_resume_accepted_retrieval_release(
            self.releases,
            genome,
            skills=tuple(skill.to_payload() for skill in self.library.all()),
            audit=self.audit,
            parent_release=parent_release,
        )
        policy = embed_retrieval_release(
            finalist.state.bundle.policy,
            release,
            changelog="Accepted package Retrieval coordinate.",
        )
        policy = replace(
            policy,
            retrieval_skill_source=self.library.clone(persist=False, read_only=True),
        )
        selected = phase_parent.with_policy(policy, target="retrieval")
        self.cache.promote_cached_bundle(
            finalist.state.bundle,
            selected.bundle,
            selected.registry,
            self.schedule.tasks_for("dev20", self.task_map),
            stage="dev20",
        )
        return selected


class _SmokeRetrievalProposer:
    def __init__(
        self, proposer: RetrievalGenomeProposer, library: RetrievalSkillLibrary
    ) -> None:
        self.proposer = proposer
        self.library = library

    def propose(
        self,
        parent: PackageCoordinateState,
        feedback: PackageProposalFeedback,
        *,
        generation: int,
        child_count: int,
    ) -> tuple[PackageCandidate, ...]:
        if child_count != 1:
            raise ValueError("smoke Retrieval proposes exactly one Child")
        parent_genome = parent.bundle.policy.retrieval_genome
        if parent_genome is None:
            raise ValueError("smoke Retrieval requires a bound Parent")
        number = int(parent_genome.version[1:]) + 1
        slot = self.proposer.propose_slot(
            parent_genome,
            scope=CHILD_SCOPES[0],
            version=f"v{number:03d}",
            generation=generation,
            feedback=feedback.to_payload(),
            skill_library=self.library.clone(persist=False, read_only=True),
        )
        reason: str | None = None
        state = parent
        if slot.genome is None:
            reason = "invalid_schema"
        else:
            try:
                policy = embed_retrieval_candidate(
                    parent.bundle.policy,
                    slot.genome,
                    self.library.clone(persist=False, read_only=True),
                    changelog="Non-formal smoke Retrieval Child.",
                )
                state = parent.with_policy(policy, target="retrieval")
                if (
                    retrieval_behavior_fingerprint(slot.genome)
                    == retrieval_behavior_fingerprint(parent_genome)
                ):
                    reason = "duplicate_child"
                    state = parent
            except Exception:
                reason = "cross_coordinate_change"
                state = parent
        identity = proposal_fingerprint(
            target="retrieval",
            generation=generation,
            slot=0,
            payload={
                "raw_proposal_sha256": slot.proposal_sha256,
                "invalid_reason": reason,
            },
        )
        return (PackageCandidate(0, "retrieval", state, identity, reason),)


class _SmokeDecisionProposer:
    def __init__(self, engine: PackageDecisionEvolutionEngine) -> None:
        self.engine = engine

    def propose(
        self,
        parent: PackageCoordinateState,
        feedback: PackageProposalFeedback,
        *,
        generation: int,
        child_count: int,
    ) -> tuple[PackageCandidate, ...]:
        if child_count != 1:
            raise ValueError("smoke Decision proposes exactly one Child")
        policy = parent.bundle.policy
        self.engine._version = int(policy.version[1:]) + 1
        raw = self.engine.mutate(
            policy,
            _proposal_evaluation(policy, feedback.parent_summary),
            child_index=0,
        )
        reason: str | None = None
        state = parent
        try:
            state = parent.with_policy(raw, target="decision")
            before = package_principal_fingerprints(parent.bundle)
            after = package_principal_fingerprints(state.bundle)
            changed = tuple(name for name in before if before[name] != after[name])
            if changed != ("decision",):
                raise ValueError("Decision smoke Child crossed coordinate ownership")
        except Exception:
            reason = "cross_coordinate_change"
            state = parent
        identity = proposal_fingerprint(
            target="decision",
            generation=generation,
            slot=0,
            payload={"policy": raw.to_payload(), "invalid_reason": reason},
        )
        return (PackageCandidate(0, "decision", state, identity, reason),)


class _SmokeNumericalProposer:
    def __init__(
        self,
        proposer: ChampionProposerAdapter,
        materializer: NumericalPackageMaterializer,
        build_rows: Sequence[object],
        fold_manifest: GroupFoldManifest,
        tasks: Sequence[ContextTask],
    ) -> None:
        self.proposer = proposer
        self.materializer = materializer
        self.build_rows = tuple(build_rows)
        self.fold_manifest = fold_manifest
        self.tasks = tuple(tasks)

    def propose(
        self,
        parent: PackageCoordinateState,
        feedback: PackageProposalFeedback,
        *,
        generation: int,
        child_count: int,
    ) -> tuple[PackageCandidate, ...]:
        del feedback
        if child_count != 1:
            raise ValueError("smoke Numerical proposes exactly one Child")
        release = parse_numerical_supply_release(
            parent.bundle.to_payload()["numerical_release_payload"]
        )
        anchor = parse_champion_release(
            cast(dict[str, object], release.to_payload()["anchor_release_payload"])
        )
        raw_identity = _digest({"parent": release.fingerprint, "generation": generation})
        state = parent
        reason: str | None = "materialization_failed"
        try:
            recipes = self.proposer.propose(
                anchor,
                ProposerEvidence("adaptive_train_build_diagnostic", False, (), ()),
                generation=generation + 1,
            )
            reviewed = {row.candidate_name for row in self.build_rows}
            recipe = next(
                item
                for item in sorted(recipes, key=champion_fingerprint)
                if isinstance(item, ChampionRecipe)
                and set(item.parents).issubset(reviewed)
            )
            full = fit_champion_recipe(recipe, self.build_rows, anchor)
            folds = []
            for fold in range(5):
                ids = {
                    task_id
                    for task_id, assigned in self.fold_manifest.task_fold_map.items()
                    if assigned != fold
                }
                rows = tuple(row for row in self.build_rows if row.task_id in ids)
                folds.append((fold, fit_champion_recipe(recipe, rows, anchor)))
            family = "combined"
            if recipe.kind == "select":
                entry = self.materializer.screening_policy.get(recipe.parents[0])
                if entry is None:
                    raise ValueError("unknown Numerical smoke parent")
                family = entry.family
            alternative = NumericalAlternativeSpec(
                candidate_id=recipe.name,
                family=cast(str, family),
                materializer_kind="champion",
                recipe_payload=recipe.to_payload(),
                full_build_policy_payload=full.to_payload(),
                build_fold_policy_payloads=tuple(
                    (fold, policy.to_payload()) for fold, policy in folds
                ),
                assumption_ids=tuple(item.assumption_id for item in recipe.assumptions),
                failure_conditions=tuple(
                    item.failure_condition for item in recipe.assumptions
                ),
            )
            by_family = {item.family: item for item in release.alternatives}
            by_family[alternative.family] = alternative
            alternatives = tuple(
                by_family[family]
                for family in ("statistical", "tsfm", "combined", "atlas_overlay")
                if family in by_family
            )
            child_release = NumericalSupplyRelease(
                schema_version=1,
                version=f"n{int(release.version[1:]) + 1:03d}",
                parent_sha256=release.fingerprint,
                anchor_release_payload=dict(release.anchor_release_payload),
                alternatives=alternatives,
                atlas_release_sha256=release.atlas_release_sha256,
                source_fingerprints={
                    **release.source_fingerprints,
                    "smoke_fit": _digest(
                        {
                            "recipe": recipe.to_payload(),
                            "full": full.to_payload(),
                            "folds": [policy.to_payload() for _fold, policy in folds],
                        }
                    ),
                },
                runtime_fingerprints=release.runtime_fingerprints,
            )
            smoke_materializer = object.__new__(NumericalPackageMaterializer)
            smoke_materializer.__dict__.update(self.materializer.__dict__)
            smoke_materializer.fold_manifest = self.fold_manifest
            registry = _build_registry(self.tasks, child_release, smoke_materializer)
            state = parent.with_numerical(child_release, registry)
            raw_identity = champion_fingerprint(recipe)
            reason = None
        except Exception:
            pass
        identity = proposal_fingerprint(
            target="numerical",
            generation=generation,
            slot=0,
            payload={"raw_proposal_sha256": raw_identity, "invalid_reason": reason},
        )
        return (PackageCandidate(0, "numerical", state, identity, reason),)


class _SmokeCoordinatePhaseRunner:
    """One-Child non-formal runner; formal three-Child validation stays untouched."""

    def __init__(
        self,
        target: str,
        proposer: object,
        evaluator: PackageCacheBackedEvaluator,
        schedule: _SmokeSchedule,
        task_map: Mapping[str, ContextTask],
        artifact_store: PackageArtifactStore,
        *,
        retrieval_publisher: _RetrievalPublisher | None = None,
    ) -> None:
        self.target = cast(str, target)
        self.proposer = proposer
        self.evaluator = evaluator
        self.schedule = schedule
        self.task_map = dict(task_map)
        self.artifact_store = artifact_store
        self.retrieval_publisher = retrieval_publisher
        self.gate_config = PackageGateConfig()

    def _evaluate(
        self, state: PackageCoordinateState, stage: str, *, cache_only: bool = False
    ) -> PackageEvaluation:
        return self.evaluator.evaluate(
            state.bundle,
            state.registry,
            self.schedule.tasks_for(stage, self.task_map),
            stage=stage,
            cache_only=cache_only,
        )

    def run(
        self,
        parent: PackageCoordinateState,
        initial: PackageCoordinateState,
        *,
        generation: int,
    ) -> PackageCoordinatePhaseOutcome:
        parent_screen = self._evaluate(parent, "screen8")
        feedback = PackageProposalFeedback.from_evaluations(
            parent=parent_screen,
            gate_names=(
                "minimum_relative_joint_gain",
                "maximum_task_joint_regret",
                "p95_srmse",
            ),
        )
        candidates = tuple(
            self.proposer.propose(
                parent, feedback, generation=generation, child_count=1
            )
        )
        if len(candidates) != 1 or candidates[0].slot != 0:
            raise ValueError("smoke proposer must return singleton slot zero")
        candidate = candidates[0]
        self.artifact_store.record_candidate_evidence(
            self.target,
            generation,
            candidate.proposal_sha256,
            {
                "slot": 0,
                "invalid_reason": candidate.invalid_reason,
                "bundle_sha256": candidate.state.bundle.fingerprint(),
                "formal_run": False,
            },
        )
        if candidate.invalid_reason is not None:
            return PackageCoordinatePhaseOutcome(
                target=cast(str, self.target),
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason="the smoke Child was structurally invalid",
                evidence=(),
            )
        evidence: list[PackageStageEvidence] = []
        finalist_evaluation: PackageEvaluation | None = None
        for stage in ("build64", "calibration16", "dev20"):
            parent_evaluation = self._evaluate(parent, stage)
            initial_evaluation = self._evaluate(initial, stage)
            child_evaluation = self._evaluate(candidate.state, stage)
            failures = package_full_gate_failures(
                child_evaluation,
                parent_evaluation,
                self.gate_config,
                stage={
                    "build64": "build",
                    "calibration16": "calibration",
                    "dev20": "dev",
                }[stage],
                fold_manifest=(
                    self.schedule.fold_manifest if stage == "build64" else None
                ),
                initial=initial_evaluation,
            )
            record = PackageStageEvidence(
                stage,
                parent.bundle.fingerprint(),
                candidate.state.bundle.fingerprint(),
                parent_evaluation,
                child_evaluation,
                failures,
                True,
            )
            evidence.append(record)
            self.artifact_store.record_stage_evidence(record)
            if failures:
                return PackageCoordinatePhaseOutcome(
                    target=cast(str, self.target),
                    parent=parent,
                    finalist=None,
                    selected=parent,
                    accepted=False,
                    improved=False,
                    reason=f"the smoke Child failed the {stage} gate",
                    evidence=tuple(evidence),
                )
            finalist_evaluation = child_evaluation
        assert finalist_evaluation is not None
        selected = candidate.state
        if self.target == "retrieval":
            if self.retrieval_publisher is None:
                raise ValueError("smoke Retrieval requires a publisher")
            selected = self.retrieval_publisher.publish(candidate, parent)
        replay = self.evaluator.evaluate(
            selected.bundle,
            selected.registry,
            self.schedule.tasks_for("dev20", self.task_map),
            stage="dev20",
            cache_only=True,
        )
        if replay.result_bytes() != finalist_evaluation.result_bytes():
            return PackageCoordinatePhaseOutcome(
                target=cast(str, self.target),
                parent=parent,
                finalist=None,
                selected=parent,
                accepted=False,
                improved=False,
                reason="smoke cache replay changed finalist behavior",
                evidence=tuple(evidence),
            )
        acceptance = {
            "schema_version": 1,
            "formal_run": False,
            "target": self.target,
            "parent_bundle_sha256": parent.bundle.fingerprint(),
            "finalist_bundle_sha256": candidate.state.bundle.fingerprint(),
            "selected_bundle_sha256": selected.bundle.fingerprint(),
            "proposal_sha256": candidate.proposal_sha256,
            "schedule_sha256": self.schedule.fingerprint,
            "stage_evidence": [item.to_payload() for item in evidence],
            "replay_result_sha256": hashlib.sha256(replay.result_bytes()).hexdigest(),
        }
        evidence_sha256 = _digest(acceptance)
        self.artifact_store.record_acceptance_evidence(evidence_sha256, acceptance)
        if not self.artifact_store.contains_evidence(evidence_sha256):
            raise ValueError("smoke acceptance evidence was not durable")
        return PackageCoordinatePhaseOutcome(
            target=cast(str, self.target),
            parent=parent,
            finalist=candidate,
            selected=selected.seal_acceptance(evidence_sha256),
            accepted=True,
            improved=True,
            reason="accepted after non-formal smoke cache replay",
            evidence=tuple(evidence),
        )


def _step_payload(
    step: PackageCoordinateStep, accepted: PackageCoordinateState
) -> dict[str, object]:
    return {
        "generation": step.generation,
        "target": step.target,
        "accepted": step.accepted,
        "reason": step.reason,
        "parent_fingerprints": dict(step.parent_fingerprints),
        "child_fingerprints": dict(step.child_fingerprints),
        "accepted_fingerprints": dict(step.accepted_fingerprints),
        "changed_modules": list(step.changed_modules),
        "parent_bytes_sha256": step.parent_bytes_sha256,
        "child_bytes_sha256": step.child_bytes_sha256,
        "accepted_bytes_sha256": step.accepted_bytes_sha256,
        "parent_registry_sha256": step.parent_registry_sha256,
        "child_registry_sha256": step.child_registry_sha256,
        "accepted_registry_sha256": step.accepted_registry_sha256,
        "public_test_accessed": step.public_test_accessed,
        "accepted_bundle_payload": accepted.bundle.to_payload(),
    }


class _CheckpointRecorder:
    def __init__(
        self,
        store: PackageArtifactStore,
        *,
        run_sha256: str,
        schedule_sha256: str,
        initial: PackageCoordinateState,
        checkpoint: PackageCheckpoint | None,
    ) -> None:
        self.store = store
        self.run_sha256 = run_sha256
        self.schedule_sha256 = schedule_sha256
        self.initial = initial
        self.steps = list(checkpoint.completed_steps if checkpoint else ())
        self.current = initial

    def bind_current(self, current: PackageCoordinateState) -> None:
        self.current = current

    def record(
        self, step: PackageCoordinateStep, accepted: PackageCoordinateState
    ) -> None:
        if step.generation < len(self.steps):
            return
        if step.generation != len(self.steps):
            raise PackageArtifactError("checkpoint coordinate generation is not contiguous")
        payload = _step_payload(step, accepted)
        self.store.append_coordinate_step(payload)
        if step.accepted:
            self.store.publish_accepted_bundle(accepted.bundle.to_payload())
        self.steps.append(payload)
        self.current = accepted
        self.store.write_checkpoint(
            PackageCheckpoint(
                schema_version=1,
                run_sha256=self.run_sha256,
                schedule_sha256=self.schedule_sha256,
                initial_bundle_payload=self.initial.bundle.to_payload(),
                current_bundle_payload=accepted.bundle.to_payload(),
                completed_steps=tuple(self.steps),
                candidate_fingerprints=(),
                cache_fingerprints=(),
                consumed_stages=(),
            )
        )


class _CheckpointingPhase:
    def __init__(self, phase: object, recorder: _CheckpointRecorder, *, offset: int) -> None:
        self.phase = phase
        self.recorder = recorder
        self.offset = offset
        self.target = phase.target

    def run(
        self,
        parent: PackageCoordinateState,
        initial: PackageCoordinateState,
        *,
        generation: int,
    ) -> PackageCoordinatePhaseOutcome:
        actual = generation + self.offset
        outcome = self.phase.run(parent, initial, generation=actual)
        step, selected = PackageCoordinateController._apply(
            actual, self.target, parent, outcome
        )
        self.recorder.record(step, selected)
        if (
            self.target == "retrieval"
            and not selected.bundle.policy.has_accepted_retrieval_release
        ):
            self.recorder.record(
                _skip_step(
                    actual + 1,
                    selected,
                    "Decision phase requires a non-v000 accepted Retrieval release",
                ),
                selected,
        )
        return outcome


def _state_from_payload(
    payload: Mapping[str, object],
    *,
    tasks: Sequence[ContextTask],
    materializer: NumericalPackageMaterializer,
    library: RetrievalSkillLibrary,
) -> PackageCoordinateState:
    raw = dict(payload)
    if raw.pop("schema_version", None) != 2:
        raise PackageArtifactError("checkpoint package bundle schema changed")
    policy_payload = raw.pop("policy", None)
    if not isinstance(policy_payload, Mapping):
        raise PackageArtifactError("checkpoint package policy is malformed")
    policy = replace(
        HarnessPolicy(**dict(policy_payload)),
        retrieval_skill_source=library.clone(persist=False, read_only=True),
    )
    bundle = PackageCoordinateBundle(policy=policy, **raw)
    release = parse_numerical_supply_release(
        cast(dict[str, object], bundle.to_payload()["numerical_release_payload"])
    )
    registry = _build_registry(tasks, release, materializer)
    if registry.fingerprint != bundle.numerical_manifest_sha256:
        raise PackageArtifactError("checkpoint Numerical registry fingerprint changed")
    return PackageCoordinateState(bundle, registry)


def _run_controller(
    phases: tuple[object | None, object | None, object | None],
    *,
    cycles: int,
    offset: int,
    parent: PackageCoordinateState,
    initial: PackageCoordinateState,
    recorder: _CheckpointRecorder,
) -> PackageCoordinateState:
    wrapped = tuple(
        None
        if phase is None
        else _CheckpointingPhase(phase, recorder, offset=offset)
        for phase in phases
    )
    controller = PackageCoordinateController(
        wrapped[0], wrapped[1], wrapped[2], cycles=cycles
    )
    selected, trace = controller.run(parent, initial)
    for step in trace:
        actual = replace(step, generation=step.generation + offset)
        if actual.generation >= len(recorder.steps):
            recorder.record(actual, selected)
    return selected


def _run_manifest_core(
    args: argparse.Namespace,
    *,
    split_manifest: Mapping[str, object],
    train: Sequence[ContextTask],
    dev: Sequence[ContextTask],
    tasks: Sequence[ContextTask],
    schedule: PackageStageSchedule | _SmokeSchedule,
    source_fingerprints: Mapping[str, str],
    runtime_fingerprints: Mapping[str, str],
    champion_path: Path,
    seed: RetrievalRelease,
    atlas_status: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "formal_run": not args.smoke,
        "configuration_sha256": _configuration_identity(args),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "train_task_ids_sha256": _digest([task.numeric.task_id for task in train]),
        "dev_task_ids_sha256": _digest([task.numeric.task_id for task in dev]),
        "counts": (
            {"train": 80, "build": 64, "calibration": 16, "dev": 20}
            if not args.smoke
            else {"train": 80, "build": 8, "calibration": 2, "dev": 2}
        ),
        "schedule_sha256": schedule.fingerprint,
        "source_fingerprints": dict(source_fingerprints),
        "runtime_fingerprints": dict(runtime_fingerprints),
        "input_fingerprints": {
            "champion_release": _file_sha256(champion_path),
            "retrieval_seed_manifest": seed.manifest_file_sha256,
            "retrieval_skills": seed.skills_file_sha256,
            "registered_tasks": _digest(
                [task_registry_fingerprint(task) for task in tasks]
            ),
        },
        "atlas": dict(atlas_status),
        "retrieval_strict_gain_target": "final",
        "public_test_accessed": False,
    }


def _load_or_create_checkpoint(
    args: argparse.Namespace,
    *,
    artifact_store: PackageArtifactStore,
    initial: PackageCoordinateState,
    run_sha256: str,
    schedule_sha256: str,
    tasks: Sequence[ContextTask],
    materializer: NumericalPackageMaterializer,
    library: RetrievalSkillLibrary,
) -> tuple[PackageCoordinateState, PackageCheckpoint]:
    if args.resume:
        checkpoint = artifact_store.load_checkpoint(
            expected_run_sha256=run_sha256,
            expected_schedule_sha256=schedule_sha256,
        )
        if _digest(checkpoint.initial_bundle_payload) != _digest(
            initial.bundle.to_payload()
        ):
            raise PackageArtifactError("resume initial bundle fingerprint changed")
        current = _state_from_payload(
            checkpoint.current_bundle_payload,
            tasks=tasks,
            materializer=materializer,
            library=library,
        )
        return current, checkpoint
    checkpoint = PackageCheckpoint(
        schema_version=1,
        run_sha256=run_sha256,
        schedule_sha256=schedule_sha256,
        initial_bundle_payload=initial.bundle.to_payload(),
        current_bundle_payload=initial.bundle.to_payload(),
        completed_steps=(),
        candidate_fingerprints=(),
        cache_fingerprints=(),
        consumed_stages=(),
    )
    artifact_store.write_checkpoint(checkpoint)
    return initial, checkpoint


def _build_phases(
    args: argparse.Namespace,
    *,
    initial: PackageCoordinateState,
    tasks: Sequence[ContextTask],
    task_map: Mapping[str, ContextTask],
    schedule: PackageStageSchedule | _SmokeSchedule,
    formal_schedule: PackageStageSchedule,
    build_rows: Sequence[object],
    materializer: NumericalPackageMaterializer,
    module: object,
    portfolio: object,
    screening: object,
    library: RetrievalSkillLibrary,
    authority_seed: RetrievalRelease,
    releases: Path,
    artifact_store: PackageArtifactStore,
    runtime_fingerprints: Mapping[str, str],
    split_sha256: str,
) -> tuple[object, object, object]:
    llm = CodexCLIClient(
        CodexCLIConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=900,
            cache_dir=Path(args.authority_dir) / "llm-cache",
        )
    )

    def skills_for(policy: HarnessPolicy) -> RetrievalSkillLibrary:
        supplied = policy.retrieval_skill_source
        return (
            supplied.clone(persist=False, read_only=True)
            if isinstance(supplied, RetrievalSkillLibrary)
            else library.clone(persist=False, read_only=True)
        )

    def retrieval_factory(policy: HarnessPolicy) -> TwoStageRetrievalAgent:
        genome = policy.retrieval_genome
        if genome is None:
            raise ValueError("package policy has no Retrieval Genome")
        return TwoStageRetrievalAgent(llm, genome, skills_for(policy))

    def decision_factory(policy: HarnessPolicy) -> DecisionAgent:
        return DecisionAgent(llm, None, prompt=policy.decision_prompt)

    cached = PackageCacheBackedEvaluator(
        PackagePipelineEvaluator(retrieval_factory, decision_factory),
        Path(args.authority_dir) / "package-inference-cache",
        runtime_fingerprints=runtime_fingerprints,
    )
    retrieval_config = RetrievalEvolutionConfig(
        generations=1,
        random_seed=args.seed,
        strict_gain_target="final",
        resume=args.resume,
    )
    retrieval_genomes = RetrievalGenomeProposer(
        llm,
        transient_retries=retrieval_config.transient_retries,
        version_origin=authority_seed.genome.version,
    )
    decision_evaluator = PackageDecisionEvaluator(
        initial.registry,
        retrieval_factory,
        decision_factory,
        dependency_fingerprints={
            "retrieval_factory": runtime_fingerprints["retrieval_runtime"],
            "decision_factory": runtime_fingerprints["decision_runtime"],
            "bridge_runtime": runtime_fingerprints["bridge_runtime"],
        },
    )
    decision_engine = PackageDecisionEvolutionEngine(
        llm,
        decision_evaluator,
        CoEvolutionConfig(
            generations=1,
            children_per_generation=args.children_per_coordinate,
            mode="genome",
            target="decision",
            checkpoint_path=None,
            successive_halving=False,
        ),
    )
    champion_proposer = ChampionProposerAdapter.codex_cli(
        identity=_digest(
            {"model": args.model, "reasoning_effort": args.reasoning_effort}
        ),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        inventory=_inventory(module, portfolio, screening),
        timeout_seconds=900,
        cache_dir=str(Path(args.authority_dir) / "llm-cache"),
    )
    publisher = _RetrievalPublisher(
        releases,
        library,
        cached,
        schedule,
        task_map,
        split_sha256=split_sha256,
        runtime_fingerprints=runtime_fingerprints,
        train_count=(8 if args.smoke else 80),
        dev_count=(2 if args.smoke else 20),
    )
    if args.smoke:
        assert isinstance(schedule, _SmokeSchedule)
        proposers = (
            _SmokeNumericalProposer(
                champion_proposer,
                materializer,
                build_rows,
                schedule.fold_manifest,
                tasks,
            ),
            _SmokeRetrievalProposer(retrieval_genomes, library),
            _SmokeDecisionProposer(decision_engine),
        )
        return tuple(
            _SmokeCoordinatePhaseRunner(
                target,
                proposer,
                cached,
                schedule,
                task_map,
                artifact_store,
                retrieval_publisher=(publisher if target == "retrieval" else None),
            )
            for target, proposer in zip(
                ("numerical", "retrieval", "decision"), proposers, strict=True
            )
        )  # type: ignore[return-value]

    numerical = NumericalCandidateProposer(
        NumericalPackageProposer(
            proposer=champion_proposer,
            materializer=materializer,
            build_rows=build_rows,
            fold_manifest=formal_schedule.fold_manifest,
            tasks=tasks,
        )
    )
    proposers = (
        numerical,
        RetrievalCandidateProposer(retrieval_genomes, skill_library=library),
        DecisionCandidateProposer(decision_engine),
    )
    return tuple(
        PackageCoordinatePhaseRunner(
            target,
            proposer,
            cached,
            formal_schedule,
            task_map,
            PackageGateConfig(),
            artifact_store,
            retrieval_publisher=(publisher if target == "retrieval" else None),
            child_count=3,
        )
        for target, proposer in zip(
            ("numerical", "retrieval", "decision"), proposers, strict=True
        )
    )  # type: ignore[return-value]


def _resume_controller(
    args: argparse.Namespace,
    *,
    phases: tuple[object, object, object],
    initial: PackageCoordinateState,
    current: PackageCoordinateState,
    recorder: _CheckpointRecorder,
) -> PackageCoordinateState:
    completed = len(recorder.steps)
    maximum = args.cycles * 3
    if completed >= maximum:
        return current
    if completed and completed % 3 == 0 and not any(
        bool(step["accepted"]) for step in recorder.steps[-3:]
    ):
        return current
    cycle = completed // 3
    position = completed % 3
    if position:
        partial = tuple(
            None if index < position else phase
            for index, phase in enumerate(phases)
        )
        current = _run_controller(
            cast(tuple[object | None, object | None, object | None], partial),
            cycles=1,
            offset=cycle * 3,
            parent=current,
            initial=initial,
            recorder=recorder,
        )
        cycle_steps = recorder.steps[cycle * 3 : (cycle + 1) * 3]
        cycle += 1
        if not any(bool(step["accepted"]) for step in cycle_steps):
            return current
    if cycle < args.cycles:
        current = _run_controller(
            cast(tuple[object | None, object | None, object | None], phases),
            cycles=args.cycles - cycle,
            offset=cycle * 3,
            parent=current,
            initial=initial,
            recorder=recorder,
        )
    return current




def _execute_run(
    args: argparse.Namespace,
    split_manifest: Mapping[str, object],
    tasks: tuple[ContextTask, ...],
) -> int:
    if _early_resume_guard(args):
        return 0
    repo = Path(args.repo).resolve()
    champion_path = Path(args.numerical_champion_release).resolve()
    seed_path = Path(args.retrieval_seed_release).resolve()
    skills_path = Path(args.retrieval_skills).resolve()
    output = Path(args.output_dir).resolve()
    authority = Path(args.authority_dir).resolve()
    if not repo.is_dir() or not champion_path.is_file() or not seed_path.is_dir():
        raise ValueError("package evolution prerequisite path is missing")
    _clean_git_source(repo)
    source_files = _source_files(repo)
    source_fingerprints = {
        name: _file_sha256(path) for name, path in source_files
    }
    champion = parse_champion_release(read_json_object(champion_path))
    if "toto_2_0" not in {
        champion.policy.recipe.fallback_parent,
        *champion.policy.recipe.parents,
    }:
        raise ValueError("Numerical Champion must retain Toto as the safe anchor")

    supplied_seed = _load_retrieval_release_for_operator(seed_path)
    if (
        supplied_seed.genome.version != "v000"
        or supplied_seed.manifest.get("state") != "seed"
    ):
        raise ValueError("Retrieval seed release must be authoritative v000")
    if (
        not skills_path.is_file()
        or _file_sha256(skills_path) != supplied_seed.skills_file_sha256
    ):
        raise ValueError("--retrieval-skills differs from the seed release snapshot")
    releases = authority / "retrieval_releases"
    releases.mkdir(parents=True, exist_ok=True)
    authority_seed_path = releases / "v000"
    if not authority_seed_path.exists():
        shutil.copytree(seed_path, authority_seed_path)
    authority_seed = _load_retrieval_release_for_operator(authority_seed_path)
    if (
        authority_seed.genome.fingerprint() != supplied_seed.genome.fingerprint()
        or authority_seed.manifest_file_sha256 != supplied_seed.manifest_file_sha256
        or authority_seed.skills_file_sha256 != supplied_seed.skills_file_sha256
    ):
        raise ValueError("authority Retrieval seed differs from the supplied seed")
    library = RetrievalSkillLibrary._from_loaded_release(authority_seed).clone(
        persist=False, read_only=True
    )

    partitions = split_manifest["partitions"]
    if not isinstance(partitions, Mapping):
        raise ValueError("split partitions are malformed")
    train_ids = tuple(partitions["train"]["task_ids"])
    train = tasks[: len(train_ids)]
    dev = tasks[len(train_ids) :]
    task_map = {task.numeric.task_id: task for task in tasks}
    formal_schedule = PackageStageSchedule.build(
        tuple(task.numeric for task in train),
        tuple(task.numeric for task in dev),
        seed=args.seed,
    )
    schedule: PackageStageSchedule | _SmokeSchedule = (
        _SmokeSchedule.build(formal_schedule, task_map)
        if args.smoke
        else formal_schedule
    )

    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    candidates = _reviewed_candidates(module, portfolio, screening)
    runtimes = _runtime_registry(args)
    store: ForecastStore | None = None
    try:
        store = ForecastStore(
            args.forecast_store,
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(args),
            cache_only=bool(args.resume),
        )
        runtime_fingerprints = _runtime_fingerprints(
            args,
            source_fingerprints=source_fingerprints,
            forecast_store=store,
            seed_release=authority_seed,
        )
        build_ids = (
            schedule.build8_ids
            if isinstance(schedule, _SmokeSchedule)
            else schedule.build64_ids
        )
        build_tasks = tuple(task_map[task_id] for task_id in build_ids)
        fold_manifest = schedule.fold_manifest
        build_rows = _materialize_champion_rows(
            store,
            tuple(task.numeric for task in build_tasks),
            candidates,
            screening,
            dict(fold_manifest.task_fold_map),
            "build",
        )

        atlas: AtlasRelease | None = None
        atlas_status: dict[str, object]
        try:
            if args.atlas_release:
                atlas = parse_atlas_release(read_json_object(args.atlas_release))
            else:
                if args.smoke:
                    raise ValueError("Atlas requires the formal 64-task Build authority")
                atlas_rows = _materialize_atlas_rows(
                    store,
                    tuple(task.numeric for task in build_tasks),
                    candidates,
                    screening,
                    split="build",
                    hindcast_config=HindcastConfig(),
                )
                atlas = fit_atlas_release(atlas_rows, fold_manifest, AtlasPolicy())
            atlas.validate_manifest(fold_manifest)
            atlas_status = {"available": True, "sha256": atlas.fingerprint}
            output.mkdir(parents=True, exist_ok=True)
            atlas_bytes = canonical_json_bytes(atlas.to_payload())
            atlas_path = output / "atlas_release.json"
            if atlas_path.exists() and atlas_path.read_bytes() != atlas_bytes:
                raise PackageArtifactError("immutable Atlas release changed")
            if not atlas_path.exists():
                atlas_path.write_bytes(atlas_bytes)
        except Exception as error:
            if args.atlas_release:
                raise
            atlas = None
            atlas_status = {
                "available": False,
                "reason": f"{type(error).__name__}: Atlas fitting unavailable",
            }

        materializer = NumericalPackageMaterializer(
            forecast_store=store,
            screening_policy=screening,
            fold_manifest=formal_schedule.fold_manifest,
            original_tasks=tasks,
            source_fingerprints=source_fingerprints,
            runtime_fingerprints={
                "forecast_store": store.identity_hash,
                "model_runtime": runtime_fingerprints["model_runtime"],
            },
            combined_policies=portfolio.combined,
            atlas_release=atlas if not args.smoke else None,
            decision_policy=DecisionPolicy(),
            hindcast_config=HindcastConfig(),
        )
        supply = _initial_supply_release(
            champion,
            candidates,
            source_fingerprints={
                **source_fingerprints,
                "champion_release": _file_sha256(champion_path),
            },
            runtime_fingerprints={
                "forecast_store": store.identity_hash,
                "model_runtime": runtime_fingerprints["model_runtime"],
            },
            atlas=atlas if not args.smoke else None,
        )
        registry = _build_registry(tasks, supply, materializer)
        seed_policy = replace(
            embed_retrieval_release(
                HarnessPolicy(), authority_seed, changelog="Package co-evolution seed."
            ),
            retrieval_skill_source=library,
        )
        initial = PackageCoordinateState(
            PackageCoordinateBundle(
                generation=0,
                coordinate="seed",
                parent_sha256=None,
                numerical_release_payload=supply.to_payload(),
                numerical_release_sha256=supply.fingerprint,
                numerical_manifest_sha256=registry.fingerprint,
                policy=seed_policy,
                runtime_fingerprints=runtime_fingerprints,
                acceptance_evidence_sha256=None,
            ),
            registry,
        )

        run_core = _run_manifest_core(
            args,
            split_manifest=split_manifest,
            train=train,
            dev=dev,
            tasks=tasks,
            schedule=schedule,
            source_fingerprints=source_fingerprints,
            runtime_fingerprints=runtime_fingerprints,
            champion_path=champion_path,
            seed=supplied_seed,
            atlas_status=atlas_status,
        )
        run_sha256 = _digest(run_core)
        artifact_store = PackageArtifactStore(output)
        artifact_store.write_run_manifest({**run_core, "run_sha256": run_sha256})
        artifact_store.write_schedule(schedule.to_payload())
        current, checkpoint = _load_or_create_checkpoint(
            args,
            artifact_store=artifact_store,
            initial=initial,
            run_sha256=run_sha256,
            schedule_sha256=schedule.fingerprint,
            tasks=tasks,
            materializer=materializer,
            library=library,
        )
        recorder = _CheckpointRecorder(
            artifact_store,
            run_sha256=run_sha256,
            schedule_sha256=schedule.fingerprint,
            initial=initial,
            checkpoint=checkpoint,
        )
        recorder.bind_current(current)
        phases = _build_phases(
            args,
            initial=initial,
            tasks=tasks,
            task_map=task_map,
            schedule=schedule,
            formal_schedule=formal_schedule,
            build_rows=build_rows,
            materializer=materializer,
            module=module,
            portfolio=portfolio,
            screening=screening,
            library=library,
            authority_seed=authority_seed,
            releases=releases,
            artifact_store=artifact_store,
            runtime_fingerprints=runtime_fingerprints,
            split_sha256=cast(str, split_manifest["manifest_sha256"]),
        )
        current = _resume_controller(
            args,
            phases=phases,
            initial=initial,
            current=current,
            recorder=recorder,
        )
        accepted_steps = sum(bool(step["accepted"]) for step in recorder.steps)
        artifact_store.complete(
            current.bundle.to_payload(),
            accepted_steps=accepted_steps,
            rejected_steps=len(recorder.steps) - accepted_steps,
            formal_run=not args.smoke,
        )
        return 0
    finally:
        if store is not None:
            store.close()
        runtimes.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_mode(args)
    split, train_ids, dev_ids = _validated_split(args.split_file)
    tasks = load_context_tasks_by_ids(args.tasks_file, (*train_ids, *dev_ids))
    if tuple(task.numeric.task_id for task in tasks) != (*train_ids, *dev_ids):
        raise ValueError("loaded tasks do not match exact Train and Dev membership")
    return _execute_run(args, split, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
