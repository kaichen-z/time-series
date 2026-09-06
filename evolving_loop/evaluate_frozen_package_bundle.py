"""Evaluate one sealed package co-evolution bundle on Public-99 exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.llm import CodexCLIClient, CodexCLIConfig, TransientLLMError
from common.metrics import linear_quantile
from common.payload import canonical_json_bytes, read_json_object
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, load_context_tasks_by_ids
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.numerical_two_stage import run_numerical_two_stage
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateState,
    PackageCoordinateStep,
    package_principal_fingerprints,
)
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.package_stage_runner import PackageStageSchedule
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkill,
    RetrievalSkillLibrary,
    _activate_verified_release_library,
    _record_digest,
)
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from evolving_loop.run_package_coevolution import (
    _build_registry,
    _clean_git_source,
    _digest,
    _file_sha256,
    _load_screening_policy,
    _source_files,
    _split_digest,
)
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.module import read_module
from numerical_agent.evolution.numerical_selector import DecisionPolicy, HindcastConfig
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.specialist_atlas import parse_atlas_release
from numerical_agent.evolution.task_local_evolution import GroupFoldManifest
from numerical_agent.main import _add_tsfm_runtime_options, _runtime_registry
from numerical_agent.run_selector_evolution import _forecast_runtime_identity


_RUNTIME_KEYS = frozenset(
    {
        "bridge_runtime",
        "numerical_runtime",
        "retrieval_runtime",
        "decision_runtime",
        "retrieval_verifier",
        "metric_policy",
        "model_runtime",
        "llm_runtime",
    }
)
_METADATA = {
    "benchmark_role": "historically_consumed_final_regression",
    "selection_used_public": False,
    "public_result_may_feed_evolution": False,
    "primary_metrics": ["smae", "srmse"],
    "probabilistic_metric_reported": False,
}


class FrozenPackageEvaluationError(ValueError):
    """Raised when the frozen-run or one-shot Public contract is violated."""


class _RegistryReference(FrozenNumericalPackageRegistry):
    """Non-executable registry identity used while authenticating bundle artifacts."""

    def __init__(self, fingerprint: str, release_sha256: str) -> None:
        self.fingerprint = fingerprint
        self._release_sha256 = release_sha256
        self._packages = MappingProxyType({})

    @property
    def task_ids(self) -> tuple[str, ...]:
        return ()

    def package_for(self, task: ContextTask):  # pragma: no cover - fail-closed guard
        del task
        raise FrozenPackageEvaluationError(
            "verified registry references must be rebuilt from Public histories"
        )


@dataclass(frozen=True)
class NamedPackageState:
    name: str
    state: PackageCoordinateState
    direct_parent_name: str | None


@dataclass(frozen=True)
class VerifiedPackageRun:
    run_manifest: Mapping[str, object]
    schedule: PackageStageSchedule
    initial_state: PackageCoordinateState
    final_state: PackageCoordinateState
    trace: tuple[PackageCoordinateStep, ...]
    runtime_fingerprints: Mapping[str, str]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FrozenPackageEvaluationError("artifact is not canonical JSON") from error


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _read_canonical_json(path: Path, label: str) -> dict[str, object]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenPackageEvaluationError(f"{label} artifact is missing or malformed") from error
    if not isinstance(payload, dict) or encoded != _canonical_bytes(payload):
        raise FrozenPackageEvaluationError(f"{label} artifact is not canonical")
    return payload


def _schedule_from_payload(payload: Mapping[str, object]) -> PackageStageSchedule:
    try:
        fold = cast(Mapping[str, object], payload["fold_manifest"])
        groups = tuple(
            (
                str(row["group_sha256"]),
                tuple(cast(Sequence[str], row["task_ids"])),
                int(row["fold"]),
            )
            for row in cast(Sequence[Mapping[str, object]], fold["groups"])
        )
        manifest = GroupFoldManifest(
            schema_version=cast(int, fold["schema_version"]),
            seed=cast(int, fold["seed"]),
            fold_count=cast(int, fold["fold_count"]),
            groups=groups,
            grouping_fingerprint=cast(str, fold["grouping_fingerprint"]),
        )
        return PackageStageSchedule(
            seed=cast(int, payload["seed"]),
            screen8_ids=tuple(cast(Sequence[str], payload["screen8_ids"])),
            screen32_ids=tuple(cast(Sequence[str], payload["screen32_ids"])),
            build64_ids=tuple(cast(Sequence[str], payload["build64_ids"])),
            calibration16_ids=tuple(cast(Sequence[str], payload["calibration16_ids"])),
            dev20_ids=tuple(cast(Sequence[str], payload["dev20_ids"])),
            fold_manifest=manifest,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FrozenPackageEvaluationError("evolution schedule is malformed") from error


def _bundle_from_payload(payload: object) -> PackageCoordinateBundle:
    if not isinstance(payload, Mapping):
        raise FrozenPackageEvaluationError("package bundle payload is malformed")
    raw = dict(payload)
    if raw.pop("schema_version", None) != 2:
        raise FrozenPackageEvaluationError("package bundle schema is not supported")
    policy_payload = raw.pop("policy", None)
    if not isinstance(policy_payload, Mapping):
        raise FrozenPackageEvaluationError("package bundle policy is malformed")
    try:
        return PackageCoordinateBundle(
            policy=HarnessPolicy(**dict(policy_payload)),
            **raw,
        )
    except (TypeError, ValueError) as error:
        raise FrozenPackageEvaluationError("package bundle is invalid") from error


def _state_from_bundle(bundle: PackageCoordinateBundle) -> PackageCoordinateState:
    registry = _RegistryReference(
        bundle.numerical_manifest_sha256, bundle.numerical_release_sha256
    )
    return PackageCoordinateState(bundle, registry)


def _step_from_payload(payload: Mapping[str, object]) -> PackageCoordinateStep:
    try:
        return PackageCoordinateStep(
            generation=cast(int, payload["generation"]),
            target=cast(str, payload["target"]),
            accepted=cast(bool, payload["accepted"]),
            reason=cast(str, payload["reason"]),
            parent_fingerprints=cast(Mapping[str, str], payload["parent_fingerprints"]),
            child_fingerprints=cast(Mapping[str, str], payload["child_fingerprints"]),
            accepted_fingerprints=cast(
                Mapping[str, str], payload["accepted_fingerprints"]
            ),
            changed_modules=tuple(cast(Sequence[str], payload["changed_modules"])),
            parent_bytes_sha256=cast(str, payload["parent_bytes_sha256"]),
            child_bytes_sha256=cast(str, payload["child_bytes_sha256"]),
            accepted_bytes_sha256=cast(str, payload["accepted_bytes_sha256"]),
            parent_registry_sha256=cast(str, payload["parent_registry_sha256"]),
            child_registry_sha256=cast(str, payload["child_registry_sha256"]),
            accepted_registry_sha256=cast(str, payload["accepted_registry_sha256"]),
            public_test_accessed=cast(bool, payload.get("public_test_accessed", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FrozenPackageEvaluationError("canonical trace row is malformed") from error


def _has_public_access_marker(root: Path) -> bool:
    def marked(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                (
                    key in {"public_test_accessed", "selection_used_public"}
                    and item is not False
                )
                or marked(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(marked(item) for item in value)
        return False

    for path in sorted((*root.rglob("*.json"), *root.rglob("*.jsonl"))):
        try:
            values = (
                [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
                if path.suffix == ".jsonl"
                else [json.loads(path.read_text("utf-8"))]
            )
        except (OSError, json.JSONDecodeError) as error:
            raise FrozenPackageEvaluationError("evolution artifact is malformed") from error
        if any(marked(value) for value in values):
            return True
    return False


def verify_frozen_package_run(
    evolution_dir: str | Path,
    final_bundle_path: str | Path,
    runtime_fingerprints: Mapping[str, str],
) -> VerifiedPackageRun:
    """Authenticate a complete, Public-free, direct-lineage evolution run."""
    root = Path(evolution_dir).resolve()
    final_path = Path(final_bundle_path).resolve()
    if not root.is_dir() or final_path != root / "final_bundle.json":
        raise FrozenPackageEvaluationError("final bundle must be the canonical run artifact")
    completion_path = root / "evaluation_complete.json"
    if not completion_path.is_file():
        raise FrozenPackageEvaluationError("Public evaluation requires a complete evolution")
    completion = _read_canonical_json(completion_path, "complete evolution")
    if (
        completion.get("status") != "complete"
        or completion.get("formal_run") is not True
        or completion.get("public_test_accessed") is not False
    ):
        raise FrozenPackageEvaluationError("Public evaluation requires a complete evolution")
    manifest = _read_canonical_json(root / "run_manifest.json", "run manifest")
    manifest_core = {key: value for key, value in manifest.items() if key != "run_sha256"}
    if (
        manifest.get("formal_run") is not True
        or manifest.get("run_sha256") != _digest(manifest_core)
        or manifest.get("public_test_accessed") is not False
    ):
        raise FrozenPackageEvaluationError("complete evolution run manifest is invalid")
    runtime = dict(runtime_fingerprints)
    if (
        set(runtime) != _RUNTIME_KEYS
        or manifest.get("runtime_fingerprints") != runtime
    ):
        raise FrozenPackageEvaluationError("Public runtime differs from frozen evolution")
    if _has_public_access_marker(root):
        raise FrozenPackageEvaluationError("evolution artifacts record Public access")
    schedule_payload = _read_canonical_json(root / "group_folds.json", "schedule")
    schedule = _schedule_from_payload(schedule_payload)
    if manifest.get("schedule_sha256") != schedule.fingerprint:
        raise FrozenPackageEvaluationError("evolution schedule fingerprint drifted")
    checkpoint = _read_canonical_json(root / "checkpoint.json", "checkpoint")
    if (
        checkpoint.get("run_sha256") != manifest.get("run_sha256")
        or checkpoint.get("schedule_sha256") != schedule.fingerprint
    ):
        raise FrozenPackageEvaluationError("evolution checkpoint identity drifted")
    initial = _state_from_bundle(_bundle_from_payload(checkpoint.get("initial_bundle_payload")))
    if dict(initial.bundle.runtime_fingerprints) != runtime:
        raise FrozenPackageEvaluationError("initial package runtime drifted")

    trace_path = root / "coordinate_trace.jsonl"
    try:
        trace_payloads = tuple(
            json.loads(line)
            for line in trace_path.read_text("utf-8").splitlines()
            if line
        )
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenPackageEvaluationError("canonical trace is missing or malformed") from error
    if not trace_payloads or any(not isinstance(row, Mapping) for row in trace_payloads):
        raise FrozenPackageEvaluationError("canonical trace is incomplete")
    current = initial
    steps: list[PackageCoordinateStep] = []
    for generation, row in enumerate(trace_payloads):
        step = _step_from_payload(cast(Mapping[str, object], row))
        accepted_bundle = _bundle_from_payload(row.get("accepted_bundle_payload"))
        parent_principals = dict(package_principal_fingerprints(current.bundle))
        accepted_principals = dict(package_principal_fingerprints(accepted_bundle))
        changed = tuple(
            name
            for name in ("numerical", "retrieval", "decision")
            if parent_principals[name] != accepted_principals[name]
        )
        valid = (
            step.generation == generation
            and step.target == ("numerical", "retrieval", "decision")[generation % 3]
            and step.parent_bytes_sha256 == current.bundle.fingerprint()
            and dict(step.parent_fingerprints) == parent_principals
            and step.accepted_bytes_sha256 == accepted_bundle.fingerprint()
            and dict(step.accepted_fingerprints) == accepted_principals
            and dict(step.child_fingerprints) == accepted_principals
            and step.child_bytes_sha256 == accepted_bundle.fingerprint()
            and step.parent_registry_sha256
            == current.bundle.numerical_manifest_sha256
            and step.child_registry_sha256
            == accepted_bundle.numerical_manifest_sha256
            and step.accepted_registry_sha256
            == accepted_bundle.numerical_manifest_sha256
            and dict(accepted_bundle.runtime_fingerprints) == runtime
        )
        if step.accepted:
            valid = valid and changed == (step.target,) and step.changed_modules == changed
            valid = valid and accepted_bundle.parent_sha256 == current.bundle.fingerprint()
            valid = valid and accepted_bundle.acceptance_evidence_sha256 is not None
        else:
            valid = valid and accepted_bundle.canonical_bytes() == current.bundle.canonical_bytes()
            valid = valid and step.changed_modules == ()
        if not valid:
            raise FrozenPackageEvaluationError("canonical trace is detached or non-contiguous")
        current = _state_from_bundle(accepted_bundle)
        steps.append(step)
    final_payload = _read_canonical_json(final_path, "final bundle")
    final = _state_from_bundle(_bundle_from_payload(final_payload))
    final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    if (
        completion.get("final_bundle_sha256") != final_hash
        or final.bundle.canonical_bytes() != current.bundle.canonical_bytes()
        or checkpoint.get("current_bundle_payload") != final_payload
        or checkpoint.get("completed_steps") != list(trace_payloads)
        or completion.get("accepted_steps") != sum(step.accepted for step in steps)
        or completion.get("rejected_steps") != sum(not step.accepted for step in steps)
    ):
        raise FrozenPackageEvaluationError("final bundle does not close the canonical trace")
    return VerifiedPackageRun(
        run_manifest=MappingProxyType(dict(manifest)),
        schedule=schedule,
        initial_state=initial,
        final_state=final,
        trace=tuple(steps),
        runtime_fingerprints=MappingProxyType(dict(sorted(runtime.items()))),
    )


def _synthetic_state(
    numerical: PackageCoordinateBundle, policy: HarnessPolicy
) -> PackageCoordinateState:
    bundle = PackageCoordinateBundle(
        generation=0,
        coordinate="seed",
        parent_sha256=None,
        numerical_release_payload=numerical.numerical_release_payload,
        numerical_release_sha256=numerical.numerical_release_sha256,
        numerical_manifest_sha256=numerical.numerical_manifest_sha256,
        policy=policy,
        runtime_fingerprints=numerical.runtime_fingerprints,
        acceptance_evidence_sha256=None,
    )
    return _state_from_bundle(bundle)


def build_attribution_states(
    verified: VerifiedPackageRun,
) -> tuple[NamedPackageState, ...]:
    """Construct four nested immutable coordinate snapshots without fitting."""
    if not isinstance(verified, VerifiedPackageRun):
        raise TypeError("attribution requires a verified package run")
    initial = verified.initial_state.bundle
    final = verified.final_state.bundle
    seed_policy = initial.policy
    final_policy = final.policy
    retrieval_plus_seed_decision = replace(
        seed_policy,
        version=final_policy.version,
        parent=final_policy.parent,
        retrieval_prompt=final_policy.retrieval_prompt,
        retrieval_skills=final_policy.retrieval_skills,
        retrieval_release_payload=final_policy.retrieval_release_payload,
        retrieval_release_sha256=final_policy.retrieval_release_sha256,
    )
    return (
        NamedPackageState("initial_toto", verified.initial_state, None),
        NamedPackageState(
            "final_numerical_seed_context",
            _synthetic_state(final, seed_policy),
            "initial_toto",
        ),
        NamedPackageState(
            "final_numerical_retrieval_seed_decision",
            _synthetic_state(final, retrieval_plus_seed_decision),
            "final_numerical_seed_context",
        ),
        NamedPackageState(
            "final_bundle",
            verified.final_state,
            "final_numerical_retrieval_seed_decision",
        ),
    )


def _state_summary(named: NamedPackageState, evaluation: PackageEvaluation) -> dict[str, object]:
    rows = evaluation.task_rows

    def values(field: str) -> list[float]:
        return [float(getattr(row, field)) for row in rows]

    def distribution(field: str) -> dict[str, float]:
        data = values(field)
        return {
            "median": float(statistics.median(data)) if data else 0.0,
            "p90": float(linear_quantile(data, 0.90)) if data else 0.0,
            "p95": float(linear_quantile(data, 0.95)) if data else 0.0,
            "max": max(data, default=0.0),
        }

    release = parse_numerical_supply_release(
        cast(dict[str, object], _plain(named.state.bundle.numerical_release_payload))
    )
    families = {item.candidate_id: item.family for item in release.alternatives}
    selected = Counter(row.selected_candidate_id for row in rows)
    family_counts = Counter(
        families.get(row.selected_candidate_id, "anchor") for row in rows
    )
    return {
        "task_count": evaluation.task_count,
        "coverage": evaluation.coverage,
        "mean_smae": evaluation.mean_smae,
        "mean_srmse": evaluation.mean_srmse,
        "mean_joint": evaluation.mean_joint,
        "mean_smae_raw": evaluation.mean_smae_raw,
        "mean_srmse_raw": evaluation.mean_srmse_raw,
        "smae": distribution("final_smae"),
        "srmse": distribution("final_srmse"),
        "smae_raw": distribution("final_smae_raw"),
        "srmse_raw": distribution("final_srmse_raw"),
        "clipped_count": evaluation.clipped_count,
        "catastrophic_count": evaluation.catastrophic_count,
        "invalid_count": evaluation.invalid_count,
        "fallback_count": evaluation.fallback_count,
        "selected_alternative_counts": dict(sorted(selected.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "artifact_hashes": {
            "bundle_sha256": named.state.bundle.fingerprint(),
            "numerical_release_sha256": named.state.bundle.numerical_release_sha256,
            "numerical_registry_sha256": named.state.bundle.numerical_manifest_sha256,
            "principal_fingerprints": dict(
                package_principal_fingerprints(named.state.bundle)
            ),
        },
    }


def _comparison(
    child: PackageEvaluation, parent: PackageEvaluation
) -> dict[str, int | float]:
    left = {row.task_id: row for row in parent.task_rows}
    right = {row.task_id: row for row in child.task_rows}
    if set(left) != set(right) or set(left) != set(child.expected_task_ids):
        raise FrozenPackageEvaluationError("frozen state scores have different membership")
    wins = ties = losses = 0
    for task_id in sorted(left):
        delta = right[task_id].joint - left[task_id].joint
        if delta < -1e-12:
            wins += 1
        elif delta > 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_delta_smae": child.mean_smae - parent.mean_smae,
        "mean_delta_srmse": child.mean_srmse - parent.mean_srmse,
        "mean_delta_joint": child.mean_joint - parent.mean_joint,
    }


def score_frozen_states(
    tasks: Sequence[ContextTask],
    states: Sequence[NamedPackageState],
    evaluator: object,
) -> Mapping[str, object]:
    """Score nested states and build a report that can never feed evolution."""
    resolved = tuple(tasks)
    named = tuple(states)
    if not resolved or len({task.numeric.task_id for task in resolved}) != len(resolved):
        raise FrozenPackageEvaluationError("Public tasks must be a non-empty unique set")
    if not callable(getattr(evaluator, "evaluate_state", None)):
        raise TypeError("frozen scorer requires evaluate_state(named, tasks)")
    expected_names = (
        "initial_toto",
        "final_numerical_seed_context",
        "final_numerical_retrieval_seed_decision",
        "final_bundle",
    )
    if tuple(item.name for item in named) != expected_names:
        raise FrozenPackageEvaluationError("frozen scorer requires the four attribution states")
    task_ids = {task.numeric.task_id for task in resolved}
    evaluations: dict[str, PackageEvaluation] = {}
    per_task: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    scored_bundles: dict[
        str, tuple[PackageEvaluation, Mapping[str, Sequence[str]]]
    ] = {}
    for item in named:
        bundle_sha256 = item.state.bundle.fingerprint()
        cached = scored_bundles.get(bundle_sha256)
        if cached is None:
            evaluation = evaluator.evaluate_state(item, resolved)
            details = (
                evaluator.details_for(item.name)
                if callable(getattr(evaluator, "details_for", None))
                else {}
            )
            scored_bundles[bundle_sha256] = (evaluation, details)
        else:
            evaluation, details = cached
        if not isinstance(evaluation, PackageEvaluation):
            raise TypeError("frozen evaluator returned an invalid PackageEvaluation")
        if (
            evaluation.public_test_accessed
            or evaluation.coverage != 1.0
            or set(evaluation.expected_task_ids) != task_ids
        ):
            raise FrozenPackageEvaluationError("Public frozen state did not score completely")
        evaluations[item.name] = evaluation
        summaries[item.name] = _state_summary(item, evaluation)
        per_task[item.name] = [
            {
                "task_id": row.task_id,
                "forecast": list(row.final_forecast),
                "selected_candidate_id": row.selected_candidate_id,
                "supporting_document_ids": list(
                    cast(Mapping[str, Sequence[str]], details).get(row.task_id, ())
                ),
                "smae": row.final_smae,
                "srmse": row.final_srmse,
                "smae_raw": row.final_smae_raw,
                "srmse_raw": row.final_srmse_raw,
            }
            for row in evaluation.task_rows
        ]
    comparisons: dict[str, object] = {}
    attribution: list[dict[str, object]] = []
    for index in range(1, len(named)):
        child = named[index]
        parent = named[index - 1]
        comparison = _comparison(evaluations[child.name], evaluations[parent.name])
        comparisons[f"{child.name}_vs_{parent.name}"] = comparison
        attribution.append(
            {
                "state": child.name,
                "direct_parent": parent.name,
                "coordinate": ("numerical", "retrieval", "decision")[index - 1],
                **comparison,
            }
        )
    baseline = named[0]
    for child in named[1:]:
        key = f"{child.name}_vs_{baseline.name}"
        comparisons.setdefault(
            key, _comparison(evaluations[child.name], evaluations[baseline.name])
        )
    return {
        "schema_version": 1,
        "metadata": dict(_METADATA),
        "states": summaries,
        "comparisons": comparisons,
        "coordinate_attribution": attribution,
        "per_task": per_task,
    }


@dataclass
class _RuntimeResources:
    runtimes: object
    store: ForecastStore
    module: object
    portfolio: object
    screening: object
    source_fingerprints: Mapping[str, str]

    def close(self) -> None:
        self.store.close()
        self.runtimes.close()


def _runtime_authority(
    args: argparse.Namespace, run_manifest: Mapping[str, object]
) -> tuple[dict[str, str], _RuntimeResources]:
    repo = Path(args.repo).resolve()
    _clean_git_source(repo)
    source_fingerprints = {
        name: _file_sha256(path) for name, path in _source_files(repo)
    }
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    runtimes = _runtime_registry(args)
    try:
        store = ForecastStore(
            args.forecast_store,
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(args),
            cache_only=False,
        )
    except BaseException:
        runtimes.close()
        raise
    model = {"model": args.model, "reasoning_effort": args.reasoning_effort}
    try:
        seed_manifest = cast(Mapping[str, object], run_manifest["input_fingerprints"])[
            "retrieval_seed_manifest"
        ]
        root = Path(__file__).parent
        runtime = {
            "bridge_runtime": _file_sha256(root / "numerical_two_stage.py"),
            "numerical_runtime": _digest(
                {"source": source_fingerprints, "forecast_store": store.identity_hash}
            ),
            "retrieval_runtime": _digest(
                {
                    "implementation": _file_sha256(
                        root / "retrieval_agent" / "two_stage_agent.py"
                    ),
                    "seed_release": seed_manifest,
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
                {
                    **model,
                    "client": _file_sha256(Path(__file__).parents[1] / "common" / "llm.py"),
                }
            ),
        }
    except BaseException:
        store.close()
        runtimes.close()
        raise
    return runtime, _RuntimeResources(
        runtimes, store, module, portfolio, screening, source_fingerprints
    )


def _library_from_policy(policy: HarnessPolicy, path: Path) -> RetrievalSkillLibrary:
    skills = tuple(
        RetrievalSkill._from_storage_payload(cast(Mapping[str, object], item))
        for item in policy.retrieval_skills
    )
    active = {_record_digest(skill) for skill in skills if skill.is_active}
    library = object.__new__(RetrievalSkillLibrary)
    library.path = path
    library.persist = False
    library._read_only = True
    library._active_record_origins = {digest: "verified_release" for digest in active}
    library._file_sha256 = None
    library._skills = RetrievalSkillLibrary._validated_index(
        skills, active_record_hashes=active
    )
    return _activate_verified_release_library(library) if active else library


class _PublicStateEvaluator:
    def __init__(
        self,
        args: argparse.Namespace,
        verified: VerifiedPackageRun,
        resources: _RuntimeResources,
    ) -> None:
        self.args = args
        self.verified = verified
        self.resources = resources
        self._registries: dict[str, FrozenNumericalPackageRegistry] = {}
        self._details: dict[str, Mapping[str, Sequence[str]]] = {}
        self.llm = CodexCLIClient(
            CodexCLIConfig(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=900,
                cache_dir=Path(args.output_dir) / "llm-cache",
            )
        )
        atlas_path = Path(args.evolution_dir) / "atlas_release.json"
        self.atlas = (
            parse_atlas_release(read_json_object(atlas_path))
            if atlas_path.is_file()
            else None
        )

    def _registry(
        self, named: NamedPackageState, tasks: Sequence[ContextTask]
    ) -> FrozenNumericalPackageRegistry:
        release = parse_numerical_supply_release(
            cast(dict[str, object], _plain(named.state.bundle.numerical_release_payload))
        )
        cached = self._registries.get(release.fingerprint)
        if cached is not None:
            return cached
        materializer = object.__new__(NumericalPackageMaterializer)
        materializer.forecast_store = self.resources.store
        materializer.screening_policy = self.resources.screening
        materializer.fold_manifest = self.verified.schedule.fold_manifest
        materializer.source_fingerprints = dict(self.resources.source_fingerprints)
        materializer.runtime_fingerprints = dict(release.runtime_fingerprints)
        materializer.combined_policies = tuple(self.resources.portfolio.combined)
        materializer.atlas_release = self.atlas
        materializer.decision_policy = DecisionPolicy()
        materializer.hindcast_config = HindcastConfig()
        materializer.diagnostics_registry = None
        registry = _build_registry(tasks, release, materializer)
        self._registries[release.fingerprint] = registry
        return registry

    def evaluate_state(
        self, named: NamedPackageState, tasks: Sequence[ContextTask]
    ) -> PackageEvaluation:
        registry = self._registry(named, tasks)
        policy = named.state.bundle.policy
        library = _library_from_policy(
            policy, Path(self.args.output_dir) / f"{named.name}-skills.json"
        )

        def retrieval_factory(bound: HarnessPolicy) -> TwoStageRetrievalAgent:
            genome = bound.retrieval_genome
            if genome is None:
                raise FrozenPackageEvaluationError("frozen state has no Retrieval Genome")
            return TwoStageRetrievalAgent(
                self.llm, genome, library.clone(persist=False, read_only=True)
            )

        def decision_factory(bound: HarnessPolicy) -> DecisionAgent:
            return DecisionAgent(self.llm, None, prompt=bound.decision_prompt)

        projected = replace(
            named.state.bundle, numerical_manifest_sha256=registry.fingerprint
        )
        rows = []
        diagnostics = []
        details: dict[str, Sequence[str]] = {}
        expected_retrieval = policy.retrieval_genome
        if expected_retrieval is None:
            raise FrozenPackageEvaluationError("frozen state has no Retrieval Genome")
        expected_retrieval_sha = expected_retrieval.fingerprint()
        expected_decision_sha = hashlib.sha256(
            policy.decision_prompt.encode("utf-8")
        ).hexdigest()
        for task in tasks:
            package = registry.package_for(task)
            retrieval = retrieval_factory(policy)
            decision = decision_factory(policy)
            try:
                result = run_numerical_two_stage(
                    task,
                    package,
                    retrieval,
                    decision,
                    preserve_round1_on_round2_failure=True,
                )
            except TransientLLMError:
                raise
            except (TypeError, ValueError) as error:
                scored = PackagePipelineEvaluator._score_contract_fallback(
                    task, package, error, metric_cap=5.0
                )
                details[task.numeric.task_id] = ()
            else:
                if (
                    result.fingerprints.get("retrieval_genome")
                    != expected_retrieval_sha
                    or result.fingerprints.get("decision_prompt")
                    != expected_decision_sha
                ):
                    raise FrozenPackageEvaluationError(
                        "Public inference drifted from the frozen contextual runtime"
                    )
                scored = PackagePipelineEvaluator._score_result(
                    task, package, result, metric_cap=5.0
                )
                details[task.numeric.task_id] = tuple(
                    result.final_decision.supporting_document_ids
                )
            rows.append(scored.row)
            diagnostics.append(scored.diagnostics)
        names = sorted({key for item in diagnostics for key in item})
        aggregates = {
            key: statistics.fmean(float(item[key]) for item in diagnostics)
            for key in names
        }
        self._details[named.name] = details
        return PackageEvaluation.from_rows(
            projected.fingerprint(),
            rows,
            tuple(task.numeric.task_id for task in tasks),
            aggregates,
        )

    def details_for(self, name: str) -> Mapping[str, Sequence[str]]:
        return self._details.get(name, {})

    def close(self) -> None:
        return None


def _public_evaluator(
    args: argparse.Namespace,
    verified: VerifiedPackageRun,
    resources: _RuntimeResources,
) -> _PublicStateEvaluator:
    return _PublicStateEvaluator(args, verified, resources)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-dir", required=True)
    parser.add_argument("--final-bundle", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--forecast-store", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--resume-unscored-start",
        action="store_true",
        help="resume an output containing only the canonical started marker",
    )
    _add_tsfm_runtime_options(parser)
    return parser


def _claim_output(
    output: Path,
    evolution: Path,
    *,
    resume_unscored_start: bool = False,
) -> None:
    if output == evolution or evolution in output.parents:
        raise FrozenPackageEvaluationError(
            "Public output must be separate from the evolution directory"
        )
    started = output / "evaluation_started.json"
    marker = canonical_json_bytes(
        {"schema_version": 1, "status": "public_evaluation_started"}
    )
    existed = output.exists()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_complete.json").exists():
        raise FrozenPackageEvaluationError("Public output already completed")
    if existed:
        entries = tuple(output.iterdir())
        if entries:
            if (
                resume_unscored_start
                and set(entries) == {started}
                and started.read_bytes() == marker
            ):
                return
            if set(entries) == {started}:
                raise FrozenPackageEvaluationError("Public output was already claimed")
            raise FrozenPackageEvaluationError("Public output already contains artifacts")
    try:
        with started.open("xb") as handle:
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FrozenPackageEvaluationError("Public output was already claimed") from error


def _public_membership(
    split_path: Path, verified: VerifiedPackageRun
) -> tuple[str, ...]:
    split = read_json_object(split_path)
    claimed = split.get("manifest_sha256")
    unsigned = {key: value for key, value in split.items() if key != "manifest_sha256"}
    try:
        public = tuple(split["partitions"]["public_test"]["task_ids"])
    except (KeyError, TypeError) as error:
        raise FrozenPackageEvaluationError("split has no Public-99 membership") from error
    if (
        claimed != _split_digest(unsigned)
        or claimed != verified.run_manifest.get("split_manifest_sha256")
        or split.get("target_sizes") != {"train": 80, "dev": 20, "public_test": 99}
        or split.get("actual_sizes") != {"train": 80, "dev": 20, "public_test": 99}
        or len(public) != 99
        or len(set(public)) != 99
        or any(type(task_id) is not str or not task_id for task_id in public)
    ):
        raise FrozenPackageEvaluationError("split does not bind exact Public-99 membership")
    return public


def _publish(output: Path, report: Mapping[str, object]) -> None:
    report_payload = dict(report)
    (output / "public_report.json").write_bytes(canonical_json_bytes(report_payload))
    with (output / "public_forecasts.jsonl").open("xb") as handle:
        per_task = cast(Mapping[str, Sequence[Mapping[str, object]]], report_payload["per_task"])
        for state_name, rows in per_task.items():
            for row in rows:
                handle.write(canonical_json_bytes({"state": state_name, **dict(row)}) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    (output / "evaluation_complete.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "public_regression_complete",
                "public_task_count": 99,
                "selection_used_public": False,
                "public_result_may_feed_evolution": False,
                "report_sha256": hashlib.sha256(
                    canonical_json_bytes(report_payload)
                ).hexdigest(),
            }
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evolution = Path(args.evolution_dir).resolve()
    output = Path(args.output_dir).resolve()
    if (output / "evaluation_complete.json").exists():
        raise FrozenPackageEvaluationError("Public output was already claimed")
    manifest = _read_canonical_json(evolution / "run_manifest.json", "run manifest")
    resources: _RuntimeResources | None = None
    evaluator: object | None = None
    try:
        runtime, resources = _runtime_authority(args, manifest)
        verified = verify_frozen_package_run(evolution, args.final_bundle, runtime)
        _claim_output(
            output,
            evolution,
            resume_unscored_start=args.resume_unscored_start,
        )
        public_ids = _public_membership(Path(args.split_file), verified)
        tasks = load_context_tasks_by_ids(args.tasks_file, public_ids)
        if tuple(task.numeric.task_id for task in tasks) != public_ids:
            raise FrozenPackageEvaluationError("loaded tasks do not match exact Public-99 membership")
        evaluator = _public_evaluator(args, verified, resources)
        report = score_frozen_states(tasks, build_attribution_states(verified), evaluator)
        _publish(output, report)
        return 0
    finally:
        if evaluator is not None and callable(getattr(evaluator, "close", None)):
            evaluator.close()
        if resources is not None:
            resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
