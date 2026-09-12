"""Root-owned, resumable orchestration for bounded real V2 evolution."""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from common.payload import strict_json_loads

from ..budget import BudgetLedger, BudgetPlan, ResourceUse
from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from ..store import V2RunStore, write_once_json
from .contracts import (
    PROFILE_SCHEDULES,
    RealEvolutionCheckpointV2,
    RealEvolutionManifestV2,
    RealRunResultV2,
    RealStageRecordV2,
)


_STAGES = ("p2", "p3", "p4", "p5")
_PHASES = {
    "p2": ("P2_RUNNING", "P2_SEALED"),
    "p3": ("P3_RUNNING", "P3_SEALED"),
    "p4": ("P4_RUNNING", "P4_SEALED"),
    "p5": ("P5_RUNNING", "P5_SEALED"),
}


class RealRunnerError(ValueError):
    """Raised when a root real-run artifact cannot be safely adopted."""


class _RealStageBudgetExhausted(RuntimeError):
    """Internal signal that bounded preparation consumed the child grant."""


def _json_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if hasattr(value, "to_payload"):
        value = value.to_payload()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    # Canonical serialization is the shared strict JSON validator.
    raw = canonical_v2_bytes(value)
    copied = strict_json_loads(raw.decode("utf-8"), context=field)
    assert isinstance(copied, dict)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class SealedStageV2:
    """Validated child boundary accepted by the root state machine."""

    completion_sha256: str
    handoff_payload: Mapping[str, object]
    summary: Mapping[str, object]
    public_test_accessed: bool

    def __post_init__(self) -> None:
        require_sha256(self.completion_sha256, "completion_sha256")
        object.__setattr__(
            self, "handoff_payload", _json_mapping(self.handoff_payload, field="handoff_payload")
        )
        object.__setattr__(self, "summary", _json_mapping(self.summary, field="summary"))
        if type(self.public_test_accessed) is not bool:
            raise ValueError("public_test_accessed must be a boolean")


@dataclass(frozen=True, slots=True)
class RealStageContextV2:
    """Narrow Host capability supplied to one child bridge."""

    stage: str
    output_dir: Path
    grant_seconds: int
    manifest: RealEvolutionManifestV2
    manifest_sha256: str
    model_binding_sha256: str
    handoffs: Mapping[str, str]
    deadline_monotonic: float | None = None
    monotonic: Callable[[], float] = time.monotonic
    read_only: bool = False

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("stage must be p2, p3, p4, or p5")
        if type(self.grant_seconds) is not int or self.grant_seconds <= 0:
            raise ValueError("grant_seconds must be positive")
        require_sha256(self.manifest_sha256, "manifest_sha256")
        require_sha256(self.model_binding_sha256, "model_binding_sha256")
        for stage, digest in self.handoffs.items():
            if stage not in _STAGES:
                raise ValueError("handoffs must use real stage names")
            require_sha256(digest, f"handoffs.{stage}")
        object.__setattr__(self, "handoffs", MappingProxyType(dict(self.handoffs)))
        if self.deadline_monotonic is not None and (
            type(self.deadline_monotonic) not in (int, float)
            or isinstance(self.deadline_monotonic, bool)
            or not math.isfinite(float(self.deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be a finite number or None")
        if not callable(self.monotonic):
            raise ValueError("monotonic must be callable")
        if type(self.read_only) is not bool:
            raise ValueError("read_only must be a boolean")

    def remaining_seconds(self) -> int:
        if self.deadline_monotonic is None:
            return self.grant_seconds
        now = float(self.monotonic())
        if not math.isfinite(now):
            raise RealRunnerError("stage monotonic clock is invalid")
        return max(0, math.floor(float(self.deadline_monotonic) - now))


@dataclass(frozen=True, slots=True)
class RealStagePorts:
    """Typed seams for the four existing project runners and their seals."""

    run_p2: Callable[[RealStageContextV2], object]
    seal_p2: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p3: Callable[[RealStageContextV2], object]
    seal_p3: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p4: Callable[[RealStageContextV2], object]
    seal_p4: Callable[[RealStageContextV2, object], SealedStageV2]
    run_p5: Callable[[RealStageContextV2], object]
    seal_p5: Callable[[RealStageContextV2, object], SealedStageV2]
    begin_finalization: Callable[[], None] | None = None
    requires_stage_wrappers: bool = False

    def __post_init__(self) -> None:
        for name in (
            "run_p2", "seal_p2", "run_p3", "seal_p3", "run_p4", "seal_p4", "run_p5", "seal_p5",
        ):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")
        if self.begin_finalization is not None and not callable(self.begin_finalization):
            raise ValueError("begin_finalization must be callable or None")
        if type(self.requires_stage_wrappers) is not bool:
            raise ValueError("requires_stage_wrappers must be a boolean")


@dataclass(frozen=True, slots=True)
class PreparedRealP2InputsV2:
    """Closed current-epoch P2 inputs derived from the admitted Champion."""

    config_payload: Mapping[str, object]
    seed_payload: Mapping[str, object]
    task_manifest_payload: Mapping[str, object]
    input_sha256s: Mapping[str, str]
    evidence_path: Path
    dictionary: object


def _context_task_payload(task: object) -> dict[str, object]:
    numeric = getattr(task, "numeric")
    return {
        "task_id": numeric.task_id,
        "entity_name": numeric.entity_name,
        "history_values": list(numeric.history_values),
        "future_values": list(numeric.future_values),
        "prediction_length": numeric.prediction_length,
        "frequency": numeric.frequency,
        "seasonal_period": numeric.seasonal_period,
        "target_name": getattr(task, "target_name"),
        "target_description": getattr(task, "target_description"),
        "history_timestamps": list(getattr(task, "history_timestamps")),
        "future_timestamps": list(getattr(task, "future_timestamps")),
        "documents": [
            {"document_id": item.document_id, "content": item.content}
            for item in getattr(task, "documents")
        ],
        "gt_evidence": list(getattr(task, "gt_evidence")),
    }


def _semantic_dictionary(screening: object):
    """Project the admitted rich screening policy into the complete Dictionary."""
    from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry

    entries = []
    for item in screening.entries:
        clauses = item.applicability.any_of
        applicability = clauses[0].reason_codes() if clauses else ()
        entries.append(
            FilterEntry(
                item.name,
                item.family,
                item.status,
                tuple(applicability),
                item.reason,
            )
        )
    return FilterDictionary(tuple(entries))


def _dictionary_payload(dictionary: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entries": [asdict(entry) for entry in dictionary.entries],
    }


def _dictionary_from_payload(payload: Mapping[str, object]):
    from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry

    if set(payload) != {"schema_version", "entries"} or payload["schema_version"] != 1:
        raise RealRunnerError("prepared real Dictionary has an invalid schema")
    rows = payload["entries"]
    if type(rows) is not list:
        raise RealRunnerError("prepared real Dictionary entries must be a list")
    try:
        return FilterDictionary(
            tuple(
                FilterEntry(
                    row["name"], row["family"], row["status"],
                    tuple(row["applicability"]), row["reason"],
                )
                for row in rows
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RealRunnerError("prepared real Dictionary is invalid") from error


def _initial_prior_rows(
    host: object,
    candidates: tuple[tuple[str, str], ...],
    *,
    check_deadline: Callable[[], None] | None = None,
):
    """Read full-horizon Train rows from the shared cache without hindcasting."""
    from numerical_agent.evolution.execution import Task as RuntimeTask
    from numerical_agent.evolution.forecast_store import CacheMissError
    from numerical_agent.evolution.screening import profile_task
    from numerical_agent.evolution.task_local_evolution import TaskLocalTaskRow

    rows = []
    for context_task in host.train_tasks:
        task = context_task.numeric
        runtime_task = RuntimeTask(
            task.task_id,
            tuple(float(value) for value in task.history_values),
            task.prediction_length,
            task.frequency,
            tuple(float(value) for value in task.future_values),
        )
        profile = profile_task(runtime_task)
        for name, family in candidates:
            if check_deadline is not None:
                check_deadline()
            forecast = None
            failure = None
            try:
                forecast = tuple(
                    float(value)
                    for value in host.forecast_store.forecast(
                        name,
                        runtime_task.history,
                        runtime_task.horizon,
                        runtime_task.frequency,
                    )
                )
            except (CacheMissError, host.forecast_store.not_applicable) as error:
                failure = f"{type(error).__name__}: {error}"[:10000]
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=runtime_task.history,
                    truth=runtime_task.future,
                    forecast=forecast,
                    diagnostic=None,
                    split="train",
                    failure_reason=failure,
                )
            )
    return tuple(rows)


def _derived_numerical_config(
    *, grant_seconds: int, runtime_fingerprints: Mapping[str, str]
) -> dict[str, object]:
    code_root = Path(__file__).resolve().parents[3]
    payload = _read_canonical(code_root / "configs/evolution_v2/numerical_qd/pilot.json")
    payload["runtime_fingerprints"] = dict(runtime_fingerprints)
    budget = dict(payload["budget"])
    budget["hard_limit_seconds"] = grant_seconds
    ceilings = dict(budget["ceilings"])
    ceilings["wall_seconds"] = float(grant_seconds)
    budget["ceilings"] = ceilings
    payload["budget"] = budget
    from ..numerical_qd.config import NumericalQDConfigV2

    return NumericalQDConfigV2.from_payload(payload).to_payload()


def _load_prepared_real_p2(
    prepared: Path, *, manifest: RealEvolutionManifestV2
) -> PreparedRealP2InputsV2:
    from numerical_agent.evolution.task_shortlist import _dictionary_hash
    from numerical_agent.run_task_local_ensemble_evolution import (
        load_task_local_evidence_bundle,
    )

    seal = _read_canonical(prepared / "prepared_inputs.json")
    if set(seal) != {
        "schema_version", "kind", "manifest_sha256", "champion_sha256",
        "dictionary_sha256", "config_sha256", "seed_supply_sha256",
        "task_manifest_sha256", "task_local_evidence_sha256",
    } or seal["schema_version"] != 1 or seal["kind"] != "real_p2_prepared_inputs":
        raise RealRunnerError("prepared P2 seal has an invalid schema")
    if seal["manifest_sha256"] != manifest.fingerprint():
        raise RealRunnerError("prepared P2 manifest identity mismatch")
    champion_row = next(
        (row for row in manifest.files if row.role == "numerical_seed"), None
    )
    if champion_row is None or seal["champion_sha256"] != champion_row.sha256:
        raise RealRunnerError("prepared P2 Champion identity mismatch")
    config = _read_canonical(prepared / "numerical_config.json")
    seed = _read_canonical(prepared / "seed_supply.json")
    tasks = _read_canonical(prepared / "task_manifest.json")
    dictionary = _dictionary_from_payload(_read_canonical(prepared / "dictionary.json"))
    identities = {
        "config": fingerprint_payload(config),
        "seed_supply": fingerprint_payload(seed),
        "task_manifest": fingerprint_payload(tasks),
        "champion_release": seal["champion_sha256"],
        "dictionary": _dictionary_hash(dictionary),
        "task_local_evidence": seal["task_local_evidence_sha256"],
    }
    expected = {
        "config_sha256": identities["config"],
        "seed_supply_sha256": identities["seed_supply"],
        "task_manifest_sha256": identities["task_manifest"],
        "dictionary_sha256": identities["dictionary"],
    }
    if any(seal[name] != value for name, value in expected.items()):
        raise RealRunnerError("prepared P2 content identity mismatch")
    evidence = load_task_local_evidence_bundle(
        prepared, dictionary_sha256=identities["dictionary"]
    )
    if fingerprint_payload(dict(evidence.index)) != seal["task_local_evidence_sha256"]:
        raise RealRunnerError("prepared P2 task-local evidence identity mismatch")
    task_ids = {
        row["task_id"]
        for split in ("train", "dev")
        for row in tasks[split]
    }
    if len(task_ids) != 100 or set(evidence.by_task) != task_ids:
        raise RealRunnerError("prepared P2 evidence does not close Train80/Dev20")
    return PreparedRealP2InputsV2(
        MappingProxyType(config), MappingProxyType(seed), MappingProxyType(tasks),
        MappingProxyType(identities), prepared, dictionary,
    )


def prepare_real_p2_inputs(
    host: object,
    *,
    manifest: RealEvolutionManifestV2,
    repo_root: Path,
    output_dir: Path,
    grant_seconds: int,
    remaining_seconds: Callable[[], int] | None = None,
) -> PreparedRealP2InputsV2:
    """Build and seal the schema-2 Dictionary/shortlist P2 authority once."""
    from evolving_loop.run_package_coevolution import _initial_supply_release
    from numerical_agent.evolution.champion import parse_champion_release
    from numerical_agent.evolution.module import read_module
    from numerical_agent.evolution.portfolio import read_policy_file
    from numerical_agent.evolution.task_local_evolution import (
        build_group_fold_manifest,
        fit_oof_shortlist_priors,
    )
    from numerical_agent.evolution.task_shortlist import (
        TaskShortlistPolicyV1,
        _dictionary_hash,
    )
    from numerical_agent.run_champion_evolution import _load_screening_policy
    from numerical_agent.run_task_local_ensemble_evolution import (
        _CONFIDENCE_HINDCAST_CONFIG,
        _bind_shortlist_index,
        _reviewed_candidates,
        _shortlist_index,
        _shortlist_rows_for_tasks,
        _v3_oof_rows,
        load_task_local_evidence_bundle,
    )

    prepared = Path(output_dir)
    if (prepared / "prepared_inputs.json").is_file():
        return _load_prepared_real_p2(prepared, manifest=manifest)
    if prepared.exists() and any(prepared.iterdir()):
        raise RealRunnerError("unsealed prepared P2 inputs cannot be resumed")
    prepared.mkdir(parents=True, exist_ok=True)

    def check_deadline() -> None:
        if remaining_seconds is None:
            return
        remaining = remaining_seconds()
        if type(remaining) is not int or remaining <= 0:
            raise _RealStageBudgetExhausted(
                "P2 preparation consumed its bounded grant"
            )

    file_rows = {row.role: row for row in manifest.files}
    champion_row = file_rows.get("numerical_seed")
    if champion_row is None:
        raise RealRunnerError("real P2 requires an admitted ChampionRelease")
    champion_payload = _read_sha_bound_json(
        Path(repo_root) / champion_row.relative_path, champion_row.sha256
    )
    champion = parse_champion_release(champion_payload)
    source_repo = Path(host.source_repo)
    module = read_module(source_repo / "methods.py")
    portfolio = read_policy_file(source_repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(source_repo / "dictionary.py")
    candidates = _reviewed_candidates(module, portfolio, screening)
    dictionary = _semantic_dictionary(screening)
    dictionary_sha = _dictionary_hash(dictionary)
    runtime_fingerprints = {
        row.role: row.identity_sha256 for row in manifest.runtime_locations
    } | {"real_host": host.resource_reporter_sha256}
    source_fingerprints = dict(champion.source_hashes)
    source_fingerprints["dictionary"] = dictionary_sha
    supply = _initial_supply_release(
        champion,
        candidates,
        schema_version=2,
        source_fingerprints=source_fingerprints,
        runtime_fingerprints=runtime_fingerprints,
        atlas=None,
    )

    train_numeric = tuple(task.numeric for task in host.train_tasks)
    dev_numeric = tuple(task.numeric for task in host.dev_tasks)
    folds = build_group_fold_manifest(train_numeric, seed=20260903)
    check_deadline()
    prior_rows = _initial_prior_rows(
        host, candidates, check_deadline=check_deadline
    )
    check_deadline()
    fold_priors, priors = fit_oof_shortlist_priors(
        prior_rows, folds, candidate_names=tuple(name for name, _ in candidates)
    )
    policy = TaskShortlistPolicyV1()
    families = dict(candidates)
    train_rows, train_shortlists = _v3_oof_rows(
        host.forecast_store,
        train_numeric,
        manifest=folds,
        dictionary=dictionary,
        screening=screening,
        families=families,
        anchor_name=champion.policy.recipe.fallback_parent,
        policy=policy,
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
        output=prepared,
        fold_priors=dict(fold_priors),
        check_deadline=check_deadline,
    )
    check_deadline()
    dev_rows, dev_shortlists = _shortlist_rows_for_tasks(
        host.forecast_store,
        dev_numeric,
        dictionary=dictionary,
        screening=screening,
        families=families,
        priors=priors,
        anchor_name=champion.policy.recipe.fallback_parent,
        policy=policy,
        split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
        output=prepared,
        check_deadline=check_deadline,
    )
    check_deadline()
    shortlists = dict(train_shortlists)
    shortlists.update(
        {
            task.task_id: shortlist
            for task, shortlist in zip(dev_numeric, dev_shortlists, strict=True)
        }
    )
    index = _shortlist_index(
        train_numeric + dev_numeric,
        shortlists,
        train_rows + dev_rows,
        policy=policy,
    )
    evidence_run_manifest = {
        "schema_version": 1,
        "kind": "real_p2_task_local_evidence",
        "manifest_sha256": manifest.fingerprint(),
        "champion_sha256": champion_row.sha256,
        "dictionary_sha256": dictionary_sha,
        "candidate_priors_sha256": fingerprint_payload(
            {"priors": [item.to_payload() for item in priors]}
        ),
        "public_test_accessed": False,
    }
    _bind_shortlist_index(
        prepared, evidence_run_manifest, index, rows=train_rows + dev_rows
    )
    evidence = load_task_local_evidence_bundle(
        prepared, dictionary_sha256=dictionary_sha
    )
    expected_ids = {task.numeric.task_id for task in host.tasks}
    if set(evidence.by_task) != expected_ids or len(expected_ids) != 100:
        raise RealRunnerError("prepared P2 evidence does not close Train80/Dev20")

    task_manifest = {
        "schema_version": 1,
        "fold_manifest": folds.to_payload(),
        "train": [_context_task_payload(task) for task in host.train_tasks],
        "dev": [_context_task_payload(task) for task in host.dev_tasks],
    }
    remaining = grant_seconds if remaining_seconds is None else remaining_seconds()
    if type(remaining) is not int or remaining <= 0:
        raise _RealStageBudgetExhausted("P2 preparation consumed its bounded grant")
    config = _derived_numerical_config(
        grant_seconds=remaining,
        runtime_fingerprints=runtime_fingerprints,
    )
    dictionary_payload = _dictionary_payload(dictionary)
    write_once_json(prepared / "dictionary.json", dictionary_payload)
    write_once_json(prepared / "numerical_config.json", config)
    write_once_json(prepared / "seed_supply.json", supply.to_payload())
    write_once_json(prepared / "task_manifest.json", task_manifest)
    evidence_sha = fingerprint_payload(dict(evidence.index))
    seal = {
        "schema_version": 1,
        "kind": "real_p2_prepared_inputs",
        "manifest_sha256": manifest.fingerprint(),
        "champion_sha256": champion_row.sha256,
        "dictionary_sha256": dictionary_sha,
        "config_sha256": fingerprint_payload(config),
        "seed_supply_sha256": fingerprint_payload(supply.to_payload()),
        "task_manifest_sha256": fingerprint_payload(task_manifest),
        "task_local_evidence_sha256": evidence_sha,
    }
    write_once_json(prepared / "prepared_inputs.json", seal)
    return _load_prepared_real_p2(prepared, manifest=manifest)


def _stage_wrapper(
    context: RealStageContextV2,
    *,
    status: str,
    native_completion_sha256: str | None,
    bindings: Mapping[str, object],
    public_test_accessed: bool,
) -> tuple[dict[str, object], str]:
    payload = {
        "schema_version": 1,
        "stage": context.stage,
        "status": status,
        "native_completion_sha256": native_completion_sha256,
        "bindings": dict(bindings),
        "public_test_accessed": public_test_accessed,
    }
    path = context.output_dir / "root_stage_completion.json"
    if context.read_only:
        if _read_canonical(path) != payload:
            raise RealRunnerError("root stage wrapper differs from native closure")
    else:
        write_once_json(path, payload)
    return payload, fingerprint_payload(payload)


def _derived_cooperative_config(context: RealStageContextV2, host: object) -> dict[str, object]:
    code_root = Path(__file__).resolve().parents[3]
    payload = _read_canonical(
        code_root / "configs/evolution_v2/cooperative/real-luna-medium.json"
    )
    control = dict(payload["control"])
    control["profile"] = "pilot"
    control["hard_limit_seconds"] = context.grant_seconds
    control["runtime_fingerprints"] = {"real_host": host.resource_reporter_sha256}
    payload["control"] = control
    ceilings = dict(payload["resource_ceilings"])
    ceilings["wall_seconds"] = float(context.grant_seconds)
    payload["resource_ceilings"] = ceilings
    from ..cooperative import CooperativeConfigV2

    return CooperativeConfigV2.from_payload(payload).to_payload()


def build_real_stage_ports(
    host: object,
    *,
    manifest: RealEvolutionManifestV2,
    repo_root: Path,
) -> RealStagePorts:
    """Assemble authenticated production P2→P5 child runs and seals."""
    authority = Path(repo_root).resolve()

    def bind_deadline(context: RealStageContextV2) -> None:
        binder = getattr(getattr(host, "llm_client", None), "bind_deadline", None)
        if callable(binder) and context.deadline_monotonic is not None:
            binder(context.deadline_monotonic, monotonic=context.monotonic)

    def run_p2(context: RealStageContextV2) -> object:
        from .bridges import run_real_numerical

        bind_deadline(context)
        try:
            prepared = prepare_real_p2_inputs(
                host,
                manifest=manifest,
                repo_root=authority,
                output_dir=context.output_dir.parent / "prepared/p2",
                grant_seconds=context.grant_seconds,
                remaining_seconds=context.remaining_seconds,
            )
        except _RealStageBudgetExhausted:
            return {
                "schema_version": 1,
                "status": "p2_preparation_budget_exhausted",
                "public_test_accessed": False,
            }
        return run_real_numerical(
            context,
            host,
            config_payload=prepared.config_payload,
            seed_payload=prepared.seed_payload,
            task_manifest_payload=prepared.task_manifest_payload,
            input_sha256s=prepared.input_sha256s,
            task_local_evidence_path=prepared.evidence_path,
            task_local_dictionary=prepared.dictionary,
        )

    def seal_p2(context: RealStageContextV2, _result: object) -> SealedStageV2:
        from ..numerical_qd.persistence import NumericalQDRunStore

        if (
            isinstance(_result, Mapping)
            and _result.get("status") == "p2_preparation_budget_exhausted"
        ):
            wrapper, wrapper_sha = _stage_wrapper(
                context,
                status="incomplete",
                native_completion_sha256=None,
                bindings={"reason": "p2_preparation_budget_exhausted"},
                public_test_accessed=False,
            )
            return SealedStageV2(
                wrapper_sha,
                {"reason": "p2_preparation_budget_exhausted"},
                {"status": "incomplete", "root_completion": wrapper},
                False,
            )

        completion = _read_canonical(context.output_dir / "evaluation_complete.json")
        if completion.get("status") != "numerical_qd_complete":
            raise RealRunnerError("P2 did not produce a completed Numerical QD authority")
        summary = completion.get("summary")
        if not isinstance(summary, Mapping) or type(summary.get("public_test_accessed")) is not bool:
            raise RealRunnerError("P2 completion lacks Public access evidence")
        pair, pair_sha = NumericalQDRunStore(context.output_dir).load_active_frozen_pair(
            tasks=host.tasks
        )
        prepared = _load_prepared_real_p2(
            context.output_dir.parent / "prepared/p2", manifest=manifest
        )
        bindings = {
            "pair_sha256": pair_sha,
            "release_sha256": pair.release.fingerprint,
            "registry_sha256": pair.registry.fingerprint,
            "envelope_sha256": pair.envelope.fingerprint(),
            "champion_sha256": prepared.input_sha256s["champion_release"],
            "config_sha256": prepared.input_sha256s["config"],
            "task_manifest_sha256": prepared.input_sha256s["task_manifest"],
            "task_local_evidence_sha256": prepared.input_sha256s["task_local_evidence"],
        }
        _, wrapper_sha = _stage_wrapper(
            context,
            status="complete",
            native_completion_sha256=fingerprint_payload(completion),
            bindings=bindings,
            public_test_accessed=summary["public_test_accessed"],
        )
        return SealedStageV2(
            wrapper_sha, bindings, {"status": "complete", **bindings},
            summary["public_test_accessed"],
        )

    def _p2_pair(context: RealStageContextV2):
        from ..numerical_qd.persistence import NumericalQDRunStore

        return NumericalQDRunStore(context.output_dir.parent / "p2").load_active_frozen_pair(
            tasks=host.tasks
        )[0]

    def run_p3(context: RealStageContextV2) -> object:
        from .bridges import run_real_cooperative

        bind_deadline(context)
        return run_real_cooperative(
            p2=_p2_pair(context),
            host=host,
            config_payload=_derived_cooperative_config(context, host),
            output_dir=context.output_dir,
        )

    def _p3_closure(context: RealStageContextV2):
        from .bridges import load_sealed_bundle_closure

        return load_sealed_bundle_closure(
            context.output_dir.parent / "p3", tasks=tuple(host.tasks), host=host
        )

    def seal_p3(context: RealStageContextV2, _result: object) -> SealedStageV2:
        completion = _read_canonical(context.output_dir / "evaluation_complete.json")
        closure = _p3_closure(context)
        public = completion.get("public_test_accessed")
        if type(public) is not bool:
            raise RealRunnerError("P3 completion lacks Public access evidence")
        config_sha = fingerprint_payload(_derived_cooperative_config(context, host))
        bindings = {
            "active_bundle_sha256": closure.active_bundle.fingerprint(),
            "completion_sha256": closure.completion_sha256,
            "checkpoint_sha256": closure.checkpoint_sha256,
            "proposal_space_sha256": closure.proposal_space_sha256,
            "config_sha256": config_sha,
            "p2_handoff_sha256": context.handoffs["p2"],
            "p5_handoff_available": closure.p5_handoff_available,
            "p5_handoff_reason": closure.p5_handoff_reason,
        }
        _, wrapper_sha = _stage_wrapper(
            context,
            status="complete",
            native_completion_sha256=fingerprint_payload(completion),
            bindings=bindings,
            public_test_accessed=public,
        )
        return SealedStageV2(
            wrapper_sha, bindings, {"status": "complete", **bindings}, public
        )

    def _source_seed(context: RealStageContextV2, closure: object):
        from ..source import SourceVariantV2

        row = next(item for item in manifest.files if item.role == "source_seed")
        payload = _read_canonical(Path(__file__).resolve().parents[3] / row.relative_path)
        admitted = SourceVariantV2.from_payload(payload)
        rebound = SourceVariantV2.seed(
            admitted.source,
            closure.active_bundle.protocol_fingerprint,
            host.resource_reporter_sha256,
        )
        if admitted != rebound:
            raise RealRunnerError(
                "admitted Source seed provenance does not match P3/Host commitments"
            )
        return rebound, row.sha256

    def run_p4(context: RealStageContextV2) -> object:
        from ..budget import ResourceUse
        from ..source import SourceConfigV2, build_source_case_from_p3, run_source_evolution

        bind_deadline(context)
        closure = _p3_closure(context)
        seed, input_digest = _source_seed(context, closure)
        case = build_source_case_from_p3(
            closure,
            host,
            source_seed=seed,
            input_digest=input_digest,
            empty_skill_path=context.output_dir.parent / "prepared/p4-empty-skills.json",
        )
        config = SourceConfigV2(
            1, 0, 2, context.grant_seconds, 2,
            closure.active_bundle.protocol_fingerprint,
            host.resource_reporter_sha256,
            ResourceUse(
                wall_seconds=float(context.grant_seconds),
                task_executions=1000,
                llm_calls=100,
                input_tokens=1_000_000,
                output_tokens=100_000,
                subprocesses=1000,
                artifact_bytes=1_000_000_000,
            ),
        )
        resume = (context.output_dir / "checkpoint.json").is_file()
        return run_source_evolution(context.output_dir, config, case, resume=resume)

    def seal_p4(context: RealStageContextV2, result: object) -> SealedStageV2:
        from ..source import SourceRunResultV2
        from ..source.archive import SourceArchiveV2
        from ..source.authority import SourceAuthorityV2

        payload = result.to_payload() if hasattr(result, "to_payload") else result
        if not isinstance(payload, Mapping):
            raise RealRunnerError("P4 returned a malformed result")
        typed = SourceRunResultV2.from_payload(payload)
        closure = _p3_closure(context)
        seed, input_digest = _source_seed(context, closure)
        archive = SourceArchiveV2(context.output_dir / "source_archive")
        authority = SourceAuthorityV2(context.output_dir / "authority", seed)
        if (
            archive.snapshot_sha256() != typed.archive_snapshot_sha256
            or authority.active_source().fingerprint() != typed.active_source_sha256
        ):
            raise RealRunnerError("P4 result does not bind its Source authority")
        complete = typed.status == "source_evolution_complete"
        native_sha = None
        if complete:
            native = _read_canonical(context.output_dir / "evaluation_complete.json")
            if native != typed.to_payload():
                raise RealRunnerError("P4 native completion differs from its result")
            native_sha = fingerprint_payload(native)
        bindings = {
            "active_source_sha256": typed.active_source_sha256,
            "archive_snapshot_sha256": typed.archive_snapshot_sha256,
            "source_seed_input_sha256": input_digest,
            "p3_handoff_sha256": context.handoffs["p3"],
        }
        status = "complete" if complete else "incomplete"
        _, wrapper_sha = _stage_wrapper(
            context,
            status=status,
            native_completion_sha256=native_sha,
            bindings=bindings,
            public_test_accessed=typed.public_test_accessed,
        )
        return SealedStageV2(
            wrapper_sha, bindings, {"status": status, **bindings},
            typed.public_test_accessed,
        )

    def run_p5(context: RealStageContextV2) -> object:
        from ..protocol import build_protocol_case_from_p3

        bind_deadline(context)
        case = build_protocol_case_from_p3(
            _p3_closure(context), host, hard_limit_seconds=context.grant_seconds
        )
        return case.run(context.output_dir)

    def seal_p5(context: RealStageContextV2, result: object) -> SealedStageV2:
        if not isinstance(result, Mapping):
            raise RealRunnerError("P5 returned a malformed result")
        payload = dict(result)
        public = payload.get("public_test_accessed", False)
        if type(public) is not bool:
            raise RealRunnerError("P5 result lacks valid Public access evidence")
        native_sha = None
        status = "incomplete"
        bindings: dict[str, object] = {
            "p3_handoff_sha256": context.handoffs["p3"],
            "p4_handoff_sha256": context.handoffs["p4"],
        }
        if payload.get("status") == "protocol_evolution_complete":
            native = _read_canonical(context.output_dir / "completion.json")
            if native != payload:
                raise RealRunnerError("P5 native completion differs from its result")
            handoff = _read_canonical(context.output_dir / "frozen_protocol_handoff.json")
            handoff_sha = fingerprint_payload(handoff)
            if handoff_sha != native.get("frozen_handoff_sha256"):
                raise RealRunnerError("P5 frozen handoff identity mismatch")
            native_sha = fingerprint_payload(native)
            status = "complete"
            bindings.update(
                active_protocol_sha256=native["active_protocol_sha256"],
                active_release_sha256=native["active_release_sha256"],
                frozen_handoff_sha256=handoff_sha,
            )
        else:
            bindings["reason"] = (
                "p5_handoff_unavailable"
                if payload.get("status") == "p5_handoff_unavailable"
                else "protocol_evolution_incomplete"
            )
        _, wrapper_sha = _stage_wrapper(
            context,
            status=status,
            native_completion_sha256=native_sha,
            bindings=bindings,
            public_test_accessed=public,
        )
        return SealedStageV2(
            wrapper_sha, bindings, {"status": status, **bindings}, public
        )

    return RealStagePorts(
        run_p2=run_p2, seal_p2=seal_p2,
        run_p3=run_p3, seal_p3=seal_p3,
        run_p4=run_p4, seal_p4=seal_p4,
        run_p5=run_p5, seal_p5=seal_p5,
        requires_stage_wrappers=True,
    )


def _read_canonical(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except (OSError, UnicodeError, ValueError) as error:
        raise RealRunnerError(f"missing or invalid canonical artifact: {path}") from error
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise RealRunnerError(f"artifact must be canonical JSON: {path}")
    return value


def _read_sha_bound_json(path: Path, expected_sha256: str) -> dict[str, object]:
    """Read strict JSON whose admitted identity is its manifest-bound raw bytes."""
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise RealRunnerError(f"artifact identity mismatch: {path}")
        value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    except RealRunnerError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise RealRunnerError(f"missing or invalid admitted artifact: {path}") from error
    if type(value) is not dict:
        raise RealRunnerError(f"admitted artifact must be a JSON object: {path}")
    return value


def _plan(manifest: RealEvolutionManifestV2) -> BudgetPlan:
    schedule = PROFILE_SCHEDULES[manifest.profile]
    return BudgetPlan(
        schedule.total_seconds,
        float(schedule.finalization_reserve_seconds / schedule.total_seconds),
        ResourceUse(wall_seconds=float(schedule.total_seconds)),
    )


def _root_manifest(manifest: RealEvolutionManifestV2, plan: BudgetPlan) -> dict[str, object]:
    return {
        "system": "evolution_v2",
        "schema_version": 1,
        "kind": "real_root_run",
        "real_manifest": manifest.to_payload(),
        "real_manifest_sha256": manifest.fingerprint(),
        "model_binding_sha256": manifest.model.fingerprint(),
        "budget_plan_sha256": plan.fingerprint(),
    }


def _load_state(
    root: Path, manifest: RealEvolutionManifestV2, plan: BudgetPlan, monotonic: Callable[[], float]
) -> tuple[V2RunStore, BudgetLedger, RealEvolutionCheckpointV2 | None]:
    store = V2RunStore.create(root)
    expected_manifest = _root_manifest(manifest, plan)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if _read_canonical(manifest_path) != expected_manifest:
            raise RealRunnerError("root manifest does not match the supplied real manifest")
    else:
        store.write_run_manifest(expected_manifest)

    budget_path = root / "budget_plan.json"
    if budget_path.exists():
        if _read_canonical(budget_path) != plan.to_payload():
            raise RealRunnerError("root budget plan does not match the supplied profile")
    else:
        store.write_budget_plan(plan.to_payload())

    checkpoint_path = root / "checkpoint.json"
    if not checkpoint_path.exists():
        return store, BudgetLedger(plan, monotonic=monotonic), None
    checkpoint = RealEvolutionCheckpointV2.from_payload(_read_canonical(checkpoint_path))
    try:
        ledger = BudgetLedger.resume(plan, checkpoint.budget_checkpoint, monotonic=monotonic)
    except ValueError as error:
        raise RealRunnerError(f"invalid root budget checkpoint: {error}") from error
    return store, ledger, checkpoint


def _checkpoint(
    store: V2RunStore,
    ledger: BudgetLedger,
    *,
    phase: str,
    records: list[RealStageRecordV2],
    active_stage: str | None,
    carry_seconds: int,
    handoff_sha256s: Mapping[str, str],
    completion_sha256: str | None,
) -> RealEvolutionCheckpointV2:
    checkpoint = RealEvolutionCheckpointV2(
        phase,
        tuple(records),
        active_stage,
        carry_seconds,
        ledger.checkpoint(),
        dict(handoff_sha256s),
        completion_sha256,
    )
    store.write_checkpoint(checkpoint.to_payload())
    return checkpoint


def _result(
    status: str,
    manifest: RealEvolutionManifestV2,
    records: list[RealStageRecordV2],
    completion_sha256: str | None = None,
) -> RealRunResultV2:
    return RealRunResultV2(
        status, manifest.fingerprint(), manifest.model.fingerprint(), tuple(records), completion_sha256, False
    )


def _previous_progress_sha(root: Path) -> str | None:
    path = root / "progress.jsonl"
    if not path.exists():
        return None
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        return None
    raw = lines[-1]
    value = strict_json_loads(raw.decode("utf-8"), context=str(path))
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise RealRunnerError("progress rows must be canonical JSON")
    return fingerprint_payload(value)


def _append_progress_once(store: V2RunStore, root: Path, row: Mapping[str, object]) -> None:
    """Append one transition row, tolerating only an interrupted identical retry."""
    path = root / "progress.jsonl"
    expected = canonical_v2_bytes(row)
    if path.exists():
        for raw in path.read_bytes().splitlines(keepends=True):
            value = strict_json_loads(raw.decode("utf-8"), context=str(path))
            if type(value) is not dict or canonical_v2_bytes(value) != raw:
                raise RealRunnerError("progress rows must be canonical JSON")
            if raw == expected:
                return
    store.append_progress(row)


def _stage_status(sealed: SealedStageV2) -> str:
    value = sealed.summary.get("status", "complete")
    if value not in {"complete", "incomplete", "failed"}:
        raise RealRunnerError("sealed stage summary status must be complete, incomplete, or failed")
    return value


def _stage_output(
    root: Path,
    stage: str,
    completion_sha256: str,
    sealed_public_test_accessed: bool,
) -> None:
    wrapper = root / stage / "root_stage_completion.json"
    completion = _read_canonical(
        wrapper if wrapper.is_file() else root / stage / "evaluation_complete.json"
    )
    if fingerprint_payload(completion) != completion_sha256:
        raise RealRunnerError(f"{stage} completion digest does not match its sealed boundary")
    completion_public = completion.get("public_test_accessed")
    if type(completion_public) is not bool:
        raise RealRunnerError(f"{stage} completion public access evidence is missing")
    if completion_public != sealed_public_test_accessed:
        raise RealRunnerError(f"{stage} completion public access evidence must match its seal")


def _sealed_handoff_payload(stage: str, sealed: SealedStageV2) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": stage,
        "completion_sha256": sealed.completion_sha256,
        "handoff_payload": dict(sealed.handoff_payload),
        "summary": dict(sealed.summary),
        "public_test_accessed": sealed.public_test_accessed,
    }


def _require_read_only_stage_closure(root: Path, stage: str) -> None:
    required_files = {
        "p2": ("root_stage_completion.json", "evaluation_complete.json"),
        "p3": ("root_stage_completion.json", "evaluation_complete.json"),
        "p4": (
            "root_stage_completion.json",
            "evaluation_complete.json",
            "authority/active_source.json",
            "source_archive/events.jsonl",
        ),
        "p5": (
            "root_stage_completion.json",
            "completion.json",
            "frozen_protocol_handoff.json",
        ),
    }[stage]
    for relative in required_files:
        if not (root / stage / relative).is_file():
            label = (
                "root stage wrapper"
                if relative == "root_stage_completion.json"
                else "native stage artifact"
            )
            raise RealRunnerError(f"missing {label} for {stage}: {relative}")
    if stage == "p4":
        for relative in (
            "authority/sealed",
            "source_archive",
            "source_archive/objects",
        ):
            if not (root / stage / relative).is_dir():
                raise RealRunnerError(
                    f"missing native stage artifact for p4: {relative}"
                )


def _validate_sealed_records(
    root: Path,
    records: list[RealStageRecordV2],
    handoffs: Mapping[str, str],
    manifest: RealEvolutionManifestV2,
    ports: RealStagePorts,
) -> bool:
    if tuple(record.stage for record in records) != _STAGES[: len(records)]:
        raise RealRunnerError("root stage records are not a sealed prefix")
    public_accessed = False
    authenticated_handoffs: dict[str, str] = {}
    for record in records:
        if record.status != "complete" or record.completion_sha256 is None:
            raise RealRunnerError("only complete sealed records can be revalidated")
        claimed = handoffs.get(record.stage)
        if claimed is None:
            raise RealRunnerError(f"missing root handoff for {record.stage}")
        handoff = _read_canonical(root / "handoffs" / f"{record.stage}.json")
        if fingerprint_payload(handoff) != claimed:
            raise RealRunnerError(f"{record.stage} root handoff digest mismatch")
        if handoff.get("stage") != record.stage or handoff.get("completion_sha256") != record.completion_sha256:
            raise RealRunnerError(f"{record.stage} root handoff does not bind its completion")
        public = handoff.get("public_test_accessed")
        if type(public) is not bool:
            raise RealRunnerError(f"{record.stage} public access evidence is missing")
        if ports.requires_stage_wrappers:
            _require_read_only_stage_closure(root, record.stage)
        result = _read_run_result(root, record.stage)
        if result is None:
            raise RealRunnerError(f"{record.stage} sealed result is missing")
        context = _context(
            root,
            record.stage,
            record.grant_seconds,
            manifest,
            authenticated_handoffs,
            read_only=True,
        )
        sealed = getattr(ports, f"seal_{record.stage}")(context, result)
        if not isinstance(sealed, SealedStageV2) or _stage_status(sealed) != "complete":
            raise RealRunnerError(f"{record.stage} native closure is not complete")
        if _sealed_handoff_payload(record.stage, sealed) != handoff:
            raise RealRunnerError(f"{record.stage} native closure differs from root handoff")
        _stage_output(root, record.stage, record.completion_sha256, public)
        authenticated_handoffs[record.stage] = claimed
        public_accessed = public_accessed or public
    return public_accessed


def _persist_run_result(root: Path, stage: str, result: object) -> None:
    try:
        payload = _json_mapping(result, field="stage result")
    except (TypeError, ValueError):
        return
    write_once_json(root / stage / "root_run_result.json", payload)


def _read_run_result(root: Path, stage: str) -> object | None:
    path = root / stage / "root_run_result.json"
    return _read_canonical(path) if path.exists() else None


def _context(
    root: Path,
    stage: str,
    grant_seconds: int,
    manifest: RealEvolutionManifestV2,
    handoffs: Mapping[str, str],
    *,
    deadline_monotonic: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    read_only: bool = False,
) -> RealStageContextV2:
    child_root = root / stage
    if read_only:
        if not child_root.is_dir():
            raise RealRunnerError(f"missing sealed stage directory: {stage}")
    else:
        child_root.mkdir(parents=True, exist_ok=True)
    return RealStageContextV2(
        stage,
        child_root,
        grant_seconds,
        manifest,
        manifest.fingerprint(),
        manifest.model.fingerprint(),
        handoffs,
        deadline_monotonic,
        monotonic,
        read_only,
    )


def _grant(
    manifest: RealEvolutionManifestV2, ledger: BudgetLedger, stage: str, carry_seconds: int
) -> int:
    schedule = PROFILE_SCHEDULES[manifest.profile]
    index = _STAGES.index(stage)
    base = schedule.allocations[stage]
    later_base = sum(schedule.allocations[name] for name in _STAGES[index + 1 :])
    search_remaining = (
        schedule.total_seconds - schedule.finalization_reserve_seconds - ledger.charged_use.wall_seconds
    )
    return int(min(base + carry_seconds, search_remaining - later_base))


def _terminal_active_failure(
    *,
    root: Path,
    store: V2RunStore,
    ledger: BudgetLedger,
    reservation: object,
    stage: str,
    grant_seconds: int,
    records: list[RealStageRecordV2],
    handoffs: Mapping[str, str],
) -> None:
    """Conservatively close a returned-but-unverifiable child as failed."""
    ledger.close_stage(reservation, ResourceUse(wall_seconds=float(grant_seconds)))
    records.append(
        RealStageRecordV2(
            stage,
            grant_seconds,
            grant_seconds,
            "failed",
            None,
            _previous_progress_sha(root),
        )
    )
    _checkpoint(
        store,
        ledger,
        phase="FAILED",
        records=records,
        active_stage=None,
        carry_seconds=0,
        handoff_sha256s=handoffs,
        completion_sha256=None,
    )


def _summary(
    manifest: RealEvolutionManifestV2,
    plan: BudgetPlan,
    records: list[RealStageRecordV2],
    handoffs: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": manifest.fingerprint(),
        "model_binding_sha256": manifest.model.fingerprint(),
        "budget_plan_sha256": plan.fingerprint(),
        "stage_records": [record.to_payload() for record in records],
        "handoff_sha256s": dict(handoffs),
        "public_test_accessed": False,
    }


def run_real_evolution(
    output_dir: str | Path,
    manifest: RealEvolutionManifestV2,
    ports: RealStagePorts,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> RealRunResultV2:
    """Run or safely resume one immutable real P2→P5 root epoch."""
    if not isinstance(manifest, RealEvolutionManifestV2):
        raise TypeError("manifest must be RealEvolutionManifestV2")
    if not isinstance(ports, RealStagePorts):
        raise TypeError("ports must be RealStagePorts")
    if not callable(monotonic):
        raise TypeError("monotonic must be callable")

    root = Path(output_dir)
    plan = _plan(manifest)
    store, ledger, checkpoint = _load_state(root, manifest, plan, monotonic)
    records = list(checkpoint.stage_records) if checkpoint is not None else []
    handoffs = dict(checkpoint.handoff_sha256s) if checkpoint is not None else {}
    carry = checkpoint.carry_seconds if checkpoint is not None else 0

    if checkpoint is not None and checkpoint.phase == "COMPLETE":
        if checkpoint.completion_sha256 is None or checkpoint.active_stage is not None:
            raise RealRunnerError("completed root checkpoint is malformed")
        public = _validate_sealed_records(root, records, handoffs, manifest, ports)
        if public or len(records) != len(_STAGES) or not ledger.finalization_started:
            raise RealRunnerError("completed root violates its sealed finalization boundary")
        result = RealRunResultV2.from_payload(_read_canonical(root / "evaluation_complete.json"))
        expected_result = _result("complete", manifest, records, checkpoint.completion_sha256)
        if result != expected_result:
            raise RealRunnerError("completed root result does not match checkpoint")
        summary = _read_canonical(root / "result_summary.json")
        if summary != _summary(manifest, plan, records, handoffs):
            raise RealRunnerError("completed root summary does not match its checkpointed bindings")
        if fingerprint_payload(summary) != checkpoint.completion_sha256:
            raise RealRunnerError("completed root summary digest does not match checkpoint")
        return result

    if checkpoint is not None and checkpoint.phase in {"INCOMPLETE", "FAILED"} and checkpoint.active_stage is None:
        return _result("incomplete" if checkpoint.phase == "INCOMPLETE" else "failed", manifest, records)

    if checkpoint is not None and checkpoint.active_stage is not None:
        stage = checkpoint.active_stage
        open_rows = checkpoint.budget_checkpoint.get("open_reservations", [])
        active = next((row for row in open_rows if row.get("stage_id") == stage), None)
        if not isinstance(active, Mapping):
            raise RealRunnerError("active root stage has no budget reservation")
        estimate = ResourceUse.from_payload(active["estimate"])
        if estimate.wall_seconds <= 0.0 or not estimate.wall_seconds.is_integer():
            raise RealRunnerError("active root stage grant must be a positive whole second")
        grant_seconds = int(estimate.wall_seconds)
        if stage != _STAGES[len(records)]:
            raise RealRunnerError("active root stage does not follow sealed records")
        context = _context(root, stage, grant_seconds, manifest, handoffs)
        prior_result = _read_run_result(root, stage)
        if prior_result is None:
            ledger.close_stage(active["reservation_sha256"], estimate)
            records.append(RealStageRecordV2(stage, grant_seconds, grant_seconds, "incomplete", None, _previous_progress_sha(root)))
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None,
                        carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        seal = getattr(ports, f"seal_{stage}")
        try:
            sealed = seal(context, prior_result)
            if not isinstance(sealed, SealedStageV2):
                raise RealRunnerError("stage seal must return SealedStageV2")
            _stage_status(sealed)
            _stage_output(root, stage, sealed.completion_sha256, sealed.public_test_accessed)
        except Exception:
            _terminal_active_failure(
                root=root, store=store, ledger=ledger, reservation=active["reservation_sha256"],
                stage=stage, grant_seconds=grant_seconds, records=records, handoffs=handoffs,
            )
            raise
        closure = ledger.close_stage(active["reservation_sha256"], estimate)
        if not closure.allowed:
            records.append(RealStageRecordV2(stage, grant_seconds, grant_seconds, "failed", sealed.completion_sha256, _previous_progress_sha(root)))
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None,
                        carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)
        handoff_payload = _sealed_handoff_payload(stage, sealed)
        handoff_sha = fingerprint_payload(handoff_payload)
        write_once_json(root / "handoffs" / f"{stage}.json", handoff_payload)
        record = RealStageRecordV2(stage, grant_seconds, grant_seconds, _stage_status(sealed), sealed.completion_sha256, _previous_progress_sha(root))
        records.append(record)
        handoffs[stage] = handoff_sha
        row = {"schema_version": 1, "stage": stage, "stage_record_sha256": record.fingerprint(),
               "completion_sha256": sealed.completion_sha256, "handoff_sha256": handoff_sha, "status": record.status}
        _append_progress_once(store, root, row)
        if record.status != "complete":
            _checkpoint(store, ledger, phase="INCOMPLETE" if record.status == "incomplete" else "FAILED", records=records,
                        active_stage=None, carry_seconds=0, handoff_sha256s=handoffs, completion_sha256=None)
            return _result(record.status, manifest, records)
        carry = 0
        _checkpoint(store, ledger, phase=_PHASES[stage][1], records=records, active_stage=None,
                    carry_seconds=carry, handoff_sha256s=handoffs, completion_sha256=None)

    if records:
        public = _validate_sealed_records(root, records, handoffs, manifest, ports)
        if public:
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)

    for stage in _STAGES[len(records) :]:
        grant_seconds = _grant(manifest, ledger, stage, carry)
        if grant_seconds <= 0:
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None, carry_seconds=0,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        permit = ledger.reserve_stage(stage, ResourceUse(wall_seconds=float(grant_seconds)))
        if not permit.allowed:
            _checkpoint(store, ledger, phase="INCOMPLETE", records=records, active_stage=None, carry_seconds=0,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("incomplete", manifest, records)
        _checkpoint(store, ledger, phase=_PHASES[stage][0], records=records, active_stage=stage,
                    carry_seconds=carry, handoff_sha256s=handoffs, completion_sha256=None)
        start = float(monotonic())
        if not math.isfinite(start):
            raise RealRunnerError("stage monotonic interval is invalid")
        context = _context(
            root,
            stage,
            grant_seconds,
            manifest,
            handoffs,
            deadline_monotonic=start + grant_seconds,
            monotonic=monotonic,
        )
        result = getattr(ports, f"run_{stage}")(context)
        end = float(monotonic())
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise RealRunnerError("stage monotonic interval is invalid")
        charged_seconds = math.ceil(end - start)
        _persist_run_result(root, stage, result)
        try:
            sealed = getattr(ports, f"seal_{stage}")(context, result)
            if not isinstance(sealed, SealedStageV2):
                raise RealRunnerError("stage seal must return SealedStageV2")
            stage_status = _stage_status(sealed)
            _stage_output(root, stage, sealed.completion_sha256, sealed.public_test_accessed)
            closure = ledger.close_stage(permit, ResourceUse(wall_seconds=float(charged_seconds)))
        except Exception:
            _terminal_active_failure(
                root=root, store=store, ledger=ledger, reservation=permit,
                stage=stage, grant_seconds=grant_seconds, records=records, handoffs=handoffs,
            )
            raise
        if not closure.allowed:
            stage_status = "failed"
        handoff_payload = _sealed_handoff_payload(stage, sealed)
        handoff_sha = fingerprint_payload(handoff_payload)
        write_once_json(root / "handoffs" / f"{stage}.json", handoff_payload)
        record = RealStageRecordV2(stage, grant_seconds, charged_seconds, stage_status, sealed.completion_sha256, _previous_progress_sha(root))
        records.append(record)
        handoffs[stage] = handoff_sha
        _append_progress_once(
            store,
            root,
            {"schema_version": 1, "stage": stage, "stage_record_sha256": record.fingerprint(),
             "completion_sha256": sealed.completion_sha256, "handoff_sha256": handoff_sha, "status": stage_status},
        )
        carry = max(0, grant_seconds - charged_seconds)
        phase = _PHASES[stage][1] if stage_status == "complete" else ("INCOMPLETE" if stage_status == "incomplete" else "FAILED")
        _checkpoint(store, ledger, phase=phase, records=records, active_stage=None, carry_seconds=carry,
                    handoff_sha256s=handoffs, completion_sha256=None)
        if stage_status != "complete":
            return _result(stage_status, manifest, records)
        if sealed.public_test_accessed:
            _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                        handoff_sha256s=handoffs, completion_sha256=None)
            return _result("failed", manifest, records)

    if len(records) != len(_STAGES):
        raise RealRunnerError("root reached finalization without four sealed stages")
    ledger.begin_finalization()
    _checkpoint(store, ledger, phase="FINALIZING", records=records, active_stage=None, carry_seconds=carry,
                handoff_sha256s=handoffs, completion_sha256=None)
    if ports.begin_finalization is not None:
        ports.begin_finalization()
    public = _validate_sealed_records(root, records, handoffs, manifest, ports)
    if public:
        _checkpoint(store, ledger, phase="FAILED", records=records, active_stage=None, carry_seconds=carry,
                    handoff_sha256s=handoffs, completion_sha256=None)
        return _result("failed", manifest, records)
    summary = _summary(manifest, plan, records, handoffs)
    completion_sha = fingerprint_payload(summary)
    write_once_json(root / "result_summary.json", summary)
    result = _result("complete", manifest, records, completion_sha)
    store.write_completion(result.to_payload())
    _checkpoint(store, ledger, phase="COMPLETE", records=records, active_stage=None, carry_seconds=carry,
                handoff_sha256s=handoffs, completion_sha256=completion_sha)
    return result


__all__ = [
    "RealRunnerError",
    "build_real_stage_ports",
    "RealStageContextV2",
    "RealStagePorts",
    "SealedStageV2",
    "run_real_evolution",
]
