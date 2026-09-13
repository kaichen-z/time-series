"""Typed child-run bridges for bounded real Evolution V2 orchestration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

from common.payload import strict_json_loads
from evolving_loop.data import ContextTask
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.retrieval_agent.policy import RetrievalGenome

from ..budget import BudgetPlan, ResourceUse
from ..bundle import EvolutionBundleV2
from ..cli import numerical_evolve_payload
from ..contracts import canonical_v2_bytes, fingerprint_payload
from ..cooperative import (
    CooperativeArtifactCatalog,
    CooperativeConfigV2,
    CooperativePipelineAdapter,
    CooperativeRunResultV2,
    DecisionCoordinateAdapter,
    DecisionModuleV2,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
    RetrievalModuleV2,
    run_cooperative_evolution,
)
from ..cooperative.contracts import CooperativeCheckpointV2
from ..kernel import AcceptanceEvidence, EvolutionKernel, KernelAuthorityError
from ..numerical_qd.adapters import FrozenNumericalArtifactsV2
from ..numerical_qd.contracts import FrozenNumericalRegistryEnvelopeV2
from ..store import V2RunStore, write_once_json
from .host import RealHostRuntimeV2, select_real_task_projection


_DECISION_SEED_PROMPT = (
    "Select the safest valid numerical candidate using only retrieved context."
)
_DECISION_PROMPTS = ("Prefer the lowest finite complete-pipeline error.",)
_PROPOSAL_SPACE_FILE = "proposal_space_manifest.json"
_P5_UNAVAILABLE = "p5_handoff_unavailable"


@dataclass(frozen=True, slots=True)
class P3BundleClosureV2:
    """Verified executable closure of one completed cooperative run."""

    active_bundle: EvolutionBundleV2
    numerical: FrozenNumericalArtifactsV2
    retrieval: RetrievalModuleV2
    decision: DecisionModuleV2
    catalog: CooperativeArtifactCatalog
    train_tasks: tuple[ContextTask, ...]
    dev_tasks: tuple[ContextTask, ...]
    metric_cap: float
    config_sha256: str
    numerical_alternatives: tuple[FrozenNumericalArtifactsV2, ...]
    decision_prompts: tuple[str, ...]
    decision_settings_cycle: bool
    closed_candidate_bundles: tuple[EvolutionBundleV2, ...]
    acceptance_evidence: tuple[AcceptanceEvidence, ...]
    proposal_space_manifest: Mapping[str, object]
    proposal_space_sha256: str
    runtime_identity: str
    completion_sha256: str
    checkpoint_sha256: str
    p5_handoff_available: bool
    p5_handoff_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_space_manifest",
            MappingProxyType(dict(self.proposal_space_manifest)),
        )


def run_real_numerical(
    context: object,
    host: RealHostRuntimeV2,
    *,
    config_payload: Mapping[str, object],
    seed_payload: Mapping[str, object],
    task_manifest_payload: Mapping[str, object],
    input_sha256s: Mapping[str, str],
    task_local_evidence_path: Path | None = None,
    task_local_dictionary: object | None = None,
) -> dict[str, object]:
    """Invoke P2 through the payload seam using a root stage context."""
    output_dir = getattr(context, "output_dir", None)
    if output_dir is None:
        raise TypeError("real numerical context requires output_dir")
    return numerical_evolve_payload(
        config_payload,
        seed_payload,
        task_manifest_payload,
        output_dir,
        input_sha256s=input_sha256s,
        host_runtime=host,
        llm_client=host.llm_client,
        task_local_evidence_path=task_local_evidence_path,
        task_local_dictionary=task_local_dictionary,
    )


def _read_canonical(path: Path, identity: str | None = None) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = strict_json_loads(raw.decode("utf-8"), context=str(path))
        if type(payload) is not dict or canonical_v2_bytes(payload) != raw:
            raise ValueError("not canonical JSON")
        if identity is not None and fingerprint_payload(payload) != identity:
            raise ValueError("digest mismatch")
        return payload
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise KernelAuthorityError(
            f"cannot verify cooperative artifact {path.name}: {error}"
        ) from error


def _task_payload(task: ContextTask) -> object:
    if hasattr(task, "to_payload"):
        return task.to_payload()
    return asdict(task)


def _host_runtime_identity(host: RealHostRuntimeV2) -> str:
    value = getattr(host, "resource_reporter_sha256", None)
    if type(value) is not str or len(value) != 64:
        raise ValueError("real cooperative Host runtime identity is unavailable")
    return value


def _seed_retrieval(host: RealHostRuntimeV2) -> RetrievalModuleV2:
    skills = tuple(
        skill.to_payload() for skill in host.retrieval_skill_library.all()
    )
    active_ids = tuple(
        skill.skill_id
        for skill in host.retrieval_skill_library.all()
        if skill.is_active
    )
    genome = replace(RetrievalGenome.seed(), active_skill_ids=active_ids)
    source_identity = fingerprint_payload(
        {"kind": "verified_retrieval_release", "skills": list(skills)}
    )
    return RetrievalModuleV2(1, source_identity, genome.to_payload(), skills)


def _seed_decision() -> DecisionModuleV2:
    return DecisionModuleV2(1, _DECISION_SEED_PROMPT, (), True, 2, "last")


def _pair_row(pair: FrozenNumericalArtifactsV2) -> dict[str, object]:
    release_payload = pair.release.to_payload()
    registry_payload = dict(pair.registry.manifest)
    return {
        "release_sha256": pair.release.fingerprint,
        "registry_sha256": pair.registry.fingerprint,
        "envelope_sha256": pair.envelope.fingerprint(),
        "release_object_sha256": fingerprint_payload(release_payload),
        "registry_object_sha256": fingerprint_payload(registry_payload),
        "selected_genome_sha256s": list(pair.selected_genome_sha256s),
    }


def _proposal_inputs(
    numerical: FrozenNumericalArtifactsV2,
    retrieval: RetrievalModuleV2,
    decision: DecisionModuleV2,
    alternatives: tuple[FrozenNumericalArtifactsV2, ...],
    train: tuple[ContextTask, ...],
    dev: tuple[ContextTask, ...],
) -> dict[str, str]:
    pairs = [
        {
            "release_sha256": pair.release.fingerprint,
            "registry_sha256": pair.registry.fingerprint,
            "envelope_sha256": pair.envelope.fingerprint(),
        }
        for pair in alternatives
    ]
    return {
        "seed_numerical_release": numerical.release.fingerprint,
        "seed_numerical_registry": numerical.registry.fingerprint,
        "seed_numerical_envelope": numerical.envelope.fingerprint(),
        "seed_retrieval": retrieval.fingerprint(),
        "seed_decision": decision.fingerprint(),
        "numerical_alternatives": fingerprint_payload({"pairs": pairs}),
        "decision_operator": fingerprint_payload(
            {"prompts": list(_DECISION_PROMPTS), "settings_cycle": False}
        ),
        "train_tasks": fingerprint_payload(
            {"tasks": [_task_payload(task) for task in train]}
        ),
        "dev_tasks": fingerprint_payload(
            {"tasks": [_task_payload(task) for task in dev]}
        ),
    }


def _proposal_manifest(
    *,
    p2: FrozenNumericalArtifactsV2,
    retrieval: RetrievalModuleV2,
    decision: DecisionModuleV2,
    alternatives: tuple[FrozenNumericalArtifactsV2, ...],
    train: tuple[ContextTask, ...],
    dev: tuple[ContextTask, ...],
    runtime_identity: str,
    input_sha256s: Mapping[str, str],
    config: CooperativeConfigV2,
) -> dict[str, object]:
    config_payload = config.to_payload()
    return {
        "schema_version": 1,
        "kind": "real_cooperative_proposal_space",
        "seed_numerical": _pair_row(p2),
        "numerical_alternatives": [_pair_row(pair) for pair in alternatives],
        "seed_retrieval_sha256": retrieval.fingerprint(),
        "seed_decision_sha256": decision.fingerprint(),
        "decision_operator": {
            "prompts": list(_DECISION_PROMPTS),
            "settings_cycle": False,
        },
        "task_projection": {
            "train_task_ids": [task.numeric.task_id for task in train],
            "dev_task_ids": [task.numeric.task_id for task in dev],
        },
        "input_sha256s": dict(input_sha256s),
        "config": config_payload,
        "config_sha256": fingerprint_payload(config_payload),
        "metric_cap": config.metric_cap,
        "runtime_identity": runtime_identity,
    }


def run_real_cooperative(
    *,
    p2: FrozenNumericalArtifactsV2,
    host: RealHostRuntimeV2,
    config_payload: Mapping[str, object],
    output_dir: Path,
) -> CooperativeRunResultV2:
    """Run P3 over the real 4/1 projection with the exact P2 registry."""
    if type(p2) is not FrozenNumericalArtifactsV2:
        raise TypeError("real cooperative bridge requires exact P2 frozen artifacts")
    tasks = tuple(host.tasks)
    if len(tasks) != 100 or len(host.train_tasks) != 80 or len(host.dev_tasks) != 20:
        raise ValueError("real cooperative Host requires frozen Train80/Dev20 tasks")
    restored = p2.envelope.restore(tasks)
    if (
        restored.fingerprint != p2.registry.fingerprint
        or p2.release.fingerprint != p2.registry.release_sha256
    ):
        raise ValueError("real cooperative P2 registry does not bind all Host tasks")
    train, dev = select_real_task_projection(host.train_tasks, host.dev_tasks)

    config = CooperativeConfigV2.from_payload(config_payload)
    if config.control.profile not in {"pilot", "formal"}:
        raise ValueError("real cooperative bridge requires a pilot/formal profile")
    retrieval = _seed_retrieval(host)
    decision = _seed_decision()
    alternatives = tuple(
        sorted(
            getattr(host, "numerical_alternatives", ()),
            key=lambda pair: (pair.release.fingerprint, pair.registry.fingerprint),
        )
    )
    if any(type(pair) is not FrozenNumericalArtifactsV2 for pair in alternatives):
        raise TypeError("real cooperative Numerical alternatives must be frozen pairs")
    for pair in alternatives:
        if pair.envelope.restore(tasks).fingerprint != pair.registry.fingerprint:
            raise ValueError("real cooperative Numerical alternative task drift")

    pipeline = CooperativePipelineAdapter(
        CooperativeArtifactCatalog(lambda _identity, _payload: None),
        host.retrieval_factory,
        host.decision_factory,
        metric_cap=config.metric_cap,
        retrieval_skill_library=host.retrieval_skill_library,
    )
    adapters = {
        "numerical": NumericalCoordinateAdapter(alternatives),
        "retrieval": RetrievalCoordinateAdapter(),
        "decision": DecisionCoordinateAdapter(_DECISION_PROMPTS),
        "pipeline": pipeline,
        "resource_reporter": host.resource_reporter,
    }
    monotonic = getattr(host, "monotonic", None)
    if monotonic is not None:
        adapters["monotonic"] = monotonic
    expected_inputs = _proposal_inputs(
        p2, retrieval, decision, alternatives, train, dev
    )
    destination = Path(output_dir)
    resume = (destination / "run_manifest.json").is_file()
    result = run_cooperative_evolution(
        destination,
        config,
        {"numerical": p2, "retrieval": retrieval, "decision": decision},
        {"train": train, "dev": dev},
        adapters,
        resume=resume,
    )
    checkpoint = CooperativeCheckpointV2.from_payload(
        _read_canonical(destination / "cooperative_checkpoint.json")
    )
    if dict(checkpoint.input_sha256s) != expected_inputs:
        raise KernelAuthorityError(
            "cooperative checkpoint differs from proposal-space commitments"
        )
    manifest = _proposal_manifest(
        p2=p2,
        retrieval=retrieval,
        decision=decision,
        alternatives=alternatives,
        train=train,
        dev=dev,
        runtime_identity=_host_runtime_identity(host),
        input_sha256s=expected_inputs,
        config=config,
    )
    write_once_json(destination / _PROPOSAL_SPACE_FILE, manifest)
    return result


def _budget_plan(root: Path) -> BudgetPlan:
    payload = _read_canonical(root / "budget_plan.json")
    if set(payload) != {
        "schema_version",
        "hard_limit_seconds",
        "finalization_reserve_fraction",
        "search_deadline_seconds",
        "ceilings",
    } or payload["schema_version"] != 1:
        raise KernelAuthorityError("invalid cooperative budget plan")
    plan = BudgetPlan(
        hard_limit_seconds=payload["hard_limit_seconds"],
        finalization_reserve_fraction=payload["finalization_reserve_fraction"],
        ceilings=ResourceUse.from_payload(payload["ceilings"]),
    )
    if plan.to_payload() != payload:
        raise KernelAuthorityError("cooperative budget plan mismatch")
    return plan


def _load_progress(root: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = (root / "cooperative_progress.jsonl").read_bytes().splitlines(
            keepends=True
        )
    except OSError as error:
        raise KernelAuthorityError("cannot verify cooperative progress") from error
    rows = []
    for line in lines:
        value = strict_json_loads(line.decode("utf-8"), context="cooperative progress")
        if type(value) is not dict or canonical_v2_bytes(value) != line:
            raise KernelAuthorityError("cooperative progress is not canonical")
        rows.append(value)
    return tuple(rows)


def _validate_progress_continuity(
    rows: tuple[dict[str, object], ...], checkpoint: CooperativeCheckpointV2
) -> None:
    if (
        len(rows) != checkpoint.next_step
        or tuple(row.get("step") for row in rows) != tuple(range(len(rows)))
        or tuple(row.get("candidate_sha256") for row in rows)
        != checkpoint.completed_candidate_sha256s
        or sum(row.get("decision") == "accept" for row in rows)
        != checkpoint.accepted_steps
        or sum(row.get("decision") != "accept" for row in rows)
        != checkpoint.rejected_steps
    ):
        raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    if rows and (
        rows[-1].get("active_after") != checkpoint.active_bundle_sha256
        or rows[-1].get("scheduler_state_sha256")
        != checkpoint.scheduler_state_sha256
    ):
        raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    for previous, current in zip(rows, rows[1:]):
        if previous.get("active_after") != current.get("active_before"):
            raise KernelAuthorityError("cooperative progress/checkpoint mismatch")
    for row in rows:
        changed = row.get("active_before") != row.get("active_after")
        if changed != (row.get("decision") == "accept"):
            raise KernelAuthorityError("cooperative progress/checkpoint mismatch")


def _load_numerical_pair(
    root: Path,
    row: Mapping[str, object],
    tasks: tuple[ContextTask, ...],
) -> FrozenNumericalArtifactsV2:
    release_payload = _read_canonical(
        root / "objects" / f"{row['release_object_sha256']}.json",
        row["release_object_sha256"],
    )
    envelope_payload = _read_canonical(
        root / "objects" / f"{row['envelope_sha256']}.json",
        row["envelope_sha256"],
    )
    release = parse_numerical_supply_release(release_payload)
    envelope = FrozenNumericalRegistryEnvelopeV2.from_payload(envelope_payload)
    registry = envelope.restore(tasks)
    selected = tuple(row["selected_genome_sha256s"])
    pair = FrozenNumericalArtifactsV2(release, registry, envelope, selected)
    if (
        release.fingerprint != row["release_sha256"]
        or registry.fingerprint != row["registry_sha256"]
    ):
        raise KernelAuthorityError("proposal-space Numerical pair mismatch")
    return pair


def _validate_manifest_shape(payload: Mapping[str, object]) -> None:
    if set(payload) != {
        "schema_version",
        "kind",
        "seed_numerical",
        "numerical_alternatives",
        "seed_retrieval_sha256",
        "seed_decision_sha256",
        "decision_operator",
        "task_projection",
        "input_sha256s",
        "config",
        "config_sha256",
        "metric_cap",
        "runtime_identity",
    } or payload.get("schema_version") != 1 or payload.get("kind") != (
        "real_cooperative_proposal_space"
    ):
        raise KernelAuthorityError("invalid cooperative proposal-space manifest")


def _p5_handoff_available(
    active: EvolutionBundleV2, closed: tuple[EvolutionBundleV2, ...]
) -> bool:
    concrete = {active.fingerprint(), *(bundle.fingerprint() for bundle in closed)}
    return len(concrete) >= 2


def load_sealed_bundle_closure(
    root: Path,
    *,
    tasks: tuple[ContextTask, ...],
    host: RealHostRuntimeV2,
) -> P3BundleClosureV2:
    """Authenticate a completed P3 store and restore its executable catalog."""
    run_root = Path(root)
    manifest = _read_canonical(run_root / _PROPOSAL_SPACE_FILE)
    _validate_manifest_shape(manifest)
    if manifest["runtime_identity"] != _host_runtime_identity(host):
        raise KernelAuthorityError("cooperative proposal-space runtime mismatch")
    train, dev = select_real_task_projection(host.train_tasks, host.dev_tasks)
    expected_projection = {
        "train_task_ids": [task.numeric.task_id for task in train],
        "dev_task_ids": [task.numeric.task_id for task in dev],
    }
    if manifest["task_projection"] != expected_projection or tuple(tasks) != tuple(
        host.tasks
    ):
        raise KernelAuthorityError("cooperative proposal-space task projection mismatch")

    plan = _budget_plan(run_root)
    kernel = EvolutionKernel.resume(
        V2RunStore(run_root), plan, monotonic=lambda: 0.0
    )
    active = kernel.active_bundle()
    checkpoint = CooperativeCheckpointV2.from_payload(
        _read_canonical(run_root / "cooperative_checkpoint.json")
    )
    completion_payload = _read_canonical(run_root / "evaluation_complete.json")
    completion = CooperativeRunResultV2.from_payload(completion_payload)
    rows = _load_progress(run_root)
    kernel_checkpoint = _read_canonical(run_root / "checkpoint.json")
    if (
        completion.status != "cooperative_complete"
        or completion.public_test_accessed
        or completion.active_bundle_sha256 != active.fingerprint()
        or completion.scheduler_state_sha256 != checkpoint.scheduler_state_sha256
        or completion.accepted_steps != checkpoint.accepted_steps
        or completion.rejected_steps != checkpoint.rejected_steps
        or checkpoint.active_bundle_sha256 != active.fingerprint()
        or checkpoint.kernel_checkpoint_sha256
        != kernel_checkpoint.get("checkpoint_sha256")
        or tuple(row.get("arm") for row in rows) != completion.attempted_arms
    ):
        raise KernelAuthorityError("cooperative completion/checkpoint/progress mismatch")
    _validate_progress_continuity(rows, checkpoint)
    if dict(checkpoint.input_sha256s) != manifest["input_sha256s"]:
        raise KernelAuthorityError("cooperative proposal-space input mismatch")
    config_value = manifest["config"]
    if not isinstance(config_value, Mapping):
        raise KernelAuthorityError("cooperative proposal-space config mismatch")
    try:
        config = CooperativeConfigV2.from_payload(config_value)
    except (TypeError, ValueError) as error:
        raise KernelAuthorityError("cooperative proposal-space config mismatch") from error
    if (
        fingerprint_payload(config.to_payload()) != manifest["config_sha256"]
        or checkpoint.config_sha256 != manifest["config_sha256"]
        or isinstance(manifest["metric_cap"], bool)
        or not isinstance(manifest["metric_cap"], (int, float))
        or float(manifest["metric_cap"]) != config.metric_cap
    ):
        raise KernelAuthorityError("cooperative proposal-space config mismatch")
    operator = manifest["decision_operator"]
    if checkpoint.input_sha256s["decision_operator"] != fingerprint_payload(operator):
        raise KernelAuthorityError("cooperative proposal-space Decision drift")
    alternatives_payload = manifest["numerical_alternatives"]
    pair_inputs = [
        {
            "release_sha256": row["release_sha256"],
            "registry_sha256": row["registry_sha256"],
            "envelope_sha256": row["envelope_sha256"],
        }
        for row in alternatives_payload
    ]
    if checkpoint.input_sha256s["numerical_alternatives"] != fingerprint_payload(
        {"pairs": pair_inputs}
    ):
        raise KernelAuthorityError("cooperative proposal-space Numerical drift")

    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    seed_pair = _load_numerical_pair(run_root, manifest["seed_numerical"], tasks)
    catalog.add_numerical(seed_pair)
    alternatives = tuple(
        _load_numerical_pair(run_root, row, tasks) for row in alternatives_payload
    )
    for pair in alternatives:
        catalog.add_numerical(pair)
    for path in (run_root / "objects").glob("*.json"):
        payload = _read_canonical(path, path.stem)
        fields = set(payload)
        if fields == {
            "schema_version",
            "source_release_sha256",
            "genome_payload",
            "skills_payload",
        }:
            catalog.add_retrieval(RetrievalModuleV2.from_payload(payload))
        elif fields == {
            "schema_version",
            "prompt",
            "skills",
            "enable_evidence_adjustments",
            "max_evidence_adjustments",
            "aggregation",
        }:
            catalog.add_decision(DecisionModuleV2.from_payload(payload))

    active_numerical = catalog.resolve_numerical(
        active.numerical_release_sha256, active.numerical_registry_sha256
    )
    active_retrieval = catalog.resolve_retrieval(active.retrieval_release_sha256)
    active_decision = catalog.resolve_decision(active.decision_policy_sha256)
    transitions = kernel_checkpoint["completed_transitions"]
    closed_bundles = []
    evidence = []
    for row in rows:
        candidate_sha = row["candidate_sha256"]
        if candidate_sha not in transitions:
            continue
        candidate = EvolutionBundleV2.from_payload(
            _read_canonical(
                run_root / "archive/objects" / f"{candidate_sha}.json",
                candidate_sha,
            )
        )
        catalog.resolve_numerical(
            candidate.numerical_release_sha256,
            candidate.numerical_registry_sha256,
        )
        catalog.resolve_retrieval(candidate.retrieval_release_sha256)
        catalog.resolve_decision(candidate.decision_policy_sha256)
        evidence_sha = transitions[candidate_sha]["acceptance_evidence_sha256"]
        evidence_value = AcceptanceEvidence.from_payload(
            _read_canonical(
                run_root / "acceptance" / f"{evidence_sha}.json", evidence_sha
            )
        )
        if evidence_value.candidate_bundle_sha256 != candidate_sha:
            raise KernelAuthorityError("cooperative acceptance evidence mismatch")
        closed_bundles.append(candidate)
        evidence.append(evidence_value)

    available = _p5_handoff_available(active, tuple(closed_bundles))
    return P3BundleClosureV2(
        active_bundle=active,
        numerical=active_numerical,
        retrieval=active_retrieval,
        decision=active_decision,
        catalog=catalog,
        train_tasks=train,
        dev_tasks=dev,
        metric_cap=float(manifest["metric_cap"]),
        config_sha256=manifest["config_sha256"],
        numerical_alternatives=alternatives,
        decision_prompts=tuple(manifest["decision_operator"]["prompts"]),
        decision_settings_cycle=manifest["decision_operator"]["settings_cycle"],
        closed_candidate_bundles=tuple(closed_bundles),
        acceptance_evidence=tuple(evidence),
        proposal_space_manifest=manifest,
        proposal_space_sha256=fingerprint_payload(manifest),
        runtime_identity=manifest["runtime_identity"],
        completion_sha256=fingerprint_payload(completion_payload),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        p5_handoff_available=available,
        p5_handoff_reason=None if available else _P5_UNAVAILABLE,
    )


__all__ = [
    "P3BundleClosureV2",
    "load_sealed_bundle_closure",
    "run_real_cooperative",
    "run_real_numerical",
]
