"""Numerical self-evolution with one Kernel-owned resource and promotion authority.

``task_manifest`` is the adapter's committed GroupFoldManifest (or its payload).
``seed_supply`` accepts the existing NumericalSupplyRelease, its payload, or the
Task 7 ImportedNumericalSeedV2. Bootstrap inventory/policies are derived from
verified adapter sources. The optional host ``adapter.monotonic`` clock supports
deterministic execution; the default is the real monotonic clock.
"""
from __future__ import annotations

import copy
import hashlib
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile
from common.payload import strict_json_loads
from evolving_loop.package_numerical_supply import (
    NumericalSupplyRelease, bound_numerical_package, build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.package_registry import task_registry_fingerprint
from numerical_agent.evolution.champion import ChampionRecipe, EvolutionAssumption, parse_champion_release
from numerical_agent.evolution.champion_evidence import ChampionTaskRow
from numerical_agent.evolution.execution import MethodForecastError, Task as RuntimeTask
from numerical_agent.evolution.numerical_loop import run_numerical_loop
from numerical_agent.evolution.screening import _policy_payload, profile_task

from ..budget import BudgetLedger, ResourceUse, StagePermit
from ..bundle import EvolutionBundleV2
from ..contracts import canonical_v2_bytes, fingerprint_payload
from ..kernel import EvolutionKernel, SeedBootstrapAuthority, SeedBootstrapStopped
from ..store import V2RunStore, write_once_json
from .adapters import (
    ImportedNumericalSeedV2, LegacyNumericalAdapter, MaterializedNumericalChildV2, NumericalWorkStopped, _canonical_member,
    evaluate_numerical_child, freeze_qd_supply, import_numerical_seed,
    materialize_dictionary_evidence,
)
from .agent_methods import (
    CurriculumTargetV2, MutationPromptPopulationV2, MutationPromptLineageV2,
    VerifiedReusableProgramV2, apply_prompt_train_credit, derive_curriculum_targets,
    insert_prompt_child, select_prompt_lineage,
    select_reusable_programs,
)
from .config import NumericalQDConfigV2
from .artifacts import ArtifactKindV2 as ArtifactKind, validate_artifact
from .contracts import (
    ConstraintReportV2, FrozenNumericalRegistryEnvelopeV2,
    HyperbandBracketV2, HyperbandBudgetOutcomeV2, HyperbandStateV2,
    HyperbandTaskResultV2, MutationOperatorStatsV2, MutationStateV2,
    NumericalGenomeV2, NumericalInventoryV2, NumericalMemberV2,
    NumericalMutationPolicyV2, NumericalObjectiveVectorV2, NumericalProposerPromptV2,
    NumericalQDEntryV2, MorphologyCellV2, TaskCacheRowV2, TrainMutationFeedbackV2, TrainTaskV2,
)
from .descriptors import describe_history
from .hyperband import (
    _read_cache, _task_evaluation, advance_hyperband, choose_bracket,
    evaluation_cache_key, fixed_rung_manifest, pack_fold_groups,
)
from .map_elites import CounterRandom, NumericalQDArchive
from .mutation import MutationProposalV2, apply_mutation, record_train_outcome
from .nsga2 import select_survivors
from .persistence import ArtifactBytesExhausted, NumericalQDRunStore, NumericalQDStoreError
from .proposers import (
    DeterministicProposalProvider, HybridProposalProvider, LLMProposalProvider,
    NormalizedProposalBatchV2, ProviderAttemptV2, host_source_recipe,
    primitive_proposer_request,
)


_NO_DEV = {"passed": False, "parent_metrics": {}, "candidate_metrics": {}}


@dataclass(frozen=True)
class NumericalQDRunResultV2:
    status: str
    active_bundle: EvolutionBundleV2
    occupied_cells: int
    accepted_steps: int
    rejected_steps: int
    mutation_policy_sha256: str
    proposer_prompt_sha256: str
    seed_mutation_policy_sha256: str
    seed_proposer_prompt_sha256: str
    budget: dict
    public_test_accessed: bool = False


def _packed_train_task_groups(
    adapter, task_commitments, split_sha256, protocol_sha256
):
    """Bind Train tasks to a label-free whole-group ordering for 8/32/80."""

    hosts = {
        task.numeric.task_id: task
        for task in adapter.tasks
        if task.numeric.task_id in adapter.fold_manifest.task_fold_map
    }
    packed = pack_fold_groups(adapter.fold_manifest.groups)
    packed_ids = {
        task_id for bin_rows in packed for _entity_id, task_ids in bin_rows
        for task_id in task_ids
    }
    if packed_ids != set(hosts):
        raise ValueError("authenticated fold groups differ from the Train Host universe")
    groups = {}
    for bin_rows in packed:
        for entity_id, task_ids in bin_rows:
            groups[entity_id] = tuple(
                TrainTaskV2(
                    task_id,
                    entity_id,
                    task_commitments[task_id],
                    "train",
                    split_sha256,
                    protocol_sha256,
                )
                for task_id in task_ids
            )
    return groups


def _read(path):
    raw = Path(path).read_bytes()
    value = strict_json_loads(raw.decode(), context=str(path))
    if type(value) is not dict or canonical_v2_bytes(value) != raw:
        raise ValueError("runner requires canonical persisted objects")
    return value


def _materialization_failure(member_id, error):
    """Keep one bounded, traceback-free diagnostic for the generation record."""
    message = " ".join(str(error).split()) or "materialization failed"
    encoded = message.encode("utf-8")[:512]
    while True:
        try:
            message = encoded.decode("utf-8")
            break
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return {
        "member_id": member_id,
        "error_type": type(error).__name__,
        "message": message,
    }


def _persist(store, value, *, kind=None):
    payload = value.to_payload() if hasattr(value, "to_payload") else value
    from .artifacts import artifact_bytes
    identity = payload.get("checkpoint_sha256", hashlib.sha256(artifact_bytes(kind, payload)).hexdigest())
    store.write_object(identity, value, kind=kind)
    return identity


def _persist_task_local_evidence(store, evidence):
    """Copy sealed Task 4 objects with their original content identities."""
    if evidence is None:
        return
    objects = [(ArtifactKind.SHORTLIST_POLICY, evidence.policy.to_payload(), evidence.policy.fingerprint()),
        (ArtifactKind.SHORTLIST_INDEX, dict(evidence.index), fingerprint_payload(dict(evidence.index)))]
    for task_id in sorted(evidence.by_task):
        shortlist, diagnostics, shortlist_sha, diagnostics_sha = evidence.by_task[task_id]
        objects.extend(((ArtifactKind.TASK_SHORTLIST, shortlist.to_payload(), shortlist_sha),
            (ArtifactKind.HINDCAST_DIAGNOSTICS, dict(diagnostics), diagnostics_sha)))
    for kind, payload, expected in objects:
        if _persist(store, payload, kind=kind) != expected:
            raise ValueError("persisted local evidence identity mismatch")


def _available(kernel):
    values = {name: max(0.0 if type(limit) is float else 0,
                        limit - getattr(kernel.budget.charged_use, name))
              for name, limit in kernel.budget.plan.ceilings.to_payload().items()}
    values["wall_seconds"] = min(values["wall_seconds"], max(0.0,
        kernel.budget.plan.search_deadline_seconds - kernel.budget.elapsed_wall_seconds))
    return ResourceUse.from_payload(values)


def _proposal_budget(available):
    """Give model generation its own bounded window, independent of task runtime."""
    if type(available) is not ResourceUse:
        raise TypeError("proposal budget requires exact ResourceUse")
    return replace(available, wall_seconds=min(60.0, available.wall_seconds))


def _trusted_reusable_program_context(store, archive, targets, *, maximum_records):
    """Derive reusable source context solely from verified feasible archive entries."""
    target_cells = tuple(target.cell.fingerprint() for target in targets)
    records, eligible = [], set()
    for entry in sorted(archive.entries.values(), key=lambda value: value.fingerprint()):
        if not entry.constraints.feasible or entry.cell.fingerprint() not in set(target_cells):
            continue
        try:
            genome, sources, _policies = store.verify_candidate(entry.genome_sha256)
            inventory = NumericalInventoryV2.from_payload(
                store._object(genome.inventory_sha256)
            )
        except (ValueError, TypeError, NumericalQDStoreError):
            continue
        eligible.add(entry.genome_sha256)
        for member in inventory.members:
            if member.status == "quarantined":
                continue
            source = sources.get(member.source_sha256)
            if source is None:
                continue
            try:
                records.append(VerifiedReusableProgramV2(
                    1, member.member_id, entry.genome_sha256,
                    member.applicability_cells, member.source_sha256,
                    source.decode("utf-8"),
                ))
            except (UnicodeError, ValueError, TypeError):
                continue
    selected = select_reusable_programs(
        tuple(records), eligible_genome_sha256s=tuple(sorted(eligible)),
        target_cell_sha256s=tuple(sorted(set(target_cells))),
        maximum_records=maximum_records,
    ) if records and target_cells else ()
    return tuple(selected), tuple(sorted(record.fingerprint() for record in selected))


class _KernelWork:
    """Host facade: every provider reservation and close is issued by Kernel."""

    def __init__(self, kernel, parent, generation):
        self.kernel, self.parent, self.generation = kernel, parent, generation
        self.children = {}

    def reserve_stage(self, stage, estimate):
        identity = fingerprint_payload({"generation": self.generation, "stage": stage})
        child = self.parent.provisional_child("numerical", {"numerical": (
            identity, fingerprint_payload({"work_registry": identity}))})
        permit = self.kernel.reserve_evaluation(child, estimate)
        if permit.allowed:
            self.children[permit.reservation_sha256] = child
        return permit

    def close_stage(self, permit, actual, *, status="passed", objectives=None):
        child = self.children.pop(permit.reservation_sha256)
        self.kernel.close_evaluation(self.parent, child, permit=permit, status=status,
            train_objectives=objectives or {}, train_behavior_descriptors={},
            dev_comparison=_NO_DEV, resource_use=actual, account_only=True)
        checkpoint = _read(self.kernel.checkpoint_path)
        reference = checkpoint["budget_closures"][permit.reservation_sha256]
        closure = _read(self.kernel.store.root / "evaluations" /
                        reference["candidate_bundle_sha256"] / "budget_closure.json")
        return StagePermit(closure["allowed"], closure["reason"], None)

    def can_open_stage(self, estimate):
        return self.kernel.budget.can_open_stage(estimate)


class _UnavailableProvider:
    def propose(self, request):
        return NormalizedProposalBatchV2((), (), ResourceUse(), "llm", "unavailable",
            (ProviderAttemptV2("llm", ResourceUse(), "unavailable"),))


class _MaterialAccounting:
    """Bill immutable evolvable bytes; exclude only authority/control records.

    Forecasts/results, requests, sources, genomes, inventories, recipes,
    structural policies, prompts, QD entries/subsets and all release envelopes
    are material. Checkpoints, archive indices/snapshots, rung/partial closure
    records, run manifests and generation status records are control.
    """

    def __init__(self, store, kernel):
        self.store, self.kernel, self.external = store, kernel, None
        self.external_permit = None
        store.material_writer = self.write

    def write(self, relative, data, dispatch, *, kind):
        spec = validate_artifact(kind, data)
        path = self.store.directory / relative
        if not spec.billable:
            return dispatch()
        metadata = {"relative_path": relative, "kind": kind,
                    "content_sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        if path.exists() or path.is_symlink():
            self.kernel.verify_existing_material(**metadata)
            return dispatch()
        written_bytes = 0
        def written(permit):
            nonlocal written_bytes
            capability = self.kernel.prepare_material_write(permit, **metadata)
            try:
                return dispatch()
            finally:
                if path.exists() or path.is_symlink():
                    self.kernel.register_material_write(permit, capability)
                    written_bytes = len(data)
                else:
                    self.kernel.abort_material_write(permit, capability)
        if self.external is not None:
            self.external(len(data), False)
            try:
                return written(self.external_permit)
            finally:
                self.external(written_bytes, True)
        work = _KernelWork(self.kernel, self.kernel.active_bundle(), 0)
        permit = work.reserve_stage("material-" + relative, ResourceUse(artifact_bytes=len(data)))
        if not permit.allowed:
            error = ArtifactBytesExhausted if permit.reason == "artifact_bytes_exhausted" else NumericalQDStoreError
            raise error("material persistence admission denied: " + permit.reason)
        succeeded = False
        try:
            result = written(permit)
            succeeded = True
            return result
        finally:
            work.close_stage(permit, ResourceUse(artifact_bytes=written_bytes), status="passed" if succeeded else "failed")


def _bootstrap(config, release, adapter):
    """Build immutable policy metadata; never score Dev or execute proposal code."""
    train = tuple(t for t in adapter.tasks if t.numeric.task_id in adapter.fold_manifest.task_fold_map)
    protected_names = set(adapter.materializer.screening_policy.fallback_names)
    if adapter.task_local_evidence is not None:
        anchor = parse_champion_release(
            release.to_payload()["anchor_release_payload"]
        ).policy.recipe.fallback_parent
        sealed_anchors = {
            value[0].candidate_names[0]
            for value in adapter.task_local_evidence.by_task.values()
        }
        if sealed_anchors != {anchor}:
            raise ValueError("Task 4 evidence differs from the protected Anchor")
        protected_names = {anchor}
    members, policies, cells = [], {}, set()
    for sha, source in sorted(adapter.sources.items()):
        module = adapter.validate_source(source)
        for name in module.names():
            entry = adapter.materializer.screening_policy.get(name)
            if entry is None or entry.status == "quarantined" or name in protected_names:
                continue
            family = entry.family if entry.family in {"statistical", "tsfm", "combined"} else "program"
            declared = tuple(sorted({describe_history(t.numeric.history_values,
                t.numeric.prediction_length, t.numeric.frequency, family,
                config.descriptor_policy).fingerprint() for t in train}))
            cells.update(declared)
            assumption = EvolutionAssumption(name + "_history", name, "history_length", "above", "full", "select",
                "History supports the candidate.", "History is unavailable.")
            recipe = ChampionRecipe("select_" + name, "select", (name,), name, (assumption,))
            policy = recipe.to_payload()
            policies[fingerprint_payload(policy)] = policy
            members.append(NumericalMemberV2(name, family, sha, fingerprint_payload(policy), (), declared, "active"))
    if not members:
        raise ValueError("seed requires an executable verified adapter source")
    operators = tuple(sorted(config.mutation["operators"]))
    policy = NumericalMutationPolicyV2(1, {op: MutationOperatorStatsV2(0, 0, 0, 0, 0) for op in operators})
    prompt = NumericalProposerPromptV2(1, "Propose a bounded mutation using Train evidence and declared cells.",
        "numerical_mutation_batch_v1", config.proposer["max_response_bytes"], operators, None)
    state = MutationStateV2(NumericalInventoryV2(1, tuple(members)), policy, prompt, tuple(sorted(cells)),
        config.mutation["max_parents_per_child"], config.mutation["max_inventory_size"])
    screen = _policy_payload(adapter.materializer.screening_policy)
    combined = {"policies": [item.to_payload() for item in adapter.materializer.combined_policies]}
    genome = NumericalGenomeV2(1, 0, (), operators[0], state.inventory.fingerprint(),
        fingerprint_payload(screen), fingerprint_payload(combined), policy.fingerprint(), prompt.fingerprint(),
        config.runtime_fingerprints, config.kernel_protocol.fingerprint())
    return state, genome, policies, screen, combined


def _seed_registry(release, adapter, *, account_work=None, bootstrap_authority=None):
    anchor = parse_champion_release(release.to_payload()["anchor_release_payload"])
    def forecast(*args):
        if bootstrap_authority is not None:
            return bootstrap_authority.forecast(adapter, *args)
        if account_work is not None:
            account_work(ResourceUse(task_executions=1))
        return adapter.forecast_trusted(*args, account_work=account_work)
    def build(task, supplied):
        if adapter.local_evidence_for(task) is not None:
            return adapter.materialize_local_package(task, supplied, forecast)
        source = run_numerical_loop(RuntimeTask(task.numeric.task_id, task.numeric.history_values,
            task.numeric.prediction_length, task.numeric.frequency, ()),
            screening_policy=adapter.materializer.screening_policy,
            candidate_runner=forecast, champion_release=anchor)
        evidence = adapter.local_evidence_for(task)
        kwargs = {} if evidence is None else {"shortlist": evidence[0], "hindcast_diagnostics_sha256": evidence[3]}
        return bound_numerical_package(source, supplied, {item.name: item for item in source.ranked_alternatives}, **kwargs)
    return build_package_registry(adapter.tasks, release, build)


def _train_rows(adapter):
    rows = []
    for task in adapter.tasks:
        task = task.numeric
        if task.task_id not in adapter.fold_manifest.task_fold_map:
            continue
        profile = profile_task(RuntimeTask(task.task_id, task.history_values, task.prediction_length, task.frequency, ()))
        # Task 7 verifies these Host rows and recomputes forecasts from exact
        # persisted source bytes before fitting. This placeholder is not scored.
        rows.append(ChampionTaskRow(task.task_id, "host_binding", profile, task.future_values,
            (0.0,) * task.prediction_length, fold=adapter.fold_manifest.task_fold_map[task.task_id], split="build", history=task.history_values))
    return tuple(rows)


def _state_for(store, genome, config, declared_cells, policy=None, prompt=None):
    return MutationStateV2(NumericalInventoryV2.from_payload(store._object(genome.inventory_sha256)),
        policy or NumericalMutationPolicyV2.from_payload(store._object(genome.mutation_policy_sha256)),
        prompt or NumericalProposerPromptV2.from_payload(store._object(genome.proposer_prompt_sha256)),
        declared_cells, config.mutation["max_parents_per_child"], config.mutation["max_inventory_size"])


def _prompt_for_genome(store, genome, prompt_overrides):
    """Resolve the latest terminal Host-credited prompt for a sampled genome."""
    prompt_sha = prompt_overrides.get(genome.fingerprint(), genome.proposer_prompt_sha256)
    return NumericalProposerPromptV2.from_payload(store._object(prompt_sha))


def _persist_state(store, state, genome):
    for value in (state.inventory, state.mutation_policy, state.proposer_prompt, genome):
        _persist(store, value)
    return store.verify_candidate(genome.fingerprint())[0]


def _seed_prompt_population(prompt):
    mutation_prompt = NumericalProposerPromptV2(
        1,
        "Propose a bounded mutation-prompt variant using Train evidence only.",
        "numerical_mutation_batch_v1",
        prompt.max_response_bytes,
        ("policy_tune",),
        None,
    )
    return MutationPromptPopulationV2(
        1, 4, (MutationPromptLineageV2(
            1, mutation_prompt, MutationOperatorStatsV2(0, 0, 0, 0, 0)
        ),)
    )


def _host_mutation_prompt_child(parent, task_template):
    """Derive a bounded mutation prompt from a Host-accepted child prompt."""
    if type(parent) is not NumericalProposerPromptV2 or type(task_template) is not str:
        raise TypeError("Host mutation prompt child requires typed prompt inputs")
    template = task_template.strip()
    if not template:
        raise ValueError("Host mutation prompt child requires a nonempty template")
    return NumericalProposerPromptV2(
        1, template[:65536], "numerical_mutation_batch_v1",
        parent.max_response_bytes, ("policy_tune",), parent.fingerprint(),
    )
def _persist_prompt_population(store, population):
    """Persist control-only prompt population without widening material taxonomy."""
    writer = getattr(store, "material_writer", None)
    store.material_writer = None
    try:
        return _persist(store, population)
    finally:
        store.material_writer = writer


def _declared_curriculum_cells(seed_state, adapter, config):
    """Rebuild the fixed authenticated seed-cell universe as typed cells."""
    if type(seed_state) is not MutationStateV2:
        raise TypeError("declared cells require a typed seed state")
    train = tuple(
        task for task in adapter.tasks
        if task.numeric.task_id in adapter.fold_manifest.task_fold_map
    )
    families = tuple(sorted({member.family for member in seed_state.inventory.members}))
    cells = {}
    for family in families:
        for task in train:
            cell = describe_history(
                task.numeric.history_values,
                task.numeric.prediction_length,
                task.numeric.frequency,
                family,
                config.descriptor_policy,
            )
            cells[cell.fingerprint()] = cell
    if set(cells) != set(seed_state.declared_cells):
        raise ValueError("reconstructed morphology cells differ from declared universe")
    return tuple(cells[cell_sha] for cell_sha in seed_state.declared_cells)


def _latest_terminal_generation(steps):
    """Restore post-generation state, independent of pending-marker SHA order."""
    return max(
        (step for step in steps if step.get("status") != "proposal_pending"),
        key=lambda step: step["generation"], default=None,
    )


def _pending_generations_unresolved(steps, attempts):
    """Return pending generations lacking a matching terminal status."""
    terminal = {
        (step.get("generation"), step.get("proposal_attempt_sha256"), step.get("proposal_request_sha256"))
        for step in steps if step.get("status") != "proposal_pending"
    }
    return tuple(sorted(step.get("generation") for step in steps
        if step.get("status") == "proposal_pending" and not any(
            (step.get("generation"), fingerprint_payload(attempt), step.get("proposal_request_sha256")) in terminal
            for attempt in attempts
            if attempt.get("context", {}).get("generation") == step.get("generation")
            and attempt.get("context", {}).get("request_sha256") == step.get("proposal_request_sha256")
        )))


def _aggregate(values, manifest, bracket, index):
    ordered = sorted(values, key=lambda v: v.task_ids)
    fields = ("mean_capped_smae", "mean_capped_srmse", "mean_raw_joint_error", "normalized_execution_cost")
    means = {name: float(statistics.fmean(getattr(value.objectives, name) for value in ordered)) for name in fields}
    objective = NumericalObjectiveVectorV2(means["mean_capped_smae"], means["mean_capped_srmse"],
        float(linear_quantile([value.objectives.mean_capped_srmse for value in ordered], 0.95)),
        means["mean_raw_joint_error"], means["normalized_execution_cost"])
    violations = tuple(sorted({name for value in ordered for name in value.constraints.violations}))
    cells = {cell.fingerprint(): cell for value in ordered for cell in value.cells}
    use = ResourceUse()
    for value in ordered:
        use += ResourceUse.from_payload(value.resource_use)
    return replace(ordered[0], task_subset_sha256=manifest.fingerprint(), bracket=bracket, rung=index,
        task_ids=tuple(sorted(task for value in ordered for task in value.task_ids)),
        task_statuses={task: status for value in ordered for task, status in value.task_statuses.items()},
        objectives=objective, constraints=ConstraintReportV2(all(v.constraints.feasible for v in ordered), violations),
        cells=tuple(cells[sha] for sha in sorted(cells)), resource_use=use.to_payload())


def _rung(kernel, work, state, manifest, children, adapter, config, cache):
    """Task 6 pure advancement around explicit real Kernel reservations."""
    started, executed = 0.0, 0
    reported = ResourceUse()
    artifact_bytes, aggregates = 0, ()
    store = getattr(work, "store", None)
    pending, results, unpersisted = [], [], []
    for candidate in state.active_candidates:
        for task in manifest.tasks:
            local_evidence_sha256 = adapter.local_evidence_sha256_for(candidate, task)
            key = evaluation_cache_key(candidate, task.task_sha256, manifest.split_sha256,
                config.kernel_protocol.metric_policy, config.descriptor_policy.fingerprint(),
                config.runtime_fingerprints, manifest.protocol_sha256, adapter.fingerprint,
                local_evidence_sha256)
            hit = _read_cache(cache, key, candidate, task, config.kernel_protocol.metric_policy,
                config.descriptor_policy.fingerprint(), config.runtime_fingerprints, adapter.fingerprint,
                local_evidence_sha256)
            pending.append((candidate, task, key, hit, local_evidence_sha256))
    missing = sum(hit is None for _, _, _, hit, _ in pending)
    estimate = replace(_available(kernel), task_executions=missing,
                       wall_seconds=float(missing * config.adapter["task_timeout_seconds"]))
    admission = work.can_open_stage(estimate)
    if not admission.allowed:
        return None, (), admission.reason
    failure, manifest_sha = None, None
    if store is not None:
        try:
            manifest_sha = _persist(store, manifest)
        except ArtifactBytesExhausted:
            failure = "artifact_bytes_exhausted"
    estimate = (ResourceUse() if failure else replace(_available(kernel), task_executions=missing,
                wall_seconds=float(missing * config.adapter["task_timeout_seconds"])))
    admission = work.can_open_stage(estimate)
    permit = (work.reserve_stage("rung-" + state.fingerprint() + "-" + manifest.fingerprint(), estimate)
              if admission.allowed else admission)
    if not permit.allowed:
        reason = failure or permit.reason
        if store is not None:
            captured = _read(kernel.checkpoint_path)["budget"]
            work.partial_rung = {"state": state.to_payload(), "manifest_sha256": manifest_sha,
                "reason": reason, "task_results": [], "unpersisted_task_results": [],
                "budget_outcome": HyperbandBudgetOutcomeV2("blocked", reason, None,
                    ResourceUse().to_payload(), captured["checkpoint_sha256"], captured).to_payload()}
        return None, (), reason
    def account_material(size, begun):
        nonlocal artifact_bytes
        if not begun and artifact_bytes + size > estimate.artifact_bytes:
            raise ArtifactBytesExhausted("rung material artifact budget exhausted")
        if begun:
            artifact_bytes += size
    if store is not None:
        store.accounting.external = account_material
        store.accounting.external_permit = permit
    succeeded = False
    try:
        if failure:
            raise ArtifactBytesExhausted("fixed manifest admission denied")
        for candidate, task, key, hit, local_evidence_sha256 in pending:
            if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
                failure = "finalization_reserve"
                break
            if hit is not None:
                value = hit.evaluation
            else:
                before = adapter.monotonic()
                snapshot = adapter.resource_snapshot()
                executed += 1
                try:
                    value = evaluate_numerical_child(adapter, children[candidate], task,
                        descriptor_policy=config.descriptor_policy, metric_policy_sha256=config.kernel_protocol.metric_policy,
                        bracket=state.bracket.name, rung=len(state.rungs), resource_use=ResourceUse(task_executions=1))
                    value = _task_evaluation(value, candidate, task, config.kernel_protocol.metric_policy,
                        config.descriptor_policy.fingerprint(), config.runtime_fingerprints, adapter.fingerprint)
                except (ValueError, TypeError, TimeoutError):
                    failure = "invalid_response"
                    break
                finally:
                    extra = adapter.resource_delta(snapshot)
                    reported += extra
                    duration = adapter.monotonic() - before
                    started += duration
                value = replace(value, resource_use=(ResourceUse.from_payload(value.resource_use) + extra).to_payload())
                if any(getattr(reported, name) > getattr(estimate, name) for name in ResourceUse.field_names()):
                    failure = "budget_overrun"
                if duration >= config.adapter["task_timeout_seconds"]:
                    failure = failure or "timeout"
            result = HyperbandTaskResultV2(candidate, task.task_id, key,
                value.task_statuses[task.task_id], value, hit is not None, None,
                local_evidence_sha256, 2 if local_evidence_sha256 is not None else 1)
            if hit is not None and store is not None:
                # A cache hit reuses the original paid bytes, including its
                # original cache_hit flag; execution accounting stays above.
                result = HyperbandTaskResultV2.from_payload(store._read(
                    f"results/{candidate}/{task.task_sha256}.json"))
                if (result.candidate_sha256 != candidate or result.task_id != task.task_id
                        or result.cache_key != key or result.evaluation != value
                        or result.local_evidence_sha256 != local_evidence_sha256):
                    raise NumericalQDStoreError("cache row differs from its exact paid result")
            results.append((task, result))
            if store is not None and hit is None:
                try:
                    store.write_task_result(task.task_sha256, results[-1][1])
                except ArtifactBytesExhausted:
                    _, denied = results.pop()
                    unpersisted.append({"task_sha256": task.task_sha256, "task_id": task.task_id,
                        "candidate_sha256": candidate, "cache_key": key,
                        "result_sha256": fingerprint_payload(denied.to_payload()), "status": "persistence_denied",
                        "resource_use": dict(value.resource_use)})
                    raise
            if value.constraints.feasible and value.task_statuses[task.task_id] == "passed":
                cache[key] = TaskCacheRowV2(key, task.task_sha256, value,
                                             local_evidence_sha256,
                                             2 if local_evidence_sha256 is not None else 1)
            if failure:
                break
        if failure is None:
            aggregates = tuple(_aggregate([result.evaluation for _, result in results if result.candidate_sha256 == candidate],
                manifest, state.bracket.name, len(state.rungs)) for candidate in state.active_candidates)
            if store is not None:
                for aggregate in aggregates:
                    _persist(store, aggregate)
            succeeded = True
    except ArtifactBytesExhausted:
        failure = "artifact_bytes_exhausted"
    finally:
        if store is not None:
            store.accounting.external = None
            store.accounting.external_permit = None
        actual = reported + ResourceUse(task_executions=executed, wall_seconds=float(started), artifact_bytes=artifact_bytes)
        closed = work.close_stage(permit, actual, status="passed" if succeeded and failure is None else "failed")
    if not closed.allowed:
        failure = closed.reason
    if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
        failure = failure or "finalization_reserve"
    captured = _read(kernel.checkpoint_path)["budget"]
    if failure:
        work.partial_rung = {"state": state.to_payload(), "manifest_sha256": manifest_sha, "reason": failure,
            "task_results": [{"task_sha256": task.task_sha256, "result": result.to_payload()} for task, result in results],
            "unpersisted_task_results": unpersisted,
            "budget_outcome": HyperbandBudgetOutcomeV2("failed", failure, permit.reservation_sha256,
                actual.to_payload(), captured["checkpoint_sha256"], captured).to_payload()}
        return None, tuple(results), failure
    outcome = HyperbandBudgetOutcomeV2("completed", None, permit.reservation_sha256,
        actual.to_payload(), captured["checkpoint_sha256"], captured)
    return advance_hyperband(state, manifest, aggregates, outcome).state, tuple(results), None


def _dev_compare(parent_registry, child_registry, adapter, kernel, account_work):
    """Compare the frozen task-local bundles on Dev20.

    Dictionary members are exploration units, not deployable global models.  The
    package's protected selector has already used task-local hindcasts to choose
    Anchor alone or Anchor plus specialists, so promotion must score that final
    forecast rather than the newly proposed member in isolation.
    """
    scores = [[], []]
    count = 0
    ordered_tasks = tuple(sorted(adapter.tasks, key=lambda t: t.numeric.task_id))
    expected_ids = tuple(task.numeric.task_id for task in ordered_tasks)
    if any(registry.task_ids != expected_ids for registry in (parent_registry, child_registry)):
        raise ValueError("Dev registry task universe mismatch")
    # Both registries were fully fingerprint-verified when constructed/restored,
    # and their package mappings are immutable. Avoid reserializing a multi-MB
    # package on every one of the 40 Dev lookups.
    package_maps = (parent_registry._packages, child_registry._packages)
    for task in ordered_tasks:
        if task.numeric.task_id in adapter.fold_manifest.task_fold_map:
            continue
        for index, _registry in enumerate((parent_registry, child_registry)):
            if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
                return _NO_DEV
            account_work()
            count += 1
            package = package_maps[index][task.numeric.task_id]
            scores[index].append(
                drcik_point_metrics(
                    task.numeric.future_values,
                    package.final_forecast,
                    cap=5.0,
                )
            )
    metrics = [{"mean_smae": float(statistics.fmean(row["smae"] for row in values)),
                "mean_srmse": float(statistics.fmean(row["srmse"] for row in values))} for values in scores]
    passed = (count == 40 and metrics[1]["mean_smae"] < metrics[0]["mean_smae"]
              and metrics[1]["mean_srmse"] <= metrics[0]["mean_srmse"])
    return {"passed": passed, "parent_metrics": metrics[0], "candidate_metrics": metrics[1]}


def _freeze_output(kernel, work, store, adapter, parent_release, parent_registry, archive, children,
                   winner, generation, config):
    """Bill durable material envelopes, excluding authority/control persistence."""
    available = _available(kernel)
    if available.artifact_bytes == 0:
        return None, "artifact_bytes_exhausted"
    # The P2-owned export may need to fill every previously-unshortlisted
    # catalog member. Reserve the remaining governed capacity, then close with
    # exact measured use so unused capacity is returned before Dev comparison.
    estimate = available
    permit = work.reserve_stage("freeze-" + winner, estimate)
    if not permit.allowed:
        return None, permit.reason
    objects = store.directory / "objects"
    existing = set(objects.glob("*.json"))
    before = adapter.monotonic()
    result, reason = None, "freeze_failed"
    actual = ResourceUse()

    def account_work(use, *, begun=False):
        nonlocal actual
        next_use = actual + use
        if begun:
            actual = next_use
        if (
            kernel.budget.elapsed_wall_seconds
            >= kernel.budget.plan.search_deadline_seconds
            or adapter.monotonic() - before >= estimate.wall_seconds
            or any(
                getattr(next_use, name) > getattr(estimate, name)
                for name in ResourceUse.field_names()
            )
        ):
            raise NumericalWorkStopped("Dictionary evidence resource boundary")
        actual = next_use

    store.accounting.external = lambda size, begun: None  # This stage bills its new material files in finally.
    store.accounting.external_permit = permit
    try:
        occupied = {archive.entries[sha].genome_sha256 for cell in archive.cells for sha in cell.entry_sha256s} | {winner}
        verified_children = {}
        for path in sorted(objects.glob("*.json")):
            payload = store._object(path.stem)
            if "materialized_numerical_child" not in payload:
                continue
            envelope = payload["materialized_numerical_child"]
            genome_sha = fingerprint_payload(envelope["genome"])
            if genome_sha not in occupied:
                continue
            genome, sources, policies = store.verify_candidate(genome_sha)
            adapter.sources = dict(adapter.sources) | {sha: source.decode() for sha, source in sources.items()}
            child = MaterializedNumericalChildV2.from_payload(payload, adapter.tasks)
            if (child.genome != genome or adapter._recipe(child.member, policies, None, None) != child.fit.recipe
                    or child.descriptor_policy_sha256 != config.descriptor_policy.fingerprint()):
                raise ValueError("archive executable policy binding mismatch")
            if genome_sha in verified_children:
                raise ValueError("archive has conflicting executable envelopes")
            verified_children[genome_sha] = child
        if occupied != set(verified_children):
            raise ValueError("archive projection lacks a verified executable envelope")
        complete_parent_registry = materialize_dictionary_evidence(
            adapter,
            parent_release,
            parent_registry,
            account_work=account_work,
        )
        frozen = freeze_qd_supply(adapter, parent_release, complete_parent_registry, archive,
            tuple(verified_children.values()), descriptor_policy=config.descriptor_policy, version=f"n{generation:03d}",
            required_genome_sha256=winner)
        payload = {"supply": frozen.release.to_payload(), "registry": frozen.envelope.to_payload()}
        if len(canonical_v2_bytes(payload)) > estimate.artifact_bytes:
            reason = "artifact_bytes_exhausted"
        else:
            _persist_task_local_evidence(store, adapter.task_local_evidence)
            pair_sha = _persist(store, payload, kind=ArtifactKind.FROZEN_PAIR)
            pair = store._object(pair_sha)
            release = parse_numerical_supply_release(pair["supply"])
            registry = FrozenNumericalRegistryEnvelopeV2.from_payload(pair["registry"]).restore(adapter.tasks)
            result = release, registry, pair_sha
    except NumericalWorkStopped:
        reason = "materialization_budget"
    except (ValueError, TypeError, TimeoutError, MethodForecastError):
        reason = "freeze_failed"
    finally:
        store.accounting.external = None
        store.accounting.external_permit = None
        # The single typed pair embeds all package envelopes. Count each new
        # material file once, including a completed write followed by failure.
        produced = set(objects.glob("*.json")) - existing
        closed = work.close_stage(permit, replace(actual,
            artifact_bytes=sum(path.stat().st_size for path in produced if path.is_file()),
            wall_seconds=float(adapter.monotonic() - before)), status="passed" if result else "failed")
    if not closed.allowed:
        return None, closed.reason
    return result, None if result else reason


def run_numerical_qd(output_dir, config, seed_supply, task_manifest, adapter, llm_client=None,
                     resume=False, stop_after=None, finalize_after=None) -> NumericalQDRunResultV2:
    """Run to the committed resource/deadline limit, or pause after N generations.

    Stops are closed generation boundaries. A crash within an unpublished
    operation fails closed through Task 8; no partial operation is replayed.
    ``finalize_after`` is a minimum generation count and only closes early once
    the archive contains a genuinely feasible evolved entry.
    """
    if type(config) is not NumericalQDConfigV2:
        config = NumericalQDConfigV2.from_payload(config)
    if not isinstance(adapter, LegacyNumericalAdapter):
        raise TypeError("runner requires the typed LegacyNumericalAdapter boundary")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive generation count")
    if finalize_after is not None and (
        type(finalize_after) is not int or finalize_after < 1
    ):
        raise ValueError("finalize_after must be a positive generation count")
    if not resume:
        NumericalQDRunStore.preflight_fresh(output_dir)
    adapter_fingerprint = adapter.fingerprint
    operator_input_sha256s = dict(adapter.operator_input_sha256s)
    adapter = copy.copy(adapter)
    adapter.preflight_resources()
    if any(task.numeric.prediction_length > config.adapter["max_forecast_values"] for task in adapter.tasks):
        raise ValueError("task forecast size exceeds the configured adapter limit")
    adapter.monotonic = getattr(adapter, "monotonic", time.monotonic)
    manifest_payload = task_manifest.to_payload() if hasattr(task_manifest, "to_payload") else task_manifest
    if manifest_payload != adapter.fold_manifest.to_payload():
        raise ValueError("task manifest differs from adapter's exact Train80 groups")
    imported = seed_supply if type(seed_supply) is ImportedNumericalSeedV2 else None
    release = imported.release if imported else (seed_supply if type(seed_supply) is NumericalSupplyRelease
                                                else parse_numerical_supply_release(seed_supply))
    if release.schema_version == 2 and adapter.task_local_evidence is None and not adapter.legacy_bootstrap:
        raise ValueError("schema-2 production requires Task 4 evidence or explicit legacy/bootstrap mode")
    seed_state, seed_genome, seed_policies, screen, combined = _bootstrap(config, release, adapter)
    task_commitments = {task.numeric.task_id: task_registry_fingerprint(task) for task in adapter.tasks}
    inputs = {"seed_supply": fingerprint_payload(release.to_payload()),
        "task_manifest": fingerprint_payload({"folds": manifest_payload, "tasks": task_commitments}),
        "adapter": adapter_fingerprint, "sources": fingerprint_payload(dict(adapter.sources)),
        "config": config.fingerprint(), "runtimes": fingerprint_payload(dict(config.runtime_fingerprints))}
    inputs.update({f"operator_{name}": identity
                   for name, identity in operator_input_sha256s.items()})
    if adapter.fingerprint != adapter_fingerprint:
        raise ValueError("adapter identity changed after operator input binding")
    if imported:
        inputs["seed_registry"] = imported.envelope.fingerprint()
    root = Path(output_dir)
    bootstrap_resume = None
    if resume and (root / "seed_bootstrap_preflight.json").exists() and not (root / "run_manifest.json").exists():
        NumericalQDRunStore(root)._verify_root_paths(bootstrap=True)
        bootstrap_resume = SeedBootstrapAuthority.resume(V2RunStore(root), config.kernel_protocol, config.budget,
            inputs, monotonic=adapter.monotonic)
        resume = False
    if resume:
        store = NumericalQDRunStore(root)
        # Store preflight precedes the Kernel read, including finished no-op runs.
        store._verify_root_paths(require_checkpoint=True)
        kernel_payload = _read(root / "checkpoint.json")
        checkpoint = store.resume(config.fingerprint(), inputs, kernel_payload["checkpoint_sha256"],
                                  kernel_payload["budget"]["checkpoint_sha256"])
        kernel = EvolutionKernel.resume(V2RunStore(root), config.budget, monotonic=adapter.monotonic)
        active = kernel.active_bundle()
        archive = NumericalQDArchive.from_payload(store._object(checkpoint.qd_snapshot_sha256))
        state = _state_for(store, seed_genome, config, seed_state.declared_cells,
            NumericalMutationPolicyV2.from_payload(store._object(checkpoint.mutation_policy_sha256)),
            NumericalProposerPromptV2.from_payload(store._object(checkpoint.proposer_prompt_sha256)))
        active_genome = store.verify_candidate(checkpoint.active_genome_sha256)[0]
        hyperband = HyperbandStateV2.from_payload(store._object(checkpoint.hyperband_state_sha256))
        random = CounterRandom(**dict(checkpoint.counter))
        objects = [store._object(Path(name).stem) for name in checkpoint.completed_operation_sha256s
                   if name.startswith("objects/")]
        steps = [value["numerical_qd_step"] for value in objects if "numerical_qd_step" in value]
        generation = max((step["generation"] for step in steps), default=0)
        attempts = [store._read(name) for name in checkpoint.completed_operation_sha256s if name.startswith("proposals/")]
        if any(attempt.get("context", {}).get("generation", 0) > generation for attempt in attempts):
            raise NumericalQDStoreError("unfinished generation cannot be resampled; resume requires a closed generation or a new epoch")
        if _pending_generations_unresolved(steps, attempts):
            raise NumericalQDStoreError("unfinished generation cannot be resampled; resume requires a closed generation or a new epoch")
        terminal = _latest_terminal_generation(steps)
        prompt_overrides = dict(terminal.get("prompt_overrides", {})) if terminal else {}
        population_sha = (
            terminal.get("mutation_prompt_population_sha256") if terminal else None
        )
        population = (
            MutationPromptPopulationV2.from_payload(store._object(population_sha))
            if population_sha else _seed_prompt_population(
                NumericalProposerPromptV2.from_payload(
                    store._object(seed_genome.proposer_prompt_sha256)
                )
            )
        )
        previous_feedback = terminal["train_feedback"] if terminal else None
        pairs = [value for value in objects if set(value) == {"supply", "registry"}]
        pair = next(value for value in pairs if parse_numerical_supply_release(value["supply"]).fingerprint == active.numerical_release_sha256)
        parent_release = parse_numerical_supply_release(pair["supply"])
        parent_registry = FrozenNumericalRegistryEnvelopeV2.from_payload(pair["registry"]).restore(adapter.tasks)
    else:
        if bootstrap_resume is None:
            NumericalQDRunStore.preflight_fresh(root)
            kernel_store = V2RunStore.create(root)
        else:
            kernel_store = bootstrap_resume.store
        bootstrap = bootstrap_resume
        if imported:
            registry = imported.envelope.restore(adapter.tasks)
            ledger = BudgetLedger(config.budget, monotonic=adapter.monotonic)
        else:
            bootstrap_estimate = replace(config.budget.ceilings,
                wall_seconds=min(config.budget.ceilings.wall_seconds, config.budget.search_deadline_seconds / 2.0),
                artifact_bytes=config.budget.ceilings.artifact_bytes)
            preflight = {"schema_version": 1, "stage": "seed_bootstrap", "seed_supply_sha256": release.fingerprint,
                "protocol_sha256": config.kernel_protocol.fingerprint(), "budget_plan_sha256": config.budget.fingerprint(),
                "input_sha256s": inputs, "estimate": bootstrap_estimate.to_payload()}
            if bootstrap is None:
                bootstrap = SeedBootstrapAuthority(kernel_store, config.kernel_protocol, config.budget,
                    preflight, monotonic=adapter.monotonic)
            status, failure = "failed", None
            try:
                registry = _seed_registry(release, adapter, bootstrap_authority=bootstrap)
                status = "passed"
            except (SeedBootstrapStopped, NumericalWorkStopped):
                status, failure = "stopped", "resource_or_host_stop"
            except KeyboardInterrupt:
                status, failure = "interrupted", "process_interrupted"
            except (ValueError, TypeError, TimeoutError, MethodForecastError):
                failure = "materialization_failed"
            finally:
                status = bootstrap.close(status, failure)["status"]
            if status != "passed":
                raise ValueError("seed bootstrap stopped before a complete registry")
            ledger = bootstrap.budget
        active = EvolutionBundleV2(2, 0, None, release.fingerprint, registry.fingerprint,
            **dict(config.fixed_bundle_components), protocol_fingerprint=config.kernel_protocol.fingerprint(),
            runtime_fingerprints=config.runtime_fingerprints, acceptance_evidence_sha256=None)
        kernel = EvolutionKernel(kernel_store, config.kernel_protocol, ledger, seed=active,
                                 bootstrap_authority=bootstrap)
        store = NumericalQDRunStore.create(root)
        store.accounting = _MaterialAccounting(store, kernel)
        for sha, source in adapter.sources.items():
            store.write_source(sha, source.encode())
        _persist(store, config)
        _persist(store, screen, kind=ArtifactKind.SCREENING_POLICY)
        _persist(store, combined, kind=ArtifactKind.COMBINED_POLICY)
        for payload in seed_policies.values():
            _persist(store, payload, kind=ArtifactKind.RECIPE_POLICY)
        _persist_state(store, seed_state, seed_genome)
        imported_seed = import_numerical_seed(release, registry, tasks=adapter.tasks, evidence=adapter.task_local_evidence)
        _persist_task_local_evidence(store, adapter.task_local_evidence)
        _persist(store, {"supply": release.to_payload(), "registry": imported_seed.envelope.to_payload()}, kind=ArtifactKind.FROZEN_PAIR)
        state, active_genome = seed_state, seed_genome
        population = _seed_prompt_population(seed_state.proposer_prompt)
        archive = NumericalQDArchive(capacity=config.map_elites["cell_capacity"])
        hyperband = HyperbandStateV2(HyperbandBracketV2.registered("explore"), (seed_genome.fingerprint(),),
            config.hyperband["reduction_factor"], config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint(), ())
        random, generation = CounterRandom(config.seed, "numerical"), 0
        previous_feedback = None
        prompt_overrides = {}
        parent_release, parent_registry = release, registry

    if not hasattr(store, "accounting"):
        store.accounting = _MaterialAccounting(store, kernel)

    def checkpoint_state():
        current = _read(kernel.checkpoint_path)
        _persist_prompt_population(store, population)
        return store.write_state(config_sha256=config.fingerprint(), input_sha256s=inputs,
            active_bundle_sha256=_persist(store, active), active_genome_sha256=active_genome.fingerprint(),
            qd_snapshot_sha256=_persist(store, archive), mutation_policy_sha256=_persist(store, state.mutation_policy),
            proposer_prompt_sha256=_persist(store, state.proposer_prompt), hyperband_state_sha256=_persist(store, hyperband),
            counter=random.to_payload(), budget_checkpoint_sha256=_persist(store, current["budget"], kind=ArtifactKind.BUDGET_CHECKPOINT),
            kernel_checkpoint_sha256=current["checkpoint_sha256"])

    def result(status):
        current = _read(kernel.checkpoint_path)
        transitions = tuple(current["completed_transitions"].values())
        return NumericalQDRunResultV2(status, active, len(archive.cells),
            sum(row["decision"] == "accept" for row in transitions), sum(row["decision"] == "reject" for row in transitions),
            state.mutation_policy.fingerprint(), state.proposer_prompt.fingerprint(), seed_state.mutation_policy.fingerprint(),
            seed_state.proposer_prompt.fingerprint(), current["budget"])

    if (root / "evaluation_complete.json").exists():
        return result("numerical_qd_complete")
    if not resume:
        checkpoint_state()
    groups = _packed_train_task_groups(
        adapter,
        task_commitments,
        config.kernel_protocol.split_manifest,
        config.kernel_protocol.fingerprint(),
    )
    # Validate all registered boundaries before charged dispatch.
    for resource in (8, 32, 80):
        fixed_rung_manifest(groups, resource, config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint())
    cache = {}
    feedback = (TrainMutationFeedbackV2.from_payload(previous_feedback) if previous_feedback else
                TrainMutationFeedbackV2("train", sorted(config.mutation["operators"])[0], False, False, False, ()))
    while not kernel.budget.finalization_started:
        if (finalize_after is not None and generation >= finalize_after
                and archive.entries):
            break
        if stop_after is not None and generation >= stop_after:
            return result("numerical_qd_paused")
        if not kernel.budget.can_open_stage(ResourceUse(task_executions=1)).allowed:
            break
        generation += 1
        counter_start = random.counter
        sampled = archive.sample_parent(feedback, random).genome_sha256 if archive.entries else seed_genome.fingerprint()
        selected = store.verify_candidate(sampled)[0]
        parent_state = _state_for(store, selected, config, seed_state.declared_cells,
            state.mutation_policy, _prompt_for_genome(store, selected, prompt_overrides))
        selected_prompt = select_prompt_lineage(population)
        selected = replace(selected, mutation_policy_sha256=state.mutation_policy.fingerprint(),
                           proposer_prompt_sha256=parent_state.proposer_prompt.fingerprint())
        _persist_state(store, parent_state, selected)
        # Declared cells are the authenticated complete morphology universe;
        # never narrow it to whichever families happen to survive in a parent.
        curriculum_cells = _declared_curriculum_cells(seed_state, adapter, config)
        targets = derive_curriculum_targets(
            tuple(sorted(curriculum_cells, key=lambda cell: cell.fingerprint())),
            archive, feedback, maximum_targets=8,
        ) if curriculum_cells else ()
        reusable, reusable_shas = _trusted_reusable_program_context(
            store, archive, targets, maximum_records=8,
        )
        population_sha = _persist_prompt_population(store, population)
        draw = random.randbelow(2 ** 32)
        checkpoint_state()  # RNG authority is durable before any provider call.
        if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
            _persist(store, {"numerical_qd_step": {
                "generation": generation,
                "status": "finalization_reserve",
                "active_bundle_sha256": active.fingerprint(),
                "winner_genome_sha256": None,
                "proposal_attempt_sha256": None,
                "train_feedback": feedback.to_payload(),
                "mutation_prompt_population_sha256": _persist_prompt_population(store, population),
                "prompt_overrides": dict(sorted(prompt_overrides.items())),
                "materialization_failures": [],
            }}, kind=ArtifactKind.GENERATION_STATUS)
            checkpoint_state()
            break
        proposal_budget = _proposal_budget(_available(kernel))
        request = primitive_proposer_request(parent_genome=selected.to_payload(), parent_state=parent_state.to_payload(),
            selected_cells=[{"cell_sha256": cell, "member_ids": sorted(m.member_id for m in parent_state.inventory.members
                if cell in m.applicability_cells)} for cell in parent_state.declared_cells],
            train_feedback=[feedback.to_payload()], remaining_budget=proposal_budget.to_payload(),
            curriculum_targets=[target.to_payload() for target in targets],
            reusable_programs=[program.to_payload() for program in reusable],
            eligible_reusable_program_sha256s=list(reusable_shas),
            mutation_prompt=selected_prompt.mutation_prompt.to_payload(),
            mutation_prompt_population_sha256=population_sha,
            allowed_mutation_operators=sorted(config.mutation["operators"]), counter_draw=draw,
            max_proposals=config.proposer["max_proposals_per_generation"], max_response_bytes=config.proposer["max_response_bytes"])
        request_sha = _persist(store, request, kind=ArtifactKind.PROPOSER_REQUEST)
        _persist(store, {"numerical_qd_step": {
            "generation": generation, "status": "proposal_pending",
            "active_bundle_sha256": active.fingerprint(), "winner_genome_sha256": None,
            "proposal_attempt_sha256": None, "train_feedback": feedback.to_payload(),
            "materialization_failures": [], "mutation_prompt_population_sha256": population_sha,
            "proposal_request_sha256": request_sha,
        }}, kind=ArtifactKind.GENERATION_STATUS)
        checkpoint_state()  # request and pending marker precede provider dispatch
        work = _KernelWork(kernel, active, generation)
        work.store = store
        artifact_permit = work.reserve_stage("proposal-artifacts-" + request_sha, ResourceUse(artifact_bytes=
            2 * max(config.proposer["max_response_bytes"], len(parent_state.canonical_bytes())) + 4096))
        if not artifact_permit.allowed:
            break
        deterministic = DeterministicProposalProvider(monotonic=adapter.monotonic)
        if config.proposer["provider"] == "hybrid":
            provider = LLMProposalProvider(llm_client, monotonic=adapter.monotonic) if llm_client else _UnavailableProvider()
            batch = HybridProposalProvider(provider, deterministic, work).propose(request)
        else:
            permit = work.reserve_stage("proposal-" + fingerprint_payload(request), ResourceUse(wall_seconds=proposal_budget.wall_seconds))
            if not permit.allowed:
                work.close_stage(artifact_permit, ResourceUse())
                break
            batch = deterministic.propose(request)
            work.close_stage(permit, batch.resource_use)
        payload = {"provider": batch.provider, "resource_use": batch.resource_use.to_payload(),
            "failure_reason": batch.failure_reason, "proposals": [p.to_payload() for p in batch.proposals],
            "source_sha256s": [sha for sha, _ in batch.source_artifacts],
            "attempts": [{"provider": attempt.provider, "resource_use": attempt.resource_use.to_payload(),
                          "failure_reason": attempt.failure_reason} for attempt in batch.attempts]}
        attempt_payload = {"context": {"generation": generation,
            "parent_genome_sha256": selected.fingerprint(), "request_sha256": request_sha,
            "mutation_prompt_sha256": selected_prompt.mutation_prompt.fingerprint(),
            "mutation_prompt_population_sha256": population_sha,
            "curriculum_target_sha256s": [target.cell.fingerprint() for target in targets],
            "eligible_reusable_program_sha256s": list(reusable_shas),
            "counter": {"seed": random.seed, "stream": random.stream, "start": counter_start, "end": random.counter}},
            "batch": payload}
        batch_sha = fingerprint_payload(attempt_payload)
        artifact_paths = [store.directory / f"sources/{sha}.py" for sha, _ in batch.source_artifacts]
        artifact_paths.append(store.directory / f"proposals/{batch_sha}.json")
        new_paths = [path for path in artifact_paths if not path.exists()]
        succeeded = False
        try:
            store.accounting.external = lambda size, begun: None
            store.accounting.external_permit = artifact_permit
            for sha, source in batch.source_artifacts:
                store.write_source(sha, source)
            store.write_proposal_attempt(batch_sha, attempt_payload)
            succeeded = True
        finally:
            store.accounting.external = None
            store.accounting.external_permit = None
            work.close_stage(artifact_permit, ResourceUse(artifact_bytes=sum(
                path.stat().st_size for path in new_paths if path.is_file())), status="passed" if succeeded else "failed")
        batch = store._proposal(store._read(f"proposals/{batch_sha}.json"))
        if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
            _persist(store, {"numerical_qd_step": {
                "generation": generation,
                "status": "finalization_reserve",
                "active_bundle_sha256": active.fingerprint(),
                "winner_genome_sha256": None,
                "proposal_attempt_sha256": batch_sha,
                "proposal_request_sha256": request_sha,
                "train_feedback": feedback.to_payload(),
                "mutation_prompt_population_sha256": _persist_prompt_population(store, population),
                "prompt_overrides": dict(sorted(prompt_overrides.items())),
                "materialization_failures": [],
            }}, kind=ArtifactKind.GENERATION_STATUS)
            checkpoint_state()
            break
        children, child_states, attempted, budget_blocked = {}, {}, [], False
        generation_prompt_overrides = {}
        materialization_failures = []
        generation_stop_reason = None
        for proposal in batch.proposals:
            if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
                generation_stop_reason, budget_blocked = "finalization_reserve", True
                break
            attempted.append((proposal.operator, None))
            proposal_payload = proposal.to_payload()
            policy_ready = True
            for key in ("member", "child", "replacement"):
                if key not in proposal_payload:
                    continue
                member = NumericalMemberV2.from_payload(proposal_payload[key])
                structural = {"schema_version": 1, "operator": proposal.operator,
                    "parents": [m.to_payload() for name in member.parent_ids for m in parent_state.inventory.members if m.member_id == name],
                    "applicability_cells": list(member.applicability_cells)}
                if fingerprint_payload(structural) != member.policy_sha256:
                    # New source code receives one canonical Host-derived
                    # select recipe. The proposer can commit its digest, but
                    # only the Host constructs and persists the policy bytes.
                    if not (store.directory / f"objects/{member.policy_sha256}.json").exists():
                        recipe = host_source_recipe(store._source(member.source_sha256))
                        if fingerprint_payload(recipe.to_payload()) != member.policy_sha256:
                            policy_ready = False
                            break
                        _persist(store, recipe, kind=ArtifactKind.RECIPE_POLICY)
                else:
                    _persist(store, structural, kind=ArtifactKind.STRUCTURAL_POLICY)
                # Exact policy/source readback precedes the atomic transition.
                store._object(member.policy_sha256)
                store._source(member.source_sha256)
            if not policy_ready:
                continue
            candidate_state = apply_mutation(parent_state, proposal).state
            if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
                generation_stop_reason, budget_blocked = "finalization_reserve", True
                break
            genome = NumericalGenomeV2(1, generation, (selected.fingerprint(),), proposal.operator,
                candidate_state.inventory.fingerprint(), selected.screening_policy_sha256, selected.combined_policy_sha256,
                candidate_state.mutation_policy.fingerprint(), candidate_state.proposer_prompt.fingerprint(),
                config.runtime_fingerprints, config.kernel_protocol.fingerprint())
            candidate_member = _canonical_member(candidate_state)
            try:
                _persist_state(store, candidate_state, genome)
                genome, sources, policies = store.verify_candidate(genome.fingerprint())
            except ValueError as error:
                materialization_failures.append(
                    _materialization_failure(candidate_member.member_id, error)
                )
                continue
            adapter.sources = dict(adapter.sources) | {sha: data.decode() for sha, data in sources.items()}
            remaining = _available(kernel)
            # Reserve an upper bound, then release unused capacity at closure.
            # Each uncached forecast/worker start checks this bound before work.
            estimate = replace(remaining,
                wall_seconds=min(remaining.wall_seconds / 2.0,
                    remaining.task_executions * config.adapter["task_timeout_seconds"]),
                subprocesses=remaining.subprocesses)
            permit = work.reserve_stage("materialize-" + genome.fingerprint(), estimate)
            if not permit.allowed:
                budget_blocked = True
                if permit.reason in {"finalization_reserve", "finalization_started"}:
                    generation_stop_reason = "finalization_reserve"
                break
            before = adapter.monotonic()
            child, failure = None, None
            actual = ResourceUse()
            def account_work(use, *, begun=False):
                nonlocal actual
                next_use = actual + use
                if begun:
                    actual = next_use
                if (kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds
                        or adapter.monotonic() - before >= estimate.wall_seconds
                        or any(getattr(next_use, name) > getattr(estimate, name)
                               for name in ResourceUse.field_names())):
                    raise NumericalWorkStopped("materialization resource boundary")
                actual = next_use
            try:
                child = adapter.materialize_child(parent_release, genome, candidate_state,
                    member_id=candidate_member.member_id, policies=policies,
                    build_rows=_train_rows(adapter), descriptor_policy=config.descriptor_policy,
                    version=f"n{generation:03d}", parent_state=parent_state, proposal=proposal,
                    account_work=account_work, task_timeout_seconds=config.adapter["task_timeout_seconds"],
                    parent_registry=parent_registry)
            except NumericalWorkStopped as error:
                failure, budget_blocked = "materialization_budget", True
                materialization_failures.append(
                    _materialization_failure(candidate_member.member_id, error)
                )
            except (ValueError, TypeError, TimeoutError, MethodForecastError) as error:
                failure = "materialization_failed"
                materialization_failures.append(
                    _materialization_failure(candidate_member.member_id, error)
                )
            finally:
                closed = work.close_stage(permit, replace(actual, wall_seconds=float(adapter.monotonic() - before)),
                    status="failed" if failure else "passed")
            if kernel.budget.elapsed_wall_seconds >= kernel.budget.plan.search_deadline_seconds:
                child = None
                generation_stop_reason, budget_blocked = "finalization_reserve", True
            if child is not None and closed.allowed:
                child = MaterializedNumericalChildV2.from_payload(
                    store._object(_persist(store, child.to_payload(adapter.tasks), kind=ArtifactKind.EXECUTABLE_CHILD)), adapter.tasks)
                children[genome.fingerprint()] = child
                child_states[genome.fingerprint()] = candidate_state
                attempted[-1] = (proposal.operator, genome.fingerprint())
            if budget_blocked:
                break
        reason, evaluations = generation_stop_reason or "no_feasible_child", ()
        if children and _available(kernel).wall_seconds > 0:
            try:
                bracket = choose_bracket(len(children), 0.0, _available(kernel).wall_seconds, config)
            except ValueError:
                bracket = None
                budget_blocked = True
            if bracket:
                hyperband = HyperbandStateV2(bracket, tuple(sorted(children)), config.hyperband["reduction_factor"],
                    config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint(), ())
                while not hyperband.complete:
                    manifest = fixed_rung_manifest(groups, bracket.resources[len(hyperband.rungs)],
                        config.kernel_protocol.split_manifest, config.kernel_protocol.fingerprint())
                    advanced, task_results, reason = _rung(kernel, work, hyperband, manifest, children, adapter, config, cache)
                    if advanced is None:
                        if hasattr(work, "partial_rung"):
                            _persist(store, {"closed_partial_rung": work.partial_rung}, kind=ArtifactKind.PARTIAL_RUNG)
                            del work.partial_rung
                        budget_blocked = reason not in {"invalid_response", "timeout"}
                        break
                    _persist(store, manifest)
                    for task, task_result in task_results:
                        if not task_result.cache_hit:
                            store.write_task_result(task.task_sha256, task_result)
                    hyperband = advanced
                    store.write_rung(hyperband.rungs[-1])
                    _persist(store, hyperband)
                    checkpoint_state()
                if hyperband.complete:
                    evaluations = hyperband.rungs[-1].evaluations
        entries = []
        for evaluation in evaluations:
            if not evaluation.constraints.feasible:
                continue
            for cell in evaluation.cells:
                task_rows = [row.evaluation for row in cache.values() if row.evaluation.genome_sha256 == evaluation.genome_sha256
                             and cell in row.evaluation.cells]
                cell_eval = _aggregate(task_rows, hyperband.rungs[-1].manifest, hyperband.bracket.name, len(hyperband.rungs) - 1)
                subset = {"schema_version": 1, "kind": "cell_subset", "cell": cell.to_payload(),
                    "parent_manifest_sha256": hyperband.rungs[-1].manifest.fingerprint(),
                    "split_sha256": evaluation.split_sha256, "protocol_sha256": evaluation.protocol_fingerprint,
                    "task_ids": list(cell_eval.task_ids), "tasks": [task.to_payload() for task in
                        sorted(hyperband.rungs[-1].manifest.tasks, key=lambda task: task.task_id)
                        if task.task_id in cell_eval.task_ids]}
                cell_eval = replace(cell_eval, task_subset_sha256=_persist(store, subset, kind=ArtifactKind.CELL_SUBSET), cells=(cell,))
                _persist(store, cell_eval)
                entries.append(NumericalQDEntryV2(1, evaluation.genome_sha256, cell_eval.fingerprint(), cell,
                    cell_eval.task_ids, cell_eval.objectives, cell_eval.constraints, evaluation.train_diagnostic_categories))
        if entries:
            archive = archive.insert(entries)
            for entry in sorted(entries, key=lambda item: item.fingerprint()):
                store.append_qd_entry(entry)
        for operator, genome_sha in attempted:
            feasible = any(e.genome_sha256 == genome_sha and e.constraints.feasible
                for rung in hyperband.rungs for e in rung.evaluations)
            promoted = any(genome_sha in rung.survivor_sha256s
                and any(e.genome_sha256 == genome_sha and e.constraints.feasible for e in rung.evaluations)
                for index, rung in enumerate(hyperband.rungs) if index < len(hyperband.bracket.resources) - 1)
            inserted = any(e.genome_sha256 == genome_sha for e in entries)
            feedback = TrainMutationFeedbackV2("train", operator,
                feasible, promoted, inserted, ())
            if operator == "policy_tune" and genome_sha in child_states:
                child_prompt = child_states[genome_sha].proposer_prompt
                mutation_child = _host_mutation_prompt_child(
                    selected_prompt.mutation_prompt, child_prompt.template,
                )
                population = insert_prompt_child(
                    population, selected_prompt.mutation_prompt.fingerprint(), mutation_child,
                )
            # Credit belongs to the sampled genome's prompt lineage.  The
            # shared policy state may supply operator authority, but must not
            # receive feedback for a different prompt ancestry.
            updated = record_train_outcome(parent_state, feedback.to_payload())
            if operator == "policy_tune":
                population = apply_prompt_train_credit(
                    population, selected_prompt.mutation_prompt.fingerprint(), feedback,
                )
            credit_bytes = sum(len(value.canonical_bytes()) for value in (updated.mutation_policy, updated.proposer_prompt)
                if not (store.directory / f"objects/{value.fingerprint()}.json").exists())
            if kernel.budget.can_open_stage(ResourceUse(artifact_bytes=credit_bytes)).allowed:
                _persist(store, updated.mutation_policy)
                _persist(store, updated.proposer_prompt)
                state = updated
                generation_prompt_overrides[sampled] = updated.proposer_prompt.fingerprint()
                if inserted:
                    # An inserted child is the sampled lineage's executable
                    # descendant and is what MAP-Elites can sample next.
                    generation_prompt_overrides[genome_sha] = updated.proposer_prompt.fingerprint()
            else:
                # Train feedback remains in the closed generation record. A new
                # evolvable policy cannot be produced after its admission limit.
                budget_blocked = True
        _persist(store, archive)
        checkpoint_state()
        candidates = tuple(NumericalQDEntryV2(1, value.genome_sha256, value.fingerprint(), value.cells[0],
            value.task_ids, value.objectives, value.constraints, value.train_diagnostic_categories)
            for value in evaluations if value.constraints.feasible)
        winner = select_survivors(candidates, 1)[0].genome_sha256 if candidates else None
        if winner is not None:
            frozen_pair, reason = _freeze_output(kernel, work, store, adapter, parent_release, parent_registry,
                archive, children, winner, generation, config)
            if frozen_pair is None:
                budget_blocked = reason != "freeze_failed"
        else:
            frozen_pair = None
            reason = reason or "no_feasible_child"
        if frozen_pair is not None:
            frozen_release, frozen_registry, pair_sha = frozen_pair
            candidate = active.provisional_child("numerical", {"numerical": (frozen_release.fingerprint, frozen_registry.fingerprint)})
            permit = kernel.reserve_evaluation(candidate, ResourceUse(task_executions=40,
                wall_seconds=min(10.0, float(40 * config.adapter["task_timeout_seconds"]))))
            if permit.allowed:
                before = adapter.monotonic()
                comparison, count = _NO_DEV, 0
                def account_dev():
                    nonlocal count
                    count += 1
                try:
                    comparison = _dev_compare(
                        parent_registry,
                        frozen_registry,
                        adapter,
                        kernel,
                        account_dev,
                    )
                except (ValueError, TypeError, TimeoutError, MethodForecastError):
                    comparison = _NO_DEV
                finally:
                    evaluation = next(value for value in evaluations if value.genome_sha256 == winner)
                    closed = kernel.close_evaluation(active, candidate, permit=permit, status="passed" if count == 40 else "failed",
                        train_objectives=evaluation.objectives.to_payload(),
                        train_behavior_descriptors={"numerical_artifacts_sha256": pair_sha,
                            "numerical_winner_genome_sha256": winner,
                            "numerical_train_evaluation_sha256": evaluation.fingerprint(),
                            "numerical_winner_materialized_sha256": fingerprint_payload(children[winner].to_payload(adapter.tasks))},
                        dev_comparison=comparison,
                        resource_use=ResourceUse(task_executions=count, wall_seconds=float(adapter.monotonic() - before)))
                    previous = active
                    active = kernel.evaluate_transition(previous, candidate, target="numerical", evaluation=closed, permit=permit)
                if active is not previous:
                    active_genome = children[winner].genome
                    parent_release, parent_registry = frozen_release, frozen_registry
                    reason = "accepted"
                else:
                    reason = "rejected"
            else:
                reason = permit.reason
        prompt_overrides.update(generation_prompt_overrides)
        _persist(store, {"numerical_qd_step": {"generation": generation, "status": reason,
            "active_bundle_sha256": active.fingerprint(), "winner_genome_sha256": winner,
            "proposal_attempt_sha256": batch_sha, "train_feedback": feedback.to_payload(),
            "proposal_request_sha256": request_sha,
            "mutation_prompt_population_sha256": _persist_prompt_population(store, population),
            "prompt_overrides": dict(sorted(prompt_overrides.items())),
            "materialization_failures": materialization_failures}}, kind=ArtifactKind.GENERATION_STATUS)
        checkpoint_state()
        if budget_blocked:
            break
    kernel.finalize()
    checkpoint_state()
    store.write_completion({"status": "numerical_qd_complete"})
    return result("numerical_qd_complete")


__all__ = ["NumericalQDRunResultV2", "run_numerical_qd"]
