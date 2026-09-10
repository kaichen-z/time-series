"""Single-writer, immutable Numerical QD artifacts and verified runner authority."""
from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from common.payload import strict_json_loads

from .. import store as durable
from ..budget import BudgetLedger, BudgetPlan, ResourceUse
from ..bundle import EvolutionBundleV2
from ..contracts import canonical_v2_bytes, fingerprint_payload, require_sha256, _require_exact_schema
from ..kernel import EvolutionKernel
from .contracts import (
    HyperbandRungV2, HyperbandStateV2, HyperbandTaskResultV2,
    NumericalEvaluationV2, NumericalGenomeV2, NumericalInventoryV2,
    NumericalMemberV2, NumericalMutationPolicyV2, NumericalProposerPromptV2,
    NumericalQDCheckpointV2, NumericalQDEntryV2,
)
from .hyperband import evaluation_cache_key
from .map_elites import NumericalQDArchive
from .config import NumericalQDConfigV2


_MANIFEST = {"schema_version": 1, "system": "numerical_qd"}
_LAYOUT = ("objects", "sources", "proposals", "results", "rungs", "archive")
_SHA = r"[a-f0-9]{64}"
_FILES = re.compile(rf"(?:manifest\.json|archive/entries\.jsonl|"
    rf"(?:objects|proposals|rungs)/{_SHA}\.json|sources/{_SHA}\.py|results/{_SHA}/{_SHA}\.json)\Z")
_TEMP = re.compile(r"\..+\.[^.]+\.tmp\Z")


class NumericalQDStoreError(durable.StoreContractError):
    """Incomplete or conflicting durable Numerical QD state."""


def _payload(value):
    if hasattr(value, "to_payload"):
        value = value.to_payload()
    if isinstance(value, bytes):
        raw = value
        try:
            value = strict_json_loads(raw.decode("utf-8"), context="Numerical QD artifact")
        except (UnicodeError, ValueError) as error:
            raise NumericalQDStoreError("artifact is not canonical JSON") from error
        if not isinstance(value, dict) or canonical_v2_bytes(value) != raw:
            raise NumericalQDStoreError("artifact is not canonical JSON")
    if not isinstance(value, Mapping):
        raise NumericalQDStoreError("artifact must be a canonical JSON object")
    # Detach and validate values, including deeply frozen mappings.
    return strict_json_loads(canonical_v2_bytes(value).decode("utf-8"), context="artifact")


def _identity(payload):
    if "checkpoint_sha256" in payload:
        body = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        sha = fingerprint_payload(body)
        if payload["checkpoint_sha256"] != sha:
            raise NumericalQDStoreError("checkpoint body SHA mismatch")
        return sha
    return fingerprint_payload(payload)


class NumericalQDRunStore:
    """All paths are rooted in an isolated V2 run; only checkpoint is mutable.

    A caller persists dependencies before evaluation and uses verify_candidate
    to reread exact executable/policy bytes. write_state publishes only fully
    closed operations. resume never repairs, appends, or rewrites anything.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        self.directory = self.root / "numerical_qd"

    @classmethod
    def create(cls, root):
        """Create the Numerical subtree after Project 1 initializes the V2 run."""
        result = cls(root)
        result._verify_root_paths(creating=True)
        repository = Path(__file__).resolve().parents[3]
        protected = ("common", "evolving_loop", "numerical_agent", "retrieval_agent",
                     "decision_agent", "tests", "runs/frozen_two_stage")
        resolved = result.root.resolve()
        if any(resolved.is_relative_to(repository / name) for name in protected):
            raise NumericalQDStoreError("output cannot be created in source or legacy paths")
        durable.V2RunStore._require_v2_manifest(result.root)
        if result.directory.exists():
            if result._read("manifest.json") != _MANIFEST:
                raise NumericalQDStoreError("Numerical QD manifest mismatch")
            result._catalog()
        for name in _LAYOUT:
            result._safe(result.directory / name)
            durable._ensure_directory(result.directory / name)
        result._write("manifest.json", _MANIFEST)
        return result

    def _safe(self, path):
        if not path.is_relative_to(self.root):
            raise NumericalQDStoreError("path escapes V2 run")
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise NumericalQDStoreError("symlink is not a durable run path")
        return path

    def _path(self, relative):
        if not isinstance(relative, str) or (relative != "checkpoint.json" and not _FILES.fullmatch(relative)):
            raise NumericalQDStoreError("invalid Numerical QD artifact path")
        return self._safe(self.directory / relative)

    def _verify_root_paths(self, *, creating=False, require_checkpoint=False):
        """Metadata-only preflight; never open an artifact before this passes.

        Kernel authority is always required. Only create may start without the
        Numerical subtree, and only pre-publication APIs may lack its checkpoint.
        Traversal uses lstat before descending, so no symlink or special file is
        followed even when the bad node is unrelated to the first content read.
        """
        required_files = ("run_manifest.json", "budget_plan.json", "checkpoint.json",
                          "accepted_bundle.json", "promotion_history.jsonl", "archive/index.jsonl")
        try:
            for path in (*reversed(self.root.parents), self.root):
                if not stat.S_ISDIR(path.lstat().st_mode):
                    raise NumericalQDStoreError("V2 root and ancestors must be real directories")
            modes, pending = {self.root: stat.S_IFDIR}, [self.root]
            while pending:
                for path in pending.pop().iterdir():
                    mode = path.lstat().st_mode
                    if stat.S_ISDIR(mode):
                        pending.append(path)
                    elif not stat.S_ISREG(mode):
                        raise NumericalQDStoreError("V2 authority requires real files and directories")
                    modes[path] = mode

            def require(path, predicate, *, optional=False):
                mode = modes.get(path)
                if mode is None and optional:
                    return
                if mode is None or not predicate(mode):
                    raise NumericalQDStoreError(f"missing or wrong authority path type: {path}")

            for name in durable._LAYOUT:
                require(self.root / name, stat.S_ISDIR)
            for name in required_files:
                require(self.root / name, stat.S_ISREG)
            require(self.root / "evaluation_complete.json", stat.S_ISREG, optional=True)
            if creating and self.directory not in modes:
                return
            require(self.directory, stat.S_ISDIR)
            for name in _LAYOUT:
                require(self.directory / name, stat.S_ISDIR)
            require(self.directory / "manifest.json", stat.S_ISREG)
            require(self.directory / "checkpoint.json", stat.S_ISREG, optional=not require_checkpoint)
            require(self.directory / "archive/entries.jsonl", stat.S_ISREG, optional=True)
        except OSError as error:
            raise NumericalQDStoreError("cannot verify V2 authority paths") from error

    def _read(self, relative):
        try:
            return _payload(self._path(relative).read_bytes())
        except OSError as error:
            raise NumericalQDStoreError(f"missing immutable artifact: {relative}") from error

    def _write(self, relative, payload):
        payload = _payload(payload)
        path = self._path(relative)
        durable.write_once_json(path, payload)
        if self._read(relative) != payload:
            raise NumericalQDStoreError("immutable write failed canonical readback")
        return path

    def write_object(self, sha256, payload):
        self._verify_root_paths()
        require_sha256(sha256, "object SHA")
        payload = _payload(payload)
        if _identity(payload) != sha256:
            raise NumericalQDStoreError("object content SHA mismatch")
        return self._write(f"objects/{sha256}.json", payload)

    def _object(self, sha256):
        require_sha256(sha256, "object SHA")
        payload = self._read(f"objects/{sha256}.json")
        if _identity(payload) != sha256:
            raise NumericalQDStoreError("immutable object SHA mismatch")
        return payload

    def write_source(self, sha256, source):
        self._verify_root_paths()
        require_sha256(sha256, "source SHA")
        if type(source) is not bytes or hashlib.sha256(source).hexdigest() != sha256:
            raise NumericalQDStoreError("source must be exact SHA-bound bytes")
        path = self._path(f"sources/{sha256}.py")
        if path.exists():
            if path.read_bytes() != source:
                raise NumericalQDStoreError("conflicting immutable source retry")
            durable._ensure_directory(path.parent)
        else:
            durable._atomic_write(path, source)
        if self._source(sha256) != source:
            raise NumericalQDStoreError("source readback mismatch")
        return path

    def _source(self, sha256):
        require_sha256(sha256, "source SHA")
        try:
            source = self._path(f"sources/{sha256}.py").read_bytes()
        except OSError as error:
            raise NumericalQDStoreError("missing source bytes") from error
        if hashlib.sha256(source).hexdigest() != sha256:
            raise NumericalQDStoreError("source content SHA mismatch")
        return source

    def _verify_prompt(self, prompt_sha256):
        seen = set()
        while prompt_sha256 is not None:
            if prompt_sha256 in seen:
                raise NumericalQDStoreError("cyclic prompt ancestry")
            seen.add(prompt_sha256)
            prompt = NumericalProposerPromptV2.from_payload(self._object(prompt_sha256))
            prompt_sha256 = prompt.parent_prompt_sha256

    def verify_candidate(self, genome_sha256):
        """Return reread genome plus exact source/policy ancestry for the adapter."""
        self._verify_root_paths()
        from .adapters import LegacyNumericalAdapter
        from numerical_agent.evolution.champion import parse_champion_recipe

        sources, policies = {}, {}

        def member_closure(member, ancestors=()):
            if member.policy_sha256 in ancestors or len(ancestors) >= 64:
                raise NumericalQDStoreError("cyclic or excessive structural policy ancestry")
            source = self._source(member.source_sha256)
            try:
                module = LegacyNumericalAdapter.validate_source(source.decode("utf-8"))
            except (UnicodeError, ValueError) as error:
                raise NumericalQDStoreError("invalid executable source") from error
            sources[member.source_sha256] = source
            policy = self._object(member.policy_sha256)
            policies[member.policy_sha256] = policy
            if "kind" in policy:
                recipe = parse_champion_recipe(policy)
                if not set(recipe.parents) <= set(module.names()):
                    raise NumericalQDStoreError("recipe does not bind executable source")
                return
            _require_exact_schema(policy, ("schema_version", "operator", "parents", "applicability_cells"), field="structural policy")
            parents = tuple(NumericalMemberV2.from_payload(value) for value in policy["parents"])
            operator = policy["operator"]
            arity = 1 if operator in {"repair", "fork"} else 2
            if (type(policy["schema_version"]) is not int or policy["schema_version"] != 1
                    or operator not in {"repair", "fork", "combine", "route", "crossover"}
                    or len(parents) != arity or member.parent_ids != tuple(p.member_id for p in parents)
                    or member.source_sha256 != parents[0].source_sha256
                    or not set(member.applicability_cells) <= set(policy["applicability_cells"])):
                raise NumericalQDStoreError("structural policy ancestry mismatch")
            for parent in parents:
                member_closure(parent, (*ancestors, member.policy_sha256))

        pending, verified, selected = [(genome_sha256, ())], set(), None
        while pending:
            identity, ancestors = pending.pop()
            if identity in ancestors:
                raise NumericalQDStoreError("cyclic genome ancestry")
            if identity in verified:
                continue
            genome = NumericalGenomeV2.from_payload(self._object(identity))
            if selected is None:
                selected = genome
            inventory = NumericalInventoryV2.from_payload(self._object(genome.inventory_sha256))
            NumericalMutationPolicyV2.from_payload(self._object(genome.mutation_policy_sha256))
            self._verify_prompt(genome.proposer_prompt_sha256)
            self._object(genome.screening_policy_sha256)
            self._object(genome.combined_policy_sha256)
            for member in inventory.members:
                member_closure(member)
            verified.add(identity)
            pending.extend((parent, (*ancestors, identity)) for parent in genome.parent_genome_sha256s)
        return selected, sources, policies

    def _proposal(self, payload):
        from .mutation import MutationProposalV2
        from .proposers import NormalizedProposalBatchV2, ProviderAttemptV2, primitive_proposer_request

        if "context" in payload or "batch" in payload:
            envelope = _require_exact_schema(payload, ("context", "batch"), field="contextual proposal attempt")
            context = _require_exact_schema(envelope["context"],
                ("generation", "parent_genome_sha256", "request_sha256", "counter"), field="proposal context")
            generation = context["generation"]
            if type(generation) is not int or generation < 1:
                raise NumericalQDStoreError("proposal context requires a positive generation")
            for name in ("parent_genome_sha256", "request_sha256"):
                require_sha256(context[name], name)
            request = primitive_proposer_request(**self._object(context["request_sha256"]))
            parent = self.verify_candidate(context["parent_genome_sha256"])[0]
            if parent.to_payload() != request["parent_genome"] or generation <= parent.generation:
                raise NumericalQDStoreError("proposal request/Parent/generation mismatch")
            counter = _require_exact_schema(context["counter"], ("seed", "stream", "start", "end"), field="proposal counter range")
            if (any(type(counter[name]) is not int for name in ("seed", "start", "end"))
                    or not 0 <= counter["start"] < counter["end"]):
                raise NumericalQDStoreError("proposal counter range must be nonempty and ordered")
            checkpoint = NumericalQDCheckpointV2.from_payload(self._read("checkpoint.json"))
            if (counter["seed"] != checkpoint.counter["seed"] or counter["stream"] != checkpoint.counter["stream"]
                    or counter["end"] > checkpoint.counter["counter"]):
                raise NumericalQDStoreError("proposal counter range was not checkpointed before dispatch")
            payload = envelope["batch"]

        value = _require_exact_schema(payload, ("provider", "resource_use", "failure_reason",
            "proposals", "source_sha256s", "attempts"), field="proposal attempt")
        proposals = tuple(MutationProposalV2.from_payload(item) for item in value["proposals"])
        sources = tuple((sha, self._source(sha)) for sha in value["source_sha256s"])
        attempts = []
        for item in value["attempts"]:
            item = _require_exact_schema(item, ("provider", "resource_use", "failure_reason"), field="provider attempt")
            attempts.append(ProviderAttemptV2(item["provider"], ResourceUse.from_payload(item["resource_use"]), item["failure_reason"]))
        return NormalizedProposalBatchV2(proposals, sources, ResourceUse.from_payload(value["resource_use"]),
            value["provider"], value["failure_reason"], tuple(attempts))

    def write_proposal_attempt(self, attempt_sha256, payload):
        self._verify_root_paths()
        require_sha256(attempt_sha256, "attempt SHA")
        payload = _payload(payload)
        self._proposal(payload)
        if fingerprint_payload(payload) != attempt_sha256:
            raise NumericalQDStoreError("proposal attempt content SHA mismatch")
        if "context" in payload:
            for path in self._safe(self.directory / "proposals").glob("*.json"):
                existing = self._read(f"proposals/{path.name}")
                if existing.get("context") == payload["context"] and existing != payload:
                    raise NumericalQDStoreError("one proposal context cannot have conflicting completed batches")
        return self._write(f"proposals/{attempt_sha256}.json", payload)

    def write_task_result(self, task_sha256, result):
        self._verify_root_paths()
        require_sha256(task_sha256, "task SHA")
        result = HyperbandTaskResultV2.from_payload(_payload(result))
        self.verify_candidate(result.candidate_sha256)
        self._verify_task_key(task_sha256, result)
        return self._write(f"results/{result.candidate_sha256}/{task_sha256}.json", result)

    @staticmethod
    def _verify_task_key(task_sha, result):
        if result.evaluation is not None:
            value = result.evaluation
            key = evaluation_cache_key(value.genome_sha256, task_sha, value.split_sha256,
                value.metric_policy_sha256, value.descriptor_policy_sha256,
                value.runtime_fingerprints, value.protocol_fingerprint, value.execution_adapter_sha256)
            if result.cache_key != key:
                raise NumericalQDStoreError("task result cache identity mismatch")

    def _verify_rung(self, rung):
        if self._object(rung.manifest.fingerprint()) != rung.manifest.to_payload():
            raise NumericalQDStoreError("rung fixed manifest mismatch")
        for evaluation in rung.evaluations:
            self.verify_candidate(evaluation.genome_sha256)
            for task in rung.manifest.tasks:
                result = HyperbandTaskResultV2.from_payload(self._read(
                    f"results/{evaluation.genome_sha256}/{task.task_sha256}.json"))
                if (result.candidate_sha256 != evaluation.genome_sha256 or result.task_id != task.task_id
                        or result.status != evaluation.task_statuses[task.task_id]):
                    raise NumericalQDStoreError("partial or mismatched rung task result")
                expected = evaluation_cache_key(evaluation.genome_sha256, task.task_sha256,
                    evaluation.split_sha256, evaluation.metric_policy_sha256, evaluation.descriptor_policy_sha256,
                    evaluation.runtime_fingerprints, evaluation.protocol_fingerprint, evaluation.execution_adapter_sha256)
                if result.cache_key != expected:
                    raise NumericalQDStoreError("rung task result identity mismatch")

    def write_rung(self, rung):
        self._verify_root_paths()
        rung = HyperbandRungV2.from_payload(_payload(rung))
        self._verify_rung(rung)
        # This is the captured Task 6 payload; never ask a live clock/ledger for it.
        self.write_object(rung.budget_outcome.ledger_checkpoint_sha256,
                          rung.budget_outcome.to_payload()["ledger_checkpoint"])
        for evaluation in rung.evaluations:
            self.write_object(evaluation.fingerprint(), evaluation)
        return self._write(f"rungs/{rung.fingerprint()}.json", rung)

    def _entries(self):
        path = self._path("archive/entries.jsonl")
        if not path.exists():
            return ()
        records, seen, previous = [], set(), None
        for raw in path.read_bytes().splitlines(keepends=True):
            row = _payload(raw)
            _require_exact_schema(row, ("entry", "previous_sha256", "record_sha256"), field="QD log record")
            entry = NumericalQDEntryV2.from_payload(row["entry"])
            body = {"entry": row["entry"], "previous_sha256": row["previous_sha256"]}
            if (row["previous_sha256"] != previous or fingerprint_payload(body) != row["record_sha256"]
                    or entry.fingerprint() in seen):
                raise NumericalQDStoreError("QD JSONL prefix/hash chain corruption")
            if self._object(entry.fingerprint()) != entry.to_payload():
                raise NumericalQDStoreError("QD entry immutable bytes mismatch")
            evaluation = NumericalEvaluationV2.from_payload(self._object(entry.evaluation_sha256))
            if evaluation.genome_sha256 != entry.genome_sha256 or not set(entry.task_ids) <= set(evaluation.task_ids):
                raise NumericalQDStoreError("QD entry evaluation binding mismatch")
            self.verify_candidate(entry.genome_sha256)
            seen.add(entry.fingerprint())
            records.append(entry)
            previous = row["record_sha256"]
        return tuple(records)

    def append_qd_entry(self, entry):
        self._verify_root_paths()
        entry = NumericalQDEntryV2.from_payload(_payload(entry))
        existing = self._entries()
        path = self._path("archive/entries.jsonl")
        if any(value.fingerprint() == entry.fingerprint() for value in existing):
            return path
        self.verify_candidate(entry.genome_sha256)
        evaluation = NumericalEvaluationV2.from_payload(self._object(entry.evaluation_sha256))
        if evaluation.genome_sha256 != entry.genome_sha256 or not set(entry.task_ids) <= set(evaluation.task_ids):
            raise NumericalQDStoreError("QD entry evaluation binding mismatch")
        self.write_object(entry.fingerprint(), entry)
        previous = None
        if existing:
            previous = _payload(path.read_bytes().splitlines(keepends=True)[-1])["record_sha256"]
        body = {"entry": entry.to_payload(), "previous_sha256": previous}
        durable.append_jsonl(path, body | {"record_sha256": fingerprint_payload(body)})
        durable._fsync_directory(path.parent)
        if self._entries() != (*existing, entry):
            raise NumericalQDStoreError("QD log append readback mismatch")
        return path

    def _catalog(self, completed_operations=None):
        if not self._safe(self.directory).is_dir():
            raise NumericalQDStoreError("missing Numerical QD directory")
        result, directories = {}, set()
        for path in self.directory.rglob("*"):
            self._safe(path)
            if path.is_dir():
                relative = path.relative_to(self.directory).as_posix()
                if relative not in _LAYOUT and not re.fullmatch(rf"results/{_SHA}", relative):
                    raise NumericalQDStoreError("unexpected artifact directory")
                directories.add(relative)
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise NumericalQDStoreError("artifact must be a regular file")
            name = path.relative_to(self.directory).as_posix()
            if _TEMP.fullmatch(path.name):
                continue
            if name == "checkpoint.json":
                continue
            self._path(name)
            data = path.read_bytes()
            if path.suffix == ".json":
                payload = _payload(data)
                if name.startswith(("objects/", "rungs/", "proposals/")) and _identity(payload) != path.stem:
                    raise NumericalQDStoreError("content-addressed file SHA mismatch")
            elif path.suffix == ".py" and hashlib.sha256(data).hexdigest() != path.stem:
                raise NumericalQDStoreError("source file SHA mismatch")
            result[name] = hashlib.sha256(data).hexdigest()
        operations = result if completed_operations is None else completed_operations
        expected_directories = set(_LAYOUT) | {
            str(Path(name).parent) for name in operations if name.startswith("results/")
        }
        if directories != expected_directories:
            raise NumericalQDStoreError("missing or unreferenced Numerical QD directory")
        return dict(sorted(result.items()))

    def _kernel(self, expected):
        require_sha256(expected, "Kernel checkpoint SHA")
        try:
            payload = _payload(self._safe(self.root / "checkpoint.json").read_bytes())
        except OSError as error:
            raise NumericalQDStoreError("missing Kernel checkpoint") from error
        _require_exact_schema(payload, EvolutionKernel._CHECKPOINT_FIELDS, field="Kernel checkpoint")
        if (_identity(payload) != expected or payload["schema_version"] != 2
                or payload["pending_publication"] is not None or payload["terminal_recovery"] is not None
                or payload["budget"]["open_reservations"]):
            raise NumericalQDStoreError("Kernel checkpoint mismatch or incomplete operation")
        return payload

    def _verify_state(self, checkpoint):
        if self._read("manifest.json") != _MANIFEST:
            raise NumericalQDStoreError("Numerical QD manifest mismatch")
        durable.V2RunStore._require_v2_manifest(self.root)
        kernel = self._kernel(checkpoint.kernel_checkpoint_sha256)
        if self._object(checkpoint.kernel_checkpoint_sha256) != kernel:
            raise NumericalQDStoreError("Kernel immutable checkpoint mismatch")
        bundle = EvolutionBundleV2.from_payload(self._object(checkpoint.active_bundle_sha256))
        if bundle.fingerprint() != kernel["active_bundle_sha256"]:
            raise NumericalQDStoreError("Kernel active Bundle mismatch")
        self.verify_candidate(checkpoint.active_genome_sha256)
        NumericalMutationPolicyV2.from_payload(self._object(checkpoint.mutation_policy_sha256))
        self._verify_prompt(checkpoint.proposer_prompt_sha256)
        config = NumericalQDConfigV2.from_payload(self._object(checkpoint.config_sha256))
        if config.seed != checkpoint.counter["seed"]:
            raise NumericalQDStoreError("RNG seed differs from committed config")
        manifest = _payload(self._safe(self.root / "run_manifest.json").read_bytes())
        if (config.kernel_protocol.to_payload() != manifest.get("kernel_protocol")
                or config.kernel_protocol.fingerprint() != bundle.protocol_fingerprint):
            raise NumericalQDStoreError("config protocol must match the Kernel manifest and active Bundle")
        # Input bytes may be external operator files; their commitments are
        # compared with the caller on resume, never resolved as filesystem paths.
        budget = self._object(checkpoint.budget_checkpoint_sha256)
        if (checkpoint.budget_checkpoint_sha256 != kernel["budget"]["checkpoint_sha256"]
                or canonical_v2_bytes(budget) != canonical_v2_bytes(kernel["budget"])):
            raise NumericalQDStoreError("runner Budget must exactly match the Kernel-owned checkpoint")
        try:
            plan_payload = _payload(self._safe(self.root / "budget_plan.json").read_bytes())
            plan = BudgetPlan(plan_payload["hard_limit_seconds"], plan_payload["finalization_reserve_fraction"],
                              ResourceUse.from_payload(plan_payload["ceilings"]))
            if plan.to_payload() != plan_payload or config.budget != plan:
                raise NumericalQDStoreError("Budget plan exact schema mismatch")
        except OSError as error:
            raise NumericalQDStoreError("missing Budget plan") from error
        BudgetLedger.resume(plan, budget, monotonic=lambda: 0.0)
        if budget["open_reservations"]:
            raise NumericalQDStoreError("open Budget reservation")
        # _kernel has already rejected pending/terminal recovery, so Kernel
        # resume only verifies its immutable graph and committed active pointer.
        EvolutionKernel.resume(durable.V2RunStore(self.root), plan, monotonic=lambda: 0.0)
        state = HyperbandStateV2.from_payload(self._object(checkpoint.hyperband_state_sha256))
        for candidate in state.candidate_sha256s:
            self.verify_candidate(candidate)
        catalog = checkpoint.completed_operation_sha256s
        all_rungs, manifest_shas, task_paths = {}, set(), set()
        for path in catalog:
            if path.startswith("proposals/"):
                self._proposal(self._read(path))
            if not path.startswith("rungs/"):
                continue
            rung = HyperbandRungV2.from_payload(self._read(path))
            self._verify_rung(rung)
            all_rungs[rung.fingerprint()] = rung
            manifest_shas.add(rung.manifest.fingerprint())
            if self._object(rung.budget_outcome.ledger_checkpoint_sha256) != rung.budget_outcome.to_payload()["ledger_checkpoint"]:
                raise NumericalQDStoreError("rung Budget checkpoint mismatch")
            if not set(rung.budget_outcome.ledger_checkpoint["closed_reservation_sha256s"]) <= set(budget["closed_reservation_sha256s"]):
                raise NumericalQDStoreError("Budget checkpoint lost a rung closure")
            captured = rung.budget_outcome.ledger_checkpoint
            if (captured["prior_elapsed_wall_seconds"] > budget["prior_elapsed_wall_seconds"]
                    or any(captured["charged_use"][name] > budget["charged_use"][name]
                           for name in ResourceUse.field_names())):
                raise NumericalQDStoreError("Budget checkpoint rewinds a completed rung charge")
            for value in rung.evaluations:
                if self._object(value.fingerprint()) != value.to_payload():
                    raise NumericalQDStoreError("rung evaluation immutable mismatch")
                task_paths.update(f"results/{value.genome_sha256}/{task.task_sha256}.json" for task in rung.manifest.tasks)
        for rung in state.rungs:
            if rung.fingerprint() not in all_rungs:
                raise NumericalQDStoreError("missing immutable completed rung")
        represented_rungs = set()
        for path in catalog:
            if path.startswith("results/") and path not in task_paths:
                raise NumericalQDStoreError("partial uncommitted task result")
            if path.startswith("objects/"):
                value = self._read(path)
                if "task_groups" in value and Path(path).stem not in manifest_shas:
                    raise NumericalQDStoreError("open or partial fixed rung manifest")
                if "candidate_sha256s" in value and "rungs" in value:
                    persisted = HyperbandStateV2.from_payload(value)
                    represented_rungs.update(rung.fingerprint() for rung in persisted.rungs)
                    if (persisted.bracket == state.bracket and persisted.candidate_sha256s == state.candidate_sha256s
                            and len(persisted.rungs) > len(state.rungs)):
                        raise NumericalQDStoreError("checkpoint rewinds a completed Hyperband rung")
        if set(all_rungs) != represented_rungs:
            raise NumericalQDStoreError("rung lacks a completed immutable Hyperband state")
        archive = NumericalQDArchive.from_payload(self._object(checkpoint.qd_snapshot_sha256))
        entries = self._entries()
        if {value.fingerprint(): value for value in entries} != dict(archive.entries):
            raise NumericalQDStoreError("QD snapshot disagrees with durable log")
        # Replay the insertion grouping, which is itself authenticated by the
        # typed snapshot; JSONL order must follow that same committed grouping.
        ordered = tuple(sha for batch in archive.insertion_log for sha in batch.entry_sha256s)
        if tuple(value.fingerprint() for value in entries) != ordered:
            raise NumericalQDStoreError("QD snapshot insertion order mismatch")

    def write_state(self, **state):
        """Seal a complete inventory, then atomically publish its runner pointer."""
        self._verify_root_paths()
        if self._safe(self.root / "evaluation_complete.json").exists():
            raise NumericalQDStoreError("completed run cannot publish another state")
        previous_path = self._path("checkpoint.json")
        previous = None
        if previous_path.exists():
            previous = NumericalQDCheckpointV2.from_payload(self._read("checkpoint.json"))
            if self._object(previous.checkpoint_sha256) != previous.to_payload():
                raise NumericalQDStoreError("previous runner checkpoint changed")
            old_counter, counter = previous.counter, state["counter"]
            if (counter["seed"] != old_counter["seed"] or counter["stream"] != old_counter["stream"]
                    or counter["counter"] < old_counter["counter"]):
                raise NumericalQDStoreError("RNG counter drift")
            if (state["config_sha256"] != previous.config_sha256
                    or state["input_sha256s"] != dict(previous.input_sha256s)):
                raise NumericalQDStoreError("run config/input identities changed")
            # A later publication cannot bless changed immutable prefix bytes.
            for name, expected in previous.completed_operation_sha256s.items():
                if name == "archive/entries.jsonl":
                    prefixes, digest = set(), hashlib.sha256()
                    for line in self._path(name).read_bytes().splitlines(keepends=True):
                        digest.update(line)
                        prefixes.add(digest.hexdigest())
                    if expected not in prefixes:
                        raise NumericalQDStoreError("QD log lost its committed prefix")
                    continue
                if hashlib.sha256(self._path(name).read_bytes()).hexdigest() != expected:
                    raise NumericalQDStoreError("immutable operation prefix changed")
        kernel = self._kernel(state["kernel_checkpoint_sha256"])
        self.write_object(state["kernel_checkpoint_sha256"], kernel)
        catalog = self._catalog()
        published = set(previous.completed_operation_sha256s) if previous else set()
        if previous is not None:
            published.add(f"objects/{previous.checkpoint_sha256}.json")
        for name in set(catalog) - published:
            if name.startswith("objects/") and "completed_operation_sha256s" in self._read(name):
                raise NumericalQDStoreError("unpublished immutable runner checkpoint requires a new epoch")
        if previous is not None and all(previous.to_payload().get(key) == value for key, value in state.items()):
            expected = dict(previous.completed_operation_sha256s)
            expected[f"objects/{previous.checkpoint_sha256}.json"] = hashlib.sha256(previous.canonical_bytes()).hexdigest()
            if catalog == expected:
                self._verify_state(previous)
                return previous
        checkpoint = NumericalQDCheckpointV2.seal(schema_version=1,
            completed_operation_sha256s=catalog, **state)
        self._verify_state(checkpoint)
        self.write_object(checkpoint.checkpoint_sha256, checkpoint)
        durable.write_atomic_json(previous_path, checkpoint.to_payload())
        if self._read("checkpoint.json") != checkpoint.to_payload():
            raise NumericalQDStoreError("runner checkpoint canonical readback failed")
        return checkpoint

    def resume(self, config_sha256, input_sha256s, kernel_checkpoint_sha256, budget_checkpoint_sha256):
        self._verify_root_paths(require_checkpoint=True)
        checkpoint = NumericalQDCheckpointV2.from_payload(self._read("checkpoint.json"))
        for name, expected in (("config_sha256", config_sha256), ("input_sha256s", input_sha256s),
                               ("kernel_checkpoint_sha256", kernel_checkpoint_sha256),
                               ("budget_checkpoint_sha256", budget_checkpoint_sha256)):
            if getattr(checkpoint, name) != expected:
                raise NumericalQDStoreError(f"resume {name} mismatch")
        if self._object(checkpoint.checkpoint_sha256) != checkpoint.to_payload():
            raise NumericalQDStoreError("runner checkpoint immutable bytes mismatch")
        expected = dict(checkpoint.completed_operation_sha256s)
        expected[f"objects/{checkpoint.checkpoint_sha256}.json"] = hashlib.sha256(checkpoint.canonical_bytes()).hexdigest()
        if self._catalog(expected) != expected:
            raise NumericalQDStoreError("missing, changed, or partial immutable operations")
        self._verify_state(checkpoint)
        completion = self._safe(self.root / "evaluation_complete.json")
        if completion.exists():
            self._verify_completion(_payload(completion.read_bytes()), self._completion_payload(checkpoint))
        return checkpoint

    def _completion_payload(self, checkpoint):
        """Derive completion facts solely from already verified persisted state."""
        kernel = self._kernel(checkpoint.kernel_checkpoint_sha256)
        if not kernel["budget"]["finalization_started"]:
            raise NumericalQDStoreError("completion requires Kernel finalization")
        bundle = EvolutionBundleV2.from_payload(self._object(checkpoint.active_bundle_sha256))
        archive = NumericalQDArchive.from_payload(self._object(checkpoint.qd_snapshot_sha256))
        attempts, proposal_use = [], ResourceUse()
        for name in sorted(checkpoint.completed_operation_sha256s):
            if name.startswith("proposals/"):
                batch = self._proposal(self._read(name))
                attempts.extend(batch.attempts)
                proposal_use += batch.resource_use
        charged = ResourceUse.from_payload(kernel["budget"]["charged_use"])
        if any(getattr(proposal_use, name) > getattr(charged, name) for name in ResourceUse.field_names()):
            raise NumericalQDStoreError("completion proposal use exceeds Kernel-owned accounting")
        transitions = tuple(kernel["completed_transitions"].values())
        dev_accessed = False
        for transition in transitions:
            evidence_sha = require_sha256(transition["acceptance_evidence_sha256"], "acceptance evidence SHA")
            evidence = _payload(self._safe(self.root / "acceptance" / f"{evidence_sha}.json").read_bytes())
            if fingerprint_payload(evidence) != evidence_sha:
                raise NumericalQDStoreError("completion evidence content SHA mismatch")
            comparison = evidence["dev_comparison"]
            dev_accessed |= bool(comparison["parent_metrics"] or comparison["candidate_metrics"])
        return {
            "schema_version": 1, "status": "numerical_qd_complete",
            "runner_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "kernel_checkpoint_sha256": checkpoint.kernel_checkpoint_sha256,
            "budget_checkpoint_sha256": checkpoint.budget_checkpoint_sha256,
            "active_bundle_sha256": checkpoint.active_bundle_sha256,
            "active_genome_sha256": checkpoint.active_genome_sha256,
            "qd_snapshot_sha256": checkpoint.qd_snapshot_sha256,
            "summary": {
                "supply_sha256": bundle.numerical_release_sha256,
                "registry_sha256": bundle.numerical_registry_sha256,
                "bundle_sha256": bundle.fingerprint(),
                "qd_snapshot_sha256": checkpoint.qd_snapshot_sha256,
                "mutation_policy_sha256": checkpoint.mutation_policy_sha256,
                "proposer_prompt_sha256": checkpoint.proposer_prompt_sha256,
                "occupied_cells": len(archive.cells),
                "accepted_count": sum(value["decision"] == "accept" for value in transitions),
                "rejected_count": sum(value["decision"] == "reject" for value in transitions),
                "provider_attempts": len(attempts),
                "llm_attempts": sum(attempt.provider == "llm" for attempt in attempts),
                "llm_calls": proposal_use.llm_calls,
                "llm_provider_used": proposal_use.llm_calls > 0,
                "budget": kernel["budget"],
                "dev_accessed": dev_accessed,
                "public_test_accessed": False,
            },
        }

    @staticmethod
    def _verify_completion(payload, expected):
        _require_exact_schema(payload, tuple(expected), field="Numerical QD completion")
        _require_exact_schema(payload["summary"], tuple(expected["summary"]), field="completion summary")
        # Canonical comparison distinguishes boolean/count aliases as well as
        # missing fields, invented claims and changed identities or metrics.
        if canonical_v2_bytes(payload) != canonical_v2_bytes(expected):
            raise NumericalQDStoreError("completion conflicts with verified finalized authority")

    def write_completion(self, payload):
        """Accept status-only convenience or the exact derived closed envelope."""
        self._verify_root_paths(require_checkpoint=True)
        checkpoint = NumericalQDCheckpointV2.from_payload(self._read("checkpoint.json"))
        self.resume(checkpoint.config_sha256, dict(checkpoint.input_sha256s),
                    checkpoint.kernel_checkpoint_sha256, checkpoint.budget_checkpoint_sha256)
        expected = self._completion_payload(checkpoint)
        payload = _payload(payload)
        if payload != {"status": "numerical_qd_complete"}:
            self._verify_completion(payload, expected)
        path = self._safe(self.root / "evaluation_complete.json")
        durable.write_once_json(path, expected)
        self._verify_completion(_payload(path.read_bytes()), expected)
        return path


__all__ = ["NumericalQDRunStore", "NumericalQDStoreError"]
