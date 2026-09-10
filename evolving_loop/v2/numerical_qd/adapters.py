"""Host adapters for the existing Numerical parser, runtime, and frozen Supply.

Candidate code receives history only. Host fitting may use Train labels; Host
evaluation emits the closed V2 aggregate. No legacy source or release is edited.
"""
from __future__ import annotations

import ast
import hashlib
import math
import statistics
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType

from common.metrics import drcik_point_metrics, joint_scaled_error, linear_quantile
from common.sandbox import check_code
from evolving_loop.data import ContextTask
from evolving_loop.numerical_two_stage import numerical_package_fingerprint
from evolving_loop.package_metrics import PackageEvaluation
from evolving_loop.package_numerical_evolution import (
    NumericalCoordinateCandidate, NumericalPackageMaterializer, NumericalPackageProposer,
    NumericalRecipeFit, _sanitized_context_task, fit_numerical_recipe,
)
from evolving_loop.package_numerical_supply import (
    NumericalSupplyRelease, bound_numerical_package, build_package_registry,
    parse_numerical_supply_release,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry, task_registry_fingerprint
from numerical_agent.evolution.champion import ChampionRecipe, parse_champion_recipe, parse_champion_release
from numerical_agent.evolution.champion_evidence import ChampionTaskRow
from numerical_agent.evolution.execution import IsolatedForecastRuntime, MethodForecastError, Task as RuntimeTask
from numerical_agent.evolution.module import EVOLUTION_IMPORTS, EVOLUTION_DUNDERS, MODULE_HEADER, MethodModule, parse_module
from numerical_agent.evolution.morphology import (
    AssumptionGrounding, MorphologyCard, MorphologyObservation, MorphologyToolCall,
)
from numerical_agent.evolution.numerical_package import (
    NumericalForecastPackage, RankedNumericalForecast,
    _ChampionNumericalForecastPackage, _TaskLocalNumericalForecastPackage,
)
from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics, HindcastFold, SelectionArithmetic, SelectionDecision,
)
from numerical_agent.evolution.screening import TaskProfile, profile_task
from numerical_agent.evolution.task_local_evolution import GroupFoldManifest

from ..budget import ResourceUse
from ..contracts import _require_exact_schema, canonical_v2_bytes, fingerprint_payload, require_sha256
from .contracts import (
    MEMBER_FAMILIES, ConstraintReportV2, FrozenNumericalPackageEnvelopeV2,
    FrozenNumericalRegistryEnvelopeV2, MutationStateV2, NumericalEvaluationV2,
    NumericalGenomeV2, NumericalMemberV2, NumericalObjectiveVectorV2, NumericalSourceOutcomeV2,
    RungManifestV2, TrainTaskV2,
)
from .descriptors import DescriptorPolicyV2, describe_history
from .map_elites import NumericalQDArchive
from .mutation import MutationProposalV2, apply_mutation
from .nsga2 import crowding_distances, non_dominated_fronts


# Explicit constructors only: neither a module name nor a class supplied by an
# artifact is ever imported, evaluated, or unpickled.
_PACKAGE_TYPES = {cls.__name__: cls for cls in (
    NumericalForecastPackage, _ChampionNumericalForecastPackage, _TaskLocalNumericalForecastPackage,
    RankedNumericalForecast, TaskProfile, CandidateDiagnostics, HindcastFold,
    SelectionDecision, SelectionArithmetic, AssumptionGrounding, MorphologyCard,
    MorphologyToolCall, MorphologyObservation,
)}

# The legacy parser is necessary but not an OS sandbox. V2 additionally limits
# imported capabilities and attribute access to a closed, numerical-only surface.
# Checking references (not just calls) also rejects aliases/callbacks to file IO.
_SOURCE_IMPORTS = frozenset({"math", "statistics", "itertools", "functools", "collections",
                              "numpy", "numpy.linalg", "numpy.fft"})
_SOURCE_ATTRIBUTES = frozenset("""
    abs absolute accumulate acos acosh add all allclose amax amin angle any append
    arange arccos arccosh arcsin arcsinh arctan arctan2 arctanh argmax argmin argsort
    around array array_equal asarray asin asinh astype atan atan2 atanh average
    bincount bool_ broadcast_to cbrt ceil chain clip column_stack combinations comb
    concatenate conj conjugate convolve copy corrcoef cos cosh count count_nonzero
    Counter cov cumprod cumsum cycle degrees deque diagonal diff digitize divmod dot
    dtype e eig eigh eigvals eigvalsh einsum empty empty_like enumerate erf erfc exp
    exp2 expand_dims expm1 fabs factorial fft fftfreq fftshift fill finfo flat flatten
    float32 float64 floor fmax fmean fmin fmod frexp fromiter fromkeys fsum full
    full_like gamma gcd geomspace get gradient groupby harmonic_mean hstack hypot
    identity ifft ifftshift imag inf inner int32 int64 interp inv irfft isclose
    isfinite isinf isnan islice items keys ldexp lgamma linalg linspace log log10
    log1p log2 logaddexp logical_and logical_not logical_or logspace lstsq matmul
    max maximum mean median min minimum mod mode moveaxis nan nan_to_num nanargmax
    nanargmin nanmax nanmean nanmedian nanmin nanpercentile nanquantile nanstd nansum
    nanvar ndim negative newaxis nextafter norm ones ones_like outer pad partition
    percentile permutations pi pinv polyfit polyval power prod product pstdev pvariance
    quantile quantiles radians ravel real reciprocal reduce repeat reshape resize
    rfft rfftfreq roll round row_stack searchsorted shape sign sin sinh size solve
    sort split sqrt square squeeze stack starmap std stdev subtract sum svd swapaxes
    take tan tanh tau tensordot tile tolist trace transpose trapz trapezoid tri tril
    triu trunc tuple unique values var variance vdot vstack where zip_longest zeros
    zeros_like T
""".split())
_SOURCE_REFLECTION = frozenset({"getattr", "setattr", "delattr", "hasattr", "vars", "globals", "locals",
                                 "dir", "type", "object", "super", "memoryview", "help", "print"})


def _check_numerical_capabilities(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name not in _SOURCE_IMPORTS for alias in node.names):
                raise ValueError("source import is outside the closed numerical capabilities")
        elif isinstance(node, ast.ImportFrom):
            if (node.level or node.module not in _SOURCE_IMPORTS
                    or any(alias.name not in _SOURCE_ATTRIBUTES for alias in node.names)):
                raise ValueError("source import is outside the closed numerical capabilities")
        elif isinstance(node, ast.Attribute) and node.attr not in _SOURCE_ATTRIBUTES:
            raise ValueError("source attribute is outside the closed numerical capabilities")
        elif isinstance(node, ast.Name) and (node.id in _SOURCE_REFLECTION or node.id.startswith("__")):
            raise ValueError("source reflection is not a numerical capability")


def _encode(value):
    if type(value).__name__ in _PACKAGE_TYPES and _PACKAGE_TYPES[type(value).__name__] is type(value):
        return {"type": type(value).__name__, "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("package mappings require string keys")
        return {"type": "mapping", "items": {key: _encode(item) for key, item in value.items()}}
    if type(value) in (tuple, list):
        return [_encode(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        # Legacy diagnostics use infinity for unavailable scores. JSON itself
        # remains finite, and forecast constructors still reject nonfinite data.
        if math.isnan(value):
            raise ValueError("NaN is not a canonical package diagnostic")
        return {"type": "nonfinite", "value": "inf" if value > 0 else "-inf"}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("unsupported package value type")


def _decode(value):
    if type(value) is list:
        return tuple(_decode(item) for item in value)
    if type(value) is dict:
        kind = value.get("type")
        if kind == "mapping":
            _require_exact_schema(value, ("type", "items"), field="package mapping")
            if type(value["items"]) is not dict:
                raise ValueError("package mapping items must be an object")
            return {key: _decode(item) for key, item in value["items"].items()}
        if kind == "nonfinite":
            _require_exact_schema(value, ("type", "value"), field="diagnostic infinity")
            if value["value"] not in ("inf", "-inf"):
                raise ValueError("unknown nonfinite diagnostic")
            return math.inf if value["value"] == "inf" else -math.inf
        if type(kind) is not str or kind not in _PACKAGE_TYPES:
            raise ValueError("unknown package artifact type")
        cls = _PACKAGE_TYPES[kind]
        _require_exact_schema(value, ("type", "fields"), field="package artifact")
        values = _require_exact_schema(value["fields"], tuple(f.name for f in fields(cls)), field=kind)
        return cls(**{key: _decode(item) for key, item in values.items()})
    if value is None or type(value) in (str, bool, int) or type(value) is float and math.isfinite(value):
        return value
    raise ValueError("package envelope must contain strict JSON values")


def _decode_package(payload):
    try:
        package = _decode(payload)
        if type(package) not in (NumericalForecastPackage, _ChampionNumericalForecastPackage, _TaskLocalNumericalForecastPackage):
            raise ValueError("envelope root must be a NumericalForecastPackage")
        if canonical_v2_bytes(_encode(package)) != canonical_v2_bytes(payload):
            raise ValueError("package envelope is not canonical")
        return package
    except (TypeError, AttributeError, KeyError, OverflowError) as error:
        raise ValueError("invalid typed package envelope") from error


def _tasks(tasks):
    values = tuple(tasks)
    if not values or any(type(task) is not ContextTask for task in values):
        raise ValueError("tasks must be exact ContextTask values")
    if len({task.numeric.task_id for task in values}) != len(values):
        raise ValueError("task identities must be unique")
    return tuple(sorted(values, key=lambda task: task.numeric.task_id))


def _envelope(registry, tasks):
    if type(registry) is not FrozenNumericalPackageRegistry:
        raise ValueError("exact FrozenNumericalPackageRegistry required")
    tasks = _tasks(tasks)
    if tuple(task.numeric.task_id for task in tasks) != registry.task_ids:
        raise ValueError("registry task universe mismatch")
    packages, entries = {}, {}
    for task in tasks:
        package = registry.package_for(task)
        artifact = FrozenNumericalPackageEnvelopeV2(1, numerical_package_fingerprint(package), _encode(package))
        sha = artifact.fingerprint()
        packages[sha] = artifact
        entries[task.numeric.task_id] = {"task_sha256": task_registry_fingerprint(task), "package_sha256": sha}
    return FrozenNumericalRegistryEnvelopeV2(1, registry.release_sha256, registry.fingerprint,
        entries, tuple(sorted(packages)), packages)


def _restore_registry(envelope, tasks):
    tasks = _tasks(tasks)
    if set(envelope.entries) != {task.numeric.task_id for task in tasks}:
        raise ValueError("restored task universe mismatch")
    entries = []
    for task in tasks:
        entry = envelope.entries[task.numeric.task_id]
        if task_registry_fingerprint(task) != entry["task_sha256"]:
            raise ValueError("restored host task SHA mismatch")
        entries.append((task, envelope.packages[entry["package_sha256"]].restore()))
    registry = FrozenNumericalPackageRegistry(entries, release_sha256=envelope.release_sha256,
        expected_task_ids=tuple(envelope.entries))
    if registry.fingerprint != envelope.registry_sha256:
        raise ValueError("restored registry content SHA mismatch")
    return registry


@dataclass(frozen=True)
class ImportedNumericalSeedV2:
    release: NumericalSupplyRelease
    envelope: FrozenNumericalRegistryEnvelopeV2
    sources: Mapping[str, str]

    def __post_init__(self):
        if type(self.release) is not NumericalSupplyRelease or type(self.envelope) is not FrozenNumericalRegistryEnvelopeV2:
            raise ValueError("seed requires exact release and envelope")
        if self.release.fingerprint != self.envelope.release_sha256:
            raise ValueError("seed release/registry mismatch")
        object.__setattr__(self, "sources", _sources(self.sources))


@dataclass(frozen=True)
class MaterializedNumericalChildV2:
    genome: NumericalGenomeV2
    state: MutationStateV2
    member: NumericalMemberV2
    candidate: NumericalCoordinateCandidate
    fit: NumericalRecipeFit
    descriptor_policy_sha256: str
    source_sha256s: tuple[str, ...]

    def __post_init__(self):
        for name, cls in (("genome", NumericalGenomeV2), ("state", MutationStateV2),
                          ("member", NumericalMemberV2), ("candidate", NumericalCoordinateCandidate), ("fit", NumericalRecipeFit)):
            if type(getattr(self, name)) is not cls:
                raise ValueError(f"child requires exact {name}")
        if self.genome.inventory_sha256 != self.state.inventory.fingerprint() or self.member not in self.state.inventory.members:
            raise ValueError("child inventory binding mismatch")
        if self.member != _canonical_member(self.state):
            raise ValueError("child must bind the canonical member")
        NumericalCoordinateCandidate.__post_init__(self.candidate)
        NumericalRecipeFit.__post_init__(self.fit)
        require_sha256(self.descriptor_policy_sha256, "descriptor_policy_sha256")
        if type(self.source_sha256s) is not tuple or self.source_sha256s != tuple(sorted(set(self.source_sha256s))):
            raise ValueError("source dependencies must be sorted unique SHAs")
        for sha in self.source_sha256s:
            require_sha256(sha, "source dependency")
        if self.member.source_sha256 not in self.source_sha256s:
            raise ValueError("source dependencies must include the selected member source")


@dataclass(frozen=True)
class FrozenNumericalArtifactsV2:
    release: NumericalSupplyRelease
    registry: FrozenNumericalPackageRegistry
    envelope: FrozenNumericalRegistryEnvelopeV2
    selected_genome_sha256s: tuple[str, ...]

    def __post_init__(self):
        if self.release.fingerprint != self.registry.release_sha256 or self.envelope.registry_sha256 != self.registry.fingerprint:
            raise ValueError("frozen Numerical artifacts disagree")


def _canonical_member(state):
    """One executable per committed inventory: specialization, ancestry, order.

    Inventory order is itself hashed. Prefer a specialized member, then a
    structural child with more parents, breaking ties by that committed order.
    No caller choice, score, task, or uncommitted policy affects this selection.
    """
    eligible = tuple(member for member in state.inventory.members if member.status != "quarantined")
    if not eligible:
        raise ValueError("inventory has no canonical member")
    return min(eligible, key=lambda member: (member.status != "specialized", -len(member.parent_ids)))


class _VerifiedForecastStore:
    """Execute selected source bytes; only the protected legacy anchor delegates."""

    def __init__(self, directory, sources, anchor_names, trusted_store):
        self.methods, self.runtimes, self.cache = {}, {}, {}
        self.anchor_names, self.trusted_store = set(anchor_names), trusted_store
        for sha, source in sorted(sources.items()):
            if hashlib.sha256(source.encode()).hexdigest() != sha:
                raise ValueError("executable source SHA mismatch")
            for name in LegacyNumericalAdapter.validate_source(source).names():
                if name in self.methods or name in self.anchor_names:
                    raise ValueError("ambiguous executable source or protected anchor identity")
                self.methods[name] = sha
        self.paths = {}
        for sha, source in sorted(sources.items()):
            path = Path(directory) / f"{sha}.py"
            path.write_text(MODULE_HEADER + "\n" + source, encoding="utf-8")
            self.paths[sha] = path

    def forecast(self, name, history, horizon, frequency):
        key = (name, tuple(history), horizon, frequency)
        if key not in self.cache:
            if name in self.methods:
                sha = self.methods[name]
                if sha not in self.runtimes:
                    self.runtimes[sha] = IsolatedForecastRuntime(self.paths[sha])
                value = self.runtimes[sha].forecast(name, history, horizon, frequency)
            elif name in self.anchor_names:
                value = self.trusted_store.forecast(name, history, horizon, frequency)
            else:
                raise MethodForecastError("method is outside the verified executable source closure")
            self.cache[key] = tuple(value)
        return self.cache[key]

    def close(self):
        for runtime in self.runtimes.values():
            runtime.close()


@contextmanager
def _source_bound_materializer(materializer, sources, anchor):
    if type(materializer) is not NumericalPackageMaterializer:
        raise ValueError("source binding requires an exact NumericalPackageMaterializer")
    with tempfile.TemporaryDirectory(prefix="numerical-qd-materialize-") as directory:
        store = _VerifiedForecastStore(directory, sources, anchor.policy.recipe.parents, materializer.forecast_store)
        try:
            # Reuse the legacy materializer without mutating the injected instance.
            # Old cached diagnostics may refer to different source bytes: recompute.
            bound = NumericalPackageMaterializer(forecast_store=store,
                screening_policy=materializer.screening_policy, fold_manifest=materializer.fold_manifest,
                original_tasks=materializer.original_tasks, source_fingerprints=materializer.source_fingerprints,
                runtime_fingerprints=materializer.runtime_fingerprints, combined_policies=materializer.combined_policies,
                atlas_release=materializer.atlas_release, decision_policy=materializer.decision_policy,
                hindcast_config=materializer.hindcast_config, diagnostics_registry=None)
            yield bound
        finally:
            store.close()


def _verified_build_rows(rows, tasks, manifest, recipe, anchor, store):
    hosts = {task.numeric.task_id: task.numeric for task in tasks
             if task.numeric.task_id in manifest.task_fold_map}
    supplied = tuple(rows)
    if not supplied or any(type(row) is not ChampionTaskRow for row in supplied):
        raise ValueError("fitting requires exact host Train rows")
    if {row.task_id for row in supplied} != set(hosts):
        raise ValueError("fitting requires the exact Train task universe")
    profiles = {name: profile_task(RuntimeTask(name, task.history_values, task.prediction_length,
                                               task.frequency, ())) for name, task in hosts.items()}
    for row in supplied:
        task = hosts[row.task_id]
        if (row.split != "build" or row.fold != manifest.task_fold_map[row.task_id]
                or row.truth != task.future_values or row.history != task.history_values
                or row.profile != profiles[row.task_id]):
            raise ValueError("fitting row does not match its trusted Train task")
    # Forecasts and diagnostics are not caller authority. Rebuild fitting rows
    # from the SAME byte-bound store used for full materialization below.
    names = sorted(set(recipe.parents) | set(anchor.policy.recipe.parents))
    return tuple(ChampionTaskRow(task_id=name, candidate_name=candidate, profile=profiles[name],
        truth=task.future_values, forecast=store.forecast(candidate, task.history_values,
            task.prediction_length, task.frequency), fold=manifest.task_fold_map[name],
        split="build", history=task.history_values)
        for name, task in sorted(hosts.items()) for candidate in names)


def _sources(sources):
    if not isinstance(sources, Mapping):
        raise ValueError("sources must be a SHA-to-text mapping")
    values = {}
    for sha, text in sources.items():
        require_sha256(sha, "source SHA")
        if type(text) is not str or hashlib.sha256(text.encode()).hexdigest() != sha:
            raise ValueError("source content SHA mismatch")
        LegacyNumericalAdapter.validate_source(text)
        values[sha] = text
    return MappingProxyType(values)


def import_numerical_seed(release, registry, *, tasks, source_paths=()) -> ImportedNumericalSeedV2:
    if type(release) is not NumericalSupplyRelease:
        raise ValueError("exact NumericalSupplyRelease required")
    release = parse_numerical_supply_release(release.to_payload())
    sources = {}
    for path in source_paths:
        data = Path(path).read_bytes()
        sources[hashlib.sha256(data).hexdigest()] = data.decode("utf-8")
    return ImportedNumericalSeedV2(release, _envelope(registry, tasks), sources)


class LegacyNumericalAdapter:
    """Trusted host boundary; injected materializers retain the legacy signature."""

    def __init__(self, *, materializer, tasks, fold_manifest, sources, proposer=None,
                 host_evaluator=None, host_evaluator_sha256=None):
        self.tasks = _tasks(tasks)
        if len(self.tasks) != 100 or type(fold_manifest) is not GroupFoldManifest or fold_manifest.fold_count != 5:
            raise ValueError("adapter requires exactly 80 grouped Train plus 20 Dev tasks")
        if len(fold_manifest.task_fold_map) != 80 or not set(fold_manifest.task_fold_map) <= {t.numeric.task_id for t in self.tasks}:
            raise ValueError("adapter requires the exact 80-task Train fold manifest")
        if not callable(getattr(materializer, "materialize", None)):
            raise ValueError("materializer must implement the legacy typed boundary")
        if type(materializer) is NumericalPackageMaterializer:
            if materializer.fold_manifest != fold_manifest or _tasks(materializer.original_tasks) != self.tasks:
                raise ValueError("materializer host-task binding mismatch")
        if proposer is not None and type(proposer) is not NumericalPackageProposer:
            raise ValueError("proposer must be an exact NumericalPackageProposer")
        if proposer is not None and (proposer.materializer is not materializer
                or proposer.fold_manifest != fold_manifest or _tasks(proposer.tasks) != self.tasks):
            raise ValueError("proposer must bind the adapter's materializer and tasks")
        if host_evaluator is not None and not callable(host_evaluator):
            raise ValueError("host_evaluator must implement the PackageEvaluation boundary")
        if host_evaluator is not None:
            require_sha256(host_evaluator_sha256, "host_evaluator_sha256")
        elif host_evaluator_sha256 is not None:
            raise ValueError("host evaluator identity requires a host evaluator")
        self.materializer, self.fold_manifest, self.proposer = materializer, fold_manifest, proposer
        self.host_evaluator = host_evaluator
        self.host_evaluator_sha256 = host_evaluator_sha256
        self.sources = _sources(sources)

    @property
    def candidate_tasks(self):
        return tuple(_sanitized_context_task(task) for task in self.tasks)

    @property
    def fingerprint(self):
        return fingerprint_payload({"implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    "host_evaluator": self.host_evaluator_sha256})

    def propose_legacy(self, release, registry, feedback, *, generation, task_evidence=None):
        """Retain the existing proposer, Train fitting, and three-child contract."""
        if self.proposer is None:
            raise ValueError("no legacy NumericalPackageProposer was supplied")
        return self.proposer.propose(release, registry, feedback, generation=generation,
            child_count=3, task_evidence=task_evidence)

    @staticmethod
    def validate_source(source: str) -> MethodModule:
        if type(source) is not str or not source or len(source.encode()) > 1_048_576:
            raise ValueError("source must be bounded nonempty text")
        check_code(source, EVOLUTION_IMPORTS, EVOLUTION_DUNDERS)
        module = parse_module(source)
        _check_numerical_capabilities(source)
        return module

    def forecast_source(self, source, method_name, task):
        self.validate_source(source)
        if type(task) is not ContextTask:
            raise ValueError("forecast requires a ContextTask")
        sha = hashlib.sha256(source.encode()).hexdigest()
        # Execute the verified bytes in a disposable path, never in a legacy path.
        with tempfile.TemporaryDirectory(prefix="numerical-qd-") as folder:
            path = Path(folder) / "methods.py"
            path.write_text(MODULE_HEADER + "\n" + source, encoding="utf-8")
            runtime = IsolatedForecastRuntime(path)
            try:
                forecast = runtime.forecast(method_name, task.numeric.history_values,
                    task.numeric.prediction_length, task.numeric.frequency)
            except MethodForecastError:
                return NumericalSourceOutcomeV2(sha, task.numeric.task_id, "invalid", (), "execution_failure")
            finally:
                runtime.close()
        return NumericalSourceOutcomeV2(sha, task.numeric.task_id, "passed", forecast, None)

    def _recipe(self, member: NumericalMemberV2, policies, parent_state: MutationStateV2 | None,
                proposal: MutationProposalV2 | None, _ancestors=(), _source_dependencies=None) -> ChampionRecipe:
        if member.source_sha256 not in self.sources:
            raise ValueError("member source is not verified")
        if _source_dependencies is not None:
            _source_dependencies.add(member.source_sha256)
        if member.policy_sha256 in _ancestors or len(_ancestors) >= 64:
            raise ValueError("cyclic or excessively deep policy ancestry")
        if member.policy_sha256 in policies:
            policy = policies[member.policy_sha256]
            if isinstance(policy, Mapping) and "kind" in policy:
                policy = parse_champion_recipe(dict(policy))
            if type(policy) is ChampionRecipe:
                if fingerprint_payload(policy.to_payload()) != member.policy_sha256:
                    raise ValueError("typed recipe policy SHA mismatch")
                if not set(policy.parents) <= set(self.validate_source(self.sources[member.source_sha256]).names()):
                    raise ValueError("recipe executable is absent from the verified member source")
                return parse_champion_recipe(policy.to_payload())
            policy = _require_exact_schema(policy, ("schema_version", "operator", "parents", "applicability_cells"), field="structural policy")
        else:
            if parent_state is None or proposal is None:
                raise ValueError("unrecognized or non-reconstructible policy digest")
            parents = tuple(next(m for m in parent_state.inventory.members if m.member_id == name) for name in member.parent_ids)
            policy = dict(schema_version=1, operator=proposal.operator,
                parents=[parent.to_payload() for parent in parents], applicability_cells=list(member.applicability_cells))
        operator = policy["operator"]
        if type(policy["schema_version"]) is not int or policy["schema_version"] != 1 or operator not in {"repair", "fork", "combine", "route", "crossover"}:
            raise ValueError("unrecognized structural policy schema/operator")
        parents = tuple(NumericalMemberV2.from_payload(value) for value in policy["parents"])
        expected_arity = 1 if operator in {"repair", "fork"} else 2
        if (fingerprint_payload(policy) != member.policy_sha256 or len(parents) != expected_arity
                or member.source_sha256 != parents[0].source_sha256
                or member.parent_ids != tuple(parent.member_id for parent in parents)
                or not set(member.applicability_cells) <= set(policy["applicability_cells"])):
            raise ValueError("structural policy SHA or reused source mismatch")
        recipes = tuple(self._recipe(parent, policies, None, None, (*_ancestors, member.policy_sha256), _source_dependencies) for parent in parents)
        if operator in {"repair", "fork"}:
            return replace(recipes[0], name=member.member_id)
        if len(recipes) != 2 or any(recipe.kind != "select" for recipe in recipes):
            raise ValueError("structural policy requires two reconstructible select parents")
        kind = "route" if operator == "route" else "weighted"
        names = tuple(recipe.parents[0] for recipe in recipes)
        # The route condition belongs to the first parent; the second is its
        # explicit fallback. Train-only fitting chooses the existing threshold grid.
        assumptions = tuple(replace(a, operator=kind) for a in recipes[0].assumptions)
        return ChampionRecipe(member.member_id, kind, names, names[1], assumptions)

    def materialize_child(self, parent_release, genome, state, *, member_id, policies,
                          build_rows, descriptor_policy: DescriptorPolicyV2, version,
                          parent_state=None, proposal=None) -> MaterializedNumericalChildV2:
        if type(genome) is not NumericalGenomeV2 or type(state) is not MutationStateV2:
            raise ValueError("typed genome and state required")
        if (genome.inventory_sha256 != state.inventory.fingerprint()
                or genome.mutation_policy_sha256 != state.mutation_policy.fingerprint()
                or genome.proposer_prompt_sha256 != state.proposer_prompt.fingerprint()):
            raise ValueError("genome/state policy binding mismatch")
        if type(self.materializer) is NumericalPackageMaterializer:
            if dict(genome.runtime_fingerprints) != self.materializer.runtime_fingerprints:
                raise ValueError("genome/runtime fingerprint mismatch")
            if genome.screening_policy_sha256 != self.materializer.screening_policy.fingerprint():
                raise ValueError("genome/screening policy fingerprint mismatch")
            combined_sha = fingerprint_payload({"policies": [policy.to_payload() for policy in self.materializer.combined_policies]})
            if genome.combined_policy_sha256 != combined_sha:
                raise ValueError("genome/combined policy fingerprint mismatch")
        if (parent_state is None) != (proposal is None):
            raise ValueError("structural reconstruction requires Parent and proposal")
        if proposal is not None and apply_mutation(parent_state, proposal).state != state:
            raise ValueError("state does not match Parent plus typed proposal")
        member = _canonical_member(state)
        if member.member_id != member_id:
            raise ValueError("materialization requires the canonical member for this genome")
        source_dependencies = set()
        recipe = self._recipe(member, policies, parent_state, proposal, _source_dependencies=source_dependencies)
        anchor = parse_champion_release(parent_release.to_payload()["anchor_release_payload"])
        with _source_bound_materializer(self.materializer,
                {sha: self.sources[sha] for sha in source_dependencies}, anchor) as materializer:
            rows = _verified_build_rows(build_rows, self.tasks, self.fold_manifest, recipe, anchor,
                                        materializer.forecast_store)
            fit = fit_numerical_recipe(recipe, rows, self.fold_manifest, anchor)
            candidate = materializer.materialize(parent_release, fit, self.candidate_tasks,
                version=version, generation=genome.generation)
        if type(candidate) is not NumericalCoordinateCandidate or candidate.invalid_reason is not None:
            raise ValueError("materializer did not return a valid typed candidate")
        if candidate.registry.task_ids != tuple(task.numeric.task_id for task in self.tasks):
            raise ValueError("materialized task coverage mismatch")
        sources = dict(candidate.release.source_fingerprints) | {
            "genome": genome.fingerprint(), "inventory": state.inventory.fingerprint(),
            "screening_policy": genome.screening_policy_sha256, "combined_policy": genome.combined_policy_sha256,
            "mutation_policy": genome.mutation_policy_sha256, "proposer_prompt": genome.proposer_prompt_sha256,
            "member_source": member.source_sha256, "member_policy": member.policy_sha256,
            "source_dependencies": fingerprint_payload({"sha256s": sorted(source_dependencies)}),
            "translated_recipe": fingerprint_payload(recipe.to_payload()), "descriptor_policy": descriptor_policy.fingerprint(),
        }
        release = replace(candidate.release, source_fingerprints=sources,
            anchor_release_payload=candidate.release.to_payload()["anchor_release_payload"])
        def builder(task, supplied):
            source = candidate.registry.package_for(task)
            available = {item.name: item for item in source.ranked_alternatives}
            if not _applicable(member, task, descriptor_policy):
                available.pop(recipe.name, None)
            return bound_numerical_package(source, supplied, available, history=task.numeric.history_values,
                task_fold=self.fold_manifest.task_fold_map.get(task.numeric.task_id))
        registry = build_package_registry(self.tasks, release, builder)
        result = NumericalCoordinateCandidate(release, registry, candidate.proposal_sha256)
        return MaterializedNumericalChildV2(genome, state, member, result, fit, descriptor_policy.fingerprint(),
            tuple(sorted(source_dependencies)))


def _applicable(member, task, policy):
    # A structural combined child inherits the parents' morphology cells, whose
    # family axis can differ. Match the five history axes in any registered family.
    return any(describe_history(task.numeric.history_values, task.numeric.prediction_length,
        task.numeric.frequency, family, policy).fingerprint() in member.applicability_cells for family in MEMBER_FAMILIES)


def evaluate_numerical_child(adapter, child, manifest, *, descriptor_policy: DescriptorPolicyV2, metric_policy_sha256,
                             bracket, rung, normalized_execution_cost=0.0, metric_cap=5.0,
                             resource_use=None) -> NumericalEvaluationV2:
    """Score this member's eligible forecast (or safe anchor) on committed Train only."""
    if type(child) is not MaterializedNumericalChildV2 or type(manifest) not in (RungManifestV2, TrainTaskV2):
        raise ValueError("typed materialized child and Train manifest required")
    if child.descriptor_policy_sha256 != descriptor_policy.fingerprint():
        raise ValueError("evaluation descriptor policy mismatch")
    if manifest.protocol_sha256 != child.genome.protocol_fingerprint:
        raise ValueError("evaluation protocol mismatch")
    hosts = {task.numeric.task_id: task for task in adapter.tasks}
    identities = manifest.tasks if type(manifest) is RungManifestV2 else (manifest,)
    task_ids = tuple(sorted(identity.task_id for identity in identities))
    scores, cells = [], {}
    for identity in identities:
        if identity.task_id not in adapter.fold_manifest.task_fold_map:
            raise ValueError("Dev tasks cannot enter Numerical QD evaluation")
        task = hosts[identity.task_id]
        if task_registry_fingerprint(task) != identity.task_sha256 or task.numeric.entity_name != identity.entity_id:
            raise ValueError("Train task content or group mismatch")
        package = child.candidate.registry.package_for(task)
        target = next((item for item in package.ranked_alternatives if item.name == child.fit.recipe.name), package.protected_baseline)
        if adapter.host_evaluator is None:
            scores.append(drcik_point_metrics(task.numeric.future_values, target.forecast, cap=metric_cap))
        cell = describe_history(task.numeric.history_values, task.numeric.prediction_length, task.numeric.frequency,
            child.member.family, descriptor_policy)
        cells[cell.fingerprint()] = cell
    statuses, violations = {task: "passed" for task in task_ids}, []
    if adapter.host_evaluator is None:
        objectives = NumericalObjectiveVectorV2(
            float(statistics.fmean(score["smae"] for score in scores)),
            float(statistics.fmean(score["srmse"] for score in scores)),
            float(linear_quantile([score["srmse"] for score in scores], 0.95)),
            float(statistics.fmean(joint_scaled_error(score["smae_raw"], score["srmse_raw"]) for score in scores)),
            normalized_execution_cost,
        )
    else:
        result = adapter.host_evaluator(child.candidate, tuple(hosts[task] for task in task_ids))
        if type(result) is not PackageEvaluation:
            raise ValueError("host evaluator must return an exact PackageEvaluation")
        PackageEvaluation.__post_init__(result)
        if (result.candidate_sha256 != child.candidate.proposal_sha256
                or result.expected_task_ids != task_ids or result.public_test_accessed):
            raise ValueError("PackageEvaluation candidate, Train membership, or public boundary mismatch")
        for row in result.task_rows:
            task = hosts[row.task_id]
            if (row.numerical_package_sha256 != numerical_package_fingerprint(child.candidate.registry.package_for(task))
                    or row.entity_name != task.numeric.entity_name or len(row.final_forecast) != task.numeric.prediction_length):
                raise ValueError("PackageEvaluation row does not bind the trusted task/package")
            if row.invalid_count:
                statuses[row.task_id] = "invalid"
        for task_id in result.missing_task_ids:
            statuses[task_id] = "invalid"
        if result.coverage != 1.0:
            violations.append("coverage")
        if result.invalid_count:
            violations.append("new_invalid_task")
        objectives = NumericalObjectiveVectorV2(result.mean_smae, result.mean_srmse, result.p95_srmse,
            float(joint_scaled_error(result.mean_smae_raw, result.mean_srmse_raw)), normalized_execution_cost)
    return NumericalEvaluationV2(1, child.genome.fingerprint(), child.candidate.release.fingerprint,
        child.candidate.registry.fingerprint, manifest.fingerprint(), manifest.split_sha256,
        metric_policy_sha256, descriptor_policy.fingerprint(), adapter.fingerprint,
        child.genome.runtime_fingerprints, manifest.protocol_sha256, "train", bracket, rung,
        task_ids, statuses, objectives,
        ConstraintReportV2(not violations, tuple(sorted(violations))), tuple(cells[sha] for sha in sorted(cells)),
        tuple(sorted(violations)), (),
        (resource_use or ResourceUse()).to_payload())


def freeze_qd_supply(adapter, parent_release, parent_registry, archive, children, *,
                     descriptor_policy: DescriptorPolicyV2, version) -> FrozenNumericalArtifactsV2:
    """Greedy occupied-cell coverage, then constrained rank/crowding/cost/SHA."""
    if type(archive) is not NumericalQDArchive:
        raise ValueError("projection requires an exact QD archive")
    if parent_registry.release_sha256 != parent_release.fingerprint:
        raise ValueError("projection Parent release/registry mismatch")
    archive = NumericalQDArchive.from_payload(archive.to_payload())
    by_genome = {}
    for child in children:
        if type(child) is not MaterializedNumericalChildV2:
            raise ValueError("projection requires typed materialized children")
        sha = child.genome.fingerprint()
        if sha in by_genome:
            raise ValueError("projection children must have unique genomes")
        if child.descriptor_policy_sha256 != descriptor_policy.fingerprint():
            raise ValueError("projection descriptor policy mismatch")
        if not set(child.source_sha256s) <= set(adapter.sources):
            raise ValueError("projection has unverified source dependencies")
        if child.candidate.release.anchor_release_payload != parent_release.anchor_release_payload:
            raise ValueError("projection must retain the exact safe anchor")
        by_genome[sha] = child
    entries = {sha: archive.entries[sha] for cell in archive.cells for sha in cell.entry_sha256s
               if archive.entries[sha].genome_sha256 in by_genome and archive.entries[sha].constraints.feasible}
    rankings = {}
    for rank, front in enumerate(non_dominated_fronts(entries.values())):
        distances = crowding_distances(front)
        for entry in front:
            rankings[entry.fingerprint()] = (rank, -distances[entry.fingerprint()],
                entry.objectives.normalized_execution_cost, entry.fingerprint())
    choices = {}
    for genome_sha, child in by_genome.items():
        rows = tuple(entry for entry in entries.values() if entry.genome_sha256 == genome_sha)
        if not rows:
            continue
        specs = tuple(spec for spec in child.candidate.release.alternatives if spec.candidate_id == child.fit.recipe.name)
        if len(specs) != 1:
            raise ValueError("materialized member lacks one exact legacy alternative")
        choices[genome_sha] = (child, specs[0], {row.cell.fingerprint() for row in rows},
            min(rankings[row.fingerprint()] for row in rows))
    selected, covered, families = [], set(), set()
    while len(selected) < 4:
        eligible = [sha for sha, (_, spec, _, _) in choices.items() if spec.family not in families]
        if not eligible:
            break
        sha = min(eligible, key=lambda sha: (-len(choices[sha][2] - covered), *choices[sha][3][:3], sha))
        child, spec, cells, _ = choices.pop(sha)
        selected.append((sha, child, spec))
        covered.update(cells)
        families.add(spec.family)
    def commitment(values):
        return fingerprint_payload({"sha256s": sorted(set(values))})
    sources = dict(parent_release.source_fingerprints) | {
        "qd_snapshot": archive.fingerprint(), "descriptor_policy": descriptor_policy.fingerprint(),
        "genomes": commitment(sha for sha, _, _ in selected),
        "inventories": commitment(child.genome.inventory_sha256 for _, child, _ in selected),
        "screening_policies": commitment(child.genome.screening_policy_sha256 for _, child, _ in selected),
        "combined_policies": commitment(child.genome.combined_policy_sha256 for _, child, _ in selected),
        "mutation_policies": commitment(child.genome.mutation_policy_sha256 for _, child, _ in selected),
        "proposer_prompts": commitment(child.genome.proposer_prompt_sha256 for _, child, _ in selected),
        "selected_sources": commitment(sha for _, child, _ in selected for sha in child.source_sha256s),
        "selected_member_policies": commitment(child.member.policy_sha256 for _, child, _ in selected),
    }
    release = replace(parent_release, version=version, parent_sha256=parent_release.fingerprint,
        alternatives=tuple(spec for _, _, spec in selected), source_fingerprints=sources,
        anchor_release_payload=parent_release.to_payload()["anchor_release_payload"])
    def builder(task, supplied):
        source = parent_registry.package_for(task)
        available = {source.protected_baseline.name: source.protected_baseline}
        for _, child, spec in selected:
            package = child.candidate.registry.package_for(task)
            if package.protected_baseline.forecast != source.protected_baseline.forecast:
                raise ValueError("projection changed the safe anchor forecast")
            available.update({item.name: item for item in package.ranked_alternatives if item.name == spec.candidate_id})
        return bound_numerical_package(source, supplied, available, history=task.numeric.history_values,
            task_fold=adapter.fold_manifest.task_fold_map.get(task.numeric.task_id))
    registry = build_package_registry(adapter.tasks, release, builder)
    return FrozenNumericalArtifactsV2(release, registry, _envelope(registry, adapter.tasks), tuple(sha for sha, _, _ in selected))
