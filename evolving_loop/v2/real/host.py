"""Fixed Host runtime for bounded real Evolution V2 stages."""
from __future__ import annotations

import hashlib
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from collections.abc import Mapping

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig
from evolving_loop.data import ContextTask, load_context_tasks_by_ids
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.main import _runtime_registry
from numerical_agent.run_champion_evolution import _load_screening_policy
from numerical_agent.run_selector_evolution import _forecast_runtime_identity
from numerical_agent.evolution.screening import ScreeningPolicy

from ..budget import ResourceUse
from ..contracts import fingerprint_payload
from ..numerical_qd.adapters import LegacyNumericalAdapter
from .contracts import RealEvolutionManifestV2

# The split validator is the existing fixed 80/20/99 Host admission boundary.
from evolving_loop.run_package_coevolution import _validated_split


_FILE_ROLES = {
    "split",
    "tasks",
    "numerical_seed",
    "numerical_source_seed",
    "forecast_cache",
    "retrieval_seed",
    "source_seed",
}
_RUNTIME_ROLES = {
    "python",
    "runtime",
    "task_loader",
    "forecast_store",
    "model_cache",
    "codex_cli",
}
_INPUT_KINDS = {
    "split": "file",
    "tasks": "directory",
    "numerical_seed": "file",
    "numerical_source_seed": "file",
    "forecast_cache": "directory",
    "retrieval_seed": "directory",
    "source_seed": "file",
}
_CODE_FILE_ROLES = frozenset({"numerical_source_seed", "source_seed"})
_CODE_RUNTIME_ROLES = frozenset({"task_loader", "forecast_store"})
_RUNTIME_KINDS = {
    "python": "file",
    "runtime": "file",
    "task_loader": "file",
    "forecast_store": "file",
    "model_cache": "model_cache",
    "codex_cli": "file",
}
EXPECTED_REAL_FORECAST_STORE_IDENTITY = (
    "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """Return Git's SHA-1 object identity for one Hugging Face CAS blob."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_kind(path: Path, kind: str, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise ValueError(f"real Host {role} identity target is unavailable") from error
    expected = stat.S_ISREG(mode) if kind == "file" else stat.S_ISDIR(mode)
    if not expected:
        raise ValueError(f"real Host {role} identity target has wrong kind")
    return resolved


def _directory_content_identity(path: Path, role: str) -> str:
    root = _require_kind(path, "directory", role)
    entries = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(root).as_posix()
        mode = child.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"real Host {role} directory contains a symlink")
        if stat.S_ISDIR(mode):
            entries.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(mode):
            entries.append(
                {"path": relative, "kind": "file", "sha256": _sha256_file(child)}
            )
        else:
            raise ValueError(f"real Host {role} directory has a special entry")
    return fingerprint_payload(
        {
            "schema_version": 1,
            "kind": "real_directory_content_identity",
            "entries": entries,
        }
    )


def _model_cache_identity(path: Path) -> str:
    """Authenticate Hugging Face Git-SHA1 and LFS-SHA256 CAS objects."""
    root = _require_kind(path, "directory", "model_cache")
    entries = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(root).as_posix()
        mode = child.lstat().st_mode
        if stat.S_ISLNK(mode):
            try:
                target = child.resolve(strict=True)
            except OSError as error:
                raise ValueError("real Host model_cache has a broken link") from error
            if not target.is_relative_to(root):
                raise ValueError("real Host model_cache link escapes its root")
            if not stat.S_ISREG(target.stat().st_mode):
                raise ValueError("real Host model_cache link target is not a file")
            entries.append(
                {
                    "path": relative,
                    "kind": "link",
                    "target": target.relative_to(root).as_posix(),
                    "size": target.stat().st_size,
                }
            )
        elif stat.S_ISDIR(mode):
            entries.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(mode):
            parts = child.relative_to(root).parts
            entry = {"path": relative, "kind": "file", "size": child.stat().st_size}
            if "blobs" in parts:
                if len(child.name) not in {40, 64} or any(
                    character not in "0123456789abcdef" for character in child.name
                ):
                    raise ValueError("real Host model_cache has a malformed CAS blob name")
                actual = (
                    _git_blob_sha1(child)
                    if len(child.name) == 40
                    else _sha256_file(child)
                )
                if actual != child.name:
                    raise ValueError(
                        "real Host model_cache CAS blob content digest mismatch"
                    )
                entry["cas_digest"] = child.name
            else:
                entry["sha256"] = _sha256_file(child)
            entries.append(entry)
        else:
            raise ValueError("real Host model_cache has a special entry")
    return fingerprint_payload(
        {
            "schema_version": 1,
            "kind": "huggingface_model_cache_semantic_identity",
            "entries": entries,
        }
    )


def resolve_real_input_identity(role: str, path: Path) -> str:
    """Return raw-file or closed-tree SHA-256 for one declared input role."""
    kind = _INPUT_KINDS.get(role)
    if kind is None:
        raise ValueError(f"unknown real Host input role: {role}")
    resolved = _require_kind(Path(path), kind, role)
    return _sha256_file(resolved) if kind == "file" else _directory_content_identity(
        resolved, role
    )


def resolve_real_runtime_identity(role: str, path: Path) -> str:
    """Return executable/source bytes or the model-cache CAS semantic identity."""
    kind = _RUNTIME_KINDS.get(role)
    if kind is None:
        raise ValueError(f"unknown real Host runtime role: {role}")
    if kind == "model_cache":
        return _model_cache_identity(Path(path))
    return _sha256_file(_require_kind(Path(path), kind, role))


def _verify_manifest_identities(manifest, files, locations) -> None:
    for row in manifest.files:
        if resolve_real_input_identity(row.role, files[row.role]) != row.sha256:
            raise ValueError(f"real Host input identity mismatch for {row.role}")
    for row in manifest.runtime_locations:
        if row.role not in locations:
            continue
        if (
            resolve_real_runtime_identity(row.role, locations[row.role])
            != row.identity_sha256
        ):
            raise ValueError(f"real Host runtime identity mismatch for {row.role}")


def _numerical_evolution_sources(manifest, files, source_repo):
    """Load the one declared mutation seed; the full catalog stays cache-backed."""
    from numerical_agent.evolution.module import read_module

    seed_path = files["numerical_source_seed"]
    seed_sha256 = _sha256_file(seed_path)
    if manifest.l0_fingerprints.get("numerical_source_seed") != seed_sha256:
        raise ValueError("real Host numerical source seed L0 identity mismatch")
    try:
        seed_source = seed_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError("real Host numerical source seed is not UTF-8") from error
    seed_module = LegacyNumericalAdapter.validate_source(seed_source)
    if seed_module.names() != ("naive_last",):
        raise ValueError("real Host numerical source seed must define only naive_last")

    historical = read_module(source_repo / "methods.py").get("naive_last")
    if historical is None:
        raise ValueError("real Host numerical source origin is unavailable")
    origin_sha256 = hashlib.sha256(historical.source.encode("utf-8")).hexdigest()
    if manifest.l0_fingerprints.get("numerical_source_origin") != origin_sha256:
        raise ValueError("real Host numerical source origin L0 identity mismatch")
    return {seed_sha256: seed_source}, seed_sha256, origin_sha256


def _confined(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("real Host path escapes repository root")
    return path


def _runtime_location(root: Path, relative: str, role: str) -> Path:
    """Resolve a manifest runtime target without relaxing input confinement.

    Runtime entries name an executable from the repository namespace (for
    example ``.venv/bin/python``).  That name is confined before resolution;
    only its final executable target may live outside the checkout because a
    virtualenv launcher is normally a symlink to the host interpreter.
    """
    if role == "codex_cli" and relative == "codex" and not (root / relative).exists():
        found = shutil.which("codex")
        if found is None:
            raise ValueError("real Host codex_cli identity target is unavailable")
        return Path(found).resolve(strict=True)
    lexical = root / relative
    if not lexical.absolute().is_relative_to(root.resolve()):
        raise ValueError("real Host runtime path escapes repository root")
    try:
        return lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError("real Host runtime identity target is unavailable") from error


def _forecast_model_cache_root(location: Path) -> Path:
    """Recover the historical HF_HOME from its authenticated ``hub`` tree."""
    return location.parent if location.name == "hub" else location


def select_real_task_projection(
    train_tasks: tuple[ContextTask, ...],
    dev_tasks: tuple[ContextTask, ...],
    *,
    train_size: int = 4,
    dev_size: int = 1,
) -> tuple[tuple[ContextTask, ...], tuple[ContextTask, ...]]:
    """Choose the first feasible entity-disjoint Train{N}/Dev{M} in Host order.

    Only entity identities participate in selection; task labels and metrics
    remain outside this projection boundary. Return the original Host tasks.
    The default Train4/Dev1 preserves the frozen projection behavior.
    """
    if type(train_size) is not int or train_size < 1:
        raise ValueError("projection train_size must be a positive integer")
    if type(dev_size) is not int or dev_size < 1:
        raise ValueError("projection dev_size must be a positive integer")
    train_entity_pool = {task.numeric.entity_name for task in train_tasks}
    dev_selected: list[ContextTask] = []
    dev_entities: set[str] = set()
    for dev_task in dev_tasks:
        entity = dev_task.numeric.entity_name
        if entity in dev_entities:
            continue
        # Preserve "first feasible Dev in Host order": only commit a Dev task if
        # enough entity-disjoint Train tasks still remain for it.
        tentative = dev_entities | {entity}
        if len(train_entity_pool - tentative) < train_size:
            continue
        dev_entities = tentative
        dev_selected.append(dev_task)
        if len(dev_selected) == dev_size:
            break
    if len(dev_selected) != dev_size:
        raise ValueError(
            f"real projection requires {dev_size} entity-disjoint Dev tasks"
        )
    seen_entities = set(dev_entities)
    selected: list[ContextTask] = []
    for train_task in train_tasks:
        entity = train_task.numeric.entity_name
        if entity in seen_entities:
            continue
        seen_entities.add(entity)
        selected.append(train_task)
        if len(selected) == train_size:
            break
    if len(selected) != train_size:
        raise ValueError(
            f"real projection requires {train_size} entity-disjoint Train tasks"
        )
    return tuple(selected), tuple(dev_selected)


@dataclass(slots=True)
class RealHostRuntimeV2:
    """One shared real task, numerical-cache, and agent runtime boundary."""

    manifest: RealEvolutionManifestV2
    tasks: tuple[ContextTask, ...]
    train_tasks: tuple[ContextTask, ...]
    dev_tasks: tuple[ContextTask, ...]
    forecast_store: ForecastStore
    runtime_registry: object
    llm_client: CodexCLIClient
    retrieval_skill_library: RetrievalSkillLibrary
    source_repo: Path
    screening_policy: ScreeningPolicy
    resource_reporter_sha256: str
    sources: Mapping[str, str] = field(default_factory=dict)
    p3_dictionary: object | None = field(default=None, repr=False)
    resource_kinds: tuple[str, ...] = ()
    projection_train_size: int = 4
    projection_dev_size: int = 1
    projection_fold_count: int = 2
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.screening_policy) is not ScreeningPolicy:
            raise TypeError("real Host requires an exact ScreeningPolicy")
        sources = dict(self.sources)
        if any(
            type(identity) is not str
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            or type(source) is not str
            or not source.strip()
            for identity, source in sources.items()
        ):
            raise ValueError("real Host sources must be SHA-identified source text")
        object.__setattr__(self, "sources", MappingProxyType(sources))
        if self.p3_dictionary is not None:
            from ..cooperative import P3NumericalDictionaryV2

            if type(self.p3_dictionary) is not P3NumericalDictionaryV2:
                raise TypeError("real Host p3_dictionary must be a P3 Dictionary closure")

    def retrieval_factory(self, genome, skills=None) -> TwoStageRetrievalAgent:
        library = self.retrieval_skill_library if skills is None else skills
        if type(library) is not RetrievalSkillLibrary:
            raise TypeError("real Retrieval factory requires verified Skills")
        return TwoStageRetrievalAgent(
            self.llm_client,
            genome,
            library.clone(persist=False, read_only=True),
        )

    def decision_factory(self, module) -> DecisionAgent:
        prompt = getattr(module, "prompt", module)
        if type(prompt) is not str or not prompt:
            raise ValueError("real Decision factory requires a nonempty prompt")
        return DecisionAgent(self.llm_client, None, prompt=prompt)

    def resource_reporter(self) -> ResourceUse:
        # UTF-8 byte counts are the V2 token-budget proxy used by the proposers.
        return ResourceUse(
            llm_calls=int(getattr(self.llm_client, "calls", 0)),
            input_tokens=int(getattr(self.llm_client, "input_tokens", 0)),
            output_tokens=int(getattr(self.llm_client, "output_tokens", 0)),
            subprocesses=int(getattr(self.llm_client, "subprocesses", 0)),
        )

    def close(self) -> None:
        """Close all owned numerical runtimes once, attempting every closure."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for resource in (self.forecast_store, self.runtime_registry):
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - close remaining owners
                errors.append(error)
        if errors:
            raise errors[0]


def _filter_tsfm_workers_config(path: Path) -> Path | None:
    """Remove worker environments whose interpreters are absent; return None if all gone."""
    import json as _json
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, _json.JSONDecodeError):
        return path
    envs = payload.get("environments", {})
    kept = {
        name: entry for name, entry in envs.items()
        if not (isinstance(entry, dict)
                and isinstance(entry.get("interpreter"), str)
                and not Path(entry["interpreter"]).expanduser().exists())
    }
    if kept == envs:
        return path
    if not kept:
        return None
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
        prefix="real-host-workers-"
    )
    _json.dump({**payload, "environments": kept}, tmp, separators=(",", ":"))
    tmp.close()
    return Path(tmp.name)


def build_real_host(
    manifest: RealEvolutionManifestV2,
    *,
    repo_root: Path,
    output_dir: Path,
    code_root: Path | None = None,
    projection_train_size: int = 4,
    projection_dev_size: int = 1,
    projection_fold_count: int = 2,
) -> RealHostRuntimeV2:
    """Build the fixed cache-only Host from one already verified manifest."""
    if type(projection_train_size) is not int or projection_train_size < 1:
        raise ValueError("projection_train_size must be a positive integer")
    if type(projection_dev_size) is not int or projection_dev_size < 1:
        raise ValueError("projection_dev_size must be a positive integer")
    if (
        type(projection_fold_count) is not int
        or projection_fold_count < 1
        or projection_train_size % projection_fold_count != 0
    ):
        raise ValueError(
            "projection_fold_count must be a positive divisor of projection_train_size"
        )
    if type(manifest) is not RealEvolutionManifestV2:
        raise TypeError("real Host requires a RealEvolutionManifestV2")
    root = Path(repo_root).resolve()
    code = root if code_root is None else Path(code_root).resolve()
    files = {
        row.role: _confined(code if row.role in _CODE_FILE_ROLES else root, row.relative_path)
        for row in manifest.files
    }
    _use_claude = shutil.which("codex") is None and shutil.which("claude") is not None
    locations = {
        row.role: _runtime_location(
            code if row.role in _CODE_RUNTIME_ROLES else root, row.relative_path, row.role
        )
        for row in manifest.runtime_locations
        if not (row.role == "codex_cli" and _use_claude)
    }
    _expected_roles = _RUNTIME_ROLES - ({"codex_cli"} if _use_claude else set())
    if set(files) != _FILE_ROLES or set(locations) != _expected_roles:
        raise ValueError("real Host manifest requires every file and runtime role")
    _verify_manifest_identities(manifest, files, locations)

    source_repo = _confined(root, "runs/method_evolution/v001")
    for role, name in (
        ("methods", "methods.py"),
        ("policies", "policies.py"),
        ("skills", "skills.py"),
        ("dictionary", "dictionary.py"),
    ):
        expected = manifest.l0_fingerprints.get(role)
        if expected is not None and _sha256_file(source_repo / name) != expected:
            raise ValueError(f"real Host L0 source identity mismatch for {role}")
    sources, numerical_source_seed_sha256, numerical_source_origin_sha256 = (
        _numerical_evolution_sources(manifest, files, source_repo)
    )

    _split, train_ids, dev_ids = _validated_split(
        files["split"], allow_label_informed_regression_split=True
    )
    tasks = load_context_tasks_by_ids(files["tasks"], (*train_ids, *dev_ids))
    if tuple(task.numeric.task_id for task in tasks) != (*train_ids, *dev_ids):
        raise ValueError("real Host tasks differ from frozen Train/Dev membership")

    methods_path = source_repo / "methods.py"
    skills_path = source_repo / "skills.py"
    portfolio = read_policy_file(source_repo / "policies.py")
    screening = _load_screening_policy(source_repo / "dictionary.py")
    workers_config_path = locations["runtime"]
    workers_config_path = _filter_tsfm_workers_config(workers_config_path)
    runtime_args = SimpleNamespace(
        tsfm_runtimes="chronos,timesfm",
        chronos_device_map="cpu",
        model_cache_dir=_forecast_model_cache_root(locations["model_cache"]),
        tsfm_workers_config=workers_config_path,
        acknowledged_model_licenses="CC-BY-NC-4.0" if workers_config_path else None,
    )
    runtimes = _runtime_registry(runtime_args)
    forecast_store = None
    try:
        forecast_store = ForecastStore(
            files["forecast_cache"],
            methods_path,
            skills_path if skills_path.is_file() else None,
            portfolio,
            runtimes,
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(runtime_args),
            cache_only=True,
            identity_hash_override=(
                manifest.l0_fingerprints.get("forecast_store") if _use_claude else None
            ),
        )
        expected_store = manifest.l0_fingerprints.get(
            "forecast_store", EXPECTED_REAL_FORECAST_STORE_IDENTITY
        )
        if forecast_store.identity_hash != expected_store:
            raise ValueError("real ForecastStore identity mismatch")
        if _use_claude:
            _claude_binary = shutil.which("claude")
            # The manifest model is contract-frozen to gpt-5.6-luna, which maps to
            # the Claude CLI default. EVOLVE_CLAUDE_MODEL overrides the proposer model
            # (e.g. "haiku") without touching the frozen manifest binding.
            import os as _os
            _claude_model = _os.environ.get("EVOLVE_CLAUDE_MODEL") or (
                manifest.model.name if not manifest.model.name.startswith("gpt") else None
            )
            llm = ClaudeCLIClient(
                ClaudeCLIConfig(
                    binary=_claude_binary,
                    model=_claude_model,
                    timeout_seconds=900,
                    cache_dir=Path(output_dir) / "llm-cache",
                )
            )
        else:
            llm = CodexCLIClient(
                CodexCLIConfig(
                    binary=str(locations["codex_cli"]),
                    model=manifest.model.name,
                    reasoning_effort=manifest.model.reasoning_effort,
                    timeout_seconds=900,
                    cache_dir=Path(output_dir) / "llm-cache",
                )
            )
        library = RetrievalSkillLibrary.from_release(
            files["retrieval_seed"]
        ).clone(persist=False, read_only=True)
        reporter_sha = fingerprint_payload(
            {
                "schema_version": 1,
                "kind": "cache_only_real_host_resource_reporter",
                "forecast_store_identity": forecast_store.identity_hash,
                "numerical_source_seed_sha256": numerical_source_seed_sha256,
                "numerical_source_origin_sha256": numerical_source_origin_sha256,
                "runtime_identity": {
                    row.role: row.identity_sha256
                    for row in manifest.runtime_locations
                },
                "resource_kinds": [
                    "input_tokens", "llm_calls", "output_tokens", "subprocesses"
                ],
                "llm_accounting": "codex_utf8_bytes_v1",
            }
        )
        return RealHostRuntimeV2(
            manifest=manifest,
            tasks=tasks,
            train_tasks=tasks[:80],
            dev_tasks=tasks[80:],
            forecast_store=forecast_store,
            runtime_registry=runtimes,
            llm_client=llm,
            retrieval_skill_library=library,
            source_repo=source_repo,
            screening_policy=screening,
            resource_reporter_sha256=reporter_sha,
            sources=sources,
            resource_kinds=(
                "input_tokens", "llm_calls", "output_tokens", "subprocesses"
            ),
            projection_train_size=projection_train_size,
            projection_dev_size=projection_dev_size,
            projection_fold_count=projection_fold_count,
        )
    except BaseException:
        if forecast_store is not None:
            forecast_store.close()
        runtimes.close()
        raise


__all__ = [
    "select_real_task_projection",
    "EXPECTED_REAL_FORECAST_STORE_IDENTITY",
    "RealHostRuntimeV2",
    "build_real_host",
    "resolve_real_input_identity",
    "resolve_real_runtime_identity",
]
