"""Fixed Host runtime for bounded real Evolution V2 stages."""
from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from common.llm import CodexCLIClient, CodexCLIConfig
from evolving_loop.data import ContextTask, load_context_tasks_by_ids
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.forecast_store import ForecastStore
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.main import _runtime_registry
from numerical_agent.run_champion_evolution import _load_screening_policy
from numerical_agent.run_selector_evolution import _forecast_runtime_identity

from ..budget import ResourceUse
from ..contracts import fingerprint_payload
from .contracts import RealEvolutionManifestV2

# The split validator is the existing fixed 80/20/99 Host admission boundary.
from evolving_loop.run_package_coevolution import _validated_split


_FILE_ROLES = {
    "split",
    "tasks",
    "numerical_seed",
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
    "forecast_cache": "directory",
    "retrieval_seed": "directory",
    "source_seed": "file",
}
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
    """Fingerprint Hugging Face CAS metadata without rereading model weights."""
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
            is_cas_blob = "blobs" in parts and all(
                character in "0123456789abcdef" for character in child.name
            ) and len(child.name) in {40, 64}
            entry = {"path": relative, "kind": "file", "size": child.stat().st_size}
            if is_cas_blob:
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
        if (
            resolve_real_runtime_identity(row.role, locations[row.role])
            != row.identity_sha256
        ):
            raise ValueError(f"real Host runtime identity mismatch for {row.role}")


def _confined(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("real Host path escapes repository root")
    return path


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
    resource_reporter_sha256: str
    resource_kinds: tuple[str, ...] = ()
    _closed: bool = field(default=False, init=False, repr=False)

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

    @staticmethod
    def resource_reporter() -> ResourceUse:
        # Cache hits are local and task/wall accounting is owned by each stage.
        # Cache misses raise CacheMissError before any ungoverned model work.
        return ResourceUse()

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


def build_real_host(
    manifest: RealEvolutionManifestV2,
    *,
    repo_root: Path,
    output_dir: Path,
) -> RealHostRuntimeV2:
    """Build the fixed cache-only Host from one already verified manifest."""
    if type(manifest) is not RealEvolutionManifestV2:
        raise TypeError("real Host requires a RealEvolutionManifestV2")
    root = Path(repo_root).resolve()
    files = {row.role: _confined(root, row.relative_path) for row in manifest.files}
    locations = {
        row.role: _confined(root, row.relative_path)
        for row in manifest.runtime_locations
    }
    if set(files) != _FILE_ROLES or set(locations) != _RUNTIME_ROLES:
        raise ValueError("real Host manifest requires every file and runtime role")
    _verify_manifest_identities(manifest, files, locations)

    _split, train_ids, dev_ids = _validated_split(
        files["split"], allow_label_informed_regression_split=True
    )
    tasks = load_context_tasks_by_ids(files["tasks"], (*train_ids, *dev_ids))
    if tuple(task.numeric.task_id for task in tasks) != (*train_ids, *dev_ids):
        raise ValueError("real Host tasks differ from frozen Train/Dev membership")

    source_repo = _confined(root, "runs/method_evolution/v001")
    methods_path = source_repo / "methods.py"
    skills_path = source_repo / "skills.py"
    portfolio = read_policy_file(source_repo / "policies.py")
    screening = _load_screening_policy(source_repo / "dictionary.py")
    runtime_args = SimpleNamespace(
        tsfm_runtimes="chronos,timesfm",
        chronos_device_map="cpu",
        model_cache_dir=locations["model_cache"],
        tsfm_workers_config=locations["runtime"],
        acknowledged_model_licenses="CC-BY-NC-4.0",
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
        )
        expected_store = manifest.l0_fingerprints.get(
            "forecast_store", EXPECTED_REAL_FORECAST_STORE_IDENTITY
        )
        if expected_store != EXPECTED_REAL_FORECAST_STORE_IDENTITY:
            raise ValueError("real manifest ForecastStore identity mismatch")
        if forecast_store.identity_hash != EXPECTED_REAL_FORECAST_STORE_IDENTITY:
            raise ValueError("real ForecastStore identity mismatch")
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
                "runtime_identity": {
                    row.role: row.identity_sha256
                    for row in manifest.runtime_locations
                },
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
            resource_reporter_sha256=reporter_sha,
        )
    except BaseException:
        if forecast_store is not None:
            forecast_store.close()
        runtimes.close()
        raise


__all__ = [
    "EXPECTED_REAL_FORECAST_STORE_IDENTITY",
    "RealHostRuntimeV2",
    "build_real_host",
    "resolve_real_input_identity",
    "resolve_real_runtime_identity",
]
