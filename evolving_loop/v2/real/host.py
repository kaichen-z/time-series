"""Fixed Host runtime for bounded real Evolution V2 stages."""
from __future__ import annotations

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
EXPECTED_REAL_FORECAST_STORE_IDENTITY = (
    "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
)


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
]
