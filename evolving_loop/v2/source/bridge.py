"""Production Host bridge from a sealed P3 closure to Source V2."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..contracts import require_sha256
from ..cooperative import (
    CooperativeArtifactCatalog,
    CooperativePipelineAdapter,
    DecisionCoordinateAdapter,
    NumericalCoordinateAdapter,
    RetrievalCoordinateAdapter,
)
from ..real.bridges import P3BundleClosureV2
from ..real.host import RealHostRuntimeV2, select_real_task_projection
from .contracts import SourceVariantV2
from .meta import SourceMetaEvaluatorV2


@dataclass(slots=True)
class SourceBundleCaseV2:
    """Reconstructible Source V2 input bound to one sealed P3 bundle."""

    seed_source: SourceVariantV2
    seed_bundle: object
    train_tasks: tuple[object, ...]
    dev_tasks: tuple[object, ...]
    catalog_factory: object
    adapters_factory: object
    pipeline_factory: object
    input_digest: str
    fold_count: int = 2
    evaluator: SourceMetaEvaluatorV2 = field(init=False)

    def __post_init__(self) -> None:
        self.evaluator = SourceMetaEvaluatorV2(self)


def _catalog_from_closure(closure: P3BundleClosureV2) -> CooperativeArtifactCatalog:
    catalog = CooperativeArtifactCatalog(lambda _identity, _payload: None)
    catalog.add_numerical(closure.numerical)
    for pair in closure.numerical_alternatives:
        catalog.add_numerical(pair)
    catalog.add_retrieval(closure.retrieval)
    catalog.add_decision(closure.decision)
    return catalog


def build_source_case_from_p3(
    closure: P3BundleClosureV2,
    host: RealHostRuntimeV2,
    *,
    source_seed: SourceVariantV2,
    input_digest: str,
    empty_skill_path: Path,
) -> SourceBundleCaseV2:
    """Bind the exact sealed P3 Bundle to real Source V2 evaluation adapters."""
    if type(closure) is not P3BundleClosureV2:
        raise TypeError("source bridge requires a sealed P3 closure")
    if type(host) is not RealHostRuntimeV2:
        raise TypeError("source bridge requires a real Host runtime")
    if type(source_seed) is not SourceVariantV2:
        raise TypeError("source bridge requires an admitted SourceVariantV2 seed")
    if source_seed.operator != "seed" or source_seed.parent_source_sha256 is not None:
        raise ValueError("source bridge requires the admitted versioned seed policy")
    require_sha256(input_digest, "source input_digest")
    if (
        source_seed.protocol_fingerprint != closure.active_bundle.protocol_fingerprint
        or source_seed.runtime_fingerprint != closure.runtime_identity
        or closure.runtime_identity != host.resource_reporter_sha256
    ):
        raise ValueError("source seed commitments do not match the sealed P3 closure")
    train, dev = select_real_task_projection(
        host.train_tasks,
        host.dev_tasks,
        train_size=host.projection_train_size,
        dev_size=host.projection_dev_size,
    )
    if tuple(closure.train_tasks) != train or tuple(closure.dev_tasks) != dev:
        raise ValueError("source bridge P3 task projection does not match Host")

    seed_bytes = closure.active_bundle.canonical_bytes()

    def catalog_factory():
        catalog = _catalog_from_closure(closure)
        seed = closure.active_bundle
        if seed.canonical_bytes() != seed_bytes:
            raise ValueError("sealed P3 Bundle changed during Source reconstruction")
        return catalog, seed

    def adapters_factory(_catalog):
        return {
            "numerical": NumericalCoordinateAdapter(closure.numerical_alternatives),
            "retrieval": RetrievalCoordinateAdapter(),
            "decision": DecisionCoordinateAdapter(closure.decision_prompts),
        }

    def pipeline_factory(catalog):
        return CooperativePipelineAdapter(
            catalog,
            host.retrieval_factory,
            host.decision_factory,
            metric_cap=closure.metric_cap,
            retrieval_skill_library=host.retrieval_skill_library,
            empty_skill_path=Path(empty_skill_path),
        )

    return SourceBundleCaseV2(
        seed_source=source_seed,
        seed_bundle=closure.active_bundle,
        train_tasks=tuple(closure.train_tasks),
        dev_tasks=tuple(closure.dev_tasks),
        catalog_factory=catalog_factory,
        adapters_factory=adapters_factory,
        pipeline_factory=pipeline_factory,
        input_digest=input_digest,
        fold_count=getattr(host, "projection_fold_count", 2),
    )


__all__ = ["SourceBundleCaseV2", "build_source_case_from_p3"]
