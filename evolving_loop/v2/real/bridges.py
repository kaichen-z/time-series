"""Typed child-run bridges for bounded real Evolution V2 orchestration."""
from __future__ import annotations

from collections.abc import Mapping

from common.llm import LLMClient

from ..cli import numerical_evolve_payload
from .host import RealHostRuntimeV2


def run_real_numerical(
    context: object,
    host: RealHostRuntimeV2,
    *,
    config_payload: Mapping[str, object],
    seed_payload: Mapping[str, object],
    task_manifest_payload: Mapping[str, object],
    input_sha256s: Mapping[str, str],
    llm_client: LLMClient | None = None,
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
        llm_client=host.llm_client if llm_client is None else llm_client,
    )


__all__ = ["run_real_numerical"]
