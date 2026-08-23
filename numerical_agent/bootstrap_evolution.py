"""Bootstrap the standalone method-evolution Git repository from the verified catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

from common.llm import (
    ClaudeCLIClient,
    ClaudeCLIConfig,
    CodexCLIClient,
    CodexCLIConfig,
    QwenClient,
)
from common.tracing import configure

from .evolution.repository_bootstrap import bootstrap_repository
from .evolution.seed import seed_definitions

LLM_BACKENDS = ("codex", "qwen", "claude")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--catalog",
        default="numerical_agent/datasets/forecast_method_dataset_v002.json",
    )
    parser.add_argument("--family", default="statistical")
    parser.add_argument("--attempts-per-method", type=int, default=2)
    parser.add_argument("--llm-backend", choices=LLM_BACKENDS, default="codex")
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--claude-cache-dir", default=None)
    parser.add_argument("--claude-timeout", type=int, default=900)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo)
    definitions, excluded = seed_definitions(args.catalog, family=args.family)
    configure(repo / ".bootstrap" / "bootstrap_trace.jsonl")
    result = bootstrap_repository(
        repo,
        definitions,
        excluded,
        _llm_client(args),
        attempts_per_method=args.attempts_per_method,
    )
    print(
        f"seeded {result.succeeded}/{result.total} methods at {result.commit}; "
        f"failed={result.failed}, resumed={result.resumed}"
    )
    return 0


def _llm_client(args: argparse.Namespace):
    if args.llm_backend == "codex":
        cache_dir = args.codex_cache_dir or str(Path(args.repo) / "codex-cache")
        return CodexCLIClient(
            CodexCLIConfig(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning_effort,
                timeout_seconds=args.codex_timeout,
                cache_dir=cache_dir,
            )
        )
    if args.llm_backend == "claude":
        cache_dir = args.claude_cache_dir or str(Path(args.repo) / "claude-cache")
        return ClaudeCLIClient(
            ClaudeCLIConfig(
                model=args.claude_model,
                timeout_seconds=args.claude_timeout,
                cache_dir=cache_dir,
            )
        )
    options = {
        name: value
        for name, value in {
            "model_id": args.model_id,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
        }.items()
        if value is not None
    }
    return QwenClient(**options)


if __name__ == "__main__":
    raise SystemExit(main())
