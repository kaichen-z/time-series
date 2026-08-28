"""Write the seed methods module: one LLM call per catalog definition, foundation models verbatim.

This is the entry point behind runs/method_evolution/full98 -- 70 classical, 23 neural, and the
five foundation-model wrappers. Re-running is cheap when the LLM client caches by prompt, so a
bootstrap interrupted partway resumes rather than paying for every method again.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, CodexCLIClient, CodexCLIConfig, QwenClient
from common.payload import read_json_object
from common.tracing import configure

from .evolution import METHODS_FILENAME, bootstrap
from .evolution import foundation_methods
from .evolution.module import read_module
from .evolution.seed import FOUNDATION_SEEDS, full_seed_definitions

LLM_BACKENDS = ("claude", "codex", "qwen")
DEFAULT_CATALOG = "numerical_agent/datasets/forecast_method_dataset_v002.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="where to create the seeded repository")
    parser.add_argument(
        "--definitions",
        default=None,
        help="a definitions.json written earlier; defaults to reading the catalog directly",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--llm-backend", choices=LLM_BACKENDS, default="claude")
    parser.add_argument("--claude-model", default="haiku")
    parser.add_argument("--claude-cache-dir", default=None)
    parser.add_argument("--claude-timeout", type=int, default=None)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--limit", type=int, default=0, help="write only the first N model-written methods"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written and make no calls"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo)
    configure(repo / "bootstrap_trace.jsonl")

    written, preset = _definitions(args)
    if args.limit:
        written = written[: args.limit]

    print(
        f"bootstrap: {len(written)} methods to write with {args.llm_backend}"
        f"{f' ({args.claude_model})' if args.llm_backend == 'claude' else ''}, "
        f"{len(preset)} seeded verbatim, into {repo}"
    )
    if args.dry_run:
        for definition in written[:5]:
            print(f"  would write {definition['name']}")
        print(f"  ... and {max(0, len(written) - 5)} more")
        for method in preset:
            print(f"  verbatim {method.name}")
        return 0

    module = bootstrap(repo, written, _llm_client(args), preset=preset)
    print(f"bootstrap: wrote {len(module.names())} methods to {repo / METHODS_FILENAME}")
    missing = {str(d["name"]) for d in written} - set(module.names())
    if missing:
        # A definition the model failed on is skipped rather than fatal, so it has to be named.
        print(f"bootstrap: {len(missing)} definitions produced no valid method: {sorted(missing)}")
    return 0


def _definitions(args: argparse.Namespace):
    """Split the seed into what the model writes and what is handed in verbatim."""
    if args.definitions:
        payload = read_json_object(args.definitions)
        definitions = list(payload["definitions"])  # type: ignore[index]
    else:
        definitions, _excluded = full_seed_definitions(args.catalog)

    foundation = set(FOUNDATION_SEEDS.values())
    written = [d for d in definitions if str(d["name"]) not in foundation]
    preset = read_module(foundation_methods.__file__).methods
    return written, preset


def _llm_client(args: argparse.Namespace):
    if args.llm_backend == "claude":
        return ClaudeCLIClient(ClaudeCLIConfig(**_present(
            model=args.claude_model,
            timeout_seconds=args.claude_timeout,
            cache_dir=args.claude_cache_dir,
        )))
    if args.llm_backend == "codex":
        return CodexCLIClient(CodexCLIConfig(**_present(model=args.codex_model)))
    return QwenClient(**_present(model_id=args.model_id, device=args.device))


def _present(**values: object) -> dict[str, object]:
    return {name: value for name, value in values.items() if value is not None}


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
