"""Create a method-evolution repository seeded with the hand-written exemplar methods.

The original 93-method seed was produced by a throwaway invocation that was never committed,
so there was no way to reproduce it. This is that entry point, and it needs no LLM: the seed
is checked-in source.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .evolution import METHODS_FILENAME, commit_module, init_repo, run_git
from .evolution import example_methods
from .evolution.module import read_module, write_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="where to create the evolution repository")
    parser.add_argument(
        "--methods",
        default=None,
        help="a module of seed methods; defaults to the checked-in exemplars",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing seeded module"
    )
    return parser


def seed(repo: str | Path, methods_path: str | Path | None = None, *, force: bool = False) -> str:
    """Write the seed module into a fresh repository and return the seed commit."""
    root = Path(repo)
    destination = root / METHODS_FILENAME
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass force=True to overwrite it")

    source = Path(methods_path) if methods_path else Path(example_methods.__file__)
    module = read_module(source)

    init_repo(root)
    write_module(destination, module)
    return commit_module(
        root,
        f"seed {len(module.names())} composed forecasting methods",
        list(module.names()),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commit = seed(args.repo, args.methods, force=args.force)
    module = read_module(Path(args.repo) / METHODS_FILENAME)
    print(f"seeded {len(module.names())} methods at {args.repo}, commit {commit}")
    for name in module.names():
        print(f"  {name}")
    print(run_git(Path(args.repo), "log", "--oneline"))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
