"""Writes metadata.json for a Dr-CiK submissions/<method>/ folder.

forecasts.jsonl is already produced by `run.py --out`; this only adds the required manifest.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_TYPES = ("agentic", "direct_llm", "ts_model", "statistical", "retrieval")

REPRODUCTION_URL = "https://github.com/khoutaibi-iliass/time-series"
CONTACT_EMAIL = "khoutaibi.iliass@gmail.com"


@dataclass(frozen=True)
class Metadata:
    method: str
    organization: str
    type: str
    base_model: str
    reproduction: str
    contact: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {self.type!r}")


def write_metadata(destination: Path, metadata: Metadata) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def check_forecasts(path: Path, min_samples: int = 100) -> list[str]:
    """Sanity checks the SUBMISSION.md verification step will run: full coverage, plausible shape."""
    problems = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        benchmark_id = record["benchmark_id"]
        if benchmark_id in seen:
            problems.append(f"duplicate benchmark_id {benchmark_id}")
        seen.add(benchmark_id)
        samples = record["samples"]
        if len(samples) < min_samples:
            problems.append(f"{benchmark_id}: only {len(samples)} samples, need >= {min_samples}")
        lengths = {len(path_) for path_ in samples}
        if len(lengths) > 1:
            problems.append(f"{benchmark_id}: ragged trajectory lengths {lengths}")
        if not all(value == value and abs(value) != float("inf")
                   for path_ in samples for value in path_):
            problems.append(f"{benchmark_id}: contains a non-finite value")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="submissions/<method>/metadata.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--organization", default="Harvard AI and Robotics Lab")
    parser.add_argument("--type", required=True, choices=VALID_TYPES)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--check", default=None, help="also validate this forecasts.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = Metadata(
        method=args.method,
        organization=args.organization,
        type=args.type,
        base_model=args.base_model,
        reproduction=REPRODUCTION_URL,
        contact=CONTACT_EMAIL,
        notes=args.notes,
    )
    write_metadata(Path(args.out), metadata)
    print(f"wrote {args.out}")
    if args.check:
        problems = check_forecasts(Path(args.check))
        if problems:
            print(f"{len(problems)} problem(s):")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"{args.check}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
