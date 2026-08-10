"""Per-generation checkpoints, so a crashed multi-hour evolve run resumes where it stopped."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .evaluate import EvalResult, TaskResult

logger = logging.getLogger(__name__)

_GEN_RE = re.compile(r"^gen_(\d+)\.json$")


@dataclass(frozen=True)
class GenerationRecord:
    """Everything one generation produced, enough to resume or plot the run afterwards."""

    generation: int
    population: tuple[str, ...]
    eval_results: tuple[EvalResult, ...]
    elite: tuple[str, ...]
    dev_score: float | None = None
    stalled: bool = False
    bundle_paths: dict[str, str] = field(default_factory=dict)


def _to_eval_result(row: dict) -> EvalResult:
    """Rebuild an EvalResult from its serialized form."""
    return EvalResult(
        individual_id=row["individual_id"],
        mean_score=row["mean_score"],
        task_results=tuple(TaskResult(**item) for item in row["task_results"]),
        worst=tuple(TaskResult(**item) for item in row.get("worst", [])),
    )


def save_generation(checkpoint_dir: str | Path, record: GenerationRecord) -> Path:
    """Write one generation's record atomically."""
    directory = Path(checkpoint_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gen_{record.generation:03d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_generation(checkpoint_dir: str | Path, generation: int) -> GenerationRecord | None:
    """Read one generation's record, or None if it was never completed."""
    path = Path(checkpoint_dir).expanduser().resolve() / f"gen_{generation:03d}.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("checkpoint %s is unreadable, treating the generation as incomplete", path)
        return None
    return GenerationRecord(
        generation=row["generation"],
        population=tuple(row["population"]),
        eval_results=tuple(_to_eval_result(item) for item in row["eval_results"]),
        elite=tuple(row["elite"]),
        dev_score=row.get("dev_score"),
        stalled=row.get("stalled", False),
        bundle_paths=row.get("bundle_paths", {}),
    )


def latest_generation(checkpoint_dir: str | Path) -> int:
    """Return the highest completed generation number, or -1 when none exist."""
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        return -1
    numbers = [int(match.group(1)) for entry in directory.iterdir() if (match := _GEN_RE.match(entry.name))]
    return max(numbers, default=-1)
