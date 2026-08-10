"""Load, save, and version the JSON bundle files that evolution mutates."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models import Bundle, FewshotExample

AGENTS = ("coding", "retrieval", "decision")
_VERSION_RE = re.compile(r"^v(\d+)\.json$")
SEED_DIR = Path(__file__).parent


def _to_bundle(row: dict, agent: str, version: str) -> Bundle:
    """Build a Bundle from a parsed JSON row, tolerating absent optional fields."""
    return Bundle(
        bundle_id=f"{agent}/{version}",
        agent=agent,
        version=version,
        parent=row.get("parent"),
        system_prompt=row["system_prompt"],
        fewshot_examples=tuple(
            FewshotExample(input=item["input"], output=item["output"]) for item in row.get("fewshot_examples", [])
        ),
        code_templates=dict(row.get("code_templates", {})),
        notes_from_evolver=row.get("notes_from_evolver", ""),
        hyperparameters=dict(row.get("hyperparameters", {})),
        created_at=row.get("created_at", ""),
    )


def load_bundle(path: str | Path) -> Bundle:
    """Read one bundle JSON file, deriving agent/version from its directory and filename."""
    resolved = Path(path).expanduser().resolve()
    row = json.loads(resolved.read_text(encoding="utf-8"))
    return _to_bundle(row, agent=resolved.parent.name, version=resolved.stem)


def save_bundle(bundle: Bundle, bundles_dir: str | Path) -> Path:
    """Write a bundle to {bundles_dir}/{agent}/{version}.json and return the path."""
    directory = Path(bundles_dir).expanduser().resolve() / bundle.agent
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": bundle.version,
        "parent": bundle.parent,
        "created_at": bundle.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system_prompt": bundle.system_prompt,
        "fewshot_examples": [{"input": item.input, "output": item.output} for item in bundle.fewshot_examples],
        "code_templates": bundle.code_templates,
        "notes_from_evolver": bundle.notes_from_evolver,
        "hyperparameters": bundle.hyperparameters,
    }
    path = directory / f"{bundle.version}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def next_version(bundles_dir: str | Path, agent: str) -> str:
    """Return the next unused v### version string for an agent's bundle directory."""
    directory = Path(bundles_dir).expanduser().resolve() / agent
    highest = -1
    if directory.is_dir():
        for entry in directory.iterdir():
            match = _VERSION_RE.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"v{highest + 1:03d}"


def load_seed(agent: str) -> Bundle:
    """Load the committed hand-written v000 seed bundle for an agent."""
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent {agent!r}, expected one of {AGENTS}")
    return load_bundle(SEED_DIR / agent / "v000.json")
