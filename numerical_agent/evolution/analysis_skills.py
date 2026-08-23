"""Validate and inject the history-only skill module used by evolved forecasters."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

from common.sandbox import ALLOWED_IMPORTS, FORBIDDEN_NAMES, check_code

from .analysis_skills_template import *  # noqa: F403 - this module is the public skill API
from .analysis_skills_template import ANALYSIS_SKILL_NAMES


DEFAULT_SKILLS_PATH = Path(__file__).with_name("analysis_skills_template.py")
_SKILL_IMPORTS = ALLOWED_IMPORTS | frozenset({"__future__"})


def validate_skill_source(source: str) -> None:
    """Reject external-state access and public functions outside the fixed skill API."""
    check_code(source, _SKILL_IMPORTS)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "forecast":
                raise ValueError("forbidden skill name: forecast")
            if not node.name.startswith("_") and node.name not in ANALYSIS_SKILL_NAMES:
                raise ValueError(f"forbidden skill name: {node.name}")
            arguments = tuple(argument.arg for argument in node.args.args)
            if not node.name.startswith("_") and arguments not in {
                ("history",),
                ("history", "frequency"),
            }:
                raise ValueError(f"skill {node.name} is not history-only")
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and inner.id in FORBIDDEN_NAMES:
                raise ValueError(f"forbidden name: {inner.id}")


def load_skill_module(path: str | Path | None = None) -> ModuleType:
    """Load one validated skill source under an isolated private module name."""
    source_path = Path(path) if path is not None else DEFAULT_SKILLS_PATH
    source = source_path.read_text(encoding="utf-8")
    validate_skill_source(source)
    spec = importlib.util.spec_from_file_location(
        f"forecast_analysis_skills_{abs(hash(source_path.resolve()))}", source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import analysis skills from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in ANALYSIS_SKILL_NAMES if not callable(getattr(module, name, None))]
    if missing:
        raise ImportError(f"analysis skill module is missing {missing!r}")
    return module


def skill_namespace(path: str | Path | None = None) -> dict[str, object]:
    """Return only the reviewed callable surface injected into a method module."""
    module = load_skill_module(path)
    return {name: getattr(module, name) for name in ANALYSIS_SKILL_NAMES}
