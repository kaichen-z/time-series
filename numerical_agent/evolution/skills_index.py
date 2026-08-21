"""Render the frozen skill library as a compact API index for the evolution prompt.

The whole library is far too long to paste into a prompt that already carries the module and
its measurements, and showing the bodies invites the model to reimplement them. Signatures plus
the first line of each docstring are enough to compose against.
"""
from __future__ import annotations

import ast
from pathlib import Path

from . import primite_ts_skills

SKILLS_PATH = Path(primite_ts_skills.__file__)

# Section comments in the library carry the axis names; this maps each to its prompt heading.
AXIS_TITLES: dict[str, str] = {
    "Axis A": "Structure inference",
    "Axis B": "Cleaning",
    "Axis C": "Decomposition",
    "Axis D": "Segmentation",
    "Axis E": "Representation",
    "Axis F": "Models",
    "Axis G": "Matching",
    "Axis H": "Features",
}


def _signature(node: ast.FunctionDef) -> str:
    # ast.unparse on an `arguments` node omits the parentheses, so add them back.
    return f"{node.name}({ast.unparse(node.args)})"


def _summary(node: ast.FunctionDef) -> str:
    text = (ast.get_docstring(node) or "").strip()
    return text.split("\n", 1)[0] if text else ""


def _axis_lines(source: str) -> dict[int, str]:
    """Map the line number of each axis banner to its axis key."""
    found: dict[int, str] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip("# ").strip()
        for key in AXIS_TITLES:
            if stripped.startswith(key):
                found[number] = key
    return found


def render_index(path: str | Path | None = None) -> str:
    """Build the axis-grouped index of every public skill, its signature and its one-line doc."""
    source = Path(path or SKILLS_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banners = _axis_lines(source)

    grouped: dict[str, list[str]] = {key: [] for key in AXIS_TITLES}
    constants: list[str] = []

    for node in tree.body:
        # The option constants are annotated assignments (NAME: tuple[str, ...] = (...)),
        # so both Assign and AnnAssign have to be considered.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            name = getattr(target, "id", "")
            if name.isupper() and isinstance(node.value, (ast.Tuple, ast.List)):
                try:
                    options = ", ".join(repr(v) for v in ast.literal_eval(node.value))
                except ValueError:
                    continue
                constants.append(f"{name} = ({options})")
            continue
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        axis = max((line for line in banners if line < node.lineno), default=None)
        key = banners.get(axis, "Axis A") if axis else "Axis A"
        grouped[key].append(f"    {_signature(node)}\n        {_summary(node)}")

    blocks = [
        "The frozen skill library is imported as P. Call its skills as P.name(...); never "
        "reimplement one.",
        "",
        "Types: Series=list[float], Breaks=list[int], Spectrum=list[(freq, amplitude, phase)], "
        "Atoms=list[Series], Codes=list[float], Features=dict[str, float].",
        "",
        "Every fit_* returns a Model with .extrapolate(horizon) -> Series, .fitted() -> Series, "
        ".residuals() -> Series and .params. Extrapolation only ever happens through "
        "Model.extrapolate, which is what keeps horizon semantics unambiguous.",
        "",
        "Skills raise the same NotApplicable the methods do, so a precondition a skill checks "
        "does not need checking again.",
        "",
    ]
    for key, title in AXIS_TITLES.items():
        if not grouped[key]:
            continue
        blocks.append(f"{key}: {title}")
        blocks.extend(grouped[key])
        blocks.append("")

    if constants:
        blocks.append("Allowed option values:")
        blocks.extend(f"    {line}" for line in constants)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


if __name__ == "__main__":  # pragma: no cover - convenience for eyeballing the output
    print(render_index())
