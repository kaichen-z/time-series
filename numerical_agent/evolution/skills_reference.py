"""Render the frozen skill library as a compact API index for the evolution prompt.

The whole library is far too long to paste into a prompt that already carries the module and
its measurements, and showing the bodies invites the model to reimplement them. Signatures plus
the first line of each docstring are enough to compose against.
"""
from __future__ import annotations

import ast
from pathlib import Path

from . import analysis_skills

SKILLS_PATH = Path(analysis_skills.__file__)

# The library's own section comments are the headings, in the order the prompt should show them.
SECTION_TITLES: tuple[str, ...] = (
    "Structure inference",
    "Cleaning",
    "Decomposition",
    "Segmentation",
    "Models",
    "Matching",
    "Features",
)


def _signature(node: ast.FunctionDef) -> str:
    # ast.unparse on an `arguments` node omits the parentheses, so add them back.
    return f"{node.name}({ast.unparse(node.args)})"


def _summary(node: ast.FunctionDef) -> str:
    text = (ast.get_docstring(node) or "").strip()
    return text.split("\n", 1)[0] if text else ""


def _section_lines(source: str) -> dict[int, str]:
    """Map the line number of each axis banner to its axis key."""
    found: dict[int, str] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip("# ").strip()
        for title in SECTION_TITLES:
            if stripped == title:
                found[number] = title
    return found


def build_skills_reference(path: str | Path | None = None) -> str:
    """List every public skill under its section heading, with its signature and one-line doc."""
    source = Path(path or SKILLS_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banners = _section_lines(source)

    grouped: dict[str, list[str]] = {title: [] for title in SECTION_TITLES}
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
        above = max((line for line in banners if line < node.lineno), default=None)
        key = banners.get(above, SECTION_TITLES[0]) if above else SECTION_TITLES[0]
        grouped[key].append(f"    {_signature(node)}\n        {_summary(node)}")

    blocks = [
        "The frozen skill library is imported as P. Call its skills as P.name(...); never "
        "reimplement one.",
        "",
        "Types: Series=list[float], Breaks=list[int], Spectrum=list[(freq, amplitude, phase)], "
        "Features=dict[str, float].",
        "",
        "Every fit_* returns a Model with .extrapolate(horizon) -> Series, .fitted() -> Series, "
        ".residuals() -> Series and .params. Extrapolation only ever happens through "
        "Model.extrapolate, which is what keeps horizon semantics unambiguous.",
        "",
        "Skills raise the same NotApplicable the methods do, so a precondition a skill checks "
        "does not need checking again.",
        "",
    ]
    for title in SECTION_TITLES:
        if not grouped[title]:
            continue
        blocks.append(f"{title}:")
        blocks.extend(grouped[title])
        blocks.append("")

    if constants:
        blocks.append("Allowed option values:")
        blocks.extend(f"    {line}" for line in constants)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


if __name__ == "__main__":  # pragma: no cover - convenience for eyeballing the output
    print(build_skills_reference())
