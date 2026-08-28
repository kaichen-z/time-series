"""A readable record of what each generation did, summarized by a small local model.

The git log holds the operations and their stated reasons, but reading it back means running
git and reconstructing the narrative every time. This writes that narrative once, as it happens,
into one file per run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from common.llm import LLMClient

MEMORY_FILENAME = "memory.md"

SUMMARIZE_GENERATION_PROMPT = """You summarize one generation of an automated experiment that
evolves a Python module of forecasting methods. Each generation an LLM reads measured results and
returns operations -- delete, rewrite, merge -- which are applied and committed.

Write 3 to 6 sentences of plain prose for someone reading the run back later: what changed this
generation, the reasons given, and what that suggests about where the module is heading. Name
methods by their function names. State only what the input supports; if the input is thin, say
less rather than inventing motivation. No bullet points, no headings, no preamble -- start with
the substance."""


def banner(generation: int) -> str:
    """The separator between generations, in the form the run log uses."""
    return f"======== Gen {generation} ============"


def build_summary_request(
    *,
    generation: int,
    applied: Sequence[str],
    method_count: int,
    val_best_smae: float | None = None,
    reasoning: str = "",
) -> str:
    """Build the request describing one generation to the summarizing model."""
    lines = [
        f"Generation {generation}.",
        f"The module holds {method_count} methods after this generation.",
    ]
    if val_best_smae is not None:
        lines.append(f"Best validation sMAE any method reached: {val_best_smae}.")
    if applied:
        lines.append(f"\n{len(applied)} operations were applied:")
        lines.extend(f"- {summary}" for summary in applied)
    else:
        lines.append("\nNo operations were applied this generation.")
    if reasoning.strip():
        # The model's own text is the only place its reasoning appears; the commit message
        # keeps just the per-operation reasons.
        lines.append("\nThe model's response, for context:\n")
        lines.append(reasoning.strip()[:6000])
    return "\n".join(lines)


def summarize_generation(
    llm: LLMClient,
    *,
    generation: int,
    applied: Sequence[str],
    method_count: int,
    val_best_smae: float | None = None,
    reasoning: str = "",
) -> str:
    """Ask the summarizing model to describe one generation in prose."""
    request = build_summary_request(
        generation=generation,
        applied=applied,
        method_count=method_count,
        val_best_smae=val_best_smae,
        reasoning=reasoning,
    )
    response = llm.complete(
        system=SUMMARIZE_GENERATION_PROMPT,
        messages=[{"role": "user", "content": request}],
        temperature=0.2,
    )
    return response.text.strip()


def fallback_summary(applied: Sequence[str], method_count: int) -> str:
    """What to record when the summarizing model is unavailable: the operations, verbatim.

    A generation with no entry at all reads later as a generation that did nothing, which is a
    worse record than an unpolished one.
    """
    if not applied:
        return f"No operations were applied. The module holds {method_count} methods."
    lines = [f"{len(applied)} operations were applied (summarizer unavailable):"]
    lines.extend(f"- {summary}" for summary in applied)
    lines.append(f"The module holds {method_count} methods.")
    return "\n".join(lines)


def append_memory(repo: str | Path, generation: int, text: str) -> Path:
    """Append one generation's entry to the run's memory file, creating it if needed."""
    destination = Path(repo) / MEMORY_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    entry = f"{banner(generation)}\n\n{text.strip()}\n\n"
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return destination


def record_generation(
    repo: str | Path,
    llm: LLMClient | None,
    *,
    generation: int,
    applied: Sequence[str],
    method_count: int,
    val_best_smae: float | None = None,
    reasoning: str = "",
) -> Path:
    """Summarize and record one generation, falling back to the raw operations on any failure.

    A failed summary must never cost a generation: the module is already committed by the time
    this runs, so anything raised here would discard work that succeeded.
    """
    text = ""
    if llm is not None:
        try:
            text = summarize_generation(
                llm,
                generation=generation,
                applied=applied,
                method_count=method_count,
                val_best_smae=val_best_smae,
                reasoning=reasoning,
            )
        except Exception as exc:  # the summarizer is a convenience, never a dependency
            text = f"{fallback_summary(applied, method_count)}\n\nSummarizer failed: {exc}"
    if not text.strip():
        text = fallback_summary(applied, method_count)
    return append_memory(repo, generation, text)
